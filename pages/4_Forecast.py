"""Page 4 - Modeling & Forecast Generation."""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Forecast | Demand Forecasting Studio", layout="wide")

from app.ui.shared import setup_page  # noqa: E402

setup_page()

import plotly.graph_objects as go  # noqa: E402

from app.core.config import MAX_HORIZON_WEEKS, MIN_HORIZON_WEEKS  # noqa: E402
from app.services.features import build_holiday_features  # noqa: E402
from app.services.models import available_models  # noqa: E402

import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402

st.header("4) Modeling & Forecast Generation")

validated = st.session_state.validated
if not validated:
    st.info("Upload and validate data first (page 1).")
    st.stop()

clean = validated["clean_df"].copy()

# --- Determine available models based on history / degraded mode ---
report = validated.get("report")
summary = getattr(report, "summary", {}) if not isinstance(report, dict) else report.get("summary", {})
degraded_mode = summary.get("degraded_mode", False)

model_map = available_models()
SIMPLE_MODELS = ["seasonal_naive", "moving_average", "drift"]

if degraded_mode:
    st.warning(
        "Degraded mode: history is too short for seasonal models. "
        "Only simple models are available."
    )
    available_model_ids = SIMPLE_MODELS
else:
    available_model_ids = list(model_map.keys())

horizon = st.slider(
    "Forecast horizon (weeks)",
    min_value=MIN_HORIZON_WEEKS,
    max_value=MAX_HORIZON_WEEKS,
    value=104,
    step=1,
)
auto_mode = st.checkbox("Auto mode (recommended)", value=True)
selected_models = st.multiselect(
    "Select models",
    options=available_model_ids,
    default=available_model_ids if auto_mode else SIMPLE_MODELS[:2],
)
holiday_mode = st.selectbox(
    "Holiday calendar",
    options=["None", "US", "Custom (future extension)"],
    index=0,
)
st.session_state.holiday_mode = holiday_mode
st.caption(
    f"Holiday mode: {holiday_mode}. Calendar effects applied through exogenous regressors when provided."
)

if st.button("Generate Forecasts", type="primary"):
    holiday_hist = build_holiday_features(clean["date"].min(), clean["date"].max(), mode=holiday_mode)
    model_input = clean.merge(holiday_hist, on="date", how="left")
    future_dates = pd.date_range(
        model_input["date"].max() + pd.Timedelta(weeks=1), periods=horizon, freq="W-MON"
    )
    holiday_future = build_holiday_features(future_dates.min(), future_dates.max(), mode=holiday_mode)

    forecasts: dict[str, pd.DataFrame] = {}
    for model_id in selected_models:
        try:
            if model_id == "sarimax":
                forecasts[model_id] = model_map[model_id](model_input, horizon, exog_future=holiday_future)
            else:
                forecasts[model_id] = model_map[model_id](model_input, horizon)
        except Exception as ex:
            st.warning(f"Model `{model_id}` failed: {ex}")

    if forecasts:
        st.session_state.forecasts = forecasts
        st.session_state.adjusted_forecasts = {}
        # Reset backtest results when forecasts change
        st.session_state.fold_map = {}
        st.session_state.backtest_errors = {}
        st.session_state.rank_df = pd.DataFrame()
        st.session_state.selected_model = None
        st.session_state.explanation = ""
        st.success(f"Generated {len(forecasts)} model forecasts.")
    else:
        st.error("No forecasts were generated.")

if st.session_state.forecasts:
    st.markdown("**Forecast Preview**")
    preview_model = st.selectbox(
        "Preview model",
        options=list(st.session_state.forecasts.keys()),
        key="modeling_preview_model",
    )
    preview_df = st.session_state.forecasts[preview_model]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=clean["date"], y=clean["y"], mode="lines", name="Actuals"))
    fig.add_trace(
        go.Scatter(x=preview_df["date"], y=preview_df["forecast"], mode="lines", name=f"{preview_model} forecast")
    )
    fig.add_trace(
        go.Scatter(x=preview_df["date"], y=preview_df["upper_95"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    )
    fig.add_trace(
        go.Scatter(
            x=preview_df["date"],
            y=preview_df["lower_95"],
            fill="tonexty",
            fillcolor="rgba(59,130,246,0.15)",
            line=dict(width=0),
            name="95% interval",
        )
    )
    fig.update_layout(title=f"Forecast Preview ({preview_model})", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(preview_df.head(20), use_container_width=True)
