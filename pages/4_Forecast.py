"""Page 4 - Modeling & Forecast Generation."""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Forecast | Demand Forecasting Studio", layout="wide")

from app.ui.shared import setup_page  # noqa: E402

setup_page()

import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

from app.core.config import MAX_HORIZON_WEEKS, MIN_HORIZON_WEEKS  # noqa: E402
from app.services.features import build_holiday_features  # noqa: E402
from app.services.models import available_models  # noqa: E402

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

# ─────────────────────────────────────────────────────────────────────────────
# Model Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("⚙️ Advanced: Model Hyperparameters", expanded=False):
    st.caption(
        "Adjust model-specific parameters to explore different forecasting behaviours. "
        "Changes take effect on the next 'Generate Forecasts' run."
    )

    hp_col1, hp_col2 = st.columns(2)

    with hp_col1:
        st.markdown("**Seasonal Naive**")
        sn_period = st.number_input(
            "Seasonal period (weeks)", min_value=4, max_value=104, value=52, step=4,
            key="hp_sn_period",
            help="Number of weeks in one seasonal cycle. Default 52 = annual.",
        )

        st.markdown("**Moving Average**")
        ma_window = st.slider(
            "Rolling window (weeks)", min_value=4, max_value=52, value=13,
            key="hp_ma_window",
            help="Wider window = smoother, slower to react. Narrow = more responsive.",
        )

        st.markdown("**ETS (Exponential Smoothing)**")
        ets_damped = st.checkbox("Damped trend", value=True, key="hp_ets_damped",
                                  help="Prevents ETS from projecting the trend indefinitely — recommended for long horizons.")
        ets_sp = st.number_input(
            "Seasonal periods", min_value=4, max_value=104, value=52, step=4,
            key="hp_ets_sp",
            help="Must match the dominant seasonality in your data (typically 52 for weekly data).",
        )

    with hp_col2:
        st.markdown("**SARIMAX**")
        s_col1, s_col2, s_col3 = st.columns(3)
        sarimax_p = s_col1.slider("p (AR)", 0, 3, 1, key="hp_sarimax_p",
                                   help="Autoregressive order — how many past values to include.")
        sarimax_d = s_col2.slider("d (diff)", 0, 2, 1, key="hp_sarimax_d",
                                   help="Differencing order — 1 removes a linear trend, 0 if already stationary.")
        sarimax_q = s_col3.slider("q (MA)", 0, 3, 1, key="hp_sarimax_q",
                                   help="Moving average order for the error term.")
        sp_col1, sp_col2, sp_col3 = st.columns(3)
        sarimax_P = sp_col1.slider("P (seasonal AR)", 0, 2, 1, key="hp_sarimax_P")
        sarimax_D = sp_col2.slider("D (seasonal diff)", 0, 1, 0, key="hp_sarimax_D")
        sarimax_Q = sp_col3.slider("Q (seasonal MA)", 0, 2, 1, key="hp_sarimax_Q")
        st.caption("Seasonal period S is fixed at 52 weeks.")

        st.markdown("**ML Lag (Random Forest)**")
        rf_n = st.slider(
            "n_estimators", min_value=50, max_value=500, value=300, step=50,
            key="hp_rf_n",
            help="More trees = more stable predictions but slower. 200–300 is a good default.",
        )
        rf_lags = st.slider(
            "Max lag (weeks)", min_value=4, max_value=52, value=52,
            key="hp_rf_lags",
            help="Maximum lag to include as a feature. Reduce to 13 or 26 for shorter-memory patterns.",
        )

# Build params dict consumed when running models
model_params: dict[str, dict] = {
    "seasonal_naive": {"seasonal_period": int(sn_period)},
    "moving_average": {"window": int(ma_window)},
    "ets": {"damped_trend": bool(ets_damped), "seasonal_periods": int(ets_sp)},
    "sarimax": {
        "order": (int(sarimax_p), int(sarimax_d), int(sarimax_q)),
        "seasonal_order": (int(sarimax_P), int(sarimax_D), int(sarimax_Q), 52),
    },
    "ml_lag_rf": {"n_estimators": int(rf_n), "max_lags": int(rf_lags)},
}

# ─────────────────────────────────────────────────────────────────────────────
# Generate Forecasts
# ─────────────────────────────────────────────────────────────────────────────
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
            extra = model_params.get(model_id, {})
            if model_id == "sarimax":
                forecasts[model_id] = model_map[model_id](model_input, horizon, exog_future=holiday_future, **extra)
            else:
                forecasts[model_id] = model_map[model_id](model_input, horizon, **extra)
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

# ─────────────────────────────────────────────────────────────────────────────
# Forecast Preview
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.forecasts:
    st.markdown("**Forecast Preview**")
    preview_model = st.selectbox(
        "Preview model",
        options=list(st.session_state.forecasts.keys()),
        key="modeling_preview_model",
    )
    preview_df = st.session_state.forecasts[preview_model]

    # Bridge: prepend last actual point so forecast traces connect visually
    last_date = clean["date"].iloc[-1]
    last_y = float(clean["y"].iloc[-1])
    bridge_dates = pd.concat([pd.Series([last_date]), preview_df["date"]]).reset_index(drop=True)
    bridge_y = pd.concat([pd.Series([last_y]), preview_df["forecast"]]).reset_index(drop=True)
    bridge_upper = pd.concat([pd.Series([last_y]), preview_df["upper_95"]]).reset_index(drop=True)
    bridge_lower = pd.concat([pd.Series([last_y]), preview_df["lower_95"]]).reset_index(drop=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=clean["date"], y=clean["y"], mode="lines", name="Actuals", line=dict(color="#1f2937")))
    fig.add_trace(
        go.Scatter(
            x=bridge_dates,
            y=bridge_y,
            mode="lines",
            name=f"{preview_model} forecast",
            line=dict(color="#0ea5e9"),
        )
    )
    fig.add_trace(
        go.Scatter(x=bridge_dates, y=bridge_upper, line=dict(width=0), showlegend=False, hoverinfo="skip")
    )
    fig.add_trace(
        go.Scatter(
            x=bridge_dates,
            y=bridge_lower,
            fill="tonexty",
            fillcolor="rgba(59,130,246,0.15)",
            line=dict(width=0),
            name="95% interval",
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title=f"Forecast Preview ({preview_model})",
        template="plotly_white",
        yaxis=dict(rangemode="tozero"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(preview_df.head(20), use_container_width=True)
