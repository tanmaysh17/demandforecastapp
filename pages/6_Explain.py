"""Page 6 - Explainability & Recommendation."""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Explain | Demand Forecasting Studio", layout="wide")

from app.ui.shared import setup_page  # noqa: E402

setup_page()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

st.header("6) Explainability & Recommendation")

validated = st.session_state.validated
forecasts = st.session_state.forecasts
rank_df = st.session_state.rank_df
selected_model = st.session_state.selected_model

if not validated or not forecasts:
    st.info("Run forecasting and backtesting first (pages 4 & 5).")
    st.stop()

clean = validated["clean_df"]
if selected_model is None:
    selected_model = list(forecasts.keys())[0]
    st.session_state.selected_model = selected_model

# Last actual point used to bridge all forecast traces
last_date = clean["date"].iloc[-1]
last_y = float(clean["y"].iloc[-1])

# --- Actuals + all forecasts overlay ---
fig = go.Figure()
fig.add_trace(
    go.Scatter(x=clean["date"], y=clean["y"], mode="lines", name="Actuals", line=dict(color="#1f2937"))
)
for f in forecasts.values():
    mid = f["model_id"].iloc[0]
    width = 2.5 if mid == selected_model else 1
    # Bridge: prepend last actual so each model connects from the same anchor
    bridge_dates = pd.concat([pd.Series([last_date]), f["date"]]).reset_index(drop=True)
    bridge_y = pd.concat([pd.Series([last_y]), f["forecast"]]).reset_index(drop=True)
    fig.add_trace(go.Scatter(
        x=bridge_dates, y=bridge_y, mode="lines", name=mid, line=dict(width=width),
    ))

sf = forecasts.get(selected_model)
if sf is not None:
    bridge_dates_sf = pd.concat([pd.Series([last_date]), sf["date"]]).reset_index(drop=True)
    bridge_upper = pd.concat([pd.Series([last_y]), sf["upper_95"]]).reset_index(drop=True)
    bridge_lower = pd.concat([pd.Series([last_y]), sf["lower_95"]]).reset_index(drop=True)
    fig.add_trace(
        go.Scatter(x=bridge_dates_sf, y=bridge_upper, line=dict(width=0), showlegend=False, hoverinfo="skip")
    )
    fig.add_trace(
        go.Scatter(
            x=bridge_dates_sf,
            y=bridge_lower,
            fill="tonexty",
            fillcolor="rgba(59,130,246,0.15)",
            line=dict(width=0),
            name="95% interval",
            hoverinfo="skip",
        )
    )
fig.update_layout(
    title="Actuals and Forecasts",
    xaxis_title="Date",
    yaxis_title="Demand",
    template="plotly_white",
    yaxis=dict(rangemode="tozero"),
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "All forecasts are bridged from the last actual data point. "
    "The highlighted (thicker) line is the recommended model. "
    "The shaded band shows the 95% prediction interval for the selected model."
)

# --- Recommendation rationale ---
st.markdown("**Model recommendation rationale**")
st.write(
    st.session_state.explanation
    or "Model selected based on backtest performance, stability, and simplicity."
)

if not rank_df.empty:
    selected_row = rank_df[rank_df["model_id"] == selected_model]
    if not selected_row.empty:
        st.write("Selected model metrics", selected_row.iloc[0].to_dict())

# --- Residual diagnostics ---
selected_folds = st.session_state.fold_map.get(selected_model, [])
if selected_folds:
    st.subheader("Residual Diagnostics")
    st.caption(
        "Residuals are the differences between what the model predicted and what actually happened "
        "during the backtest holdout period. "
        "**Good residuals** look like random noise with no pattern — they should fluctuate around zero. "
        "**Patterns in residuals** (trends, seasonality) suggest the model is missing something systematic."
    )
    fold = sorted(selected_folds, key=lambda x: x.holdout_weeks)[-1]
    resid = np.array(fold.actual) - np.array(fold.predicted)
    resid_df = pd.DataFrame({"step": np.arange(1, len(resid) + 1), "residual": resid})
    st.plotly_chart(
        px.line(resid_df, x="step", y="residual", title="Residuals over Holdout Period")
        .update_layout(template="plotly_white"),
        use_container_width=True,
    )
    max_lag = min(20, len(resid) - 1)
    if max_lag > 1:
        st.caption(
            "**Residual autocorrelation:** if bars are significant (outside the dashed lines), "
            "the model is leaving predictable patterns on the table. "
            "Large spikes at lag 52 → missed seasonality. Large lag-1 spike → consider ARIMA or ML models."
        )
        acf_vals = [1.0]
        for lag in range(1, max_lag + 1):
            acf_vals.append(float(np.corrcoef(resid[:-lag], resid[lag:])[0, 1]))
        acf_df = pd.DataFrame({"lag": np.arange(0, max_lag + 1), "acf": acf_vals})
        st.plotly_chart(
            px.bar(acf_df, x="lag", y="acf", title="Residual Autocorrelation")
            .update_layout(template="plotly_white"),
            use_container_width=True,
        )
