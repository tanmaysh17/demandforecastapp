"""Page 2 - EDA & Data Health."""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="EDA & Data Health | Demand Forecasting Studio", layout="wide")

from app.ui.shared import setup_page  # noqa: E402

setup_page()

import plotly.express as px  # noqa: E402

from app.services.eda import compute_growth_metrics, stationarity_indicator  # noqa: E402

st.header("2) EDA & Data Health")

validated = st.session_state.validated
if not validated:
    st.info("Upload and validate data first (page 1).")
    st.stop()

report = validated.get("report")
summary = getattr(report, "summary", {}) if not isinstance(report, dict) else report.get("summary", {})
clean = validated["clean_df"]

# --- Metrics ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("History (weeks)", summary.get("history_weeks"))
c2.metric("Missing weeks", summary.get("missing_weeks"))
c3.metric("Outliers", summary.get("outlier_count"))
c4.metric("Negative values", summary.get("negative_count"))

if summary.get("degraded_mode"):
    st.warning("Degraded mode active: limited model selection available on the Forecast page.")

# --- Issue list ---
issues_raw = getattr(report, "issues", []) if not isinstance(report, dict) else report.get("issues", [])
sev_color = {"error": "red", "warning": "orange", "info": "blue"}
if not issues_raw:
    st.success("No data quality issues detected.")
else:
    for issue in issues_raw:
        if hasattr(issue, "level"):
            lvl, chk, msg, det = issue.level, issue.check, issue.message, issue.details
        else:
            lvl, chk, msg, det = issue["level"], issue["check"], issue["message"], issue["details"]
        color = sev_color.get(lvl, "gray")
        st.markdown(f"- :{color}[**{lvl.upper()}**] `{chk}`: {msg} | `{det}`")

# --- Demand trend ---
import plotly.graph_objects as go  # noqa: E402

fig = go.Figure()
fig.add_trace(go.Scatter(x=clean["date"], y=clean["y"], mode="lines", name="Demand"))
flagged = clean[clean["is_outlier"] | clean["is_change_point"]]
if not flagged.empty:
    fig.add_scatter(
        x=flagged["date"],
        y=flagged["y"],
        mode="markers",
        marker=dict(color="red", size=8),
        name="Outlier / Change point",
    )
fig.update_layout(title="Demand Trend", xaxis_title="Date", yaxis_title="Demand", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# --- Growth metrics ---
growth = compute_growth_metrics(clean)
st.plotly_chart(
    px.line(
        growth,
        x="date",
        y=["wow_growth", "rolling_4w_growth", "rolling_13w_growth"],
        title="Growth Rates",
    ),
    use_container_width=True,
)

# --- Stationarity ---
stationarity = stationarity_indicator(clean["y"])
st.write("Stationarity indicator", stationarity)
