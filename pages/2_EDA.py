"""Page 2 - EDA & Data Health."""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="EDA & Data Health | Demand Forecasting Studio", layout="wide")

from app.ui.shared import setup_page  # noqa: E402

setup_page()

import numpy as np  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

from app.services.eda import compute_growth_metrics, stationarity_indicator  # noqa: E402

st.header("2) EDA & Data Health")

validated = st.session_state.validated
if not validated:
    st.info("Upload and validate data first (page 1).")
    st.stop()

report = validated.get("report")
summary = getattr(report, "summary", {}) if not isinstance(report, dict) else report.get("summary", {})
clean = validated["clean_df"]

# --- Metrics row ---
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("History (weeks)", summary.get("history_weeks"))
c2.metric("Missing weeks", summary.get("missing_weeks"))
c3.metric("Outliers", summary.get("outlier_count"))
c4.metric("Negative values", summary.get("negative_count"))
cv = clean["y"].std() / clean["y"].mean() if clean["y"].mean() != 0 else float("nan")
c5.metric("Coeff. of Variation", f"{cv:.1%}" if np.isfinite(cv) else "N/A")
st.caption(
    "**CV (Coefficient of Variation):** measures demand volatility relative to its mean. "
    "CV > 30% indicates high variability — models with interval estimates become especially important."
)

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

# ─────────────────────────────────────────────────────────────────────────────
# 1. Demand Trend
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Demand Trend")
st.caption(
    "The primary time series view. Look for: upward/downward trend, seasonal peaks and troughs, "
    "and sudden level shifts. Red markers indicate weeks flagged as outliers or structural change points — "
    "inspect these before running models."
)
fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(x=clean["date"], y=clean["y"], mode="lines", name="Demand", line=dict(color="#0ea5e9")))
flagged = clean[clean["is_outlier"] | clean["is_change_point"]]
if not flagged.empty:
    fig_trend.add_scatter(
        x=flagged["date"],
        y=flagged["y"],
        mode="markers",
        marker=dict(color="red", size=8),
        name="Outlier / Change point",
    )
fig_trend.update_layout(
    xaxis_title="Date",
    yaxis_title="Demand",
    template="plotly_white",
    yaxis=dict(rangemode="tozero"),
    margin=dict(t=20, b=20),
)
st.plotly_chart(fig_trend, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Weekly Demand Distribution
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Weekly Demand Distribution")
st.caption(
    "Shows the spread of weekly demand values. A right-skewed distribution (long tail to the right) is "
    "common in retail — a few high-demand weeks drive the mean above the median. "
    "Wide spread (high std) means forecasting uncertainty will also be high. "
    "The dashed lines show mean (blue) and median (green)."
)
fig_hist = px.histogram(
    clean,
    x="y",
    nbins=40,
    labels={"y": "Weekly Demand"},
    color_discrete_sequence=["#93c5fd"],
)
mean_val = float(clean["y"].mean())
median_val = float(clean["y"].median())
fig_hist.add_vline(x=mean_val, line_dash="dash", line_color="#0284c7", annotation_text=f"Mean: {mean_val:,.0f}")
fig_hist.add_vline(x=median_val, line_dash="dash", line_color="#16a34a", annotation_text=f"Median: {median_val:,.0f}")
fig_hist.update_layout(template="plotly_white", xaxis_title="Weekly Demand", yaxis_title="Count", margin=dict(t=10, b=20))
st.plotly_chart(fig_hist, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Year-over-Year Demand Overlay
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Year-over-Year Demand by Week")
st.caption(
    "Each line represents one calendar year, plotted by ISO week number (1–52). "
    "Consistent peaks and troughs across years confirm repeatable seasonality — a key signal that "
    "seasonal models (ETS, SARIMAX, Seasonal Naive) will perform well. "
    "Diverging years may indicate trend changes or external shocks worth investigating."
)
yoy_df = clean.copy()
yoy_df["year"] = yoy_df["date"].dt.year.astype(str)
yoy_df["week"] = yoy_df["date"].dt.isocalendar().week.astype(int)
fig_yoy = px.line(
    yoy_df,
    x="week",
    y="y",
    color="year",
    labels={"week": "ISO Week of Year", "y": "Demand", "year": "Year"},
)
fig_yoy.update_layout(
    template="plotly_white",
    xaxis=dict(tickmode="linear", tick0=1, dtick=4),
    yaxis=dict(rangemode="tozero"),
    margin=dict(t=10, b=20),
)
st.plotly_chart(fig_yoy, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Growth Rates
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Growth Rates")
st.caption(
    "**WoW (Week-over-Week):** noisy, shows short-term swings. "
    "**Rolling 4-week:** smooths weekly noise, reveals monthly momentum. "
    "**Rolling 13-week:** reveals quarterly trend direction. "
    "A consistently positive 13-week growth rate suggests a structural uptrend that models should capture."
)
growth = compute_growth_metrics(clean)
st.plotly_chart(
    px.line(
        growth,
        x="date",
        y=["wow_growth", "rolling_4w_growth", "rolling_13w_growth"],
        labels={"value": "Growth Rate", "variable": "Metric"},
    ).update_layout(template="plotly_white", margin=dict(t=10, b=20)),
    use_container_width=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Autocorrelation (ACF)
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Autocorrelation (ACF)")
st.caption(
    "Shows how correlated today's demand is with demand k weeks ago (lag k). "
    "**Spike at lag 52:** confirms strong annual seasonality. "
    "**Spikes at lag 4 or 13:** monthly or quarterly patterns. "
    "**High autocorrelation at lag 1:** the series has short-term momentum — SARIMAX and ML models will benefit. "
    "Red bars are statistically significant (outside the dashed ±1.96/√n threshold)."
)
series = clean["y"].dropna()
max_lag = min(60, len(series) - 2)
n = len(series)
sig_bound = 1.96 / np.sqrt(n)
acf_vals = [1.0]
for lag in range(1, max_lag + 1):
    acf_vals.append(float(series.autocorr(lag=lag)))
import pandas as pd  # noqa: E402
acf_df = pd.DataFrame({"lag": range(max_lag + 1), "acf": acf_vals})
colors = ["#93c5fd"] + ["#ef4444" if abs(v) > sig_bound else "#93c5fd" for v in acf_vals[1:]]
fig_acf = go.Figure()
fig_acf.add_trace(go.Bar(x=acf_df["lag"], y=acf_df["acf"], marker_color=colors, name="ACF"))
fig_acf.add_hline(y=sig_bound, line_dash="dash", line_color="#94a3b8", annotation_text=f"+{sig_bound:.3f}")
fig_acf.add_hline(y=-sig_bound, line_dash="dash", line_color="#94a3b8", annotation_text=f"-{sig_bound:.3f}")
fig_acf.update_layout(
    template="plotly_white",
    xaxis_title="Lag (weeks)",
    yaxis_title="Autocorrelation",
    yaxis=dict(range=[-1.05, 1.05]),
    margin=dict(t=10, b=20),
)
st.plotly_chart(fig_acf, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Stationarity Test
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Stationarity Test (ADF)")
st.caption(
    "The Augmented Dickey-Fuller test checks whether the series has a unit root (non-stationary). "
    "**Non-stationary** means the mean or variance changes over time — common for trending demand data. "
    "SARIMAX handles this with differencing (d=1). ETS with damped trend also adapts. "
    "If non-stationary, avoid using Moving Average or Drift models for long horizons."
)
stationarity = stationarity_indicator(clean["y"])
if stationarity.get("status") == "insufficient_data":
    st.info("Insufficient data for stationarity test (need ≥ 30 observations).")
else:
    stat_val = stationarity.get("adf_stat", float("nan"))
    p_val = stationarity.get("p_value", float("nan"))
    is_stationary = stationarity.get("status") == "stationary"
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("ADF Statistic", f"{stat_val:.4f}")
    col_s2.metric("p-value", f"{p_val:.4f}")
    col_s3.metric("Result", "Stationary ✅" if is_stationary else "Non-stationary ⚠️")
    if is_stationary:
        st.success(
            "The series is stationary (p < 0.05). Its statistical properties are stable over time — "
            "all model classes should perform reliably."
        )
    else:
        st.warning(
            "The series appears non-stationary (p ≥ 0.05). The mean or variance may be changing over time. "
            "Use SARIMAX (d=1) or ETS for best results. Moving Average and Drift forecasts may drift "
            "away from reality on long horizons."
        )
