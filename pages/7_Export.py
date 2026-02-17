"""Page 7 - Export & Reporting."""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Export | Demand Forecasting Studio", layout="wide")

from app.ui.shared import setup_page  # noqa: E402

setup_page()

import pandas as pd  # noqa: E402

from app.services.eda import compute_growth_metrics  # noqa: E402
from app.services.reporting import build_forecast_table, build_one_page_summary, export_to_excel  # noqa: E402

st.header("7) Export & Reporting")

validated = st.session_state.validated
forecasts = st.session_state.forecasts
rank_df = st.session_state.rank_df
selected_model = st.session_state.selected_model

if not validated or not forecasts or selected_model is None:
    st.info("Generate and evaluate forecasts first (pages 4 & 5).")
    st.stop()

clean = validated["clean_df"]
forecast_df = st.session_state.adjusted_forecasts.get(selected_model, forecasts[selected_model]).copy()

# --- Manual Adjustments ---
st.markdown("**Manual Adjustments (Audit Trailed)**")
c1, c2, c3 = st.columns(3)
with c1:
    start_date = st.date_input("Adjustment start", value=forecast_df["date"].min().date(), key="adj_start")
with c2:
    end_date = st.date_input("Adjustment end", value=forecast_df["date"].max().date(), key="adj_end")
with c3:
    pct = st.number_input("Adjustment %", min_value=-50.0, max_value=50.0, value=0.0, step=0.5, key="adj_pct")
reason = st.text_input("Adjustment reason", value="", key="adj_reason")

if st.button("Apply Manual Adjustment"):
    mask = (forecast_df["date"] >= pd.to_datetime(start_date)) & (
        forecast_df["date"] <= pd.to_datetime(end_date)
    )
    before = forecast_df.loc[mask, "forecast"].copy()
    factor = 1 + (pct / 100.0)
    forecast_df.loc[mask, ["forecast", "lower_80", "upper_80", "lower_95", "upper_95"]] *= factor
    st.session_state.adjusted_forecasts[selected_model] = forecast_df
    st.session_state.audit_trail.append(
        {
            "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
            "model_id": selected_model,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "adjustment_pct": pct,
            "rows_affected": int(mask.sum()),
            "original_mean_forecast": float(before.mean()) if len(before) else None,
            "adjusted_mean_forecast": float(forecast_df.loc[mask, "forecast"].mean()) if mask.any() else None,
            "reason": reason,
        }
    )
    st.success("Manual adjustment applied and audit log updated.")

# --- Build report tables ---
forecast_table = build_forecast_table(clean, forecast_df)
growth = compute_growth_metrics(clean)

report = validated.get("report")
summary_dict = getattr(report, "summary", {}) if not isinstance(report, dict) else report.get("summary", {})
issues_raw = getattr(report, "issues", []) if not isinstance(report, dict) else report.get("issues", [])

summary = build_one_page_summary(
    selected_model=selected_model,
    horizon=len(forecast_df),
    rank_df=rank_df,
    validation_summary=summary_dict,
    explanation=st.session_state.explanation,
    growth_outlook={
        "wow_growth_latest": float(growth["wow_growth"].dropna().iloc[-1])
        if not growth["wow_growth"].dropna().empty
        else None,
        "trailing_52w_growth_latest": float(growth["trailing_52w_growth"].dropna().iloc[-1])
        if not growth["trailing_52w_growth"].dropna().empty
        else None,
    },
)

issues_list = []
for i in issues_raw:
    if hasattr(i, "level"):
        issues_list.append({"level": i.level, "check": i.check, "message": i.message, "details": str(i.details)})
    else:
        issues_list.append({"level": i["level"], "check": i["check"], "message": i["message"], "details": str(i["details"])})
issues_df = pd.DataFrame(issues_list)
audit_df = pd.DataFrame(st.session_state.audit_trail)

# --- Preview ---
st.dataframe(forecast_table.tail(30), use_container_width=True)
if not rank_df.empty:
    st.dataframe(rank_df, use_container_width=True)
st.dataframe(summary, use_container_width=True)
if not audit_df.empty:
    st.dataframe(audit_df, use_container_width=True)

# --- Downloads ---
excel_bytes = export_to_excel(forecast_table, rank_df, summary, issues_df, audit_trail=audit_df)
st.download_button(
    "Download Excel Report",
    data=excel_bytes,
    file_name="forecast_report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
st.download_button(
    "Download Forecast CSV",
    data=forecast_table.to_csv(index=False).encode("utf-8"),
    file_name="forecast_table.csv",
    mime="text/csv",
)
