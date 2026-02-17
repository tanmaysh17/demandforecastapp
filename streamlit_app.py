from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.core.config import MAX_HORIZON_WEEKS, MIN_HORIZON_WEEKS, RANDOM_SEED
from app.services.backtesting import rolling_backtest
from app.services.data_loader import build_template, infer_columns, read_tabular
from app.services.eda import cagr, compute_growth_metrics, decompose_weekly, detect_seasonality_periods, seasonal_indices, stationarity_indicator
from app.services.features import build_holiday_features, prepare_weekly_exog
from app.services.models import available_models, forecast_ensemble
from app.services.reporting import build_forecast_table, build_one_page_summary, export_to_excel
from app.services.selection import build_explanation, rank_models, summarize_evaluations
from app.services.validation import validate_and_prepare

np.random.seed(RANDOM_SEED)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("forecasting_app")

st.set_page_config(page_title="Demand Forecasting Studio", layout="wide")
st.title("Demand Forecasting Studio")
st.caption("Enterprise-ready weekly demand forecasting with validation, backtests, explainability, and export.")


def _render_validation_report(report):
    sev_color = {"error": "red", "warning": "orange", "info": "blue"}
    if not report.issues:
        st.success("No data quality issues detected.")
        return
    for issue in report.issues:
        color = sev_color.get(issue.level, "gray")
        st.markdown(
            f"- :{color}[**{issue.level.upper()}**] `{issue.check}`: {issue.message} | Details: `{issue.details}`"
        )


def _plot_actuals_forecast(history: pd.DataFrame, forecasts: list[pd.DataFrame], selected_model: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["date"], y=history["y"], mode="lines", name="Actuals", line=dict(color="#1f2937")))
    for f in forecasts:
        mid = 3 if f["model_id"].iloc[0] == selected_model else 1
        fig.add_trace(
            go.Scatter(
                x=f["date"],
                y=f["forecast"],
                mode="lines",
                name=f["model_id"].iloc[0],
                line=dict(width=mid),
            )
        )
    sf = next((f for f in forecasts if f["model_id"].iloc[0] == selected_model), None)
    if sf is not None:
        fig.add_trace(
            go.Scatter(
                x=sf["date"],
                y=sf["upper_95"],
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sf["date"],
                y=sf["lower_95"],
                fill="tonexty",
                fillcolor="rgba(59,130,246,0.15)",
                line=dict(width=0),
                name="95% interval",
                hoverinfo="skip",
            )
        )
    fig.update_layout(title="Actuals and Forecasts", xaxis_title="Date", yaxis_title="Demand", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)


tabs = st.tabs(
    [
        "1) Upload & Mapping",
        "2) EDA & Data Health",
        "3) Features & Seasonality",
        "4) Modeling & Forecasting",
        "5) Backtesting & Accuracy",
        "6) Explainability",
        "7) Export & Reporting",
    ]
)

if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "validated" not in st.session_state:
    st.session_state.validated = None
if "forecasts" not in st.session_state:
    st.session_state.forecasts = {}
if "rank_df" not in st.session_state:
    st.session_state.rank_df = pd.DataFrame()
if "explanation" not in st.session_state:
    st.session_state.explanation = ""
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "fold_map" not in st.session_state:
    st.session_state.fold_map = {}
if "adjusted_forecasts" not in st.session_state:
    st.session_state.adjusted_forecasts = {}
if "audit_trail" not in st.session_state:
    st.session_state.audit_trail = []
if "holiday_mode" not in st.session_state:
    st.session_state.holiday_mode = "None"

with tabs[0]:
    st.subheader("Upload Weekly Data")
    template_df = build_template()
    csv_bytes = template_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV Template", data=csv_bytes, file_name="forecast_template.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload file (CSV or Excel)", type=["csv", "xlsx", "xls"])
    if uploaded:
        try:
            raw = read_tabular(uploaded.name, uploaded.read())
            st.session_state.raw_df = raw
            st.write("Data preview")
            st.dataframe(raw.head(20), use_container_width=True)
        except Exception as ex:
            st.error(f"Could not read file: {ex}")

    raw_df = st.session_state.raw_df
    if raw_df is not None:
        inferred = infer_columns(raw_df)
        st.subheader("Column Mapping")
        all_cols = list(raw_df.columns)
        inferred_date = inferred.date_col if inferred and inferred.date_col in all_cols else all_cols[0]
        date_col = st.selectbox(
            "Date column",
            options=all_cols,
            index=all_cols.index(inferred_date),
        )
        inferred_target = inferred.target_col if inferred and inferred.target_col in all_cols else all_cols[0]
        target_col = st.selectbox(
            "Demand column",
            options=all_cols,
            index=all_cols.index(inferred_target),
        )
        optional_options = [c for c in all_cols if c not in (date_col, target_col)]
        inferred_optional = inferred.optional_cols if inferred else []
        safe_optional_default = [c for c in inferred_optional if c in optional_options]
        optional_cols = st.multiselect(
            "Optional dimensions / external regressors",
            options=optional_options,
            default=safe_optional_default,
        )

        if st.button("Validate Data", type="primary"):
            report = validate_and_prepare(raw_df, date_col, target_col, optional_cols)
            weekly_exog = prepare_weekly_exog(raw_df, date_col, optional_cols)
            clean = report.cleaned_df.merge(weekly_exog, how="left", on="date")
            st.session_state.validated = {"report": report, "clean_df": clean, "date_col": date_col, "target_col": target_col}
            logger.info("Validation completed. Summary=%s", report.summary)

with tabs[1]:
    st.subheader("Data Health Diagnostics")
    validated = st.session_state.validated
    if not validated:
        st.info("Upload and validate data first.")
    else:
        report = validated["report"]
        clean = validated["clean_df"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("History (weeks)", report.summary["history_weeks"])
        c2.metric("Missing weeks", report.summary["missing_weeks"])
        c3.metric("Outliers", report.summary["outlier_count"])
        c4.metric("Negative values", report.summary["negative_count"])
        _render_validation_report(report)

        fig = px.line(clean, x="date", y="y", title="Demand Trend")
        flagged = clean[clean["is_outlier"] | clean["is_change_point"]]
        if not flagged.empty:
            fig.add_scatter(
                x=flagged["date"],
                y=flagged["y"],
                mode="markers",
                marker=dict(color="red", size=8),
                name="Outlier/Change point",
            )
        st.plotly_chart(fig, use_container_width=True)

        growth = compute_growth_metrics(clean)
        st.plotly_chart(px.line(growth, x="date", y=["wow_growth", "rolling_4w_growth", "rolling_13w_growth"], title="Growth Rates"), use_container_width=True)

        stationarity = stationarity_indicator(clean["y"])
        st.write("Stationarity indicator", stationarity)

with tabs[2]:
    st.subheader("Feature Engineering and Seasonality")
    validated = st.session_state.validated
    if not validated:
        st.info("Upload and validate data first.")
    else:
        clean = validated["clean_df"]
        periods = detect_seasonality_periods(clean["y"])
        st.write("Detected seasonal periods", periods)
        decomp = decompose_weekly(clean.set_index("date")["y"], period=52)
        if decomp is not None:
            decomp_reset = decomp.reset_index(names="date")
            st.plotly_chart(px.line(decomp_reset, x="date", y="trend", title="Trend Component"), use_container_width=True)
            st.plotly_chart(px.line(decomp_reset, x="date", y="seasonal", title="Seasonal Component"), use_container_width=True)
            st.plotly_chart(px.line(decomp_reset, x="date", y="resid", title="Residual Component"), use_container_width=True)
        indices = seasonal_indices(clean)
        st.plotly_chart(px.bar(indices, x="week_of_year", y="seasonality_index", title="Seasonality Index by Week"), use_container_width=True)

with tabs[3]:
    st.subheader("Modeling and Forecast Generation")
    validated = st.session_state.validated
    if not validated:
        st.info("Upload and validate data first.")
    else:
        clean = validated["clean_df"].copy()
        horizon = st.slider("Forecast horizon (weeks)", min_value=MIN_HORIZON_WEEKS, max_value=MAX_HORIZON_WEEKS, value=104, step=1)
        model_map = available_models()
        auto_mode = st.checkbox("Auto mode (recommended)", value=True)
        selected_models = st.multiselect("Select models", options=list(model_map.keys()), default=list(model_map.keys()) if auto_mode else ["seasonal_naive", "ets"])
        holiday_mode = st.selectbox("Holiday calendar", options=["None", "US", "Custom (future extension)"], index=0)
        st.session_state.holiday_mode = holiday_mode
        st.caption(f"Holiday mode selected: {holiday_mode}. Calendar effects are applied through exogenous regressors when provided.")

        if st.button("Generate Forecasts", type="primary"):
            report = validated["report"]
            has_blocking_quality = any(i.level == "error" for i in report.issues)
            if has_blocking_quality and auto_mode:
                selected_models = ["seasonal_naive", "moving_average", "drift", "ets"]
                st.warning("Data quality issues detected. Auto mode switched to robust model subset.")

            holiday_hist = build_holiday_features(clean["date"].min(), clean["date"].max(), mode=holiday_mode)
            model_input = clean.merge(holiday_hist, on="date", how="left")
            future_dates = pd.date_range(model_input["date"].max() + pd.Timedelta(weeks=1), periods=horizon, freq="W-MON")
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
            fig_preview = go.Figure()
            fig_preview.add_trace(go.Scatter(x=clean["date"], y=clean["y"], mode="lines", name="Actuals"))
            fig_preview.add_trace(go.Scatter(x=preview_df["date"], y=preview_df["forecast"], mode="lines", name=f"{preview_model} forecast"))
            fig_preview.add_trace(go.Scatter(x=preview_df["date"], y=preview_df["upper_95"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig_preview.add_trace(
                go.Scatter(
                    x=preview_df["date"],
                    y=preview_df["lower_95"],
                    fill="tonexty",
                    fillcolor="rgba(59,130,246,0.15)",
                    line=dict(width=0),
                    name="95% interval",
                )
            )
            fig_preview.update_layout(title=f"Forecast Preview ({preview_model})", template="plotly_white")
            st.plotly_chart(fig_preview, use_container_width=True)
            st.dataframe(preview_df.head(20), use_container_width=True)

with tabs[4]:
    st.subheader("Backtesting and Accuracy")
    validated = st.session_state.validated
    forecasts = st.session_state.forecasts
    if not validated or not forecasts:
        st.info("Generate forecasts first.")
    else:
        clean = validated["clean_df"]
        holdout_options = st.multiselect("Holdout windows", options=[13, 26, 52], default=[13, 26, 52])
        holdouts = tuple(sorted(holdout_options))
        model_map = available_models()

        if st.button("Run Backtests"):
            fold_map = {}
            for model_id in forecasts.keys():
                if model_id not in model_map:
                    continue
                folds = rolling_backtest(clean, model_id, model_map[model_id], holdouts=holdouts)
                fold_map[model_id] = folds
            st.session_state.fold_map = fold_map
            evaluations = summarize_evaluations(fold_map)
            rank_df = rank_models(evaluations)
            st.session_state.rank_df = rank_df

            if not rank_df.empty:
                baseline_candidates = rank_df[rank_df["model_id"].isin(["seasonal_naive", "moving_average", "drift", "ets"])]
                top_models = rank_df.head(3)["model_id"].tolist()
                if len(top_models) >= 2:
                    st.session_state.forecasts["ensemble_top"] = forecast_ensemble([st.session_state.forecasts[m] for m in top_models if m in st.session_state.forecasts])
                st.dataframe(rank_df, use_container_width=True)
                st.success(f"Top model: {rank_df.iloc[0]['model_id']}")
                top_model = rank_df.iloc[0]["model_id"]
                st.session_state.selected_model = top_model
                base_row = baseline_candidates.iloc[0] if not baseline_candidates.empty else None
                issues = [f"{i.check}:{i.message}" for i in validated["report"].issues]
                st.session_state.explanation = build_explanation(rank_df.iloc[0], base_row, issues)
            else:
                st.warning("No valid backtest results; consider shorter holdouts or more history.")
        if not st.session_state.rank_df.empty:
            model_for_plot = st.selectbox("Backtest plot model", options=st.session_state.rank_df["model_id"].tolist())
            folds = st.session_state.fold_map.get(model_for_plot, [])
            if folds:
                fold = sorted(folds, key=lambda x: x.holdout_weeks)[-1]
                bt_df = pd.DataFrame(
                    {
                        "date": pd.to_datetime(fold.dates),
                        "actual": fold.actual,
                        "predicted": fold.predicted,
                        "lower_95": fold.lower_95,
                        "upper_95": fold.upper_95,
                    }
                )
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=bt_df["date"], y=bt_df["actual"], mode="lines", name="Actual"))
                fig_bt.add_trace(go.Scatter(x=bt_df["date"], y=bt_df["predicted"], mode="lines", name="Predicted"))
                fig_bt.add_trace(go.Scatter(x=bt_df["date"], y=bt_df["upper_95"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
                fig_bt.add_trace(
                    go.Scatter(
                        x=bt_df["date"],
                        y=bt_df["lower_95"],
                        fill="tonexty",
                        fillcolor="rgba(16,185,129,0.12)",
                        line=dict(width=0),
                        name="Backtest 95% interval",
                    )
                )
                fig_bt.update_layout(title=f"Backtest: Predicted vs Actual ({model_for_plot})", template="plotly_white")
                st.plotly_chart(fig_bt, use_container_width=True)

with tabs[5]:
    st.subheader("Explainability and Recommendation")
    validated = st.session_state.validated
    forecasts = st.session_state.forecasts
    rank_df = st.session_state.rank_df
    selected_model = st.session_state.selected_model

    if not validated or not forecasts:
        st.info("Run forecasting and backtesting first.")
    else:
        clean = validated["clean_df"]
        if selected_model is None:
            selected_model = list(forecasts.keys())[0]
            st.session_state.selected_model = selected_model

        _plot_actuals_forecast(clean, list(forecasts.values()), selected_model)
        st.markdown("**Model recommendation rationale**")
        st.write(st.session_state.explanation or "Model selected based on backtest performance, stability, and simplicity.")

        if not rank_df.empty:
            selected = rank_df[rank_df["model_id"] == selected_model]
            if not selected.empty:
                st.write("Selected model metrics", selected.iloc[0].to_dict())
        selected_folds = st.session_state.fold_map.get(selected_model, [])
        if selected_folds:
            fold = sorted(selected_folds, key=lambda x: x.holdout_weeks)[-1]
            resid = np.array(fold.actual) - np.array(fold.predicted)
            resid_df = pd.DataFrame({"step": np.arange(1, len(resid) + 1), "residual": resid})
            st.plotly_chart(px.line(resid_df, x="step", y="residual", title="Residual Diagnostics"), use_container_width=True)
            max_lag = min(20, len(resid) - 1)
            if max_lag > 1:
                acf_vals = [1.0]
                for lag in range(1, max_lag + 1):
                    acf_vals.append(float(np.corrcoef(resid[:-lag], resid[lag:])[0, 1]))
                acf_df = pd.DataFrame({"lag": np.arange(0, max_lag + 1), "acf": acf_vals})
                st.plotly_chart(px.bar(acf_df, x="lag", y="acf", title="Residual Autocorrelation"), use_container_width=True)

with tabs[6]:
    st.subheader("Export and Reporting")
    validated = st.session_state.validated
    forecasts = st.session_state.forecasts
    rank_df = st.session_state.rank_df
    selected_model = st.session_state.selected_model

    if not validated or not forecasts or selected_model is None:
        st.info("Generate and evaluate forecasts first.")
    else:
        clean = validated["clean_df"]
        forecast_df = st.session_state.adjusted_forecasts.get(selected_model, forecasts[selected_model]).copy()

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
            mask = (forecast_df["date"] >= pd.to_datetime(start_date)) & (forecast_df["date"] <= pd.to_datetime(end_date))
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

        forecast_table = build_forecast_table(clean, forecast_df)
        growth = compute_growth_metrics(clean)
        summary = build_one_page_summary(
            selected_model=selected_model,
            horizon=len(forecast_df),
            rank_df=rank_df,
            validation_summary=validated["report"].summary,
            explanation=st.session_state.explanation,
            growth_outlook={
                "wow_growth_latest": float(growth["wow_growth"].dropna().iloc[-1]) if not growth["wow_growth"].dropna().empty else None,
                "trailing_52w_growth_latest": float(growth["trailing_52w_growth"].dropna().iloc[-1]) if not growth["trailing_52w_growth"].dropna().empty else None,
            },
        )
        issues_df = pd.DataFrame([{"level": i.level, "check": i.check, "message": i.message, "details": str(i.details)} for i in validated["report"].issues])
        audit_df = pd.DataFrame(st.session_state.audit_trail)

        st.dataframe(forecast_table.tail(30), use_container_width=True)
        st.dataframe(rank_df, use_container_width=True)
        st.dataframe(summary, use_container_width=True)
        if not audit_df.empty:
            st.dataframe(audit_df, use_container_width=True)

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

st.sidebar.header("Run Controls")
st.sidebar.write("All operations are logged and deterministic via fixed random seed.")
if st.sidebar.button("Reset Session"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

if st.session_state.validated:
    clean = st.session_state.validated["clean_df"]
    if len(clean) >= 52:
        years = len(clean) / 52
        v = cagr(clean["y"], years)
        st.sidebar.metric("Historical CAGR", f"{v:.2%}" if v is not None else "N/A")
