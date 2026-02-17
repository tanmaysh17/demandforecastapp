"""Page 1 - Upload & Mapping: file upload, column mapping, validation settings."""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Upload & Mapping | Demand Forecasting Studio", layout="wide")

from app.ui.shared import setup_page  # noqa: E402

setup_page()

import logging  # noqa: E402

import plotly.express as px  # noqa: E402

from app.core.config import ValidationThresholds  # noqa: E402
from app.services.data_loader import build_template, infer_columns, read_tabular  # noqa: E402
from app.services.features import prepare_weekly_exog  # noqa: E402
from app.services.validation import validate_and_prepare  # noqa: E402

logger = logging.getLogger("forecasting_app")

st.header("1) Upload & Mapping")

# --- Template download ---
template_df = build_template()
csv_bytes = template_df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV Template", data=csv_bytes, file_name="forecast_template.csv", mime="text/csv")

# --- File upload ---
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
    date_col = st.selectbox("Date column", options=all_cols, index=all_cols.index(inferred_date))

    inferred_target = inferred.target_col if inferred and inferred.target_col in all_cols else all_cols[0]
    target_col = st.selectbox("Demand column", options=all_cols, index=all_cols.index(inferred_target))

    optional_options = [c for c in all_cols if c not in (date_col, target_col)]
    inferred_optional = inferred.optional_cols if inferred else []
    safe_optional_default = [c for c in inferred_optional if c in optional_options]
    optional_cols = st.multiselect(
        "Optional dimensions / external regressors",
        options=optional_options,
        default=safe_optional_default,
    )

    # --- Demand preview chart (reactive — updates as user changes dropdowns) ---
    try:
        preview_df = raw_df[[date_col, target_col]].dropna().copy()
        preview_df[date_col] = __import__("pandas").to_datetime(preview_df[date_col], errors="coerce")
        preview_df = preview_df.dropna(subset=[date_col]).sort_values(date_col)
        if not preview_df.empty:
            fig_prev = px.line(
                preview_df,
                x=date_col,
                y=target_col,
                title=f"Preview: {target_col} over time",
                labels={date_col: "Date", target_col: "Demand"},
            )
            fig_prev.update_layout(
                template="plotly_white",
                yaxis=dict(rangemode="tozero"),
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig_prev, use_container_width=True)
            st.caption(
                "This chart updates as you change the column selections above. "
                "Verify the trend looks as expected before proceeding to validation."
            )
    except Exception:
        pass  # Silently skip if column types aren't compatible yet

    # --- Data preparation settings ---
    st.subheader("Data Preparation Settings")
    c1, c2 = st.columns(2)
    with c1:
        imputation_method = st.selectbox(
            "Missing week imputation",
            options=["linear", "ffill", "zero"],
            index=0,
            help="How to fill weeks with no data: linear interpolation, forward-fill, or zero.",
        )
        outlier_action = st.selectbox(
            "Outlier handling",
            options=["flag", "remove", "cap"],
            index=0,
            help="flag=mark but keep, remove=replace with interpolated value, cap=winsorise to 3.5·MAD.",
        )

    with c2:
        with st.expander("Advanced: Validation Thresholds"):
            max_missing = st.slider("Max allowed missing ratio", 0.01, 0.20, 0.03, step=0.01,
                                    help="Fraction of weeks that may be missing before raising an issue.")
            max_outlier = st.slider("Max allowed outlier ratio", 0.01, 0.20, 0.08, step=0.01,
                                    help="Fraction of weeks flagged as outliers before raising an issue.")
            min_history = st.number_input("Min history weeks (degraded mode threshold)", 26, 208, 104, step=4,
                                          help="Below this, only simple models are available (degraded mode).")
            min_full_models = st.number_input("Min history for full models", 52, 312, 156, step=4,
                                              help="Below this, a warning is raised for complex models.")

    thresholds = ValidationThresholds(
        max_missing_ratio=max_missing,
        max_outlier_ratio=max_outlier,
        min_history_weeks=int(min_history),
        min_history_for_full_models=int(min_full_models),
    )

    if st.button("Validate Data", type="primary"):
        report = validate_and_prepare(
            raw_df,
            date_col,
            target_col,
            optional_cols,
            thresholds=thresholds,
            imputation_method=imputation_method,
            outlier_action=outlier_action,
        )
        weekly_exog = prepare_weekly_exog(raw_df, date_col, optional_cols)
        clean = report.cleaned_df.merge(weekly_exog, how="left", on="date")
        st.session_state.validated = {
            "report": report,
            "clean_df": clean,
            "date_col": date_col,
            "target_col": target_col,
        }
        # Reset downstream state when data changes
        st.session_state.forecasts = {}
        st.session_state.rank_df = __import__("pandas").DataFrame()
        st.session_state.fold_map = {}
        st.session_state.backtest_errors = {}
        st.session_state.selected_model = None
        st.session_state.explanation = ""
        logger.info("Validation completed. Summary=%s", report.summary)

    if st.session_state.validated:
        report = st.session_state.validated["report"]
        summary = getattr(report, "summary", {}) if not isinstance(report, dict) else report.get("summary", {})
        degraded = summary.get("degraded_mode", False)
        if degraded:
            st.warning(
                f"Degraded mode: only {summary.get('history_weeks', '?')} weeks of history — "
                "advanced models (ETS, SARIMAX, ML) will be restricted on the Forecast page."
            )
        else:
            st.success("Data validated successfully. Proceed to the EDA page.")
