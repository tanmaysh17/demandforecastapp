"""Page 5 - Backtesting & Accuracy."""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Backtest | Demand Forecasting Studio", layout="wide")

from app.ui.shared import setup_page  # noqa: E402

setup_page()

import plotly.graph_objects as go  # noqa: E402
import pandas as pd  # noqa: E402

from app.services.backtesting import rolling_backtest  # noqa: E402
from app.services.models import available_models, forecast_ensemble  # noqa: E402
from app.services.selection import build_explanation, compute_ensemble_weights, rank_models, summarize_evaluations  # noqa: E402

st.header("5) Backtesting & Accuracy")

validated = st.session_state.validated
forecasts = st.session_state.forecasts
if not validated or not forecasts:
    st.info("Generate forecasts first (page 4).")
    st.stop()

clean = validated["clean_df"]
holdout_options = st.multiselect("Holdout windows", options=[13, 26, 52], default=[13, 26, 52])
holdouts = tuple(sorted(holdout_options))
model_map = available_models()

if st.button("Run Backtests", type="primary"):
    fold_map: dict = {}
    all_errors: dict[str, list[str]] = {}
    for model_id in forecasts:
        if model_id not in model_map:
            continue
        folds, errors = rolling_backtest(clean, model_id, model_map[model_id], holdouts=holdouts)
        fold_map[model_id] = folds
        if errors:
            all_errors[model_id] = errors

    st.session_state.fold_map = fold_map
    st.session_state.backtest_errors = all_errors

    # Surface any backtest failures
    if all_errors:
        with st.expander("Backtest warnings / failures", expanded=True):
            for mid, errs in all_errors.items():
                for e in errs:
                    st.warning(f"`{mid}`: {e}")

    evaluations = summarize_evaluations(fold_map)
    rank_df = rank_models(evaluations)
    st.session_state.rank_df = rank_df

    if not rank_df.empty:
        top_models = rank_df.head(3)["model_id"].tolist()
        top_forecasts = [forecasts[m] for m in top_models if m in forecasts]
        if len(top_forecasts) >= 2:
            weights_dict = compute_ensemble_weights(rank_df, top_n=len(top_forecasts))
            weights_list = [weights_dict.get(m, 1.0 / len(top_forecasts)) for m in top_models if m in forecasts]
            st.session_state.forecasts["ensemble_top"] = forecast_ensemble(top_forecasts, weights=weights_list)
            weight_info = ", ".join(f"{m}: {w:.1%}" for m, w in weights_dict.items())
            st.info(f"Ensemble weights (inverse-sMAPE): {weight_info}")

        top_model = rank_df.iloc[0]["model_id"]
        st.session_state.selected_model = top_model
        baseline_candidates = rank_df[rank_df["model_id"].isin(["seasonal_naive", "moving_average", "drift", "ets"])]
        base_row = baseline_candidates.iloc[0] if not baseline_candidates.empty else None
        report = validated.get("report")
        issues_raw = getattr(report, "issues", []) if not isinstance(report, dict) else report.get("issues", [])
        issues = []
        for i in issues_raw:
            if hasattr(i, "check"):
                issues.append(f"{i.check}: {i.message}")
            else:
                issues.append(f"{i['check']}: {i['message']}")
        st.session_state.explanation = build_explanation(rank_df.iloc[0], base_row, issues)
        st.success(f"Top model: {top_model}")
        st.dataframe(rank_df, use_container_width=True)
    else:
        st.warning("No valid backtest results; consider shorter holdouts or more history.")

# Show previously computed errors if page is revisited
if st.session_state.backtest_errors:
    with st.expander("Previous backtest warnings"):
        for mid, errs in st.session_state.backtest_errors.items():
            for e in errs:
                st.warning(f"`{mid}`: {e}")

# --- Backtest plot ---
rank_df = st.session_state.rank_df
if not rank_df.empty:
    st.dataframe(rank_df, use_container_width=True)
    model_for_plot = st.selectbox("Backtest plot model", options=rank_df["model_id"].tolist())
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
        st.caption(
            "This chart compares what the model would have predicted vs what actually happened "
            "during the holdout window. Closer tracking = better in-sample accuracy. "
            "Wide intervals that capture the actuals = well-calibrated uncertainty."
        )
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(
            x=bt_df["date"], y=bt_df["actual"], mode="lines", name="Actual",
            line=dict(color="#1f2937"),
        ))
        fig_bt.add_trace(go.Scatter(
            x=bt_df["date"], y=bt_df["predicted"], mode="lines", name="Predicted",
            line=dict(color="#0ea5e9", dash="dash"),
        ))
        fig_bt.add_trace(
            go.Scatter(x=bt_df["date"], y=bt_df["upper_95"], line=dict(width=0), showlegend=False, hoverinfo="skip")
        )
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
        fig_bt.update_layout(
            title=f"Backtest: Predicted vs Actual ({model_for_plot})",
            template="plotly_white",
            yaxis=dict(rangemode="tozero"),
        )
        st.plotly_chart(fig_bt, use_container_width=True)
