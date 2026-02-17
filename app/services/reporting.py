from __future__ import annotations

import io
from datetime import datetime

import pandas as pd


def build_forecast_table(
    history: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> pd.DataFrame:
    hist = history[["date", "y"]].copy()
    hist["actual"] = hist["y"]
    hist["forecast"] = pd.NA
    hist["lower_80"] = pd.NA
    hist["upper_80"] = pd.NA
    hist["lower_95"] = pd.NA
    hist["upper_95"] = pd.NA
    hist["model_id"] = "actual"
    hist = hist.drop(columns=["y"])
    combined = pd.concat([hist, forecast_df], ignore_index=True, sort=False)
    return combined[
        ["date", "actual", "forecast", "lower_80", "upper_80", "lower_95", "upper_95", "model_id"]
    ].sort_values("date")


def build_one_page_summary(
    selected_model: str,
    horizon: int,
    rank_df: pd.DataFrame,
    validation_summary: dict,
    explanation: str,
    growth_outlook: dict[str, float | None],
) -> pd.DataFrame:
    best_smape = rank_df.iloc[0]["smape"] if not rank_df.empty else None
    data = {
        "generated_at": datetime.utcnow().isoformat(),
        "selected_model": selected_model,
        "horizon_weeks": horizon,
        "methodology": "Weekly time-series CV with baseline and advanced models; ranked by error, stability, coverage, and simplicity.",
        "best_smape": best_smape,
        "history_weeks": validation_summary.get("history_weeks"),
        "missing_weeks": validation_summary.get("missing_weeks"),
        "outlier_count": validation_summary.get("outlier_count"),
        "growth_wow_latest": growth_outlook.get("wow_growth_latest"),
        "growth_t52_latest": growth_outlook.get("trailing_52w_growth_latest"),
        "assumptions_and_risks": explanation,
    }
    return pd.DataFrame([data])


def export_to_excel(
    forecast_table: pd.DataFrame,
    comparison_table: pd.DataFrame,
    summary_table: pd.DataFrame,
    validation_issues: pd.DataFrame,
    audit_trail: pd.DataFrame | None = None,
) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        forecast_table.to_excel(writer, index=False, sheet_name="forecast")
        comparison_table.to_excel(writer, index=False, sheet_name="model_comparison")
        summary_table.to_excel(writer, index=False, sheet_name="summary")
        validation_issues.to_excel(writer, index=False, sheet_name="data_quality")
        if audit_trail is not None and not audit_trail.empty:
            audit_trail.to_excel(writer, index=False, sheet_name="manual_adjustments")
    buffer.seek(0)
    return buffer.read()
