from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from app.core.config import ValidationThresholds, WEEKLY_FREQ
from app.core.types import ValidationIssue, ValidationReport


def _mad_outliers(series: pd.Series, z: float = 3.5) -> pd.Series:
    values = series.astype(float)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return pd.Series(False, index=series.index)
    modified_z = 0.6745 * (values - median) / mad
    return pd.Series(np.abs(modified_z) > z, index=series.index)


def _change_point_flags(series: pd.Series, window: int = 13) -> pd.Series:
    if len(series) < (window * 3):
        return pd.Series(False, index=series.index)
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    mean_shift = rolling_mean.diff(window).abs()
    threshold = mean_shift.mean(skipna=True) + 2 * mean_shift.std(skipna=True)
    flags = mean_shift > threshold
    return flags.reindex(series.index, fill_value=False)


def validate_and_prepare(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    optional_cols: list[str] | None = None,
    thresholds: ValidationThresholds | None = None,
    imputation_method: str = "linear",
    outlier_action: str = "flag",
) -> ValidationReport:
    """Validate and prepare weekly demand data.

    Parameters
    ----------
    imputation_method:
        How to fill missing weeks. One of ``"linear"`` (default, interpolate),
        ``"ffill"`` (forward-fill then back-fill), or ``"zero"`` (fill with 0).
    outlier_action:
        What to do with MAD-detected outliers. One of ``"flag"`` (default,
        mark but keep), ``"remove"`` (replace with interpolated value), or
        ``"cap"`` (winsorise to median ± 3.5·MAD).
    """
    thresholds = thresholds or ValidationThresholds()
    optional_cols = optional_cols or []
    issues: list[ValidationIssue] = []

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    if work[date_col].isna().any():
        issues.append(
            ValidationIssue(
                level="error",
                check="date_parse",
                message="Some date values could not be parsed and were removed.",
                details={"rows_removed": int(work[date_col].isna().sum())},
            )
        )
        work = work.dropna(subset=[date_col])

    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    if work[target_col].isna().any():
        issues.append(
            ValidationIssue(
                level="error",
                check="target_numeric",
                message="Non-numeric target values found; affected rows were removed.",
                details={"rows_removed": int(work[target_col].isna().sum())},
            )
        )
        work = work.dropna(subset=[target_col])

    dup_cols = [date_col] + [c for c in optional_cols if c in work.columns]
    duplicate_mask = work.duplicated(subset=dup_cols, keep=False)
    if duplicate_mask.any():
        issues.append(
            ValidationIssue(
                level="warning",
                check="duplicates",
                message="Duplicate rows detected for date and selected dimensions.",
                details={"duplicate_rows": int(duplicate_mask.sum())},
            )
        )
        work = work.groupby(dup_cols, as_index=False)[target_col].sum()

    work = work.sort_values(date_col)
    weekly = (
        work.set_index(date_col)
        .resample(WEEKLY_FREQ)[target_col]
        .sum(min_count=1)
        .to_frame(name=target_col)
    )
    missing_weeks = int(weekly[target_col].isna().sum())
    if missing_weeks > 0:
        method_label = {"linear": "linear interpolation", "ffill": "forward-fill", "zero": "zero-fill"}.get(
            imputation_method, imputation_method
        )
        issues.append(
            ValidationIssue(
                level="warning",
                check="missing_weeks",
                message=f"Missing weekly periods detected and imputed by {method_label}.",
                details={"missing_weeks": missing_weeks, "imputation_method": imputation_method},
            )
        )
        if imputation_method == "ffill":
            weekly[target_col] = weekly[target_col].ffill().bfill()
        elif imputation_method == "zero":
            weekly[target_col] = weekly[target_col].fillna(0.0)
        else:
            weekly[target_col] = weekly[target_col].interpolate(method="linear").bfill().ffill()

    negatives = (weekly[target_col] < 0).sum()
    negative_ratio = negatives / max(len(weekly), 1)
    if negatives > 0:
        lvl = "warning" if negative_ratio <= thresholds.max_negative_ratio else "error"
        issues.append(
            ValidationIssue(
                level=lvl,
                check="negative_values",
                message="Negative demand values detected.",
                details={"negative_rows": int(negatives), "negative_ratio": float(negative_ratio)},
            )
        )

    outliers = _mad_outliers(weekly[target_col])
    outlier_ratio = float(outliers.mean()) if len(outliers) else 0.0
    if outlier_ratio > 0:
        action_label = {"flag": "flagged", "remove": "removed (interpolated)", "cap": "capped at 3.5·MAD"}.get(
            outlier_action, "flagged"
        )
        lvl = "warning" if outlier_ratio <= thresholds.max_outlier_ratio else "error"
        issues.append(
            ValidationIssue(
                level=lvl,
                check="outliers",
                message=f"Outlier points detected with robust MAD method and {action_label}.",
                details={"outlier_count": int(outliers.sum()), "outlier_ratio": outlier_ratio, "outlier_action": outlier_action},
            )
        )
        if outlier_action == "remove" and outliers.any():
            weekly.loc[outliers, target_col] = np.nan
            weekly[target_col] = weekly[target_col].interpolate(method="linear").bfill().ffill()
            # Recompute outlier mask after removal (now all clean)
            outliers = pd.Series(False, index=weekly.index)
        elif outlier_action == "cap" and outliers.any():
            vals = weekly[target_col].astype(float)
            median = float(np.median(vals))
            mad = float(np.median(np.abs(vals - median)))
            half_range = 3.5 * mad / 0.6745
            weekly.loc[outliers, target_col] = weekly.loc[outliers, target_col].clip(
                lower=median - half_range, upper=median + half_range
            )
            outliers = pd.Series(False, index=weekly.index)

    cp_flags = _change_point_flags(weekly[target_col])
    if cp_flags.any():
        issues.append(
            ValidationIssue(
                level="warning",
                check="structural_breaks",
                message="Potential structural breaks or level shifts detected.",
                details={"candidate_break_points": int(cp_flags.sum())},
            )
        )

    history_weeks = len(weekly)
    degraded_mode = history_weeks < thresholds.min_history_weeks
    if degraded_mode:
        issues.append(
            ValidationIssue(
                level="warning",
                check="sparse_history",
                message=(
                    "Insufficient history for robust yearly seasonality. "
                    "Degraded mode: only simple models (Seasonal Naive, Moving Average, Drift) will be available."
                ),
                details={"history_weeks": history_weeks, "minimum_recommended": thresholds.min_history_weeks},
            )
        )
    elif history_weeks < thresholds.min_history_for_full_models:
        issues.append(
            ValidationIssue(
                level="warning",
                check="limited_history",
                message="History length is limited; robust/simple models may be preferred.",
                details={
                    "history_weeks": history_weeks,
                    "minimum_for_full_models": thresholds.min_history_for_full_models,
                },
            )
        )

    cleaned = weekly.reset_index().rename(columns={date_col: "date", target_col: "y"})
    cleaned["is_outlier"] = outliers.values
    cleaned["is_change_point"] = cp_flags.values

    summary = {
        "rows_input": int(len(df)),
        "rows_cleaned": int(len(cleaned)),
        "history_weeks": int(history_weeks),
        "missing_weeks": missing_weeks,
        "outlier_count": int(outliers.sum()),
        "negative_count": int(negatives),
        "degraded_mode": degraded_mode,
        "thresholds": asdict(thresholds),
    }
    return ValidationReport(issues=issues, summary=summary, cleaned_df=cleaned)

