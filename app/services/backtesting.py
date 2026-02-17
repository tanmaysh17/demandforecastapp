from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.config import DEFAULT_HOLDOUTS
from app.core.types import BacktestFoldResult


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    denom = np.where(denom == 0, 1e-9, denom)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def mase(y_true: np.ndarray, y_pred: np.ndarray, insample: np.ndarray, m: int = 52) -> float:
    if len(insample) <= m:
        scale = np.mean(np.abs(np.diff(insample))) if len(insample) > 1 else 1.0
    else:
        scale = np.mean(np.abs(insample[m:] - insample[:-m]))
    scale = max(float(scale), 1e-9)
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower_80: np.ndarray,
    upper_80: np.ndarray,
    lower_95: np.ndarray,
    upper_95: np.ndarray,
    insample: np.ndarray,
) -> dict[str, float]:
    return {
        "smape": smape(y_true, y_pred),
        "mase": mase(y_true, y_pred, insample),
        "rmse": rmse(y_true, y_pred),
        "coverage_80": interval_coverage(y_true, lower_80, upper_80),
        "coverage_95": interval_coverage(y_true, lower_95, upper_95),
    }


def rolling_backtest(
    df: pd.DataFrame,
    model_id: str,
    model_fn,
    holdouts: tuple[int, ...] = DEFAULT_HOLDOUTS,
) -> tuple[list[BacktestFoldResult], list[str]]:
    """Run rolling time-series backtests.

    Returns (fold_results, errors) where errors is a list of human-readable
    failure messages for any holdout fold that could not be evaluated.
    """
    results: list[BacktestFoldResult] = []
    errors: list[str] = []
    n = len(df)
    for h in holdouts:
        if n <= h + 30:
            errors.append(f"holdout={h}w skipped: not enough history ({n} rows, need >{h + 30})")
            continue
        train = df.iloc[: n - h].copy()
        test = df.iloc[n - h :].copy()
        try:
            pred = model_fn(train, h)
            metrics = evaluate_predictions(
                y_true=test["y"].values,
                y_pred=pred["forecast"].values,
                lower_80=pred["lower_80"].values,
                upper_80=pred["upper_80"].values,
                lower_95=pred["lower_95"].values,
                upper_95=pred["upper_95"].values,
                insample=train["y"].values,
            )
            results.append(BacktestFoldResult(model_id=model_id, holdout_weeks=h, metrics=metrics))
            results[-1].dates = list(test["date"].values)
            results[-1].actual = list(test["y"].astype(float).values)
            results[-1].predicted = list(pred["forecast"].astype(float).values)
            results[-1].lower_95 = list(pred["lower_95"].astype(float).values)
            results[-1].upper_95 = list(pred["upper_95"].astype(float).values)
        except Exception as ex:
            errors.append(f"holdout={h}w failed: {type(ex).__name__}: {ex}")
            continue
    return results, errors
