from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.config import RankingWeights
from app.core.types import ModelEvaluation


MODEL_COMPLEXITY = {
    "seasonal_naive": 0.1,
    "moving_average": 0.15,
    "drift": 0.2,
    "ets": 0.35,
    "sarimax": 0.6,
    "ml_lag_rf": 0.75,
    "ensemble_top": 0.8,
}


def summarize_evaluations(fold_map: dict[str, list]) -> list[ModelEvaluation]:
    evaluations: list[ModelEvaluation] = []
    for model_id, folds in fold_map.items():
        if not folds:
            continue
        metric_df = pd.DataFrame([f.metrics for f in folds])
        avg = metric_df.mean(numeric_only=True).to_dict()
        std = metric_df.std(numeric_only=True).fillna(0.0).to_dict()
        evaluations.append(
            ModelEvaluation(
                model_id=model_id,
                avg_metrics={k: float(v) for k, v in avg.items()},
                std_metrics={k: float(v) for k, v in std.items()},
                fold_results=folds,
                complexity_score=MODEL_COMPLEXITY.get(model_id, 0.5),
            )
        )
    return evaluations


def rank_models(evaluations: list[ModelEvaluation], weights: RankingWeights | None = None) -> pd.DataFrame:
    weights = weights or RankingWeights()
    if not evaluations:
        return pd.DataFrame()

    rows = []
    for e in evaluations:
        error = 0.6 * e.avg_metrics.get("smape", np.nan) + 0.4 * e.avg_metrics.get("mase", np.nan) * 100
        stability = 0.6 * e.std_metrics.get("smape", 0.0) + 0.4 * e.std_metrics.get("mase", 0.0) * 100
        coverage_penalty = (
            abs(0.8 - e.avg_metrics.get("coverage_80", 0.0)) + abs(0.95 - e.avg_metrics.get("coverage_95", 0.0))
        ) * 100
        composite = (
            weights.error_weight * error
            + weights.stability_weight * stability
            + weights.coverage_weight * coverage_penalty
            + weights.complexity_penalty_weight * e.complexity_score * 100
        )
        rows.append(
            {
                "model_id": e.model_id,
                "smape": e.avg_metrics.get("smape", np.nan),
                "mase": e.avg_metrics.get("mase", np.nan),
                "rmse": e.avg_metrics.get("rmse", np.nan),
                "coverage_80": e.avg_metrics.get("coverage_80", np.nan),
                "coverage_95": e.avg_metrics.get("coverage_95", np.nan),
                "stability_smape_std": e.std_metrics.get("smape", np.nan),
                "complexity_score": e.complexity_score,
                "composite_score": composite,
            }
        )
    rank_df = pd.DataFrame(rows).sort_values("composite_score", ascending=True).reset_index(drop=True)
    rank_df["rank"] = np.arange(1, len(rank_df) + 1)
    return rank_df


def build_explanation(selected_row: pd.Series, baseline_row: pd.Series | None, issues: list[str]) -> str:
    model = selected_row["model_id"]
    smape = selected_row["smape"]
    stable = selected_row["stability_smape_std"]
    text = [
        f"Selected model: {model}.",
        f"It achieved average sMAPE of {smape:.2f}% and fold-to-fold stability (sMAPE std) of {stable:.2f}.",
    ]
    if baseline_row is not None:
        delta = baseline_row["smape"] - smape
        text.append(
            f"Compared with the best baseline, the selected model improved sMAPE by {delta:.2f} percentage points."
            if delta > 0
            else "Selected model performs similarly to baseline; simplicity and stability were prioritized."
        )
    if issues:
        text.append(f"Data quality considerations: {'; '.join(issues)}.")
    text.append("Forecast reliability assumes future demand drivers remain comparable to historical patterns.")
    return " ".join(text)

