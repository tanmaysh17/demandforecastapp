from __future__ import annotations

from dataclasses import dataclass


RANDOM_SEED = 42
MIN_HORIZON_WEEKS = 52
MAX_HORIZON_WEEKS = 204
DEFAULT_HOLDOUTS = (13, 26, 52)
WEEKLY_FREQ = "W-MON"


@dataclass(frozen=True)
class ValidationThresholds:
    max_missing_ratio: float = 0.03
    max_outlier_ratio: float = 0.08
    min_history_weeks: int = 104
    min_history_for_full_models: int = 156
    max_negative_ratio: float = 0.05


@dataclass(frozen=True)
class RankingWeights:
    error_weight: float = 0.55
    stability_weight: float = 0.25
    coverage_weight: float = 0.15
    complexity_penalty_weight: float = 0.05

