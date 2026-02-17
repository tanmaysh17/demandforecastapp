from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class ValidationIssue:
    level: str
    check: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    issues: list[ValidationIssue]
    summary: dict[str, Any]
    cleaned_df: pd.DataFrame


@dataclass
class ModelForecast:
    model_id: str
    forecast_df: pd.DataFrame
    fit_summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestFoldResult:
    model_id: str
    holdout_weeks: int
    metrics: dict[str, float]
    dates: list[pd.Timestamp] = field(default_factory=list)
    actual: list[float] = field(default_factory=list)
    predicted: list[float] = field(default_factory=list)
    lower_95: list[float] = field(default_factory=list)
    upper_95: list[float] = field(default_factory=list)


@dataclass
class ModelEvaluation:
    model_id: str
    avg_metrics: dict[str, float]
    std_metrics: dict[str, float]
    fold_results: list[BacktestFoldResult]
    complexity_score: float
