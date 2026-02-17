from __future__ import annotations

import pandas as pd

from app.services.models import available_models
from app.services.validation import validate_and_prepare


def _sample_df(n: int = 160) -> pd.DataFrame:
    dates = pd.date_range("2021-01-04", periods=n, freq="W-MON")
    y = [100 + i * 0.2 + (i % 52) * 0.8 for i in range(n)]
    return pd.DataFrame({"week_start_date": dates, "demand": y})


def test_validation_produces_clean_weekly_series():
    df = _sample_df()
    report = validate_and_prepare(df, "week_start_date", "demand")
    assert "history_weeks" in report.summary
    assert report.cleaned_df["y"].isna().sum() == 0


def test_models_generate_horizon():
    df = _sample_df()
    report = validate_and_prepare(df, "week_start_date", "demand")
    clean = report.cleaned_df[["date", "y"]]
    horizon = 52
    for model_id, model_fn in available_models().items():
        out = model_fn(clean, horizon)
        assert len(out) == horizon, f"{model_id} failed horizon length"
        assert {"forecast", "lower_80", "upper_80", "lower_95", "upper_95"}.issubset(set(out.columns))

