from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import periodogram
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller


def compute_growth_metrics(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["wow_growth"] = work["y"].pct_change(1)
    work["rolling_4w_growth"] = work["y"].rolling(4).sum().pct_change(4)
    work["rolling_13w_growth"] = work["y"].rolling(13).sum().pct_change(13)
    work["trailing_52w"] = work["y"].rolling(52).sum()
    work["trailing_52w_prior"] = work["trailing_52w"].shift(52)
    work["trailing_52w_growth"] = (work["trailing_52w"] / work["trailing_52w_prior"]) - 1
    work["yoy_growth"] = work["y"] / work["y"].shift(52) - 1
    return work


def cagr(series: pd.Series, years: float) -> float | None:
    if years <= 0 or len(series) < 2:
        return None
    start, end = float(series.iloc[0]), float(series.iloc[-1])
    if start <= 0 or end <= 0:
        return None
    return (end / start) ** (1 / years) - 1


def seasonal_indices(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["week_of_year"] = work["date"].dt.isocalendar().week.astype(int)
    seasonality = work.groupby("week_of_year")["y"].mean()
    index = seasonality / seasonality.mean()
    return index.reset_index(name="seasonality_index")


def detect_seasonality_periods(series: pd.Series, top_k: int = 3) -> list[int]:
    values = series.values.astype(float)
    if len(values) < 30:
        return []
    freqs, power = periodogram(values - np.mean(values))
    candidates: list[tuple[int, float]] = []
    for f, p in zip(freqs[1:], power[1:]):
        if f <= 0:
            continue
        period = int(round(1 / f))
        if 2 <= period <= 104:
            candidates.append((period, p))
    candidates.sort(key=lambda x: x[1], reverse=True)
    periods: list[int] = []
    for period, _ in candidates:
        if all(abs(period - existing) > 2 for existing in periods):
            periods.append(period)
        if len(periods) >= top_k:
            break
    if 52 not in periods and len(series) >= 104:
        periods.insert(0, 52)
    return periods[:top_k]


def decompose_weekly(series: pd.Series, period: int = 52) -> pd.DataFrame | None:
    if len(series) < period * 2:
        return None
    result = seasonal_decompose(series, model="additive", period=period, extrapolate_trend="freq")
    return pd.DataFrame(
        {
            "trend": result.trend,
            "seasonal": result.seasonal,
            "resid": result.resid,
        },
        index=series.index,
    )


def stationarity_indicator(series: pd.Series) -> dict[str, float | str]:
    clean = series.dropna()
    if len(clean) < 30:
        return {"status": "insufficient_data"}
    stat, pval, *_ = adfuller(clean, autolag="AIC")
    return {"adf_stat": float(stat), "p_value": float(pval), "status": "stationary" if pval < 0.05 else "non_stationary"}

