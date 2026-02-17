from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from app.services.features import make_lagged_features

warnings.filterwarnings("ignore")


def _interval_bounds(
    point_forecast: np.ndarray,
    residuals: np.ndarray,
    horizon: int,
    coverage: float,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = 1 - coverage
    z = norm.ppf(1 - alpha / 2)
    sigma = np.std(residuals) if len(residuals) > 1 else np.std(point_forecast) * 0.15
    sigma = max(float(sigma), 1e-6)
    scale = sigma * np.sqrt(np.arange(1, horizon + 1))
    lower = point_forecast - z * scale
    upper = point_forecast + z * scale
    return lower, upper


def _build_output(model_id: str, dates: pd.DatetimeIndex, point_forecast: np.ndarray, residuals: np.ndarray) -> pd.DataFrame:
    f = np.asarray(point_forecast, dtype=float)
    l80, u80 = _interval_bounds(f, residuals, len(f), 0.80)
    l95, u95 = _interval_bounds(f, residuals, len(f), 0.95)
    return pd.DataFrame(
        {
            "date": dates,
            "forecast": f,
            "lower_80": l80,
            "upper_80": u80,
            "lower_95": l95,
            "upper_95": u95,
            "model_id": model_id,
        }
    )


def forecast_seasonal_naive(train: pd.DataFrame, horizon: int) -> pd.DataFrame:
    series = train["y"].values
    seasonal_period = 52 if len(series) >= 52 else max(1, len(series))
    base = series[-seasonal_period:]
    repeated = np.resize(base, horizon)
    dates = pd.date_range(train["date"].max() + pd.Timedelta(weeks=1), periods=horizon, freq="W-MON")
    residuals = series[seasonal_period:] - series[:-seasonal_period] if len(series) > seasonal_period else np.diff(series)
    return _build_output("seasonal_naive", dates, repeated, residuals)


def forecast_moving_average(train: pd.DataFrame, horizon: int, window: int = 13) -> pd.DataFrame:
    series = train["y"].values
    rolling_mean = pd.Series(series).rolling(window=min(window, len(series))).mean().iloc[-1]
    point = np.full(horizon, rolling_mean)
    dates = pd.date_range(train["date"].max() + pd.Timedelta(weeks=1), periods=horizon, freq="W-MON")
    residuals = series - pd.Series(series).rolling(window=min(window, len(series))).mean().bfill().values
    return _build_output("moving_average", dates, point, residuals)


def forecast_drift(train: pd.DataFrame, horizon: int) -> pd.DataFrame:
    series = train["y"].values
    if len(series) < 2:
        point = np.full(horizon, series[-1] if len(series) else 0.0)
    else:
        slope = (series[-1] - series[0]) / (len(series) - 1)
        point = np.array([series[-1] + slope * (i + 1) for i in range(horizon)])
    dates = pd.date_range(train["date"].max() + pd.Timedelta(weeks=1), periods=horizon, freq="W-MON")
    residuals = np.diff(series) if len(series) > 1 else np.array([0.0])
    return _build_output("drift", dates, point, residuals)


def forecast_ets(train: pd.DataFrame, horizon: int) -> pd.DataFrame:
    series = train["y"].astype(float).values
    seasonal = "add" if len(series) >= 104 else None
    seasonal_periods = 52 if seasonal else None
    model = ExponentialSmoothing(
        series,
        trend="add",
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        damped_trend=True,
        initialization_method="estimated",
    )
    fit = model.fit(optimized=True)
    forecast = fit.forecast(horizon)
    fitted = fit.fittedvalues
    residuals = series[-len(fitted) :] - fitted
    dates = pd.date_range(train["date"].max() + pd.Timedelta(weeks=1), periods=horizon, freq="W-MON")
    return _build_output("ets", dates, forecast, residuals)


def forecast_sarima(train: pd.DataFrame, horizon: int, exog_future: pd.DataFrame | None = None) -> pd.DataFrame:
    series = train["y"].astype(float).values
    exog_cols = [c for c in train.columns if c not in ("date", "y")]
    exog = train[exog_cols] if exog_cols else None
    seasonal_order = (1, 0, 1, 52) if len(series) >= 120 else (0, 0, 0, 0)
    model = SARIMAX(
        endog=series,
        exog=exog,
        order=(1, 1, 1),
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    if exog_cols:
        if exog_future is None or exog_future.empty:
            exog_future = pd.DataFrame(np.tile(exog.tail(1).values, (horizon, 1)), columns=exog_cols)
        pred = fit.get_forecast(steps=horizon, exog=exog_future[exog_cols])
    else:
        pred = fit.get_forecast(steps=horizon)
    point = pred.predicted_mean.values
    resid = fit.resid
    dates = pd.date_range(train["date"].max() + pd.Timedelta(weeks=1), periods=horizon, freq="W-MON")
    return _build_output("sarimax", dates, point, resid)


def forecast_ml_lag(train: pd.DataFrame, horizon: int) -> pd.DataFrame:
    feats = make_lagged_features(train)
    usable = feats.dropna().copy()
    if usable.empty or len(usable) < 30:
        return forecast_moving_average(train, horizon)

    feature_cols = [c for c in usable.columns if c not in ("date", "y")]
    X = usable[feature_cols]
    y = usable["y"]
    model = RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=3)
    model.fit(X, y)

    history = train.copy()
    preds = []
    for _ in range(horizon):
        feat_hist = make_lagged_features(history).dropna()
        row = feat_hist.iloc[-1:][feature_cols]
        pred = float(model.predict(row)[0])
        next_date = history["date"].max() + pd.Timedelta(weeks=1)
        next_row = {"date": next_date, "y": pred}
        for c in train.columns:
            if c not in ("date", "y"):
                next_row[c] = history[c].iloc[-1]
        history = pd.concat([history, pd.DataFrame([next_row])], ignore_index=True)
        preds.append(pred)

    fitted_preds = model.predict(X)
    residuals = y.values - fitted_preds
    dates = pd.date_range(train["date"].max() + pd.Timedelta(weeks=1), periods=horizon, freq="W-MON")
    return _build_output("ml_lag_rf", dates, np.array(preds), residuals)


def forecast_ensemble(forecasts: list[pd.DataFrame]) -> pd.DataFrame:
    base = forecasts[0][["date"]].copy()
    for col in ["forecast", "lower_80", "upper_80", "lower_95", "upper_95"]:
        base[col] = np.mean([f[col].values for f in forecasts], axis=0)
    base["model_id"] = "ensemble_top"
    return base


def available_models() -> dict[str, Callable[..., pd.DataFrame]]:
    return {
        "seasonal_naive": forecast_seasonal_naive,
        "moving_average": forecast_moving_average,
        "drift": forecast_drift,
        "ets": forecast_ets,
        "sarimax": forecast_sarima,
        "ml_lag_rf": forecast_ml_lag,
    }

