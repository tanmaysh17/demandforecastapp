from __future__ import annotations

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


def prepare_weekly_exog(df: pd.DataFrame, date_col: str, exog_cols: list[str]) -> pd.DataFrame:
    if not exog_cols:
        return pd.DataFrame(columns=["date"])
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    keep = [date_col] + [c for c in exog_cols if c in work.columns]
    work = work[keep].dropna(subset=[date_col])
    for col in keep[1:]:
        if not pd.api.types.is_numeric_dtype(work[col]):
            work[col] = pd.to_numeric(work[col], errors="coerce")
    weekly = work.set_index(date_col).resample("W-MON").mean(numeric_only=True).reset_index()
    return weekly.rename(columns={date_col: "date"})


def make_lagged_features(
    df: pd.DataFrame,
    lags: tuple[int, ...] = (1, 2, 3, 4, 13, 26, 52),
    rolling_windows: tuple[int, ...] = (4, 13, 26),
) -> pd.DataFrame:
    work = df.copy()
    for lag in lags:
        work[f"lag_{lag}"] = work["y"].shift(lag)
    for window in rolling_windows:
        work[f"roll_mean_{window}"] = work["y"].shift(1).rolling(window).mean()
        work[f"roll_std_{window}"] = work["y"].shift(1).rolling(window).std()
    work["week_of_year"] = work["date"].dt.isocalendar().week.astype(int)
    work["month"] = work["date"].dt.month
    work["quarter"] = work["date"].dt.quarter
    return work


def build_holiday_features(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    mode: str = "None",
) -> pd.DataFrame:
    dates = pd.date_range(start_date, end_date, freq="W-MON")
    out = pd.DataFrame({"date": dates, "holiday_week_count": 0.0})
    if mode != "US":
        return out
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=start_date - pd.Timedelta(days=7), end=end_date + pd.Timedelta(days=7))
    hdf = pd.DataFrame({"holiday_date": holidays})
    hdf["date"] = hdf["holiday_date"] - pd.to_timedelta(hdf["holiday_date"].dt.weekday, unit="D")
    hdf["date"] = hdf["date"] + pd.Timedelta(days=0)
    weekly = hdf.groupby("date").size().rename("holiday_week_count").reset_index()
    out = out.merge(weekly, on="date", how="left", suffixes=("", "_x"))
    out["holiday_week_count"] = out["holiday_week_count_x"].fillna(out["holiday_week_count"])
    return out[["date", "holiday_week_count"]]
