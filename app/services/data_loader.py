from __future__ import annotations

import io
from dataclasses import dataclass

import pandas as pd


@dataclass
class ColumnMapping:
    date_col: str
    target_col: str
    optional_cols: list[str]


def read_tabular(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    name = file_name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(io.BytesIO(file_bytes))
    else:
        raise ValueError("Unsupported file type. Please upload CSV or Excel.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def infer_columns(df: pd.DataFrame) -> ColumnMapping | None:
    if df.empty:
        return None

    lowered = {c.lower(): c for c in df.columns}
    date_candidates = [
        lowered.get("date"),
        lowered.get("week"),
        lowered.get("week_start"),
        lowered.get("week_start_date"),
    ]
    date_col = next((c for c in date_candidates if c), None)

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    demand_candidates = [
        lowered.get("demand"),
        lowered.get("sales"),
        lowered.get("qty"),
        lowered.get("quantity"),
    ]
    target_col = next((c for c in demand_candidates if c), None)
    if not target_col and numeric_cols:
        target_col = numeric_cols[0]

    if date_col and target_col:
        optional = [c for c in df.columns if c not in (date_col, target_col)]
        return ColumnMapping(date_col=date_col, target_col=target_col, optional_cols=optional)
    return None


def build_template() -> pd.DataFrame:
    dates = pd.date_range("2022-01-03", periods=104, freq="W-MON")
    return pd.DataFrame(
        {
            "week_start_date": dates,
            "demand": 100 + (pd.Series(range(len(dates))) * 0.25) + (pd.Series(range(len(dates))) % 52) * 0.8,
            "product": "SKU-A",
            "region": "NA",
            "promotion_flag": (pd.Series(range(len(dates))) % 10 == 0).astype(int),
        }
    )
