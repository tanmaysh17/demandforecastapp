"""Home page: welcome screen and navigation guide."""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Demand Forecasting Studio", layout="wide")
st.title("Demand Forecasting Studio")
st.caption("Enterprise-ready weekly demand forecasting with validation, backtests, explainability, and export.")

from app.ui.shared import setup_page  # noqa: E402

setup_page()

st.markdown(
    """
## Welcome

Use the sidebar to navigate between pages:

| Page | Description |
|------|-------------|
| **1 - Upload & Mapping** | Upload your CSV / Excel file and map columns |
| **2 - EDA & Data Health** | Data quality report, outlier flags, growth metrics |
| **3 - Features & Seasonality** | Seasonal decomposition, seasonality indices |
| **4 - Forecast** | Generate forecasts with multiple models |
| **5 - Backtest** | Rolling holdout accuracy evaluation and model ranking |
| **6 - Explain** | Model recommendation rationale and residual diagnostics |
| **7 - Export** | Manual adjustments, audit trail, and Excel/CSV download |

**Typical workflow:** Upload → EDA → Forecast → Backtest → Explain → Export
"""
)
