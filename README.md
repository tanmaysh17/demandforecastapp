# Demand Forecasting Studio

Production-quality weekly demand forecasting web app for business users.

## Features
- User accounts (signup/login) with secure password hashing
- Per-user saved sessions (save/load/delete) persisted in SQLite
- Excel upload + column mapping
- Automated data quality validation:
  - Missing weeks, duplicates, non-numeric values
  - Outliers, negative values, sparse history, structural breaks
- Weekly EDA:
  - Trend, decomposition, growth metrics, seasonality diagnostics
- Forecasting:
  - Baselines: seasonal naive, moving average, drift, ETS baseline
  - Advanced: ETS, SARIMAX, lag-based ML (Random Forest)
  - Optional ensemble of top models
- Time-series-aware evaluation:
  - Rolling holdouts (13, 26, 52 weeks)
  - sMAPE, MASE, RMSE, interval coverage (80%/95%)
- Explainability:
  - Model ranking with performance/stability/plausibility
  - Plain-language model recommendation
- Export:
  - Forecast table, model comparison, one-page summary (Excel/CSV)

## Quick Start
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Authentication and Session Storage
- App uses local SQLite database `app_data.db` for users and saved sessions.
- Passwords are stored as salted PBKDF2-SHA256 hashes.
- Each signed-in user can save and reload forecasting work-in-progress sessions.

## Input Schema
Supported file types:
- CSV (`.csv`)
- Excel (`.xlsx`, `.xls`)

Required:
- Weekly date column (week-start date preferred)
- Numeric demand/sales column

Optional:
- Product, region, channel, customer, price, promotion flags, external regressors

## Project Structure
```text
app/
  core/
    config.py
    types.py
  services/
    data_loader.py
    validation.py
    eda.py
    features.py
    models.py
    backtesting.py
    selection.py
    reporting.py
streamlit_app.py
tests/
```

## Extensibility
Architecture is designed for future support of:
- Hierarchical forecasting
- Scenario planning
- Manual overrides + audit trail
