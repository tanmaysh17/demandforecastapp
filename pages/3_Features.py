"""Page 3 - Features & Seasonality."""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Features & Seasonality | Demand Forecasting Studio", layout="wide")

from app.ui.shared import setup_page  # noqa: E402

setup_page()

import plotly.express as px  # noqa: E402

from app.services.eda import decompose_weekly, detect_seasonality_periods, seasonal_indices  # noqa: E402

st.header("3) Features & Seasonality")

validated = st.session_state.validated
if not validated:
    st.info("Upload and validate data first (page 1).")
    st.stop()

clean = validated["clean_df"]

periods = detect_seasonality_periods(clean["y"])
st.write("Detected seasonal periods", periods)

decomp = decompose_weekly(clean.set_index("date")["y"], period=52)
if decomp is not None:
    decomp_reset = decomp.reset_index(names="date")
    st.plotly_chart(px.line(decomp_reset, x="date", y="trend", title="Trend Component"), use_container_width=True)
    st.plotly_chart(
        px.line(decomp_reset, x="date", y="seasonal", title="Seasonal Component"), use_container_width=True
    )
    st.plotly_chart(
        px.line(decomp_reset, x="date", y="resid", title="Residual Component"), use_container_width=True
    )
else:
    st.info("Insufficient history for seasonal decomposition (need ≥ 104 weeks).")

indices = seasonal_indices(clean)
st.plotly_chart(
    px.bar(indices, x="week_of_year", y="seasonality_index", title="Seasonality Index by Week"),
    use_container_width=True,
)
