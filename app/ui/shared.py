"""Shared UI utilities: session state initialisation and sidebar."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.core.config import RANDOM_SEED
from app.services.persistence import (
    delete_user_session,
    deserialize_app_state,
    get_user_session_payload,
    init_db,
    list_user_sessions,
    save_user_session,
    serialize_app_state,
)
from app.services.eda import cagr

np.random.seed(RANDOM_SEED)

ANON_USER_ID = 1

_STATE_DEFAULTS: dict = {
    "raw_df": None,
    "validated": None,
    "forecasts": {},
    "rank_df": pd.DataFrame(),
    "explanation": "",
    "selected_model": None,
    "fold_map": {},
    "backtest_errors": {},
    "adjusted_forecasts": {},
    "audit_trail": [],
    "holiday_mode": "None",
    "dark_mode": False,
}

# ── CSS base (always injected) ────────────────────────────────────────────────
_CSS_BASE = """
<style>
/* ── Global font (iOS system stack) ─────────────────────────────────── */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
                 "Helvetica Neue", Arial, sans-serif !important;
}

/* ── Hide Streamlit's auto-generated sidebar nav ────────────────────── */
[data-testid="stSidebarNav"] { display: none !important; }

/* ── Sidebar: no scroll ──────────────────────────────────────────────── */
section[data-testid="stSidebar"] > div:first-child {
    overflow-y: hidden !important;
}

/* ── Download buttons ────────────────────────────────────────────────── */
div.stDownloadButton > button {
    background: #34C759 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    transition: background 0.15s ease !important;
    box-shadow: none !important;
}
div.stDownloadButton > button:hover {
    background: #2DB84E !important;
}

/* ── Primary buttons ─────────────────────────────────────────────────── */
div.stButton > button[kind="primary"] {
    background: #007AFF !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.6rem !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    transition: background 0.15s ease !important;
    box-shadow: none !important;
}
div.stButton > button[kind="primary"]:hover { background: #0071E3 !important; }
div.stButton > button[kind="primary"]:active { background: #0062CC !important; }

/* ── Expanders ───────────────────────────────────────────────────────── */
div[data-testid="stExpander"] {
    border-radius: 10px !important;
}
</style>
"""

# ── Light mode CSS ────────────────────────────────────────────────────────────
_CSS_LIGHT = """
<style>
section[data-testid="stSidebar"] {
    background: #F2F2F7 !important;
}
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] * {
    color: #1C1C1E !important;
}
section[data-testid="stSidebar"] hr { border-color: #D1D1D6 !important; }
section[data-testid="stSidebar"] button {
    background: #FFFFFF !important;
    border: 1px solid #C7C7CC !important;
    border-radius: 10px !important;
    color: #1C1C1E !important;
}
section[data-testid="stSidebar"] button:hover { background: #E5E5EA !important; }
section[data-testid="stSidebar"] [data-testid="stPageLink"] p,
section[data-testid="stSidebar"] [data-testid="stPageLink"] span,
section[data-testid="stSidebar"] [data-testid="stPageLink"] a {
    color: #007AFF !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}
section[data-testid="stSidebar"] [data-testid="stPageLink"]:hover {
    background: #E5E5EA !important;
    border-radius: 8px !important;
}
div.stButton > button[kind="secondary"],
div.stButton > button:not([kind]) {
    background: #FFFFFF !important;
    border: 1px solid #C7C7CC !important;
    border-radius: 10px !important;
    padding: 0.45rem 1.2rem !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: #1C1C1E !important;
    transition: background 0.15s ease !important;
    box-shadow: none !important;
}
div.stButton > button[kind="secondary"]:hover,
div.stButton > button:not([kind]):hover {
    background: #F2F2F7 !important;
    border-color: #AEAEB2 !important;
}
div[data-testid="metric-container"] {
    background: #FFFFFF !important;
    border: 1px solid #E5E5EA !important;
    border-radius: 12px !important;
    padding: 0.75rem 1rem 0.5rem !important;
    box-shadow: none !important;
}
div[data-testid="stExpander"] {
    border: 1px solid #E5E5EA !important;
    background: #FFFFFF !important;
}
</style>
"""

# ── Dark mode CSS ─────────────────────────────────────────────────────────────
_CSS_DARK = """
<style>
/* Main app background */
.stApp, .stApp > div {
    background-color: #1C1C1E !important;
}
/* All main content text */
.stApp p, .stApp span, .stApp label, .stApp li,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stMarkdown, .stMarkdown * {
    color: #F2F2F7 !important;
}
/* Sidebar */
section[data-testid="stSidebar"] {
    background: #2C2C2E !important;
}
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] * {
    color: #F2F2F7 !important;
}
section[data-testid="stSidebar"] hr { border-color: #48484A !important; }
section[data-testid="stSidebar"] button {
    background: #3A3A3C !important;
    border: 1px solid #48484A !important;
    border-radius: 10px !important;
    color: #F2F2F7 !important;
}
section[data-testid="stSidebar"] button:hover { background: #48484A !important; }
section[data-testid="stSidebar"] [data-testid="stPageLink"] p,
section[data-testid="stSidebar"] [data-testid="stPageLink"] span,
section[data-testid="stSidebar"] [data-testid="stPageLink"] a {
    color: #0A84FF !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}
section[data-testid="stSidebar"] [data-testid="stPageLink"]:hover {
    background: #3A3A3C !important;
    border-radius: 8px !important;
}
/* Secondary / default buttons */
div.stButton > button[kind="secondary"],
div.stButton > button:not([kind]) {
    background: #3A3A3C !important;
    border: 1px solid #48484A !important;
    border-radius: 10px !important;
    padding: 0.45rem 1.2rem !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: #F2F2F7 !important;
    transition: background 0.15s ease !important;
    box-shadow: none !important;
}
div.stButton > button[kind="secondary"]:hover,
div.stButton > button:not([kind]):hover {
    background: #48484A !important;
}
/* Metric tiles */
div[data-testid="metric-container"] {
    background: #2C2C2E !important;
    border: 1px solid #3A3A3C !important;
    border-radius: 12px !important;
    padding: 0.75rem 1rem 0.5rem !important;
    box-shadow: none !important;
}
div[data-testid="metric-container"] * { color: #F2F2F7 !important; }
/* Expanders */
div[data-testid="stExpander"] {
    border: 1px solid #3A3A3C !important;
    background: #2C2C2E !important;
}
div[data-testid="stExpander"] * { color: #F2F2F7 !important; }
div[data-testid="stExpander"] summary { color: #F2F2F7 !important; }
/* Input fields */
input, textarea,
.stTextInput input,
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea {
    background: #2C2C2E !important;
    color: #F2F2F7 !important;
    border-color: #48484A !important;
}
/* Select boxes */
div[data-baseweb="select"] > div {
    background: #2C2C2E !important;
    border-color: #48484A !important;
    color: #F2F2F7 !important;
}
div[data-baseweb="popover"] { background: #2C2C2E !important; }
div[data-baseweb="menu"] { background: #2C2C2E !important; }
div[data-baseweb="menu"] li { color: #F2F2F7 !important; }
div[data-baseweb="menu"] li:hover { background: #3A3A3C !important; }
/* Number inputs */
div[data-baseweb="input"] { background: #2C2C2E !important; }
/* Sliders */
div[data-testid="stSlider"] p { color: #F2F2F7 !important; }
/* Tables */
table { background: #2C2C2E !important; }
th { background: #3A3A3C !important; color: #F2F2F7 !important; }
td { background: #2C2C2E !important; color: #F2F2F7 !important; }
/* DataFrame */
.stDataFrame { background: #2C2C2E !important; }
/* Dividers */
hr { border-color: #3A3A3C !important; }
/* Tabs */
div[data-testid="stTabs"] button { color: #F2F2F7 !important; }
div[data-testid="stTabs"] button[aria-selected="true"] { color: #0A84FF !important; }
/* Caption text */
.stCaption, .stCaption * { color: #8E8E93 !important; }
/* Alerts / info boxes */
div[data-testid="stAlert"] { background: #2C2C2E !important; }
div[data-testid="stAlert"] * { color: #F2F2F7 !important; }
</style>
"""


def _get_css(dark: bool) -> str:
    return _CSS_BASE + (_CSS_DARK if dark else _CSS_LIGHT)


def _init_state() -> None:
    for key, default in _STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _render_sidebar() -> None:
    dark = st.session_state.get("dark_mode", False)
    text_color = "#F2F2F7" if dark else "#1C1C1E"
    sub_color = "#8E8E93"

    # ── Logo + dark mode toggle ────────────────────────────────────────────
    logo_col, toggle_col = st.sidebar.columns([3, 1])
    with logo_col:
        st.markdown(
            f"""
<div style="display:flex;align-items:center;gap:10px;padding:0.6rem 0 1.0rem 0;">
  <svg width="34" height="34" viewBox="0 0 34 34" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect width="34" height="34" rx="9" fill="#007AFF"/>
    <polyline points="5,26 11,17 17,21 23,11 29,9"
              stroke="white" stroke-width="2.5" stroke-linecap="round"
              stroke-linejoin="round" fill="none"/>
  </svg>
  <div>
    <div style="font-weight:700;font-size:0.92rem;color:{text_color};line-height:1.15;
                font-family:-apple-system,BlinkMacSystemFont,sans-serif;">Demand</div>
    <div style="font-weight:400;font-size:0.74rem;color:{sub_color};line-height:1.15;
                font-family:-apple-system,BlinkMacSystemFont,sans-serif;">Forecasting Studio</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
    with toggle_col:
        st.markdown("<div style='padding-top:0.55rem;'></div>", unsafe_allow_html=True)
        label = "☀️" if dark else "🌙"
        if st.button(label, key="_dm_toggle", help="Toggle dark / light mode"):
            st.session_state.dark_mode = not dark
            st.rerun()

    # ── Combined page nav with status icons ───────────────────────────────
    validated = st.session_state.validated
    forecasts = st.session_state.forecasts
    rank_df = st.session_state.rank_df
    selected_model = st.session_state.selected_model

    def _icon(done: bool) -> str:
        return "✅" if done else "⭕"

    nav_pages = [
        ("streamlit_app.py",        "🏠  Home"),
        ("pages/1_Upload.py",       f"{_icon(validated is not None)}  Upload & Mapping"),
        ("pages/2_EDA.py",          f"{_icon(validated is not None)}  EDA & Data Health"),
        ("pages/3_Features.py",     f"{_icon(validated is not None)}  Features & Seasonality"),
        ("pages/4_Forecast.py",     f"{_icon(bool(forecasts))}  Forecast"),
        ("pages/5_Backtest.py",     f"{_icon(not rank_df.empty)}  Backtest"),
        ("pages/6_Explain.py",      f"{_icon(selected_model is not None)}  Explain"),
        ("pages/7_Export.py",       f"{_icon(not rank_df.empty)}  Export"),
        ("pages/8_Launch_Curve.py", "🚀  Launch Curve"),
    ]
    for path, label in nav_pages:
        st.sidebar.page_link(path, label=label)

    # ── Sessions (compact expander) ───────────────────────────────────────
    st.sidebar.divider()
    with st.sidebar.expander("💾  Sessions"):
        save_name = st.text_input("Session name", value="default", key="_ss_name")
        if st.button("Save session", key="_ss_save"):
            payload = serialize_app_state(dict(st.session_state))
            save_user_session(ANON_USER_ID, save_name.strip() or "default", payload)
            st.success("Saved.")

        saved = list_user_sessions(ANON_USER_ID)
        if saved:
            labels = [f"{x['session_name']}  ({x['updated_at'][:10]})" for x in saved]
            idx = st.selectbox(
                "Saved runs",
                options=list(range(len(saved))),
                format_func=lambda i: labels[i],
                key="_ss_sel",
            )
            c1, c2 = st.columns(2)
            if c1.button("Load", key="_ss_load"):
                payload = get_user_session_payload(ANON_USER_ID, saved[idx]["id"])
                if payload:
                    restored = deserialize_app_state(payload)
                    st.session_state.raw_df = restored["raw_df"]
                    st.session_state.validated = restored["validated"]
                    st.session_state.forecasts = restored["forecasts"] or {}
                    st.session_state.rank_df = (
                        restored["rank_df"] if restored["rank_df"] is not None else pd.DataFrame()
                    )
                    st.session_state.explanation = restored["explanation"]
                    st.session_state.selected_model = restored["selected_model"]
                    st.session_state.adjusted_forecasts = restored["adjusted_forecasts"] or {}
                    st.session_state.audit_trail = restored["audit_trail"] or []
                    st.session_state.holiday_mode = restored["holiday_mode"] or "None"
                    st.session_state.fold_map = {}
                    st.session_state.backtest_errors = {}
                    st.success("Session loaded.")
                    st.rerun()
            if c2.button("Delete", key="_ss_del"):
                delete_user_session(ANON_USER_ID, saved[idx]["id"])
                st.rerun()

    # ── Reset + CAGR ──────────────────────────────────────────────────────
    st.sidebar.divider()
    if st.sidebar.button("↺  Reset Session", key="_reset"):
        dark_pref = st.session_state.get("dark_mode", False)
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.session_state.dark_mode = dark_pref  # preserve theme preference
        st.rerun()

    if st.session_state.validated:
        clean = st.session_state.validated["clean_df"]
        if len(clean) >= 52:
            years = len(clean) / 52
            v = cagr(clean["y"], years)
            st.sidebar.metric("Historical CAGR", f"{v:.2%}" if v is not None else "N/A")


def setup_page() -> None:
    """Call at the top of every page: init state and render sidebar."""
    init_db()
    _init_state()
    dark = st.session_state.get("dark_mode", False)
    st.markdown(_get_css(dark), unsafe_allow_html=True)
    _render_sidebar()
