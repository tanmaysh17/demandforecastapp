"""Shared UI utilities: session state initialisation, auth guard, sidebar."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.core.config import RANDOM_SEED
from app.services.persistence import (
    authenticate_user,
    create_user,
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
    "auth_user_id": None,
    "auth_username": None,
}

_CSS = """
<style>
/* ── Primary buttons ─────────────────────────────────────────────────── */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.55rem 1.6rem;
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    box-shadow: 0 2px 6px rgba(2, 132, 199, 0.35);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
div.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 14px rgba(2, 132, 199, 0.45);
}
div.stButton > button[kind="primary"]:active {
    transform: translateY(0);
    box-shadow: 0 1px 4px rgba(2, 132, 199, 0.3);
}

/* ── Secondary / default buttons ────────────────────────────────────── */
div.stButton > button[kind="secondary"],
div.stButton > button:not([kind]) {
    border: 1.5px solid #cbd5e1;
    border-radius: 8px;
    padding: 0.45rem 1.2rem;
    font-size: 0.9rem;
    font-weight: 500;
    transition: border-color 0.15s ease, background 0.15s ease;
}
div.stButton > button[kind="secondary"]:hover,
div.stButton > button:not([kind]):hover {
    border-color: #0ea5e9;
    background: #f0f9ff;
}

/* ── Download buttons ────────────────────────────────────────────────── */
div.stDownloadButton > button {
    border-radius: 8px;
    font-weight: 500;
}

/* ── Metric tiles ────────────────────────────────────────────────────── */
div[data-testid="metric-container"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.75rem 1rem 0.5rem;
}

/* ── Top account bar divider ─────────────────────────────────────────── */
.acct-bar {
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 0.4rem;
    margin-bottom: 0.2rem;
}

/* ── Sidebar progress ────────────────────────────────────────────────── */
div[data-testid="stSidebar"] .workflow-step {
    font-size: 0.82rem;
    line-height: 1.6;
}
</style>
"""


def _init_state() -> None:
    for key, default in _STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _render_auth_wall() -> None:
    """Show login / signup forms and stop page execution if not authenticated."""
    st.subheader("Account Access")
    left, right = st.columns(2)
    with left:
        with st.form("login_form"):
            st.markdown("**Sign in**")
            login_user = st.text_input("Username", key="login_user")
            login_pass = st.text_input("Password", type="password", key="login_pass")
            if st.form_submit_button("Sign in", type="primary"):
                ok, user_id = authenticate_user(login_user, login_pass)
                if ok and user_id is not None:
                    st.session_state.auth_user_id = user_id
                    st.session_state.auth_username = login_user.strip()
                    st.success("Signed in.")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
    with right:
        with st.form("signup_form"):
            st.markdown("**Create account**")
            signup_user = st.text_input("New username", key="signup_user")
            signup_pass = st.text_input("New password", type="password", key="signup_pass")
            signup_pass2 = st.text_input("Confirm password", type="password", key="signup_pass2")
            if st.form_submit_button("Create account"):
                if signup_pass != signup_pass2:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = create_user(signup_user, signup_pass)
                    if ok:
                        st.success(msg + " Please sign in.")
                    else:
                        st.error(msg)
    st.stop()


def _render_sidebar() -> None:
    # ── Workflow progress ──────────────────────────────────────────────────
    validated = st.session_state.validated
    forecasts = st.session_state.forecasts
    rank_df = st.session_state.rank_df
    selected_model = st.session_state.selected_model

    steps = [
        ("1. Upload & Mapping", validated is not None),
        ("2. EDA & Data Health", validated is not None),
        ("3. Features & Seasonality", validated is not None),
        ("4. Forecast", bool(forecasts)),
        ("5. Backtesting", not rank_df.empty),
        ("6. Explain", selected_model is not None),
        ("7. Export", not rank_df.empty),
    ]
    done_count = sum(1 for _, done in steps if done)
    pct = done_count / len(steps)

    st.sidebar.markdown("**Workflow Progress**")
    st.sidebar.progress(pct, text=f"{int(pct * 100)}% complete")
    step_lines = "\n".join(
        f"{'✅' if done else '🔴'} {name}" for name, done in steps
    )
    st.sidebar.markdown(
        f"<div class='workflow-step'>{step_lines.replace(chr(10), '<br>')}</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.divider()

    # ── Account ────────────────────────────────────────────────────────────
    st.sidebar.header("Account")
    st.sidebar.write(f"Signed in as `{st.session_state.auth_username}`")
    if st.sidebar.button("Sign out"):
        st.session_state.auth_user_id = None
        st.session_state.auth_username = None
        st.rerun()

    st.sidebar.markdown("**Saved Sessions**")
    save_name = st.sidebar.text_input("Session name", value="default")
    if st.sidebar.button("Save current session"):
        payload = serialize_app_state(dict(st.session_state))
        save_user_session(st.session_state.auth_user_id, save_name.strip() or "default", payload)
        st.sidebar.success("Session saved.")

    saved = list_user_sessions(st.session_state.auth_user_id)
    if saved:
        session_labels = [f"{x['session_name']} ({x['updated_at']})" for x in saved]
        idx = st.sidebar.selectbox(
            "Saved runs", options=list(range(len(saved))), format_func=lambda i: session_labels[i]
        )
        col_a, col_b = st.sidebar.columns(2)
        if col_a.button("Load selected"):
            payload = get_user_session_payload(st.session_state.auth_user_id, saved[idx]["id"])
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
                st.sidebar.success("Session loaded.")
                st.rerun()
        if col_b.button("Delete selected"):
            delete_user_session(st.session_state.auth_user_id, saved[idx]["id"])
            st.sidebar.success("Session deleted.")
            st.rerun()

    st.sidebar.header("Run Controls")
    st.sidebar.write("All operations are logged and deterministic via fixed random seed.")
    if st.sidebar.button("Reset Session"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    if st.session_state.validated:
        clean = st.session_state.validated["clean_df"]
        if len(clean) >= 52:
            years = len(clean) / 52
            v = cagr(clean["y"], years)
            st.sidebar.metric("Historical CAGR", f"{v:.2%}" if v is not None else "N/A")


def setup_page(title: str = "Demand Forecasting Studio") -> None:
    """Call at the top of every page: init state, enforce auth, render sidebar."""
    init_db()
    _init_state()
    # Inject CSS
    st.markdown(_CSS, unsafe_allow_html=True)
    if st.session_state.auth_user_id is None:
        _render_auth_wall()
    else:
        _render_sidebar()
        # Top-right account chip
        _, acct_col = st.columns([8, 1])
        with acct_col:
            st.markdown(
                f"<div class='acct-bar' style='text-align:right; color:#64748b; font-size:0.8rem;'>"
                f"👤 {st.session_state.auth_username}</div>",
                unsafe_allow_html=True,
            )
