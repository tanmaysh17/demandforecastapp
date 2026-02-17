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
    if st.session_state.auth_user_id is None:
        _render_auth_wall()
    else:
        _render_sidebar()
