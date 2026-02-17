from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
from datetime import datetime
from typing import Any

import pandas as pd


def init_db(db_path: str = "app_data.db") -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_name TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return f"{salt.hex()}:{digest.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt_hex, digest_hex = stored_hash.split(":")
        expected = bytes.fromhex(digest_hex)
        computed = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), 120_000)
        return hmac.compare_digest(expected, computed)
    except Exception:
        return False


def create_user(username: str, password: str, db_path: str = "app_data.db") -> tuple[bool, str]:
    username = username.strip()
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    conn = sqlite3.connect(db_path)
    try:
        password_hash = _hash_password(password)
        conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, password_hash, datetime.utcnow().isoformat()),
        )
        conn.commit()
        return True, "Account created."
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    finally:
        conn.close()


def authenticate_user(username: str, password: str, db_path: str = "app_data.db") -> tuple[bool, int | None]:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT id, password_hash FROM users WHERE username = ?", (username.strip(),)).fetchone()
        if not row:
            return False, None
        user_id, stored_hash = row
        if _verify_password(password, stored_hash):
            return True, int(user_id)
        return False, None
    finally:
        conn.close()


def _serialize_df(df: pd.DataFrame | None) -> dict[str, Any] | None:
    if df is None:
        return None
    return {"format": "split", "data": df.to_json(orient="split", date_format="iso")}


def _deserialize_df(obj: dict[str, Any] | None) -> pd.DataFrame | None:
    if not obj:
        return None
    if obj.get("format") != "split":
        return None
    return pd.read_json(obj["data"], orient="split")


def serialize_app_state(state: dict[str, Any]) -> str:
    validated = state.get("validated")
    validated_payload = None
    if validated:
        report = validated.get("report")
        issues: list[dict[str, Any]] = []
        summary: dict[str, Any] = {}
        if report:
            if isinstance(report, dict):
                issues = report.get("issues", [])
                summary = report.get("summary", {})
            else:
                issues = [
                    {"level": i.level, "check": i.check, "message": i.message, "details": i.details}
                    for i in report.issues
                ]
                summary = report.summary
        validated_payload = {
            "clean_df": _serialize_df(validated.get("clean_df")),
            "summary": summary,
            "issues": issues,
            "date_col": validated.get("date_col"),
            "target_col": validated.get("target_col"),
        }

    payload = {
        "raw_df": _serialize_df(state.get("raw_df")),
        "validated": validated_payload,
        "forecasts": {k: _serialize_df(v) for k, v in state.get("forecasts", {}).items()},
        "rank_df": _serialize_df(state.get("rank_df")),
        "explanation": state.get("explanation", ""),
        "selected_model": state.get("selected_model"),
        "adjusted_forecasts": {k: _serialize_df(v) for k, v in state.get("adjusted_forecasts", {}).items()},
        "audit_trail": state.get("audit_trail", []),
        "holiday_mode": state.get("holiday_mode", "None"),
        "saved_at": datetime.utcnow().isoformat(),
    }
    return json.dumps(payload)


def deserialize_app_state(payload_json: str) -> dict[str, Any]:
    obj = json.loads(payload_json)
    validated_obj = obj.get("validated")
    validated = None
    if validated_obj:
        validated = {
            "report": {
                "summary": validated_obj.get("summary", {}),
                "issues": validated_obj.get("issues", []),
            },
            "clean_df": _deserialize_df(validated_obj.get("clean_df")),
            "date_col": validated_obj.get("date_col"),
            "target_col": validated_obj.get("target_col"),
        }

    return {
        "raw_df": _deserialize_df(obj.get("raw_df")),
        "validated": validated,
        "forecasts": {k: _deserialize_df(v) for k, v in obj.get("forecasts", {}).items()},
        "rank_df": _deserialize_df(obj.get("rank_df")),
        "explanation": obj.get("explanation", ""),
        "selected_model": obj.get("selected_model"),
        "adjusted_forecasts": {k: _deserialize_df(v) for k, v in obj.get("adjusted_forecasts", {}).items()},
        "audit_trail": obj.get("audit_trail", []),
        "holiday_mode": obj.get("holiday_mode", "None"),
        "saved_at": obj.get("saved_at"),
    }


def save_user_session(user_id: int, session_name: str, payload_json: str, db_path: str = "app_data.db") -> None:
    now = datetime.utcnow().isoformat()
    conn = sqlite3.connect(db_path)
    try:
        existing = conn.execute(
            "SELECT id FROM user_sessions WHERE user_id = ? AND session_name = ?",
            (user_id, session_name),
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE user_sessions SET payload_json = ?, updated_at = ? WHERE id = ?",
                (payload_json, now, int(existing[0])),
            )
        else:
            conn.execute(
                "INSERT INTO user_sessions (user_id, session_name, payload_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, session_name, payload_json, now, now),
            )
        conn.commit()
    finally:
        conn.close()


def list_user_sessions(user_id: int, db_path: str = "app_data.db") -> list[dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, session_name, updated_at FROM user_sessions WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
        return [{"id": int(r[0]), "session_name": r[1], "updated_at": r[2]} for r in rows]
    finally:
        conn.close()


def get_user_session_payload(user_id: int, session_id: int, db_path: str = "app_data.db") -> str | None:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT payload_json FROM user_sessions WHERE user_id = ? AND id = ?",
            (user_id, session_id),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def delete_user_session(user_id: int, session_id: int, db_path: str = "app_data.db") -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DELETE FROM user_sessions WHERE user_id = ? AND id = ?", (user_id, session_id))
        conn.commit()
    finally:
        conn.close()
