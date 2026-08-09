"""SQLite persistence for call sessions, transcripts, and caller profiles.

Each voice call gets one row in `call_sessions`; every user/agent message in
that call gets one row in `messages`. Returning callers get one row in
`callers`, keyed by a stable caller id (SIP phone number when available,
otherwise the room participant identity), holding only short structured
facts — never full medical notes. Uses the stdlib `sqlite3` module — no
extra dependency needed for a starter template with modest write volume.
"""

import json
import sqlite3
import time
import uuid
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "agent.db"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS call_sessions (
                id TEXT PRIMARY KEY,
                room_name TEXT NOT NULL,
                participant_identity TEXT,
                started_at REAL NOT NULL,
                ended_at REAL,
                close_reason TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES call_sessions(id),
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS callers (
                user_id TEXT PRIMARY KEY,
                name TEXT,
                language_preference TEXT,
                facts TEXT NOT NULL DEFAULT '{}',
                last_interaction REAL NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def create_call_session(room_name: str) -> str:
    session_id = str(uuid.uuid4())
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO call_sessions (id, room_name, started_at) VALUES (?, ?, ?)",
            (session_id, room_name, time.time()),
        )
        conn.commit()
    finally:
        conn.close()
    return session_id


def set_participant(session_id: str, participant_identity: str) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE call_sessions SET participant_identity = ? WHERE id = ?",
            (participant_identity, session_id),
        )
        conn.commit()
    finally:
        conn.close()


def save_message(session_id: str, role: str, content: str, created_at: float) -> None:
    if not content:
        return
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO messages (id, session_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), session_id, role, content, created_at),
        )
        conn.commit()
    finally:
        conn.close()


def get_caller(user_id: str) -> dict | None:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM callers WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row is None:
            return None
        caller = dict(row)
        caller["facts"] = json.loads(caller["facts"])
        return caller
    finally:
        conn.close()


def upsert_caller(
    user_id: str,
    *,
    name: str | None = None,
    language_preference: str | None = None,
    facts: dict[str, str] | None = None,
) -> None:
    """Create or update a caller record.

    Only overwrites `name`/`language_preference` when a new value is given,
    and shallow-merges `facts` on top of whatever was already stored so a
    later call doesn't erase facts learned in an earlier one. `facts` should
    hold short structured values (e.g. an age band or a one-line triage
    outcome) — never written-out medical notes.
    """
    existing = get_caller(user_id)
    merged_facts = {**(existing["facts"] if existing else {}), **(facts or {})}
    merged_name = name or (existing["name"] if existing else None)
    merged_language = language_preference or (
        existing["language_preference"] if existing else None
    )

    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO callers (user_id, name, language_preference, facts, last_interaction)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                name = excluded.name,
                language_preference = excluded.language_preference,
                facts = excluded.facts,
                last_interaction = excluded.last_interaction
            """,
            (
                user_id,
                merged_name,
                merged_language,
                json.dumps(merged_facts),
                time.time(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def delete_caller(user_id: str) -> bool:
    """Permanently delete a caller's saved profile. Returns True if a row
    existed and was deleted, False if there was nothing to delete."""
    conn = get_connection()
    try:
        cursor = conn.execute("DELETE FROM callers WHERE user_id = ?", (user_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def end_call_session(session_id: str, close_reason: str) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE call_sessions SET ended_at = ?, close_reason = ? WHERE id = ?",
            (time.time(), close_reason, session_id),
        )
        conn.commit()
    finally:
        conn.close()
