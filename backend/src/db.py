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
        # One row per outbound dial attempt (see agent.py's OUTBOUND CALLS
        # handling). `outcome` is one of: answered, no_answer, busy,
        # voicemail, immediate_hangup, failed. `next_retry_at` is set only
        # for outcomes the retry policy says to retry, and only while
        # `retried` is still 0 — scripts/retry_outbound_calls.py flips it to
        # 1 once it has dispatched the follow-up attempt, so a due row is
        # never picked up twice.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS outbound_call_attempts (
                id TEXT PRIMARY KEY,
                phone_number TEXT NOT NULL,
                call_type TEXT,
                detail TEXT,
                attempt_number INTEGER NOT NULL,
                outcome TEXT NOT NULL,
                attempted_at REAL NOT NULL,
                next_retry_at REAL,
                retried INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_outbound_attempts_retry "
            "ON outbound_call_attempts(next_retry_at, retried)"
        )
        # Human-escalation requests (see src/escalation.py and the HUMAN
        # ESCALATION section of agent.py's SYSTEM_PROMPT). `reason` is one of
        # the two escalation categories (escalation.ESCALATION_REASONS);
        # `urgency` is low/medium/high/emergency; `status` is
        # open/in_progress/resolved. `who`/`what_happened`/`already_checked`
        # are short, redacted summaries only — never a full transcript or
        # sensitive identifiers (see escalation.redact). `called_back` tracks
        # whether scripts/escalation_followup_calls.py has already placed the
        # Day-6 outbound callback for a resolved request, so it's never
        # placed twice.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS escalations (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                reason TEXT NOT NULL,
                urgency TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                who TEXT,
                what_happened TEXT NOT NULL,
                already_checked TEXT,
                language TEXT,
                preferred_contact TEXT,
                notify_status TEXT,
                resolution_note TEXT,
                called_back INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                resolved_at REAL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_escalations_user_reason "
            "ON escalations(user_id, reason, status)"
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


def record_outbound_attempt(
    *,
    phone_number: str,
    call_type: str | None,
    detail: str | None,
    attempt_number: int,
    outcome: str,
    next_retry_at: float | None,
) -> str:
    """Log one outbound dial attempt and its outcome. Returns the row id."""
    attempt_id = str(uuid.uuid4())
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO outbound_call_attempts
                (id, phone_number, call_type, detail, attempt_number, outcome,
                 attempted_at, next_retry_at, retried)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                attempt_id,
                phone_number,
                call_type,
                detail,
                attempt_number,
                outcome,
                time.time(),
                next_retry_at,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return attempt_id


def due_outbound_retries(now: float) -> list[dict]:
    """Attempts whose retry delay has elapsed and haven't been retried yet.
    Consumed by scripts/retry_outbound_calls.py.
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT * FROM outbound_call_attempts
            WHERE next_retry_at IS NOT NULL AND next_retry_at <= ? AND retried = 0
            ORDER BY next_retry_at ASC
            """,
            (now,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def mark_outbound_retried(attempt_id: str) -> None:
    """Marks an attempt's retry as dispatched, so due_outbound_retries()
    never returns it again."""
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE outbound_call_attempts SET retried = 1 WHERE id = ?",
            (attempt_id,),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Human escalation requests (see src/escalation.py)
# ---------------------------------------------------------------------------

# Short, speakable-over-the-phone reference id — e.g. "A1B2C3D4" — rather
# than a full uuid4, since the agent has to read this out loud and the
# caller may need to repeat it back to a human later.
_ESCALATION_ID_LEN = 8

_URGENCY_RANK = {"low": 0, "medium": 1, "high": 2, "emergency": 3}


def _new_escalation_id() -> str:
    return uuid.uuid4().hex[:_ESCALATION_ID_LEN].upper()


def find_open_escalation(user_id: str, reason: str) -> dict | None:
    """An existing not-yet-resolved request for this caller and reason, if
    any — used to dedupe instead of opening a second ticket for the same
    problem. Returns None if `user_id` is falsy (nothing to dedupe against
    for an unidentified caller).
    """
    if not user_id:
        return None
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT * FROM escalations
            WHERE user_id = ? AND reason = ? AND status != 'resolved'
            ORDER BY created_at DESC LIMIT 1
            """,
            (user_id, reason),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def create_escalation(
    *,
    user_id: str | None,
    reason: str,
    urgency: str,
    who: str | None,
    what_happened: str,
    already_checked: str | None,
    language: str | None,
    preferred_contact: str | None,
    notify_status: str,
) -> dict:
    """Open a new escalation request. Returns the created row (including its
    short `id`, which is what the caller is told as their reference)."""
    escalation_id = _new_escalation_id()
    now = time.time()
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO escalations
                (id, user_id, reason, urgency, status, who, what_happened,
                 already_checked, language, preferred_contact, notify_status,
                 called_back, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'open', ?, ?, ?, ?, ?, ?, 0, ?, ?)
            """,
            (
                escalation_id,
                user_id,
                reason,
                urgency,
                who,
                what_happened,
                already_checked,
                language,
                preferred_contact,
                notify_status,
                now,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return get_escalation(escalation_id)


def bump_existing_escalation(
    escalation_id: str, *, urgency: str, additional_note: str
) -> dict:
    """Update an already-open escalation instead of creating a duplicate —
    raises its urgency if the new report is more urgent than what's on file,
    and appends a short note so a human reviewing it can see both reports.
    Never lowers urgency or overwrites the original `what_happened`.
    """
    existing = get_escalation(escalation_id)
    new_urgency = (
        urgency
        if _URGENCY_RANK.get(urgency, 0) > _URGENCY_RANK.get(existing["urgency"], 0)
        else existing["urgency"]
    )
    merged_note = f"{existing['what_happened']}\n[update] {additional_note}"
    conn = get_connection()
    try:
        conn.execute(
            """
            UPDATE escalations
            SET urgency = ?, what_happened = ?, updated_at = ?
            WHERE id = ?
            """,
            (new_urgency, merged_note, time.time(), escalation_id),
        )
        conn.commit()
    finally:
        conn.close()
    return get_escalation(escalation_id)


def get_escalation(escalation_id: str) -> dict | None:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM escalations WHERE id = ?", (escalation_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_escalations(status: str | None = None) -> list[dict]:
    """All escalations, newest first. Pass `status` to filter to one of
    open/in_progress/resolved; omit for all of them. Backs
    scripts/escalation_dashboard.py.
    """
    conn = get_connection()
    try:
        if status:
            rows = conn.execute(
                "SELECT * FROM escalations WHERE status = ? ORDER BY created_at DESC",
                (status,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM escalations ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def set_escalation_status(
    escalation_id: str, status: str, resolution_note: str | None = None
) -> None:
    """Move a request through open -> in_progress -> resolved. Sets
    `resolved_at` the moment it becomes resolved (used by
    scripts/escalation_followup_calls.py to find requests ready for a
    callback)."""
    conn = get_connection()
    try:
        conn.execute(
            """
            UPDATE escalations
            SET status = ?, resolution_note = COALESCE(?, resolution_note),
                resolved_at = CASE WHEN ? = 'resolved' THEN ? ELSE resolved_at END,
                updated_at = ?
            WHERE id = ?
            """,
            (status, resolution_note, status, time.time(), time.time(), escalation_id),
        )
        conn.commit()
    finally:
        conn.close()


def due_escalation_callbacks() -> list[dict]:
    """Resolved requests for a phone-number caller that haven't had their
    Day-6 outbound callback placed yet. Consumed by
    scripts/escalation_followup_calls.py.
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT * FROM escalations
            WHERE status = 'resolved' AND called_back = 0
                AND user_id IS NOT NULL AND user_id LIKE '+%'
            ORDER BY resolved_at ASC
            """
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def set_escalation_notify_status(escalation_id: str, notify_status: str) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE escalations SET notify_status = ? WHERE id = ?",
            (notify_status, escalation_id),
        )
        conn.commit()
    finally:
        conn.close()


def mark_escalation_called_back(escalation_id: str) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE escalations SET called_back = 1 WHERE id = ?", (escalation_id,)
        )
        conn.commit()
    finally:
        conn.close()
