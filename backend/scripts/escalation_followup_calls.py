"""Place a Day-6 outbound callback for every human-escalation request that a
human has marked resolved (via scripts/escalation_dashboard.py or directly
in the escalations table) but hasn't been called back about yet.

    uv run python scripts/escalation_followup_calls.py

Meant to be run periodically (e.g. every few minutes via cron), same as
scripts/retry_outbound_calls.py — it does one pass and exits, it isn't a
scheduler itself. Queries db.due_escalation_callbacks() for resolved
requests where the caller's id looks like a phone number (i.e. this was a
SIP call, not a browser session with no reachable number) and dispatches an
call_type="escalation_followup" outbound call for each one exactly like
scripts/make_outbound_call.py does, then marks it called-back so this script
never dials the same request twice.

Requires the backend worker already running (`uv run python src/agent.py
dev`) — this only queues the jobs — and LIVEKIT_SIP_OUTBOUND_TRUNK_ID set,
same as any other outbound call.
"""

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_ROOT / ".env.local")

sys.path.insert(0, str(BACKEND_ROOT / "src"))

import livekit.api as api  # noqa: E402

import db  # noqa: E402
import escalation  # noqa: E402


def _detail_for(row: dict) -> str:
    reason_label = escalation.ESCALATION_REASONS.get(row["reason"], row["reason"])
    detail = f"reference {row['id']}, about: {reason_label}."
    if row["resolution_note"]:
        detail += f" Resolution: {row['resolution_note']}"
    return detail


async def main() -> None:
    db.init_db()
    due = db.due_escalation_callbacks()
    if not due:
        print("No resolved escalations awaiting a callback.")
        return

    lkapi = api.LiveKitAPI(
        url=os.environ["LIVEKIT_URL"],
        api_key=os.environ["LIVEKIT_API_KEY"],
        api_secret=os.environ["LIVEKIT_API_SECRET"],
    )
    try:
        for row in due:
            metadata = {
                "phone_number": row["user_id"],
                "call_type": "escalation_followup",
                "detail": _detail_for(row),
            }
            room_name = f"outbound-escalation-{uuid.uuid4().hex[:12]}"
            await lkapi.agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(
                    agent_name="my-agent",
                    room=room_name,
                    metadata=json.dumps(metadata),
                )
            )
            db.mark_escalation_called_back(row["id"])
            print(
                f"Callback placed for escalation {row['id']} -> {row['user_id']} "
                f"in room {room_name!r}."
            )
    finally:
        await lkapi.aclose()


if __name__ == "__main__":
    asyncio.run(main())
