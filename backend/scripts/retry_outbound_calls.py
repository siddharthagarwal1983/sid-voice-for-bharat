"""Re-dispatch outbound calls whose retry delay has elapsed.

    uv run python scripts/retry_outbound_calls.py

Meant to be run periodically (e.g. every few minutes via cron) — it does
one pass and exits, it isn't a long-running scheduler itself. Queries
db.due_outbound_retries() for attempts where config/retry_policy.json
scheduled a follow-up and the delay has passed, dispatches each one
exactly like scripts/make_outbound_call.py does but with attempt_number
incremented, then marks it retried so this script never re-dispatches the
same due row twice.

Requires the backend worker already running (`uv run python src/agent.py
dev`), same as make_outbound_call.py — this only queues the jobs.
"""

import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_ROOT / ".env.local")

sys.path.insert(0, str(BACKEND_ROOT / "src"))

import livekit.api as api  # noqa: E402

import db  # noqa: E402

RETRY_POLICY_PATH = BACKEND_ROOT / "config" / "retry_policy.json"


def _retries_enabled() -> bool:
    try:
        with RETRY_POLICY_PATH.open(encoding="utf-8") as f:
            return json.load(f).get("enabled", False)
    except (OSError, json.JSONDecodeError):
        return False


async def main() -> None:
    if not _retries_enabled():
        print(
            f'Retries are disabled ({RETRY_POLICY_PATH} has "enabled": false) — '
            "nothing to do. Set it to true to turn retries back on."
        )
        return

    db.init_db()
    due = db.due_outbound_retries(time.time())
    if not due:
        print("No outbound retries due.")
        return

    lkapi = api.LiveKitAPI(
        url=os.environ["LIVEKIT_URL"],
        api_key=os.environ["LIVEKIT_API_KEY"],
        api_secret=os.environ["LIVEKIT_API_SECRET"],
    )
    try:
        for attempt in due:
            metadata = {
                "phone_number": attempt["phone_number"],
                "attempt_number": attempt["attempt_number"] + 1,
            }
            if attempt["call_type"]:
                metadata["call_type"] = attempt["call_type"]
            if attempt["detail"]:
                metadata["detail"] = attempt["detail"]

            room_name = f"outbound-retry-{uuid.uuid4().hex[:12]}"
            await lkapi.agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(
                    agent_name="my-agent",
                    room=room_name,
                    metadata=json.dumps(metadata),
                )
            )
            db.mark_outbound_retried(attempt["id"])
            print(
                f"Retried {attempt['phone_number']} (was: {attempt['outcome']}, "
                f"attempt {metadata['attempt_number']}) in room {room_name!r}."
            )
    finally:
        await lkapi.aclose()


if __name__ == "__main__":
    asyncio.run(main())
