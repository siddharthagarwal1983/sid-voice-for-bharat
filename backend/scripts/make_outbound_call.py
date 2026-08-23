"""Place an outbound HealthMitra call to a phone number.

    uv run python scripts/make_outbound_call.py +919876543210
    uv run python scripts/make_outbound_call.py +919876543210 \\
        --type medication_reminder --detail "your evening metformin dose"
    uv run python scripts/make_outbound_call.py +919876543210 --type triage_followup

This doesn't dial the phone directly — it dispatches the `my-agent` worker
into a fresh room with the phone number (and optional call reason) in the
job metadata. The worker (src/agent.py) reads that metadata and does the
actual dialing via the SIP outbound trunk once it picks up the job; see
`_extract_outbound_call` / `_dial_outbound_participant` there. Requires:
  - The backend worker already running (`uv run python src/agent.py dev`),
    since dispatch just queues a job for it to pick up.
  - LIVEKIT_SIP_OUTBOUND_TRUNK_ID set (see scripts/setup_outbound_trunk.py).

--type, if given, makes this a purposeful call that opens with the actual
reason for calling instead of the generic greeting (see the OUTBOUND CALLS
section of SYSTEM_PROMPT in src/agent.py):
  - medication_reminder / vaccination_reminder: pass --detail describing
    what to remind them about, e.g. "your evening metformin dose".
  - triage_followup: no --detail needed — it uses the caller's
    last_triage_outcome already on file (see save_caller_profile), keyed by
    this same phone number. If nothing is on file, the agent just asks
    generally how they've been since last time.
"""

import argparse
import asyncio
import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env.local")

import livekit.api as api  # noqa: E402

CALL_TYPES = ("medication_reminder", "vaccination_reminder", "triage_followup")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("phone_number", help="E.164 format, e.g. +919876543210")
    parser.add_argument(
        "--type",
        dest="call_type",
        choices=CALL_TYPES,
        default=None,
        help="Purposeful call reason. Omit for a plain outbound call with the generic greeting.",
    )
    parser.add_argument(
        "--detail",
        default=None,
        help="What to remind them about (medication_reminder / vaccination_reminder only).",
    )
    args = parser.parse_args()

    if not args.phone_number.startswith("+"):
        parser.error(
            f"Phone number should be in E.164 format (e.g. +919876543210), got: {args.phone_number!r}"
        )
    if (
        args.call_type in ("medication_reminder", "vaccination_reminder")
        and not args.detail
    ):
        parser.error(
            f"--type {args.call_type} requires --detail (what to remind them about)"
        )
    if args.call_type == "triage_followup" and args.detail:
        parser.error(
            "--detail is ignored for triage_followup — it uses last_triage_outcome on file instead"
        )

    return args


async def main() -> None:
    args = parse_args()
    room_name = f"outbound-{uuid.uuid4().hex[:12]}"

    metadata = {"phone_number": args.phone_number}
    if args.call_type:
        metadata["call_type"] = args.call_type
    if args.detail:
        metadata["detail"] = args.detail

    lkapi = api.LiveKitAPI(
        url=os.environ["LIVEKIT_URL"],
        api_key=os.environ["LIVEKIT_API_KEY"],
        api_secret=os.environ["LIVEKIT_API_SECRET"],
    )
    try:
        await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name="my-agent",
                room=room_name,
                metadata=json.dumps(metadata),
            )
        )
        reason = f" ({args.call_type})" if args.call_type else ""
        print(
            f"Dispatched outbound call to {args.phone_number}{reason} in room {room_name!r}."
        )
        print("Watch the worker's logs (src/agent.py dev) for call progress.")
    finally:
        await lkapi.aclose()


if __name__ == "__main__":
    asyncio.run(main())
