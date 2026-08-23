"""One-time setup: create the LiveKit SIP outbound trunk that routes calls
through Twilio, using the TWILIO_* credentials in .env.local.

Run once per LiveKit project:

    uv run python scripts/setup_outbound_trunk.py

Prints the resulting trunk ID — paste it into .env.local (and Railway/prod
env) as LIVEKIT_SIP_OUTBOUND_TRUNK_ID. Safe to re-run: it always creates a
new trunk rather than silently reusing one, so check `lk sip outbound list`
(or list_outbound_trunk) first if you suspect one already exists, to avoid
accumulating duplicates.
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import livekit.api as api

load_dotenv(Path(__file__).resolve().parent.parent / ".env.local")

REQUIRED_VARS = [
    "LIVEKIT_URL",
    "LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET",
    "TWILIO_SIP_TERM_URI",
    "TWILIO_SIP_USERNAME",
    "TWILIO_SIP_PASSWORD",
    "TWILIO_PHONE_NUMBER",
]


async def main() -> None:
    missing = [v for v in REQUIRED_VARS if not os.environ.get(v)]
    if missing:
        print(f"Missing required env vars in .env.local: {', '.join(missing)}")
        raise SystemExit(1)

    lkapi = api.LiveKitAPI(
        url=os.environ["LIVEKIT_URL"],
        api_key=os.environ["LIVEKIT_API_KEY"],
        api_secret=os.environ["LIVEKIT_API_SECRET"],
    )
    try:
        trunk = api.SIPOutboundTrunkInfo(
            name="HealthMitra Twilio Outbound",
            # The Twilio Elastic SIP Trunk's termination URI — where LiveKit
            # sends outbound calls to reach the PSTN via Twilio.
            address=os.environ["TWILIO_SIP_TERM_URI"],
            # Caller ID for outbound calls — must be a number on this Twilio
            # trunk.
            numbers=[os.environ["TWILIO_PHONE_NUMBER"]],
            # Twilio Credential List auth (username/password), configured on
            # the Twilio trunk to accept calls from LiveKit.
            auth_username=os.environ["TWILIO_SIP_USERNAME"],
            auth_password=os.environ["TWILIO_SIP_PASSWORD"],
        )
        resp = await lkapi.sip.create_outbound_trunk(
            api.CreateSIPOutboundTrunkRequest(trunk=trunk)
        )
        print(f"Created outbound trunk: {resp.sip_trunk_id}")
        print("Set this in .env.local as:")
        print(f"  LIVEKIT_SIP_OUTBOUND_TRUNK_ID={resp.sip_trunk_id}")
    finally:
        await lkapi.aclose()


if __name__ == "__main__":
    asyncio.run(main())
