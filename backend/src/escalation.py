"""Human-escalation: when HealthMitra should stop and hand a caller off to a
person, and how that request is packaged, deduped, and delivered.

Two escalation reasons only (see the HUMAN ESCALATION section of
SYSTEM_PROMPT in agent.py for exactly when the agent is told to use each):

1. red_flag_symptom  — the caller has a red-flag/emergency symptom, or asks
   the agent to diagnose them or name a specific medicine (a Hard Refusal
   the agent can't resolve on its own). This is on top of, not instead of,
   the immediate spoken 108/hospital guidance — 108 is the emergency
   response; this ticket gets a human to review and follow up.
2. unresolved_request — the agent genuinely couldn't resolve the caller's
   need with its own tools (knowledge base found nothing relevant, facility
   lookup found nothing, or the caller directly asked for a human).

Never sent without the caller's explicit consent — see create_escalation in
agent.py, which refuses to call db.create_escalation at all unless
consent_given=True, the same pattern as Day 4's save_caller_profile.
"""

import logging
import os
import re

import httpx

logger = logging.getLogger("agent.escalation")

URGENCY_LEVELS = ("low", "medium", "high", "emergency")

ESCALATION_REASONS = {
    "red_flag_symptom": "Red-flag symptom or request for diagnosis — needs clinical human review",
    "unresolved_request": "Caller's issue could not be resolved by the assistant",
}

_WEBHOOK_TIMEOUT_S = 5.0

# ---------------------------------------------------------------------------
# Redaction — defense in depth. The system prompt already instructs the
# agent to never pass sensitive values into a summary in the first place;
# this is a second, code-level backstop in case it does anyway, since a
# regex can't be talked out of doing its job the way a prompt can be.
# ---------------------------------------------------------------------------

# 6+ consecutive digits (optionally space/dash-grouped) — covers OTPs, PINs,
# account numbers, card numbers, and Aadhaar numbers without needing to name
# each format individually.
_LONG_NUMBER_RE = re.compile(r"(?:\d[\d\s-]{4,}\d){1}(?<=\d)")
_LONG_NUMBER_MIN_DIGITS = 6

# A sensitive keyword followed by whatever value comes after it up to the
# next sentence boundary — e.g. "password is hunter2" -> "password [redacted]".
_KEYWORD_VALUE_RE = re.compile(
    r"\b(password|otp|pin|cvv|one[- ]time code)\b[^.,;\n]*",
    re.IGNORECASE,
)


def redact(text: str | None) -> str:
    """Strip likely OTPs/PINs/passwords/account numbers out of free text
    before it's stored or sent anywhere. Never raises; empty input returns
    empty output.
    """
    if not text:
        return ""

    def _replace_long_number(match: re.Match) -> str:
        digit_count = sum(ch.isdigit() for ch in match.group(0))
        return (
            "[redacted]" if digit_count >= _LONG_NUMBER_MIN_DIGITS else match.group(0)
        )

    result = _LONG_NUMBER_RE.sub(_replace_long_number, text)
    result = _KEYWORD_VALUE_RE.sub(lambda m: f"{m.group(1)} [redacted]", result)
    return result


# ---------------------------------------------------------------------------
# Delivery — a real, external destination for the request, best-effort.
#
# A local database row + backend/scripts/escalation_dashboard.py is always
# the source of truth (works with zero configuration). If
# ESCALATION_WEBHOOK_URL is also set, each new/updated escalation is POSTed
# there too — this one URL works as-is for a Slack "Incoming Webhook" (reads
# the `text` field) or a generic JSON sink (e.g. webhook.site for testing);
# for Discord, use a Discord webhook URL with "/slack" appended, which makes
# Discord accept the same Slack-shaped payload.
# ---------------------------------------------------------------------------


def notify_webhook(escalation: dict) -> str:
    """Best-effort delivery to an external destination. Returns
    "sent"/"failed"/"not_configured" — never raises, so a flaky webhook can
    never break the call or lose the (already-saved-to-SQLite) request.
    """
    url = os.environ.get("ESCALATION_WEBHOOK_URL")
    if not url:
        return "not_configured"

    summary = (
        f":rotating_light: HealthMitra escalation `{escalation['id']}` "
        f"[{escalation['urgency'].upper()}] — {ESCALATION_REASONS.get(escalation['reason'], escalation['reason'])}\n"
        f"Who: {escalation['who'] or 'unidentified caller'}\n"
        f"What happened: {escalation['what_happened']}\n"
        f"Already checked: {escalation['already_checked'] or '-'}\n"
        f"Language: {escalation['language'] or 'unknown'} | "
        f"Preferred follow-up: {escalation['preferred_contact'] or 'unspecified'}"
    )
    try:
        response = httpx.post(
            url,
            json={"text": summary, "content": summary, "escalation": escalation},
            timeout=_WEBHOOK_TIMEOUT_S,
        )
        response.raise_for_status()
        return "sent"
    except (httpx.TimeoutException, httpx.HTTPError) as exc:
        logger.warning(
            "Escalation webhook delivery failed for %s: %s", escalation["id"], exc
        )
        return "failed"
