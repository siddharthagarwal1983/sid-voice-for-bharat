# Backend — Voice Agent with Murf Falcon TTS

The Python backend for the Voice Agent Starter. It runs a real-time voice AI pipeline using [LiveKit Agents](https://docs.livekit.io/agents), connecting Murf Falcon TTS, Deepgram STT, and Google Gemini into a single conversational agent.

## How It Works

```
User speaks → [Deepgram STT] → text → [Gemini LLM] → response → [Murf Falcon TTS] → audio → User hears
```

LiveKit handles the real-time audio transport. The agent connects to LiveKit as a participant, listens for user speech, and responds with synthesized audio.

## Setup

### 1. Install dependencies

```bash
cd backend
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env.local
```

Fill in your keys in `.env.local`:

| Variable             | Where to get it                                           |
| -------------------- | --------------------------------------------------------- |
| `LIVEKIT_URL`        | [LiveKit Cloud](https://cloud.livekit.io/) → Settings     |
| `LIVEKIT_API_KEY`    | [LiveKit Cloud](https://cloud.livekit.io/) → Settings     |
| `LIVEKIT_API_SECRET` | [LiveKit Cloud](https://cloud.livekit.io/) → Settings     |
| `MURF_API_KEY`       | [murf.ai/api/dashboard](https://murf.ai/api/dashboard)    |
| `DEEPGRAM_API_KEY`   | [deepgram.com](https://console.deepgram.com/)             |
| `GOOGLE_API_KEY`     | [aistudio.google.com](https://aistudio.google.com/apikey) |

For LiveKit Cloud users, you can auto-populate LiveKit credentials:

```bash
lk cloud auth
lk app env -w -d .env.local
```

### 3. Download models

```bash
uv run python src/agent.py download-files
```

This downloads Silero VAD and the LiveKit turn detector models.

### 4. Run the agent

```bash
# Development mode (auto-reload)
uv run python src/agent.py dev

# Or test directly in your terminal (no frontend needed)
uv run python src/agent.py console

# Production
uv run python src/agent.py start
```

## Configuration

All configuration lives in [`src/agent.py`](src/agent.py).

### System prompt

The `SYSTEM_PROMPT` constant at the top of `agent.py` controls what your agent does. Change it to build any voice-powered use case.

#### Example prompts

**Customer Support (default):**

```
You are a friendly and efficient customer support agent for a tech company. Help users with account issues, billing questions, and product troubleshooting. Be concise, empathetic, and solution-oriented. If you don't know something, say so honestly and offer to escalate.
```

**Language Tutor:**

```
You are a patient and encouraging language tutor helping the user practice conversational Spanish. Speak primarily in Spanish but switch to English to explain grammar or vocabulary when needed. Correct mistakes gently and suggest better phrasing. Keep conversations natural and fun.
```

**AI Receptionist:**

```
You are a professional receptionist for a medical clinic. Help callers schedule appointments, answer questions about office hours and services, and take messages for doctors. Be warm but efficient. Ask for the caller's name and reason for calling upfront.
```

**Interview Coach:**

```
You are an experienced interview coach. Conduct mock interviews with the user for software engineering roles. Ask one behavioral or technical question at a time, let the user answer fully, then give specific feedback on their response — what was strong, what could improve, and a suggested reframe. Keep the tone encouraging but honest.
```

**Sales Assistant:**

```
You are a knowledgeable sales assistant for an electronics store. Help customers find the right product by asking about their needs, budget, and preferences. Compare options clearly, highlight trade-offs, and make a recommendation. Never be pushy — focus on helping the customer make the best decision for them.
```

**Fitness Coach:**

```
You are an upbeat personal fitness coach. Help users plan workouts, suggest exercises for specific muscle groups, and answer questions about form and technique. Ask about their fitness level and any injuries before recommending exercises. Keep instructions clear and motivating.
```

**Storyteller / Bedtime Narrator:**

```
You are a creative storyteller who tells original bedtime stories for children aged 4–8. Ask the child (or parent) for a character name, a favorite animal, and a setting, then weave a short, calming story. Use vivid but simple language. End each story on a peaceful, sleepy note.
```

**Meeting Summarizer:**

```
You are a meeting assistant. The user will describe what happened in a meeting or read you their notes. Summarize the key decisions, action items (with owners if mentioned), and any open questions. Be concise and structured. Ask clarifying questions if something is ambiguous.
```

**Trivia Game Host:**

```
You are an enthusiastic trivia game host. Ask the user one trivia question at a time from a mix of categories — science, history, pop culture, geography, and sports. Wait for their answer, tell them if they're right or wrong, give a brief fun fact, then move to the next question. Keep score and announce it every 5 questions.
```

**Mental Health Check-in Companion:**

```
You are a gentle, non-clinical wellness companion. Help users talk through their day, reflect on how they're feeling, and practice simple grounding exercises like deep breathing or gratitude lists. You are not a therapist — if the user expresses serious distress or mentions self-harm, gently encourage them to reach out to a professional or crisis helpline.
```

### Voice

Set the `voice` argument in the `murf.TTS(...)` call:

```python
tts=murf.TTS(
    voice="en-US-matthew",    # Change this
    style="Conversation",
    tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
    text_pacing=True
)
```

Some voice options:

| Voice ID | Description                      |
| -------- | -------------------------------- |
| `Anisha` | Indian English, female (default) |
| `Pooja`  | Indian English, female           |
| `Samar`  | Indian English, male             |
| `Amara`  | US English, female               |
| `Hazel`  | UK English, female               |
| `Bertie` | UK English, male                 |
| `Gordon` | US English, male                 |

Browse all 150+ voices: [Murf Voice Library](https://murf.ai/api/docs/voices-styles/voice-library).

### STT (Speech-to-Text)

Default is Deepgram Nova-3. Change in the `AgentSession(stt=...)` call:

```python
stt=deepgram.STT(model="nova-3")
```

### LLM

Default is Google Gemini. To switch:

- **Gemini (default):** Set `GOOGLE_API_KEY` in `.env.local`
- **OpenAI:** Set `OPENAI_API_KEY`, install `livekit-agents[openai]`, and change the `llm=` argument

## Health Data Tools

The default HealthMitra prompt (see `SYSTEM_PROMPT` in `agent.py`) gives the agent two domain-specific function tools it decides to call on its own, mid-conversation, without being asked. Both live in [`src/health_tools.py`](src/health_tools.py) and are wired up as `@function_tool` methods on `Assistant` in [`src/agent.py`](src/agent.py).

### 1. `classify_symptom_triage` — LOCAL data, not live

Routes a caller's described symptoms to `emergency` / `phc` / `home_care` using a **fixed, hand-built keyword ruleset** (`_TRIAGE_RULES` in `health_tools.py`), modeled on the same red-flag list already in the system prompt (chest pain, breathlessness, heavy bleeding, high fever, dehydration, etc.).

This is intentionally **not** a live medical API — there isn't a reliable public one for this, and for a no-diagnosis assistant a fixed, inspectable ruleset is arguably the safer choice anyway: every routing decision traces back to one specific matched keyword, never to an ungrounded LLM guess. The tool result includes `ruleset_version` so it's clear which version of the rules produced the routing.

The agent calls this as soon as a caller describes symptoms, before deciding how to route the call — see the `SYMPTOM TRIAGE` section of `SYSTEM_PROMPT`.

### 2. `find_nearby_health_facility` — LIVE data, with a local fallback

Looks up nearby government hospitals/PHCs for a district. It tries a **live** fetch against [OpenStreetMap's Nominatim](https://nominatim.org/) search API first (free, keyless, real public data — no API key needed). If that fails — timeout, rate limit, no network, all realistic on the connections these callers are on — it falls back to a **small hand-curated local list** (`_LOCAL_FALLBACK` in `health_tools.py`) of well-known government hospitals for about a dozen major Indian districts, and is explicit with the caller about which source it used ("fetched live just now" vs. "from a saved local reference list, which may not be fully up to date"). If neither source has anything for the caller's district, the agent says so plainly and points them to the 104 health helpline or their local ASHA worker — it never invents a facility name or address.

Every result also carries a `fetched_at` timestamp, since "the hospital I found yesterday" and "the hospital I found just now" aren't interchangeable for someone deciding where to go right now.

To see the failure path yourself, simulate the data source being down:

```bash
uv run python -c "
import sys; sys.path.insert(0, 'src')
import health_tools as h, httpx
httpx.get = lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectTimeout('simulated'))
print(h.find_facilities('Mumbai'))       # -> falls back to the local list
print(h.find_facilities('SomeVillage'))  # -> honest 'not_found', nothing invented
"
```

### Tool chaining with Day 4's caller memory

`find_nearby_health_facility` takes an optional `district` argument. If the model doesn't pass one, the tool itself calls `db.get_caller()` — the same lookup backing the Day 4 `lookup_caller` tool — and reuses the district saved on that caller's profile (if they previously consented to `save_caller_profile` saving one). This chain is implemented in code, not just prompted, so a returning caller who already shared their district is never asked for it twice. `save_caller_profile` now also accepts a `district` field for this purpose.

### Pushed to the UI

Both tools publish their structured result (not just spoken text) to the room via `local_participant.send_text(..., topic=...)` on `"healthmitra-triage"` / `"healthmitra-facility"`. The frontend (`frontend/components/app/health-data-panel.tsx`) subscribes with `useTextStream` and renders each result as its own card in a horizontal rail — new cards land on the right and the rail auto-scrolls to them, sliding earlier cards left rather than growing the page taller. Facility names/addresses and the triage level are visible on screen while the agent is still speaking them, not just spoken once and gone.

`agent.py`'s `metrics_collected` handler publishes STT/TTS pipeline latency the same way, on topic `"healthmitra-metrics"` — EOU transcription delay for "how long until it heard you," TTS time-to-first-byte for "how long until it started speaking." The frontend shows both live in `frontend/components/app/metrics-panel.tsx`, bottom-left of the page.

## Outbound calling

HealthMitra can place outbound calls — e.g. a proactive check-in — through a Twilio SIP trunk, in addition to answering inbound calls.

### One-time setup

1. Fill in the `TWILIO_*` variables in `.env.local` (Elastic SIP Trunking → your trunk → **Termination** for `TWILIO_SIP_TERM_URI`, **Credential Lists** for the SIP username/password).
2. Create the LiveKit outbound trunk that routes through it:
   ```bash
   uv run python scripts/setup_outbound_trunk.py
   ```
   This prints a trunk ID (`ST_...`) — paste it into `.env.local` as `LIVEKIT_SIP_OUTBOUND_TRUNK_ID`. Re-running the script always creates a new trunk, so don't run it twice without checking `LiveKitAPI().sip.list_outbound_trunk()` first.

### Placing a call

With the worker running (`uv run python src/agent.py dev`):

```bash
uv run python scripts/make_outbound_call.py +9198XXXXXXXX
```

This dispatches `my-agent` into a fresh room with the phone number in the job's dispatch metadata (`{"phone_number": "+91..."}`). `my_agent()` in `agent.py` reads that metadata (`_extract_outbound_call`), dials out over the trunk and blocks until answered (`_dial_outbound_participant`, `wait_until_answered=True`), then falls into the exact same caller-lookup-and-greeting flow as an inbound call — from the agent's perspective, an answered plain outbound call looks identical to an inbound one from that point on. If the call isn't answered, the job ends cleanly (logged, no crash, no attempt to talk to an empty room).

Note this places a real call and may incur Twilio charges — there's no simulated/dry-run mode.

### Purposeful calls — reminders & follow-ups

Pass `--type` to open with an actual reason for calling instead of the generic greeting (see the `OUTBOUND CALLS` section of `SYSTEM_PROMPT`):

```bash
# Medication / vaccination reminders — --detail says what to remind them about
uv run python scripts/make_outbound_call.py +9198XXXXXXXX \
  --type medication_reminder --detail "your evening metformin dose"

uv run python scripts/make_outbound_call.py +9198XXXXXXXX \
  --type vaccination_reminder --detail "your second flu vaccine dose, due this week"

# Follow-up after a triage escalation — no --detail needed
uv run python scripts/make_outbound_call.py +9198XXXXXXXX --type triage_followup
```

`triage_followup` doesn't take a `--detail` — it pulls `last_triage_outcome` from the caller's saved profile (the same one `save_caller_profile` writes to during any call, keyed by phone number) and has the agent reference it naturally — e.g. "you'd mentioned going to the PHC — were you able to, and how are you feeling now?" If nothing is on file for that number, the agent just asks generally how they've been since the last conversation instead of inventing a reason.

An unrecognized or omitted `--type` falls back to the plain generic-greeting call above — this is a graceful default, not an error.

### Call outcomes & retries

Inbound calls only have one shape — the caller reached you. Outbound calls have several ways to *not* reach a live conversation, each handled differently:

| Outcome | Detected | Behavior | Retry (when enabled) |
| --- | --- | --- | --- |
| `no_answer` | SIP timeout (408/480/487), or LiveKit's own `ringing_timeout` cancellation — see below | Job ends immediately, nothing spoken | 30 min later, up to 3 attempts |
| `busy` | SIP busy (486/600/603) from the dial itself | Job ends immediately, nothing spoken | 10 min later, up to 3 attempts |
| `voicemail` | The LLM recognizes an answering-machine greeting after the opening line (`report_voicemail_detected` tool) — there's no SIP-level or platform AMD signal available over raw SIP trunking, so this is a judgment call, same as any other tool | Leaves one short, call_type-appropriate message, then hangs up | 4 hours later, up to 2 attempts |
| `immediate_hangup` | The SIP participant disconnects within 8s of answering, without the agent having called `end_call` itself | Job ends, nothing more said | 1 day later, up to 2 attempts (don't be pushy about a likely rejection) |
| `failed` | Anything else — invalid number, no trunk configured, carrier/account error | Job ends immediately | None — needs a human to look at it, not an automatic retry |
| `answered` | Connected, no special case triggered | Normal conversation, ends via `end_call` | None — it worked |

Classification is built on **two** real errors captured from this project's own LiveKit SIP trunk, not just one — worth knowing since they look different:
- A call the far end actively rejects carries a real SIP status in `TwirpError.metadata['sip_status_code']` (e.g. a carrier/account rejection).
- A call that just **rings out** — nobody answers before `ringing_timeout` — raises `TwirpError(code="canceled", message="...sip request timed out...")` with **no `sip_status_code` in metadata at all**, since LiveKit cancels client-side rather than forwarding a status the far end never sent. Classifying on metadata alone missed this entirely and silently mislabeled every ring-timeout as `failed` — caught by placing a real no-answer test call, not by reading the docs. See `_classify_sip_error` in `agent.py`.

`immediate_hangup` is a plain `participant_disconnected` timer check. Every outcome is logged to the `outbound_call_attempts` table (`db.py`) via `db.record_outbound_attempt`.

#### Retry rules live in `config/retry_policy.json`, not in code

```json
{
  "enabled": false,
  "outcomes": {
    "no_answer": { "max_attempts": 3, "delay_minutes": 30 },
    "busy": { "max_attempts": 3, "delay_minutes": 10 },
    "voicemail": { "max_attempts": 2, "delay_minutes": 240 },
    "immediate_hangup": { "max_attempts": 2, "delay_minutes": 1440 }
  }
}
```

`agent.py`'s `next_retry_at()` re-reads this file on every retry decision (not cached at import), so changing a delay or attempt cap takes effect on the next outbound call without restarting the worker. The top-level `"enabled"` flag is a single master on/off switch for all outbound retries — it's currently **`false`**: outcomes are still classified and recorded normally, nothing just gets automatically retried yet. Flip it to `true` when you're ready to turn retries on.

Retries are never executed inline in the job that hit the outcome — they're queued and picked up by a separate pass:

```bash
uv run python scripts/retry_outbound_calls.py
```

This queries `db.due_outbound_retries()` for attempts whose delay has elapsed, re-dispatches each one with `attempt_number` incremented (so the policy's `max_attempts` cap is respected), and marks it retried so it's never picked up twice. Run this periodically — e.g. on a cron schedule — it does one pass and exits rather than running as its own scheduler. While `"enabled": false`, it prints a clear message and exits immediately instead of silently doing nothing.

## Testing

The project includes an eval suite based on the LiveKit Agents [testing framework](https://docs.livekit.io/agents/build/testing/):

```bash
uv run pytest
```

Tests are in [`tests/test_agent.py`](tests/test_agent.py) and use LLM-as-judge evaluations to verify the agent behaves correctly (friendly greetings, grounding, refusing harmful requests).

To run tests in CI, you'll need to add `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET` as repository secrets.

## Deployment

### Railway

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.com/deploy/tIVCF1?referralCode=cNjn2P&utm_medium=integration&utm_source=template&utm_campaign=generic)

Set these environment variables in Railway:

- `MURF_API_KEY`
- `DEEPGRAM_API_KEY`
- `GOOGLE_API_KEY`
- `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`

### Docker

A production-ready [Dockerfile](Dockerfile) is included:

```bash
docker build -t murf-voice-agent .
docker run --env-file .env.local murf-voice-agent
```

## Project Structure

```
backend/
├── src/
│   └── agent.py          # Agent entrypoint — pipeline, prompt, config
├── config/
│   └── retry_policy.json # Outbound retry delays/attempt caps + on/off switch
├── scripts/
│   ├── setup_outbound_trunk.py   # One-time: create the Twilio outbound SIP trunk
│   ├── make_outbound_call.py     # Place an outbound call
│   └── retry_outbound_calls.py   # Re-dispatch due outbound retries (run periodically)
├── tests/
│   └── test_agent.py     # LLM-judged eval suite
├── .env.example           # Environment variable template
├── pyproject.toml         # Python dependencies (uv)
├── Dockerfile             # Production container
└── railway.toml           # Railway deploy config
```

## Links

- [Murf Falcon TTS Docs](https://murf.ai/api/docs/text-to-speech/streaming)
- [Murf Voice Library](https://murf.ai/api/docs/voices-styles/voice-library)
- [LiveKit Agents Docs](https://docs.livekit.io/agents)
- [Deepgram Nova-3 Docs](https://developers.deepgram.com)

## License

MIT — see [LICENSE](LICENSE).
