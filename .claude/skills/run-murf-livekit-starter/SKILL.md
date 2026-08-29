---
name: run-murf-livekit-starter
description: Build, run, and drive the HealthMitra voice agent (murf-livekit-starter) — a LiveKit + Murf TTS + Deepgram voice assistant with a Next.js frontend. Use when asked to start, run, launch, or screenshot the app, take a screenshot of the call UI, place a test call, or run the backend test suite.
---

This is a two-process app that must run together: a Python LiveKit
agent worker (`backend/`) and a Next.js frontend (`frontend/`) that
serves the call UI and mints LiveKit tokens. Both connect out to a real
LiveKit Cloud project (no local `livekit-server` needed once
`.env.local` points at a cloud URL). Drive it with the committed
Playwright driver at `.claude/skills/run-murf-livekit-starter/driver.mjs`
— it loads the call UI, starts a call, waits for the agent to join and
greet, screenshots each stage, and ends the call.

All paths below are relative to the repo root.

## Prerequisites

Already satisfied in this environment (verified 2026-08-29): `uv`,
`pnpm`, and Node are on `PATH`. If starting from scratch:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # uv (Python)
npm install -g pnpm                               # pnpm (Node)
```

## Setup

```bash
cd backend && uv sync && uv run python src/agent.py download-files && cd ..
cd frontend && pnpm install && cd ..
```

Env vars — `backend/.env.local` and `frontend/.env.local` (copy from
each dir's `.env.example`):

```bash
# backend/.env.local — required
LIVEKIT_URL=...            # wss://<project>.livekit.cloud
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
MURF_API_KEY=...
DEEPGRAM_API_KEY=...
GOOGLE_API_KEY=...         # or OPENAI_API_KEY, depending on LLM choice

# frontend/.env.local — required
LIVEKIT_URL=...            # must match the backend's LiveKit project
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
```

In this repo both files already exist and are filled in (pointing at
a real LiveKit Cloud project, `aurora-y6tokydz`) — nothing to configure.

Driver dependencies (one-time, isolated from the app's own
`package.json`):

```bash
cd .claude/skills/run-murf-livekit-starter && npm install
npx playwright install chromium   # no-op if already cached
```

## Build

No separate build step for local dev — both services run in dev mode.

## Run (agent path)

Start both services in the background and wait for each to report ready.
`start_app.sh` also tries to launch a local `livekit-server --dev`, which
is unnecessary (and just runs unused) when `.env.local` already points
at LiveKit Cloud — start the two app processes directly instead:

```bash
(cd backend && uv run python src/agent.py dev > /tmp/backend_agent.log 2>&1 &)
(cd frontend && pnpm dev > /tmp/frontend_dev.log 2>&1 &)

for i in $(seq 1 30); do curl -sf http://localhost:3000 >/dev/null && break; sleep 1; done
for i in $(seq 1 30); do grep -q "registered worker" /tmp/backend_agent.log 2>/dev/null && break; sleep 1; done
```

Then drive it:

```bash
cd .claude/skills/run-murf-livekit-starter
node driver.mjs                 # full call: connect, greet, end call
node driver.mjs --landing-only  # just load the page, no call
```

Screenshots land in `.claude/skills/run-murf-livekit-starter/screenshots/`:
`1-landing.png`, `2-connecting.png`, `3-connected.png`, `4-ended.png`
(the last three only for the full run). The driver prints the visible
call-UI text and any browser console errors, and exits non-zero if
there were console errors.

Stop cleanly when done:

```bash
lsof -ti:3000 -sTCP:LISTEN | xargs -r kill
pkill -f "src/agent.py dev"
```

## Run (human path)

```bash
./start_app.sh   # from repo root — starts livekit-server --dev (if installed), backend, and frontend
```

Open `http://localhost:3000`, click "Start talking", grant mic access,
talk to HealthMitra. Ctrl-C to stop (kills the three backgrounded jobs).

## Test

```bash
cd backend && uv run pytest tests/test_agent.py -q
```

13 tests. Expect all to pass, but see Gotchas below — some are flaky.

---

## Gotchas

- **LLM-judge tests are flaky, not broken.** `test_grounding` and
  `test_specialist_does_not_bounce_back_on_stale_context` each failed
  once in this session on a nitpicky judge verdict over an already-correct
  response, then passed cleanly on immediate rerun (3x and 1x
  respectively). Before treating a single failure here as a real
  regression, rerun just that test.
- **Playwright `.click()` returns before the UI re-renders.** Screenshotting
  immediately after clicking "Start talking" can still show the landing
  page (confirmed: identical screenshot + file size to the pre-click
  shot). `driver.mjs` waits for `text=Connecting` (3s timeout, ignored if
  missed — it's often too transient to catch) before taking that shot.
- **A double-triggered "Start talking" click leaves an orphaned LiveKit
  room.** It idles for ~35s and the agent force-disconnects it
  automatically (`Room idle for 35s — force-disconnecting room ...` in
  the backend log) — harmless, but don't mistake it for an error if you
  see two `voice_assistant_room_*` ids in one driver run.
- **No local `timeout` command on macOS/zsh.** Use a `for i in $(seq 1 N); do ...; sleep 1; done` polling loop instead (used above), not `timeout N bash -c '...'`.
- **`start_app.sh` always tries `livekit-server --dev`** even though
  this repo's `.env.local` points at LiveKit Cloud — it's a harmless
  unused extra process, not a sign of misconfiguration.

## Troubleshooting

- **`command not found: timeout`** (macOS/zsh): see the polling-loop
  gotcha above.
- **Driver hangs on `waitForSelector("text=Listening")`**: check
  `/tmp/backend_agent.log` for `registered worker` — if the worker
  never registered, the LiveKit credentials in `backend/.env.local`
  are likely wrong or the project URL doesn't match `frontend/.env.local`.
- **Port 3000 already in use on relaunch**: a previous `pnpm dev` is
  still bound. `lsof -ti:3000 -sTCP:LISTEN | xargs -r kill` before
  restarting.
