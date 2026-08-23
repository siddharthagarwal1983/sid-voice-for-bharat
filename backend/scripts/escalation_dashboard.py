"""A minimal local dashboard for human-escalation requests (see
src/escalation.py and the HUMAN ESCALATION section of src/agent.py's
SYSTEM_PROMPT).

    uv run python scripts/escalation_dashboard.py
    uv run python scripts/escalation_dashboard.py --port 8800

Then open http://localhost:8000 (or your --port). Every request the agent
creates via create_escalation is a real row in the same SQLite database
(data/agent.db) the rest of the backend already uses — this is just a
read/update view over it, no separate service to run or configure. Lets a
human see open/in-progress/resolved requests and mark one resolved (with an
optional short note); marking one resolved is what makes it eligible for
scripts/escalation_followup_calls.py to place the Day-6 outbound callback.

Stdlib only (http.server + sqlite3, both already used elsewhere in this
project) — no new dependency for a page that's meant to be a simple, always-
available fallback destination even with no external services configured.
"""

import argparse
import contextlib
import html
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT / "src"))

import db  # noqa: E402
import escalation  # noqa: E402

_STATUS_ORDER = ("open", "in_progress", "resolved")
_URGENCY_COLOR = {
    "emergency": "#c0392b",
    "high": "#e67e22",
    "medium": "#d4ac0d",
    "low": "#7f8c8d",
}


def _fmt_time(ts: float | None) -> str:
    if not ts:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def _row_html(row: dict) -> str:
    urgency_color = _URGENCY_COLOR.get(row["urgency"], "#7f8c8d")
    reason_label = escalation.ESCALATION_REASONS.get(row["reason"], row["reason"])
    resolve_form = (
        f"""
        <form method="post" action="/resolve" class="resolve-form">
            <input type="hidden" name="id" value="{html.escape(row["id"])}">
            <input type="text" name="resolution_note" placeholder="Resolution note (optional)">
            <button type="submit">Mark resolved</button>
        </form>
        """
        if row["status"] != "resolved"
        else f"<span class='muted'>Resolved {_fmt_time(row['resolved_at'])}"
        + (
            " &middot; callback placed"
            if row["called_back"]
            else " &middot; callback pending"
            if str(row.get("user_id") or "").startswith("+")
            else ""
        )
        + "</span>"
    )
    return f"""
    <tr>
        <td><code>{html.escape(row["id"])}</code></td>
        <td><span class="badge" style="background:{urgency_color}">{html.escape(row["urgency"])}</span></td>
        <td>{html.escape(row["status"])}</td>
        <td>{html.escape(reason_label)}</td>
        <td>{html.escape(row["who"] or "unidentified")}</td>
        <td>{html.escape(row["what_happened"] or "")}</td>
        <td>{html.escape(row["already_checked"] or "-")}</td>
        <td>{html.escape(row["language"] or "-")}</td>
        <td>{html.escape(row["preferred_contact"] or "-")}</td>
        <td>{html.escape(row["notify_status"] or "-")}</td>
        <td>{_fmt_time(row["created_at"])}</td>
        <td>{resolve_form}</td>
    </tr>
    """


def _page_html() -> str:
    rows = db.list_escalations()
    by_status = {s: [r for r in rows if r["status"] == s] for s in _STATUS_ORDER}
    sections = []
    for status in _STATUS_ORDER:
        section_rows = by_status[status]
        sections.append(
            f"<h2>{status} ({len(section_rows)})</h2>"
            + (
                """
                <table>
                <thead><tr>
                    <th>Ref</th><th>Urgency</th><th>Status</th><th>Reason</th>
                    <th>Who</th><th>What happened</th><th>Already checked</th>
                    <th>Language</th><th>Preferred contact</th><th>Notify</th>
                    <th>Created</th><th></th>
                </tr></thead>
                <tbody>
                """
                + "".join(_row_html(r) for r in section_rows)
                + "</tbody></table>"
                if section_rows
                else "<p class='muted'>None.</p>"
            )
        )

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>HealthMitra — Escalations</title>
<style>
    body {{ font-family: -apple-system, sans-serif; margin: 2rem; color: #222; }}
    h1 {{ margin-bottom: 0.2rem; }}
    .muted {{ color: #888; font-size: 0.9rem; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; font-size: 0.85rem; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f5f5f5; }}
    .badge {{ color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.75rem; text-transform: uppercase; }}
    .resolve-form {{ display: flex; gap: 4px; }}
    .resolve-form input[type=text] {{ width: 140px; }}
    code {{ font-size: 0.8rem; }}
</style>
</head>
<body>
<h1>HealthMitra — Human Escalations</h1>
<p class="muted">Local dashboard over data/agent.db. Auto-refreshes every 15s. Set ESCALATION_WEBHOOK_URL in .env.local to also deliver these to Slack/Discord/a generic webhook.</p>
{"".join(sections)}
<script>setTimeout(() => location.reload(), 15000);</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter default logging
        print(f"[dashboard] {self.address_string()} - {fmt % args}")

    def do_GET(self):
        if self.path != "/":
            self.send_response(404)
            self.end_headers()
            return
        body = _page_html().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/resolve":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        fields = parse_qs(self.rfile.read(length).decode("utf-8"))
        escalation_id = fields.get("id", [""])[0]
        note = fields.get("resolution_note", [""])[0] or None
        if escalation_id:
            db.set_escalation_status(escalation_id, "resolved", resolution_note=note)
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    db.init_db()
    server = HTTPServer(("localhost", args.port), Handler)
    print(f"Escalation dashboard at http://localhost:{args.port}")
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()


if __name__ == "__main__":
    main()
