"""Day-8 call-outcome dashboard for HealthMitra (Health Access track).

    uv run python scripts/call_dashboard.py
    uv run python scripts/call_dashboard.py --port 8801

Then open http://localhost:8000 (or your --port).

CALL SUCCESS DEFINITION (see src/agent.py for the full note): a call is
successful if the caller received safe guidance (a symptom-triage routing
decision, or a grounded scheme/eligibility answer) or an appropriate human
escalation. Every number on this page comes from real call_sessions rows
written by src/agent.py's session "close" handler as actual browser/SIP
calls happen — nothing here is hardcoded or simulated.

Stdlib only (http.server + sqlite3), same pattern as
scripts/escalation_dashboard.py — a read-only view over the same
data/agent.db the rest of the backend already writes to, no separate
service to run or configure.

PRIVACY: this page only ever reads the small set of outcome columns on
call_sessions (id, timestamps, channel, language, outcome, failure
category, a short track-outcome label, latency). It never reads the
`messages` table (full transcripts), `callers` table (names/medical facts),
or `participant_identity` (often a phone number) — see db.list_recent_calls
and db.get_dashboard_stats, which select only those safe columns by
construction. Do not add a query here that joins in any of those.
"""

import argparse
import contextlib
import html
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT / "src"))

import db  # noqa: E402

_FAILURE_LABEL = {
    "user_declined": "User declined to continue",
    "incomplete": "Incomplete — no resolution reached",
    "tool_failure": "Tool failure",
    "api_error": "STT/LLM/TTS error",
    "no_response": "No response from caller",
    "hangup": "Caller hung up",
}
_FAILURE_COLOR = {
    "user_declined": "#8e44ad",
    "incomplete": "#7f8c8d",
    "tool_failure": "#c0392b",
    "api_error": "#c0392b",
    "no_response": "#d4ac0d",
    "hangup": "#e67e22",
}


def _fmt_time(ts: float | None) -> str:
    if not ts:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def _fmt_duration(started: float | None, ended: float | None) -> str:
    if not started or not ended:
        return "-"
    secs = max(0, round(ended - started))
    return f"{secs // 60}m {secs % 60:02d}s" if secs >= 60 else f"{secs}s"


def _parse_query(path: str) -> dict:
    return {k: v[0] for k, v in parse_qs(urlparse(path).query).items() if v[0]}


def _stat_card(label: str, value: str, color: str = "#222") -> str:
    return f"""
    <div class="card">
        <div class="card-value" style="color:{color}">{html.escape(value)}</div>
        <div class="card-label">{html.escape(label)}</div>
    </div>
    """


def _bar_chart(breakdown: dict, colors: dict, empty_label: str) -> str:
    if not breakdown:
        return f"<p class='muted'>{html.escape(empty_label)}</p>"
    total = sum(breakdown.values())
    rows = []
    for key, count in sorted(breakdown.items(), key=lambda kv: -kv[1]):
        pct = round(count / total * 100, 1) if total else 0
        color = colors.get(key, "#3498db")
        rows.append(f"""
        <div class="bar-row">
            <div class="bar-label">{html.escape(key)}</div>
            <div class="bar-track"><div class="bar-fill" style="width:{pct}%;background:{color}"></div></div>
            <div class="bar-count">{count}</div>
        </div>
        """)
    return "".join(rows)


def _daily_chart(daily: dict) -> str:
    if not daily:
        return "<p class='muted'>No calls yet in this range.</p>"
    max_total = max(bucket["total"] for bucket in daily.values()) or 1
    cols = []
    for day, bucket in daily.items():
        total = bucket["total"]
        success = bucket["success"]
        height = round(total / max_total * 100)
        success_pct = round(success / total * 100) if total else 0
        cols.append(f"""
        <div class="day-col" title="{html.escape(day)}: {total} calls, {success} successful">
            <div class="day-bar" style="height:{height}%">
                <div class="day-bar-success" style="height:{success_pct}%"></div>
            </div>
            <div class="day-label">{html.escape(day[5:])}</div>
        </div>
        """)
    return f"<div class='day-chart'>{''.join(cols)}</div>"


def _row_html(row: dict) -> str:
    outcome = row["outcome"]
    if outcome == "success":
        outcome_badge = "<span class='badge' style='background:#27ae60'>success</span>"
    else:
        cat = row["failure_category"] or "unknown"
        color = _FAILURE_COLOR.get(cat, "#7f8c8d")
        label = _FAILURE_LABEL.get(cat, cat)
        outcome_badge = f"<span class='badge' style='background:{color}'>{html.escape(label)}</span>"
    latency = (
        f"{round(row['avg_response_latency_ms'])} ms"
        if row["avg_response_latency_ms"] is not None
        else "-"
    )
    return f"""
    <tr>
        <td>{_fmt_time(row["started_at"])}</td>
        <td>{_fmt_duration(row["started_at"], row["ended_at"])}</td>
        <td>{html.escape(row["channel"] or "-")}</td>
        <td>{html.escape(row["language"] or "-")}</td>
        <td>{outcome_badge}</td>
        <td>{html.escape(row["track_outcome"] or "-")}</td>
        <td>{latency}</td>
    </tr>
    """


def _page_html(query: dict) -> str:
    channel = query.get("channel") or None
    language = query.get("language") or None
    days = query.get("days")
    date_from = time.time() - int(days) * 86400 if days and days.isdigit() else None

    stats = db.get_dashboard_stats(
        date_from=date_from, channel=channel, language=language
    )
    recent = db.list_recent_calls(limit=50, channel=channel, language=language)
    filter_options = db.list_call_filter_options()

    def _option(value: str, current: str | None, label: str | None = None) -> str:
        selected = " selected" if value == (current or "") else ""
        return f"<option value='{html.escape(value)}'{selected}>{html.escape(label or value)}</option>"

    channel_options = "".join(
        [_option("", channel, "All channels")]
        + [_option(c, channel) for c in filter_options["channels"]]
    )
    language_options = "".join(
        [_option("", language, "All languages")]
        + [_option(lang, language) for lang in filter_options["languages"]]
    )
    days_options = "".join(
        _option(v, days, label)
        for v, label in [
            ("", "All time"),
            ("1", "Last 24h"),
            ("7", "Last 7 days"),
            ("30", "Last 30 days"),
        ]
    )

    avg_latency = (
        f"{round(stats['avg_latency_ms'])} ms" if stats["avg_latency_ms"] else "-"
    )

    table_rows = (
        "".join(_row_html(r) for r in recent)
        if recent
        else "<tr><td colspan='7' class='muted'>No calls recorded yet.</td></tr>"
    )

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>HealthMitra — Call Outcomes</title>
<style>
    body {{ font-family: -apple-system, sans-serif; margin: 2rem; color: #222; background: #fafafa; }}
    h1 {{ margin-bottom: 0.2rem; }}
    h2 {{ margin-top: 2rem; }}
    .muted {{ color: #888; font-size: 0.9rem; }}
    .filters {{ margin: 1rem 0; display: flex; gap: 0.5rem; align-items: center; }}
    select, button {{ padding: 6px 10px; font-size: 0.9rem; }}
    .cards {{ display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }}
    .card {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 1rem 1.5rem; min-width: 140px; }}
    .card-value {{ font-size: 2rem; font-weight: 700; }}
    .card-label {{ color: #666; font-size: 0.85rem; margin-top: 0.2rem; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 1rem; font-size: 0.85rem; background: white; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f5f5f5; }}
    .badge {{ color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.75rem; }}
    .bar-row {{ display: flex; align-items: center; gap: 0.5rem; margin: 4px 0; font-size: 0.85rem; }}
    .bar-label {{ width: 160px; }}
    .bar-track {{ flex: 1; background: #eee; border-radius: 4px; height: 14px; overflow: hidden; }}
    .bar-fill {{ height: 100%; }}
    .bar-count {{ width: 30px; text-align: right; }}
    .day-chart {{ display: flex; gap: 6px; align-items: flex-end; height: 140px; background: white; border: 1px solid #ddd; border-radius: 8px; padding: 1rem; }}
    .day-col {{ display: flex; flex-direction: column; align-items: center; justify-content: flex-end; height: 100%; width: 32px; }}
    .day-bar {{ width: 18px; background: #e74c3c; border-radius: 3px 3px 0 0; display: flex; flex-direction: column; justify-content: flex-end; min-height: 2px; }}
    .day-bar-success {{ background: #27ae60; width: 100%; border-radius: 3px 3px 0 0; }}
    .day-label {{ font-size: 0.65rem; color: #888; margin-top: 4px; }}
    .two-col {{ display: flex; gap: 2rem; flex-wrap: wrap; }}
    .two-col > div {{ flex: 1; min-width: 280px; }}
</style>
</head>
<body>
<h1>HealthMitra — Call Outcomes</h1>
<p class="muted">
    Success = the caller received safe guidance (symptom triage or a grounded scheme/eligibility answer)
    or an appropriate human escalation. Everything below is read from real call_sessions rows written as
    calls actually happen — no hardcoded numbers. Auto-refreshes every 15s.
</p>

<form class="filters" method="get">
    <label>Channel <select name="channel" onchange="this.form.submit()">{channel_options}</select></label>
    <label>Language <select name="language" onchange="this.form.submit()">{language_options}</select></label>
    <label>Range <select name="days" onchange="this.form.submit()">{days_options}</select></label>
    <noscript><button type="submit">Apply</button></noscript>
</form>

<div class="cards">
    {_stat_card("Total calls", str(stats["total_calls"]))}
    {_stat_card("Successful", str(stats["successful_calls"]), "#27ae60")}
    {_stat_card("Failed", str(stats["failed_calls"]), "#c0392b")}
    {_stat_card("Success rate", f"{stats['success_rate']}%", "#2980b9")}
    {_stat_card("Avg. response latency", avg_latency, "#8e44ad")}
</div>

<div class="two-col">
    <div>
        <h2>Calls per day</h2>
        {_daily_chart(stats["daily"])}
    </div>
    <div>
        <h2>Failure breakdown</h2>
        {_bar_chart(stats["failure_breakdown"], _FAILURE_COLOR, "No failed calls in this range.")}
        <h2>Outcome detail</h2>
        {_bar_chart(stats["track_breakdown"], {}, "No calls in this range.")}
    </div>
</div>

<h2>Recent calls</h2>
<table>
<thead><tr>
    <th>Started</th><th>Duration</th><th>Channel</th><th>Language</th>
    <th>Outcome</th><th>Detail</th><th>Latency</th>
</tr></thead>
<tbody>
{table_rows}
</tbody>
</table>

<script>setTimeout(() => location.reload(), 15000);</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter default logging
        print(f"[call-dashboard] {self.address_string()} - {fmt % args}")

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != "/":
            self.send_response(404)
            self.end_headers()
            return
        body = _page_html(_parse_query(self.path)).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    db.init_db()
    server = HTTPServer(("localhost", args.port), Handler)
    print(f"Call outcome dashboard at http://localhost:{args.port}")
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()


if __name__ == "__main__":
    main()
