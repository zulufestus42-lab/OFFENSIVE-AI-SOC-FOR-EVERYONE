"""
Forward each /v1/scan result to the Offense AI dashboard.

Your main.py route is synchronous, so this helper is synchronous too. It is called from a
FastAPI BackgroundTask, which runs AFTER the response is already sent to the caller, so it
never slows down a scan and a dashboard outage can never break /v1/scan.

Setup on Render (FastAPI service → Environment):
    DASHBOARD_INGEST_URL   = https://<your-dashboard-backend>/v1/events
    DASHBOARD_INGEST_TOKEN = <same value as INGEST_TOKEN on the dashboard backend>

If DASHBOARD_INGEST_URL is unset, the push is skipped silently, so this is safe to deploy
before the dashboard backend exists.
"""

import os

import httpx

DASHBOARD_INGEST_URL = os.environ.get("DASHBOARD_INGEST_URL")
DASHBOARD_INGEST_TOKEN = os.environ.get("DASHBOARD_INGEST_TOKEN", "")


def push_to_dashboard(attack_log: dict, scanned_text: str | None = None) -> None:
    """Send one attack_log (the exact dict /v1/scan returns) to the dashboard."""
    if not DASHBOARD_INGEST_URL:
        return
    event = dict(attack_log)
    if scanned_text:
        # Shown in the dashboard log's "Signature" column. Trimmed so a huge payload
        # can't bloat the request; it is always rendered as text, never executed.
        event["INPUT"] = str(scanned_text)[:200]
    try:
        httpx.post(
            DASHBOARD_INGEST_URL,
            json=event,
            headers={"Authorization": f"Bearer {DASHBOARD_INGEST_TOKEN}"},
            timeout=3.0,
        )
    except httpx.HTTPError as exc:
        # Never raise: the scan has already been answered and logged to SQLite.
        print(f"[dashboard] push failed: {exc}")
