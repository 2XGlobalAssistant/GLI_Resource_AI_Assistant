"""
logging_utils.py
----------------
Lightweight query + feedback logger for the 2X GLI Assistant.

Tier 1 (always on): structured app.logger.info() lines that appear in
                     Render's log stream.

Tier 2 (opt-in):    POST to Airtable when AIRTABLE_API_KEY and
                     AIRTABLE_BASE_ID are set as environment variables.
                     Enable by adding those vars in your Render dashboard —
                     no code changes needed.

Airtable setup (when ready):
  1. Create a free Airtable account at airtable.com
  2. Create a new Base called e.g. "GLI Assistant Logs"
  3. Add two tables: "Queries" and "Feedback"
  4. In each table, create fields matching the keys in QUERY_FIELDS /
     FEEDBACK_FIELDS below (all Single line text or Number as noted)
  5. Go to airtable.com/create/tokens → create a token with
     data.records:write scope for your base
  6. Add to Render environment:
       AIRTABLE_API_KEY   = your token
       AIRTABLE_BASE_ID   = appXXXXXXXXXXXXXX  (from your base URL)
       AIRTABLE_QUERY_TABLE    = Queries    (or whatever you named it)
       AIRTABLE_FEEDBACK_TABLE = Feedback
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _post_to_airtable(table_env_var: str, fields: Dict[str, Any]) -> None:
    """
    Fire-and-forget POST to Airtable.  Runs in a background thread so it
    never adds latency to the user-facing request.  Silently swallows all
    errors — logging should never break the main app.
    """
    api_key = os.getenv("AIRTABLE_API_KEY", "").strip()
    base_id = os.getenv("AIRTABLE_BASE_ID", "").strip()
    table   = os.getenv(table_env_var, "").strip()

    if not (api_key and base_id and table):
        return  # Airtable not configured — skip silently

    def _send():
        try:
            import urllib.request
            url     = f"https://api.airtable.com/v0/{base_id}/{table}"
            payload = json.dumps({"fields": fields}, ensure_ascii=False).encode()
            req     = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type":  "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5):
                pass
        except Exception as exc:
            logger.warning(f"[logging_utils] Airtable POST failed: {exc}")

    threading.Thread(target=_send, daemon=True).start()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_query(
    question:         str,
    answer:           str,
    route:            Optional[str] = None,
    coverage:         Optional[str] = None,
    traffic_light:    Optional[str] = None,
    expertise:        Optional[str] = None,
    institution_type: Optional[str] = None,
    response_ms:      Optional[int] = None,
) -> None:
    """
    Log a completed query.  Called from app.py after rag_query() returns.
    """
    timestamp = _now_iso()

    # --- Tier 1: Render log stream ---
    # One structured line per query; easy to grep / filter in Render dashboard
    logger.info(
        "[QUERY] ts=%s route=%s coverage=%s confidence=%s expertise=%s "
        "institution=%s ms=%s | q=%s",
        timestamp,
        route          or "—",
        coverage       or "—",
        traffic_light  or "—",
        expertise      or "—",
        institution_type or "—",
        response_ms    if response_ms is not None else "—",
        question[:120],   # truncate very long questions in the log line
    )

    # --- Tier 2: Airtable (fire-and-forget, only if configured) ---
    _post_to_airtable(
        "AIRTABLE_QUERY_TABLE",
        {
            "Timestamp":        timestamp,
            "Question":         question,
            "Answer":           answer[:2000],   # Airtable cell limit
            "Route":            route            or "",
            "Coverage":         coverage         or "",
            "Confidence":       traffic_light    or "",
            "Expertise":        expertise        or "",
            "Institution Type": institution_type or "",
            "Response MS":      response_ms      or 0,
        },
    )


def log_feedback(
    feedback:  str,
    page:      Optional[str] = None,
    timestamp: Optional[str] = None,
) -> None:
    """
    Log a feedback submission.  Called from app.py's /beta-feedback route.
    """
    ts = timestamp or _now_iso()

    # --- Tier 1: Render log stream ---
    logger.info(
        "[FEEDBACK] ts=%s page=%s | %s",
        ts,
        page or "—",
        feedback[:200],
    )

    # --- Tier 2: Airtable ---
    _post_to_airtable(
        "AIRTABLE_FEEDBACK_TABLE",
        {
            "Timestamp": ts,
            "Page":      page     or "",
            "Feedback":  feedback,
        },
    )
