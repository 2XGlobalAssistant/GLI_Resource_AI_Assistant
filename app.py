"""
app.py — Flask web server for the 2X GLI Assistant
---------------------------------------------------
This is the entry point for the application. It receives questions from the
chat interface (index.html), passes them to the AI pipeline (rag.py), and
returns answers back to the browser.

When Render.com starts the app, it runs this file via gunicorn (see render.yaml).
You should not normally need to edit this file.
"""

import os
import time
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

from rag import load_index, rag_query       # The AI pipeline that answers questions
from logging_utils import log_query, log_feedback  # Saves queries and feedback to logs

# Load environment variables from the .env file (e.g. OPENAI_API_KEY).
# Must happen before anything that needs those variables.
load_dotenv()

app = Flask(__name__)

# Work out where the search index lives relative to this file.
# The index (index.pkl) is the pre-built vector database of all the GLI documents.
PROJECT_ROOT = Path(__file__).resolve().parent
INDEX_PATH = PROJECT_ROOT / "data" / "index.pkl"

# If the index file is missing, stop immediately with a helpful error.
# Fix: run update_library.py to rebuild it.
if not INDEX_PATH.exists():
    raise FileNotFoundError(
        f"Missing {INDEX_PATH}. Run: python update_library.py and choose option 1."
    )

# Load the index into memory once at startup — not on every request.
# This makes responses much faster.
INDEX = load_index(INDEX_PATH)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _normalize_history(raw_history) -> list | None:
    """
    Converts conversation history into a consistent format before passing it
    to the AI. The frontend can send history in two different shapes, so this
    function accepts both and outputs the same format either way.

    Format A — what the chat UI sends (question/answer pairs):
        [{"question": "...", "answer": "..."}, ...]

    Format B — standard message format (role/content):
        [{"role": "user"|"assistant", "content": "..."}, ...]

    Returns None if history is missing, empty, or unreadable.
    Only the last 12 turns are kept to avoid sending too much text to the AI.
    """
    if not isinstance(raw_history, list) or not raw_history:
        return None

    first = raw_history[0]
    if not isinstance(first, dict):
        return None

    # Format A: question/answer pairs → convert to role/content
    if "question" in first or "answer" in first:
        normalized = []
        for item in raw_history[-12:]:
            q = (item.get("question") or "").strip()
            a = (item.get("answer")   or "").strip()
            if q:
                normalized.append({"role": "user",      "content": q})
            if a:
                normalized.append({"role": "assistant", "content": a})
        return normalized or None

    # Format B: already in role/content format — just validate and pass through
    valid = [
        m for m in raw_history[-12:]
        if isinstance(m, dict)
        and (m.get("role") or "").strip().lower() in {"user", "assistant"}
        and (m.get("content") or "").strip()
    ]
    return valid or None


# ---------------------------------------------------------------------------
# URL routes — these define what the server responds to
# ---------------------------------------------------------------------------

@app.get("/")
def home():
    # Serves the main chat interface (templates/index.html)
    return render_template("index.html")


@app.get("/health")
def health():
    # Simple health check endpoint. Render uses this to confirm the app is running.
    # Visit /health in a browser and you should see {"ok": true}.
    return {"ok": True}


@app.post("/chat")
def chat():
    """
    Main endpoint: receives a question from the chat UI, runs it through the AI
    pipeline, and returns an answer with sources.

    The frontend sends a JSON body like:
        {
          "question": "What are the 2X thresholds for Kenya?",
          "expertise": "general",
          "institution_type": "Fund Manager",
          "history": [...]   <- previous turns in the conversation
        }

    Returns a JSON object with "answer", "sources", and "meta" fields.
    """
    payload  = request.get_json(force=True) or {}
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Missing 'question'"}), 400

    # k = how many document chunks to retrieve before generating an answer.
    # Higher k = more context but slower. Default of 12 works well for most questions.
    k = int(payload.get("k") or 12)

    # Extract optional context about the user — used to tailor the answer.
    # e.g. "expert" users get more technical language; "Fund Manager" gets fund-focused answers.
    response_mode    = payload.get("response_mode") or payload.get("mode") or None
    expertise        = payload.get("expertise")     or payload.get("level") or "general"
    audience         = payload.get("audience")      or None
    institution_type = payload.get("institution_type") or payload.get("institution") or None
    jurisdiction     = payload.get("jurisdiction")  or payload.get("geography") or None
    asset_class      = payload.get("asset_class")   or payload.get("instrument") or None
    history          = _normalize_history(payload.get("history"))

    # Record when we started so we can log how long the answer took.
    t_start = time.monotonic()

    try:
        # Hand off to the AI pipeline in rag.py — this is where the actual work happens.
        result = rag_query(
            question=question,
            index=INDEX,
            k=k,
            response_mode=response_mode,
            expertise=expertise,
            audience=audience,
            institution_type=institution_type,
            jurisdiction=jurisdiction,
            asset_class=asset_class,
            history=history,
        )

        response_ms = int((time.monotonic() - t_start) * 1000)
        meta        = result.get("meta", {})

        # Log the query in the background. This runs without blocking the response
        # so the user does not have to wait for the log to finish writing.
        # If logging fails for any reason, we swallow the error — it must never
        # prevent the user from getting their answer.
        try:
            log_query(
                question=question,
                answer=result.get("answer", ""),
                route=meta.get("route"),
                coverage=meta.get("coverage"),
                traffic_light=meta.get("traffic_light"),
                expertise=expertise,
                institution_type=institution_type,
                response_ms=response_ms,
            )
        except Exception:
            pass

        return jsonify(result)

    except Exception as e:
        app.logger.exception("Error in /chat")
        return jsonify({"error": "Server error", "detail": str(e)}), 500


@app.post("/ask-batch")
def ask_batch():
    """
    Batch endpoint: runs multiple questions in one request and returns all answers.
    Used for testing the assistant against a list of questions at once.

    Expects a JSON body like:
        {
          "questions": ["Question 1", "Question 2", ...],
          "expertise": "general"
        }

    Returns {"results": [...]} where each item matches the /chat response format.
    """
    payload   = request.get_json(force=True) or {}
    questions = payload.get("questions") or []

    if not isinstance(questions, list) or not questions:
        return jsonify({"error": "Missing 'questions' (list)"}), 400

    k = int(payload.get("k") or 12)

    # Same optional context fields as /chat
    response_mode    = payload.get("response_mode") or payload.get("mode") or None
    expertise        = payload.get("expertise") or "general"
    audience         = payload.get("audience")  or None
    institution_type = payload.get("institution_type") or payload.get("institution") or None
    jurisdiction     = payload.get("jurisdiction") or payload.get("geography") or None
    asset_class      = payload.get("asset_class")  or payload.get("instrument") or None
    history          = _normalize_history(payload.get("history"))

    results = []

    for q in questions:
        question = (q or "").strip()

        # Skip blank entries rather than crashing
        if not question:
            results.append({
                "answer":    "Please enter a question.",
                "sources":   [],
                "retrieved": [],
                "meta":      {"coverage": "none"},
            })
            continue

        t_start = time.monotonic()

        result = rag_query(
            question=question,
            index=INDEX,
            k=k,
            response_mode=response_mode,
            expertise=expertise,
            audience=audience,
            institution_type=institution_type,
            jurisdiction=jurisdiction,
            asset_class=asset_class,
            history=history,
        )

        response_ms = int((time.monotonic() - t_start) * 1000)
        meta        = result.get("meta", {})

        # Log each question separately, same as /chat
        try:
            log_query(
                question=question,
                answer=result.get("answer", ""),
                route=meta.get("route"),
                coverage=meta.get("coverage"),
                traffic_light=meta.get("traffic_light"),
                expertise=expertise,
                institution_type=institution_type,
                response_ms=response_ms,
            )
        except Exception:
            pass

        # Pull the answer out of whichever field the pipeline used
        answer = (
            result.get("answer")
            or result.get("response")
            or result.get("final")
            or "Sorry, I was not able to generate a response."
        )

        results.append({
            "answer":    answer,
            "sources":   result.get("sources",   []),
            "retrieved": result.get("retrieved", []),
            "meta":      meta,
        })

    return jsonify({"results": results})


@app.post("/beta-feedback")
def beta_feedback():
    """
    Receives feedback submitted through the feedback box in the chat UI.
    Saves it via logging_utils (which writes to the Render log stream and
    optionally posts to Google Forms).

    Always returns HTTP 200, even if saving fails — the user should always
    see the thank-you message regardless of any backend issues.
    """
    payload  = request.get_json(force=True) or {}
    feedback = (payload.get("feedback") or "").strip()

    if not feedback:
        return jsonify({"error": "Missing 'feedback'"}), 400

    try:
        log_feedback(
            feedback=feedback,
            page=payload.get("page") or "",
            timestamp=payload.get("timestamp") or None,
        )
    except Exception as e:
        app.logger.warning(f"Feedback logging failed: {e}")
        # Return 200 anyway — logging errors should never be visible to the user

    return jsonify({"ok": True})


if __name__ == "__main__":
    # This block only runs when you start the app directly with `python app.py`.
    # On Render, gunicorn starts the app instead (see render.yaml).
    port = int(os.getenv("FLASK_RUN_PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
