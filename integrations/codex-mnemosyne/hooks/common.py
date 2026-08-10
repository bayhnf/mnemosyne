"""Shared helpers for Mnemosyne Codex hook scripts.

Design constraints (from Task 8 brief):
  - stdlib JSON only; no third-party deps in the hook path
  - stable event IDs (deterministic from session+turn+role)
  - bounded recall context (SessionStart <=6/800, UserPromptSubmit <=8/1200)
  - visible structured failures (non-sensitive systemMessage), fail-open (exit 0)
  - 0600 transport-only spool; never searchable, deleted after ack
  - never parse transcripts

The hook scripts read JSON from stdin, call these helpers, and write one JSON
object to stdout.  Mnemosyne remains the sole persistent memory provider; this
module is a thin adapter.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import datetime
from typing import Any, Dict, Optional

# Ensure the repository root (where the ``mnemosyne`` package lives) is on
# ``sys.path`` when a hook runs as a standalone script.  Python only adds the
# hook's own directory to ``sys.path[0]``, so ``import mnemosyne`` would fail
# without this.  The hooks dir is ``integrations/codex-mnemosyne/hooks``;
# the repo root is three levels up.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Identity / event-id helpers
# ---------------------------------------------------------------------------

PRODUCER = "codex"


def stable_event_id(session_id: str, turn_id: str, role: str) -> str:
    """Deterministic event id from session+turn+role.

    Same inputs always produce the same event_id, so replaying a hook for the
    same logical event is idempotent (native ingest deduplicates by event_id).
    """
    material = f"{session_id}|{turn_id}|{role}".encode()
    return "cx-" + hashlib.sha256(material).hexdigest()[:24]


def stable_turn_id(session_id: str, prompt_or_message: str) -> str:
    """Deterministic turn id from session + content hash."""
    material = f"{session_id}|{prompt_or_message}".encode()
    return "turn-" + hashlib.sha256(material).hexdigest()[:16]


def _content_hash(content: str) -> str:
    """SHA-256 of the content (matches Mnemosyne IngestEvent contract)."""
    return hashlib.sha256(content.encode()).hexdigest()


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# stdin / stdout helpers
# ---------------------------------------------------------------------------


def read_stdin() -> Dict[str, Any]:
    """Read the Codex hook JSON payload from stdin.  Non-fatal on bad input."""
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return {}
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError, OSError):
        return {}


def emit(payload: Dict[str, Any]) -> None:
    """Write one JSON object to stdout and flush."""
    sys.stdout.write(json.dumps(payload, default=str) + "\n")
    sys.stdout.flush()


def emit_context(hook_event_name: str, additional_context: str) -> None:
    """Emit additionalContext for injection into the Codex turn."""
    ctx = additional_context.strip()
    if not ctx:
        emit({})
        return
    emit(
        {
            "hookSpecificOutput": {
                "hookEventName": hook_event_name,
                "additionalContext": ctx,
            }
        }
    )


def emit_system_message(message: str) -> None:
    """Emit a visible system message (non-sensitive warning)."""
    if not message:
        emit({})
        return
    emit({"systemMessage": message})


def emit_noop() -> None:
    emit({})


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _env_or(default: str, *names: str) -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


def actor_id() -> str:
    return _env_or("codex-actor", "MNEMOSYNE_CODEX_ACTOR_ID", "CODEX_ACTOR_ID")


def actor_type() -> str:
    return _env_or("human", "MNEMOSYNE_CODEX_ACTOR_TYPE", "CODEX_ACTOR_TYPE")


def project_id(cwd: Optional[str] = None) -> str:
    env_val = os.environ.get("MNEMOSYNE_CODEX_PROJECT_ID") or os.environ.get(
        "CODEX_PROJECT_ID"
    )
    if env_val:
        return env_val
    if cwd:
        return hashlib.sha256(cwd.encode()).hexdigest()[:12]
    return "codex-project"


def spool_path() -> str:
    """Resolve the transport spool path (default under the data dir)."""
    env_val = os.environ.get("MNEMOSYNE_CODEX_SPOOL_PATH")
    if env_val:
        return env_val
    data_dir = os.environ.get("MNEMOSYNE_DATA_DIR") or os.path.join(
        os.path.expanduser("~"), ".hermes", "mnemosyne", "data"
    )
    return os.path.join(data_dir, "codex-spool.db")


# ---------------------------------------------------------------------------
# Bounded recall context formatting
# ---------------------------------------------------------------------------


def format_recall_context(
    results: list, retrieval_mode: str = "", degradation: Optional[list] = None
) -> str:
    """Render bounded recall results as a compact, non-sensitive context block.

    Each result is one line.  The block is wrapped so Codex sees a clear
    boundary.  No raw memory ids, payload hashes, or internal metadata leak.
    """
    if not results:
        return ""
    lines = []
    lines.append('<mnemosyne-recall source="codex-hook" format="digest">')
    for r in results:
        content = str(r.get("content", "")).strip()
        source = str(r.get("source", "")).strip()
        importance = r.get("importance")
        tag_parts = []
        if source:
            tag_parts.append(source)
        if importance is not None:
            try:
                tag_parts.append(f"imp={float(importance):.2f}")
            except (TypeError, ValueError):
                pass
        tag = f" [{', '.join(tag_parts)}]" if tag_parts else ""
        # Truncate very long items; recall_bounded already enforces max_item_tokens
        # but keep a hard ceiling as defence in depth.
        if len(content) > 500:
            content = content[:497] + "..."
        lines.append(f"-{tag} {content}")
    lines.append("</mnemosyne-recall>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Native ingest / recall wrappers (fail-open, returns structured outcome)
# ---------------------------------------------------------------------------


class Outcome:
    """Structured outcome of a native operation.  Non-sensitive diagnostics only."""

    def __init__(self, ok: bool, error_code: str = "", error_message: str = ""):
        self.ok = ok
        self.error_code = error_code
        # error_message is for the spool/log only — never emitted to the user.
        self.error_message = error_message

    def __bool__(self) -> bool:
        return self.ok


def native_ingest(event_dict: Dict[str, Any]) -> Outcome:
    """Call mnemosyne.remember_event with an IngestEvent built from event_dict.

    Returns Outcome(ok=True) on stored/duplicate, Outcome(ok=False, ...) on
    conflict/rejected/exception.  Never raises.
    """
    try:
        from mnemosyne.core.memory import Mnemosyne
        from mnemosyne.core.inhale import IngestEvent
    except Exception as exc:  # pragma: no cover - import failure path
        return Outcome(False, "import_error", str(exc)[:200])

    try:
        content = event_dict["content"]
        ev = IngestEvent(
            event_id=event_dict["event_id"],
            producer=event_dict.get("producer", PRODUCER),
            actor_id=event_dict.get("actor_id", actor_id()),
            project_id=event_dict.get("project_id", project_id()),
            session_id=event_dict["session_id"],
            turn_id=event_dict["turn_id"],
            role=event_dict["role"],
            content=content,
            content_hash=_content_hash(content),
            occurred_at=event_dict.get("occurred_at", _now_iso()),
            metadata=event_dict.get("metadata"),
        )
        m = Mnemosyne(
            session_id=event_dict["session_id"],
            author_id=ev.actor_id,
            author_type=actor_type(),
            channel_id=ev.project_id,
        )
        receipt = m.remember_event(ev)
        status = getattr(receipt, "status", "")
        if status in ("stored", "duplicate"):
            return Outcome(True)
        # conflict / rejected
        return Outcome(False, f"ingest_{status}", f"receipt status={status}")
    except Exception as exc:
        return Outcome(False, "ingest_exception", str(exc)[:200])


def native_recall(
    query: str, top_k: int, max_tokens: int, session_id: str = "codex-recall"
) -> tuple[list, str, list]:
    """Call mnemosyne.recall_bounded.  Returns (results, retrieval_mode, degradation).

    Uses the caller's session_id so session-scoped memories are visible.
    Re-indexes any pending/degraded receipts first so a fresh hook process
    (separate connection from the one that ingested) sees FTS-indexed rows.
    This is idempotent and cheap.  Never raises; on failure returns
    ([], "error", [reason]).
    """
    try:
        from mnemosyne.core.memory import Mnemosyne
        from mnemosyne.core.recall_bounded import RecallPolicy
    except Exception as exc:  # pragma: no cover
        return [], "error", [f"import_error:{str(exc)[:80]}"]

    try:
        m = Mnemosyne(
            session_id=session_id,
            author_id=actor_id(),
            author_type=actor_type(),
            channel_id=project_id(),
        )
        # Best-effort: re-index pending/degraded receipts so this fresh
        # connection sees the latest FTS rows.  Never raises.
        try:
            m.retry_pending_ingest(limit=50)
        except Exception:
            pass
        policy = RecallPolicy(top_k=top_k, max_tokens=max_tokens)
        envelope = m.recall_bounded(query, policy)
        return (
            list(envelope.results),
            getattr(envelope, "retrieval_mode", ""),
            list(getattr(envelope, "degradation_reasons", []) or []),
        )
    except Exception as exc:
        return [], "error", [f"recall_exception:{str(exc)[:80]}"]


# ---------------------------------------------------------------------------
# Transport-only spool (0600, never searchable, ack-deleted)
# ---------------------------------------------------------------------------

_SPOOL_SCHEMA = """
CREATE TABLE IF NOT EXISTS spooled_events (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    spooled_at TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0
);
"""


def _ensure_spool(path: str) -> None:
    """Create the spool db with mode 0600 if it does not exist."""
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, mode=0o700, exist_ok=True)
    # Create/touch with 0600 before any sqlite open so the mode is guaranteed.
    if not os.path.exists(path):
        fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
        os.close(fd)
    else:
        os.chmod(path, 0o600)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_SPOOL_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def spool_event(path: str, event_dict: Dict[str, Any]) -> None:
    """Persist one event to the transport spool (0600). Never raises."""
    try:
        _ensure_spool(path)
        conn = sqlite3.connect(path)
        try:
            conn.execute(
                "INSERT INTO spooled_events (event_id, payload_json, spooled_at, attempts) VALUES (?, ?, ?, 0)",
                (
                    event_dict.get("event_id", ""),
                    json.dumps(event_dict, default=str),
                    _now_iso(),
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        # Last-resort: the spool itself failed.  We cannot do anything more;
        # the hook must still exit 0 (fail-open for Codex).
        pass


def spool_count(path: str) -> int:
    """Return the number of spooled events (for tests)."""
    if not os.path.exists(path):
        return 0
    conn = sqlite3.connect(path)
    try:
        row = conn.execute("SELECT COUNT(*) FROM spooled_events").fetchone()
        return int(row[0]) if row else 0
    finally:
        conn.close()


def flush_spool(path: str, env: Optional[Dict[str, Any]] = None) -> int:
    """Attempt to deliver all spooled events via native ingest.

    Deletes each row after a successful ack (stored/duplicate).  Returns the
    number of rows successfully flushed.  Never raises.
    """
    if env:
        for k, v in env.items():
            os.environ[k] = str(v)
    if not os.path.exists(path):
        return 0
    _ensure_spool(path)
    conn = sqlite3.connect(path)
    flushed = 0
    try:
        rows = conn.execute(
            "SELECT rowid, payload_json FROM spooled_events ORDER BY rowid"
        ).fetchall()
        for rowid, payload_json in rows:
            try:
                event_dict = json.loads(payload_json)
            except (json.JSONDecodeError, TypeError):
                # Corrupt row — remove it so it does not block the queue.
                conn.execute("DELETE FROM spooled_events WHERE rowid = ?", (rowid,))
                conn.commit()
                continue
            outcome = native_ingest(event_dict)
            if outcome.ok:
                conn.execute("DELETE FROM spooled_events WHERE rowid = ?", (rowid,))
                conn.commit()
                flushed += 1
            else:
                conn.execute(
                    "UPDATE spooled_events SET attempts = attempts + 1 WHERE rowid = ?",
                    (rowid,),
                )
                conn.commit()
    finally:
        conn.close()
    return flushed


# ---------------------------------------------------------------------------
# Failure message helpers
# ---------------------------------------------------------------------------

_MEMORY_DOWN_MSG = (
    "Mnemosyne memory is temporarily unavailable. "
    "This turn will proceed without recall; "
    "the event has been safely spooled for later delivery."
)


def memory_down_message() -> str:
    """A visible, non-sensitive warning for when memory is unreachable."""
    return _MEMORY_DOWN_MSG


# ---------------------------------------------------------------------------
# Convenience: ingest with spool fallback
# ---------------------------------------------------------------------------


def ingest_or_spool(event_dict: Dict[str, Any]) -> Outcome:
    """Try native ingest; on failure, spool the event for later delivery.

    If MNEMOSYNE_CODEX_FORCE_SPOOL=1 is set, always spool (for testing the
    failure path without breaking the native DB).
    """
    force = os.environ.get("MNEMOSYNE_CODEX_FORCE_SPOOL", "") == "1"
    if not force:
        outcome = native_ingest(event_dict)
        if outcome.ok:
            return outcome
    # Spool for later delivery.
    spool_event(spool_path(), event_dict)
    return Outcome(False, "spooled", "event spooled for later delivery")
