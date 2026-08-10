"""Shared helpers for Mnemosyne Codex hook scripts.

Design constraints (Task 8 brief + round-1 review):
  - stdlib JSON only; no third-party deps in the hook path
  - persistent cross-session memory via one deterministic opaque memory scope
    derived from actor + project (NOT the ephemeral Codex session_id)
  - stable event IDs (deterministic from scope+host_turn_id+role)
  - bounded recall context (SessionStart <=6/800, UserPromptSubmit <=8/1200)
  - honest, content-free visible failures (distinct per condition), fail-open
  - 0600 transport-only spool; never searchable, ack-deleted only
  - bounded spool: idempotent event IDs, finite capacity, terminal retry state,
    no silent deletion of corrupt rows, additive schema
  - SessionEnd returns before the 3s Codex ceiling
  - PLUGIN_DATA for default writable plugin state; no repo-root import trick
  - never parse transcripts
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import sqlite3
import sys
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Installed-plugin discovery: no repo-root sys.path trick.
#
# Codex runs hooks by path; Python only puts the hook's own directory on
# sys.path. The mnemosyne package must be importable as an installed package
# (pip install -e . from the repo, or pip install mnemosyne). We deliberately
# do NOT inject the source repository root, so the installed plugin behaves
# identically whether the source tree is present or not.
#
# For the test-time path (hooks imported by tests under the source tree), the
# mnemosyne package is already importable because tests run from a repo where
# it is installed/editable.
# ---------------------------------------------------------------------------

PRODUCER = "codex"

# Spool bounded-capacity and retry policy.
_SPOOL_MAX_ROWS = 32
_SPOOL_MAX_ATTEMPTS = 8


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _content_hash(content: str) -> str:
    """SHA-256 of the content (matches Mnemosyne IngestEvent contract)."""
    return hashlib.sha256(content.encode()).hexdigest()


# ---------------------------------------------------------------------------
# stdin / stdout helpers
# ---------------------------------------------------------------------------


def read_stdin() -> Dict[str, Any]:
    """Read the Codex hook JSON payload from stdin.

    Returns {} for empty/non-object input (fail-open). A valid JSON value
    that is not an object (string, number, array, bool) also returns {},
    so the hook emits a harmless JSON response instead of a traceback.
    """
    try:
        raw = sys.stdin.read()
    except OSError:
        return {}
    if not raw or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(parsed, dict):
        # Valid JSON but not an object — fail open without a traceback.
        return {}
    return parsed


def emit(payload: Dict[str, Any]) -> None:
    """Write one JSON object to stdout and flush."""
    sys.stdout.write(json.dumps(payload, default=str) + "\n")
    sys.stdout.flush()


def emit_context(hook_event_name: str, additional_context: str) -> None:
    """Emit additionalContext for injection into the Codex turn.

    Preserves an existing systemMessage by coexisting in the same object:
    documented hook output shape supports both systemMessage and
    hookSpecificOutput simultaneously.
    """
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
# Config: actor, project, memory scope, paths
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
    """Opaque project id. Never leaks the raw cwd/path: returns a short hash."""
    env_val = os.environ.get("MNEMOSYNE_CODEX_PROJECT_ID") or os.environ.get(
        "CODEX_PROJECT_ID"
    )
    if env_val:
        return env_val
    if cwd:
        return "cwd-" + hashlib.sha256(cwd.encode()).hexdigest()[:12]
    return "codex-project"


def memory_scope(actor: str, project: str) -> str:
    """One deterministic opaque memory scope per actor + project.

    This is the single key used as Mnemosyne(session_id=...) for BOTH native
    ingest (IngestEvent.session_id) and bounded recall, so memories persist
    across Codex sessions for the same actor+project. Different actor OR
    project yields a different scope => no cross-isolation recall.

    The scope is opaque (never widens to global/shared and never leaks raw
    actor/project/path values).
    """
    material = f"{actor}|{project}".encode()
    return "mem-" + hashlib.sha256(material).hexdigest()[:16]


def _plugin_data_dir() -> str:
    """Default writable plugin state dir from PLUGIN_DATA (Codex extension).

    Falls back to a mnemosyne-local data dir. Never uses $HOME directly.
    """
    for name in ("PLUGIN_DATA", "CLAUDE_PLUGIN_DATA"):
        v = os.environ.get(name)
        if v:
            return v
    env_data = os.environ.get("MNEMOSYNE_DATA_DIR")
    if env_data:
        return env_data
    # Last-resort default: a plugin-local subdir (not $HOME).
    return os.path.join("/tmp", "codex-mnemosyne-data")


def spool_path() -> str:
    """Resolve the transport spool path under PLUGIN_DATA."""
    env_val = os.environ.get("MNEMOSYNE_CODEX_SPOOL_PATH")
    if env_val:
        return env_val
    return os.path.join(_plugin_data_dir(), "codex-spool.db")


# ---------------------------------------------------------------------------
# Identity / event-id helpers
# ---------------------------------------------------------------------------


def stable_event_id(scope: str, turn_id: str, role: str) -> str:
    """Deterministic event id from memory scope + host turn id + role.

    Same inputs always produce the same event_id, so replaying a hook for the
    same logical event is idempotent (native ingest deduplicates by
    event_id, and the spool deduplicates by event_id).
    """
    material = f"{scope}|{turn_id}|{role}".encode()
    return "cx-" + hashlib.sha256(material).hexdigest()[:24]


def fallback_turn_id(scope: str, content: str) -> str:
    """Deterministic content fallback turn id, used only when host turn_id is
    absent. ponytail: ceiling = collision across identical content in the same
    scope; upgrade path = always rely on host turn_id when Codex provides it.
    """
    material = f"{scope}|{content}".encode()
    return "turn-" + hashlib.sha256(material).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Bounded recall context formatting
# ---------------------------------------------------------------------------


def format_recall_context(
    results: list, retrieval_mode: str = "", degradation: Optional[list] = None
) -> str:
    """Render bounded recall results as a compact, non-sensitive context block.

    Each result is one line. No raw memory ids, payload hashes, scope, or
    internal metadata leak.
    """
    if not results:
        return ""
    lines = ['<mnemosyne-recall source="codex-hook" format="digest">']
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
        # Hard ceiling as defence in depth (recall_bounded already bounds).
        if len(content) > 500:
            content = content[:497] + "..."
        lines.append(f"-{tag} {content}")
    lines.append("</mnemosyne-recall>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Native ingest / recall wrappers (fail-open, returns structured outcome)
# ---------------------------------------------------------------------------


class Outcome:
    """Structured outcome of a native operation.

    error_message is for internal logic only — never emitted to the user.
    """

    def __init__(self, ok: bool, error_code: str = "", error_message: str = ""):
        self.ok = ok
        self.error_code = error_code
        self.error_message = error_message

    def __bool__(self) -> bool:
        return self.ok


def _import_mnemosyne() -> Tuple[bool, str]:
    """Probe whether the mnemosyne package is importable.

    Returns (importable, error_code). error_code is empty on success.
    """
    try:
        import mnemosyne  # noqa: F401
        from mnemosyne.core.memory import Mnemosyne  # noqa: F401
        from mnemosyne.core.inhale import IngestEvent  # noqa: F401
        from mnemosyne.core.recall_bounded import RecallPolicy  # noqa: F401
    except Exception:
        return False, "package_absent"
    return True, ""


def native_ingest(event_dict: Dict[str, Any]) -> Outcome:
    """Call mnemosyne remember_event with an IngestEvent built from event_dict.

    The memory scope (derived from actor+project) is used as the
    Mnemosyne.session_id so memories persist across Codex sessions and are
    isolated per actor+project. The ephemeral Codex session_id is kept only
    in event metadata for non-recall provenance.

    Returns Outcome(ok=True) on stored/duplicate, Outcome(ok=False, ...) on
    conflict/rejected/exception/package-absent. Never raises.
    """
    try:
        from mnemosyne.core.memory import Mnemosyne
        from mnemosyne.core.inhale import IngestEvent
    except Exception:
        return Outcome(False, "package_absent", "")

    try:
        content = event_dict["content"]
        scope = event_dict["scope"]
        ev = IngestEvent(
            event_id=event_dict["event_id"],
            producer=event_dict.get("producer", PRODUCER),
            actor_id=event_dict["actor_id"],
            project_id=event_dict["project_id"],
            session_id=scope,
            turn_id=event_dict["turn_id"],
            role=event_dict["role"],
            content=content,
            content_hash=_content_hash(content),
            occurred_at=event_dict.get("occurred_at", _now_iso()),
            metadata=event_dict.get("metadata"),
        )
        m = Mnemosyne(
            session_id=scope,
            author_id=ev.actor_id,
            author_type=actor_type(),
            channel_id=ev.project_id,
        )
        receipt = m.remember_event(ev)
        status = getattr(receipt, "status", "")
        if status in ("stored", "duplicate"):
            return Outcome(True)
        return Outcome(False, f"ingest_{status}", "")
    except Exception:
        return Outcome(False, "ingest_exception", "")


def native_recall(
    query: str,
    top_k: int,
    max_tokens: int,
    scope: str,
    actor: str,
    project: str,
) -> Tuple[list, str, list]:
    """Call mnemosyne recall_bounded using the same memory scope as ingest.

    Returns (results, retrieval_mode, degradation). Never raises; on failure
    returns ([], "error", []).
    """
    try:
        from mnemosyne.core.memory import Mnemosyne
        from mnemosyne.core.recall_bounded import RecallPolicy
    except Exception:
        return [], "error", []

    try:
        m = Mnemosyne(
            session_id=scope,
            author_id=actor,
            author_type=actor_type(),
            channel_id=project,
        )
        policy = RecallPolicy(
            top_k=top_k,
            max_tokens=max_tokens,
            actor_ids=(actor,),
            project_ids=(project,),
            session_ids=(scope,),
        )
        envelope = m.recall_bounded(query, policy)
        return (
            list(envelope.results),
            getattr(envelope, "retrieval_mode", ""),
            list(getattr(envelope, "degradation_reasons", []) or []),
        )
    except Exception:
        return [], "error", []


# ---------------------------------------------------------------------------
# Honest, content-free visible messages (one per condition)
# ---------------------------------------------------------------------------

_MSG_PACKAGE_ABSENT = (
    "Mnemosyne memory is not installed. "
    "Install the mnemosyne Python package (pip install mnemosyne) to enable "
    "persistent memory."
)
_MSG_RECALL_UNAVAILABLE = (
    "Mnemosyne recall is temporarily unavailable. This turn will proceed "
    "without recalled context."
)
_MSG_INGEST_FAILED_QUEUED = (
    "Mnemosyne ingest was unavailable; the event was durably queued for later delivery."
)
_MSG_INGEST_FAILED_NOT_QUEUED = (
    "Mnemosyne ingest was unavailable and the event could not be queued. "
    "The event was not persisted."
)
_MSG_QUEUE_FULL = (
    "Mnemosyne memory queue is full; the event was not persisted to avoid "
    "unbounded growth."
)
_MSG_SESSION_END_PENDING = "Mnemosyne session end: some events remain pending delivery."
_MSG_SESSION_END_DELIVERED = "Mnemosyne session end complete."


def message_package_absent() -> str:
    return _MSG_PACKAGE_ABSENT


def message_recall_unavailable() -> str:
    return _MSG_RECALL_UNAVAILABLE


def message_ingest_queued() -> str:
    return _MSG_INGEST_FAILED_QUEUED


def message_ingest_not_queued() -> str:
    return _MSG_INGEST_FAILED_NOT_QUEUED


def message_queue_full() -> str:
    return _MSG_QUEUE_FULL


def message_session_end_pending() -> str:
    return _MSG_SESSION_END_PENDING


def message_session_end_delivered() -> str:
    return _MSG_SESSION_END_DELIVERED


# ---------------------------------------------------------------------------
# Transport-only spool (0600, never searchable, ack-deleted, bounded)
# ---------------------------------------------------------------------------

# Additive schema: the prior local spool had the columns below without
# `terminal`; we add `terminal` and `scope`/`role` columns for the bounded
# retry policy WITHOUT breaking an existing prior-local spool db.
_SPOOL_SCHEMA = """
CREATE TABLE IF NOT EXISTS spooled_events (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    payload_json TEXT NOT NULL,
    spooled_at TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    terminal INTEGER NOT NULL DEFAULT 0
);
"""


def _ensure_spool(path: str) -> None:
    """Create the spool db with mode 0600 if it does not exist.

    Additive migration: if a prior schema lacks the `terminal` column, add it
    without dropping data.
    """
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, mode=0o700, exist_ok=True)
    created = not os.path.exists(path)
    if created:
        fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
        os.close(fd)
    else:
        os.chmod(path, 0o600)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_SPOOL_SCHEMA)
        # Additive migration for prior local spool schema.
        cols = {
            r[1] for r in conn.execute("PRAGMA table_info(spooled_events)").fetchall()
        }
        if "terminal" not in cols:
            conn.execute(
                "ALTER TABLE spooled_events ADD COLUMN terminal INTEGER NOT NULL DEFAULT 0"
            )
        conn.commit()
    finally:
        conn.close()


def _spool_is_full(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT COUNT(*) FROM spooled_events").fetchone()
    return int(row[0]) >= _SPOOL_MAX_ROWS


def spool_put(path: str, event_dict: Dict[str, Any]) -> str:
    """Persist one event to the transport spool (idempotent by event_id).

    Returns one of: "stored" (new row written), "duplicate" (event_id already
    present), "full" (capacity reached, nothing written), "error" (write
    failed). Never raises.
    """
    try:
        _ensure_spool(path)
        conn = sqlite3.connect(path)
        try:
            eid = event_dict.get("event_id", "")
            existing = conn.execute(
                "SELECT 1 FROM spooled_events WHERE event_id = ?", (eid,)
            ).fetchone()
            if existing is not None:
                return "duplicate"
            if _spool_is_full(conn):
                return "full"
            conn.execute(
                "INSERT INTO spooled_events "
                "(event_id, payload_json, spooled_at, attempts, terminal) "
                "VALUES (?, ?, ?, 0, 0)",
                (eid, json.dumps(event_dict, default=str), _now_iso()),
            )
            conn.commit()
            return "stored"
        finally:
            conn.close()
    except Exception:
        return "error"


def spool_count(path: str) -> int:
    """Return the number of spooled events (for tests)."""
    if not os.path.exists(path):
        return 0
    try:
        conn = sqlite3.connect(path)
        try:
            row = conn.execute("SELECT COUNT(*) FROM spooled_events").fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()
    except Exception:
        return 0


def _deadline_remaining(start: float, budget_s: float) -> float:
    return max(0.0, budget_s - (datetime.datetime.now().timestamp() - start))


def flush_spool(
    path: str,
    env: Optional[Dict[str, Any]] = None,
    budget_s: float = 2.5,
) -> Tuple[int, int, int]:
    """Attempt to deliver spooled events via native ingest within a deadline.

    Deletes a row only after a successful ack (stored/duplicate). Terminal
    rows (attempts >= _SPOOL_MAX_ATTEMPTS) are retained, not retried. Corrupt
    rows are retained (never silently deleted).

    Returns (flushed, retained_pending, retained_terminal). Never raises.
    Guarantees return before `budget_s` seconds elapse even if ingest hangs,
    by checking the deadline between each row and never blocking on a single
    ingest beyond the remaining budget.
    """
    if env:
        for k, v in env.items():
            os.environ[k] = str(v)
    if not os.path.exists(path):
        return 0, 0, 0
    _ensure_spool(path)
    start = datetime.datetime.now().timestamp()
    flushed = 0
    retained_pending = 0
    retained_terminal = 0
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute(
            "SELECT rowid, event_id, payload_json, attempts, terminal "
            "FROM spooled_events WHERE terminal = 0 ORDER BY rowid"
        ).fetchall()
        for rowid, event_id, payload_json, attempts, terminal in rows:
            if _deadline_remaining(start, budget_s) <= 0:
                retained_pending += 1
                continue
            try:
                event_dict = json.loads(payload_json)
            except (json.JSONDecodeError, TypeError, ValueError):
                # Corrupt row — retain, never silently delete.
                retained_pending += 1
                continue
            outcome = native_ingest(event_dict)
            if outcome.ok:
                conn.execute("DELETE FROM spooled_events WHERE rowid = ?", (rowid,))
                conn.commit()
                flushed += 1
            else:
                new_attempts = attempts + 1
                is_terminal = 1 if new_attempts >= _SPOOL_MAX_ATTEMPTS else 0
                conn.execute(
                    "UPDATE spooled_events SET attempts = ?, terminal = ? WHERE rowid = ?",
                    (new_attempts, is_terminal, rowid),
                )
                conn.commit()
                if is_terminal:
                    retained_terminal += 1
                else:
                    retained_pending += 1
        # Count pre-existing terminal rows (not selected above).
        trow = conn.execute(
            "SELECT COUNT(*) FROM spooled_events WHERE terminal = 1"
        ).fetchone()
        retained_terminal += int(trow[0]) if trow else 0
    except Exception:
        pass
    finally:
        conn.close()
    return flushed, retained_pending, retained_terminal


def flush_spool_bounded(path: str, budget_s: float = 2.0) -> Tuple[int, int, int]:
    """Run flush_spool under a hard wall-clock deadline.

    Uses signal.SIGALRM (Unix) to guarantee return within budget_s seconds even
    if a single native ingest call blocks/hangs. On timeout, returns immediately
    with whatever was flushed so far; unprocessed rows are retained. On
    platforms without SIGALRM, falls back to the between-row deadline in
    flush_spool (which still bounds fast-per-row cases).

    ponytail: ceiling = a single ingest call blocking longer than budget;
    SIGALRM interrupts it. Upgrade path = run flush in a child process with
    subprocess timeout if SIGALRM is ever unavailable.
    """
    import signal

    result = {"flushed": 0, "pending": 0, "terminal": 0}

    def _alarm_handler(signum, frame):
        raise TimeoutError("flush deadline exceeded")

    old_handler = None
    had_alarm = hasattr(signal, "SIGALRM")
    if had_alarm:
        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        # Budget in whole seconds, minimum 1.
        signal.setitimer(signal.ITIMER_REAL, max(1.0, budget_s))
    try:
        f, p_, t_ = flush_spool(path, budget_s=budget_s)
        result["flushed"] = f
        result["pending"] = p_
        result["terminal"] = t_
    except TimeoutError:
        # Deadline hit mid-flush; whatever was acked is already committed per-row.
        pass
    except Exception:
        pass
    finally:
        if had_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
    return result["flushed"], result["pending"], result["terminal"]


# ---------------------------------------------------------------------------
# Convenience: ingest with spool fallback + honest messaging
# ---------------------------------------------------------------------------


def ingest_or_spool(event_dict: Dict[str, Any]) -> Tuple[Outcome, str]:
    """Try native ingest; on failure, spool for later delivery.

    Returns (Outcome, spool_status). spool_status is one of:
    "stored", "duplicate", "full", "error", "" (ingest succeeded, not spooled).

    If MNEMOSYNE_CODEX_FORCE_SPOOL=1, always attempt the spool path (for
    testing the failure path without breaking the native DB).
    """
    force = os.environ.get("MNEMOSYNE_CODEX_FORCE_SPOOL", "") == "1"
    if not force:
        outcome = native_ingest(event_dict)
        if outcome.ok:
            return outcome, ""
    status = spool_put(spool_path(), event_dict)
    return Outcome(False, "spooled", ""), status
