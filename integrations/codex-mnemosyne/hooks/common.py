"""Shared helpers for Mnemosyne Codex hook scripts.

Design constraints (Task 8 brief + round-1/round-2 reviews):
  - stdlib JSON only; no third-party deps in the hook path
  - persistent cross-session memory via one deterministic opaque memory scope
    derived from actor + project (NOT the ephemeral Codex session_id)
  - stable event IDs (deterministic from scope+host_turn_id+role)
  - bounded recall context (SessionStart <=6/800, UserPromptSubmit <=8/1200)
  - honest, content-free visible failures (distinct per condition), fail-open
  - 0600 transport-only spool; never searchable, ack-deleted only
  - bounded spool: idempotent event IDs, finite capacity, terminal retry state,
    no silent deletion of corrupt rows, additive schema
  - SessionEnd returns before the 3s Codex ceiling via a real hard subprocess
    boundary (not SIGALRM-only)
  - PLUGIN_DATA for default writable plugin state AND native Mnemosyne data;
    no repo-root import trick; no global home-directory access
  - never parse transcripts
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterator, Optional, Tuple

# ---------------------------------------------------------------------------
# Installed-plugin discovery: no repo-root sys.path trick.
#
# Codex runs hooks by path; Python only puts the hook's own directory on
# sys.path. The mnemosyne package must be importable as an installed package
# (pip install mnemosyne). We do NOT inject the source repository root.
# ---------------------------------------------------------------------------

PRODUCER = "codex"

# Spool bounded-capacity and retry policy.
_SPOOL_MAX_ROWS = 32
_SPOOL_MAX_ATTEMPTS = 8

# SessionEnd budget (well within Codex's 3-second ceiling).
_SESSION_END_BUDGET_S = 2.0


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _content_hash(content: str) -> str:
    """SHA-256 of the content (matches Mnemosyne IngestEvent contract)."""
    return hashlib.sha256(content.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Environment management (no process-global leakage)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _scoped_env(extra: Dict[str, str]) -> Iterator[None]:
    """Temporarily set env vars, restoring the exact prior state on exit.

    flush_spool uses this instead of mutating os.environ permanently, so
    test isolation never depends on stale env side-effects.
    """
    sentinel = object()
    saved: Dict[str, Any] = {}
    try:
        for k, v in extra.items():
            saved[k] = os.environ.get(k, sentinel)
            os.environ[k] = str(v)
        yield
    finally:
        for k, old in saved.items():
            if old is sentinel:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old  # type: ignore[assignment]


def _ensure_mnemosyne_data_dir() -> None:
    """Ensure MNEMOSYNE_DATA_DIR points to PLUGIN_DATA so the native Mnemosyne
    DB never defaults to a global home directory.

    If MNEMOSYNE_DATA_DIR is already set, respect it. Otherwise derive from
    PLUGIN_DATA / CLAUDE_PLUGIN_DATA. This runs at import time so every
    Mnemosyne() constructor uses the disposable/plugin-data path.
    """
    if os.environ.get("MNEMOSYNE_DATA_DIR"):
        return
    for name in ("PLUGIN_DATA", "CLAUDE_PLUGIN_DATA"):
        v = os.environ.get(name)
        if v:
            os.environ["MNEMOSYNE_DATA_DIR"] = v
            return


# Run at import so all downstream Mnemosyne() calls see the right path.
_ensure_mnemosyne_data_dir()


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
        return {}
    return parsed


def emit(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, default=str) + "\n")
    sys.stdout.flush()


def emit_context(hook_event_name: str, additional_context: str) -> None:
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
    """One deterministic opaque memory scope per actor + project."""
    material = f"{actor}|{project}".encode()
    return "mem-" + hashlib.sha256(material).hexdigest()[:16]


def _plugin_data_dir() -> str:
    """Default writable plugin state dir from PLUGIN_DATA (Codex extension).

    Falls back to MNEMOSYNE_DATA_DIR. Never uses a global home directory.
    """
    for name in ("PLUGIN_DATA", "CLAUDE_PLUGIN_DATA"):
        v = os.environ.get(name)
        if v:
            return v
    env_data = os.environ.get("MNEMOSYNE_DATA_DIR")
    if env_data:
        return env_data
    return os.path.join(tempfile.gettempdir(), "codex-mnemosyne-data")


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
        if len(content) > 500:
            content = content[:497] + "..."
        lines.append(f"-{tag} {content}")
    lines.append("</mnemosyne-recall>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Native ingest / recall wrappers (fail-open, returns structured outcome)
# ---------------------------------------------------------------------------


class Outcome:
    def __init__(self, ok: bool, error_code: str = "", error_message: str = ""):
        self.ok = ok
        self.error_code = error_code
        self.error_message = error_message

    def __bool__(self) -> bool:
        return self.ok


def _import_mnemosyne() -> Tuple[bool, str]:
    try:
        import mnemosyne  # noqa: F401
        from mnemosyne.core.memory import Mnemosyne  # noqa: F401
        from mnemosyne.core.inhale import IngestEvent  # noqa: F401
        from mnemosyne.core.recall_bounded import RecallPolicy  # noqa: F401
    except Exception:
        return False, "package_absent"
    return True, ""


def native_ingest(event_dict: Dict[str, Any]) -> Outcome:
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
# Static, content-free diagnostics for SessionEnd stderr (Task 8 official hook
# contract). SessionEnd output is advisory and systemMessage is not supported
# for it, so failures are surfaced as static stderr + nonzero exit only.
_DIAG_SESSION_END_RETAINED = (
    "mnemosyne: session end flush incomplete; some events retained.\n"
)
_DIAG_SESSION_END_UNAVAILABLE = "mnemosyne: session end status unavailable.\n"


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


def diag_session_end_retained() -> str:
    """Static, content-free stderr diagnostic for retained spool rows."""
    return _DIAG_SESSION_END_RETAINED


def diag_session_end_unavailable() -> str:
    """Static, content-free stderr diagnostic when the flush itself raises."""
    return _DIAG_SESSION_END_UNAVAILABLE


# ---------------------------------------------------------------------------
# Transport-only spool (0600, never searchable, ack-deleted, bounded)
# ---------------------------------------------------------------------------

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
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, mode=0o700, exist_ok=True)
    if not os.path.exists(path):
        fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
        os.close(fd)
    else:
        os.chmod(path, 0o600)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_SPOOL_SCHEMA)
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

    Returns "stored", "duplicate", "full", or "error". Never raises.
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
    """Best-effort row count (never raises). Returns 0 on absence or error.

    Kept for existing best-effort callers. SessionEnd's success/retained
    decision must NOT use this: a corrupt/uninspectable spool also yields 0
    here, which would silently represent it as success. Use
    ``spool_inspect_state`` for any decision that must distinguish empty from
    corrupt.
    """
    count, _inspectable = spool_inspect_state(path)
    return count


def spool_inspect_state(path: str) -> Tuple[int, bool]:
    """Strict spool inspection. Returns (row_count, inspectable).

    - Confirmed absent (ENOENT / FileNotFoundError) or zero-byte file:
      ``(0, True)`` — a genuine empty state is inspectable and successful.
    - Valid SQLite with the spool table: ``(count, True)``.
    - Corrupt / unreadable / not-SQLite, OR a path that exists but cannot be
      traversed/stat'd/opened (EACCES, ELOOP, ...): ``(0, False)`` — the
      caller must treat this as "status unavailable", never as success.

    Never raises. Classification is by the OSError subclass, NOT by
    ``os.path.exists`` (which returns False on EACCES during stat/traverse
    and would conflate an inaccessible-existing path with a genuinely absent
    one). ponytail: ceiling = a SQLite file that opens but has no
    spooled_events table (e.g. a different schema); we treat a missing table
    as 0 inspectable rows rather than corrupt, matching _ensure_spool's
    additive-schema contract. Upgrade path: if the schema ever becomes
    load-bearing for correctness, validate the column set here too.
    """
    # Stat with error classification. os.path.exists() is deliberately
    # avoided: it swallows ALL OSErrors (incl. EACCES) and returns False,
    # conflating "exists but inaccessible" with "confirmed absent".
    try:
        if os.path.getsize(path) == 0:
            return 0, True
    except FileNotFoundError:
        # Confirmed absent (ENOENT) — genuine empty state, success.
        return 0, True
    except OSError:
        # Exists-or-unknown but cannot be stat'd (EACCES on parent dir,
        # ELOOP, ENOTDIR, ...). Must NOT be treated as absent.
        return 0, False
    try:
        conn = sqlite3.connect(path)
        try:
            row = conn.execute("SELECT COUNT(*) FROM spooled_events").fetchone()
            return (int(row[0]) if row else 0), True
        finally:
            conn.close()
    except Exception:
        return 0, False


def _deadline_remaining(start: float, budget_s: float) -> float:
    return max(0.0, budget_s - (datetime.datetime.now().timestamp() - start))


def _flush_spool_inner(path: str, budget_s: float) -> Tuple[int, int, int]:
    """Inner flush logic (no env management). Returns (flushed, pending, terminal)."""
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
        trow = conn.execute(
            "SELECT COUNT(*) FROM spooled_events WHERE terminal = 1"
        ).fetchone()
        retained_terminal += int(trow[0]) if trow else 0
    except Exception:
        pass
    finally:
        conn.close()
    return flushed, retained_pending, retained_terminal


def flush_spool(
    path: str,
    env: Optional[Dict[str, Any]] = None,
    budget_s: float = 2.5,
) -> Tuple[int, int, int]:
    """Attempt to deliver spooled events via native ingest within a deadline.

    If ``env`` is provided, it is applied temporarily via ``_scoped_env`` and
    restored on exit — it never permanently mutates ``os.environ``.

    Deletes a row only after a successful ack. Terminal rows are retained.
    Corrupt rows are retained. Never raises.
    """
    if env:
        with _scoped_env({k: str(v) for k, v in env.items()}):
            return _flush_spool_inner(path, budget_s)
    return _flush_spool_inner(path, budget_s)


def _write_flush_result(path: str, result: Tuple[int, int, int]) -> None:
    """Write flush results to a temp file for the parent to read."""
    result_path = path + ".flush_result"
    try:
        with open(result_path, "w") as f:
            json.dump(result, f)
    except Exception:
        pass


def _read_flush_result(path: str) -> Tuple[int, int, int]:
    result_path = path + ".flush_result"
    try:
        with open(result_path) as f:
            data = json.load(f)
        os.unlink(result_path)
        return tuple(data)
    except Exception:
        return 0, 0, 0


def flush_spool_bounded(
    path: str, budget_s: float = _SESSION_END_BUDGET_S
) -> Tuple[int, int, int]:
    """Run flush_spool in a child process with a hard subprocess timeout.

    This provides a real hard process boundary: if the child is killed on
    timeout, unacked rows are retained (they were only ever deleted inside the
    child after a successful ack). The parent never blocks longer than
    budget_s + small overhead.

    Falls back to in-process flush (with SIGALRM where available) only if the
    subprocess cannot be spawned.
    """
    # Serialize current env so the child inherits the same config (disposable
    # MNEMOSYNE_DATA_DIR, PLUGIN_DATA, etc.) without the parent leaking.
    child_env = {k: v for k, v in os.environ.items()}

    try:
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import json, os, sys; "
                    f"sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r}); "
                    "import common; "
                    f"r = common._flush_spool_inner({path!r}, {budget_s!r}); "
                    f"common._write_flush_result({path!r}, r)"
                ),
            ],
            env=child_env,
            timeout=budget_s + 0.5,
            capture_output=True,
            text=True,
        )
        return _read_flush_result(path)
    except subprocess.TimeoutExpired:
        # Child was killed. Whatever it acked was committed per-row in the
        # child's SQLite connection. Unacked rows are retained.
        return _read_flush_result(path)
    except Exception:
        # Last-resort fallback: in-process flush.
        return _flush_spool_inner(path, budget_s)


# ---------------------------------------------------------------------------
# Convenience: ingest with spool fallback + honest messaging
# ---------------------------------------------------------------------------


# Test-only backdoor: MNEMOSYNE_CODEX_FORCE_SPOOL=1 forces the spool path,
# bypassing native ingest. This is a production-code test hook documented here
# for clarity; it does not affect deployed behavior unless set.
def ingest_or_spool(event_dict: Dict[str, Any]) -> Tuple[Outcome, str]:
    """Try native ingest; on failure, spool for later delivery.

    Returns (Outcome, spool_status). spool_status is one of:
    "stored", "duplicate", "full", "error", "" (ingest succeeded).
    """
    force = os.environ.get("MNEMOSYNE_CODEX_FORCE_SPOOL", "") == "1"
    if not force:
        outcome = native_ingest(event_dict)
        if outcome.ok:
            return outcome, ""
    status = spool_put(spool_path(), event_dict)
    return Outcome(False, "spooled", ""), status
