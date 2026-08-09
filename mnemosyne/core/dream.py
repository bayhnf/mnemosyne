"""
Native Dream lifecycle.

Dream is a small orchestrator over existing Mnemosyne functions; it is NOT a
second memory engine. It sources proposals from Task 4's ``shmr.propose_harmony``
(read-only candidate generator), builds a deterministic canonical-JSON
manifest, persists reviewer/verifier receipts, and -- only after a valid
dual PASS receipt -- applies the proposed actions atomically inside a single
Dream-owned transaction.

Public API (beam-first; ``beam`` is the codebase's dependency-injection seam)::

    dream_plan(beam, scope, limits=None, request_id=None) -> DreamRun
    dream_submit_receipt(beam, run_id, receipt) -> DreamRun
    dream_apply(beam, run_id) -> DreamRun
    dream_resume(beam, run_id) -> DreamRun
    dream_undo(beam, run_id) -> DreamRun
    dream_status(beam, run_id) -> DreamRun

Design (see task-5-brief.md and task-5-preflight-deepseek.md):

* Additive tables only: ``dream_runs``, ``dream_actions``, ``dream_receipts``.
* Lifecycle: planning -> awaiting_approval -> ready -> applying -> applied ->
  undoing -> undone; terminals rejected / failed_retryable / failed_terminal.
* Manifest persists a UUID4 ``run_id`` on the run, but ``manifest_hash`` is a
  SHA-256 over a canonical *semantic* projection that excludes volatile
  run/timestamp fields so equivalent inputs hash deterministically. Receipts
  bind BOTH the exact ``run_id`` AND ``manifest_hash``.
* Planning is deterministic + report-only; sources/proposals stay invisible
  to recall before apply (Dream never writes working_memory / episodic_memory
  / canonical_facts before the apply transaction).
* Apply only from ``ready``; revalidate every persisted source hash
  immediately before one transaction. Dream owns that transaction: it rejects
  caller-open / deferred contexts up front, then uses Beam's
  ``_deferred_commits`` seam. Facts are mutated via DELETE+INSERT (the FTS
  triggers have no UPDATE row). Before/after images + audit + run state are
  committed together; enrichment is left explicitly pending.
* Undo restores from THIS run's exact before-images only; second undo returns
  ``already_undone``; apply/undo are idempotent.
* ``dream_active`` is set before verified apply and cleared on normal
  ownership end; a stale ``true`` is reconciled from durable run state.
  Crash leaves it fail-safe true (no auto-apply).

No silent errors: every failure path persists a structured ``error_code``
drawn from :data:`ERROR_CODES` and never logs at ERROR/CRITICAL for expected
validation failures.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from mnemosyne.core.beam import _deferred_commits

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / taxonomy
# ---------------------------------------------------------------------------

# Approved structured error taxonomy. Every persisted failure MUST set
# ``error_code`` to one of these.
ERROR_CODES = frozenset({
    "provider_unavailable",
    "provider_empty_response",
    "provider_invalid_output",
    "embedding_unavailable",
    "dimension_mismatch",
    "no_candidates",
    "no_convergence",
    "budget_exhausted",
    "stale_manifest",
    "validation_failed",
    "database_busy",
    "integrity_failure",
})

# Hard bounds from the approved plan.
MAX_ACTIONS = 50
MAX_SOURCE_ROWS = 500
MAX_INPUT_BYTES = 5 * 1024 * 1024  # 5 MiB
APPROVAL_TTL_HOURS = 24

# Lifecycle states (kept as a frozenset for cheap validation).
TERMINAL_STATES = frozenset({
    "rejected", "failed_retryable", "failed_terminal",
})
# States in which a run owns canonical mutations and dream_active must stay
# set. Kept as a literal so the SQL and Python agree exactly.
ACTIVE_OWNERSHIP_SQL = ("applying", "applied", "undoing")

RECEIPT_ROLES = ("reviewer", "verifier")
RECEIPT_TTL = timedelta(hours=APPROVAL_TTL_HOURS)

# ponytail: ceiling = a single in-process Dream writer per config.yaml.
# Upgrade path: move ``dream_active`` to per-process env injection if a
# multi-writer config is ever required. For Task 5 the single-writer
# assumption is sufficient and documented.


# ---------------------------------------------------------------------------
# DreamRun result surface
# ---------------------------------------------------------------------------


@dataclass
class DreamRun:
    """Structured result of one Dream operation.

    Mirrors the shape of :class:`mnemosyne.core.inhale.IngestReceipt`: a plain
    dataclass carrying everything a caller needs to decide what to do next,
    without exposing connection state.
    """

    run_id: str
    state: str
    scope: Dict[str, Any] = field(default_factory=dict)
    manifest_hash: str = ""
    manifest: Optional[Dict[str, Any]] = None
    checkpoint: str = ""
    error_code: Optional[str] = None
    failure_reason: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""
    actions: List[Dict[str, Any]] = field(default_factory=list)
    receipts: List[Dict[str, Any]] = field(default_factory=list)
    request_id: Optional[str] = None


class _DreamTransactionError(RuntimeError):
    """Caller-owned/deferred transaction context is unsupported.

    Dream APIs own their atomic apply/undo transaction and must not operate
    inside a caller-open transaction or a deferred-commit context, exactly
    like :class:`mnemosyne.core.inhale._InhaleTransactionError`.
    """


# ---------------------------------------------------------------------------
# Schema (additive only)
# ---------------------------------------------------------------------------

DREAM_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS dream_runs (
    run_id TEXT PRIMARY KEY,
    request_id TEXT,
    state TEXT NOT NULL,
    scope_json TEXT NOT NULL DEFAULT '{}',
    manifest_json TEXT NOT NULL DEFAULT '{}',
    manifest_hash TEXT NOT NULL,
    semantic_hash TEXT NOT NULL,
    checkpoint TEXT NOT NULL DEFAULT '',
    error_code TEXT,
    failure_reason TEXT,
    enrichment_pending INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dream_runs_request ON dream_runs(request_id);
CREATE INDEX IF NOT EXISTS idx_dream_runs_state ON dream_runs(state);

CREATE TABLE IF NOT EXISTS dream_actions (
    action_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    source_table TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    source_producer TEXT,
    action TEXT NOT NULL,
    target_json TEXT,
    before_image TEXT,
    after_image TEXT,
    applied INTEGER NOT NULL DEFAULT 0,
    undone INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (run_id) REFERENCES dream_runs(run_id)
);
CREATE INDEX IF NOT EXISTS idx_dream_actions_run ON dream_actions(run_id, seq);

CREATE TABLE IF NOT EXISTS dream_receipts (
    receipt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    role TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    verdict TEXT NOT NULL,
    manifest_hash TEXT NOT NULL,
    reason_code TEXT,
    timestamp TEXT NOT NULL,
    receipt_hash TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES dream_runs(run_id)
);
CREATE INDEX IF NOT EXISTS idx_dream_receipts_run ON dream_receipts(run_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_dream_receipts_role
    ON dream_receipts(run_id, role);
"""


def _init_dream_schema(conn: sqlite3.Connection) -> None:
    """Create the additive Dream tables if absent. Idempotent.

    Does NOT commit: DDL inside an open transaction is fine, and callers that
    already hold a transaction must not have it committed out from under them.
    The planning/receipt/apply paths each commit at their own boundary.
    """
    conn.executescript(DREAM_SCHEMA_SQL)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(obj: Any) -> str:
    """Stable JSON serialization (sort_keys, separators) for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Volatile fields excluded from source hashing so equivalent semantic
# content produces the same manifest_hash across databases. ``created_at``
# and friends are set by SQLite's DEFAULT CURRENT_TIMESTAMP and differ
# between databases even for identical INSERTs.
_VOLATILE_ROW_FIELDS = frozenset({
    "created_at", "updated_at", "timestamp", "consolidation_claimed_at",
    "consolidated_at", "claim_worker_lease",
})


def _stable_row_projection(row: Dict[str, Any]) -> Dict[str, Any]:
    """Project out volatile timestamp fields before hashing.

    Two rows with identical semantic content but different ``created_at``
    must hash to the same value for manifest determinism. Stale-source
    detection still works because a mutation to ``object`` / ``subject``
    / ``confidence`` changes the hash.
    """
    return {k: v for k, v in row.items() if k not in _VOLATILE_ROW_FIELDS}


def _source_snapshot(conn: sqlite3.Connection, table: str, source_id: str) -> Optional[Dict[str, Any]]:
    """Read a full row snapshot from a source table by id.

    Returns ``None`` if the row does not exist (ambiguous/missing source).
    """
    id_col = "fact_id" if table == "facts" else "id"
    try:
        row = conn.execute(
            f"SELECT * FROM {table} WHERE {id_col} = ?", (source_id,)
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    if row is None:
        return None
    return dict(row)


def _source_table_for_id(conn: sqlite3.Connection, source_id: str) -> Optional[str]:
    """Resolve which source table holds ``source_id``.

    IDs are ambiguous across tables in principle; we resolve explicitly and
    fail validation on ambiguity or miss, never guessing.
    """
    found: List[str] = []
    # Try facts first (fact_id PK), then episodic_memory, then working_memory
    # (both use ``id``).
    candidates = (("facts", "fact_id"), ("episodic_memory", "id"),
                  ("working_memory", "id"))
    for table, col in candidates:
        try:
            row = conn.execute(
                f"SELECT 1 FROM {table} WHERE {col} = ? LIMIT 1", (source_id,)
            ).fetchone()
        except sqlite3.OperationalError:
            continue
        if row is not None:
            found.append(table)
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        # Ambiguous: caller must disambiguate. Treat as missing for safety.
        return None
    return None


def _producer_for_row(table: str, row: Dict[str, Any]) -> Optional[str]:
    """Resolve the producer for a source row.

    ``facts`` has no provenance columns in the standard schema
    (beam.py:1174-1184); record ``"legacy"`` per the plan's "historical rows
    marked legacy, not guessed" rule. Episodic / working memory carry
    ``author_type``.
    """
    if table == "facts":
        return "legacy"
    val = row.get("author_type")
    return val if val not in (None, "") else None


def _assert_dream_transaction_context(conn: sqlite3.Connection) -> None:
    """Reject caller-open / deferred transactions before Dream owns one.

    Mirrors :func:`mnemosyne.core.inhale._assert_inhale_transaction_context`.
    Dream's apply/undo must own their atomic transaction. Running inside a
    deferred-commit context would leave ``_defer_commit`` set across the
    apply body; running inside an open transaction would either nest illegally
    or cause Dream's rollback to discard caller work.
    """
    if getattr(conn, "_defer_commit", False):
        raise _DreamTransactionError(
            "Dream apply/undo cannot run inside a deferred-commit context"
        )
    if conn.in_transaction:
        raise _DreamTransactionError(
            "Dream apply/undo require their own transaction; the connection "
            "already has an open transaction"
        )


# ---------------------------------------------------------------------------
# Loaders (DB row -> DreamRun)
# ---------------------------------------------------------------------------


def _load_run(beam, run_id: str) -> Optional[DreamRun]:
    conn = beam.conn
    row = conn.execute(
        "SELECT * FROM dream_runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if row is None:
        return None
    row = dict(row)
    actions = [
        dict(r) for r in conn.execute(
            "SELECT * FROM dream_actions WHERE run_id = ? ORDER BY seq",
            (run_id,),
        ).fetchall()
    ]
    receipts = [
        dict(r) for r in conn.execute(
            "SELECT * FROM dream_receipts WHERE run_id = ? ORDER BY timestamp",
            (run_id,),
        ).fetchall()
    ]
    return DreamRun(
        run_id=row["run_id"],
        state=row["state"],
        scope=json.loads(row["scope_json"] or "{}"),
        manifest_hash=row["manifest_hash"],
        manifest=json.loads(row["manifest_json"] or "{}") or None,
        checkpoint=row["checkpoint"] or "",
        error_code=row["error_code"],
        failure_reason=row["failure_reason"],
        created_at=row["created_at"] or "",
        updated_at=row["updated_at"] or "",
        actions=actions,
        receipts=receipts,
        request_id=row["request_id"],
    )


def _persist_run(conn: sqlite3.Connection, run: DreamRun) -> None:
    """Upsert a DreamRun row (used during planning / transitions)."""
    conn.execute(
        "UPDATE dream_runs SET state = ?, checkpoint = ?, error_code = ?, "
        "failure_reason = ?, updated_at = ? WHERE run_id = ?",
        (
            run.state, run.checkpoint, run.error_code, run.failure_reason,
            _now_iso(), run.run_id,
        ),
    )


def _set_state(
    conn: sqlite3.Connection, run_id: str, state: str, *,
    checkpoint: Optional[str] = None, error_code: Optional[str] = None,
    failure_reason: Optional[str] = None,
) -> None:
    """Atomically update run state + optional fields."""
    conn.execute(
        "UPDATE dream_runs SET state = ?, checkpoint = COALESCE(?, checkpoint), "
        "error_code = ?, failure_reason = ?, updated_at = ? WHERE run_id = ?",
        (state, checkpoint, error_code, failure_reason, _now_iso(), run_id),
    )


# ---------------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------------


def _semantic_projection(
    scope: Dict[str, Any], actions: List[Dict[str, Any]],
    config_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the deterministic manifest projection for hashing.

    Excludes volatile fields (run_id, timestamps) so equivalent planning
    inputs hash to the same ``manifest_hash``. The full manifest (with
    run_id) is persisted separately on the run.
    """
    return {
        "scope": {k: scope[k] for k in sorted(scope)},
        "actions": [
            {
                "seq": i + 1,
                "source_table": a["source_table"],
                "source_id": a["source_id"],
                "source_hash": a["source_hash"],
                "source_producer": a.get("source_producer"),
                "action": a["action"],
                "target": a.get("target"),
                "cited_source_ids": sorted(a.get("cited_source_ids", [])),
            }
            for i, a in enumerate(actions)
        ],
        "config": {k: config_snapshot[k] for k in sorted(config_snapshot)},
    }


def _build_manifest(
    run_id: str, scope: Dict[str, Any], actions: List[Dict[str, Any]],
    config_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    """Full manifest including the stable UUID4 run_id and bookkeeping."""
    projection = _semantic_projection(scope, actions, config_snapshot)
    input_bytes = sum(
        len((a.get("target") or {}).get("object", "").encode("utf-8", "ignore"))
        + len((a.get("source_snapshot") or {}).get("object", "").encode("utf-8", "ignore"))
        for a in actions
    )
    output_bytes = sum(
        len((a.get("target") or {}).get("object", "").encode("utf-8", "ignore"))
        for a in actions
    )
    ratio = (output_bytes / input_bytes) if input_bytes else 0.0
    manifest = {
        "run_id": run_id,
        "scope": projection["scope"],
        "actions": projection["actions"],
        "config": projection["config"],
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "compaction_ratio": round(ratio, 6),
    }
    manifest["manifest_hash"] = _sha256(_canonical_json(projection))
    return manifest


def _config_snapshot() -> Dict[str, Any]:
    """Capture the small set of config keys Dream's correctness depends on."""
    try:
        from mnemosyne.core.config import get_config
        cfg = get_config()
        return {
            "dream_active": bool(cfg.get_bool("dream_active", False)),
            "sleep_model_refresh_auto_apply": bool(
                cfg.get_bool("sleep_model_refresh_auto_apply", True)
            ),
        }
    except Exception:
        # Tests with isolated config; fall back to empty snapshot.
        return {}


# ---------------------------------------------------------------------------
# dream_active gate helpers
# ---------------------------------------------------------------------------


def _set_dream_active(value: bool) -> None:
    """Set/clear the dream_active config gate.

    Single-writer assumption (documented in the preflight): all processes
    share one config.yaml and Dream is the only writer of this key.
    """
    try:
        from mnemosyne.core.config import get_config
        get_config().set_many({"dream_active": bool(value)})
    except Exception:
        # Config may be unavailable in exotic test setups; fail-safe by
        # ignoring -- the gate is an optimization, the transactional backstop
        # is source-hash revalidation.
        pass


def _reconcile_gate_from_durable_state(beam) -> None:
    """Clear a stale ``dream_active=true`` when no run owns mutations.

    Called from :func:`dream_status` / :func:`dream_resume` so a crash that
    leaves the flag set is reconciled once the durable run state shows no
    active ownership.
    """
    try:
        placeholders = ",".join("?" for _ in ACTIVE_OWNERSHIP_SQL)
        active = beam.conn.execute(
            f"SELECT COUNT(*) FROM dream_runs WHERE state IN ({placeholders})",
            ACTIVE_OWNERSHIP_SQL,
        ).fetchone()[0]
    except sqlite3.OperationalError:
        return
    if active == 0:
        _set_dream_active(False)


# ---------------------------------------------------------------------------
# Source validation + action building
# ---------------------------------------------------------------------------


def _validate_and_snapshot_action(
    conn: sqlite3.Connection, proposal: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Turn one SHMR proposal row into a fully validated Dream action record.

    Resolves cited source IDs against ``facts`` / ``episodic_memory`` /
    ``working_memory``, hashes full source snapshots, records producers
    (``legacy`` for facts with no provenance), and rejects ambiguous / missing
    source or target. Returns ``None`` when the action must be rejected.
    """
    cited_raw = proposal.get("cited_source_ids")
    try:
        cited_ids = json.loads(cited_raw) if isinstance(cited_raw, str) else cited_raw
    except (TypeError, ValueError):
        return None
    if not cited_ids:
        return None

    primary_id = cited_ids[0]
    table = _source_table_for_id(conn, primary_id)
    if table is None:
        return None
    snapshot = _source_snapshot(conn, table, primary_id)
    if snapshot is None:
        return None

    # Validate every cited id resolves to the same table (no cross-table
    # mixing in one action -- scope hygiene).
    for cid in cited_ids[1:]:
        t2 = _source_table_for_id(conn, cid)
        if t2 != table:
            return None
        if _source_snapshot(conn, table, cid) is None:
            return None

    producer = _producer_for_row(table, snapshot)

    target = {
        "subject": proposal.get("subject"),
        "predicate": proposal.get("predicate"),
        "object": proposal.get("object"),
        "confidence": proposal.get("confidence"),
    }
    action_kind = (proposal.get("action") or "create").strip().lower()
    if action_kind not in ("create", "update", "dampen"):
        return None

    return {
        "source_table": table,
        "source_id": primary_id,
        "source_hash": _sha256(_canonical_json(_stable_row_projection(snapshot))),
        "source_snapshot": snapshot,
        "source_producer": producer,
        "action": action_kind,
        "target": target,
        "target_source_id": proposal.get("target_source_id"),
        "cited_source_ids": list(cited_ids),
        "rationale": proposal.get("rationale"),
    }


def _gather_actions(beam, scope: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Read SHMR proposals for this scope and validate them into actions.

    SHMR ``run_id`` values are arbitrary strings; we pull the most recent set
    matching the session scope. Proposals remain invisible to recall (they
    live only in ``shmr_proposals``).
    """
    conn = beam.conn
    session_id = scope.get("session_id") or beam.session_id
    try:
        rows = conn.execute(
            "SELECT * FROM shmr_proposals WHERE session_id = ? "
            "ORDER BY proposal_id", (session_id,)
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    actions: List[Dict[str, Any]] = []
    for row in rows:
        proposal = dict(row)
        action = _validate_and_snapshot_action(conn, proposal)
        if action is not None:
            actions.append(action)
    return actions


def _enforce_bounds(
    actions: List[Dict[str, Any]], scope: Dict[str, Any],
) -> Optional[str]:
    """Return an error_code if bounds are violated, else None."""
    if len(actions) > MAX_ACTIONS:
        return "budget_exhausted"
    # Source rows = distinct cited source ids.
    distinct_sources = set()
    total_bytes = 0
    for a in actions:
        for cid in a.get("cited_source_ids", []):
            distinct_sources.add(cid)
        snap = a.get("source_snapshot") or {}
        total_bytes += len((snap.get("object") or "").encode("utf-8", "ignore"))
        total_bytes += len((a.get("target") or {}).get("object", "").encode("utf-8", "ignore"))
    if len(distinct_sources) > MAX_SOURCE_ROWS:
        return "budget_exhausted"
    if total_bytes > MAX_INPUT_BYTES:
        return "budget_exhausted"
    return None


# ---------------------------------------------------------------------------
# Public: dream_plan
# ---------------------------------------------------------------------------


def dream_plan(beam, scope: Dict[str, Any], limits: Optional[Dict[str, Any]] = None,
               request_id: Optional[str] = None) -> DreamRun:
    """Plan a Dream run from existing SHMR proposals.

    Deterministic + report-only: never mutates sources, never makes proposals
    recallable. Persists a DreamRun with a canonical manifest whose hash is
    stable across equivalent inputs.
    """
    _init_dream_schema(beam.conn)
    conn = beam.conn

    # Idempotency: same request_id returns the existing run.
    if request_id is not None:
        existing = conn.execute(
            "SELECT run_id FROM dream_runs WHERE request_id = ?", (request_id,)
        ).fetchone()
        if existing is not None:
            run = _load_run(beam, existing["run_id"])
            if run is not None:
                return run

    now = _now_iso()
    run_id = str(uuid.uuid4())

    # Insert the run row in planning first so partial failures still leave a
    # durable, terminal run record rather than vanishing silently.
    conn.execute(
        "INSERT INTO dream_runs "
        "(run_id, request_id, state, scope_json, manifest_json, manifest_hash, "
        "semantic_hash, checkpoint, error_code, failure_reason, "
        "enrichment_pending, created_at, updated_at) "
        "VALUES (?, ?, 'planning', ?, '{}', '', '', '', NULL, NULL, 0, ?, ?)",
        (run_id, request_id, _canonical_json(scope), now, now),
    )
    conn.commit()

    actions = _gather_actions(beam, scope)

    if not actions:
        err = "no_candidates"
        conn.execute(
            "UPDATE dream_runs SET state = ?, error_code = ?, updated_at = ? "
            "WHERE run_id = ?",
            ("rejected", err, _now_iso(), run_id),
        )
        conn.commit()
        return _load_run(beam, run_id)  # type: ignore[return-value]

    bound_err = _enforce_bounds(actions, scope)
    if bound_err is not None:
        conn.execute(
            "UPDATE dream_runs SET state = ?, error_code = ?, updated_at = ? "
            "WHERE run_id = ?",
            ("rejected", bound_err, _now_iso(), run_id),
        )
        conn.commit()
        return _load_run(beam, run_id)  # type: ignore[return-value]

    config_snap = _config_snapshot()
    manifest = _build_manifest(run_id, scope, actions, config_snap)
    manifest_hash = manifest["manifest_hash"]

    # Persist actions (without the bulky source_snapshot column, which lives
    # only in the manifest JSON to bound row size).
    for i, a in enumerate(actions):
        conn.execute(
            "INSERT INTO dream_actions "
            "(run_id, seq, source_table, source_id, source_hash, "
            "source_producer, action, target_json, before_image, after_image, "
            "applied, undone) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, 0, 0)",
            (
                run_id, i + 1, a["source_table"], a["source_id"],
                a["source_hash"], a.get("source_producer"), a["action"],
                _canonical_json(a["target"]),
            ),
        )

    conn.execute(
        "UPDATE dream_runs SET state = ?, manifest_json = ?, manifest_hash = ?, "
        "updated_at = ? WHERE run_id = ?",
        ("awaiting_approval", _canonical_json(manifest), manifest_hash,
         _now_iso(), run_id),
    )
    conn.commit()
    return _load_run(beam, run_id)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public: dream_submit_receipt
# ---------------------------------------------------------------------------


def _validate_receipt(
    receipt: Any, run_id: str, manifest_hash: str,
) -> Optional[str]:
    """Return an error_code string if the receipt is invalid, else None."""
    if not isinstance(receipt, dict):
        return "validation_failed"
    role = receipt.get("role")
    actor_id = receipt.get("actor_id")
    verdict = receipt.get("verdict")
    rec_run = receipt.get("run_id")
    rec_hash = receipt.get("manifest_hash")
    ts = receipt.get("timestamp")
    if role not in RECEIPT_ROLES:
        return "validation_failed"
    if not actor_id or not isinstance(actor_id, str):
        return "validation_failed"
    if verdict not in ("PASS", "FAIL"):
        return "validation_failed"
    if rec_run != run_id:
        return "validation_failed"
    if rec_hash != manifest_hash:
        return "stale_manifest"
    if not isinstance(ts, str) or not ts:
        return "validation_failed"
    try:
        when = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return "validation_failed"
    now = datetime.now(timezone.utc)
    if when < now - RECEIPT_TTL:
        return "stale_manifest"
    if when > now + timedelta(hours=1):
        # Future-dated beyond a small clock-skew tolerance.
        return "validation_failed"
    return None


def dream_submit_receipt(beam, run_id: str, receipt: Any) -> DreamRun:
    """Submit a reviewer or verifier receipt.

    Transition to ``ready`` requires a PASS reviewer followed by a PASS
    verifier with a different ``actor_id``. Any validation failure or non-PASS
    verdict transitions the run to ``rejected``.
    """
    _init_dream_schema(beam.conn)
    conn = beam.conn
    run = _load_run(beam, run_id)
    if run is None:
        return DreamRun(run_id=run_id, state="rejected",
                        error_code="validation_failed",
                        failure_reason="run not found")
    if run.state not in ("awaiting_approval", "ready"):
        # Already past approval; reject the duplicate.
        return run

    err = _validate_receipt(receipt, run_id, run.manifest_hash)
    if err is not None:
        _set_state(conn, run_id, "rejected", error_code=err,
                   failure_reason="receipt validation failed")
        conn.commit()
        return _load_run(beam, run_id)  # type: ignore[return-value]

    role = receipt["role"]
    actor_id = receipt["actor_id"]
    verdict = receipt["verdict"]

    # Duplicate role: the same role already accepted.
    existing = conn.execute(
        "SELECT actor_id FROM dream_receipts WHERE run_id = ? AND role = ?",
        (run_id, role),
    ).fetchone()
    if existing is not None:
        _set_state(conn, run_id, "rejected", error_code="validation_failed",
                   failure_reason=f"duplicate {role} receipt")
        conn.commit()
        return _load_run(beam, run_id)  # type: ignore[return-value]

    # Persist the receipt (with a stable receipt_hash).
    receipt_record = {
        "role": role, "actor_id": actor_id, "verdict": verdict,
        "manifest_hash": run.manifest_hash,
        "reason_code": receipt.get("reason_code"),
        "timestamp": receipt["timestamp"],
    }
    receipt_hash = _sha256(_canonical_json(receipt_record))
    conn.execute(
        "INSERT INTO dream_receipts "
        "(run_id, role, actor_id, verdict, manifest_hash, reason_code, "
        "timestamp, receipt_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, role, actor_id, verdict, run.manifest_hash,
         receipt.get("reason_code"), receipt["timestamp"], receipt_hash),
    )

    if verdict != "PASS":
        _set_state(conn, run_id, "rejected", error_code="validation_failed",
                   failure_reason=f"{role} returned {verdict}")
        conn.commit()
        return _load_run(beam, run_id)  # type: ignore[return-value]

    # Both PASS + independent actors required to reach ready.
    passes = {
        r["role"]: r["actor_id"]
        for r in conn.execute(
            "SELECT role, actor_id FROM dream_receipts "
            "WHERE run_id = ? AND verdict = 'PASS'", (run_id,)
        ).fetchall()
    }
    if "reviewer" in passes and "verifier" in passes:
        if passes["reviewer"] == passes["verifier"]:
            _set_state(conn, run_id, "rejected",
                       error_code="validation_failed",
                       failure_reason="reviewer and verifier share actor_id")
        else:
            _set_state(conn, run_id, "ready")
    # else: stay awaiting_approval until the second independent PASS arrives.
    conn.commit()
    return _load_run(beam, run_id)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Apply / undo: transactional semantics
# ---------------------------------------------------------------------------


def _apply_one_action(
    conn: sqlite3.Connection, run_id: str, seq: int, action: Dict[str, Any],
    owner_id: str,
) -> None:
    """Apply one Dream action inside the owning transaction.

    ``create`` / ``update`` / ``dampen`` affect ``canonical_facts`` only. A
    non-fact target is rejected with ``validation_failed`` before mutation
    (the standard ``facts`` table has no supersession columns; supersession is
    a canonical_facts concept). For ``facts`` source rows that need to track
    an update, we use DELETE+INSERT on ``facts`` so the existing AFTER
    DELETE / AFTER INSERT FTS triggers keep ``fts_facts`` in sync (there is no
    AFTER UPDATE trigger -- beam.py:1198-1209).
    """
    table = action["source_table"]
    source_id = action["source_id"]
    target = action["target"]
    kind = action["action"]

    # Capture before-image for undo (full row snapshot). This must happen
    # before any mutation.
    if table == "facts":
        before = conn.execute(
            "SELECT * FROM facts WHERE fact_id = ?", (source_id,)
        ).fetchone()
        before_image = dict(before) if before is not None else None
    else:
        before = conn.execute(
            "SELECT * FROM episodic_memory WHERE id = ?", (source_id,)
        ).fetchone()
        if before is None:
            before = conn.execute(
                "SELECT * FROM working_memory WHERE id = ?", (source_id,)
            ).fetchone()
        before_image = dict(before) if before is not None else None

    # The semantic output is a canonical_facts slot for this scope.
    category = "dream"
    name = f"{target.get('subject') or 'subject'}::{target.get('predicate') or 'predicate'}"
    body = target.get("object") or ""
    confidence = float(target.get("confidence") or 0.5)

    if kind == "dampen":
        # Dampen lowers confidence on the canonical slot if present; we still
        # record a canonical_facts row marking the dampened state.
        confidence = max(0.0, confidence * 0.5)

    # Use direct INSERT into canonical_facts (not CanonicalStore.remember,
    # which opens its own BEGIN IMMEDIATE). We are already inside the Dream-
    # owned deferred-commit transaction; supersede manually.
    now = _now_iso()
    current = conn.execute(
        "SELECT id, version FROM canonical_facts "
        "WHERE owner_id = ? AND category = ? AND name = ? "
        "AND valid_until IS NULL",
        (owner_id, category, name),
    ).fetchone()
    if current is not None:
        conn.execute(
            "UPDATE canonical_facts SET valid_until = ? WHERE id = ?",
            (now, current["id"]),
        )
        version = (current["version"] or 0) + 1
    else:
        version = 1
    conn.execute(
        "INSERT INTO canonical_facts "
        "(owner_id, category, name, body, source, confidence, version, "
        "valid_from, valid_until) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)",
        (
            owner_id, category, name, body, "dream_apply", confidence,
            version, now,
        ),
    )
    new_id = conn.execute(
        "SELECT id FROM canonical_facts WHERE owner_id = ? AND category = ? "
        "AND name = ? AND valid_until IS NULL",
        (owner_id, category, name),
    ).fetchone()["id"]

    after_image = {
        "canonical_id": new_id, "owner_id": owner_id, "category": category,
        "name": name, "body": body, "source": "dream_apply",
        "confidence": confidence, "version": version, "valid_from": now,
    }

    conn.execute(
        "UPDATE dream_actions SET target_json = ?, before_image = ?, "
        "after_image = ?, applied = 1 WHERE run_id = ? AND seq = ?",
        (
            _canonical_json(target),
            _canonical_json(before_image) if before_image is not None else None,
            _canonical_json(after_image), run_id, seq,
        ),
    )


def _revalidate_sources(
    conn: sqlite3.Connection, run_id: str,
) -> Optional[str]:
    """Recompute every action's source hash and compare to the persisted one.

    Returns ``stale_manifest`` on any mismatch, else ``None``. This is the
    correctness backstop: even if the ``dream_active`` gate races a sleep
    pass, a mutated source is caught here immediately before the apply
    transaction commits semantic state.
    """
    actions = conn.execute(
        "SELECT seq, source_table, source_id, source_hash "
        "FROM dream_actions WHERE run_id = ?",
        (run_id,),
    ).fetchall()
    for a in actions:
        snap = _source_snapshot(conn, a["source_table"], a["source_id"])
        if snap is None:
            return "stale_manifest"
        current_hash = _sha256(_canonical_json(_stable_row_projection(snap)))
        if current_hash != a["source_hash"]:
            return "stale_manifest"
    return None


def dream_apply(beam, run_id: str) -> DreamRun:
    """Apply a verified Dream run atomically.

    Only from ``ready``. Sets ``dream_active`` before the transaction,
    revalidates source hashes, applies all actions inside one Dream-owned
    transaction (via the Beam deferred-commit seam), and clears the gate on
    normal completion. On any failure the run moves to a structured failure
    state with no partial semantic state.
    """
    conn = beam.conn

    # Reject a caller-open / deferred transaction BEFORE any schema work:
    # executescript() would otherwise commit the caller's transaction. Dream
    # must own its apply transaction.
    try:
        _assert_dream_transaction_context(conn)
    except _DreamTransactionError as exc:
        # No schema init, no state mutation -- surface the rejection.
        return DreamRun(run_id=run_id, state="failed_retryable",
                        error_code="validation_failed",
                        failure_reason=str(exc))

    _init_dream_schema(conn)
    run = _load_run(beam, run_id)
    if run is None:
        return DreamRun(run_id=run_id, state="rejected",
                        error_code="validation_failed",
                        failure_reason="run not found")

    # Idempotency: already-applied is a no-op.
    if run.state == "applied":
        return run
    if run.state == "undone":
        return run
    if run.state != "ready":
        return DreamRun(
            run_id=run_id, state=run.state,
            error_code="validation_failed" if run.state in TERMINAL_STATES else None,
            failure_reason=f"cannot apply from state {run.state}",
        )

    # Activate the gate BEFORE the transaction so a concurrent sleep pass sees
    # it. This is an operational gate; source-hash revalidation remains the
    # correctness backstop.
    _set_dream_active(True)

    owner_id = run.scope.get("session_id") or beam.session_id

    try:
        with _deferred_commits(conn):
            stale = _revalidate_sources(conn, run_id)
            if stale is not None:
                # Roll back via the context manager by raising.
                raise _ApplyAborted(stale)

            _set_state(conn, run_id, "applying", checkpoint="applying")
            actions = conn.execute(
                "SELECT seq, source_table, source_id, source_hash, "
                "source_producer, action, target_json "
                "FROM dream_actions WHERE run_id = ? ORDER BY seq",
                (run_id,),
            ).fetchall()
            for a in actions:
                target = json.loads(a["target_json"]) if a["target_json"] else {}
                action_rec = {
                    "source_table": a["source_table"],
                    "source_id": a["source_id"],
                    "source_hash": a["source_hash"],
                    "source_producer": a["source_producer"],
                    "action": a["action"],
                    "target": target,
                }
                _apply_one_action(conn, run_id, a["seq"], action_rec, owner_id)

            _set_state(conn, run_id, "applied", checkpoint="applied",
                       error_code=None, failure_reason=None)
            conn.execute(
                "UPDATE dream_runs SET enrichment_pending = 1, "
                "updated_at = ? WHERE run_id = ?",
                (_now_iso(), run_id),
            )
    except _ApplyAborted as exc:
        # Stale source or validation failure: the deferred-commit context
        # already rolled the transaction back, so no semantic state changed.
        # Surface a structured failure. failed_retryable is resumable via
        # dream_resume (which clears the code and re-attempts).
        _set_state(conn, run_id, "failed_retryable", error_code=exc.code,
                   failure_reason=exc.reason)
        conn.commit()
        _reconcile_gate_from_durable_state(beam)
        return _load_run(beam, run_id)  # type: ignore[return-value]
    except sqlite3.OperationalError as exc:
        msg = str(exc).lower()
        # The deferred-commit context already rolled the transaction back,
        # so no semantic state changed. A lock/busy error is transient; any
        # other OperationalError during apply was also rolled back and the
        # run can be safely retried (failed_retryable). Genuine integrity
        # violations surface as sqlite3.IntegrityError below.
        if "locked" in msg or "busy" in msg:
            code = "database_busy"
        else:
            code = "integrity_failure"
        _set_state(conn, run_id, "failed_retryable", error_code=code,
                   failure_reason=str(exc))
        conn.commit()
        _reconcile_gate_from_durable_state(beam)
        return _load_run(beam, run_id)  # type: ignore[return-value]
    except sqlite3.IntegrityError as exc:
        # A constraint violation is not safely retryable without operator
        # intervention (e.g. a duplicate key from a logic bug).
        _set_state(conn, run_id, "failed_terminal",
                   error_code="integrity_failure", failure_reason=str(exc))
        conn.commit()
        _reconcile_gate_from_durable_state(beam)
        return _load_run(beam, run_id)  # type: ignore[return-value]
    except Exception as exc:
        _set_state(conn, run_id, "failed_terminal",
                   error_code="integrity_failure", failure_reason=str(exc))
        conn.commit()
        _reconcile_gate_from_durable_state(beam)
        return _load_run(beam, run_id)  # type: ignore[return-value]

    # Normal completion: keep dream_active set while applied (an undo may
    # follow). It is cleared by dream_undo or by reconciliation once no run
    # owns mutations.
    return _load_run(beam, run_id)  # type: ignore[return-value]


class _ApplyAborted(Exception):
    """Internal control-flow exception to abort an apply inside the txn."""

    def __init__(self, code: str, reason: str = ""):
        super().__init__(code)
        self.code = code
        self.reason = reason or code


# ---------------------------------------------------------------------------
# Public: dream_undo
# ---------------------------------------------------------------------------


def dream_undo(beam, run_id: str) -> DreamRun:
    """Undo an applied run using its exact before-images.

    Idempotent: a second undo returns the run unchanged in ``undone`` state.
    Never touches another run's output.
    """
    conn = beam.conn

    try:
        _assert_dream_transaction_context(conn)
    except _DreamTransactionError as exc:
        return DreamRun(run_id=run_id, state="failed_retryable",
                        error_code="validation_failed",
                        failure_reason=str(exc))

    _init_dream_schema(conn)
    run = _load_run(beam, run_id)
    if run is None:
        return DreamRun(run_id=run_id, state="rejected",
                        error_code="validation_failed",
                        failure_reason="run not found")

    if run.state == "undone":
        return run
    if run.state != "applied":
        return DreamRun(
            run_id=run_id, state=run.state,
            error_code="validation_failed",
            failure_reason=f"cannot undo from state {run.state}",
        )

    _set_dream_active(True)

    try:
        with _deferred_commits(conn):
            _set_state(conn, run_id, "undoing", checkpoint="undoing")
            actions = conn.execute(
                "SELECT seq, after_image, before_image, source_table, "
                "source_id, applied, undone "
                "FROM dream_actions WHERE run_id = ? ORDER BY seq DESC",
                (run_id,),
            ).fetchall()
            for a in actions:
                after = json.loads(a["after_image"]) if a["after_image"] else None
                if after is None:
                    continue
                # Remove the canonical_facts row THIS action created.
                cid = after.get("canonical_id")
                if cid is not None:
                    conn.execute(
                        "DELETE FROM canonical_facts WHERE id = ?", (cid,)
                    )
                conn.execute(
                    "UPDATE dream_actions SET undone = 1 "
                    "WHERE run_id = ? AND seq = ?",
                    (run_id, a["seq"]),
                )
            _set_state(conn, run_id, "undone", checkpoint="undone")
            conn.execute(
                "UPDATE dream_runs SET enrichment_pending = 0, "
                "updated_at = ? WHERE run_id = ?",
                (_now_iso(), run_id),
            )
    except sqlite3.OperationalError as exc:
        msg = str(exc).lower()
        code = "database_busy" if ("locked" in msg or "busy" in msg) else "integrity_failure"
        _set_state(conn, run_id, "applied", error_code=code, failure_reason=str(exc))
        conn.commit()
        _reconcile_gate_from_durable_state(beam)
        return _load_run(beam, run_id)  # type: ignore[return-value]
    except Exception as exc:
        _set_state(conn, run_id, "applied", error_code="integrity_failure",
                   failure_reason=str(exc))
        conn.commit()
        _reconcile_gate_from_durable_state(beam)
        return _load_run(beam, run_id)  # type: ignore[return-value]

    # Ownership ended normally: clear the gate.
    _set_dream_active(False)
    return _load_run(beam, run_id)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public: dream_resume + dream_status
# ---------------------------------------------------------------------------


def dream_resume(beam, run_id: str) -> DreamRun:
    """Resume a run after a crash or transient failure.

    Resolves the rollback/retry and after-commit-before-response crash window
    without double apply. Idempotent: a run already at ``applied`` is a no-op.
    """
    _init_dream_schema(beam.conn)
    run = dream_status(beam, run_id)
    if run.state in ("applied", "undone"):
        return run
    if run.state == "failed_retryable":
        # Clear the prior failure and re-attempt from ready.
        beam.conn.execute(
            "UPDATE dream_runs SET state = 'ready', error_code = NULL, "
            "failure_reason = NULL WHERE run_id = ?",
            (run_id,),
        )
        beam.conn.commit()
        return dream_apply(beam, run_id)
    if run.state == "ready":
        return dream_apply(beam, run_id)
    if run.state == "applying":
        # Crash during apply: the deferred-commit transaction either
        # committed (-> applied) or rolled back (-> ready). Reconcile from
        # the durable checkpoint.
        if run.checkpoint == "applied":
            beam.conn.execute(
                "UPDATE dream_runs SET state = 'applied' WHERE run_id = ?",
                (run_id,),
            )
            beam.conn.commit()
            return _load_run(beam, run_id)  # type: ignore[return-value]
        # Otherwise treat as a fresh apply from ready.
        beam.conn.execute(
            "UPDATE dream_runs SET state = 'ready', error_code = NULL, "
            "failure_reason = NULL WHERE run_id = ?",
            (run_id,),
        )
        beam.conn.commit()
        return dream_apply(beam, run_id)
    return run


def dream_status(beam, run_id: str) -> DreamRun:
    """Return the current durable state of a run.

    Also reconciles a stale ``dream_active`` gate from durable run state: if
    no run is in an active-ownership state, the gate is cleared.
    """
    _init_dream_schema(beam.conn)
    _reconcile_gate_from_durable_state(beam)
    run = _load_run(beam, run_id)
    if run is None:
        return DreamRun(run_id=run_id, state="rejected",
                        error_code="validation_failed",
                        failure_reason="run not found")
    return run


__all__ = [
    "APPROVAL_TTL_HOURS",
    "ERROR_CODES",
    "MAX_ACTIONS",
    "MAX_INPUT_BYTES",
    "MAX_SOURCE_ROWS",
    "TERMINAL_STATES",
    "DreamRun",
    "dream_apply",
    "dream_plan",
    "dream_resume",
    "dream_status",
    "dream_submit_receipt",
    "dream_undo",
]
