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


def _reject_if_caller_holds_transaction(beam, run_id: str = "") -> Optional[DreamRun]:
    """Every public Dream entrypoint must call this BEFORE any schema DDL.

    ``_init_dream_schema`` calls ``executescript``, which implicitly COMMITs
    any pending transaction on the connection. That would destroy a caller's
    open transaction and commit its uncommitted rows. Detect the caller-owned
    context up front and return a structured rejection instead.

    Returns a ``DreamRun`` rejection if the caller holds a transaction, else
    ``None`` and the caller may proceed.
    """
    try:
        _assert_dream_transaction_context(beam.conn)
    except _DreamTransactionError as exc:
        return DreamRun(
            run_id=run_id, state="failed_retryable",
            error_code="validation_failed", failure_reason=str(exc),
        )
    return None


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
) -> Dict[str, Any]:
    """Build the deterministic manifest projection for hashing.

    Excludes volatile fields (run_id, timestamps) AND runtime config gates
    (``dream_active``, which Dream itself toggles) so equivalent planning
    inputs hash to the same ``manifest_hash`` regardless of runtime state.
    The full manifest persists an audit-only config snapshot separately.
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
    }


def _build_manifest(
    run_id: str, scope: Dict[str, Any], actions: List[Dict[str, Any]],
    config_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    """Full manifest including the stable UUID4 run_id and bookkeeping.

    ``manifest_hash`` covers only the semantic projection (scope + actions +
    sources); the ``config`` snapshot is persisted audit-only and does NOT
    enter the hash, so toggling ``dream_active`` does not change the hash for
    identical planning inputs.
    """
    projection = _semantic_projection(scope, actions)
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
        # Audit-only: runtime config snapshot, NOT part of manifest_hash.
        "config_audit": {k: config_snapshot[k] for k in sorted(config_snapshot)},
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "compaction_ratio": round(ratio, 6),
    }
    manifest["manifest_hash"] = _sha256(_canonical_json(projection))
    return manifest


def _config_snapshot() -> Dict[str, Any]:
    """Capture the small set of config keys Dream depends on, for audit only.

    This snapshot is persisted on the manifest for audit/reproducibility but
    does NOT enter the manifest_hash (see :func:`_semantic_projection`). A
    failure to read config yields an empty snapshot + a warning, not a silent
    swallow -- the caller still proceeds because config is audit-only.
    """
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
        logger.warning("config audit snapshot failed")
        return {"config_unavailable": True}


# ---------------------------------------------------------------------------
# dream_active gate helpers
# ---------------------------------------------------------------------------


def _set_dream_active(value: bool) -> Optional[str]:
    """Set/clear the dream_active config gate.

    Single-writer assumption (documented in the preflight): all processes
    share one config.yaml and Dream is the only writer of this key.

    Returns an error_code string on failure (the caller surfaces a structured
    failure rather than silently proceeding fail-open), or ``None`` on
    success. Clearing (``value=False``) is best-effort and returns ``None``
    even on failure because the failure direction is fail-safe (auto-apply
    stays off); only setting (``value=True``) must not fail silently.
    """
    try:
        from mnemosyne.core.config import get_config
        get_config().set_many({"dream_active": bool(value)})
    except Exception:
        if value:
            logger.warning("dream_active gate could not be set")
            return "validation_failed"
        # Clearing is best-effort; fail-safe direction.
        return None
    return None


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


# Proposal statuses eligible for Dream planning. ``proposed`` is the fresh
# status shmr writes; ``rolled_back`` proposals must never be planned.
ELIGIBLE_PROPOSAL_STATUSES = ("proposed",)


# Alias map between the public Dream scope contract and Task-4 ``scope_json``
# storage fields. Task-4 ``_scope_from_row`` (shmr.py:659-670) persists
# ``session_id``, ``author_id``, ``author_type``, ``channel_id``; the public
# Dream scope contract uses ``session_id``, ``actor_id``, ``producer``,
# ``project_id``. These are the same logical fields under different names.
# We normalize both sides to the canonical (public) name at the trust
# boundary before comparing, so a mismatch on ANY logical field rejects.
_SCOPE_ALIASES = {
    # canonical public name -> tuple of all known aliases (including itself)
    "actor_id": ("actor_id", "author_id"),
    "producer": ("producer", "author_type"),
    "project_id": ("project_id", "channel_id"),
}


def _normalize_scope(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a raw scope dict to canonical logical field names.

    ``session_id`` is always passed through (exact match, no alias). For the
    three provenance fields, the first non-null value among the canonical name
    and its Task-4 storage aliases wins, recorded under the canonical name.
    Fields not in the alias map are preserved verbatim so future fields are
    not silently dropped.
    """
    out: Dict[str, Any] = {}
    sid = raw.get("session_id")
    if sid is not None:
        out["session_id"] = sid
    for canonical, aliases in _SCOPE_ALIASES.items():
        for alias in aliases:
            val = raw.get(alias)
            if val is not None and val != "":
                out[canonical] = val
                break
    # Preserve any non-alias fields verbatim (forward compatibility).
    known = {"session_id"}
    for aliases in _SCOPE_ALIASES.values():
        known.update(aliases)
    for k, v in raw.items():
        if k not in known and v is not None:
            out.setdefault(k, v)
    return out


def _scope_matches(proposal_scope_raw: Dict[str, Any],
                   plan_scope_raw: Dict[str, Any]) -> bool:
    """Strict logical-field scope-provenance match.

    Both sides are normalized to canonical names (``actor_id``,
    ``producer``, ``project_id``) via :func:`_normalize_scope` before
    comparison. A proposal is eligible only if, for every logical provenance
    field, either both sides declare the same value, or neither side declares
    it. If either side declares a field the other does not match/declare,
    the proposal is rejected. ``session_id`` is always required and exact.
    """
    proposal = _normalize_scope(proposal_scope_raw)
    plan = _normalize_scope(plan_scope_raw)
    for canonical in _SCOPE_ALIASES:
        p_val = proposal.get(canonical)
        plan_val = plan.get(canonical)
        if p_val is not None and plan_val is not None and p_val != plan_val:
            return False
        # Fail closed in both directions: if EITHER side declares this
        # logical provenance field and the other does not, reject. No
        # bare-session exception -- a session-only plan must not select a
        # proposal that declares actor/producer/project, and vice versa.
        if (p_val is None) != (plan_val is None):
            return False
    return True


def _gather_actions(
    beam, scope: Dict[str, Any],
) -> tuple:  # (actions, consumed_proposal_ids)
    """Read ELIGIBLE SHMR proposals for this scope and validate them.

    Filters:
      * ``status`` must be in ELIGIBLE_PROPOSAL_STATUSES (excludes
        ``rolled_back`` and already-claimed proposals).
      * ``scope_json`` provenance must match the plan scope (no cross-actor
        / cross-project leakage).
    Returns the validated actions AND the list of ``proposal_id``s consumed,
    so :func:`dream_plan` can mark them claimed atomically (mutating the
    proposal queue status, never source memory).
    """
    conn = beam.conn
    session_id = scope.get("session_id") or beam.session_id
    try:
        rows = conn.execute(
            "SELECT * FROM shmr_proposals WHERE session_id = ? "
            "AND status IN (" + ",".join("?" for _ in ELIGIBLE_PROPOSAL_STATUSES) + ") "
            "ORDER BY proposal_id",
            (session_id, *ELIGIBLE_PROPOSAL_STATUSES),
        ).fetchall()
    except sqlite3.OperationalError:
        return [], []
    actions: List[Dict[str, Any]] = []
    consumed: List[int] = []
    for row in rows:
        proposal = dict(row)
        # Scope-provenance filter.
        try:
            proposal_scope = json.loads(proposal.get("scope_json") or "{}")
        except (TypeError, ValueError):
            proposal_scope = {}
        if not _scope_matches(proposal_scope, scope):
            continue
        action = _validate_and_snapshot_action(conn, proposal)
        if action is not None:
            action["proposal_id"] = proposal["proposal_id"]
            actions.append(action)
            consumed.append(proposal["proposal_id"])
    return actions, consumed


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
    rejection = _reject_if_caller_holds_transaction(beam)
    if rejection is not None:
        return rejection
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

    # I6-B: the ENTIRE gather + claim + run/action persistence happens inside
    # ONE BEGIN IMMEDIATE transaction (serialized via _deferred_commits). This
    # closes the TOCTOU window: a concurrent dream_plan on another connection
    # blocks on BEGIN IMMEDIATE until this transaction commits, so two plans
    # can never both SELECT the same 'proposed' proposal and both claim it.
    # The claim UPDATE additionally has a CAS guard (status='proposed') +
    # row-count validation as defense-in-depth.
    try:
        with _deferred_commits(conn):
            conn.execute("BEGIN IMMEDIATE")

            actions, consumed_ids = _gather_actions(beam, scope)

            if not actions:
                # No eligible proposals. Persist a terminal run so the caller
                # has a durable structured result.
                conn.execute(
                    "INSERT INTO dream_runs "
                    "(run_id, request_id, state, scope_json, manifest_json, "
                    "manifest_hash, semantic_hash, checkpoint, error_code, "
                    "failure_reason, enrichment_pending, created_at, updated_at) "
                    "VALUES (?, ?, 'rejected', ?, '{}', '', '', '', "
                    "'no_candidates', NULL, 0, ?, ?)",
                    (run_id, request_id, _canonical_json(scope), now,
                     _now_iso()),
                )
                return _load_run(beam, run_id)  # type: ignore[return-value]

            bound_err = _enforce_bounds(actions, scope)
            if bound_err is not None:
                conn.execute(
                    "INSERT INTO dream_runs "
                    "(run_id, request_id, state, scope_json, manifest_json, "
                    "manifest_hash, semantic_hash, checkpoint, error_code, "
                    "failure_reason, enrichment_pending, created_at, updated_at) "
                    "VALUES (?, ?, 'rejected', ?, '{}', '', '', '', "
                    "?, NULL, 0, ?, ?)",
                    (run_id, request_id, _canonical_json(scope), bound_err,
                     now, _now_iso()),
                )
                return _load_run(beam, run_id)  # type: ignore[return-value]

            # CAS claim: only claim proposals still in 'proposed' status. The
            # BEGIN IMMEDIATE write lock serializes concurrent plans, but the
            # CAS guard + row-count check is defense-in-depth for any path
            # that claims outside this transaction. If a competitor claimed
            # first (or the proposal was rolled_back between gather and
            # claim), the row count will be less than expected -> no_candidates.
            if consumed_ids:
                placeholders = ",".join("?" for _ in consumed_ids)
                cursor = conn.execute(
                    f"UPDATE shmr_proposals SET status = 'dream_claimed' "
                    f"WHERE proposal_id IN ({placeholders}) "
                    f"AND status IN ("
                    + ",".join("?" for _ in ELIGIBLE_PROPOSAL_STATUSES)
                    + ")",
                    tuple(consumed_ids) + tuple(ELIGIBLE_PROPOSAL_STATUSES),
                )
                if cursor.rowcount != len(consumed_ids):
                    # A competitor claimed at least one proposal between our
                    # gather and claim (or the proposal was concurrently
                    # invalidated). Abort as no_candidates -- the caller can
                    # re-plan once new proposals are available.
                    conn.execute(
                        "INSERT INTO dream_runs "
                        "(run_id, request_id, state, scope_json, manifest_json, "
                        "manifest_hash, semantic_hash, checkpoint, error_code, "
                        "failure_reason, enrichment_pending, created_at, "
                        "updated_at) "
                        "VALUES (?, ?, 'rejected', ?, '{}', '', '', '', "
                        "'no_candidates', 'proposal claimed by another run', "
                        "0, ?, ?)",
                        (run_id, request_id, _canonical_json(scope), now,
                         _now_iso()),
                    )
                    return _load_run(beam, run_id)  # type: ignore[return-value]

            config_snap = _config_snapshot()
            manifest = _build_manifest(run_id, scope, actions, config_snap)
            manifest_hash = manifest["manifest_hash"]

            # Persist the run row + actions in the SAME transaction as the
            # claim. A failure rolls everything back, leaving proposals
            # eligible.
            conn.execute(
                "INSERT INTO dream_runs "
                "(run_id, request_id, state, scope_json, manifest_json, "
                "manifest_hash, semantic_hash, checkpoint, error_code, "
                "failure_reason, enrichment_pending, created_at, updated_at) "
                "VALUES (?, ?, 'awaiting_approval', ?, ?, ?, '', '', NULL, "
                "NULL, 0, ?, ?)",
                (run_id, request_id, _canonical_json(scope),
                 _canonical_json(manifest), manifest_hash, now, _now_iso()),
            )
            for i, a in enumerate(actions):
                conn.execute(
                    "INSERT INTO dream_actions "
                    "(run_id, seq, source_table, source_id, source_hash, "
                    "source_producer, action, target_json, before_image, "
                    "after_image, applied, undone) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, 0, 0)",
                    (
                        run_id, i + 1, a["source_table"], a["source_id"],
                        a["source_hash"], a.get("source_producer"),
                        a["action"], _canonical_json(a["target"]),
                    ),
                )
    except sqlite3.OperationalError as exc:
        msg = str(exc).lower()
        if "locked" in msg or "busy" in msg:
            # A concurrent planner holds the write lock; surface a retryable
            # structured failure rather than silently double-consuming.
            code = "database_busy"
            state = "failed_retryable"
        else:
            code = "integrity_failure"
            state = "failed_terminal"
        conn.execute(
            "INSERT OR IGNORE INTO dream_runs "
            "(run_id, request_id, state, scope_json, manifest_json, "
            "manifest_hash, semantic_hash, checkpoint, error_code, "
            "failure_reason, enrichment_pending, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, '{}', '', '', '', ?, ?, 0, ?, ?)",
            (run_id, request_id, state, _canonical_json(scope), code,
             code, now, _now_iso()),
        )
        conn.commit()
        return _load_run(beam, run_id)  # type: ignore[return-value]

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
    # Reject timezone-naive timestamps: a naive datetime would crash the
    # comparison below with TypeError (can't compare offset-naive and
    # offset-aware). Treat as malformed, not a crash.
    if when.tzinfo is None:
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
    rejection = _reject_if_caller_holds_transaction(beam, run_id)
    if rejection is not None:
        return rejection
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

    ``create`` / ``update`` / ``dampen`` affect ``canonical_facts`` only. The
    standard ``facts`` table has no supersession columns, so supersession is a
    canonical_facts concept. ``canonical_facts`` has no FTS/vector index
    (canonical.py:109-143), so there is no trigger or vector write to keep in
    sync -- Dream writes canonical_facts directly via supersede-by-
    ``valid_until`` + INSERT.

    C1 (exact undo): the before-image persisted on the action is the EXACT
    canonical slot row that existed immediately before this action (or ``None``
    if no prior current row existed). Undo uses that to restore the prior row
    to its exact prior state (``valid_until=NULL``, prior body/version/source/
    confidence), so a pre-existing value is never left superseded.
    """
    target = action["target"]
    kind = action["action"]

    # The semantic output is a canonical_facts slot for this scope.
    category = "dream"
    name = f"{target.get('subject') or 'subject'}::{target.get('predicate') or 'predicate'}"
    body = target.get("object") or ""
    confidence = float(target.get("confidence") or 0.5)

    if kind == "dampen":
        # Dampen lowers confidence on the canonical slot if present; we still
        # record a canonical_facts row marking the dampened state.
        confidence = max(0.0, confidence * 0.5)

    # Capture the EXACT canonical before-image BEFORE any mutation. This is
    # what undo restores. ``None`` means no prior current row existed.
    prior_row = conn.execute(
        "SELECT * FROM canonical_facts "
        "WHERE owner_id = ? AND category = ? AND name = ? "
        "AND valid_until IS NULL",
        (owner_id, category, name),
    ).fetchone()
    canonical_before = dict(prior_row) if prior_row is not None else None

    # Use direct INSERT into canonical_facts (not CanonicalStore.remember,
    # which opens its own BEGIN IMMEDIATE). We are already inside the Dream-
    # owned deferred-commit transaction; supersede manually.
    now = _now_iso()
    if canonical_before is not None:
        conn.execute(
            "UPDATE canonical_facts SET valid_until = ? WHERE id = ?",
            (now, canonical_before["id"]),
        )
        version = (canonical_before["version"] or 0) + 1
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
            _canonical_json(canonical_before) if canonical_before is not None else None,
            _canonical_json(after_image), run_id, seq,
        ),
    )


def _check_approval_freshness(
    conn: sqlite3.Connection, run_id: str,
) -> Optional[str]:
    """Re-check that the run's PASS receipts are still within the approval TTL.

    Called from :func:`dream_apply` immediately before mutation, inside the
    apply transaction. Returns ``stale_manifest`` if any PASS receipt is now
    older than APPROVAL_TTL_HOURS, else ``None``.
    """
    rows = conn.execute(
        "SELECT timestamp FROM dream_receipts "
        "WHERE run_id = ? AND verdict = 'PASS'", (run_id,)
    ).fetchall()
    if not rows:
        return "stale_manifest"
    now = datetime.now(timezone.utc)
    for r in rows:
        try:
            when = datetime.fromisoformat(
                r["timestamp"].replace("Z", "+00:00")
            )
        except (ValueError, AttributeError):
            return "stale_manifest"
        if when.tzinfo is None or when < now - RECEIPT_TTL:
            return "stale_manifest"
    return None


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
    rejection = _reject_if_caller_holds_transaction(beam, run_id)
    if rejection is not None:
        return rejection
    conn = beam.conn
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
    # correctness backstop. If the gate cannot be set, surface a structured
    # failure rather than proceeding fail-open (sleep would race Dream).
    gate_err = _set_dream_active(True)
    if gate_err is not None:
        _set_state(conn, run_id, "failed_retryable", error_code=gate_err,
                   failure_reason="dream_active gate could not be set")
        conn.commit()
        return _load_run(beam, run_id)  # type: ignore[return-value]

    owner_id = run.scope.get("session_id") or beam.session_id

    try:
        with _deferred_commits(conn):
            # I4: acquire a write lock (BEGIN IMMEDIATE) BEFORE source
            # revalidation so a concurrent writer cannot mutate a source
            # between the check and the semantic write. _deferred_commits
            # suppresses the inner commit() that BEGIN IMMEDIATE would
            # otherwise trigger via the autocommit boundary; the transaction
            # is committed once at the end of the block.
            conn.execute("BEGIN IMMEDIATE")
            stale = _revalidate_sources(conn, run_id)
            if stale is not None:
                # Roll back via the context manager by raising.
                raise _ApplyAborted(stale)
            # I1: re-check approval TTL at apply time. A ready run whose
            # newest receipt is now older than the approval TTL must not
            # apply, even if it was valid at submission. Surface as
            # stale_manifest (approval window expired).
            ttl_err = _check_approval_freshness(conn, run_id)
            if ttl_err is not None:
                raise _ApplyAborted(ttl_err)

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
                   failure_reason=code)
        conn.commit()
        _reconcile_gate_from_durable_state(beam)
        return _load_run(beam, run_id)  # type: ignore[return-value]
    except sqlite3.IntegrityError:
        # A constraint violation is not safely retryable without operator
        # intervention (e.g. a duplicate key from a logic bug).
        _set_state(conn, run_id, "failed_terminal",
                   error_code="integrity_failure",
                   failure_reason="integrity_failure")
        conn.commit()
        _reconcile_gate_from_durable_state(beam)
        return _load_run(beam, run_id)  # type: ignore[return-value]
    except Exception:
        _set_state(conn, run_id, "failed_terminal",
                   error_code="integrity_failure",
                   failure_reason="integrity_failure")
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
    rejection = _reject_if_caller_holds_transaction(beam, run_id)
    if rejection is not None:
        return rejection
    conn = beam.conn
    _init_dream_schema(conn)
    run = _load_run(beam, run_id)
    if run is None:
        return DreamRun(run_id=run_id, state="rejected",
                        error_code="validation_failed",
                        failure_reason="run not found")

    if run.state == "undone":
        # Idempotent: explicit non-silent signal that the run is already
        # undone. Brief item 5 requires a second undo to return
        # ``already_undone``; we surface it via failure_reason + checkpoint
        # rather than inventing a new state or error-taxonomy code.
        return DreamRun(
            run_id=run_id, state="undone", checkpoint="undone",
            failure_reason="already_undone",
        )
    if run.state != "applied":
        return DreamRun(
            run_id=run_id, state=run.state,
            error_code="validation_failed",
            failure_reason=f"cannot undo from state {run.state}",
        )

    gate_err = _set_dream_active(True)
    if gate_err is not None:
        _set_state(conn, run_id, "failed_retryable", error_code=gate_err,
                   failure_reason="dream_active gate could not be set")
        conn.commit()
        return _load_run(beam, run_id)  # type: ignore[return-value]

    try:
        with _deferred_commits(conn):
            # I4: hold the write lock across undo so the before-image restore
            # is atomic against concurrent writers.
            conn.execute("BEGIN IMMEDIATE")
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
                # C1: restore the EXACT canonical before-image. If a prior
                # current row was superseded by this action, make it current
                # again by clearing valid_until. If no prior row existed
                # (before_image is None), there is nothing to restore -- the
                # slot was brand-new and the DELETE above fully reverses it.
                before = json.loads(a["before_image"]) if a["before_image"] else None
                if before is not None:
                    conn.execute(
                        "UPDATE canonical_facts SET valid_until = NULL "
                        "WHERE id = ?",
                        (before["id"],),
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
        _set_state(conn, run_id, "applied", error_code=code,
                   failure_reason=code)
        conn.commit()
        _reconcile_gate_from_durable_state(beam)
        return _load_run(beam, run_id)  # type: ignore[return-value]
    except Exception:
        _set_state(conn, run_id, "applied", error_code="integrity_failure",
                   failure_reason="integrity_failure")
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
    rejection = _reject_if_caller_holds_transaction(beam, run_id)
    if rejection is not None:
        return rejection
    _init_dream_schema(beam.conn)
    run = dream_status(beam, run_id)
    if run.state in ("applied", "undone"):
        return run
    if run.state == "failed_retryable":
        # M7: do not endlessly retry stale_manifest. A stale source will not
        # heal on retry; the operator must re-plan after fixing the source.
        # Other failed_retryable codes (database_busy, transient) are safe to
        # retry once.
        if run.error_code == "stale_manifest":
            return run
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
    rejection = _reject_if_caller_holds_transaction(beam, run_id)
    if rejection is not None:
        return rejection
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
