"""
Native Inhale: durable, receipt-backed ingest for Mnemosyne.

The existing ``BeamMemory.remember()`` contract is untouched; these APIs are
additive. Every ingest event is an idempotency key: the raw memory row(s),
the ``ingest_receipts`` row, and the sync event commit in one SQLite
transaction, then indexing/enrichment runs *outside* that transaction and the
receipt moves from ``pending`` to ``ready`` / ``degraded`` /
``failed_retryable`` / ``failed_terminal``.

A crash after the raw commit leaves a durable ``pending`` receipt that
``retry_pending_ingest()`` can re-run without duplicating memory.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from mnemosyne.core import beam as _beam_mod
from mnemosyne.core.sync import _parse_sync_timestamp
from mnemosyne.core.filters import (
    classify_memory_write,
    get_write_classifier_mode,
    redact_memory_artifact,
)

logger = logging.getLogger(__name__)

ROLES = frozenset({"user", "assistant", "tool", "system"})


class _InhaleTransactionError(RuntimeError):
    """Caller-owned/deferred transaction context is unsupported.

    The Inhale APIs own a short atomic transaction and must not (a) leave a
    ``_defer_commit`` flag set across network enrichment, or (b) roll back a
    transaction the caller opened. Detect and reject this before any
    mutation so the caller's context is preserved untouched.
    """


def _iso_from_epoch(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def _now_epoch() -> float:
    return datetime.now(timezone.utc).timestamp()


MAX_ATTEMPTS = 5
MAX_CONTENT_CHARS = 1_000_000
MAX_FIELD_CHARS = 255
MAX_METADATA_BYTES = 128 * 1024
RETRYABLE_INDEX_STATES = ("pending", "failed_retryable", "degraded")
# A retry worker holds a claim for at most this long; stale claims older than
# this are reclaimed deterministically so a crashed worker cannot strand a
# receipt. Keep short enough that enrichment never overlaps a DB transaction.
CLAIM_LEASE_SECONDS = 60.0
_MEMORIA_SOURCE_TABLES = (
    "memoria_facts",
    "memoria_timelines",
    "memoria_kg",
    "memoria_instructions",
    "memoria_preferences",
)

# Serializes SyncEngine construction per process so two threads ingesting
# into the same fresh DB cannot race the engine's additive ALTER TABLEs.
_ENGINE_LOCK = threading.Lock()


@dataclass(frozen=True)
class IngestEvent:
    """Trust-boundary ingest payload with a stable external event id."""

    event_id: str
    producer: str
    actor_id: str
    project_id: str
    session_id: str
    turn_id: str
    role: str
    content: str
    content_hash: str
    occurred_at: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class TurnEvent(IngestEvent):
    """An ingest event that is explicitly one turn in a session."""


@dataclass
class IngestReceipt:
    """Structured outcome of one ingest attempt."""

    event_id: str
    payload_hash: str
    memory_ids: List[str]
    status: str
    index_status: str
    attempts: int
    last_error_code: Optional[str] = None
    last_error_at: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RetryReport:
    """Counts of one ``retry_pending_ingest`` pass."""

    attempted: int = 0
    succeeded: int = 0
    degraded: int = 0
    failed_retryable: int = 0
    failed_terminal: int = 0
    receipts: List[IngestReceipt] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public APIs (module level; BeamMemory mirrors these as methods)
# ---------------------------------------------------------------------------


def remember_event(beam, event: IngestEvent) -> IngestReceipt:
    """Durably ingest one event and return its receipt.

    ``status``: stored | duplicate | conflict | rejected.
    ``index_status``: pending while indexing, then ready / degraded /
    failed_retryable / failed_terminal.
    """
    if not isinstance(event, IngestEvent):
        raise TypeError("event must be an IngestEvent")
    errors = _validate_event(event)
    if errors:
        return _reject(event, errors)

    # Native admission: classify before any payload hashing, dedup lookup,
    # transaction, or memory write. Secrets are always rejected with zero
    # durable writes; CoT/approval artifacts are canonicalized (warn),
    # rejected (strict), or unchanged (off). On reject, return a
    # content-free receipt WITHOUT entering a transaction or writing any row.
    admitted, admit_errors, admit_labels, admission_marker = _admit_event(event)
    if admit_errors:
        return _reject(
            event,
            admit_errors,
            error_code="admission_rejected",
            labels=admit_labels,
        )
    event = admitted

    payload_hash = _payload_hash(event)
    conn = beam.conn
    _assert_inhale_transaction_context(conn)
    engine = _get_sync_engine(beam)
    now = _now_iso()
    provenance = _provenance(event)
    receipt_meta = json.dumps(
        {"_ingest": provenance, **(admission_marker or {})},
        sort_keys=True,
        default=str,
    )

    _begin_write(conn)
    try:
        row = conn.execute(
            "SELECT * FROM ingest_receipts WHERE event_id = ?", (event.event_id,)
        ).fetchone()
        if row is not None:
            if row["payload_hash"] == payload_hash:
                # Identical payload replay: the original is already stored.
                # Return the ORIGINAL receipt (its truthful index state,
                # e.g. ready/pending) surfaced as a duplicate. The dedup
                # authority is payload_hash, which the conflict path never
                # touches.
                conn.rollback()
                return _receipt_from_row(row, status="duplicate")
            # Event id reused with a different payload: structured conflict.
            # The original receipt row is the indexing-lifecycle authority
            # and is NOT mutated -- not status, not index_status, not
            # memory_ids, not payload_hash, not attempts, not error fields.
            # Conflict observability is a separate durable audit trail so a
            # pending/failed_retryable original stays exactly as retryable
            # as before the conflict.
            conn.execute(
                """INSERT INTO ingest_conflicts
                   (event_id, stored_payload_hash, conflicting_payload_hash,
                    observed_at)
                   VALUES (?, ?, ?, ?)""",
                (event.event_id, row["payload_hash"], payload_hash, now),
            )
            conn.commit()
            logger.warning(
                "ingest conflict event_id=%r: stored payload_hash=%s, "
                "conflicting payload_hash=%s (original receipt untouched)",
                event.event_id,
                row["payload_hash"],
                payload_hash,
            )
            # Synthesize the conflict outcome WITHOUT persisting it on the
            # original lifecycle row.
            return _conflict_receipt(row, payload_hash, now)

        memory_id = _memory_id_for_event(event.event_id)
        metadata_json = json.dumps(
            _memory_metadata(event, admission_marker), sort_keys=True, default=str
        )
        conn.execute(
            """INSERT INTO working_memory
               (id, content, source, timestamp, session_id, importance,
                metadata_json, veracity, memory_type, trust_tier,
                author_id, author_type, channel_id, scope)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                memory_id,
                event.content,
                event.producer,
                event.occurred_at,
                event.session_id,
                0.5,
                metadata_json,
                "unknown",
                None,
                _beam_mod._source_to_trust_tier(event.producer),
                event.actor_id,
                event.producer,
                event.project_id,
                "session",
            ),
        )
        conn.execute(
            """INSERT INTO ingest_receipts
               (event_id, payload_hash, memory_ids, status, index_status,
                attempts, last_error_code, last_error_at, created_at,
                updated_at, metadata_json)
               VALUES (?, ?, ?, 'stored', 'pending', 1, NULL, NULL, ?, ?, ?)""",
            (
                event.event_id,
                payload_hash,
                json.dumps([memory_id]),
                now,
                now,
                receipt_meta,
            ),
        )
        engine.log_event(
            memory_id,
            "CREATE",
            payload=_sync_payload(event, metadata_json),
            commit=False,
        )
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except sqlite3.Error:
            pass
        raise

    try:
        index_status, error_code = _index_memory(
            beam, memory_id, event.content, event.producer, event.occurred_at
        )
        _finalize_receipt(beam, event.event_id, index_status, error_code, 1)
    except Exception:
        # Crash window: keep the receipt retryable and stay loud. If the
        # finalize itself dies the receipt stays 'pending' and retry recovers.
        try:
            _finalize_receipt(
                beam, event.event_id, "failed_retryable", "indexing_failed", 1
            )
        except Exception:
            pass
        raise

    beam._invalidate_query_cache_after_remember_commit()
    return _receipt_from_row(
        conn.execute(
            "SELECT * FROM ingest_receipts WHERE event_id = ?", (event.event_id,)
        ).fetchone()
    )


def remember_turn(beam, turn: TurnEvent) -> IngestReceipt:
    """Durably ingest one turn (an event carrying a turn id)."""
    if not isinstance(turn, TurnEvent):
        raise TypeError("turn must be a TurnEvent")
    return remember_event(beam, turn)


def remember_turns_atomic(beam, turns: List["TurnEvent"]) -> List[IngestReceipt]:
    """Durably ingest multiple turns atomically in one SQLite transaction.

    All turns whose validation passes are inserted (working_memory row +
    ingest_receipts row + sync event) inside a single transaction. If any
    insert fails, the entire batch is rolled back -- no partial state is
    left durable. Validation rejections are returned as ``rejected`` receipts
    without entering the transaction (a rejected event has no partial state,
    matching the single-event contract).

    Indexing/enrichment runs per-event outside the atomic transaction after
    the commit, so a vector-store failure leaves the receipt retryable
    (``pending`` / ``failed_retryable``) but does not un-roll the durable
    memory row.

    Callers that need true multi-event atomicity (e.g. a Hermes sync_turn
    with both user+assistant roles available) should use this instead of
    calling ``remember_turn`` in a loop, which commits each event separately
    and cannot roll back a first-side event when the second side fails.
    """
    if not isinstance(turns, list):
        raise TypeError("turns must be a list")
    for turn in turns:
        if not isinstance(turn, TurnEvent):
            raise TypeError("each turn must be a TurnEvent")

    # Phase 1: validate all events. Rejected events return a rejected
    # receipt at their original input position and are skipped -- they
    # never enter the transaction (a corrected resubmission with the same
    # event id must be able to succeed, so a rejected key is never burned).
    # receipts is sized to len(turns) so every position maps 1:1 to an
    # input turn; a rejected receipt can never be overwritten or suppressed.
    valid_turns: List[tuple] = []
    receipts: List[Optional[IngestReceipt]] = [None] * len(turns)
    for i, turn in enumerate(turns):
        errors = _validate_event(turn)
        if errors:
            receipts[i] = _reject(turn, errors)
            continue
        # Native admission runs in the phase-1 validation pass, before the
        # transaction: rejected events never enter it (their event id stays
        # reusable), and canonicalized events are what gets hashed/deduped.
        admitted, admit_errors, admit_labels, admission_marker = _admit_event(turn)
        if admit_errors:
            receipts[i] = _reject(
                turn,
                admit_errors,
                error_code="admission_rejected",
                labels=admit_labels,
            )
        else:
            valid_turns.append((i, admitted, admission_marker))

    if not valid_turns:
        return receipts

    conn = beam.conn
    _assert_inhale_transaction_context(conn)
    engine = _get_sync_engine(beam)
    now = _now_iso()
    # Track the memory rows + positions so indexing runs after commit.
    pending_index: List[tuple] = []

    _begin_write(conn)
    try:
        for idx, turn, admission_marker in valid_turns:
            payload_hash = _payload_hash(turn)
            row = conn.execute(
                "SELECT * FROM ingest_receipts WHERE event_id = ?",
                (turn.event_id,),
            ).fetchone()
            if row is not None:
                if row["payload_hash"] == payload_hash:
                    # Idempotent replay: record the duplicate receipt at this
                    # position; nothing to insert.
                    receipts[idx] = _receipt_from_row(row, status="duplicate")
                    continue
                # Conflict: record it in the audit trail but do NOT mutate
                # the original lifecycle row.
                conn.execute(
                    """INSERT INTO ingest_conflicts
                       (event_id, stored_payload_hash, conflicting_payload_hash,
                        observed_at)
                       VALUES (?, ?, ?, ?)""",
                    (turn.event_id, row["payload_hash"], payload_hash, now),
                )
                logger.warning(
                    "ingest conflict event_id=%r: stored payload_hash=%s, "
                    "conflicting payload_hash=%s (original receipt untouched)",
                    turn.event_id,
                    row["payload_hash"],
                    payload_hash,
                )
                receipts[idx] = _conflict_receipt(row, payload_hash, now)
                continue

            memory_id = _memory_id_for_event(turn.event_id)
            metadata_json = json.dumps(
                _memory_metadata(turn, admission_marker), sort_keys=True, default=str
            )
            conn.execute(
                """INSERT INTO working_memory
                   (id, content, source, timestamp, session_id, importance,
                    metadata_json, veracity, memory_type, trust_tier,
                    author_id, author_type, channel_id, scope)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    memory_id,
                    turn.content,
                    turn.producer,
                    turn.occurred_at,
                    turn.session_id,
                    0.5,
                    metadata_json,
                    "unknown",
                    None,
                    _beam_mod._source_to_trust_tier(turn.producer),
                    turn.actor_id,
                    turn.producer,
                    turn.project_id,
                    "session",
                ),
            )
            conn.execute(
                """INSERT INTO ingest_receipts
                   (event_id, payload_hash, memory_ids, status, index_status,
                    attempts, last_error_code, last_error_at, created_at,
                    updated_at, metadata_json)
                   VALUES (?, ?, ?, 'stored', 'pending', 1, NULL, NULL, ?, ?, ?)""",
                (
                    turn.event_id,
                    payload_hash,
                    json.dumps([memory_id]),
                    now,
                    now,
                    json.dumps(
                        {
                            "_ingest": _provenance(turn),
                            **(admission_marker or {}),
                        },
                        sort_keys=True,
                        default=str,
                    ),
                ),
            )
            engine.log_event(
                memory_id,
                "CREATE",
                payload=_sync_payload(turn, metadata_json),
                commit=False,
            )
            pending_index.append((idx, turn, memory_id))
            receipts[idx] = None  # placeholder; filled after commit
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except sqlite3.Error:
            pass
        raise

    # Phase 2: index/enrich each newly-stored receipt outside the atomic
    # transaction. A failure here leaves the receipt retryable (pending /
    # failed_retryable) but does NOT un-roll the durable row.
    for pos, turn, memory_id in pending_index:
        try:
            index_status, error_code = _index_memory(
                beam,
                memory_id,
                turn.content,
                turn.producer,
                turn.occurred_at,
            )
            _finalize_receipt(beam, turn.event_id, index_status, error_code, 1)
        except Exception:
            try:
                _finalize_receipt(
                    beam,
                    turn.event_id,
                    "failed_retryable",
                    "indexing_failed",
                    1,
                )
            except Exception:
                pass
            raise
        receipts[pos] = _receipt_from_row(
            conn.execute(
                "SELECT * FROM ingest_receipts WHERE event_id = ?",
                (turn.event_id,),
            ).fetchone()
        )

    beam._invalidate_query_cache_after_remember_commit()
    return receipts


def retry_pending_ingest(beam, limit: int = 100) -> RetryReport:
    """Re-run indexing/enrichment for stored receipts that are not ready.

    Never re-inserts the raw memory row or the sync event: the receipt is the
    authority, and enrichment re-runs are idempotent (MEMORIA rows keyed by
    source_memory_id are reset before re-extraction; graph/annotation writes
    are INSERT OR REPLACE / INSERT OR IGNORE).

    Concurrency: each receipt is claimed via an atomic lease before
    enrichment so two workers cannot duplicate work. Claims are acquired and
    released in short SQLite transactions; enrichment (network/embedding)
    never runs while a transaction is open. Stale claims (crashed worker /
    expired lease) are reclaimed deterministically. Release and finalize are
    ownership-guarded by claim_worker_id, so a worker whose lease was
    reclaimed cannot clear or overwrite the new owner's claim.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    conn = beam.conn
    _assert_inhale_transaction_context(conn)
    now = _now_epoch()
    now_iso = _iso_from_epoch(now)
    worker_id = f"{threading.get_ident()}:{now}:{_event_id_token()}"
    placeholders = ",".join("?" * len(RETRYABLE_INDEX_STATES))
    # Select candidates outside any transaction; the claim is the real gate.
    candidates = conn.execute(
        f"""SELECT event_id, memory_ids, attempts, payload_hash
            FROM ingest_receipts
            WHERE status = 'stored'
              AND index_status IN ({placeholders})
            ORDER BY created_at
            LIMIT ?""",
        (*RETRYABLE_INDEX_STATES, limit),
    ).fetchall()
    report = RetryReport()
    for row in candidates:
        event_id = row["event_id"]
        attempts = int(row["attempts"] or 0)

        # Claim the receipt in a short transaction. The atomic UPDATE ensures
        # only one worker wins; a stale lease (older than now) is reclaimable.
        lease_iso = _iso_from_epoch(now + CLAIM_LEASE_SECONDS)
        claimed = _try_claim(conn, event_id, worker_id, lease_iso, now_iso)
        if not claimed:
            continue  # Another worker owns it (live claim).

        try:
            if attempts >= MAX_ATTEMPTS:
                if not _finalize_claimed(
                    conn,
                    event_id,
                    "failed_terminal",
                    "max_attempts_exceeded",
                    attempts,
                    release=True,
                    worker_id=worker_id,
                ):
                    continue  # Claim was reclaimed; the new owner finalizes.
                report.attempted += 1
                report.failed_terminal += 1
                report.receipts.append(_final_row(conn, event_id))
                continue

            memory_rows = _load_memory_rows(conn, row["memory_ids"])
            if not memory_rows:
                if not _finalize_claimed(
                    conn,
                    event_id,
                    "failed_terminal",
                    "memory_row_missing",
                    attempts + 1,
                    release=True,
                    worker_id=worker_id,
                ):
                    continue  # Claim was reclaimed; the new owner finalizes.
                report.attempted += 1
                report.failed_terminal += 1
                report.receipts.append(_final_row(conn, event_id))
                continue

            final_status, final_code = "ready", None
            severity = {
                "ready": 0,
                "degraded": 1,
                "failed_retryable": 2,
                "failed_terminal": 3,
            }
            for memory_row in memory_rows:
                status, code = _index_memory(
                    beam,
                    memory_row["id"],
                    memory_row["content"],
                    memory_row["source"],
                    memory_row["timestamp"],
                )
                # Worst state wins so the receipt never overclaims.
                if severity[status] > severity[final_status]:
                    final_status, final_code = status, code

            if not _finalize_claimed(
                conn,
                event_id,
                final_status,
                final_code,
                attempts + 1,
                release=True,
                worker_id=worker_id,
            ):
                continue  # Claim was reclaimed; the new owner finalizes.
        except Exception:
            # Crash during enrichment: release the claim so the receipt is
            # not stranded (its index_status stays as-is, still retryable).
            # Ownership-guarded: if the claim was reclaimed meanwhile, this is
            # a no-op and the new owner's claim stays intact.
            try:
                _release_claim(conn, event_id, worker_id)
            except Exception:
                pass
            raise

        report.attempted += 1
        if final_status == "ready":
            report.succeeded += 1
        elif final_status == "degraded":
            report.degraded += 1
        elif final_status == "failed_retryable":
            report.failed_retryable += 1
        else:
            report.failed_terminal += 1
        report.receipts.append(_final_row(conn, event_id))
    return report


def ingest_status(
    beam, event_id: Optional[str] = None, limit: int = 100
) -> List[IngestReceipt]:
    """Return content-free receipt data for stored ingest events.

    One stable idempotency surface for status, diagnostics, and doctor: reads
    only the ingest_receipts table and returns IngestReceipt rows with
    NO content field (trust boundary: never surface raw event content through a
    status/audit path). event_id returns at most that one receipt; limit
    bounds the scan (positive int, validated at the trust boundary).

    Receipts are returned newest-first by created_at so callers see the most
    recent ingest state without an unbounded scan.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    conn = beam.conn
    if event_id is not None:
        row = conn.execute(
            "SELECT * FROM ingest_receipts WHERE event_id = ?", (event_id,)
        ).fetchone()
        return [_receipt_from_row(row)] if row is not None else []
    rows = conn.execute(
        "SELECT * FROM ingest_receipts ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [_receipt_from_row(r) for r in rows]


def _try_claim(
    conn: sqlite3.Connection,
    event_id: str,
    worker_id: str,
    lease_iso: str,
    now_iso: str,
) -> bool:
    """Atomically claim a receipt for this worker in a short transaction.

    Wins only if the receipt is unclaimed or its lease has expired (stale
    claim from a crashed worker). Returns True if this worker now owns it.
    """
    placeholders = ",".join("?" * len(RETRYABLE_INDEX_STATES))
    _begin_write(conn)
    try:
        # Re-check the lifecycle atomically at claim time to close the TOCTOU
        # gap: candidate selection runs before this transaction, so a delayed
        # worker could reach the claim AFTER another worker already finalized
        # the receipt to 'ready'/'failed_terminal'. The claim UPDATE must
        # therefore require status='stored' AND a still-retryable index_status
        # in addition to lease availability, so the late worker neither claims,
        # enriches, nor overwrites the truthful terminal state.
        cur = conn.execute(
            f"""UPDATE ingest_receipts
               SET claim_worker_id = ?,
                   claim_worker_lease = ?
               WHERE event_id = ?
                 AND status = 'stored'
                 AND index_status IN ({placeholders})
                 AND (
                   claim_worker_id IS NULL
                   OR claim_worker_lease IS NULL
                   OR claim_worker_lease < ?
                 )""",
            (worker_id, lease_iso, event_id, *RETRYABLE_INDEX_STATES, now_iso),
        )
        conn.commit()
        return cur.rowcount == 1
    except Exception:
        try:
            conn.rollback()
        except sqlite3.Error:
            pass
        raise


def _release_claim(conn: sqlite3.Connection, event_id: str, worker_id: str) -> None:
    """Clear THIS worker's claim (ownership-guarded CAS).

    A worker whose lease was reclaimed by another worker must not clear the
    new owner's live claim; the WHERE clause makes the stale release a no-op.
    """
    _begin_write(conn)
    try:
        conn.execute(
            """UPDATE ingest_receipts
               SET claim_worker_id = NULL,
                   claim_worker_lease = NULL
               WHERE event_id = ? AND claim_worker_id = ?""",
            (event_id, worker_id),
        )
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except sqlite3.Error:
            pass
        raise


def _finalize_claimed(
    conn: sqlite3.Connection,
    event_id: str,
    index_status: str,
    error_code: Optional[str],
    attempts: int,
    *,
    release: bool,
    worker_id: str,
) -> bool:
    """Move THIS worker's claimed receipt to its terminal/retryable state and
    release the claim in one short transaction.

    Ownership-guarded CAS: the UPDATE applies only while the row is still
    claimed by ``worker_id``. Returns True if this worker still owned the
    claim (state was written); False if another worker reclaimed it, so the
    caller must not report a receipt it no longer owns.
    """
    now = _now_iso()
    _begin_write(conn)
    try:
        cur = conn.execute(
            """UPDATE ingest_receipts
               SET index_status = ?,
                   attempts = ?,
                   last_error_code = ?,
                   last_error_at = ?,
                   updated_at = ?"""
            + (", claim_worker_id = NULL, claim_worker_lease = NULL" if release else "")
            + """ WHERE event_id = ? AND claim_worker_id = ?""",
            (
                index_status,
                attempts,
                error_code,
                now if error_code else None,
                now,
                event_id,
                worker_id,
            ),
        )
        conn.commit()
        return cur.rowcount == 1
    except Exception:
        try:
            conn.rollback()
        except sqlite3.Error:
            pass
        raise


def _event_id_token() -> str:
    """Short unique token for worker ids."""
    return hashlib.sha256(_now_iso().encode("utf-8")).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Validation (trust boundary)
# ---------------------------------------------------------------------------


def _validate_event(event: IngestEvent) -> List[str]:
    errors: List[str] = []
    for field_name in (
        "event_id",
        "producer",
        "actor_id",
        "project_id",
        "session_id",
        "turn_id",
        "role",
        "content",
        "content_hash",
        "occurred_at",
    ):
        value = getattr(event, field_name)
        if not isinstance(value, str) or not value:
            errors.append(f"{field_name}: required")
            continue
        limit = MAX_CONTENT_CHARS if field_name == "content" else MAX_FIELD_CHARS
        if len(value) > limit:
            errors.append(f"{field_name}: exceeds {limit} characters")
    if errors:
        return errors

    if event.role not in ROLES:
        errors.append(f"role: must be one of {sorted(ROLES)}")
    for field_name in (
        "event_id",
        "producer",
        "actor_id",
        "project_id",
        "session_id",
        "turn_id",
    ):
        if any(ord(c) < 32 or ord(c) == 127 for c in getattr(event, field_name)):
            errors.append(f"{field_name}: control characters not allowed")
    try:
        digest = hashlib.sha256(event.content.encode("utf-8")).hexdigest()
        if digest.lower() != event.content_hash.lower():
            errors.append("content_hash: does not match sha256(content)")
    except UnicodeEncodeError:
        errors.append("content: not valid UTF-8")
    try:
        _parse_sync_timestamp(event.occurred_at)
    except (TypeError, ValueError):
        errors.append("occurred_at: invalid ISO timestamp")

    metadata = event.metadata
    if metadata is not None:
        if not isinstance(metadata, dict):
            errors.append("metadata: must be a dict or None")
        else:
            try:
                raw = json.dumps(metadata, sort_keys=True, default=str)
            except Exception:
                # Trust boundary: a hostile __str__ must become 'rejected',
                # not an unhandled exception.
                errors.append("metadata: not JSON-serializable")
            else:
                if len(raw.encode("utf-8")) > MAX_METADATA_BYTES:
                    errors.append(f"metadata: exceeds {MAX_METADATA_BYTES} bytes")
    return errors


def _reject(
    event: IngestEvent,
    errors: List[str],
    *,
    error_code: str = "validation_failed",
    labels: Optional[List[str]] = None,
) -> IngestReceipt:
    """Structured rejection at the trust boundary.

    Deliberately NOT persisted: a rejected event has no partial state, and a
    corrected resubmission with the same stable event id must be able to
    succeed (burning the key on a rejected payload would turn every producer
    fix into a permanent conflict).
    """
    try:
        payload_hash = _payload_hash(event)
    except Exception:
        payload_hash = hashlib.sha256(
            str(getattr(event, "event_id", "")).encode("utf-8", "replace")
        ).hexdigest()
    logger.error(
        "ingest rejected event_id=%r (%s): %s",
        getattr(event, "event_id", None),
        "; ".join(errors),
        type(event).__name__,
    )
    metadata: Dict[str, Any] = {"errors": errors}
    if labels:
        # Label-only admission diagnostics: never the matched value or raw
        # artifact text. ``labels`` arrive from Task 1's classifier warnings,
        # which carry pattern labels (e.g. ``api_key_prefix``), not secrets.
        metadata["admission_labels"] = labels
    now = _now_iso()
    return IngestReceipt(
        event_id=getattr(event, "event_id", "") or "",
        payload_hash=payload_hash,
        memory_ids=[],
        status="rejected",
        index_status="failed_terminal",
        attempts=0,
        last_error_code=error_code,
        last_error_at=now,
        created_at=now,
        updated_at=now,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Native admission policy (Task 2)
# ---------------------------------------------------------------------------


def _admit_event(
    event: IngestEvent,
) -> Tuple[Optional[IngestEvent], List[str], List[str], Optional[Dict[str, Any]]]:
    """Classify an event at the trust boundary before any durable write.

    Returns ``(admitted_event, reject_errors, reject_labels, admission_marker)``:
      - ``admitted_event`` is the event to hash/persist. It is ``event`` when
        no rewrite is required, or a canonical redacted copy (built via
        ``dataclasses.replace``) when the classifier rewrote content and/or
        metadata. It is ``None`` when the event must be rejected.
      - ``reject_errors`` is empty unless the event must be rejected.
      - ``reject_labels`` carries Task 1 classifier warning labels
        (pattern/label only, never matched values or raw artifact text).
      - ``admission_marker`` is ``None`` or a label-only marker
        (``{"_admission": {"rewritten": reason}}``) for a warn canonicalization.
        Callers persist it in existing stored metadata / receipt metadata
        fields; it is never written into ``event.metadata``, so it cannot
        alter the payload hash or break original/redacted-replay dedupe.

    Native policy:
      - Secrets are always rejected with zero durable writes, independently of
        ``MNEMOSYNE_WRITE_CLASSIFIER``. The secret is checked in both
        ``event.content`` and the canonical serialized ``event.metadata``.
      - CoT/approval artifacts:
          ``warn``   -> canonicalize (deterministic redaction via Task 1's
                       ``redact_memory_artifact``) and proceed; the redacted
                       event is what gets hashed and deduplicated, and the
                       returned marker is persisted by callers.
          ``strict`` -> reject.
          ``off``    -> unchanged (compatibility preserved).
    """
    content_decision = classify_memory_write(event.content)

    metadata_canonical = json.dumps(
        event.metadata or {}, sort_keys=True, separators=(",", ":"), default=str
    )
    metadata_decision = classify_memory_write(metadata_canonical)

    reject_errors: List[str] = []
    reject_labels: List[str] = []

    # --- Secrets: always reject, independent of classifier mode. ---
    if content_decision.reason == "secret_detected":
        reject_errors.append("content: secret_detected")
        reject_labels.extend(content_decision.warnings)
    if metadata_decision.reason == "secret_detected":
        reject_errors.append("metadata: secret_detected")
        reject_labels.extend(metadata_decision.warnings)

    if reject_errors:
        return None, reject_errors, reject_labels, None

    # --- CoT/approval artifacts: mode-dependent handling. ---
    mode = get_write_classifier_mode()
    artifact_reasons = {"reasoning_artifact", "approval_artifact"}
    is_artifact = (
        content_decision.reason in artifact_reasons
        or metadata_decision.reason in artifact_reasons
    )

    if mode == "strict" and is_artifact:
        reason = content_decision.reason or metadata_decision.reason
        reject_errors.append(f"admission: {reason}")
        return None, reject_errors, reject_labels, None

    if mode == "warn" and is_artifact:
        # Deterministic canonicalization: the placeholder token is derived
        # from the stable reason label, never from the artifact. The
        # canonical event is what gets hashed/deduped, so an original
        # (canonicalized here) and a direct redacted replay collide as
        # duplicates. Content and metadata artifacts are redacted
        # independently; the label-only admission marker is persisted by
        # callers (never inside event.metadata, so the payload hash is
        # unchanged by the marker itself).
        marker_reason = content_decision.reason or metadata_decision.reason
        rewritten: Dict[str, Any] = {}
        if content_decision.reason in artifact_reasons:
            redacted = redact_memory_artifact(event.content, content_decision.reason)
            # Recompute content_hash so the canonical event is byte-for-byte
            # consistent: a later direct replay of the redacted form must
            # collide as a duplicate (same content, same content_hash,
            # same payload_hash), not a conflict.
            rewritten["content"] = redacted
            rewritten["content_hash"] = hashlib.sha256(
                redacted.encode("utf-8")
            ).hexdigest()
        if metadata_decision.reason in artifact_reasons:
            # Whole-metadata canonical form: one fixed label-only key holding
            # the stable redaction token, so no artifact text survives in any
            # key or value and the canonical form is deterministic.
            # ponytail: drops non-artifact metadata keys; per-value redaction
            # if preserving sibling keys ever matters.
            rewritten["metadata"] = {
                "_redacted": redact_memory_artifact(
                    metadata_canonical, metadata_decision.reason
                )
            }
        return (
            replace(event, **rewritten),
            [],
            [],
            {"_admission": {"rewritten": marker_reason}},
        )

    # off mode, or warn/strict with no artifact: unchanged.
    return event, [], [], None


# ---------------------------------------------------------------------------
# Transaction + index pipeline helpers
# ---------------------------------------------------------------------------


def _begin_write(conn: sqlite3.Connection) -> None:
    """BEGIN IMMEDIATE, recovering from a stale open transaction once."""
    try:
        conn.execute("BEGIN IMMEDIATE")
    except sqlite3.OperationalError as exc:
        if "cannot start a transaction" in str(exc).lower():
            conn.rollback()
            conn.execute("BEGIN IMMEDIATE")
        else:
            raise


def _assert_inhale_transaction_context(conn: sqlite3.Connection) -> None:
    """Reject caller-owned / deferred transaction contexts before mutation.

    The Inhale API must own its short atomic transaction. Running inside a
    deferred-commit context would leave ``_defer_commit`` set across network
    enrichment; running inside an already-open transaction would either nest
    illegally or cause the API's rollback to discard the caller's work. In
    both cases, reject loudly before touching anything.
    """
    if getattr(conn, "_defer_commit", False):
        raise _InhaleTransactionError(
            "Inhale APIs cannot run inside a deferred-commit context "
            "(_defer_commit is active); they require their own atomic "
            "transaction. Open a fresh BeamMemory or finish the deferred "
            "batch before ingesting."
        )
    if conn.in_transaction:
        raise _InhaleTransactionError(
            "Inhale APIs require their own transaction; the connection "
            "already has an open transaction. Commit or roll back the "
            "caller-owned transaction before calling ingest."
        )


def _index_memory(beam, memory_id: str, content: str, source: str, timestamp: str):
    """Run embedding + enrichment outside any ingest transaction.

    Returns (index_status, error_code). The receipt must never be moved to
    'ready' when the vector write failed, so the vector store is probed
    explicitly and sqlite-vec failures are surfaced (strict_vec=True).
    """
    conn = beam.conn
    error_code: Optional[str] = None
    if _beam_mod._embeddings.available():
        try:
            vec = _beam_mod._embeddings.embed([content])
            if vec is None or len(vec) != 1:
                raise RuntimeError("embed returned no vector")
            _beam_mod._store_working_embedding(conn, memory_id, vec[0], strict_vec=True)
        except Exception as exc:
            error_code = _classify_embedding_error(exc)
            logger.warning("inhale: embedding failed (%s)", type(exc).__name__)
            try:
                conn.rollback()
            except sqlite3.Error:
                pass
    else:
        error_code = "embedding_unavailable"

    if error_code is None and not _beam_mod._wm_vec_available(conn):
        error_code = "vector_index_unavailable"

    try:
        with _beam_mod._guarded_transaction(conn):
            _reset_memoria_extraction(conn, memory_id)
            beam.extract_and_store_facts(
                content, message_idx=0, source_memory_id=memory_id
            )
        beam._add_temporal_triple(memory_id, timestamp, source, content)
        beam._ingest_graph_and_veracity(memory_id, content, source, "unknown")
    except Exception as exc:
        error_code = error_code or "enrichment_failed"
        logger.warning("inhale: enrichment failed (%s)", type(exc).__name__)

    if error_code is None:
        return "ready", None
    if error_code == "embedding_failure":
        return "failed_retryable", error_code
    if error_code == "dimension_mismatch":
        return "failed_terminal", error_code
    return "degraded", error_code


def _classify_embedding_error(exc: Exception) -> str:
    if "dimension" in str(exc).lower():
        return "dimension_mismatch"
    return "embedding_failure"


def _reset_memoria_extraction(conn: sqlite3.Connection, memory_id: str) -> None:
    """Make MEMORIA re-extraction idempotent across retries."""
    for table in _MEMORIA_SOURCE_TABLES:
        conn.execute(f"DELETE FROM {table} WHERE source_memory_id = ?", (memory_id,))


def _finalize_receipt(
    beam, event_id: str, index_status: str, error_code: Optional[str], attempts: int
) -> None:
    """Move one receipt to its truthful terminal/retryable index state."""
    now = _now_iso()
    beam.conn.execute(
        """UPDATE ingest_receipts
           SET index_status = ?,
               attempts = ?,
               last_error_code = ?,
               last_error_at = ?,
               updated_at = ?
           WHERE event_id = ?""",
        (
            index_status,
            attempts,
            error_code,
            now if error_code else None,
            now,
            event_id,
        ),
    )
    beam.conn.commit()


def _final_row(conn: sqlite3.Connection, event_id: str) -> IngestReceipt:
    return _receipt_from_row(
        conn.execute(
            "SELECT * FROM ingest_receipts WHERE event_id = ?", (event_id,)
        ).fetchone()
    )


def _load_memory_rows(conn: sqlite3.Connection, memory_ids_json: str) -> List[Any]:
    try:
        ids = json.loads(memory_ids_json or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(ids, list):
        return []
    rows = []
    for memory_id in ids:
        row = conn.execute(
            "SELECT id, content, source, timestamp FROM working_memory WHERE id = ?",
            (str(memory_id),),
        ).fetchone()
        if row is not None:
            rows.append(row)
    return rows


def _get_sync_engine(beam) -> Any:
    """Lazily build one SyncEngine per BeamMemory (thread-local connection)."""
    engine = getattr(beam, "_inhale_sync_engine", None)
    if engine is not None:
        return engine
    with _ENGINE_LOCK:
        engine = getattr(beam, "_inhale_sync_engine", None)
        if engine is None:
            from mnemosyne.core.sync import SyncEngine

            engine = SyncEngine(beam)
            beam._inhale_sync_engine = engine
    return engine


# ---------------------------------------------------------------------------
# Payload / receipt mapping
# ---------------------------------------------------------------------------


def _payload_hash(event: IngestEvent) -> str:
    payload = {
        "producer": event.producer,
        "actor_id": event.actor_id,
        "project_id": event.project_id,
        "session_id": event.session_id,
        "turn_id": event.turn_id,
        "role": event.role,
        "content": event.content,
        "content_hash": event.content_hash,
        "occurred_at": event.occurred_at,
        "metadata": event.metadata,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _provenance(event: IngestEvent) -> Dict[str, str]:
    return {
        "event_id": event.event_id,
        "producer": event.producer,
        "actor_id": event.actor_id,
        "project_id": event.project_id,
        "session_id": event.session_id,
        "turn_id": event.turn_id,
        "role": event.role,
    }


def _memory_metadata(
    event: IngestEvent, admission_marker: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    metadata = dict(event.metadata or {})
    if admission_marker:
        metadata.update(admission_marker)
    metadata["_ingest"] = _provenance(event)
    return metadata


def _sync_payload(event: IngestEvent, metadata_json: str) -> Dict[str, Any]:
    return {
        "content": event.content,
        "source": event.producer,
        "importance": 0.5,
        "metadata_json": metadata_json,
        "memory_type": None,
        "veracity": "unknown",
        "valid_until": None,
    }


def _memory_id_for_event(event_id: str) -> str:
    # Stable across retries: one event id always maps to the same memory row.
    return hashlib.sha256(event_id.encode("utf-8")).hexdigest()[:16]


def _receipt_from_row(row: Any, status: Optional[str] = None) -> IngestReceipt:
    if row is None:
        raise RuntimeError("receipt row missing")
    try:
        memory_ids = json.loads(row["memory_ids"] or "[]")
    except json.JSONDecodeError:
        memory_ids = []
    try:
        metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else None
    except json.JSONDecodeError:
        metadata = None
    return IngestReceipt(
        event_id=row["event_id"],
        payload_hash=row["payload_hash"],
        memory_ids=memory_ids if isinstance(memory_ids, list) else [],
        status=status or row["status"],
        index_status=row["index_status"],
        attempts=int(row["attempts"] or 0),
        last_error_code=row["last_error_code"],
        last_error_at=row["last_error_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        metadata=metadata,
    )


def _conflict_receipt(row: Any, conflicting_hash: str, now: str) -> IngestReceipt:
    """Build a structured conflict outcome for THIS attempt without
    persisting conflict bookkeeping on the original lifecycle row.

    The returned receipt reports the conflict (status='conflict') but keeps
    the original's payload_hash, memory_ids, and indexing state so the
    caller sees both the rejection and the truthful underlying state.
    """
    try:
        memory_ids = json.loads(row["memory_ids"] or "[]")
    except json.JSONDecodeError:
        memory_ids = []
    return IngestReceipt(
        event_id=row["event_id"],
        payload_hash=row["payload_hash"],
        memory_ids=memory_ids if isinstance(memory_ids, list) else [],
        status="conflict",
        index_status=row["index_status"],
        attempts=int(row["attempts"] or 0),
        last_error_code="event_id_conflict",
        last_error_at=now,
        created_at=row["created_at"],
        updated_at=now,
        metadata={
            "conflicting_payload_hash": conflicting_hash,
            "original_index_status": row["index_status"],
        },
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
