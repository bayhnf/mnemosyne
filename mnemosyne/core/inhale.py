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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from mnemosyne.core import beam as _beam_mod
from mnemosyne.core.sync import _parse_sync_timestamp

logger = logging.getLogger(__name__)

ROLES = frozenset({"user", "assistant", "tool", "system"})
MAX_ATTEMPTS = 5
MAX_CONTENT_CHARS = 1_000_000
MAX_FIELD_CHARS = 255
MAX_METADATA_BYTES = 128 * 1024
RETRYABLE_INDEX_STATES = ("pending", "failed_retryable", "degraded")
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

    payload_hash = _payload_hash(event)
    conn = beam.conn
    engine = _get_sync_engine(beam)
    now = _now_iso()
    provenance = _provenance(event)
    receipt_meta = json.dumps({"_ingest": provenance}, sort_keys=True, default=str)

    _begin_write(conn)
    try:
        row = conn.execute(
            "SELECT * FROM ingest_receipts WHERE event_id = ?", (event.event_id,)
        ).fetchone()
        if row is not None:
            if row["payload_hash"] == payload_hash:
                # Identical payload replay: the original is already stored,
                # so this attempt is a duplicate regardless of whether an
                # intervening different-payload attempt later flipped the
                # row to 'conflict'. The dedup authority is payload_hash,
                # which the conflict path never overwrites.
                conn.rollback()
                return _receipt_from_row(row, status="duplicate")
            # Event id reused with a different payload: terminal, no mutation.
            # CRITICAL: do NOT overwrite payload_hash, memory_ids,
            # metadata_json, or created_at -- those are the original
            # receipt's identity and must stay intact so a later identical
            # replay of the ORIGINAL payload still deduplicates. Only the
            # conflict bookkeeping (status/index/error counters) flips.
            conflicting_hash = payload_hash
            conn.execute(
                """UPDATE ingest_receipts
                   SET status = 'conflict',
                       index_status = 'failed_terminal',
                       attempts = attempts + 1,
                       last_error_code = 'event_id_conflict',
                       last_error_at = ?,
                       updated_at = ?
                   WHERE event_id = ?""",
                (now, now, event.event_id),
            )
            conn.commit()
            logger.warning(
                "ingest conflict event_id=%r: stored payload_hash=%s, "
                "conflicting payload_hash=%s (original preserved)",
                event.event_id,
                row["payload_hash"],
                conflicting_hash,
            )
            return _receipt_from_row(
                conn.execute(
                    "SELECT * FROM ingest_receipts WHERE event_id = ?",
                    (event.event_id,),
                ).fetchone()
            )

        memory_id = _memory_id_for_event(event.event_id)
        metadata_json = json.dumps(_memory_metadata(event), sort_keys=True, default=str)
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


def retry_pending_ingest(beam, limit: int = 100) -> RetryReport:
    """Re-run indexing/enrichment for stored receipts that are not ready.

    Never re-inserts the raw memory row or the sync event: the receipt is the
    authority, and enrichment re-runs are idempotent (MEMORIA rows keyed by
    source_memory_id are reset before re-extraction; graph/annotation writes
    are INSERT OR REPLACE / INSERT OR IGNORE).
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    conn = beam.conn
    placeholders = ",".join("?" * len(RETRYABLE_INDEX_STATES))
    rows = conn.execute(
        f"""SELECT * FROM ingest_receipts
            WHERE status = 'stored'
              AND index_status IN ({placeholders})
            ORDER BY created_at
            LIMIT ?""",
        (*RETRYABLE_INDEX_STATES, limit),
    ).fetchall()
    report = RetryReport()
    # ponytail: no per-event claim row; concurrent retry workers on the SAME
    # event can both re-run enrichment (memory rows stay safe — the receipt is
    # the authority). Add a claimed/worker state when multi-worker retry is a
    # requirement.
    for row in rows:
        event_id = row["event_id"]
        attempts = int(row["attempts"] or 0)
        if attempts >= MAX_ATTEMPTS:
            _finalize_receipt(
                beam, event_id, "failed_terminal", "max_attempts_exceeded", attempts
            )
            report.attempted += 1
            report.failed_terminal += 1
            report.receipts.append(_final_row(conn, event_id))
            continue

        memory_rows = _load_memory_rows(conn, row["memory_ids"])
        if not memory_rows:
            _finalize_receipt(
                beam, event_id, "failed_terminal", "memory_row_missing", attempts + 1
            )
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

        _finalize_receipt(beam, event_id, final_status, final_code, attempts + 1)
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


def _reject(event: IngestEvent, errors: List[str]) -> IngestReceipt:
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
    now = _now_iso()
    return IngestReceipt(
        event_id=getattr(event, "event_id", "") or "",
        payload_hash=payload_hash,
        memory_ids=[],
        status="rejected",
        index_status="failed_terminal",
        attempts=0,
        last_error_code="validation_failed",
        last_error_at=now,
        created_at=now,
        updated_at=now,
        metadata={"errors": errors},
    )


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
            logger.warning(
                "inhale: embedding failed for %s (%s): %s",
                memory_id,
                type(exc).__name__,
                exc,
            )
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
        logger.warning(
            "inhale: enrichment failed for %s (%s): %s",
            memory_id,
            type(exc).__name__,
            exc,
        )

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


def _memory_metadata(event: IngestEvent) -> Dict[str, Any]:
    metadata = dict(event.metadata or {})
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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
