"""
Task 2: Durable native ingest receipts (Inhale).

Regression coverage for the receipt-backed ingest API:
  - ingest_receipts schema is additive and created on init
  - memory row + receipt + sync event commit atomically
  - duplicate / conflict / rejected event semantics
  - crash windows before receipt commit and after commit (pending index)
  - truthful index states: ready / degraded / failed_retryable /
    failed_terminal, and retry without duplicate memory
  - concurrent duplicate races

The embedding backend is mocked for deterministic state-machine coverage;
a real sqlite-vec + fastembed regression (test_real_sqlite_vec_upsert_marks_
receipt_ready) guards the live vector path and skips where the extension is
unavailable.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from mnemosyne.core import beam as beam_module
from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.inhale import (
    IngestEvent,
    IngestReceipt,
    TurnEvent,
    _InhaleTransactionError,
    _iso_from_epoch,
    retry_pending_ingest,
)
from mnemosyne.core.sync import SyncEngine


RECEIPT_COLUMNS = {
    "event_id",
    "payload_hash",
    "memory_ids",
    "status",
    "index_status",
    "attempts",
    "last_error_code",
    "last_error_at",
    "created_at",
    "updated_at",
}


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


@pytest.fixture
def beam(temp_db):
    return BeamMemory(session_id="inhale-session", db_path=temp_db)


@pytest.fixture
def vec_ready(monkeypatch):
    """Mock a healthy embedding + sqlite-vec pipeline."""
    monkeypatch.setattr(beam_module._embeddings, "available", lambda: True)
    monkeypatch.setattr(
        beam_module._embeddings,
        "embed",
        lambda texts: [[0.5] * beam_module.EMBEDDING_DIM for _ in texts],
    )
    monkeypatch.setattr(beam_module, "_wm_vec_available", lambda conn: True)
    monkeypatch.setattr(beam_module, "_store_working_embedding", lambda *a, **k: None)


def _event(**overrides) -> IngestEvent:
    fields: Dict[str, object] = {
        "event_id": "evt-1",
        "producer": "codex",
        "actor_id": "actor-1",
        "project_id": "project-1",
        "session_id": "session-1",
        "turn_id": "turn-1",
        "role": "user",
        "content": "latency dropped to 250ms",
        "content_hash": "",
        "occurred_at": "2026-08-10T01:02:03Z",
        "metadata": None,
    }
    fields.update(overrides)
    if not fields.get("content_hash"):
        fields["content_hash"] = hashlib.sha256(
            str(fields["content"]).encode("utf-8")
        ).hexdigest()
    return IngestEvent(**fields)  # type: ignore[arg-type]


def _turn(**overrides) -> TurnEvent:
    event = _event(**overrides)
    return TurnEvent(**vars(event))


def _receipt_rows(conn) -> List[dict]:
    return [dict(row) for row in conn.execute("SELECT * FROM ingest_receipts")]


def _working_rows(conn) -> List[dict]:
    return [dict(row) for row in conn.execute("SELECT * FROM working_memory")]


def _sync_event_count(conn) -> int:
    return conn.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0]


# ---------------------------------------------------------------------------
# Schema / basic ingest contract
# ---------------------------------------------------------------------------


def test_init_creates_ingest_receipts_table(temp_db):
    BeamMemory(session_id="s", db_path=temp_db)
    conn = beam_module._get_connection(temp_db)
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(ingest_receipts)").fetchall()
    }
    assert RECEIPT_COLUMNS <= columns


def test_remember_event_stores_memory_receipt_and_sync_event_atomically(
    beam, vec_ready
):
    receipt = beam.remember_event(_event())

    assert receipt.status == "stored"
    assert receipt.index_status == "ready"
    assert receipt.attempts == 1
    assert receipt.last_error_code is None
    assert len(receipt.memory_ids) == 1

    rows = _working_rows(beam.conn)
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == receipt.memory_ids[0]
    assert row["content"] == "latency dropped to 250ms"
    assert row["author_type"] == "codex"
    assert row["author_id"] == "actor-1"
    assert row["channel_id"] == "project-1"
    assert row["session_id"] == "session-1"
    assert row["timestamp"] == "2026-08-10T01:02:03Z"
    provenance = json.loads(row["metadata_json"])["_ingest"]
    assert provenance["event_id"] == "evt-1"
    assert provenance["turn_id"] == "turn-1"
    assert provenance["role"] == "user"

    receipts = _receipt_rows(beam.conn)
    assert len(receipts) == 1
    assert receipts[0]["status"] == "stored"
    assert receipts[0]["index_status"] == "ready"
    assert json.loads(receipts[0]["metadata_json"])["_ingest"]["role"] == "user"

    assert _sync_event_count(beam.conn) == 1


def test_remember_turn_is_a_durable_ingest(beam, vec_ready):
    receipt = beam.remember_turn(_turn(turn_id="turn-9", role="assistant"))
    assert receipt.status == "stored"
    assert receipt.index_status == "ready"
    row = _working_rows(beam.conn)[0]
    assert json.loads(row["metadata_json"])["_ingest"]["role"] == "assistant"


# ---------------------------------------------------------------------------
# Duplicate / conflict / rejection semantics
# ---------------------------------------------------------------------------


def test_100_identical_replays_return_duplicate_without_duplicate_memory(
    beam, vec_ready
):
    first = beam.remember_event(_event())
    assert first.status == "stored"

    for _ in range(100):
        replay = beam.remember_event(_event())
        assert replay.status == "duplicate"
        assert replay.event_id == first.event_id
        assert replay.payload_hash == first.payload_hash
        assert replay.memory_ids == first.memory_ids

    assert len(_working_rows(beam.conn)) == 1
    assert len(_receipt_rows(beam.conn)) == 1
    assert _sync_event_count(beam.conn) == 1


def test_same_event_id_different_payload_is_conflict_without_mutation(beam, vec_ready):
    first = beam.remember_event(_event())
    assert first.status == "stored"
    orig_index_status = first.index_status
    conflicted = beam.remember_event(
        _event(content="latency dropped to 500ms", event_id="evt-1")
    )

    assert conflicted.status == "conflict"
    assert conflicted.last_error_code == "event_id_conflict"
    # Conflict does NOT conflate with indexing lifecycle: the returned
    # receipt surfaces the ORIGINAL index state, never failed_terminal.
    assert conflicted.index_status == orig_index_status
    assert len(_working_rows(beam.conn)) == 1
    assert _sync_event_count(beam.conn) == 1

    persisted = _receipt_rows(beam.conn)[0]
    # The durable original receipt stays 'stored' with its real index state.
    assert persisted["status"] == "stored"
    assert persisted["index_status"] == orig_index_status

    # Replaying the conflicting payload is still a conflict (audit grows).
    replay = beam.remember_event(
        _event(content="latency dropped to 500ms", event_id="evt-1")
    )
    assert replay.status == "conflict"
    assert replay.index_status == orig_index_status


def test_validation_rejection_is_rejected_and_writes_nothing(beam, vec_ready):
    bad_hash = _event(content_hash="deadbeef")
    rejected = beam.remember_event(bad_hash)

    assert rejected.status == "rejected"
    assert rejected.index_status == "failed_terminal"
    assert rejected.last_error_code == "validation_failed"
    assert len(_working_rows(beam.conn)) == 0
    assert len(_receipt_rows(beam.conn)) == 0
    assert _sync_event_count(beam.conn) == 0

    bad_role = _event(role="villain")
    assert beam.remember_event(bad_role).status == "rejected"
    assert beam.remember_event(_event(metadata="not-a-dict")).status == "rejected"


def test_corrected_event_id_can_succeed_after_rejection(beam, vec_ready):
    beam.remember_event(_event(content_hash="deadbeef"))
    assert len(_working_rows(beam.conn)) == 0

    fixed = beam.remember_event(_event())
    assert fixed.status == "stored"
    assert len(_working_rows(beam.conn)) == 1


# ---------------------------------------------------------------------------
# Atomicity and crash windows
# ---------------------------------------------------------------------------


def test_sync_event_failure_rolls_back_memory_and_receipt(beam, vec_ready, monkeypatch):
    def _boom(self, *args, **kwargs):
        raise RuntimeError("sync transport exploded")

    monkeypatch.setattr(SyncEngine, "log_event", _boom)
    with pytest.raises(RuntimeError):
        beam.remember_event(_event())

    assert len(_working_rows(beam.conn)) == 0
    assert len(_receipt_rows(beam.conn)) == 0
    assert _sync_event_count(beam.conn) == 0

    monkeypatch.undo()
    receipt = beam.remember_event(_event())
    assert receipt.status == "stored"
    assert len(_working_rows(beam.conn)) == 1
    assert len(_receipt_rows(beam.conn)) == 1


def test_crash_after_commit_before_embedding_leaves_pending_and_retries_once(
    beam, vec_ready, monkeypatch
):
    import mnemosyne.core.inhale as inhale

    def _flaky_embed(texts):
        raise RuntimeError("embedding service down")

    monkeypatch.setattr(beam_module._embeddings, "embed", _flaky_embed)
    monkeypatch.setattr(
        inhale,
        "_finalize_receipt",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("process died")),
    )

    with pytest.raises(RuntimeError):
        beam.remember_event(_event())

    # Raw commit is durable; index state is truthfully pending, never ready.
    assert len(_working_rows(beam.conn)) == 1
    receipts = _receipt_rows(beam.conn)
    assert len(receipts) == 1
    assert receipts[0]["status"] == "stored"
    assert receipts[0]["index_status"] == "pending"
    assert receipts[0]["attempts"] == 1

    monkeypatch.undo()
    report = retry_pending_ingest(beam)
    assert report.attempted == 1
    assert report.succeeded == 1
    assert report.receipts[0].index_status == "ready"

    assert len(_working_rows(beam.conn)) == 1
    assert len(_receipt_rows(beam.conn)) == 1
    assert _sync_event_count(beam.conn) == 1


# ---------------------------------------------------------------------------
# Truthful index states
# ---------------------------------------------------------------------------


def test_embedding_unavailable_yields_degraded_with_memory_intact(beam, monkeypatch):
    monkeypatch.setattr(beam_module._embeddings, "available", lambda: False)
    receipt = beam.remember_event(_event())
    assert receipt.status == "stored"
    assert receipt.index_status == "degraded"
    assert receipt.last_error_code == "embedding_unavailable"
    assert len(_working_rows(beam.conn)) == 1


def test_vector_index_unavailable_yields_degraded(beam, monkeypatch):
    monkeypatch.setattr(beam_module._embeddings, "available", lambda: True)
    monkeypatch.setattr(
        beam_module._embeddings,
        "embed",
        lambda texts: [[0.5] * beam_module.EMBEDDING_DIM for _ in texts],
    )
    # Force the truthful degraded path regardless of whether this test
    # environment ships the sqlite-vec extension.
    monkeypatch.setattr(beam_module, "_wm_vec_available", lambda conn: False)
    receipt = beam.remember_event(_event())
    assert receipt.index_status == "degraded"
    assert receipt.last_error_code == "vector_index_unavailable"
    assert (
        beam.conn.execute(
            "SELECT COUNT(*) FROM memory_embeddings WHERE memory_id = ?",
            (receipt.memory_ids[0],),
        ).fetchone()[0]
        == 1
    )


def test_embedding_failure_yields_failed_retryable_then_ready(
    beam, vec_ready, monkeypatch
):
    def _flaky_embed(texts):
        raise RuntimeError("embedding service down")

    monkeypatch.setattr(beam_module._embeddings, "embed", _flaky_embed)
    receipt = beam.remember_event(_event())
    assert receipt.status == "stored"
    assert receipt.index_status == "failed_retryable"
    assert receipt.last_error_code == "embedding_failure"
    assert len(_working_rows(beam.conn)) == 1

    monkeypatch.undo()
    report = retry_pending_ingest(beam)
    assert report.attempted == 1
    assert report.succeeded == 1
    assert len(_working_rows(beam.conn)) == 1
    assert len(_receipt_rows(beam.conn)) == 1


def test_enrichment_failure_yields_degraded_then_retry_is_idempotent(
    beam, vec_ready, monkeypatch
):
    original_extract = beam.extract_and_store_facts

    def _flaky_extract(content, message_idx=0, source_memory_id=None):
        raise RuntimeError("regex extraction exploded")

    monkeypatch.setattr(beam, "extract_and_store_facts", _flaky_extract)
    receipt = beam.remember_event(_event())
    assert receipt.index_status == "degraded"
    assert receipt.last_error_code == "enrichment_failed"

    monkeypatch.setattr(beam, "extract_and_store_facts", original_extract)
    report = retry_pending_ingest(beam)
    assert report.attempted == 1
    assert report.succeeded == 1

    # Retry must not duplicate the memory row, the receipt, or enrichment rows.
    assert len(_working_rows(beam.conn)) == 1
    assert len(_receipt_rows(beam.conn)) == 1
    expected_facts = beam.conn.execute(
        "SELECT COUNT(*) FROM memoria_facts WHERE source_memory_id = ?",
        (receipt.memory_ids[0],),
    ).fetchone()[0]
    assert expected_facts > 0


# ---------------------------------------------------------------------------
# Retry report and concurrency
# ---------------------------------------------------------------------------


def test_retry_pending_ingest_respects_limit(beam, vec_ready, monkeypatch):
    import mnemosyne.core.inhale as inhale

    monkeypatch.setattr(
        inhale,
        "_finalize_receipt",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("process died")),
    )
    for i in range(3):
        with pytest.raises(RuntimeError):
            beam.remember_event(_event(event_id=f"evt-{i}"))

    monkeypatch.undo()
    assert (
        beam.conn.execute(
            "SELECT COUNT(*) FROM ingest_receipts WHERE index_status = 'pending'"
        ).fetchone()[0]
        == 3
    )

    partial = retry_pending_ingest(beam, limit=2)
    assert partial.attempted == 2
    assert partial.succeeded == 2
    assert len(partial.receipts) == 2
    assert (
        beam.conn.execute(
            "SELECT COUNT(*) FROM ingest_receipts WHERE index_status = 'pending'"
        ).fetchone()[0]
        == 1
    )

    rest = retry_pending_ingest(beam, limit=100)
    assert rest.attempted == 1
    assert rest.succeeded == 1
    assert (
        beam.conn.execute(
            "SELECT COUNT(*) FROM ingest_receipts WHERE index_status != 'ready'"
        ).fetchone()[0]
        == 0
    )


def test_concurrent_duplicate_race_produces_exactly_one_memory(temp_db, vec_ready):
    # Pre-initialize schema serially, then let each thread use its own
    # thread-local connection like production code does.
    BeamMemory(session_id="race", db_path=temp_db)
    exceptions: List[BaseException] = []
    barrier = threading.Barrier(2)
    results: List[Optional[IngestReceipt]] = [None, None]

    def worker(idx: int):
        try:
            b = BeamMemory(session_id="race", db_path=temp_db)
            b.conn.execute("PRAGMA busy_timeout=10000")
            barrier.wait()
            results[idx] = b.remember_event(_event())
        except BaseException as exc:
            exceptions.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not exceptions, f"threads raised: {exceptions}"
    statuses = sorted(r.status for r in results if r is not None)
    assert statuses == ["duplicate", "stored"]

    conn = beam_module._get_connection(temp_db)
    assert conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM ingest_receipts").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0] == 1


# ---------------------------------------------------------------------------
# Task 2 takeover: conflict must not corrupt original dedup (audit point #1)
# ---------------------------------------------------------------------------


def test_conflict_must_not_overwrite_original_payload_hash(beam, vec_ready):
    """A reused event_id with a different payload is a conflict, but the
    original receipt's payload_hash MUST be preserved so a later identical
    replay of the ORIGINAL payload still deduplicates (returns duplicate),
    not a fresh conflict."""
    original = _event(content="latency dropped to 250ms", event_id="evt-X")
    first = beam.remember_event(original)
    assert first.status == "stored"
    original_hash = first.payload_hash

    conflicting = _event(content="latency dropped to 500ms", event_id="evt-X")
    conflict = beam.remember_event(conflicting)
    assert conflict.status == "conflict"

    # The stored payload_hash must still be the ORIGINAL, not the conflicting
    # payload's hash. If the conflict overwrote it, the original could no
    # longer deduplicate.
    persisted = _receipt_rows(beam.conn)[0]
    assert persisted["payload_hash"] == original_hash, (
        "conflict path overwrote the original payload_hash; a later identical "
        "replay of the original payload would stop deduplicating"
    )

    # Replay the ORIGINAL payload: must be a duplicate, NOT a conflict.
    replay = beam.remember_event(original)
    assert replay.status == "duplicate", (
        "original payload no longer deduplicates after a conflict corrupted "
        "the stored payload_hash"
    )
    assert replay.payload_hash == original_hash


def test_conflict_does_not_create_duplicate_memory_or_sync_event(
    beam, vec_ready
):
    """A conflict must mutate nothing except flipping the receipt to conflict;
    no new working_memory row, no new sync event."""
    beam.remember_event(_event(content="original content", event_id="evt-C"))
    assert len(_working_rows(beam.conn)) == 1
    assert _sync_event_count(beam.conn) == 1

    beam.remember_event(_event(content="different content", event_id="evt-C"))
    assert len(_working_rows(beam.conn)) == 1
    assert _sync_event_count(beam.conn) == 1


def test_idempotency_across_two_independent_connections(temp_db, vec_ready):
    """Audit point #4: same event id + identical payload under two independent
    BeamMemory connections (separate thread-local conns) yields exactly one
    stored memory and one duplicate, never two stored rows."""
    BeamMemory(session_id="s1", db_path=temp_db)
    b1 = BeamMemory(session_id="s1", db_path=temp_db)
    b2 = BeamMemory(session_id="s2", db_path=temp_db)
    b1.conn.execute("PRAGMA busy_timeout=10000")
    b2.conn.execute("PRAGMA busy_timeout=10000")

    ev = _event(event_id="evt-2conn")
    r1 = b1.remember_event(ev)
    r2 = b2.remember_event(ev)

    statuses = sorted([r1.status, r2.status])
    assert statuses == ["duplicate", "stored"], statuses

    conn = beam_module._get_connection(temp_db)
    assert conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM ingest_receipts").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0] == 1


def test_conflict_then_original_replay_across_two_connections(temp_db, vec_ready):
    """The conflict-dedup invariant must hold across independent connections:
    after a conflict on conn A, replaying the original on conn B still
    deduplicates (does not become a fresh conflict or stored)."""
    BeamMemory(session_id="s1", db_path=temp_db)
    b1 = BeamMemory(session_id="s1", db_path=temp_db)
    b2 = BeamMemory(session_id="s2", db_path=temp_db)
    b1.conn.execute("PRAGMA busy_timeout=10000")
    b2.conn.execute("PRAGMA busy_timeout=10000")

    original = _event(content="original payload", event_id="evt-Y")
    conflict_ev = _event(content="conflicting payload", event_id="evt-Y")

    assert b1.remember_event(original).status == "stored"
    assert b2.remember_event(conflict_ev).status == "conflict"

    # Original replay on a third independent connection must still dedupe.
    b3 = BeamMemory(session_id="s3", db_path=temp_db)
    b3.conn.execute("PRAGMA busy_timeout=10000")
    replay = b3.remember_event(original)
    assert replay.status in ("duplicate", "conflict")
    # It must NOT be 'stored' (no duplicate memory) and the original hash
    # must be intact for dedup.
    conn = beam_module._get_connection(temp_db)
    assert conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0] == 1


# ---------------------------------------------------------------------------
# Fix Round 1: conflict must not corrupt the original receipt lifecycle
# ---------------------------------------------------------------------------


def test_conflict_does_not_mutate_original_receipt_lifecycle(beam, vec_ready):
    """A same-event_id, different-payload request returns a structured
    'conflict' outcome but changes NOTHING about the original receipt:
    status stays 'stored', index_status keeps its real value, memory_ids /
    payload_hash / attempts / original error fields are untouched."""
    first = beam.remember_event(_event(content="original content", event_id="evt-L"))
    assert first.status == "stored"
    assert first.index_status == "ready"
    orig_index_status = first.index_status
    orig_attempts = first.attempts
    orig_memory_ids = list(first.memory_ids)

    conflict = beam.remember_event(
        _event(content="different content", event_id="evt-L")
    )
    assert conflict.status == "conflict"

    persisted = _receipt_rows(beam.conn)[0]
    # The original receipt lifecycle is authoritative and untouched.
    assert persisted["status"] == "stored", (
        "conflict mutated the original receipt status"
    )
    assert persisted["index_status"] == orig_index_status, (
        "conflict mutated the original index_status"
    )
    assert persisted["attempts"] == orig_attempts, (
        "conflict inflated the original indexing attempts"
    )
    assert json.loads(persisted["memory_ids"]) == orig_memory_ids
    assert persisted["last_error_code"] is None, (
        "conflict polluted the original indexing error code"
    )

    # A later original-payload replay returns duplicate with the ORIGINAL
    # index state (ready), never failed_terminal.
    replay = beam.remember_event(_event(content="original content", event_id="evt-L"))
    assert replay.status == "duplicate"
    assert replay.index_status == orig_index_status


def test_pending_original_remains_retryable_after_conflict(beam, vec_ready, monkeypatch):
    """Exact sequence: raw commit leaves stored/pending; a conflict occurs;
    retry_pending_ingest() can still claim and finish the original without
    duplicate memory/sync rows and without stranding it terminal."""
    import mnemosyne.core.inhale as inhale

    def _flaky_embed(texts):
        raise RuntimeError("embedding service down")

    monkeypatch.setattr(beam_module._embeddings, "embed", _flaky_embed)
    r = beam.remember_event(_event(event_id="evt-PR"))
    assert r.status == "stored"
    assert r.index_status == "failed_retryable"
    assert _sync_event_count(beam.conn) == 1

    # Conflict happens while the original is still retryable.
    c = beam.remember_event(_event(content="different", event_id="evt-PR"))
    assert c.status == "conflict"

    # Original receipt is still stored + retryable (NOT terminalized).
    persisted = _receipt_rows(beam.conn)[0]
    assert persisted["status"] == "stored"
    assert persisted["index_status"] in ("pending", "failed_retryable", "degraded")

    monkeypatch.undo()
    report = retry_pending_ingest(beam)
    assert report.attempted == 1
    assert report.succeeded == 1
    assert report.receipts[0].index_status == "ready"
    # No duplicate memory/sync rows.
    assert len(_working_rows(beam.conn)) == 1
    assert _sync_event_count(beam.conn) == 1


def test_conflict_durable_audit_trail(beam, vec_ready):
    """Conflicts are durably observable without mutating the original
    receipt's indexing lifecycle."""
    beam.remember_event(_event(content="original", event_id="evt-AU"))
    beam.remember_event(_event(content="challenger-1", event_id="evt-AU"))
    beam.remember_event(_event(content="challenger-2", event_id="evt-AU"))

    audit = [
        dict(row)
        for row in beam.conn.execute("SELECT * FROM ingest_conflicts ORDER BY seq")
    ]
    assert len(audit) == 2
    for entry in audit:
        assert entry["event_id"] == "evt-AU"
        assert entry["conflicting_payload_hash"]
        assert entry["observed_at"]
    # Original receipt attempts untouched by conflicts.
    assert _receipt_rows(beam.conn)[0]["attempts"] == 1


# ---------------------------------------------------------------------------
# Fix Round 1: real sqlite-vec path (no mock)
# ---------------------------------------------------------------------------


def test_real_sqlite_vec_upsert_marks_receipt_ready(temp_db, monkeypatch):
    """When sqlite-vec is actually available, the ingest path writes to
    vec_working and the receipt reaches 'ready' on the live path -- not via
    a mock."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.enable_load_extension(True)
        import sqlite_vec

        conn.load_extension(sqlite_vec.loadable_path())
    except Exception:
        pytest.skip("sqlite-vec extension unavailable in this runtime")
    finally:
        try:
            conn.close()
        except Exception:
            pass

    BeamMemory(session_id="realvec", db_path=temp_db)
    b = BeamMemory(session_id="realvec", db_path=temp_db)
    # Use the real embedding model (no embed mock); only mocking away any
    # network is unnecessary -- fastembed runs locally.
    receipt = b.remember_event(_event(event_id="evt-RV", content="vec upsert live"))
    assert receipt.status == "stored"
    assert receipt.index_status == "ready", receipt
    assert receipt.last_error_code is None


# ---------------------------------------------------------------------------
# Fix Round 1: reject caller-owned / deferred transaction context
# ---------------------------------------------------------------------------


def test_remember_event_rejects_deferred_commit_context(beam, vec_ready):
    """The Inhale API owns its short atomic transaction; it must not run
    inside a caller's deferred-commit context (which would leave _defer_commit
    set across network enrichment or rollback a caller-owned txn)."""
    from mnemosyne.core.beam import _deferred_commits

    with _deferred_commits(beam.conn):
        with pytest.raises(_InhaleTransactionError):
            beam.remember_event(_event(event_id="evt-DF"))


def test_remember_event_rejects_open_caller_transaction(beam, vec_ready):
    """If the caller already opened a transaction (in_transaction True) the
    API must reject before mutating rather than rolling back the caller's
    work."""
    beam.conn.execute("BEGIN")
    try:
        with pytest.raises(_InhaleTransactionError):
            beam.remember_event(_event(event_id="evt-OT"))
    finally:
        beam.conn.rollback()
    # The caller's transaction was not rolled back by the API; rollback here
    # is the test's own cleanup. The API must have written nothing.
    receipt = beam.remember_event(_event(event_id="evt-OT"))
    assert receipt.status == "stored"


def test_retry_pending_ingest_rejects_deferred_commit_context(beam, vec_ready):
    from mnemosyne.core.beam import _deferred_commits

    beam.remember_event(_event(event_id="evt-RD"))
    # Force it back to pending so retry has work.
    beam.conn.execute(
        "UPDATE ingest_receipts SET index_status='pending' WHERE event_id='evt-RD'"
    )
    beam.conn.commit()
    with _deferred_commits(beam.conn):
        with pytest.raises(_InhaleTransactionError):
            retry_pending_ingest(beam)


# ---------------------------------------------------------------------------
# Fix Round 1: claim-safe concurrent retries
# ---------------------------------------------------------------------------


def test_two_retry_workers_do_not_duplicate_enrichment(
    temp_db, vec_ready, monkeypatch
):
    """Two independent retry workers on the same pending receipt: exactly one
    does enrichment; the other observes the claimed/finished state. Memory
    and sync rows stay exactly-once."""
    BeamMemory(session_id="rw", db_path=temp_db)
    b0 = BeamMemory(session_id="rw", db_path=temp_db)
    b0.conn.execute("PRAGMA busy_timeout=10000")

    # Store a receipt, leave it pending.
    import mnemosyne.core.inhale as inhale

    monkeypatch.setattr(
        inhale,
        "_finalize_receipt",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("crash")),
    )
    with pytest.raises(RuntimeError):
        b0.remember_event(_event(event_id="evt-2W"))
    monkeypatch.undo()

    persisted = _receipt_rows(b0.conn)[0]
    assert persisted["index_status"] == "pending"

    call_count = {"n": 0}
    real_index = inhale._index_memory

    def _counting_index(beam, memory_id, content, source, timestamp):
        call_count["n"] += 1
        return real_index(beam, memory_id, content, source, timestamp)

    monkeypatch.setattr(inhale, "_index_memory", _counting_index)

    results: dict = {}
    exceptions: List[BaseException] = []
    barrier = threading.Barrier(2)

    def worker(tag):
        try:
            b = BeamMemory(session_id=f"rw-{tag}", db_path=temp_db)
            b.conn.execute("PRAGMA busy_timeout=10000")
            barrier.wait()
            results[tag] = retry_pending_ingest(b)
        except BaseException as exc:
            exceptions.append(exc)

    t1 = threading.Thread(target=worker, args=("a",))
    t2 = threading.Thread(target=worker, args=("b",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not exceptions, exceptions
    total_attempted = sum(r.attempted for r in results.values())
    total_succeeded = sum(r.succeeded for r in results.values())
    assert total_attempted == 1, results
    assert total_succeeded == 1, results

    conn = beam_module._get_connection(temp_db)
    assert conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0] == 1
    assert (
        conn.execute(
            "SELECT index_status FROM ingest_receipts WHERE event_id='evt-2W'"
        ).fetchone()[0]
        == "ready"
    )

    monkeypatch.undo()


def test_stale_retry_claim_is_reclaimed(temp_db, vec_ready, monkeypatch):
    """A claim whose lease has expired (crashed worker) must be reclaimable by
    a later retry pass so the receipt is not stranded."""
    BeamMemory(session_id="sc", db_path=temp_db)
    b = BeamMemory(session_id="sc", db_path=temp_db)

    import mnemosyne.core.inhale as inhale

    monkeypatch.setattr(
        inhale,
        "_finalize_receipt",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("crash")),
    )
    with pytest.raises(RuntimeError):
        b.remember_event(_event(event_id="evt-SC"))
    monkeypatch.undo()

    # Simulate a crashed worker: claim the receipt with an expired lease.
    stale = (datetime.now(timezone.utc).timestamp() - 3600)
    b.conn.execute(
        "UPDATE ingest_receipts SET claim_worker_id='dead-worker',"
        " claim_worker_lease=? WHERE event_id='evt-SC'",
        (_iso_from_epoch(stale),),
    )
    b.conn.commit()

    report = retry_pending_ingest(b)
    assert report.attempted == 1
    assert report.succeeded == 1
    assert report.receipts[0].index_status == "ready"


# ---------------------------------------------------------------------------
# Fix Round 2: claim must re-check lifecycle at claim time (TOCTOU)
# ---------------------------------------------------------------------------


def test_delayed_worker_cannot_claim_already_finalized_receipt(
    temp_db, vec_ready, monkeypatch
):
    """A delayed candidate whose receipt another worker finalized to 'ready'
    BEFORE it reaches the claim must NOT claim, enrich, or overwrite that
    terminal lifecycle state.

    This closes the TOCTOU gap: candidate selection happens before the claim
    transaction, so the atomic claim UPDATE must re-check status='stored'
    and index_status is retryable in addition to lease availability. Here the
    delayed worker holds a pre-selected event_id and reaches _try_claim after
    another worker has already finalized the receipt to 'ready'.
    """
    BeamMemory(session_id="toctou", db_path=temp_db)
    b = BeamMemory(session_id="toctou", db_path=temp_db)
    b.conn.execute("PRAGMA busy_timeout=10000")

    # Store a receipt and leave it pending (crash during indexing).
    import mnemosyne.core.inhale as inhale

    monkeypatch.setattr(
        inhale,
        "_finalize_receipt",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("crash")),
    )
    with pytest.raises(RuntimeError):
        b.remember_event(_event(event_id="evt-TOU"))
    monkeypatch.undo()

    assert (
        b.conn.execute(
            "SELECT index_status FROM ingest_receipts WHERE event_id='evt-TOU'"
        ).fetchone()[0]
        == "pending"
    )

    # Worker B runs a full retry pass that finalizes the receipt to 'ready'
    # (claim acquired + enrichment + finalize + release).
    b2 = BeamMemory(session_id="toctou-b", db_path=temp_db)
    b2.conn.execute("PRAGMA busy_timeout=10000")
    report_b = retry_pending_ingest(b2)
    assert report_b.attempted == 1
    assert report_b.succeeded == 1
    assert (
        b2.conn.execute(
            "SELECT index_status FROM ingest_receipts WHERE event_id='evt-TOU'"
        ).fetchone()[0]
        == "ready"
    )

    # The delayed worker A holds the stale candidate (evt-TOU was pending when
    # selected) and reaches _try_claim AFTER B finalized. The claim must fail
    # because the receipt is no longer retryable, even though the lease is free.
    now_iso = inhale._iso_from_epoch(inhale._now_epoch())
    lease_iso = inhale._iso_from_epoch(inhale._now_epoch() + 60)
    claimed = inhale._try_claim(
        b.conn, "evt-TOU", "delayed-worker-a", lease_iso, now_iso
    )
    assert claimed is False, (
        "delayed worker claimed an already-ready receipt via stale candidate"
    )

    # The truthful 'ready' lifecycle state is not overwritten.
    row = b.conn.execute(
        "SELECT status, index_status, attempts FROM ingest_receipts"
        " WHERE event_id='evt-TOU'"
    ).fetchone()
    assert row[0] == "stored"
    assert row[1] == "ready"
    # Worker A did not enrich (claim refused), so attempts are unchanged.
    assert row[2] == report_b.receipts[0].attempts


def test_delayed_worker_cannot_claim_terminalized_receipt(temp_db, vec_ready, monkeypatch):
    """Same TOCTOU guard but for a receipt finalized to 'failed_terminal'
    (max attempts): the delayed worker must not resurrect it."""
    BeamMemory(session_id="tt", db_path=temp_db)
    b = BeamMemory(session_id="tt", db_path=temp_db)
    b.conn.execute("PRAGMA busy_timeout=10000")

    # Create a stored receipt, then terminalize it directly (simulating a
    # prior retry that hit max_attempts).
    b.remember_event(_event(event_id="evt-TERM"))
    b.conn.execute(
        "UPDATE ingest_receipts SET index_status='failed_terminal',"
        " attempts=?, last_error_code='max_attempts_exceeded'"
        " WHERE event_id='evt-TERM'",
        (99,),
    )
    b.conn.commit()

    # Even if we inject the event_id as a candidate (e.g. it was selected
    # before terminalization), the claim must fail because index_status is
    # no longer retryable.
    import mnemosyne.core.inhale as inhale

    now_iso = inhale._iso_from_epoch(inhale._now_epoch())
    lease_iso = inhale._iso_from_epoch(inhale._now_epoch() + 60)
    claimed = inhale._try_claim(
        b.conn, "evt-TERM", "late-worker", lease_iso, now_iso
    )
    assert claimed is False, "claimed a failed_terminal receipt"

    row = b.conn.execute(
        "SELECT status, index_status, attempts FROM ingest_receipts"
        " WHERE event_id='evt-TERM'"
    ).fetchone()
    assert row[0] == "stored"
    assert row[1] == "failed_terminal"
    assert row[2] == 99
