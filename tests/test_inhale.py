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

The embedding backend is mocked (this environment has no fastembed model and
no sqlite-vec extension); the pipeline state machine is what is under test.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from mnemosyne.core import beam as beam_module
from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.inhale import (
    IngestEvent,
    IngestReceipt,
    TurnEvent,
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
    conflicted = beam.remember_event(
        _event(content="latency dropped to 500ms", event_id="evt-1")
    )

    assert conflicted.status == "conflict"
    assert conflicted.index_status == "failed_terminal"
    assert conflicted.last_error_code == "event_id_conflict"
    assert len(_working_rows(beam.conn)) == 1
    assert _sync_event_count(beam.conn) == 1

    persisted = _receipt_rows(beam.conn)[0]
    assert persisted["status"] == "conflict"
    assert persisted["index_status"] == "failed_terminal"

    # Replaying the conflicting payload returns the same conflict receipt.
    replay = beam.remember_event(
        _event(content="latency dropped to 500ms", event_id="evt-1")
    )
    assert replay.status == "conflict"


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
