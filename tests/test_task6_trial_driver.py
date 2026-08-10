"""Task 6: deterministic, content-free local trial driver.

Test-only. Proves exactly-once ingest, conflict non-mutation, admission
reject/accept, migration report-only stability, snapshot/restore round-trip,
crash/restart/concurrency safety, and the Dream lifecycle -- without ever
contacting a network, a live model, or a remote host.

All helpers are copied inline per the brief's helper policy (never
cross-imported from sibling test files). Constructs ``BeamMemory`` inside each
test body; never at module scope (thread-local connection caches leak).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

import mnemosyne.core.beam as beam_module
import mnemosyne.core.config as config_module
from mnemosyne.core.inhale import IngestEvent, TurnEvent


# ---------------------------------------------------------------------------
# Inline ingest helpers (mirror tests/test_inhale.py:84-100, 1276-1283)
# ---------------------------------------------------------------------------


def _event(**overrides) -> IngestEvent:
    """Deterministic IngestEvent; content_hash mirrors the source helper."""
    fields: Dict[str, Any] = {
        "event_id": "evt-1",
        "producer": "codex",
        "actor_id": "actor-1",
        "project_id": "project-1",
        "session_id": "session-1",
        "turn_id": "turn-1",
        "role": "user",
        "content": "latency baseline recorded at two hundred fifty milliseconds",
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
    """TurnEvent built from _event; second position defaults to assistant."""
    fields: Dict[str, Any] = {
        "event_id": "turn-1",
        "producer": "codex",
        "actor_id": "actor-1",
        "project_id": "project-1",
        "session_id": "session-1",
        "turn_id": "turn-1",
        "role": "user",
        "content": "release checklist signed off by oncall",
        "content_hash": "",
        "occurred_at": "2026-08-10T01:02:03Z",
        "metadata": None,
    }
    fields.update(overrides)
    if not fields.get("content_hash"):
        fields["content_hash"] = hashlib.sha256(
            str(fields["content"]).encode("utf-8")
        ).hexdigest()
    return TurnEvent(**fields)  # type: ignore[arg-type]


def _table_count(beam, table: str) -> int:
    return beam.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _receipt_statuses(conn) -> List[str]:
    return [
        r[0]
        for r in conn.execute("SELECT status FROM ingest_receipts ORDER BY event_id")
    ]


def _sync_event_count(conn) -> int:
    return conn.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0]


def _stored_content(beam, event_id: str) -> Optional[str]:
    memory_id = hashlib.sha256(event_id.encode("utf-8")).hexdigest()[:16]
    row = beam.conn.execute(
        "SELECT content FROM working_memory WHERE id = ?", (memory_id,)
    ).fetchone()
    return row["content"] if row is not None else None


# ---------------------------------------------------------------------------
# Inline schema/source fingerprint helpers
# (mirror tests/test_migration_dry_run_fingerprint.py:19-32,
#  tests/test_snapshot.py:62-101)
# ---------------------------------------------------------------------------


def _schema_fingerprint(db_path) -> str:
    """Read-only hash of schema rows + user_version."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        conn.execute("PRAGMA query_only=ON")
        rows = conn.execute(
            "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()
        user_version = conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()
    return hashlib.sha256(repr((rows, user_version)).encode()).hexdigest()


def _source_fingerprint(path: Path) -> tuple:
    """(path, size, mtime_ns, sha256) for a db + its WAL/SHM sidecars."""
    parts = [
        path,
        path.with_name(path.name + "-wal"),
        path.with_name(path.name + "-shm"),
    ]
    fp = []
    for p in parts:
        if p.exists():
            st = p.stat()
            fp.append(
                (
                    str(p),
                    st.st_size,
                    st.st_mtime_ns,
                    hashlib.sha256(p.read_bytes()).hexdigest(),
                )
            )
    return tuple(fp)


def _integrity(path: Path) -> str:
    conn = sqlite3.connect(str(path))
    try:
        return conn.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        conn.close()


def _journal_mode(path: Path) -> str:
    conn = sqlite3.connect(str(path))
    try:
        return conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Inline migration bank builder
# (mirror tests/test_migration_dry_run_fingerprint.py:_wal_seed, _fresh_e7_bank)
# ---------------------------------------------------------------------------


def _wal_seed(db_path) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE IF NOT EXISTS seed_probe (x TEXT)")
    conn.execute("DELETE FROM seed_probe")
    conn.execute("INSERT INTO seed_probe VALUES ('seed')")
    conn.commit()
    conn.close()


def _fresh_e7_bank(tmp_path):
    """54-table-era bank: full init minus the two 3.11.1 tables."""
    from mnemosyne.core.memory import init_db

    db_path = tmp_path / "e7_bank.db"
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    for tbl in ("memory_events", "sync_meta"):
        conn.execute(f"DROP TABLE IF EXISTS {tbl}")
    conn.commit()
    conn.close()
    _wal_seed(db_path)
    return db_path


# ---------------------------------------------------------------------------
# Inline Dream helpers
# (mirror tests/test_dream_lifecycle.py:87-105, 216-226)
# ---------------------------------------------------------------------------


def _seed_facts(beam, rows):
    for r in rows:
        beam.conn.execute(
            "INSERT INTO facts "
            "(fact_id, session_id, subject, predicate, object, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                r["fact_id"],
                r.get("session_id", beam.session_id),
                r["subject"],
                r["predicate"],
                r["object"],
                r.get("confidence", 0.9),
            ),
        )
    beam.conn.commit()


def _pass_receipt(role, actor_id, run_id, manifest_hash, when=None):
    return {
        "role": role,
        "actor_id": actor_id,
        "run_id": run_id,
        "manifest_hash": manifest_hash,
        "verdict": "PASS",
        "reason_code": "ok",
        "timestamp": (when or datetime.now(timezone.utc).isoformat()),
    }


# ---------------------------------------------------------------------------
# Config isolation + offline embedding seams for every BeamMemory test
# (mirror tests/test_dream_lifecycle.py:66-83, 175-188)
# ---------------------------------------------------------------------------


def _force_offline_embeddings(monkeypatch):
    """Pin SHMR's embedding seam to lexical fallback; guard against network."""
    from mnemosyne.core import embeddings as _emb
    from mnemosyne.core import shmr

    monkeypatch.setattr(shmr, "_embedding_fn", lambda: None, raising=True)

    def _counting_guard(_texts):
        raise AssertionError(
            "Dream test reached the network embedding path; tests must stay "
            "offline via the deterministic lexical fallback."
        )

    monkeypatch.setattr(_emb, "embed", _counting_guard, raising=True)


def _isolate_config(tmp_path, monkeypatch):
    """Point the central config at a throwaway data dir + offline seams.

    Call before constructing any BeamMemory; resets on exit so no test touches
    a real user configuration file. Keeps retry embedding seams alive (never
    monkeypatch.undo()).
    """
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    config_module.MnemosyneConfig.reset_instance()
    _force_offline_embeddings(monkeypatch)
    # reset again on teardown via pytest's monkeypatch finalizer is not enough;
    # the singleton must be cleared explicitly.
    return tmp_path


@pytest.fixture(autouse=True)
def _reset_config_singleton():
    """Ensure the MnemosyneConfig singleton never leaks across tests."""
    config_module.MnemosyneConfig.reset_instance()
    yield
    config_module.MnemosyneConfig.reset_instance()


# ---------------------------------------------------------------------------
# Content-free report projections (strict allowlist)
# (mirror mnemosyne/cli.py:_dream_run_projection shape)
# ---------------------------------------------------------------------------


def _dream_run_projection(run) -> dict:
    """Content-free projection of one DreamRun.

    Omits scope, raw manifest, actions, before/after images, failure_reason.
    """
    receipt_counts: dict = {}
    raw_receipts = getattr(run, "receipts", None) or []
    if isinstance(raw_receipts, list):
        for r in raw_receipts:
            if not isinstance(r, dict):
                continue
            role = r.get("role", "unknown")
            verdict = r.get("verdict", "unknown")
            key = f"{role}:{verdict}"
            receipt_counts[key] = receipt_counts.get(key, 0) + 1
    action_count = (
        len(raw_actions)
        if isinstance((raw_actions := getattr(run, "actions", None)), list)
        else 0
    )
    return {
        "run_id": run.run_id,
        "state": run.state,
        "manifest_hash": getattr(run, "manifest_hash", "") or "",
        "checkpoint": getattr(run, "checkpoint", "") or "",
        "error_code": getattr(run, "error_code", None),
        "created_at": getattr(run, "created_at", "") or "",
        "updated_at": getattr(run, "updated_at", "") or "",
        "request_id": getattr(run, "request_id", None),
        "action_count": action_count,
        "receipt_counts": receipt_counts,
    }


def _dream_report_projection(applied, undone, second_undo) -> dict:
    return {
        "applied_state": applied.state,
        "undo_state": undone.state,
        "second_undo_state": second_undo.state,
        "applied": _dream_run_projection(applied),
        "undone": _dream_run_projection(undone),
        "second_undo": _dream_run_projection(second_undo),
    }


def _ingest_report_projection(receipt) -> dict:
    """Content-free projection of an IngestReceipt (allowlist only)."""
    return {
        "event_id": getattr(receipt, "event_id", None),
        "status": getattr(receipt, "status", None),
        "index_status": getattr(receipt, "index_status", None),
        "attempts": getattr(receipt, "attempts", None),
        "last_error_code": getattr(receipt, "last_error_code", None),
    }


def _retry_report_projection(report) -> dict:
    return {
        "attempted": report.attempted,
        "succeeded": report.succeeded,
        "degraded": report.degraded,
        "failed_retryable": report.failed_retryable,
        "failed_terminal": report.failed_terminal,
    }


def _snapshot_report_projection(snap_result, restore_result) -> dict:
    return {
        "integrity_check": snap_result.get("integrity_check"),
        "restore_integrity_check": restore_result.get("integrity_check"),
        "restored": restore_result.get("restored"),
    }


def _migration_report_projection(report) -> dict:
    return {
        "added": report.get("added"),
        "would_add": report.get("would_add"),
        "tables_would_add": sorted(report.get("tables_would_add") or []),
    }


def assert_content_free(report: dict) -> None:
    blob = json.dumps(report, default=str)
    for forbidden in (
        "synthetic secret",
        "sk-abcdefghij0123456789",
        "deployment completed at capacity threshold",
        "api_key=",
        "<analysis>",
        "APPROVAL RECEIPT:",
        "/home/bell",
    ):
        assert forbidden not in blob, f"report leaked: {forbidden!r}"


def _run_compact_trial_sequence(tmp_path, monkeypatch) -> list:
    """Smallest composition of cases 1.1, 1.3, 1.7, 1.8; collects projections."""
    reports: list = []

    # --- 1.1 exactly-once ingest ---
    from mnemosyne.core.beam import BeamMemory

    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    config_module.MnemosyneConfig.reset_instance()
    beam = BeamMemory(session_id="trial-sess", db_path=tmp_path / "trial.db")
    event = _event(
        event_id="evt-trial-1",
        content="deployment completed at capacity threshold",
    )
    statuses = [beam.remember_event(event).status for _ in range(5)]
    reports.append(
        {
            "case": "exactly_once",
            "stored": statuses.count("stored"),
            "duplicate": statuses.count("duplicate"),
        }
    )

    # --- 1.3 admission reject ---
    secret = _event(
        event_id="evt-trial-2",
        content="api_key=sk-abcdefghij0123456789",
    )
    r1 = beam.remember_event(secret)
    reports.append(
        {
            "case": "admission_reject",
            "projection": _ingest_report_projection(r1),
        }
    )

    # --- 1.7 snapshot/restore ---
    from mnemosyne.core.memory import init_db
    from mnemosyne.dr import snapshot

    source = tmp_path / "source.db"
    init_db(source)
    snap_result = snapshot.create_isolated_snapshot(source, tmp_path / "snaps")
    snap_path = Path(snap_result["snapshot_path"])
    target = tmp_path / "restored.db"
    restore_result = snapshot.restore_isolated_snapshot(snap_path, target)
    reports.append(
        {
            "case": "snapshot_restore",
            "projection": _snapshot_report_projection(snap_result, restore_result),
        }
    )

    # --- 1.8 Dream lifecycle ---
    dream_dir = tmp_path / "dream"
    dream_dir.mkdir()
    _isolate_config(dream_dir, monkeypatch)
    from mnemosyne.core import dream, shmr

    dbeam = BeamMemory(session_id="dream-sess", db_path=dream_dir / "dream.db")
    _seed_facts(
        dbeam,
        [
            {
                "fact_id": "f1",
                "subject": "svc-a",
                "predicate": "latency",
                "object": "baseline threshold",
            },
            {
                "fact_id": "f2",
                "subject": "svc-a",
                "predicate": "latency",
                "object": "baseline threshold amended",
            },
        ],
    )
    shmr._init_proposal_schema(dbeam.conn)
    dbeam.conn.execute(
        "INSERT INTO shmr_proposals "
        "(run_id, cluster_id, session_id, scope_json, cited_source_ids, "
        "subject, predicate, object, confidence, action, target_source_id, "
        "rationale, status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "shmr_trial",
            "cc",
            "dream-sess",
            '{"session_id": "dream-sess"}',
            '["f1"]',
            "svc-a",
            "latency",
            "baseline",
            0.9,
            "create",
            None,
            "r",
            "proposed",
        ),
    )
    dbeam.conn.commit()
    run = dream.dream_plan(
        dbeam, scope={"session_id": "dream-sess"}, request_id="req-trial-1"
    )
    dream.dream_submit_receipt(
        dbeam,
        run.run_id,
        _pass_receipt("reviewer", "trial-reviewer", run.run_id, run.manifest_hash),
    )
    run = dream.dream_submit_receipt(
        dbeam,
        run.run_id,
        _pass_receipt("verifier", "trial-verifier", run.run_id, run.manifest_hash),
    )
    applied = dream.dream_apply(dbeam, run.run_id)
    undone = dream.dream_undo(dbeam, run.run_id)
    second_undo = dream.dream_undo(dbeam, run.run_id)
    reports.append(
        {
            "case": "dream_lifecycle",
            "projection": _dream_report_projection(applied, undone, second_undo),
        }
    )

    return reports


# ===========================================================================
# Trial driver tests
# ===========================================================================


def test_trial_exactly_once_ingest_reports_one_stored_rest_duplicate(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    config_module.MnemosyneConfig.reset_instance()
    from mnemosyne.core.beam import BeamMemory

    beam = BeamMemory(session_id="trial-sess", db_path=tmp_path / "trial.db")
    event = _event(
        event_id="evt-trial-1",
        content="deployment completed at capacity threshold",
    )

    statuses = [beam.remember_event(event).status for _ in range(5)]

    assert statuses.count("stored") == 1
    assert statuses.count("duplicate") == 4
    assert _table_count(beam, "working_memory") == 1
    assert _table_count(beam, "ingest_receipts") == 1
    assert _table_count(beam, "memory_events") == 1


def test_trial_conflict_preserves_original_payload_hash_and_no_new_rows(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    config_module.MnemosyneConfig.reset_instance()
    from mnemosyne.core.beam import BeamMemory

    beam = BeamMemory(session_id="trial-sess", db_path=tmp_path / "trial.db")
    original = _event(
        event_id="evt-trial-1",
        content="deployment completed at capacity threshold",
    )
    conflict = _event(
        event_id="evt-trial-1",
        content="deployment completed at capacity threshold amended",
    )

    first = beam.remember_event(original)
    second = beam.remember_event(conflict)

    assert first.status == "stored"
    assert second.status == "conflict"
    assert _table_count(beam, "working_memory") == 1
    assert _table_count(beam, "memory_events") == 1
    replay = beam.remember_event(original)
    assert replay.status == "duplicate"


def test_trial_admission_reject_is_zero_persistence_and_reusable(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    config_module.MnemosyneConfig.reset_instance()
    from mnemosyne.core.beam import BeamMemory

    beam = BeamMemory(session_id="trial-sess", db_path=tmp_path / "trial.db")
    secret_in_content = _event(
        event_id="evt-trial-2",
        content="api_key=sk-abcdefghij0123456789",
    )
    secret_in_metadata = _event(
        event_id="evt-trial-3",
        content="clean note",
        metadata={"token": "sk-abcdefghij0123456789"},
    )

    r1 = beam.remember_event(secret_in_content)
    r2 = beam.remember_event(secret_in_metadata)

    for receipt in (r1, r2):
        assert receipt.status == "rejected"
        assert receipt.index_status == "failed_terminal"
        assert receipt.last_error_code == "admission_rejected"
        assert "sk-abcdefghij0123456789" not in repr(receipt.metadata)
    assert _table_count(beam, "working_memory") == 0
    assert _table_count(beam, "ingest_receipts") == 0
    assert _table_count(beam, "memory_events") == 0

    corrected = _event(event_id="evt-trial-2", content="corrected safe content")
    assert beam.remember_event(corrected).status == "stored"


def test_trial_crash_before_embedding_leaves_pending_and_retry_completes_once(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    config_module.MnemosyneConfig.reset_instance()
    from mnemosyne.core.beam import BeamMemory
    import mnemosyne.core.inhale as inhale
    from mnemosyne.core.inhale import retry_pending_ingest

    beam = BeamMemory(session_id="trial-sess", db_path=tmp_path / "trial.db")

    real_finalize = inhale._finalize_receipt
    monkeypatch.setattr(
        beam_module._embeddings,
        "embed",
        lambda texts: (_ for _ in ()).throw(RuntimeError("embedding service down")),
    )
    monkeypatch.setattr(
        inhale,
        "_finalize_receipt",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("process died")),
    )

    with pytest.raises(RuntimeError):
        beam.remember_event(_event(event_id="evt-trial-4"))

    # Hermetic recovery: re-pin the deterministic embedding seams explicitly
    # (never monkeypatch.undo(), which would tear down the vec-style seams the
    # retry needs to reach succeeded==1 on a host without sqlite-vec).
    monkeypatch.setattr(inhale, "_finalize_receipt", real_finalize)
    monkeypatch.setattr(beam_module._embeddings, "available", lambda: True)
    monkeypatch.setattr(
        beam_module._embeddings,
        "embed",
        lambda texts: [[0.5] * beam_module.EMBEDDING_DIM for _ in texts],
    )
    monkeypatch.setattr(beam_module, "_wm_vec_available", lambda conn: True)
    monkeypatch.setattr(beam_module, "_store_working_embedding", lambda *a, **k: None)
    report = retry_pending_ingest(beam)

    assert report.attempted == 1
    assert report.succeeded == 1
    assert _table_count(beam, "working_memory") == 1
    assert _table_count(beam, "ingest_receipts") == 1
    assert _table_count(beam, "memory_events") == 1


def test_trial_concurrent_duplicate_race_produces_exactly_one_memory(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    config_module.MnemosyneConfig.reset_instance()
    import threading
    from mnemosyne.core.beam import BeamMemory

    db = tmp_path / "race.db"
    BeamMemory(session_id="race-sess", db_path=db)  # schema init serially
    exceptions = []
    barrier = threading.Barrier(2)
    results = [None, None]

    def worker(idx):
        try:
            b = BeamMemory(session_id="race-sess", db_path=db)
            b.conn.execute("PRAGMA busy_timeout=10000")
            barrier.wait(timeout=15)
            results[idx] = b.remember_event(_event(event_id="evt-trial-5"))
        except BaseException as exc:
            exceptions.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not exceptions, f"threads raised: {exceptions}"
    statuses = sorted(r.status for r in results if r is not None)
    assert statuses == ["duplicate", "stored"]
    from mnemosyne.core import beam as beam_module

    conn = beam_module._get_connection(db)
    assert conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM ingest_receipts").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0] == 1


def test_trial_migration_dry_run_leaves_fingerprint_unchanged(tmp_path):
    db_path = _fresh_e7_bank(tmp_path)  # pre-E7 legacy bank with WAL seed
    before = _schema_fingerprint(db_path)

    from mnemosyne.migrations.e7_311_tables import migrate_311_tables

    report = migrate_311_tables(db_path, dry_run=True)

    assert report["added"] == 0
    assert report["would_add"] >= 1
    assert "memory_events" in report["tables_would_add"]
    assert _schema_fingerprint(db_path) == before


def _seed_plain_database(path: Path) -> Path:
    """Small ordinary (rollback-journal) SQLite database, like the snapshot
    suite's _seed_database. Plain-journal (not WAL) so the source has no
    -wal/-shm sidecars and a fingerprint comparison meaningfully proves the
    snapshot opened the source read-only (no sidecar churn)."""
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    conn.executemany("INSERT INTO t VALUES (?, ?)", [(1, "a"), (2, "b")])
    conn.commit()
    conn.close()
    return path


def test_trial_snapshot_is_read_only_and_restore_round_trips(tmp_path):
    from mnemosyne.dr import snapshot

    source = _seed_plain_database(tmp_path / "source.db")
    before = _source_fingerprint(source)

    snap_result = snapshot.create_isolated_snapshot(source, tmp_path / "snaps")

    assert snap_result["integrity_check"] is True
    snap_path = Path(snap_result["snapshot_path"])
    assert _integrity(snap_path) == "ok"
    assert _journal_mode(snap_path) == "delete"
    assert not snap_path.with_name(snap_path.name + "-wal").exists()
    assert not snap_path.with_name(snap_path.name + "-shm").exists()
    assert _source_fingerprint(source) == before

    target = tmp_path / "restored.db"
    restore_result = snapshot.restore_isolated_snapshot(snap_path, target)
    assert restore_result["integrity_check"] is True
    assert _schema_fingerprint(target) == _schema_fingerprint(source)


def test_trial_dream_lifecycle_applies_and_idempotently_undoes(tmp_path, monkeypatch):
    _isolate_config(tmp_path, monkeypatch)  # config + offline seams
    from mnemosyne.core.beam import BeamMemory
    from mnemosyne.core import dream, shmr

    beam = BeamMemory(session_id="dream-sess", db_path=tmp_path / "dream.db")
    _seed_facts(
        beam,
        [
            {
                "fact_id": "f1",
                "subject": "svc-a",
                "predicate": "latency",
                "object": "baseline threshold",
            },
            {
                "fact_id": "f2",
                "subject": "svc-a",
                "predicate": "latency",
                "object": "baseline threshold amended",
            },
        ],
    )
    shmr._init_proposal_schema(beam.conn)
    beam.conn.execute(
        "INSERT INTO shmr_proposals "
        "(run_id, cluster_id, session_id, scope_json, cited_source_ids, "
        "subject, predicate, object, confidence, action, target_source_id, "
        "rationale, status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "shmr_trial",
            "cc",
            "dream-sess",
            '{"session_id": "dream-sess"}',
            '["f1"]',
            "svc-a",
            "latency",
            "baseline",
            0.9,
            "create",
            None,
            "r",
            "proposed",
        ),
    )
    beam.conn.commit()

    run = dream.dream_plan(
        beam, scope={"session_id": "dream-sess"}, request_id="req-trial-1"
    )
    assert run.state == "awaiting_approval"
    dream.dream_submit_receipt(
        beam,
        run.run_id,
        _pass_receipt("reviewer", "trial-reviewer", run.run_id, run.manifest_hash),
    )
    run = dream.dream_submit_receipt(
        beam,
        run.run_id,
        _pass_receipt("verifier", "trial-verifier", run.run_id, run.manifest_hash),
    )
    assert run.state == "ready"

    applied = dream.dream_apply(beam, run.run_id)
    assert applied.state == "applied"
    undone = dream.dream_undo(beam, run.run_id)
    assert undone.state == "undone"
    second_undo = dream.dream_undo(beam, run.run_id)
    assert second_undo.state == "undone"

    report = _dream_report_projection(applied, undone, second_undo)
    assert report["applied_state"] == "applied"
    assert report["undo_state"] == "undone"
    assert report["second_undo_state"] == "undone"
    assert_content_free(report)


def test_trial_reports_contain_no_content_or_secrets(tmp_path, monkeypatch):
    reports = _run_compact_trial_sequence(tmp_path, monkeypatch)
    blob = json.dumps(reports, default=str)
    for forbidden in (
        "synthetic secret",
        "sk-abcdefghij0123456789",
        "deployment completed at capacity threshold",
        "api_key=",
        "<analysis>",
        "APPROVAL RECEIPT:",
        "/home/bell",
    ):
        assert forbidden not in blob
