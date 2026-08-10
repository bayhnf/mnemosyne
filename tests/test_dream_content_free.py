"""
Task 15: Dream diagnostics must stay content-free.

Every persisted ``failure_reason`` on the six raw-exception paths must be an
existing static error code, and the two Dream WARNING sites must not
interpolate exception text. Synthetic canary strings injected into the
exceptions must never reach the returned ``DreamRun``, the stored
``dream_runs.failure_reason`` column, or captured WARNING records.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import sqlite3
from datetime import datetime, timezone

import pytest

from mnemosyne.core import config as config_module
from mnemosyne.core import dream
from mnemosyne.core import shmr
from mnemosyne.core.beam import BeamMemory


def _force_offline_embeddings(monkeypatch):
    """Pin SHMR's embedding seam to the deterministic lexical fallback."""
    from mnemosyne.core import embeddings as _emb

    monkeypatch.setattr(shmr, "_embedding_fn", lambda: None, raising=True)

    def _counting_guard(_texts):
        raise AssertionError(
            "Dream test reached the network embedding path; tests must stay "
            "offline via the deterministic lexical fallback."
        )

    monkeypatch.setattr(_emb, "embed", _counting_guard, raising=True)


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


class _RecordingLLM:
    """Deterministic LLM seam for SHMR propose_harmony."""

    def __init__(self, replies):
        self.replies = list(replies)

    def __call__(self, prompt, system=""):
        if not self.replies:
            return ""
        return self.replies.pop(0)


def _two_cluster_facts():
    return [
        {
            "fact_id": "f1",
            "subject": "alice",
            "predicate": "likes",
            "object": "the rust programming language for systems work",
        },
        {
            "fact_id": "f2",
            "subject": "alice",
            "predicate": "likes",
            "object": "rust language for systems programming",
        },
        {
            "fact_id": "f3",
            "subject": "bob",
            "predicate": "uses",
            "object": "python for data analysis pipelines daily",
        },
        {
            "fact_id": "f4",
            "subject": "bob",
            "predicate": "uses",
            "object": "python in data analysis pipelines",
        },
    ]


def _single_cluster_replies():
    return [
        json.dumps(
            [
                {
                    "subject": "alice",
                    "predicate": "prefers",
                    "object": "rust",
                    "confidence": 0.9,
                    "action": "create",
                    "target_fact_id": "f1",
                    "rationale": "both mention rust",
                }
            ]
        ),
        json.dumps(
            [
                {
                    "subject": "bob",
                    "predicate": "prefers",
                    "object": "python",
                    "confidence": 0.88,
                    "action": "create",
                    "target_fact_id": "f3",
                    "rationale": "both mention python",
                }
            ]
        ),
    ]


@pytest.fixture(autouse=True)
def _isolate_config_and_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    config_module.MnemosyneConfig.reset_instance()
    _force_offline_embeddings(monkeypatch)
    yield
    config_module.MnemosyneConfig.reset_instance()


@pytest.fixture
def beam(tmp_path):
    b = BeamMemory(session_id="dream-sess", db_path=tmp_path / "dream.db")
    _seed_facts(b, _two_cluster_facts())
    return b


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pass_receipt(role, actor_id, run_id, manifest_hash):
    return {
        "role": role,
        "actor_id": actor_id,
        "run_id": run_id,
        "manifest_hash": manifest_hash,
        "verdict": "PASS",
        "reason_code": "ok",
        "timestamp": _now_iso(),
    }


def _force_proposals(beam):
    llm = _RecordingLLM(_single_cluster_replies())
    shmr.propose_harmony(beam, llm_call=llm, similarity_threshold=0.4)


def _plan(beam):
    return dream.dream_plan(beam, scope={"session_id": beam.session_id})


def _ready_run(beam):
    _force_proposals(beam)
    run = _plan(beam)
    run = dream.dream_submit_receipt(
        beam,
        run.run_id,
        _pass_receipt("reviewer", "r1", run.run_id, run.manifest_hash),
    )
    run = dream.dream_submit_receipt(
        beam,
        run.run_id,
        _pass_receipt("verifier", "v1", run.run_id, run.manifest_hash),
    )
    assert run.state == "ready"
    return run


def _applied_run(beam):
    run = dream.dream_apply(beam, _ready_run(beam).run_id)
    assert run.state == "applied"
    return run


def _fail_on(monkeypatch, beam, needle, exc_factory):
    original_execute = beam.conn.execute

    def flaky_execute(sql, *params):
        if isinstance(sql, str) and needle in sql:
            raise exc_factory()
        return original_execute(sql, *params)

    monkeypatch.setattr(beam.conn, "execute", flaky_execute)


def _assert_static_failure(beam, run, code, canary):
    assert run.failure_reason == code
    serialized = json.dumps(dataclasses.asdict(run), default=str)
    assert canary not in serialized
    row = beam.conn.execute(
        "SELECT failure_reason FROM dream_runs WHERE run_id = ?",
        (run.run_id,),
    ).fetchone()
    assert row is not None
    assert row["failure_reason"] == code


class TestDreamPlanContentFree:
    def test_lock_error_persists_database_busy(self, beam, monkeypatch):
        canary = "CANARY_PLAN_LOCK"
        _fail_on(
            monkeypatch,
            beam,
            "BEGIN IMMEDIATE",
            lambda: sqlite3.OperationalError(f"database is locked {canary}"),
        )
        run = dream.dream_plan(beam, scope={"session_id": beam.session_id})
        _assert_static_failure(beam, run, "database_busy", canary)

    def test_non_lock_error_persists_integrity_failure(self, beam, monkeypatch):
        canary = "CANARY_PLAN_OTHER"
        _fail_on(
            monkeypatch,
            beam,
            "BEGIN IMMEDIATE",
            lambda: sqlite3.OperationalError(f"simulated plan failure {canary}"),
        )
        run = dream.dream_plan(beam, scope={"session_id": beam.session_id})
        _assert_static_failure(beam, run, "integrity_failure", canary)


class TestDreamApplyContentFree:
    def test_busy_error_persists_database_busy(self, beam, monkeypatch):
        canary = "CANARY_APPLY_BUSY"
        ready = _ready_run(beam)
        _fail_on(
            monkeypatch,
            beam,
            "BEGIN IMMEDIATE",
            lambda: sqlite3.OperationalError(f"database is locked {canary}"),
        )
        run = dream.dream_apply(beam, ready.run_id)
        _assert_static_failure(beam, run, "database_busy", canary)

    def test_integrity_error_persists_integrity_failure(self, beam, monkeypatch):
        canary = "CANARY_APPLY_INTEGRITY"
        ready = _ready_run(beam)
        _fail_on(
            monkeypatch,
            beam,
            "INSERT INTO canonical_facts",
            lambda: sqlite3.IntegrityError(
                f"UNIQUE constraint failed: canonical_facts.id {canary}"
            ),
        )
        run = dream.dream_apply(beam, ready.run_id)
        _assert_static_failure(beam, run, "integrity_failure", canary)

    def test_generic_error_persists_integrity_failure(self, beam, monkeypatch):
        canary = "CANARY_APPLY_GENERIC"
        ready = _ready_run(beam)
        _fail_on(
            monkeypatch,
            beam,
            "INSERT INTO canonical_facts",
            lambda: RuntimeError(f"unexpected apply failure {canary}"),
        )
        run = dream.dream_apply(beam, ready.run_id)
        _assert_static_failure(beam, run, "integrity_failure", canary)


class TestDreamUndoContentFree:
    def test_non_lock_error_persists_integrity_failure(self, beam, monkeypatch):
        canary = "CANARY_UNDO_OTHER"
        applied = _applied_run(beam)
        _fail_on(
            monkeypatch,
            beam,
            "DELETE FROM canonical_facts",
            lambda: sqlite3.OperationalError(f"simulated undo failure {canary}"),
        )
        run = dream.dream_undo(beam, applied.run_id)
        _assert_static_failure(beam, run, "integrity_failure", canary)

    def test_generic_error_persists_integrity_failure(self, beam, monkeypatch):
        canary = "CANARY_UNDO_GENERIC"
        applied = _applied_run(beam)
        _fail_on(
            monkeypatch,
            beam,
            "DELETE FROM canonical_facts",
            lambda: RuntimeError(f"unexpected undo failure {canary}"),
        )
        run = dream.dream_undo(beam, applied.run_id)
        _assert_static_failure(beam, run, "integrity_failure", canary)


class TestDreamWarningsContentFree:
    def test_config_snapshot_warning_is_content_free(self, monkeypatch, caplog):
        canary = "CANARY_CONFIG_READ"

        def _broken_get_config():
            raise RuntimeError(f"config read failure {canary}")

        monkeypatch.setattr(
            config_module, "get_config", _broken_get_config, raising=True
        )
        with caplog.at_level(logging.WARNING, logger="mnemosyne.core.dream"):
            result = dream._config_snapshot()
        assert result == {"config_unavailable": True}
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("config audit snapshot failed" in r.getMessage() for r in warnings)
        assert all(canary not in r.getMessage() for r in warnings)

    def test_set_dream_active_warning_is_content_free(self, monkeypatch, caplog):
        canary = "CANARY_CONFIG_WRITE"

        def _broken_get_config():
            raise RuntimeError(f"config write failure {canary}")

        monkeypatch.setattr(
            config_module, "get_config", _broken_get_config, raising=True
        )
        with caplog.at_level(logging.WARNING, logger="mnemosyne.core.dream"):
            result = dream._set_dream_active(True)
        assert result == "validation_failed"
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            "dream_active gate could not be set" in r.getMessage() for r in warnings
        )
        assert all(canary not in r.getMessage() for r in warnings)
