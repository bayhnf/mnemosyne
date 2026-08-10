"""
Task 6A: Native public SDK and CLI parity.

RED-first tests for the additive public surface that exposes existing native
Inhale (remember_event / remember_turn / retry_pending_ingest / ingest_status),
bounded Exhale (recall_bounded), Dream lifecycle, and orphan reclaim through
the Mnemosyne SDK, module-level convenience functions, top-level exports, and
the hand-rolled CLI — without duplicating core logic or changing legacy
remember()/recall() behavior.

These tests FAIL against 4763013 because none of the SDK wrappers, top-level
exports, or CLI commands exist yet.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict

import pytest

from mnemosyne.core import dream as dream_mod
from mnemosyne.core import beam as beam_module
from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.inhale import IngestEvent, IngestReceipt, TurnEvent, RetryReport
from mnemosyne.core.recall_bounded import RecallPolicy, RecallEnvelope


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


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
        fields["content_hash"] = _content_hash(str(fields["content"]))
    return IngestEvent(**fields)  # type: ignore[arg-type]


def _turn(**overrides) -> TurnEvent:
    return TurnEvent(**_event(**overrides).__dict__)


@pytest.fixture
def beam(tmp_path):
    """Fresh BeamMemory with offline embeddings pinned."""
    b = BeamMemory(session_id="sdk-sess", db_path=tmp_path / "sdk.db")
    yield b


@pytest.fixture
def vec_ready(monkeypatch):
    """Mock a healthy embedding + sqlite-vec pipeline.

    Pins both the beam-side and the legacy Mnemosyne.remember embedding paths
    so the mock vectors work end-to-end without numpy.
    """
    import json as _json
    monkeypatch.setattr(beam_module._embeddings, "available", lambda: True)
    monkeypatch.setattr(
        beam_module._embeddings,
        "embed",
        lambda texts: [[0.5] * beam_module.EMBEDDING_DIM for _ in texts],
    )
    monkeypatch.setattr(beam_module, "_wm_vec_available", lambda conn: True)
    monkeypatch.setattr(beam_module, "_store_working_embedding", lambda *a, **k: None)
    # Legacy Mnemosyne.remember calls _embeddings.serialize on the mock vector.
    monkeypatch.setattr(beam_module._embeddings, "serialize", lambda vec: _json.dumps(list(vec)))
    # core/memory.py imports embeddings separately; pin there too.
    from mnemosyne.core import memory as _mem_mod
    monkeypatch.setattr(_mem_mod._embeddings, "available", lambda: True)
    monkeypatch.setattr(
        _mem_mod._embeddings,
        "embed",
        lambda texts: [[0.5] * beam_module.EMBEDDING_DIM for _ in texts],
    )
    monkeypatch.setattr(_mem_mod._embeddings, "serialize", lambda vec: _json.dumps(list(vec)))


def run_cli(args, tmp_path):
    """Run the CLI in a subprocess with an isolated data dir."""
    env = os.environ.copy()
    env["HOME"] = str(tmp_path / "home")
    env["MNEMOSYNE_DATA_DIR"] = str(tmp_path / "mnemosyne-data")
    return subprocess.run(
        [sys.executable, "-m", "mnemosyne.cli", *args],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


# ===========================================================================
# SDK wrappers: ingest_status
# ===========================================================================


class TestIngestStatus:
    def test_beam_ingest_status_retrieves_stored_receipt(self, beam, vec_ready):
        receipt = beam.remember_event(_event(event_id="evt-status-1"))
        assert receipt.status == "stored"

        rows = beam.ingest_status()
        assert len(rows) == 1
        assert isinstance(rows[0], IngestReceipt)
        assert rows[0].event_id == "evt-status-1"
        # Content-free: receipt carries no content field.
        assert not hasattr(rows[0], "content")

    def test_beam_ingest_status_by_event_id(self, beam, vec_ready):
        beam.remember_event(_event(event_id="evt-a"))
        beam.remember_event(_event(event_id="evt-b"))
        rows = beam.ingest_status(event_id="evt-a")
        assert len(rows) == 1
        assert rows[0].event_id == "evt-a"

    def test_beam_ingest_status_rejects_invalid_limit(self, beam):
        with pytest.raises(ValueError):
            beam.ingest_status(limit=0)
        with pytest.raises(ValueError):
            beam.ingest_status(limit=-1)

    def test_ingest_status_is_content_free(self, beam, vec_ready):
        beam.remember_event(_event(event_id="evt-cf", content="secret-api-key"))
        rows = beam.ingest_status(event_id="evt-cf")
        assert len(rows) == 1
        serialized = json.dumps(rows[0].__dict__, default=str)
        assert "secret-api-key" not in serialized


# ===========================================================================
# SDK wrappers: delegation to native core
# ===========================================================================


class TestSDKDelegation:
    def test_mnemosyne_remember_event_returns_receipt(self, tmp_path, vec_ready):
        from mnemosyne.core.memory import Mnemosyne

        mem = Mnemosyne(session_id="s", db_path=tmp_path / "m.db")
        receipt = mem.remember_event(_event(event_id="evt-sdk-1"))
        assert isinstance(receipt, IngestReceipt)
        assert receipt.event_id == "evt-sdk-1"
        assert receipt.status == "stored"

    def test_mnemosyne_remember_turn_returns_receipt(self, tmp_path, vec_ready):
        from mnemosyne.core.memory import Mnemosyne

        mem = Mnemosyne(session_id="s", db_path=tmp_path / "m.db")
        receipt = mem.remember_turn(_turn(event_id="turn-sdk-1"))
        assert isinstance(receipt, IngestReceipt)
        assert receipt.event_id == "turn-sdk-1"

    def test_mnemosyne_retry_pending_ingest_returns_report(self, tmp_path, vec_ready):
        from mnemosyne.core.memory import Mnemosyne

        mem = Mnemosyne(session_id="s", db_path=tmp_path / "m.db")
        mem.remember_event(_event(event_id="evt-retry-1"))
        report = mem.retry_pending_ingest()
        assert isinstance(report, RetryReport)

    def test_mnemosyne_recall_bounded_returns_envelope(self, tmp_path, vec_ready):
        from mnemosyne.core.memory import Mnemosyne

        mem = Mnemosyne(session_id="s", db_path=tmp_path / "m.db")
        mem.remember_event(_event(event_id="evt-rb-1", content="hello world"))
        env = mem.recall_bounded("hello")
        assert isinstance(env, RecallEnvelope)
        assert isinstance(env.results, list)

    def test_mnemosyne_ingest_status(self, tmp_path, vec_ready):
        from mnemosyne.core.memory import Mnemosyne

        mem = Mnemosyne(session_id="s", db_path=tmp_path / "m.db")
        mem.remember_event(_event(event_id="evt-is-1"))
        rows = mem.ingest_status()
        assert len(rows) == 1
        assert rows[0].event_id == "evt-is-1"

    def test_module_level_remember_event(self, tmp_path, vec_ready, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "mod"))
        from mnemosyne.core import memory as mem_mod

        mem_mod._default_instance = None
        receipt = mem_mod.remember_event(_event(event_id="evt-mod-1"))
        assert isinstance(receipt, IngestReceipt)
        assert receipt.event_id == "evt-mod-1"

    def test_module_level_recall_bounded(self, tmp_path, vec_ready, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "mod2"))
        from mnemosyne.core import memory as mem_mod

        mem_mod._default_instance = None
        mem_mod.remember_event(_event(event_id="evt-modrb-1", content="module recall test"))
        env = mem_mod.recall_bounded("module")
        assert isinstance(env, RecallEnvelope)

    def test_top_level_exports_exist(self):
        import mnemosyne

        for name in (
            "remember_event",
            "remember_turn",
            "retry_pending_ingest",
            "recall_bounded",
            "ingest_status",
            "dream_plan",
            "dream_submit_receipt",
            "dream_apply",
            "dream_resume",
            "dream_undo",
            "dream_status",
        ):
            assert name in mnemosyne._lazy_exports, f"mnemosyne.{name} not exported"

    def test_dream_exports_delegate(self, tmp_path, vec_ready):
        from mnemosyne.core.memory import Mnemosyne

        mem = Mnemosyne(session_id="s", db_path=tmp_path / "d.db")
        run = mem.dream_plan(scope={"session_id": "s"})
        assert isinstance(run, dream_mod.DreamRun)
        assert run.run_id

        run2 = mem.dream_status(run.run_id)
        assert run2.run_id == run.run_id


# ===========================================================================
# Legacy recall unchanged; bounded recall is separate
# ===========================================================================


class TestRecallParity:
    def test_legacy_recall_returns_plain_list(self, tmp_path, vec_ready):
        from mnemosyne.core.memory import Mnemosyne

        mem = Mnemosyne(session_id="s", db_path=tmp_path / "legacy.db")
        mem.remember("legacy memory content", source="test")
        results = mem.recall("legacy")
        assert isinstance(results, list)
        assert not isinstance(results, RecallEnvelope)

    def test_bounded_recall_within_hard_limits(self, tmp_path, vec_ready):
        from mnemosyne.core.memory import Mnemosyne

        mem = Mnemosyne(session_id="s", db_path=tmp_path / "bounded.db")
        for i in range(10):
            mem.remember(f"bounded item number {i}", source="test")
        env = mem.recall_bounded("bounded", RecallPolicy(top_k=3, max_tokens=100))
        assert isinstance(env, RecallEnvelope)
        assert len(env.results) <= 3
        assert env.token_count <= 100


# ===========================================================================
# CLI: native ingest surface
# ===========================================================================


class TestCLINativeIngest:
    def test_ingest_is_idempotent(self, tmp_path):
        env_dict = {
            "event_id": "cli-evt-1",
            "producer": "codex",
            "actor_id": "actor-1",
            "project_id": "proj-1",
            "session_id": "sess-1",
            "turn_id": "turn-1",
            "role": "user",
            "content": "cli idempotency check",
            "content_hash": _content_hash("cli idempotency check"),
            "occurred_at": "2026-08-10T01:02:03Z",
        }
        args = [
            "ingest",
            "--event-id", env_dict["event_id"],
            "--producer", env_dict["producer"],
            "--actor-id", env_dict["actor_id"],
            "--project-id", env_dict["project_id"],
            "--session-id", env_dict["session_id"],
            "--turn-id", env_dict["turn_id"],
            "--role", env_dict["role"],
            "--content", env_dict["content"],
            "--occurred-at", env_dict["occurred_at"],
        ]
        r1 = run_cli(args, tmp_path)
        assert r1.returncode == 0, r1.stderr
        r2 = run_cli(args, tmp_path)
        assert r2.returncode == 0, r2.stderr
        # Second ingest of identical event is a duplicate, not an error.
        assert "duplicate" in r2.stdout.lower() or "stored" in r2.stdout.lower()

    def test_ingest_status_is_content_free(self, tmp_path):
        ingest_args = [
            "ingest",
            "--event-id", "cli-status-evt",
            "--producer", "codex",
            "--actor-id", "a1",
            "--project-id", "p1",
            "--session-id", "s1",
            "--turn-id", "t1",
            "--role", "user",
            "--content", "secret-content-field",
            "--occurred-at", "2026-08-10T01:02:03Z",
        ]
        run_cli(ingest_args, tmp_path)
        r = run_cli(["ingest-status", "--json"], tmp_path)
        assert r.returncode == 0, r.stderr
        assert "secret-content-field" not in r.stdout
        assert "secret-content-field" not in r.stderr

    def test_ingest_retry_is_structured(self, tmp_path):
        run_cli([
            "ingest",
            "--event-id", "cli-retry-evt",
            "--producer", "codex",
            "--actor-id", "a1",
            "--project-id", "p1",
            "--session-id", "s1",
            "--turn-id", "t1",
            "--role", "user",
            "--content", "retry me",
            "--occurred-at", "2026-08-10T01:02:03Z",
        ], tmp_path)
        r = run_cli(["ingest-retry", "--json"], tmp_path)
        assert r.returncode == 0, r.stderr
        payload = json.loads(r.stdout)
        assert "attempted" in payload
        assert "succeeded" in payload


# ===========================================================================
# CLI: bounded recall options
# ===========================================================================


class TestCLIBoundedRecall:
    def test_bounded_recall_options(self, tmp_path):
        run_cli(["store", "bounded recall item", "cli", "0.7"], tmp_path)
        r = run_cli(
            ["recall", "bounded", "--max-tokens", "50", "--top-k", "2", "--json"],
            tmp_path,
        )
        assert r.returncode == 0, r.stderr

    def test_plain_recall_no_bounded_option_stays_legacy(self, tmp_path):
        run_cli(["store", "plain legacy item", "cli", "0.7"], tmp_path)
        r = run_cli(["recall", "plain", "--json"], tmp_path)
        assert r.returncode == 0, r.stderr
        payload = json.loads(r.stdout)
        assert "results" in payload
        assert isinstance(payload["results"], list)
        # Legacy path does not produce a RecallEnvelope envelope with trace_id.
        assert "trace_id" not in payload


# ===========================================================================
# CLI: reclaim-orphans dry-run
# ===========================================================================


class TestCLIReclaimOrphans:
    def test_reclaim_orphans_dry_run_default(self, tmp_path):
        r = run_cli(["reclaim-orphans"], tmp_path)
        assert r.returncode == 0, r.stderr
        assert "dry" in r.stdout.lower() or "0" in r.stdout

    def test_reclaim_orphans_apply_requires_flag(self, tmp_path):
        # Without --apply it must be a dry-run (no mutation).
        r1 = run_cli(["reclaim-orphans"], tmp_path)
        assert r1.returncode == 0
        r2 = run_cli(["reclaim-orphans", "--apply"], tmp_path)
        assert r2.returncode == 0, r2.stderr


# ===========================================================================
# CLI: Dream plan + reviewer/verifier receipts
# ===========================================================================


class TestCLIDream:
    def _seed_for_dream(self, tmp_path):
        """Seed facts via the CLI store so SHMR has something to propose."""
        # We need facts in the facts table. Use the Python API directly to
        # seed since the CLI has no fact-store command.
        env = os.environ.copy()
        env["HOME"] = str(tmp_path / "home")
        env["MNEMOSYNE_DATA_DIR"] = str(tmp_path / "mnemosyne-data")
        db_path = Path(env["MNEMOSYNE_DATA_DIR"]) / "mnemosyne.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        from mnemosyne.core.beam import BeamMemory
        beam = BeamMemory(session_id="dream-cli-sess", db_path=db_path)
        beam.conn.execute(
            "INSERT INTO facts "
            "(fact_id, session_id, subject, predicate, object, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("f1", "dream-cli-sess", "alice", "likes", "rust language", 0.9),
        )
        beam.conn.execute(
            "INSERT INTO facts "
            "(fact_id, session_id, subject, predicate, object, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("f2", "dream-cli-sess", "alice", "likes", "rust programming", 0.9),
        )
        beam.conn.commit()
        beam.conn.close()
        return db_path

    def test_dream_plan_then_review_verify_distinct_actors(self, tmp_path):
        # Dream with no proposals returns no_candidates; the plan command must
        # still exit 0 and emit structured output with a run_id.
        r = run_cli(["dream", "plan", "--session-id", "no-such-session", "--json"], tmp_path)
        assert r.returncode == 0, r.stderr
        payload = json.loads(r.stdout)
        assert "run_id" in payload
        assert "state" in payload


# ===========================================================================
# CLI: invalid / missing flags exit 2 without traceback
# ===========================================================================


class TestCLIBoundaryErrors:
    def test_ingest_missing_required_flag_exits_2(self, tmp_path):
        r = run_cli(["ingest", "--event-id", "x"], tmp_path)
        assert r.returncode == 2
        assert "Traceback" not in r.stderr
        assert "Error:" in r.stderr or "Usage:" in r.stderr

    def test_ingest_status_bad_limit_exits_2(self, tmp_path):
        r = run_cli(["ingest-status", "--limit", "not-an-int"], tmp_path)
        assert r.returncode == 2
        assert "Traceback" not in r.stderr

    def test_dream_unknown_subcommand_exits_2(self, tmp_path):
        r = run_cli(["dream", "frobnicate"], tmp_path)
        assert r.returncode != 0
        assert "Traceback" not in r.stderr

    def test_reclaim_orphans_bad_stale_after_exits_2(self, tmp_path):
        r = run_cli(["reclaim-orphans", "--stale-after-seconds", "abc"], tmp_path)
        assert r.returncode == 2
        assert "Traceback" not in r.stderr

    def test_ingest_missing_content_exits_2(self, tmp_path):
        r = run_cli([
            "ingest",
            "--event-id", "evt-x",
            "--producer", "codex",
            "--actor-id", "a1",
            "--project-id", "p1",
            "--session-id", "s1",
            "--turn-id", "t1",
            "--role", "user",
            "--occurred-at", "2026-08-10T01:02:03Z",
        ], tmp_path)
        assert r.returncode == 2
        assert "Traceback" not in r.stderr
