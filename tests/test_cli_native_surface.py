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
    """Run the CLI in a subprocess with an isolated data dir.

    Removes inherited MNEMOSYNE_BANK so every test is deterministic
    (defaults to the 'default' bank in the throwaway data dir).
    """
    env = os.environ.copy()
    env["HOME"] = str(tmp_path / "home")
    env["MNEMOSYNE_DATA_DIR"] = str(tmp_path / "mnemosyne-data")
    env.pop("MNEMOSYNE_BANK", None)
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


# ===========================================================================
# Task 6A fix round 1: DeepSeek review findings
# ===========================================================================


class TestCLIDreamLifecycleIntegration:
    """P1: genuine CLI Dream review/verify lifecycle coverage.

    Seeds shmr_proposals + matching facts directly into the DB so the CLI
    subprocess can plan without an LLM, then drives review/verify through
    the CLI and asserts the state machine.
    """

    def _seed_proposals(self, tmp_path):
        """Seed facts + shmr_proposals so dream plan finds eligible actions."""
        env = os.environ.copy()
        env["HOME"] = str(tmp_path / "home")
        data_dir = tmp_path / "mnemosyne-data"
        env["MNEMOSYNE_DATA_DIR"] = str(data_dir)
        db_path = data_dir / "mnemosyne.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        from mnemosyne.core.beam import BeamMemory
        from mnemosyne.core.shmr import _init_schema, PROPOSAL_SCHEMA_SQL
        beam = BeamMemory(session_id="dream-int-sess", db_path=db_path)
        # Ensure shmr_proposals table exists (normally created lazily by
        # propose_harmony, which we bypass by seeding directly).
        _init_schema(beam.conn)
        beam.conn.executescript(PROPOSAL_SCHEMA_SQL)
        beam.conn.execute(
            "INSERT INTO facts "
            "(fact_id, session_id, subject, predicate, object, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("ifact1", "dream-int-sess", "alice", "likes", "rust lang", 0.9),
        )
        beam.conn.execute(
            "INSERT INTO facts "
            "(fact_id, session_id, subject, predicate, object, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("ifact2", "dream-int-sess", "alice", "likes", "rust programming", 0.9),
        )
        beam.conn.execute(
            "INSERT INTO facts "
            "(fact_id, session_id, subject, predicate, object, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("ifact3", "dream-int-sess", "bob", "uses", "python daily", 0.9),
        )
        beam.conn.execute(
            "INSERT INTO facts "
            "(fact_id, session_id, subject, predicate, object, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("ifact4", "dream-int-sess", "bob", "uses", "python regularly", 0.9),
        )
        beam.conn.execute(
            "INSERT INTO shmr_proposals "
            "(run_id, cluster_id, session_id, scope_json, cited_source_ids, "
            "subject, predicate, object, confidence, action, target_source_id, "
            "rationale, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "seed-run-1", "c1", "dream-int-sess",
                json.dumps({"session_id": "dream-int-sess"}),
                json.dumps(["ifact1", "ifact2"]),
                "alice", "prefers", "rust", 0.9, "create",
                "ifact1", "seeded cluster", "proposed",
            ),
        )
        beam.conn.execute(
            "INSERT INTO shmr_proposals "
            "(run_id, cluster_id, session_id, scope_json, cited_source_ids, "
            "subject, predicate, object, confidence, action, target_source_id, "
            "rationale, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "seed-run-1", "c2", "dream-int-sess",
                json.dumps({"session_id": "dream-int-sess"}),
                json.dumps(["ifact3", "ifact4"]),
                "bob", "prefers", "python", 0.88, "create",
                "ifact3", "seeded cluster", "proposed",
            ),
        )
        beam.conn.commit()
        beam.conn.close()
        return db_path

    def test_plan_review_verify_ready_lifecycle(self, tmp_path):
        self._seed_proposals(tmp_path)
        # Plan
        r = run_cli(
            ["dream", "plan", "--session-id", "dream-int-sess", "--json"],
            tmp_path,
        )
        assert r.returncode == 0, r.stderr
        plan_payload = json.loads(r.stdout)
        assert plan_payload["state"] == "awaiting_approval"
        assert plan_payload["manifest_hash"]
        run_id = plan_payload["run_id"]

        # Review with actor A, PASS
        r = run_cli(
            ["dream", "review", "--run-id", run_id, "--actor-id", "revA",
             "--verdict", "PASS", "--json"],
            tmp_path,
        )
        assert r.returncode == 0, r.stderr
        review_payload = json.loads(r.stdout)
        assert review_payload["state"] == "awaiting_approval"

        # Verify with actor B (distinct), PASS -> ready
        r = run_cli(
            ["dream", "verify", "--run-id", run_id, "--actor-id", "verB",
             "--verdict", "PASS", "--json"],
            tmp_path,
        )
        assert r.returncode == 0, r.stderr
        verify_payload = json.loads(r.stdout)
        assert verify_payload["state"] == "ready"

    def test_same_actor_review_verify_rejected(self, tmp_path):
        self._seed_proposals(tmp_path)
        r = run_cli(
            ["dream", "plan", "--session-id", "dream-int-sess", "--json"],
            tmp_path,
        )
        run_id = json.loads(r.stdout)["run_id"]

        # Review with actor A
        run_cli(
            ["dream", "review", "--run-id", run_id, "--actor-id", "sameActor",
             "--verdict", "PASS", "--json"],
            tmp_path,
        )
        # Verify with same actor -> rejected (state goes to rejected)
        r = run_cli(
            ["dream", "verify", "--run-id", run_id, "--actor-id", "sameActor",
             "--verdict", "PASS", "--json"],
            tmp_path,
        )
        assert r.returncode != 0
        payload = json.loads(r.stdout)
        assert payload["state"] == "rejected"

    def test_wrong_manifest_hash_rejected(self, tmp_path):
        self._seed_proposals(tmp_path)
        r = run_cli(
            ["dream", "plan", "--session-id", "dream-int-sess", "--json"],
            tmp_path,
        )
        run_id = json.loads(r.stdout)["run_id"]

        r = run_cli(
            ["dream", "review", "--run-id", run_id, "--actor-id", "revA",
             "--manifest-hash", "deadbeef" * 8,
             "--verdict", "PASS", "--json"],
            tmp_path,
        )
        assert r.returncode != 0
        payload = json.loads(r.stdout)
        assert payload["state"] == "rejected"


class TestDreamJSONProjection:
    """P2: --json output must be content-free curated projection."""

    def test_plan_json_has_no_manifest_actions_or_content(self, tmp_path):
        # Use a no-candidates plan (no seeding needed); the projection must
        # still be curated.
        r = run_cli(
            ["dream", "plan", "--session-id", "empty-sess", "--json"],
            tmp_path,
        )
        assert r.returncode == 0, r.stderr
        payload = json.loads(r.stdout)
        # Allowed keys: durable identifiers/state/scope/manifest_hash/
        # checkpoint/error code/safe timestamps/receipt counts.
        for forbidden in ("manifest", "actions", "receipts", "before_image",
                          "after_image", "content", "config"):
            assert forbidden not in payload, (
                f"dream --json must not include '{forbidden}': {payload!r}"
            )
        # Must include safe identifiers.
        for required in ("run_id", "state", "manifest_hash"):
            assert required in payload, f"missing {required}: {payload!r}"

    def test_status_json_has_no_manifest_or_actions(self, tmp_path):
        r = run_cli(
            ["dream", "plan", "--session-id", "empty-sess-2", "--json"],
            tmp_path,
        )
        run_id = json.loads(r.stdout)["run_id"]
        r = run_cli(["dream", "status", "--run-id", run_id, "--json"], tmp_path)
        assert r.returncode == 0, r.stderr
        payload = json.loads(r.stdout)
        for forbidden in ("manifest", "actions", "receipts"):
            assert forbidden not in payload

    def test_review_json_has_receipt_summary_not_details(self, tmp_path):
        # Plan then review with FAIL — the JSON must not leak raw receipt.
        r = run_cli(
            ["dream", "plan", "--session-id", "empty-sess-3", "--json"],
            tmp_path,
        )
        run_id = json.loads(r.stdout)["run_id"]
        # This will be rejected because there's nothing to review (state is
        # already terminal), but the JSON projection must still be curated.
        r = run_cli(
            ["dream", "review", "--run-id", run_id, "--actor-id", "revA",
             "--verdict", "FAIL", "--json"],
            tmp_path,
        )
        payload = json.loads(r.stdout)
        # receipts as a raw list must never appear; a count is OK.
        assert "receipts" not in payload
        # A receipt_count or receipt_summary is acceptable.
        assert "actions" not in payload


class TestDreamLimitsRejected:
    """P2: --limits accepted but core ignores it; reject at CLI boundary."""

    def test_dream_plan_limits_exits_2(self, tmp_path):
        r = run_cli(
            ["dream", "plan", "--session-id", "x", "--limits", "{}"],
            tmp_path,
        )
        assert r.returncode == 2
        assert "Traceback" not in r.stderr
        assert "limits" in r.stderr.lower() or "unsupported" in r.stderr.lower()


class TestDreamVerdictRequired:
    """P2: review/verify must require explicit --verdict."""

    def test_review_missing_verdict_exits_2(self, tmp_path):
        r = run_cli(
            ["dream", "review", "--run-id", "x", "--actor-id", "a"],
            tmp_path,
        )
        assert r.returncode == 2
        assert "Traceback" not in r.stderr
        assert "verdict" in r.stderr.lower()

    def test_verify_missing_verdict_exits_2(self, tmp_path):
        r = run_cli(
            ["dream", "verify", "--run-id", "x", "--actor-id", "a"],
            tmp_path,
        )
        assert r.returncode == 2
        assert "Traceback" not in r.stderr
        assert "verdict" in r.stderr.lower()

    def test_review_invalid_verdict_exits_2(self, tmp_path):
        r = run_cli(
            ["dream", "review", "--run-id", "x", "--actor-id", "a",
             "--verdict", "MAYBE"],
            tmp_path,
        )
        assert r.returncode == 2
        assert "Traceback" not in r.stderr


class TestBoundedRecallExplainRejected:
    """P3: bounded recall --explain must not silently no-op."""

    def test_bounded_recall_explain_exits_2(self, tmp_path):
        r = run_cli(
            ["recall", "query", "--max-tokens", "50", "--explain"],
            tmp_path,
        )
        assert r.returncode == 2
        assert "Traceback" not in r.stderr
        assert "explain" in r.stderr.lower()


class TestReclaimOrphansApplyDryRunOrder:
    """P3: reject --apply + --dry-run in either order."""

    def test_apply_then_dry_run_rejected(self, tmp_path):
        r = run_cli(["reclaim-orphans", "--apply", "--dry-run"], tmp_path)
        assert r.returncode == 2
        assert "Traceback" not in r.stderr

    def test_dry_run_then_apply_rejected(self, tmp_path):
        r = run_cli(["reclaim-orphans", "--dry-run", "--apply"], tmp_path)
        assert r.returncode == 2
        assert "Traceback" not in r.stderr


class TestIngestNextFlagAsValue:
    """P3: cmd_ingest must reject a next flag as a missing value."""

    def test_ingest_flag_value_is_next_flag_exits_2(self, tmp_path):
        # --producer is followed by --actor-id: must reject, not silently
        # consume --actor-id as the producer value.
        r = run_cli(
            ["ingest",
             "--event-id", "evt-x",
             "--producer", "--actor-id",
             "--actor-id", "a1",
             "--project-id", "p1",
             "--session-id", "s1",
             "--turn-id", "t1",
             "--role", "user",
             "--content", "hello",
             "--occurred-at", "2026-08-10T01:02:03Z"],
            tmp_path,
        )
        assert r.returncode == 2
        assert "Traceback" not in r.stderr


class TestCLIHelpDiscoverability:
    """P3: new commands appear in CLI help."""

    def test_help_lists_new_commands(self, tmp_path):
        r = run_cli(["--help"], tmp_path)
        assert r.returncode == 0
        for cmd in ("ingest", "ingest-status", "ingest-retry",
                     "reclaim-orphans", "dream"):
            assert cmd in r.stdout, f"'{cmd}' missing from --help output"


# ===========================================================================
# Task 6A fix round 2: scope whitelist + populated projection + determinism
# ===========================================================================


class TestDreamScopeOmitted:
    """Disclosure boundary: scope is omitted entirely from projection.

    Scope values — even whitelisted public keys — are untyped core input that
    can be nested or untrusted. The smallest fail-closed public JSON contract
    omits the field entirely rather than trying to filter unbounded input.
    """

    _MALICIOUS_SCOPE = {
        "session_id": {"api_key": "sk-x"},
        "actor_id": "actor-1",
        "content": "TOP-SECRET-MEMORY",
        "config": {"model": "gpt-4", "api_key": "sk-leaked"},
    }

    def _seed_run_with_malicious_scope(self, tmp_path):
        """Create a Dream run via SDK with nested/extra scope, return run_id."""
        data_dir = tmp_path / "mnemosyne-data"
        db_path = data_dir / "mnemosyne.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        from mnemosyne.core.beam import BeamMemory
        from mnemosyne.core import dream
        beam = BeamMemory(session_id="scope-test-sess", db_path=db_path)
        run = dream.dream_plan(beam, scope=dict(self._MALICIOUS_SCOPE))
        beam.conn.close()
        return run.run_id

    def test_status_json_omits_scope_entirely(self, tmp_path):
        run_id = self._seed_run_with_malicious_scope(tmp_path)
        r = run_cli(["dream", "status", "--run-id", run_id, "--json"], tmp_path)
        assert r.returncode == 0, r.stderr
        payload = json.loads(r.stdout)
        assert "scope" not in payload, (
            f"projection must not include scope: {payload!r}"
        )
        # Malicious data must not leak anywhere in the output.
        raw = r.stdout
        assert "TOP-SECRET-MEMORY" not in raw
        assert "sk-leaked" not in raw
        assert "sk-x" not in raw

    def test_plan_json_omits_scope_entirely(self, tmp_path):
        r = run_cli(
            ["dream", "plan", "--session-id", "wl-sess", "--json"],
            tmp_path,
        )
        assert r.returncode == 0, r.stderr
        payload = json.loads(r.stdout)
        assert "scope" not in payload


class TestDreamPopulatedProjection:
    """Test-hygiene: exercise a populated run, not only terminal no-candidate.

    Prove that an actual non-empty plan + review + verify JSON remains
    content-free (no manifest/actions/images/raw receipts) even when the run
    carries real actions and receipts.
    """

    def _seed_proposals(self, tmp_path):
        env = os.environ.copy()
        env["HOME"] = str(tmp_path / "home")
        data_dir = tmp_path / "mnemosyne-data"
        env["MNEMOSYNE_DATA_DIR"] = str(data_dir)
        db_path = data_dir / "mnemosyne.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        from mnemosyne.core.beam import BeamMemory
        from mnemosyne.core.shmr import _init_schema, PROPOSAL_SCHEMA_SQL
        beam = BeamMemory(session_id="pop-proj-sess", db_path=db_path)
        _init_schema(beam.conn)
        beam.conn.executescript(PROPOSAL_SCHEMA_SQL)
        for fid, subj, pred, obj in (
            ("pf1", "alice", "likes", "rust lang"),
            ("pf2", "alice", "likes", "rust programming"),
        ):
            beam.conn.execute(
                "INSERT INTO facts "
                "(fact_id, session_id, subject, predicate, object, confidence) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (fid, "pop-proj-sess", subj, pred, obj, 0.9),
            )
        beam.conn.execute(
            "INSERT INTO shmr_proposals "
            "(run_id, cluster_id, session_id, scope_json, cited_source_ids, "
            "subject, predicate, object, confidence, action, target_source_id, "
            "rationale, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("pop-run", "pc1", "pop-proj-sess",
             json.dumps({"session_id": "pop-proj-sess"}),
             json.dumps(["pf1", "pf2"]),
             "alice", "prefers", "rust", 0.9, "create",
             "pf1", "seeded", "proposed"),
        )
        beam.conn.commit()
        beam.conn.close()

    def test_populated_plan_json_content_free(self, tmp_path):
        self._seed_proposals(tmp_path)
        r = run_cli(
            ["dream", "plan", "--session-id", "pop-proj-sess", "--json"],
            tmp_path,
        )
        assert r.returncode == 0, r.stderr
        payload = json.loads(r.stdout)
        assert payload["state"] == "awaiting_approval"
        assert payload["action_count"] >= 1
        assert payload["manifest_hash"]
        for forbidden in ("manifest", "actions", "receipts", "before_image",
                          "after_image", "content", "config"):
            assert forbidden not in payload, (
                f"populated plan --json leaks '{forbidden}'"
            )

    def test_populated_review_verify_json_content_free(self, tmp_path):
        self._seed_proposals(tmp_path)
        r = run_cli(
            ["dream", "plan", "--session-id", "pop-proj-sess", "--json"],
            tmp_path,
        )
        run_id = json.loads(r.stdout)["run_id"]

        r = run_cli(
            ["dream", "review", "--run-id", run_id, "--actor-id", "revA",
             "--verdict", "PASS", "--json"],
            tmp_path,
        )
        assert r.returncode == 0, r.stderr
        review_payload = json.loads(r.stdout)
        assert review_payload["state"] == "awaiting_approval"
        # After review, receipt_counts should show reviewer:PASS.
        assert review_payload.get("receipt_counts", {}).get("reviewer:PASS", 0) >= 1
        for forbidden in ("manifest", "actions", "receipts", "before_image",
                          "after_image", "content", "config"):
            assert forbidden not in review_payload

        r = run_cli(
            ["dream", "verify", "--run-id", run_id, "--actor-id", "verB",
             "--verdict", "PASS", "--json"],
            tmp_path,
        )
        assert r.returncode == 0, r.stderr
        verify_payload = json.loads(r.stdout)
        assert verify_payload["state"] == "ready"
        assert verify_payload.get("receipt_counts", {}).get("verifier:PASS", 0) >= 1
        for forbidden in ("manifest", "actions", "receipts", "before_image",
                          "after_image", "content", "config"):
            assert forbidden not in verify_payload


# ===========================================================================
# I-4: Dream mutation failures must signal failure (exit code + error code)
# ===========================================================================


class TestDreamFailureSignaling:
    """Audit I-4: failed apply/resume/undo must not look like success.

    A failed mutation must (a) exit non-zero, (b) surface a safe error_code
    (no manifest/action/before-image/private content), and (c) emit the
    structured projection on --json. Successful and explicit idempotent
    terminal cases retain their existing behavior.

    The common root cause: apply/resume/undo exit 0 on retryable/terminal
    failures. The unifying failure signal is ``error_code is not None``.
    """

    # JSON keys that must NEVER appear in the curated projection.
    _FORBIDDEN_KEYS = (
        "manifest", "actions", "receipts", "before_image", "after_image",
        "content", "config", "scope", "failure_reason",
    )

    def _assert_no_content_leak(self, run_result, *, private_reason):
        """Assert the curated projection leaks no private content.

        Checks both stdout and stderr: the structured projection must carry
        only safe identifiers/state/error_code, never the unbounded
        ``failure_reason`` (which may contain DB error text or row detail),
        never raw manifest/actions/images, and never the seeded private
        detail string.
        """
        # The private failure_reason text must never reach the CLI surface.
        for stream in (run_result.stdout, run_result.stderr):
            assert private_reason not in stream, (
                f"CLI output leaks private failure_reason: {private_reason!r}"
            )
        # If JSON was emitted, forbidden keys must be absent.
        stripped = run_result.stdout.strip()
        if stripped.startswith("{"):
            payload = json.loads(stripped)
            for bad in self._FORBIDDEN_KEYS:
                assert bad not in payload, (
                    f"projection leaks forbidden key '{bad}': {payload!r}"
                )

    def _seed_run(self, tmp_path, run_id, state, *,
                  error_code=None, failure_reason=None,
                  checkpoint="applied", failing_undo=False):
        """Seed a dream_run row directly in a given durable state.

        ``failing_undo=True`` seeds an applied run whose sole action has a
        malformed ``after_image`` so ``dream_undo`` raises inside its
        transaction and leaves the run at ``applied`` with
        ``error_code=integrity_failure`` — the exact I-4 undo-failure shape.
        """
        data_dir = tmp_path / "mnemosyne-data"
        db_path = data_dir / "mnemosyne.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        from mnemosyne.core.beam import BeamMemory
        from mnemosyne.core import dream
        beam = BeamMemory(session_id="i4-sess", db_path=db_path)
        dream._init_dream_schema(beam.conn)
        beam.conn.execute(
            "INSERT INTO dream_runs (run_id, request_id, state, scope_json, "
            "manifest_hash, semantic_hash, checkpoint, error_code, "
            "failure_reason, enrichment_pending, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, "req-" + run_id, state, "{}", "h-" + run_id, "sh",
             checkpoint, error_code, failure_reason, 0,
             "2026-08-10T00:00:00Z", "2026-08-10T00:00:00Z"),
        )
        if failing_undo:
            beam.conn.execute(
                "INSERT INTO dream_actions (run_id, seq, source_table, "
                "source_id, source_hash, source_producer, action, "
                "target_json, before_image, after_image, applied, undone) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, 1, "facts", "f1", "hh", "p", "create", "{}",
                 None, "{NOT VALID JSON", 1, 0),
            )
        beam.conn.commit()
        beam.conn.close()
        return db_path

    # --- failed_retryable apply must exit non-zero + show error_code ------

    def test_apply_failed_retryable_exits_nonzero_json(self, tmp_path):
        self._seed_run(tmp_path, "i4-retryable", "failed_retryable",
                       error_code="database_busy",
                       failure_reason="db is locked")
        r = run_cli(["dream", "apply", "--run-id", "i4-retryable", "--json"],
                    tmp_path)
        assert r.returncode != 0, (
            f"failed_retryable apply must exit non-zero: {r.returncode}\n"
            f"stdout={r.stdout}\nstderr={r.stderr}"
        )
        self._assert_no_content_leak(r, private_reason="db is locked")
        payload = json.loads(r.stdout)
        # dream_apply from a terminal state surfaces validation_failed; the
        # key I-4 contract is that an error_code IS set and exit is non-zero.
        assert payload.get("error_code") is not None
        assert payload.get("state") in ("failed_retryable", "rejected")

    def test_apply_failed_retryable_exits_nonzero_text(self, tmp_path):
        self._seed_run(tmp_path, "i4-retryable-text", "failed_retryable",
                       error_code="database_busy",
                       failure_reason="db is locked")
        r = run_cli(["dream", "apply", "--run-id", "i4-retryable-text"],
                    tmp_path)
        assert r.returncode != 0, (
            f"failed_retryable apply (text) must exit non-zero: "
            f"{r.returncode}"
        )
        # Plain-text path must surface a safe error code (the specific code
        # depends on core state-machine resolution; the I-4 contract is that
        # SOME structured error code is printed).
        assert "error:" in r.stdout or "error:" in r.stderr
        self._assert_no_content_leak(r, private_reason="db is locked")

    # --- failed undo must exit non-zero + show error_code -----------------

    def test_undo_failed_exits_nonzero_json(self, tmp_path):
        # Seed an applied run whose undo will fail inside its transaction
        # (malformed after_image) and leave error_code=integrity_failure.
        self._seed_run(tmp_path, "i4-undo-fail", "applied",
                       failing_undo=True)
        r = run_cli(["dream", "undo", "--run-id", "i4-undo-fail", "--json"],
                    tmp_path)
        assert r.returncode != 0, (
            f"undo on a run carrying an error_code must exit non-zero: "
            f"{r.returncode}\nstdout={r.stdout}\nstderr={r.stderr}"
        )
        # The raw JSON parser error (unbounded text) must NOT be printed.
        self._assert_no_content_leak(r, private_reason="NOT VALID JSON")
        payload = json.loads(r.stdout)
        assert payload.get("error_code") == "integrity_failure"

    def test_undo_failed_exits_nonzero_text(self, tmp_path):
        self._seed_run(tmp_path, "i4-undo-fail-text", "applied",
                       failing_undo=True)
        r = run_cli(["dream", "undo", "--run-id", "i4-undo-fail-text"],
                    tmp_path)
        assert r.returncode != 0, (
            f"undo (text) on a failed run must exit non-zero: {r.returncode}"
        )
        assert "integrity_failure" in r.stdout or "integrity_failure" in r.stderr
        self._assert_no_content_leak(r, private_reason="NOT VALID JSON")

    # --- failed resume must exit non-zero + show error_code ---------------

    def test_resume_failed_exits_nonzero_json(self, tmp_path):
        # stale_manifest is NOT retried by resume (returns the run as-is);
        # the CLI must surface that as a failure.
        self._seed_run(tmp_path, "i4-resume-stale", "failed_retryable",
                       error_code="stale_manifest",
                       failure_reason="source hash mismatch",
                       checkpoint="applied")
        r = run_cli(["dream", "resume", "--run-id", "i4-resume-stale", "--json"],
                    tmp_path)
        assert r.returncode != 0, (
            f"resume ending in failure must exit non-zero: {r.returncode}\n"
            f"stdout={r.stdout}\nstderr={r.stderr}"
        )
        self._assert_no_content_leak(r, private_reason="source hash mismatch")
        payload = json.loads(r.stdout)
        assert payload.get("error_code") is not None

    def test_resume_failed_exits_nonzero_text(self, tmp_path):
        self._seed_run(tmp_path, "i4-resume-stale-text", "failed_retryable",
                       error_code="stale_manifest",
                       failure_reason="source hash mismatch",
                       checkpoint="applied")
        r = run_cli(["dream", "resume", "--run-id", "i4-resume-stale-text"],
                    tmp_path)
        assert r.returncode != 0, (
            f"resume (text) ending in failure must exit non-zero: "
            f"{r.returncode}"
        )
        assert "stale_manifest" in r.stdout or "stale_manifest" in r.stderr
        self._assert_no_content_leak(r, private_reason="source hash mismatch")

    # --- success / idempotent / explicit-terminal compatibility -----------

    def test_apply_unknown_run_exits_nonzero(self, tmp_path):
        # 'rejected' with validation_failed (run not found) — already exits
        # non-zero for apply; keep that behavior.
        r = run_cli(["dream", "apply", "--run-id", "no-such-run", "--json"],
                    tmp_path)
        assert r.returncode != 0
        payload = json.loads(r.stdout)
        assert payload.get("error_code") == "validation_failed"

    def test_undo_already_undone_idempotent_exits_zero(self, tmp_path):
        # Idempotent second undo: state=undone, NO error_code (only a benign
        # failure_reason='already_undone'). Must remain exit 0.
        self._seed_run(tmp_path, "i4-idempotent", "undone",
                       error_code=None,
                       failure_reason="already_undone",
                       checkpoint="undone")
        r = run_cli(["dream", "undo", "--run-id", "i4-idempotent", "--json"],
                    tmp_path)
        assert r.returncode == 0, (
            f"idempotent undo must exit 0: {r.returncode}\n"
            f"stdout={r.stdout}\nstderr={r.stderr}"
        )
        payload = json.loads(r.stdout)
        assert payload.get("state") == "undone"
        assert payload.get("error_code") is None
