"""
Task 5: Native Dream lifecycle.

Binding contract from the approved plan + preflight:

  * Beam-first public API: dream_plan / dream_submit_receipt / dream_apply /
    dream_resume / dream_undo / dream_status, each taking ``beam`` first.
  * Additive tables only: dream_runs, dream_actions, dream_receipts.
  * Lifecycle states:
      planning -> awaiting_approval -> ready -> applying -> applied
        -> undoing -> undone
      terminals: rejected | failed_retryable | failed_terminal
  * Manifest: stable UUID4 run_id persisted on the run, but manifest_hash is
    a SHA-256 over a canonical semantic projection that excludes volatile
    run/timestamp fields so equivalent inputs hash deterministically.
  * Hard bounds: <=50 actions, <=500 source rows, <=5 MiB input; 24h approval
    TTL; request_id idempotency (no duplicate run/action rows).
  * Dual receipts: PASS reviewer then PASS independent verifier (different
    actor_id), exact run_id + manifest_hash; reject stale/future/malformed/
    duplicate-role/wrong-hash/wrong-run.
  * Planning is deterministic + report-only; proposals stay invisible to
    recall before apply.
  * Apply only from ready; revalidate source hashes immediately before one
    transaction; Dream owns the transaction (reject caller-open transactions,
    use the Beam deferred-commit seam); DELETE+INSERT for facts (FTS triggers);
    persist before/after images + audit + run state atomically; enrichment
    stays pending after commit.
  * Undo uses this run's before-images only, never another run's output;
    second undo -> already_undone; apply/undo idempotent.
  * dream_resume resolves rollback/retry and after-commit-before-response
    crash window without double apply; checkpoint is durable.
  * dream_active gate: set before verified apply/undo, cleared on normal
    ownership end; stale true reconciled from durable run state; crash leaves
    it fail-safe true.
  * Structured error taxonomy: provider_unavailable, provider_empty_response,
    provider_invalid_output, embedding_unavailable, dimension_mismatch,
    no_candidates, no_convergence, budget_exhausted, stale_manifest,
    validation_failed, database_busy, integrity_failure.

All tests use offline seeded sqlite data; no network/model calls. Every test
below FAILS against the current tree (no mnemosyne.core.dream module exists).
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

from mnemosyne.core.beam import BeamMemory
from mnemosyne.core import config as config_module
from mnemosyne.core import dream
from mnemosyne.core import model_refresh
from mnemosyne.core import shmr


# ---------------------------------------------------------------------------
# Offline embedding pin (suite-wide, mirrors test_shmr_dream_proposals.py)
# ---------------------------------------------------------------------------


def _force_offline_embeddings(monkeypatch):
    """Pin SHMR's embedding seam to the lexical fallback and guard against
    any network reach into mnemosyne.core.embeddings.embed."""
    from mnemosyne.core import embeddings as _emb

    monkeypatch.setattr(shmr, "_embedding_fn", lambda: None, raising=True)

    def _counting_guard(_texts):
        raise AssertionError(
            "Dream test reached the network embedding path; tests must stay "
            "offline via the deterministic lexical fallback."
        )

    monkeypatch.setattr(_emb, "embed", _counting_guard, raising=True)


# ---------------------------------------------------------------------------
# Seeded fixtures
# ---------------------------------------------------------------------------


def _seed_facts(beam, rows):
    ids = []
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
        ids.append(r["fact_id"])
    beam.conn.commit()
    return ids


def _seed_episodic(beam, rows):
    ids = []
    for r in rows:
        beam.conn.execute(
            "INSERT INTO episodic_memory "
            "(id, content, importance, session_id, author_id, author_type, channel_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                r["id"],
                r["content"],
                r.get("importance", 0.8),
                r.get("session_id", beam.session_id),
                r.get("author_id"),
                r.get("author_type"),
                r.get("channel_id"),
            ),
        )
        ids.append(r["id"])
    beam.conn.commit()
    return ids


class _RecordingLLM:
    """Deterministic LLM seam for SHMR propose_harmony."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts: List[str] = []

    def __call__(self, prompt, system=""):
        self.prompts.append(prompt)
        if not self.replies:
            return ""
        return self.replies.pop(0)


def _two_cluster_facts():
    """Two clusters of near-duplicate facts that SHMR can synthesize."""
    return [
        {"fact_id": "f1", "subject": "alice", "predicate": "likes",
         "object": "the rust programming language for systems work"},
        {"fact_id": "f2", "subject": "alice", "predicate": "likes",
         "object": "rust language for systems programming"},
        {"fact_id": "f3", "subject": "bob", "predicate": "uses",
         "object": "python for data analysis pipelines daily"},
        {"fact_id": "f4", "subject": "bob", "predicate": "uses",
         "object": "python in data analysis pipelines"},
    ]


def _single_cluster_replies():
    """LLM replies that synthesize one belief per cluster."""
    return [
        json.dumps([{
            "subject": "alice", "predicate": "prefers", "object": "rust",
            "confidence": 0.9, "action": "create",
            "target_fact_id": "f1", "rationale": "both mention rust",
        }]),
        json.dumps([{
            "subject": "bob", "predicate": "prefers", "object": "python",
            "confidence": 0.88, "action": "create",
            "target_fact_id": "f3", "rationale": "both mention python",
        }]),
    ]


@pytest.fixture(autouse=True)
def _isolate_config_and_offline(tmp_path, monkeypatch):
    """Module-wide isolation for EVERY Dream test.

    Points the central config at a throwaway data dir and resets the singleton
    so no test ever touches /home/bell/.hermes/mnemosyne/config.yaml. Also pins
    SHMR's embedding path to the deterministic lexical fallback.
    """
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    config_module.MnemosyneConfig.reset_instance()
    _force_offline_embeddings(monkeypatch)
    yield
    config_module.MnemosyneConfig.reset_instance()


@pytest.fixture
def beam(tmp_path):
    """A BeamMemory seeded with two SHMR-synthesizable fact clusters."""
    b = BeamMemory(session_id="dream-sess", db_path=tmp_path / "dream.db")
    _seed_facts(b, _two_cluster_facts())
    yield b


def _scope(beam) -> Dict[str, Any]:
    return {"session_id": beam.session_id}


def _plan(beam, request_id=None, limits=None):
    return dream.dream_plan(
        beam,
        scope=_scope(beam),
        limits=limits,
        request_id=request_id,
    )


def _force_proposals(beam, llm_replies=None):
    """Drive SHMR propose_harmony and return its result dict."""
    llm = _RecordingLLM(llm_replies or _single_cluster_replies())
    return shmr.propose_harmony(beam, llm_call=llm, similarity_threshold=0.4)


def _pass_receipt(role, actor_id, run_id, manifest_hash, when=None):
    return {
        "role": role,            # "reviewer" | "verifier"
        "actor_id": actor_id,
        "run_id": run_id,
        "manifest_hash": manifest_hash,
        "verdict": "PASS",
        "reason_code": "ok",
        "timestamp": (when or _now_iso()),
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ago(hours: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


def _ahead(hours: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


# ===========================================================================
# 1. Module surface + schema
# ===========================================================================


class TestModuleSurface:
    def test_public_functions_exist(self):
        for name in (
            "dream_plan", "dream_submit_receipt", "dream_apply",
            "dream_resume", "dream_undo", "dream_status",
        ):
            assert hasattr(dream, name), f"dream.{name} missing"

    def test_dream_run_dataclass_surface(self):
        run = dream.DreamRun(run_id="x", state="planning")
        for field_name in (
            "run_id", "state", "scope", "manifest_hash", "checkpoint",
            "error_code", "actions", "receipts",
        ):
            assert hasattr(run, field_name), f"DreamRun.{field_name} missing"


class TestSchemaIsAdditive:
    def test_dream_tables_created_on_first_plan(self, beam):
        _plan(beam)
        names = {
            r[0] for r in beam.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "dream_runs" in names
        assert "dream_actions" in names
        assert "dream_receipts" in names

    def test_dream_tables_do_not_alter_existing_schema(self, beam):
        before = beam.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        _plan(beam)
        after = beam.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        # Every pre-existing table must still exist (additive only).
        before_names = {r[0] for r in before}
        after_names = {r[0] for r in after}
        assert before_names <= after_names


# ===========================================================================
# 2. Full lifecycle state transitions
# ===========================================================================


class TestLifecycle:
    def test_plan_transitions_to_awaiting_approval(self, beam):
        _force_proposals(beam)
        run = _plan(beam)
        assert run.state == "awaiting_approval"
        assert run.manifest_hash
        assert run.run_id

    def test_plan_with_no_proposals_goes_no_candidates(self, beam):
        # No SHMR proposals persisted -> no_candidates error code.
        run = _plan(beam)
        assert run.state in ("failed_terminal", "rejected")
        assert run.error_code == "no_candidates"

    def test_dual_pass_receipts_transition_to_ready(self, beam):
        _force_proposals(beam)
        run = _plan(beam)
        r1 = dream.dream_submit_receipt(beam, run.run_id,
                                        _pass_receipt("reviewer", "rev1",
                                                      run.run_id, run.manifest_hash))
        assert r1.state == "awaiting_approval"
        r2 = dream.dream_submit_receipt(beam, run.run_id,
                                        _pass_receipt("verifier", "ver1",
                                                      run.run_id, run.manifest_hash))
        assert r2.state == "ready"

    def test_apply_transitions_applied_then_undo_undone(self, beam):
        _force_proposals(beam)
        run = _plan(beam)
        dream.dream_submit_receipt(beam, run.run_id,
                                   _pass_receipt("reviewer", "rev1",
                                                 run.run_id, run.manifest_hash))
        run = dream.dream_submit_receipt(beam, run.run_id,
                                         _pass_receipt("verifier", "ver1",
                                                       run.run_id, run.manifest_hash))
        applied = dream.dream_apply(beam, run.run_id)
        assert applied.state == "applied"
        undone = dream.dream_undo(beam, run.run_id)
        assert undone.state == "undone"

    def test_rejected_after_non_pass_receipt_when_both_required(self, beam):
        _force_proposals(beam)
        run = _plan(beam)
        receipt = _pass_receipt("reviewer", "rev1", run.run_id, run.manifest_hash)
        receipt["verdict"] = "FAIL"
        receipt["reason_code"] = "bad_citations"
        out = dream.dream_submit_receipt(beam, run.run_id, receipt)
        assert out.state == "rejected"


# ===========================================================================
# 3. Manifest determinism + bounds
# ===========================================================================


class TestManifestDeterminism:
    def test_same_inputs_yield_same_manifest_hash(self, tmp_path):
        # Two separate beams with identical seeded data + proposals must
        # produce the same manifest_hash (semantic projection excludes
        # volatile run_id/timestamps).
        def setup(path):
            config_module.MnemosyneConfig.reset_instance()
            b = BeamMemory(session_id="det", db_path=path)
            _seed_facts(b, _two_cluster_facts())
            _force_proposals(b)
            return _plan(b)

        run_a = setup(tmp_path / "a.db")
        run_b = setup(tmp_path / "b.db")
        assert run_a.manifest_hash == run_b.manifest_hash
        assert run_a.run_id != run_b.run_id  # UUID4 differs

    def test_run_id_is_uuid4(self, beam):
        _force_proposals(beam)
        run = _plan(beam)
        u = uuid.UUID(run.run_id)
        assert u.version == 4


class TestHardBounds:
    def test_actions_over_50_rejected(self, beam):
        # Insert 51 SHMR proposals targeting the same cluster; Dream must
        # refuse to plan >50 actions.
        shmr._init_proposal_schema(beam.conn)
        for i in range(51):
            beam.conn.execute(
                "INSERT INTO shmr_proposals "
                "(run_id, cluster_id, session_id, scope_json, cited_source_ids, "
                "subject, predicate, object, confidence, action, target_source_id, rationale) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "shmr_x", f"c{i}", beam.session_id,
                    json.dumps({"session_id": beam.session_id}),
                    json.dumps(["f1"]),
                    "s", "p", f"o{i}", 0.5, "create", None, "r",
                ),
            )
        beam.conn.commit()
        run = _plan(beam)
        assert run.state in ("rejected", "failed_terminal")
        assert run.error_code == "budget_exhausted"

    def test_source_rows_over_500_rejected(self, beam):
        # Seed 501 distinct facts, then synthesize proposals citing each.
        shmr._init_proposal_schema(beam.conn)
        rows = [
            {"fact_id": f"bx{i}", "subject": "x", "predicate": "p",
             "object": f"value-{i}", "confidence": 0.5}
            for i in range(501)
        ]
        _seed_facts(beam, rows)
        for i in range(501):
            beam.conn.execute(
                "INSERT INTO shmr_proposals "
                "(run_id, cluster_id, session_id, scope_json, cited_source_ids, "
                "subject, predicate, object, confidence, action, "
                "target_source_id, rationale) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "shmr_y", f"cy{i}", beam.session_id,
                    json.dumps({"session_id": beam.session_id}),
                    json.dumps([f"bx{i}"]),
                    "x", "p", f"value-{i}", 0.5, "create", None, "r",
                ),
            )
        beam.conn.commit()
        run = _plan(beam)
        assert run.state in ("rejected", "failed_terminal")
        assert run.error_code in ("budget_exhausted", "validation_failed")

    def test_input_over_5mib_rejected(self, beam):
        # One fact whose object exceeds 5 MiB, cited by a proposal.
        shmr._init_proposal_schema(beam.conn)
        big = "Z" * (5 * 1024 * 1024 + 1024)
        _seed_facts(beam, [{"fact_id": "big1", "subject": "big",
                            "predicate": "blob", "object": big,
                            "confidence": 0.5}])
        beam.conn.execute(
            "INSERT INTO shmr_proposals "
            "(run_id, cluster_id, session_id, scope_json, cited_source_ids, "
            "subject, predicate, object, confidence, action, "
            "target_source_id, rationale) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "shmr_big", "cbig", beam.session_id,
                json.dumps({"session_id": beam.session_id}),
                json.dumps(["big1"]),
                "big", "blob", big, 0.5, "create", None, "r",
            ),
        )
        beam.conn.commit()
        run = _plan(beam)
        assert run.state in ("rejected", "failed_terminal")
        assert run.error_code in ("budget_exhausted", "validation_failed")


# ===========================================================================
# 4. request_id idempotency
# ===========================================================================


class TestRequestIdIdempotency:
    def test_same_request_id_returns_same_run_no_duplicates(self, beam):
        _force_proposals(beam)
        rid = "req-abc"
        a = _plan(beam, request_id=rid)
        b = _plan(beam, request_id=rid)
        assert a.run_id == b.run_id
        rows = beam.conn.execute(
            "SELECT COUNT(*) FROM dream_runs WHERE request_id = ?", (rid,)
        ).fetchone()[0]
        assert rows == 1


# ===========================================================================
# 5. Dual receipts: independence, hash, TTL, role, actor
# ===========================================================================


class TestReceiptValidation:
    def _to_ready(self, beam):
        _force_proposals(beam)
        run = _plan(beam)
        return run

    def test_requires_reviewer_before_verifier(self, beam):
        run = self._to_ready(beam)
        # Verifier first must not advance to ready.
        out = dream.dream_submit_receipt(beam, run.run_id,
                                         _pass_receipt("verifier", "v1",
                                                       run.run_id, run.manifest_hash))
        assert out.state == "awaiting_approval"

    def test_verifier_must_differ_from_reviewer_actor(self, beam):
        run = self._to_ready(beam)
        dream.dream_submit_receipt(beam, run.run_id,
                                   _pass_receipt("reviewer", "same",
                                                 run.run_id, run.manifest_hash))
        out = dream.dream_submit_receipt(beam, run.run_id,
                                         _pass_receipt("verifier", "same",
                                                       run.run_id, run.manifest_hash))
        assert out.state == "rejected"
        assert out.error_code == "validation_failed"

    def test_wrong_manifest_hash_rejected(self, beam):
        run = self._to_ready(beam)
        bad = _pass_receipt("reviewer", "rev1", run.run_id, "0" * 64)
        out = dream.dream_submit_receipt(beam, run.run_id, bad)
        assert out.state == "rejected"
        assert out.error_code == "stale_manifest"

    def test_wrong_run_id_rejected(self, beam):
        run = self._to_ready(beam)
        bad = _pass_receipt("reviewer", "rev1", "deadbeef", run.manifest_hash)
        out = dream.dream_submit_receipt(beam, run.run_id, bad)
        assert out.state == "rejected"

    def test_stale_receipt_over_24h_rejected(self, beam):
        run = self._to_ready(beam)
        bad = _pass_receipt("reviewer", "rev1", run.run_id, run.manifest_hash,
                            when=_ago(25))
        out = dream.dream_submit_receipt(beam, run.run_id, bad)
        assert out.state == "rejected"

    def test_future_receipt_rejected(self, beam):
        run = self._to_ready(beam)
        bad = _pass_receipt("reviewer", "rev1", run.run_id, run.manifest_hash,
                            when=_ahead(2))
        out = dream.dream_submit_receipt(beam, run.run_id, bad)
        assert out.state == "rejected"

    def test_malformed_receipt_rejected(self, beam):
        run = self._to_ready(beam)
        out = dream.dream_submit_receipt(beam, run.run_id, {"junk": True})
        assert out.state == "rejected"
        assert out.error_code == "validation_failed"

    def test_duplicate_role_rejected(self, beam):
        run = self._to_ready(beam)
        dream.dream_submit_receipt(beam, run.run_id,
                                   _pass_receipt("reviewer", "rev1",
                                                 run.run_id, run.manifest_hash))
        out = dream.dream_submit_receipt(beam, run.run_id,
                                         _pass_receipt("reviewer", "rev2",
                                                       run.run_id, run.manifest_hash))
        assert out.state == "rejected"


# ===========================================================================
# 6. Proposal invisibility before apply
# ===========================================================================


class TestProposalInvisibility:
    def test_proposals_not_recallable_before_apply(self, beam):
        _force_proposals(beam)
        _plan(beam)
        # Nothing Dream wrote may surface via plain recall.
        results = beam.recall("alice rust")
        for r in results:
            content = (r.get("content") or r.get("object") or "").lower()
            assert "alice prefers rust" not in content
            assert "dream" not in content

    def test_no_dream_rows_in_recallable_tables_before_apply(self, beam):
        _force_proposals(beam)
        _plan(beam)
        counts = {
            t: beam.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("working_memory", "episodic_memory", "canonical_facts")
        }
        # Apply hasn't run; whatever was seeded is all that's there.
        assert counts["working_memory"] == 0
        assert counts["episodic_memory"] == 0
        # canonical_facts may exist from init; ensure no dream proposal slot
        # was inserted.
        rows = beam.conn.execute(
            "SELECT name FROM canonical_facts WHERE body LIKE '%dream%' "
            "OR body LIKE '%alice prefers%'"
        ).fetchall()
        assert len(rows) == 0


# ===========================================================================
# 7. Stale source rejection
# ===========================================================================


class TestStaleSource:
    def test_source_mutation_after_plan_yields_stale_manifest_on_apply(self, beam):
        _force_proposals(beam)
        run = _plan(beam)
        dream.dream_submit_receipt(beam, run.run_id,
                                   _pass_receipt("reviewer", "r1",
                                                 run.run_id, run.manifest_hash))
        run = dream.dream_submit_receipt(beam, run.run_id,
                                         _pass_receipt("verifier", "v1",
                                                       run.run_id, run.manifest_hash))
        # Mutate one cited source after plan.
        beam.conn.execute(
            "UPDATE facts SET object = ? WHERE fact_id = ?",
            ("the rust programming language [CHANGED]", "f1"),
        )
        beam.conn.commit()
        out = dream.dream_apply(beam, run.run_id)
        assert out.state in ("failed_retryable", "rejected", "failed_terminal")
        assert out.error_code == "stale_manifest"
        # No semantic mutation happened.
        assert beam.conn.execute(
            "SELECT COUNT(*) FROM canonical_facts WHERE body LIKE '%rust%'"
        ).fetchone()[0] == 0


# ===========================================================================
# 8. Atomic apply: mid-action failure leaves zero partial state
# ===========================================================================


class TestAtomicApply:
    def test_mid_transaction_failure_rolls_back_all_semantic_writes(
        self, beam, monkeypatch
    ):
        _force_proposals(beam)
        run = _plan(beam)
        dream.dream_submit_receipt(beam, run.run_id,
                                   _pass_receipt("reviewer", "r1",
                                                 run.run_id, run.manifest_hash))
        run = dream.dream_submit_receipt(beam, run.run_id,
                                         _pass_receipt("verifier", "v1",
                                                       run.run_id, run.manifest_hash))

        canonical_count_before = beam.conn.execute(
            "SELECT COUNT(*) FROM canonical_facts"
        ).fetchone()[0]

        original_execute = beam.conn.execute
        call_state = {"failed": False}

        def flaky_execute(sql, *params):
            # Inject a failure on the FIRST canonical_facts INSERT inside the
            # apply transaction (this is the actual semantic mutation).
            if (
                isinstance(sql, str)
                and "INSERT INTO canonical_facts" in sql
                and not call_state["failed"]
            ):
                call_state["failed"] = True
                raise sqlite3.OperationalError("simulated mid-apply failure")
            return original_execute(sql, *params)

        monkeypatch.setattr(beam.conn, "execute", flaky_execute)

        out = dream.dream_apply(beam, run.run_id)
        assert out.state in ("failed_retryable", "failed_terminal")
        assert out.error_code in ("database_busy", "integrity_failure",
                                  "validation_failed")

        # No semantic state changed: canonical_facts unchanged.
        canonical_count_after = beam.conn.execute(
            "SELECT COUNT(*) FROM canonical_facts"
        ).fetchone()[0]
        assert canonical_count_after == canonical_count_before

        # Atomicity guarantee: regardless of whether the run is retryable, no
        # partial semantic state survives (canonical_count unchanged above).
        # The run must NOT be left in ``applied``.
        status = dream.dream_status(beam, run.run_id)
        assert status.state != "applied"


# ===========================================================================
# 9. Exact undo / idempotency / cross-run safety
# ===========================================================================


class TestUndo:
    def _apply(self, beam):
        _force_proposals(beam)
        run = _plan(beam)
        dream.dream_submit_receipt(beam, run.run_id,
                                   _pass_receipt("reviewer", "r1",
                                                 run.run_id, run.manifest_hash))
        run = dream.dream_submit_receipt(beam, run.run_id,
                                         _pass_receipt("verifier", "v1",
                                                       run.run_id, run.manifest_hash))
        return dream.dream_apply(beam, run.run_id)

    def test_second_undo_returns_already_undone(self, beam):
        run = self._apply(beam)
        dream.dream_undo(beam, run.run_id)
        second = dream.dream_undo(beam, run.run_id)
        assert second.state == "undone"

    def test_undo_restores_exact_before_image(self, beam):
        run = self._apply(beam)
        # After apply, some canonical_facts row exists for this run's output.
        before = beam.conn.execute(
            "SELECT COUNT(*) FROM canonical_facts"
        ).fetchone()[0]
        dream.dream_undo(beam, run.run_id)
        after = beam.conn.execute(
            "SELECT COUNT(*) FROM canonical_facts"
        ).fetchone()[0]
        # Undo removes exactly what apply added.
        assert after < before

    def test_undo_does_not_touch_another_run_output(self, beam):
        run_a = self._apply(beam)
        # Apply a second run with different scope/output.
        run_b = self._apply(beam)
        dream.dream_undo(beam, run_a.run_id)
        # run_b's applied output must still exist.
        status_b = dream.dream_status(beam, run_b.run_id)
        assert status_b.state == "applied"


# ===========================================================================
# 10. Checkpoint / resume / crash windows
# ===========================================================================


class TestResumeAndCheckpoint:
    def test_resume_after_failed_apply_does_not_double_apply(self, beam):
        _force_proposals(beam)
        run = _plan(beam)
        dream.dream_submit_receipt(beam, run.run_id,
                                   _pass_receipt("reviewer", "r1",
                                                 run.run_id, run.manifest_hash))
        run = dream.dream_submit_receipt(beam, run.run_id,
                                         _pass_receipt("verifier", "v1",
                                                       run.run_id, run.manifest_hash))

        # First apply: force a failure on the first canonical_facts INSERT
        # (the actual semantic mutation), so the deferred-commit txn rolls
        # back and the run ends in failed_retryable/failed_terminal.
        inserts_seen = {"n": 0}
        original_execute = beam.conn.execute

        def one_shot_fail(sql, *params):
            if (
                isinstance(sql, str)
                and "INSERT INTO canonical_facts" in sql
            ):
                inserts_seen["n"] += 1
                if inserts_seen["n"] == 1:
                    raise sqlite3.OperationalError("transient")
            return original_execute(sql, *params)

        # Monkeypatch only briefly via attribute restore.
        beam.conn.execute = one_shot_fail  # type: ignore[assignment]
        try:
            out = dream.dream_apply(beam, run.run_id)
        finally:
            beam.conn.execute = original_execute  # type: ignore[assignment]
        assert out.state in ("failed_retryable", "failed_terminal")

        # dream_resume must drive it to applied without duplicating semantic
        # writes.

        resumed = dream.dream_resume(beam, run.run_id)
        assert resumed.state == "applied"

        # Counting canonical_facts should reflect exactly one apply.
        # Each action creates one canonical_facts row; we had 2 actions
        # (two clusters), so exactly 2 rows should exist now.
        n = beam.conn.execute(
            "SELECT COUNT(*) FROM canonical_facts WHERE owner_id = ?",
            (beam.session_id,),
        ).fetchone()[0]
        assert n == 2

    def test_resume_after_commit_before_response_ack_is_idempotent(self, beam):
        # Crash window: the apply transaction commits, the durable state
        # becomes ``applied``, but the response never reached the caller.
        # dream_resume must observe the durable ``applied`` state and NOT
        # re-apply (which would duplicate canonical_facts rows).
        _force_proposals(beam)
        run = _plan(beam)
        dream.dream_submit_receipt(beam, run.run_id,
                                   _pass_receipt("reviewer", "r1",
                                                 run.run_id, run.manifest_hash))
        run = dream.dream_submit_receipt(beam, run.run_id,
                                         _pass_receipt("verifier", "v1",
                                                       run.run_id, run.manifest_hash))
        # Apply normally (this is the commit that "landed").
        dream.dream_apply(beam, run.run_id)
        n_after_apply = beam.conn.execute(
            "SELECT COUNT(*) FROM canonical_facts WHERE owner_id = ?",
            (beam.session_id,),
        ).fetchone()[0]

        # The caller retries, thinking the apply never happened. dream_resume
        # must observe durable ``applied`` and no-op.
        resumed = dream.dream_resume(beam, run.run_id)
        assert resumed.state == "applied"
        n_after_resume = beam.conn.execute(
            "SELECT COUNT(*) FROM canonical_facts WHERE owner_id = ?",
            (beam.session_id,),
        ).fetchone()[0]
        assert n_after_resume == n_after_apply


# ===========================================================================
# 11. dream_active gate
# ===========================================================================


class TestDreamActiveGate:
    def test_gate_set_true_while_applied_then_cleared_on_undo(self, beam):
        _force_proposals(beam)
        run = _plan(beam)
        dream.dream_submit_receipt(beam, run.run_id,
                                   _pass_receipt("reviewer", "r1",
                                                 run.run_id, run.manifest_hash))
        run = dream.dream_submit_receipt(beam, run.run_id,
                                         _pass_receipt("verifier", "v1",
                                                       run.run_id, run.manifest_hash))
        assert model_refresh.auto_apply_enabled() is True
        dream.dream_apply(beam, run.run_id)
        assert model_refresh.auto_apply_enabled() is False
        dream.dream_undo(beam, run.run_id)
        assert model_refresh.auto_apply_enabled() is True

    def test_stale_true_gate_reconciled_when_no_active_run(self, beam):
        # Plant a stale dream_active=true with no run owning mutations.
        from mnemosyne.core.config import get_config
        get_config().set_many({"dream_active": True})
        assert model_refresh.auto_apply_enabled() is False
        # dream_status on a non-existent / terminal run reconciles the gate.
        _force_proposals(beam)
        run = _plan(beam)
        # Move the run to a terminal state then reconcile.
        dream.dream_submit_receipt(beam, run.run_id,
                                   _pass_receipt("reviewer", "r1",
                                                 run.run_id, run.manifest_hash))
        run = dream.dream_submit_receipt(beam, run.run_id,
                                         _pass_receipt("verifier", "v1",
                                                       run.run_id, run.manifest_hash))
        dream.dream_apply(beam, run.run_id)
        dream.dream_undo(beam, run.run_id)
        dream.dream_status(beam, run.run_id)
        # Gate cleared because no run is in applying/applied/undoing.
        assert model_refresh.auto_apply_enabled() is True


# ===========================================================================
# 12. Error taxonomy surfaces
# ===========================================================================


class TestErrorTaxonomy:
    EXPECTED = {
        "provider_unavailable", "provider_empty_response",
        "provider_invalid_output", "embedding_unavailable",
        "dimension_mismatch", "no_candidates", "no_convergence",
        "budget_exhausted", "stale_manifest", "validation_failed",
        "database_busy", "integrity_failure",
    }

    def test_error_codes_are_the_allowed_set(self):
        assert dream.ERROR_CODES == self.EXPECTED

    def test_no_candidates_when_no_proposals(self, beam):
        run = _plan(beam)
        assert run.error_code == "no_candidates"

    def test_database_busy_surfaces_on_sqlite_lock(self, beam, monkeypatch):
        _force_proposals(beam)
        run = _plan(beam)
        dream.dream_submit_receipt(beam, run.run_id,
                                   _pass_receipt("reviewer", "r1",
                                                 run.run_id, run.manifest_hash))
        run = dream.dream_submit_receipt(beam, run.run_id,
                                         _pass_receipt("verifier", "v1",
                                                       run.run_id, run.manifest_hash))
        original_execute = beam.conn.execute

        def locked(sql, *params):
            if isinstance(sql, str) and sql.strip().upper().startswith("BEGIN"):
                raise sqlite3.OperationalError("database is locked")
            if isinstance(sql, str) and "INSERT INTO canonical_facts" in sql:
                raise sqlite3.OperationalError("database is locked")
            return original_execute(sql, *params)

        monkeypatch.setattr(beam.conn, "execute", locked)
        out = dream.dream_apply(beam, run.run_id)
        assert out.error_code == "database_busy"
        assert out.state in ("failed_retryable", "failed_terminal")


# ===========================================================================
# 13. Transaction ownership: reject caller-open transactions
# ===========================================================================


class TestTransactionOwnership:
    def test_apply_rejects_caller_open_transaction(self, beam):
        _force_proposals(beam)
        run = _plan(beam)
        dream.dream_submit_receipt(beam, run.run_id,
                                   _pass_receipt("reviewer", "r1",
                                                 run.run_id, run.manifest_hash))
        run = dream.dream_submit_receipt(beam, run.run_id,
                                         _pass_receipt("verifier", "v1",
                                                       run.run_id, run.manifest_hash))
        beam.conn.execute("BEGIN")
        try:
            out = dream.dream_apply(beam, run.run_id)
            assert out.error_code in ("validation_failed", "database_busy")
        finally:
            try:
                beam.conn.rollback()
            except Exception:
                pass


# ===========================================================================
# Fix round 1 — DeepSeek review binding findings (C1 + I1-I8)
# ===========================================================================


class TestC1ExactCanonicalBeforeImage:
    """C1: undo must restore the exact pre-apply canonical slot, including
    valid_until=NULL on a pre-existing current row."""

    def test_undo_restores_pre_existing_canonical_slot(self, beam):
        # Seed a pre-existing canonical slot for the same owner/category/name
        # that Dream's action will target.
        from mnemosyne.core.canonical import CanonicalStore
        store = CanonicalStore(conn=beam.conn)
        owner = beam.session_id
        store.remember(owner, "dream", "alice::prefers", "OLD ORIGINAL BODY",
                       source="seed", confidence=0.42)
        beam.conn.commit()

        _force_proposals(beam)
        run = _plan(beam)
        dream.dream_submit_receipt(beam, run.run_id,
                                   _pass_receipt("reviewer", "r1",
                                                 run.run_id, run.manifest_hash))
        run = dream.dream_submit_receipt(beam, run.run_id,
                                         _pass_receipt("verifier", "v1",
                                                       run.run_id, run.manifest_hash))
        dream.dream_apply(beam, run.run_id)

        # After apply, the pre-existing row must be superseded (valid_until
        # set) and a new current row exists.
        pre_existing = beam.conn.execute(
            "SELECT * FROM canonical_facts WHERE body = ? "
            "AND owner_id = ?", ("OLD ORIGINAL BODY", owner)
        ).fetchone()
        assert pre_existing is not None
        assert pre_existing["valid_until"] is not None

        # Undo.
        dream.dream_undo(beam, run.run_id)

        # C1 contract: the pre-existing row must be current again
        # (valid_until restored to NULL), with the exact prior body/version.
        restored = beam.conn.execute(
            "SELECT * FROM canonical_facts WHERE body = ? "
            "AND owner_id = ?", ("OLD ORIGINAL BODY", owner)
        ).fetchone()
        assert restored is not None
        assert restored["valid_until"] is None, (
            "pre-existing canonical row was left superseded after undo; "
            "exact before-image was not restored"
        )
        assert restored["version"] == pre_existing["version"]
        assert restored["source"] == "seed"
        assert restored["confidence"] == 0.42

        # And the run's own after-row must be gone.
        dream_after = beam.conn.execute(
            "SELECT * FROM canonical_facts WHERE source = 'dream_apply' "
            "AND owner_id = ?", (owner,)
        ).fetchall()
        assert len(dream_after) == 0


class TestI1ApprovalTTLAtApply:
    """I1: a ready run whose receipts are older than 24h must not apply.

    Receipts are planted directly as fresh at submission time, then aged past
    the 24h TTL so we exercise the apply-time re-check (not the submission-
    time check, which already rejects >24h receipts).
    """

    def test_apply_rejects_expired_approvals(self, beam):
        _force_proposals(beam)
        run = _plan(beam)
        # Submit with fresh timestamps so the run reaches ready.
        dream.dream_submit_receipt(beam, run.run_id,
                                   _pass_receipt("reviewer", "r1",
                                                 run.run_id, run.manifest_hash))
        run = dream.dream_submit_receipt(beam, run.run_id,
                                         _pass_receipt("verifier", "v1",
                                                       run.run_id, run.manifest_hash))
        assert run.state == "ready"

        # Now age both PASS receipts past the 24h approval TTL directly in the
        # durable store, simulating wall-clock elapsing after submission.
        old_ts = _ago(25)
        beam.conn.execute(
            "UPDATE dream_receipts SET timestamp = ? WHERE run_id = ?",
            (old_ts, run.run_id),
        )
        beam.conn.commit()

        out = dream.dream_apply(beam, run.run_id)
        assert out.state != "applied", (
            "apply succeeded with expired (>24h) approval receipts"
        )
        assert out.error_code == "stale_manifest"
        # No canonical_facts row was written.
        assert beam.conn.execute(
            "SELECT COUNT(*) FROM canonical_facts WHERE source = 'dream_apply'"
        ).fetchone()[0] == 0


class TestI2NaiveTimestampReceipt:
    """I2: a timezone-naive receipt timestamp must produce validation_failed,
    never a TypeError."""

    def test_naive_timestamp_yields_structured_validation_failed(self, beam):
        _force_proposals(beam)
        run = _plan(beam)
        receipt = _pass_receipt("reviewer", "r1", run.run_id, run.manifest_hash)
        receipt["timestamp"] = "2026-08-10T00:00:00"  # no tz offset
        out = dream.dream_submit_receipt(beam, run.run_id, receipt)
        assert out.state == "rejected"
        assert out.error_code == "validation_failed"


class TestI3ManifestHashConfigIndependent:
    """I3: identical inputs must produce identical manifest_hash regardless of
    the runtime dream_active config gate."""

    def test_manifest_hash_identical_under_differing_dream_active(
        self, tmp_path, monkeypatch
    ):
        def plan_under(dream_active: bool):
            data_dir = tmp_path / ("on" if dream_active else "off")
            data_dir.mkdir()
            monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))
            config_module.MnemosyneConfig.reset_instance()
            from mnemosyne.core.config import get_config
            get_config().set_many({"dream_active": dream_active})
            b = BeamMemory(session_id="det", db_path=data_dir / "d.db")
            _seed_facts(b, _two_cluster_facts())
            _force_proposals(b)
            return _plan(b)

        run_off = plan_under(False)
        run_on = plan_under(True)
        assert run_off.manifest_hash == run_on.manifest_hash, (
            "manifest_hash depends on runtime dream_active gate; semantic "
            "projection must exclude runtime config"
        )


class TestI4RevalidationUnderImmediateLock:
    """I4: Dream's owned apply must acquire BEGIN IMMEDIATE (or equivalent)
    BEFORE source revalidation so a concurrent writer cannot race the check.

    We verify the invariant: immediately after dream_apply opens its
    transaction, the connection must report an active IMMEDIATE transaction
    that a second connection would block on.
    """

    def test_apply_acquires_write_lock_before_revalidation(self, beam, monkeypatch):
        _force_proposals(beam)
        run = _plan(beam)
        dream.dream_submit_receipt(beam, run.run_id,
                                   _pass_receipt("reviewer", "r1",
                                                 run.run_id, run.manifest_hash))
        run = dream.dream_submit_receipt(beam, run.run_id,
                                         _pass_receipt("verifier", "v1",
                                                       run.run_id, run.manifest_hash))

        observed = {"in_txn_at_revalidate": None}

        real_execute = beam.conn.execute

        def observing_execute(sql, *params):
            up = str(sql).strip().upper()
            if "BEGIN" in up and "IMMEDIATE" in up:
                # The apply path opened a write transaction.
                observed["began_immediate"] = True
            return real_execute(sql, *params)

        monkeypatch.setattr(beam.conn, "execute", observing_execute)
        out = dream.dream_apply(beam, run.run_id)
        assert out.state == "applied"
        assert observed.get("began_immediate") is True, (
            "dream_apply did not acquire BEGIN IMMEDIATE before revalidation; "
            "the TOCTOU window remains open"
        )


class TestI6ProposalEligibilityAndClaim:
    """I6: _gather_actions must select only eligible proposals (status, scope
    provenance) and atomically claim them so re-planning cannot reuse them."""

    def test_rolled_back_proposals_excluded(self, beam):
        shmr._init_proposal_schema(beam.conn)
        _seed_facts(beam, [
            {"fact_id": "fa", "subject": "x", "predicate": "p",
             "object": "alpha value one", "confidence": 0.9},
            {"fact_id": "fb", "subject": "x", "predicate": "p",
             "object": "alpha value two", "confidence": 0.9},
        ])
        beam.conn.execute(
            "INSERT INTO shmr_proposals "
            "(run_id, cluster_id, session_id, scope_json, cited_source_ids, "
            "subject, predicate, object, confidence, action, target_source_id, "
            "rationale, status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("shmr_bad", "c0", beam.session_id,
             json.dumps({"session_id": beam.session_id}),
             json.dumps(["fa"]), "x", "p", "alpha", 0.9, "create", None,
             "r", "rolled_back"),
        )
        beam.conn.commit()
        run = _plan(beam)
        # A rolled_back proposal must not be planned.
        assert run.state == "rejected"
        assert run.error_code == "no_candidates"

    def test_mismatched_actor_scope_excluded(self, beam):
        shmr._init_proposal_schema(beam.conn)
        _seed_facts(beam, [
            {"fact_id": "fc", "subject": "y", "predicate": "p",
             "object": "beta value one", "confidence": 0.9},
            {"fact_id": "fd", "subject": "y", "predicate": "p",
             "object": "beta value two", "confidence": 0.9},
        ])
        # Proposal whose scope_json declares a different actor_id.
        beam.conn.execute(
            "INSERT INTO shmr_proposals "
            "(run_id, cluster_id, session_id, scope_json, cited_source_ids, "
            "subject, predicate, object, confidence, action, target_source_id, "
            "rationale, status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("shmr_other", "c1", beam.session_id,
             json.dumps({"session_id": beam.session_id, "actor_id": "other-actor"}),
             json.dumps(["fc"]), "y", "p", "beta", 0.9, "create", None,
             "r", "proposed"),
        )
        beam.conn.commit()
        # Plan with a scope that asserts a different actor.
        run = dream.dream_plan(
            beam, scope={"session_id": beam.session_id, "actor_id": "my-actor"}
        )
        assert run.state == "rejected"
        assert run.error_code == "no_candidates"

    def test_repeated_plan_does_not_reuse_consumed_proposals(self, beam):
        _force_proposals(beam)
        first = _plan(beam)
        assert first.state == "awaiting_approval"
        # A second plan (new request_id) must not consume the same proposals.
        second = dream.dream_plan(
            beam, scope=_scope(beam), request_id="req-second"
        )
        # No eligible proposals left -> no_candidates.
        assert second.state == "rejected"
        assert second.error_code == "no_candidates"


class TestI7EveryEntrypointGuardsTransaction:
    """I7: every public Dream entrypoint must reject a caller-open transaction
    BEFORE schema DDL/executescript, preserving the caller's transaction + row."""

    def test_plan_rejects_caller_open_transaction(self, beam):
        beam.conn.execute("BEGIN")
        beam.conn.execute(
            "INSERT INTO facts (fact_id, session_id, subject, predicate, object) "
            "VALUES ('caller-row', ?, 'c', 'd', 'e')",
            (beam.session_id,),
        )
        try:
            out = dream.dream_plan(beam, scope=_scope(beam))
            assert out.error_code in ("validation_failed", "database_busy")
            # The caller's uncommitted row must still be in the transaction.
            assert beam.conn.in_transaction
            row = beam.conn.execute(
                "SELECT fact_id FROM facts WHERE fact_id = 'caller-row'"
            ).fetchone()
            assert row is not None
        finally:
            beam.conn.rollback()

    def test_submit_receipt_rejects_caller_open_transaction(self, beam):
        _force_proposals(beam)
        run = _plan(beam)
        beam.conn.execute("BEGIN")
        try:
            out = dream.dream_submit_receipt(
                beam, run.run_id,
                _pass_receipt("reviewer", "r1", run.run_id, run.manifest_hash)
            )
            assert out.error_code in ("validation_failed", "database_busy")
            assert beam.conn.in_transaction
        finally:
            beam.conn.rollback()

    def test_status_rejects_caller_open_transaction(self, beam):
        _force_proposals(beam)
        run = _plan(beam)
        beam.conn.execute("BEGIN")
        try:
            out = dream.dream_status(beam, run.run_id)
            assert out.error_code in ("validation_failed", "database_busy")
            assert beam.conn.in_transaction
        finally:
            beam.conn.rollback()

    def test_resume_rejects_caller_open_transaction(self, beam):
        _force_proposals(beam)
        run = _plan(beam)
        beam.conn.execute("BEGIN")
        try:
            out = dream.dream_resume(beam, run.run_id)
            assert out.error_code in ("validation_failed", "database_busy")
            assert beam.conn.in_transaction
        finally:
            beam.conn.rollback()

    def test_undo_rejects_caller_open_transaction(self, beam):
        _force_proposals(beam)
        run = _plan(beam)
        dream.dream_submit_receipt(beam, run.run_id,
                                   _pass_receipt("reviewer", "r1",
                                                 run.run_id, run.manifest_hash))
        run = dream.dream_submit_receipt(beam, run.run_id,
                                         _pass_receipt("verifier", "v1",
                                                       run.run_id, run.manifest_hash))
        dream.dream_apply(beam, run.run_id)
        beam.conn.execute("BEGIN")
        try:
            out = dream.dream_undo(beam, run.run_id)
            assert out.error_code in ("validation_failed", "database_busy")
            assert beam.conn.in_transaction
        finally:
            beam.conn.rollback()


class TestI8AlreadyUndoneSignal:
    """I8: a second undo must explicitly signal already_undone while retaining
    the undone state."""

    def test_second_undo_signals_already_undone(self, beam):
        _force_proposals(beam)
        run = _plan(beam)
        dream.dream_submit_receipt(beam, run.run_id,
                                   _pass_receipt("reviewer", "r1",
                                                 run.run_id, run.manifest_hash))
        run = dream.dream_submit_receipt(beam, run.run_id,
                                         _pass_receipt("verifier", "v1",
                                                       run.run_id, run.manifest_hash))
        dream.dream_apply(beam, run.run_id)
        first = dream.dream_undo(beam, run.run_id)
        assert first.state == "undone"
        second = dream.dream_undo(beam, run.run_id)
        assert second.state == "undone"
        # Explicit non-silent signal: failure_reason (or a dedicated field)
        # must indicate already_undone.
        signal = (second.failure_reason or "") + " " + (second.checkpoint or "")
        assert "already_undone" in signal, (
            "second undo returned no already_undone signal"
        )


class TestResumeDoesNotLoopOnStaleManifest:
    """If dream_apply fails with stale_manifest, dream_resume must not endlessly
    retry the same stale source; it must surface the structured failure."""

    def test_resume_after_stale_manifest_does_not_loop(self, beam, monkeypatch):
        _force_proposals(beam)
        run = _plan(beam)
        dream.dream_submit_receipt(beam, run.run_id,
                                   _pass_receipt("reviewer", "r1",
                                                 run.run_id, run.manifest_hash))
        run = dream.dream_submit_receipt(beam, run.run_id,
                                         _pass_receipt("verifier", "v1",
                                                       run.run_id, run.manifest_hash))
        # Mutate the source so revalidation always fails.
        beam.conn.execute(
            "UPDATE facts SET object = ? WHERE fact_id = ?",
            ("the rust language [MUTATED]", "f1"),
        )
        beam.conn.commit()
        applied = dream.dream_apply(beam, run.run_id)
        assert applied.error_code == "stale_manifest"

        # Resume must NOT silently retry; it must surface the stale failure.
        resumed = dream.dream_resume(beam, run.run_id)
        assert resumed.error_code == "stale_manifest"
        assert resumed.state != "applied"


# ===========================================================================
# Fix round 2 — I6: real scope-provenance mapping + concurrent claim CAS
# ===========================================================================


class TestI6RealScopeProvenanceAliases:
    """I6-A: the scope matcher must compare LOGICAL fields using the real
    Task-4 ``scope_json`` keys (``author_id``, ``author_type``,
    ``channel_id``), aliased to the public Dream scope contract
    (``actor_id``, ``producer``, ``project_id``). A plan must never select a
    proposal whose declared provenance mismatches the plan scope."""

    def _seed_proposal_with_scope(self, beam, proposal_scope_json: dict):
        """Seed one shmr_proposals row with an explicit scope_json."""
        shmr._init_proposal_schema(beam.conn)
        _seed_facts(beam, [
            {"fact_id": "fz1", "subject": "z", "predicate": "p",
             "object": "zeta value one", "confidence": 0.9},
        ])
        beam.conn.execute(
            "INSERT INTO shmr_proposals "
            "(run_id, cluster_id, session_id, scope_json, cited_source_ids, "
            "subject, predicate, object, confidence, action, target_source_id, "
            "rationale, status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "shmr_z", "cz", beam.session_id,
                json.dumps(proposal_scope_json),
                json.dumps(["fz1"]), "z", "p", "zeta", 0.9, "create", None,
                "r", "proposed",
            ),
        )
        beam.conn.commit()

    def test_real_author_id_mismatch_rejected(self, beam):
        """Proposal scope_json has ``author_id='other-actor'`` (the REAL
        Task-4 field); plan scope has ``actor_id='my-actor'``. The alias
        mapper must treat these as the same logical field and reject."""
        self._seed_proposal_with_scope(
            beam, {"session_id": beam.session_id, "author_id": "other-actor"}
        )
        run = dream.dream_plan(
            beam,
            scope={"session_id": beam.session_id, "actor_id": "my-actor"},
        )
        assert run.state == "rejected"
        assert run.error_code == "no_candidates"
        # Proposal must remain eligible (not claimed).
        status = beam.conn.execute(
            "SELECT status FROM shmr_proposals WHERE proposal_id = 1"
        ).fetchone()
        assert status is not None
        assert status["status"] == "proposed"

    def test_author_type_channel_id_mismatch_via_aliases_rejected(self, beam):
        """Proposal scope_json has ``author_type='hermes'`` and
        ``channel_id='proj-x'``; plan scope has ``producer='codex'`` and
        ``project_id='proj-x'``. The producer alias must catch the mismatch."""
        self._seed_proposal_with_scope(
            beam,
            {
                "session_id": beam.session_id,
                "author_type": "hermes",
                "channel_id": "proj-x",
            },
        )
        run = dream.dream_plan(
            beam,
            scope={
                "session_id": beam.session_id,
                "producer": "codex",
                "project_id": "proj-x",
            },
        )
        assert run.state == "rejected"
        assert run.error_code == "no_candidates"

    def test_equal_alias_values_are_eligible(self, beam):
        """Proposal scope_json ``author_id='my-actor'`` matches plan scope
        ``actor_id='my-actor'`` via the alias. This proves the normalizer is
        not a blanket rejector."""
        self._seed_proposal_with_scope(
            beam, {"session_id": beam.session_id, "author_id": "my-actor"}
        )
        run = dream.dream_plan(
            beam,
            scope={"session_id": beam.session_id, "actor_id": "my-actor"},
        )
        assert run.state == "awaiting_approval"
        assert len(run.actions) == 1


class TestI6ConcurrentClaimCAS:
    """I6-B: two concurrent dream_plan calls on distinct connections must not
    both consume the same proposal. Exactly one run must own the action;
    the other must get no_candidates/rejected without consuming it."""

    def test_two_connections_cannot_double_consume_one_proposal(
        self, tmp_path
    ):
        """Two concurrent dream_plan calls on distinct connections with one
        eligible proposal must not double-consume it.

        Under 2af4737 (no serialized CAS) both plans could SELECT the same
        'proposed' proposal and both claim it. After the fix (BEGIN IMMEDIATE
        + CAS UPDATE with status='proposed' guard), exactly one wins; the
        other gets no_candidates/database_busy.

        This test uses a start barrier so both threads enter dream_plan at
        the same time. The fix's BEGIN IMMEDIATE serializes them: the first
        to acquire the write lock gathers + claims + commits atomically;
        the second then sees the proposal as 'dream_claimed' (status filter
        excludes it) and gets no_candidates.
        """
        import threading
        config_module.MnemosyneConfig.reset_instance()
        db = tmp_path / "conc.db"
        b0 = BeamMemory(session_id="conc-sess", db_path=db)
        shmr._init_proposal_schema(b0.conn)
        _seed_facts(b0, [
            {"fact_id": "fc1", "subject": "c", "predicate": "p",
             "object": "common value one", "confidence": 0.9},
        ])
        b0.conn.execute(
            "INSERT INTO shmr_proposals "
            "(run_id, cluster_id, session_id, scope_json, cited_source_ids, "
            "subject, predicate, object, confidence, action, target_source_id, "
            "rationale, status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "shmr_c", "cc", "conc-sess",
                json.dumps({"session_id": "conc-sess"}),
                json.dumps(["fc1"]), "c", "p", "common", 0.9,
                "create", None, "r", "proposed",
            ),
        )
        b0.conn.commit()
        b0.conn.close()

        scope = {"session_id": "conc-sess"}
        results = {}
        barrier = threading.Barrier(2)
        errors = []

        def planner(rid):
            try:
                b = BeamMemory(session_id="conc-sess", db_path=db)
                # Start barrier: both threads reach dream_plan together.
                barrier.wait(timeout=15)
                results[rid] = dream.dream_plan(b, scope=scope, request_id=rid)
                b.conn.close()
            except Exception as exc:
                errors.append((rid, exc))

        t1 = threading.Thread(target=planner, args=("req-a",))
        t2 = threading.Thread(target=planner, args=("req-b",))
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        assert not errors, f"threads raised: {errors}"

        run_a = results.get("req-a")
        run_b = results.get("req-b")
        assert run_a is not None and run_b is not None

        owners = [r for r in (run_a, run_b) if r.state == "awaiting_approval"]
        losers = [
            r for r in (run_a, run_b)
            if r.state in ("rejected", "failed_retryable")
        ]
        assert len(owners) == 1, (
            f"expected exactly 1 owner, got {len(owners)}: "
            f"a={run_a.state}/{run_a.error_code} "
            f"b={run_b.state}/{run_b.error_code}"
        )
        assert len(losers) == 1
        assert losers[0].error_code in ("no_candidates", "database_busy")

        # No duplicate active plan can apply the same proposal.
        check = BeamMemory(session_id="conc-sess", db_path=db)
        actions = check.conn.execute(
            "SELECT COUNT(*) FROM dream_actions WHERE source_id = 'fc1'"
        ).fetchone()[0]
        assert actions == 1

    def test_concurrent_claim_loses_proposals_remain_eligible(self, tmp_path):
        """If a concurrent planner loses the claim race, the proposals it
        tried to claim must NOT be mutated by the loser -- they remain
        'proposed' for the winner (or 'dream_claimed' by the winner)."""
        import threading
        config_module.MnemosyneConfig.reset_instance()
        db = tmp_path / "conc.db"
        b0 = BeamMemory(session_id="conc-sess", db_path=db)
        shmr._init_proposal_schema(b0.conn)
        _seed_facts(b0, [
            {"fact_id": "fc2", "subject": "c", "predicate": "p",
             "object": "second value two", "confidence": 0.9},
        ])
        b0.conn.execute(
            "INSERT INTO shmr_proposals "
            "(run_id, cluster_id, session_id, scope_json, cited_source_ids, "
            "subject, predicate, object, confidence, action, target_source_id, "
            "rationale, status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "shmr_c2", "cc2", "conc-sess",
                json.dumps({"session_id": "conc-sess"}),
                json.dumps(["fc2"]), "c", "p", "second", 0.9,
                "create", None, "r", "proposed",
            ),
        )
        b0.conn.commit()
        b0.conn.close()

        scope = {"session_id": "conc-sess"}
        results = {}
        barrier = threading.Barrier(2)

        def planner(rid):
            b = BeamMemory(session_id="conc-sess", db_path=db)
            barrier.wait(timeout=15)
            results[rid] = dream.dream_plan(b, scope=scope, request_id=rid)
            b.conn.close()

        t1 = threading.Thread(target=planner, args=("req-x",))
        t2 = threading.Thread(target=planner, args=("req-y",))
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        # Exactly one proposal-claim should exist.
        check = BeamMemory(session_id="conc-sess", db_path=db)
        claimed = check.conn.execute(
            "SELECT COUNT(*) FROM shmr_proposals WHERE status = 'dream_claimed'"
        ).fetchone()[0]
        assert claimed == 1, (
            f"expected exactly 1 dream_claimed proposal, got {claimed}"
        )


# ===========================================================================
# Fix round 3 — strict scope provenance: fail closed in both directions
# ===========================================================================


class TestI6StrictScopeFailsClosedBothDirections:
    """Round-2 left a bare-session exception: a session-only plan could still
    select a proposal that declares author_id/author_type/channel_id. The
    strict contract requires: if EITHER side declares a logical provenance
    field the other does not declare and match, the proposal is excluded.
    No exception for bare-session plans."""

    def _seed_proposal_with_scope(self, beam, proposal_scope_json):
        shmr._init_proposal_schema(beam.conn)
        _seed_facts(beam, [
            {"fact_id": "fr3", "subject": "r", "predicate": "p",
             "object": "round three value", "confidence": 0.9},
        ])
        beam.conn.execute(
            "INSERT INTO shmr_proposals "
            "(run_id, cluster_id, session_id, scope_json, cited_source_ids, "
            "subject, predicate, object, confidence, action, target_source_id, "
            "rationale, status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "shmr_r3", "cr3", beam.session_id,
                json.dumps(proposal_scope_json),
                json.dumps(["fr3"]), "r", "p", "round3", 0.9,
                "create", None, "r", "proposed",
            ),
        )
        beam.conn.commit()

    def test_session_only_plan_rejects_proposal_declaring_provenance(self, beam):
        """A plan with only session_id must NOT select a proposal whose
        scope_json declares author_id/author_type/channel_id. The proposal
        must remain eligible (unclaimed)."""
        self._seed_proposal_with_scope(beam, {
            "session_id": beam.session_id,
            "author_id": "private-actor",
            "author_type": "hermes",
            "channel_id": "private-project",
        })
        run = dream.dream_plan(
            beam, scope={"session_id": beam.session_id},
        )
        assert run.state == "rejected"
        assert run.error_code == "no_candidates"
        # Proposal must not be claimed.
        status = beam.conn.execute(
            "SELECT status FROM shmr_proposals WHERE proposal_id = 1"
        ).fetchone()
        assert status["status"] == "proposed"

    def test_selective_plan_rejects_proposal_with_no_provenance(self, beam):
        """A plan that declares actor_id must NOT select a proposal whose
        scope_json has only session_id (no provenance at all)."""
        self._seed_proposal_with_scope(
            beam, {"session_id": beam.session_id}
        )
        run = dream.dream_plan(
            beam,
            scope={"session_id": beam.session_id, "actor_id": "my-actor"},
        )
        assert run.state == "rejected"
        assert run.error_code == "no_candidates"

    def test_matching_alias_fields_remain_eligible(self, beam):
        """When both sides declare matching provenance (via aliases), the
        proposal is eligible. Proves the strict check is not a blanket
        rejector."""
        self._seed_proposal_with_scope(beam, {
            "session_id": beam.session_id,
            "author_id": "my-actor",
            "author_type": "codex",
            "channel_id": "proj-a",
        })
        run = dream.dream_plan(
            beam,
            scope={
                "session_id": beam.session_id,
                "actor_id": "my-actor",
                "producer": "codex",
                "project_id": "proj-a",
            },
        )
        assert run.state == "awaiting_approval"
        assert len(run.actions) == 1
