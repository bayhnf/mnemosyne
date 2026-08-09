"""
Task 4: Safe SHMR -> Dream proposal foundation.

SHMR is a read-only candidate generator. It must:

- run against a fresh standard Mnemosyne schema (no legacy ``facts.status``),
- include stable source IDs in every LLM prompt,
- record exact cited source IDs + explicit scope on every persisted proposal,
- reject an LLM ``target_fact_id`` that is not one of this cluster's sources,
- leave all source ``facts`` / ``working_memory`` / ``episodic_memory`` rows
  byte-identical to their pre-run state, and
- roll back proposal writes when persistence fails, without touching sources.

These tests fail against the current ``harmonize()`` implementation, which
mutates ``facts``, queries a nonexistent ``status`` column, embeds no source
IDs in the prompt, accepts arbitrary ``target_fact_id`` values, and makes
real network calls. They define the contract for the new proposal-only API.
"""

from __future__ import annotations

import json
import sqlite3


from mnemosyne.core.beam import BeamMemory
from mnemosyne.core import shmr


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _seed_facts(beam, rows):
    """Insert fact rows and return the list of fact_ids written."""
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


def _facts_snapshot(beam):
    """Capture every source row that SHMR must not mutate."""
    out = {}
    for table in ("facts", "working_memory", "episodic_memory"):
        rows = beam.conn.execute(
            f"SELECT * FROM {table} ORDER BY rowid"
        ).fetchall()
        out[table] = [dict(r) for r in rows]
    return out


class _RecordingLLM:
    """Deterministic LLM seam. Records the prompt and replays a canned reply.

    Each entry in ``replies`` is returned for successive calls. The captured
    prompt is exposed so tests can assert source IDs were embedded in it.
    """

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, prompt, system=""):
        self.prompts.append(prompt)
        if not self.replies:
            return ""
        return self.replies.pop(0)


# ---------------------------------------------------------------------------
# Schema: runs against a fresh standard Mnemosyne schema
# ---------------------------------------------------------------------------


class TestFreshSchema:
    def test_propose_runs_against_brand_new_beam_without_error(self, tmp_path):
        """A freshly created BeamMemory (no migrations, no legacy tables) must
        accept SHMR without raising. The current implementation queries
        ``facts.status``, which does not exist in the standard schema."""
        beam = BeamMemory(session_id="s1", db_path=tmp_path / "fresh.db")
        _seed_facts(
            beam,
            [
                {"fact_id": "f1", "subject": "alice", "predicate": "likes",
                 "object": "rust programming language"},
                {"fact_id": "f2", "subject": "alice", "predicate": "likes",
                 "object": "the rust language for systems work"},
            ],
        )
        llm = _RecordingLLM(
            [
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
                )
            ]
        )
        # Must not raise OperationalError on missing facts.status.
        result = shmr.propose_harmony(beam, llm_call=llm, similarity_threshold=0.4)
        assert result["status"] in ("proposed", "no_convergence")
        assert result["clusters_found"] >= 1


# ---------------------------------------------------------------------------
# Scope isolation + explicit scope on persisted proposals
# ---------------------------------------------------------------------------


class TestScopeIsolation:
    def test_proposal_records_session_and_available_scope_fields(self, tmp_path):
        beam = BeamMemory(session_id="scope-sess", db_path=tmp_path / "scope.db")
        _seed_facts(
            beam,
            [
                {"fact_id": "fa", "subject": "bob", "predicate": "uses",
                 "object": "python for data analysis pipelines"},
                {"fact_id": "fb", "subject": "bob", "predicate": "uses",
                 "object": "python in data analysis"},
            ],
        )
        llm = _RecordingLLM(
            [
                json.dumps(
                    [
                        {
                            "subject": "bob",
                            "predicate": "prefers",
                            "object": "python",
                            "confidence": 0.85,
                            "action": "create",
                            "target_fact_id": "fa",
                            "rationale": "corroborated",
                        }
                    ]
                )
            ]
        )
        result = shmr.propose_harmony(beam, llm_call=llm, similarity_threshold=0.4)
        assert result["proposals_persisted"] >= 1

        rows = beam.conn.execute(
            "SELECT * FROM shmr_proposals ORDER BY rowid"
        ).fetchall()
        assert len(rows) == 1
        row = dict(rows[0])
        # Explicit scope recorded for Dream manifest (Task 5).
        assert row["session_id"] == "scope-sess"
        scope = json.loads(row["scope_json"]) if row["scope_json"] else {}
        assert scope.get("session_id") == "scope-sess"
        # Cited source IDs are recorded exactly.
        cited = json.loads(row["cited_source_ids"]) if row["cited_source_ids"] else []
        assert "fa" in cited and "fb" in cited


# ---------------------------------------------------------------------------
# Prompt contains stable source IDs
# ---------------------------------------------------------------------------


class TestPromptSourceIds:
    def test_every_prompt_contains_each_source_fact_id(self, tmp_path):
        beam = BeamMemory(session_id="p-sess", db_path=tmp_path / "prompt.db")
        ids = _seed_facts(
            beam,
            [
                {"fact_id": "src-001", "subject": "carol", "predicate": "lives",
                 "object": "in jakarta the capital of indonesia"},
                {"fact_id": "src-002", "subject": "carol", "predicate": "based",
                 "object": "jakarta capital of indonesia region"},
            ],
        )
        llm = _RecordingLLM(["[]"])  # no proposals is fine; we assert the prompt
        shmr.propose_harmony(beam, llm_call=llm, similarity_threshold=0.4)
        assert llm.prompts, "SHMR must have called the LLM at least once"
        prompt_text = "\n".join(llm.prompts)
        for fid in ids:
            assert fid in prompt_text, (
                f"source id {fid!r} must appear in the LLM prompt so the model "
                "can cite it back; got prompt lacking it"
            )


# ---------------------------------------------------------------------------
# Out-of-cluster hallucinated target rejection
# ---------------------------------------------------------------------------


class TestHallucinatedTargetRejection:
    def test_target_fact_id_not_in_cluster_is_dropped(self, tmp_path):
        beam = BeamMemory(
            session_id="hall-sess", db_path=tmp_path / "halluc.db"
        )
        _seed_facts(
            beam,
            [
                {"fact_id": "real-1", "subject": "dan", "predicate": "eats",
                 "object": "sushi every week regularly"},
                {"fact_id": "real-2", "subject": "dan", "predicate": "enjoys",
                 "object": "sushi weekly as a regular habit"},
            ],
        )
        # LLM cites a fact_id that does not exist in this cluster.
        bad = _RecordingLLM(
            [
                json.dumps(
                    [
                        {
                            "subject": "dan",
                            "predicate": "loves",
                            "object": "sushi",
                            "confidence": 0.99,
                            "action": "update",
                            "target_fact_id": "FABRICATED-NOT-IN-CLUSTER",
                            "rationale": "hallucinated target",
                        }
                    ]
                )
            ]
        )
        result = shmr.propose_harmony(beam, llm_call=bad, similarity_threshold=0.4)
        # The hallucinated proposal must not be persisted.
        rows = beam.conn.execute(
            "SELECT * FROM shmr_proposals"
        ).fetchall()
        assert len(rows) == 0, (
            "out-of-cluster target_fact_id must be rejected; "
            f"got {len(rows)} persisted proposal(s)"
        )
        assert result["proposals_rejected"] >= 1


# ---------------------------------------------------------------------------
# Source rows unchanged
# ---------------------------------------------------------------------------


class TestSourcesUnchanged:
    def test_facts_working_episodic_rows_are_byte_identical_after_run(self, tmp_path):
        beam = BeamMemory(
            session_id="immutable-sess", db_path=tmp_path / "imm.db"
        )
        _seed_facts(
            beam,
            [
                {"fact_id": "im-1", "subject": "eve", "predicate": "codes",
                 "object": "in typescript for the frontend app daily"},
                {"fact_id": "im-2", "subject": "eve", "predicate": "writes",
                 "object": "typescript on the frontend application often"},
            ],
        )
        before = _facts_snapshot(beam)

        llm = _RecordingLLM(
            [
                # update action targeting a real source id,
                # dampen action targeting a real source id,
                # plus a hallucinated target.
                json.dumps(
                    [
                        {
                            "subject": "eve",
                            "predicate": "uses",
                            "object": "typescript",
                            "confidence": 0.9,
                            "action": "update",
                            "target_fact_id": "im-1",
                            "rationale": "x",
                        },
                        {
                            "subject": "eve",
                            "predicate": "noise",
                            "object": "x",
                            "confidence": 0.2,
                            "action": "dampen",
                            "target_fact_id": "im-2",
                            "rationale": "y",
                        },
                    ]
                )
            ]
        )
        shmr.propose_harmony(beam, llm_call=llm, similarity_threshold=0.4)
        after = _facts_snapshot(beam)
        for table in ("facts", "working_memory", "episodic_memory"):
            assert before[table] == after[table], (
                f"SHMR mutated source table {table!r}; "
                f"before={before[table]!r} after={after[table]!r}"
            )


# ---------------------------------------------------------------------------
# Transaction rollback on persistence failure
# ---------------------------------------------------------------------------


class TestTransactionRollback:
    def test_persistence_failure_rolls_back_proposals_without_mutating_sources(
        self, tmp_path, monkeypatch
    ):
        beam = BeamMemory(
            session_id="rb-sess", db_path=tmp_path / "rollback.db"
        )
        _seed_facts(
            beam,
            [
                {"fact_id": "rb-1", "subject": "frank", "predicate": "runs",
                 "object": "marathons on weekends in the park"},
                {"fact_id": "rb-2", "subject": "frank", "predicate": "jogs",
                 "object": "marathon distances every weekend morning"},
            ],
        )
        before = _facts_snapshot(beam)

        llm = _RecordingLLM(
            [
                json.dumps(
                    [
                        {
                            "subject": "frank",
                            "predicate": "trains",
                            "object": "marathons",
                            "confidence": 0.8,
                            "action": "create",
                            "target_fact_id": "rb-1",
                            "rationale": "z",
                        }
                    ]
                )
            ]
        )

        # Force the proposal INSERT to fail mid-run by dropping the target
        # table after SHMR has read sources but before it persists.
        original_execute = beam.conn.execute

        call_count = {"n": 0}

        def flaky_execute(sql, *params):
            call_count["n"] += 1
            if isinstance(sql, str) and "INSERT INTO shmr_proposals" in sql:
                raise sqlite3.OperationalError("simulated persistence failure")
            return original_execute(sql, *params)

        monkeypatch.setattr(beam.conn, "execute", flaky_execute)

        # Must not raise out; SHMR swallows the persistence error and rolls back.
        result = shmr.propose_harmony(beam, llm_call=llm, similarity_threshold=0.4)
        assert result["status"] == "rolled_back"

        # No partial proposal rows.
        rows = beam.conn.execute(
            "SELECT * FROM shmr_proposals"
        ).fetchall()
        assert len(rows) == 0

        # Sources untouched.
        after = _facts_snapshot(beam)
        for table in ("facts", "working_memory", "episodic_memory"):
            assert before[table] == after[table]
