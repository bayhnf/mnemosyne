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


# ===========================================================================
# Fix round 1 — DeepSeek review binding findings
# ===========================================================================
#
# Each block below targets one binding finding. Tests are written before
# production edits and must fail (RED) for the documented old behavior.


def _force_offline_embeddings(monkeypatch):
    """Pin SHMR's embedding path to the deterministic lexical fallback.

    Finding 3: the test suite must remain offline even when fastembed is
    installed but its model is uncached (which would otherwise trigger a
    network download inside _embeddings.embed). We (a) force shmr's
    _embedding_fn seam to return None so _gather_candidates selects the
    lexical fallback without invoking any embed backend, and (b) plant a
    counting guard on mnemosyne.core.embeddings.embed so the test can prove
    the network path was never reached.
    """
    from mnemosyne.core import embeddings as _emb

    calls = {"n": 0}

    def _counting_guard(_texts):
        calls["n"] += 1
        raise AssertionError(
            "SHMR test reached the network embedding path; tests must stay "
            "offline via the deterministic lexical fallback."
        )

    monkeypatch.setattr(shmr, "_embedding_fn", lambda: None, raising=True)
    monkeypatch.setattr(_emb, "embed", _counting_guard, raising=True)
    return calls


def _seed_episodic(beam, rows):
    """Insert episodic_memory rows with provenance columns."""
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


# ---------------------------------------------------------------------------
# Finding 1: proposal-only must hold across the public SHMR surface
# ---------------------------------------------------------------------------


class TestPublicSurfaceIsProposalOnly:
    """Finding 1: the public harmonize() entry point must not mutate sources.

    The legacy harmonize() calls _apply_beliefs(), UPDATEs facts, writes
    harmonic_beliefs, and queries the nonexistent facts.status column. It is
    part of the supported public surface (documented in docs/shmr.md). After
    the fix it must either delegate to the proposal-only path or be removed
    from the public surface; in either case calling it must leave source rows
    byte-identical and must not raise on a fresh schema.
    """

    def test_harmonize_does_not_mutate_facts_rows(self, tmp_path, monkeypatch):
        _force_offline_embeddings(monkeypatch)
        beam = BeamMemory(session_id="pub1", db_path=tmp_path / "pub1.db")
        _seed_facts(
            beam,
            [
                {"fact_id": "h-1", "subject": "gina", "predicate": "likes",
                 "object": "rust programming language a lot"},
                {"fact_id": "h-2", "subject": "gina", "predicate": "likes",
                 "object": "the rust language for systems"},
            ],
        )
        before = _facts_snapshot(beam)

        # Inject a deterministic LLM that would, under the old code, trigger
        # update + dampen actions against source fact_ids.
        monkeypatch.setattr(
            shmr, "_call_llm",
            lambda prompt, system="": json.dumps(
                [
                    {
                        "subject": "gina", "predicate": "prefers", "object": "rust",
                        "confidence": 0.9, "action": "update",
                        "target_fact_id": "h-1", "rationale": "x",
                    },
                    {
                        "subject": "gina", "predicate": "noise", "object": "x",
                        "confidence": 0.2, "action": "dampen",
                        "target_fact_id": "h-2", "rationale": "y",
                    },
                ]
            ),
        )

        # Must not raise (legacy code hit OperationalError on facts.status).
        result = shmr.harmonize(beam, similarity_threshold=0.3)

        after = _facts_snapshot(beam)
        for table in ("facts", "working_memory", "episodic_memory"):
            assert before[table] == after[table], (
                f"public harmonize() mutated source table {table!r}"
            )
        # And it must not claim it harmonized (applied) anything.
        assert result.get("status") != "harmonized", (
            "public harmonize() reported status='harmonized', which implies it "
            "applied beliefs to sources"
        )

    def test_public_shmr_has_no_source_mutating_entrypoint(self, tmp_path, monkeypatch):
        """No public shmr.* callable may UPDATE/DELETE facts,
        working_memory, or episodic_memory."""
        _force_offline_embeddings(monkeypatch)
        beam = BeamMemory(session_id="pub2", db_path=tmp_path / "pub2.db")
        _seed_facts(
            beam,
            [
                {"fact_id": "p-1", "subject": "hank", "predicate": "uses",
                 "object": "python for data analysis pipelines"},
                {"fact_id": "p-2", "subject": "hank", "predicate": "uses",
                 "object": "python in data analysis scripts"},
            ],
        )
        before = _facts_snapshot(beam)

        # Exercise every public-ish entry that takes a beam.
        monkeypatch.setattr(shmr, "_call_llm", lambda prompt, system="": "[]")
        for fn_name in ("harmonize", "propose_harmony"):
            fn = getattr(shmr, fn_name, None)
            if fn is None:
                continue
            kwargs = {"similarity_threshold": 0.3}
            if fn_name == "propose_harmony":
                kwargs["llm_call"] = lambda prompt, system="": "[]"
            try:
                fn(beam, **kwargs)
            except TypeError:
                pass

        after = _facts_snapshot(beam)
        for table in ("facts", "working_memory", "episodic_memory"):
            assert before[table] == after[table]


# ---------------------------------------------------------------------------
# Finding 2: propose_harmony must be atomic for the whole run
# ---------------------------------------------------------------------------


class TestRunLevelAtomicity:
    """Finding 2: a failure on a later cluster must roll back ALL clusters."""

    def test_multi_cluster_failure_rolls_back_earlier_cluster_rows(
        self, tmp_path, monkeypatch
    ):
        _force_offline_embeddings(monkeypatch)
        beam = BeamMemory(session_id="atom", db_path=tmp_path / "atom.db")
        # Two disjoint clusters: python facts and rust facts.
        _seed_facts(
            beam,
            [
                {"fact_id": "py-a", "subject": "ivy", "predicate": "uses",
                 "object": "python data analysis"},
                {"fact_id": "py-b", "subject": "ivy", "predicate": "uses",
                 "object": "python analysis scripts"},
                {"fact_id": "rs-a", "subject": "jack", "predicate": "likes",
                 "object": "rust systems language"},
                {"fact_id": "rs-b", "subject": "jack", "predicate": "likes",
                 "object": "rust for systems"},
            ],
        )
        before = _facts_snapshot(beam)

        # LLM returns a valid proposal for each cluster (two calls).
        llm = _RecordingLLM(
            [
                json.dumps(
                    [
                        {
                            "subject": "ivy", "predicate": "prefers", "object": "python",
                            "confidence": 0.8, "action": "create",
                            "target_source_id": None, "rationale": "p1",
                        }
                    ]
                ),
                json.dumps(
                    [
                        {
                            "subject": "jack", "predicate": "prefers", "object": "rust",
                            "confidence": 0.8, "action": "create",
                            "target_source_id": None, "rationale": "p2",
                        }
                    ]
                ),
            ]
        )

        # Sabotage the proposal INSERT but only on the SECOND cluster's rows.
        # We detect "second cluster" by watching for the jack/rust content.
        original_execute = beam.conn.execute
        insert_seen = {"n": 0}

        def flaky_execute(sql, *params):
            if isinstance(sql, str) and "INSERT INTO shmr_proposals" in sql:
                insert_seen["n"] += 1
                # First cluster (python) inserts succeed; second cluster (rust)
                # is the 2nd INSERT and must raise, AFTER the 1st succeeded.
                if insert_seen["n"] == 2:
                    raise sqlite3.OperationalError("simulated late failure")
            return original_execute(sql, *params)

        monkeypatch.setattr(beam.conn, "execute", flaky_execute)

        result = shmr.propose_harmony(
            beam, llm_call=llm, similarity_threshold=0.3, min_cluster_size=2
        )

        # At least one INSERT must have succeeded before the failure, proving
        # the test actually exercises cross-cluster atomicity.
        assert insert_seen["n"] >= 2, (
            "test setup error: expected >=2 INSERT attempts (one per cluster); "
            f"got {insert_seen['n']}"
        )
        # The whole run must roll back: zero proposals persisted.
        assert result["status"] == "rolled_back", result
        rows = beam.conn.execute("SELECT * FROM shmr_proposals").fetchall()
        assert len(rows) == 0, (
            f"run-level rollback failed: {len(rows)} proposal(s) survived"
        )
        # Truthful counters: nothing claimed persisted.
        assert result["proposals_persisted"] == 0
        # Sources untouched.
        after = _facts_snapshot(beam)
        for table in ("facts", "working_memory", "episodic_memory"):
            assert before[table] == after[table]


# ---------------------------------------------------------------------------
# Finding 3: tests deterministically prevent external embedding access
# ---------------------------------------------------------------------------


class TestNoNetworkEmbeddings:
    """Finding 3: prove the lexical fallback is used and the network path
    is never reached, even if fastembed is installed with an uncached model."""

    def test_propose_harmony_uses_lexical_fallback_never_network(
        self, tmp_path, monkeypatch
    ):
        embed_calls = _force_offline_embeddings(monkeypatch)
        beam = BeamMemory(session_id="off", db_path=tmp_path / "off.db")
        _seed_facts(
            beam,
            [
                {"fact_id": "o-1", "subject": "kate", "predicate": "uses",
                 "object": "python for analysis"},
                {"fact_id": "o-2", "subject": "kate", "predicate": "uses",
                 "object": "python in analysis"},
            ],
        )
        llm = _RecordingLLM(["[]"])
        # Must not raise the AssertionError planted in _force_offline_embeddings.
        result = shmr.propose_harmony(
            beam, llm_call=llm, similarity_threshold=0.3
        )
        assert result["status"] in ("no_convergence", "proposed")
        assert result["clusters_found"] >= 1
        # And prove the embedding backend was never consulted.
        assert embed_calls["n"] == 0, (
            f"SHMR called the embedding backend {embed_calls['n']} time(s); "
            "the lexical fallback must run instead to stay offline."
        )


# ---------------------------------------------------------------------------
# Finding 4: malformed LLM confidence is rejected, not raised
# ---------------------------------------------------------------------------


class TestMalformedConfidenceRejected:
    """Finding 4: non-numeric / NaN / Infinity / out-of-range confidence must
    be counted as rejected untrusted output, not raise or leave partial state."""

    def test_non_string_confidence_is_rejected_not_raised(self, tmp_path, monkeypatch):
        _force_offline_embeddings(monkeypatch)
        beam = BeamMemory(session_id="conf", db_path=tmp_path / "conf.db")
        _seed_facts(
            beam,
            [
                {"fact_id": "c-1", "subject": "liam", "predicate": "uses",
                 "object": "python for data analysis pipelines"},
                {"fact_id": "c-2", "subject": "liam", "predicate": "uses",
                 "object": "python data analysis scripts"},
            ],
        )
        llm = _RecordingLLM(
            [
                json.dumps(
                    [
                        {
                            "subject": "liam", "predicate": "prefers", "object": "python",
                            "confidence": "not a number", "action": "create",
                            "target_source_id": None, "rationale": "bad conf",
                        }
                    ]
                )
            ]
        )
        result = shmr.propose_harmony(
            beam, llm_call=llm, similarity_threshold=0.3
        )
        assert result["status"] == "no_convergence"
        rows = beam.conn.execute("SELECT * FROM shmr_proposals").fetchall()
        assert len(rows) == 0
        assert result["proposals_rejected"] >= 1, (
            "malformed-confidence proposal must be counted as rejected"
        )

    def test_nan_and_infinity_confidence_rejected(self, tmp_path, monkeypatch):
        _force_offline_embeddings(monkeypatch)
        beam = BeamMemory(session_id="nan", db_path=tmp_path / "nan.db")
        _seed_facts(
            beam,
            [
                {"fact_id": "n-1", "subject": "mia", "predicate": "uses",
                 "object": "python data analysis pipelines"},
                {"fact_id": "n-2", "subject": "mia", "predicate": "uses",
                 "object": "python data analysis scripts"},
            ],
        )
        llm = _RecordingLLM(
            [
                # NaN and Infinity are valid JSON5 but Python's json module
                # accepts them as float('nan')/float('inf'). They must be
                # rejected, not clamped.
                '[{"subject":"mia","predicate":"p","object":"o",'
                '"confidence":NaN,"action":"create","target_source_id":null},'
                '{"subject":"mia","predicate":"p","object":"o",'
                '"confidence":Infinity,"action":"create","target_source_id":null}]'
            ]
        )
        result = shmr.propose_harmony(
            beam, llm_call=llm, similarity_threshold=0.3
        )
        rows = beam.conn.execute("SELECT * FROM shmr_proposals").fetchall()
        assert len(rows) == 0
        assert result["proposals_rejected"] >= 2


# ---------------------------------------------------------------------------
# Finding 5: do not silently mix provenance inside a cluster
# ---------------------------------------------------------------------------


class TestProvenancePartition:
    """Finding 5: candidates with different session / author_type (producer) /
    author_id (actor) / channel_id (project) must not be silently merged."""

    def test_mixed_author_types_are_not_merged_into_one_cluster(
        self, tmp_path, monkeypatch
    ):
        _force_offline_embeddings(monkeypatch)
        beam = BeamMemory(session_id="prov", db_path=tmp_path / "prov.db")
        # Two episodic memories with identical text but different producers
        # (author_type). Under the old code they would cluster together and
        # the persisted scope would silently pick cluster[0] as authority.
        _seed_episodic(
            beam,
            [
                {
                    "id": "e-human", "content": "nora uses python for data analysis",
                    "author_type": "human", "author_id": "nora", "channel_id": "proj-x",
                },
                {
                    "id": "e-agent", "content": "nora uses python for data analysis",
                    "author_type": "agent", "author_id": "agent-7", "channel_id": "proj-x",
                },
            ],
        )
        llm = _RecordingLLM(
            [json.dumps([{"subject": "nora", "predicate": "uses", "object": "python",
                          "confidence": 0.8, "action": "create",
                          "target_source_id": None, "rationale": "r"}])]
        )
        shmr.propose_harmony(
            beam, llm_call=llm, similarity_threshold=0.3, min_cluster_size=2
        )
        rows = beam.conn.execute("SELECT * FROM shmr_proposals").fetchall()
        # Either no cluster formed (partitioned) or each persisted proposal
        # records a single, consistent scope — never a mixed one.
        for r in rows:
            scope = json.loads(dict(r)["scope_json"])
            author_types = {scope.get("author_type")}
            assert len(author_types) == 1, (
                f"mixed provenance in one proposal scope: {scope}"
            )
        # And specifically: we must NOT have merged the human + agent memory
        # into a single cluster that cites both ids with one authority scope.
        for r in rows:
            cited = set(json.loads(dict(r)["cited_source_ids"]))
            assert not ({"e-human", "e-agent"} <= cited), (
                "human and agent provenance were silently merged into one proposal"
            )


# ---------------------------------------------------------------------------
# Finding 6: cited_source_ids must be truthful per-proposal citations
# ---------------------------------------------------------------------------


class TestTruthfulCitations:
    """Finding 6: cited_source_ids must record the ids the LLM actually cited
    for THAT proposal (validated as a subset of the cluster), not the entire
    cluster's ids under a single authority."""

    def test_cited_ids_are_the_actual_per_proposal_citations(
        self, tmp_path, monkeypatch
    ):
        _force_offline_embeddings(monkeypatch)
        beam = BeamMemory(session_id="cite", db_path=tmp_path / "cite.db")
        _seed_facts(
            beam,
            [
                {"fact_id": "ci-1", "subject": "oscar", "predicate": "uses",
                 "object": "python data analysis pipelines"},
                {"fact_id": "ci-2", "subject": "oscar", "predicate": "uses",
                 "object": "python data analysis scripts"},
                {"fact_id": "ci-3", "subject": "oscar", "predicate": "uses",
                 "object": "python data analysis notebooks"},
            ],
        )
        # LLM cites only ci-1 and ci-2 for its single proposal, NOT ci-3.
        llm = _RecordingLLM(
            [
                json.dumps(
                    [
                        {
                            "subject": "oscar", "predicate": "prefers",
                            "object": "python", "confidence": 0.85,
                            "action": "create", "target_source_id": None,
                            "rationale": "r",
                            "cited_source_ids": ["ci-1", "ci-2"],
                        }
                    ]
                )
            ]
        )
        result = shmr.propose_harmony(
            beam, llm_call=llm, similarity_threshold=0.3, min_cluster_size=2
        )
        assert result["proposals_persisted"] >= 1
        rows = beam.conn.execute("SELECT * FROM shmr_proposals").fetchall()
        assert len(rows) == 1
        cited = set(json.loads(dict(rows[0])["cited_source_ids"]))
        # Must be exactly what the model cited, validated against the cluster,
        # not the whole cluster's id set.
        assert cited == {"ci-1", "ci-2"}, (
            f"expected cited_source_ids == {{ci-1, ci-2}}, got {cited}"
        )
        assert "ci-3" not in cited

    def test_citation_of_id_not_in_cluster_is_rejected(
        self, tmp_path, monkeypatch
    ):
        _force_offline_embeddings(monkeypatch)
        beam = BeamMemory(session_id="cite2", db_path=tmp_path / "cite2.db")
        _seed_facts(
            beam,
            [
                {"fact_id": "ck-1", "subject": "penny", "predicate": "uses",
                 "object": "python data analysis pipelines"},
                {"fact_id": "ck-2", "subject": "penny", "predicate": "uses",
                 "object": "python data analysis scripts"},
            ],
        )
        # Model cites an id that is NOT in the cluster.
        llm = _RecordingLLM(
            [
                json.dumps(
                    [
                        {
                            "subject": "penny", "predicate": "prefers",
                            "object": "python", "confidence": 0.85,
                            "action": "create", "target_source_id": None,
                            "rationale": "r",
                            "cited_source_ids": ["ck-1", "FABRICATED"],
                        }
                    ]
                )
            ]
        )
        result = shmr.propose_harmony(
            beam, llm_call=llm, similarity_threshold=0.3
        )
        rows = beam.conn.execute("SELECT * FROM shmr_proposals").fetchall()
        assert len(rows) == 0, (
            "proposal with an out-of-cluster cited id must be rejected"
        )
        assert result["proposals_rejected"] >= 1
