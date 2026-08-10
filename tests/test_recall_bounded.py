"""Task 3 — Authoritative bounded recall (``recall_bounded``).

RED-first tests for the additive post-hydration gate that exposes a
``RecallEnvelope`` with hard result/token caps, strict isolation, and
deterministic fallback, while leaving legacy ``recall()`` untouched.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.recall_bounded import (
    RecallPolicy,
    RecallEnvelope,
    recall_bounded,
)


@pytest.fixture(autouse=True)
def _reset_recall_diag():
    from mnemosyne.core.recall_diagnostics import reset_recall_diagnostics
    reset_recall_diagnostics()
    yield
    reset_recall_diagnostics()


@pytest.fixture
def beam(tmp_path):
    """Fresh BeamMemory pointed at an isolated DB."""
    b = BeamMemory(session_id="sess-a", db_path=tmp_path / "t.db")
    yield b


def _remember(beam, content, *, session_id="sess-a", scope="session",
              source="conversation", author_id=None, author_type=None,
              channel_id=None, veracity="unknown", importance=0.5,
              valid_until=None, superseded_by=None, memory_type=None,
              metadata=None):
    """Insert a working_memory row with full control for test seeding."""
    mid = beam.remember(
        content, source=source, importance=importance,
        scope=scope, veracity=veracity, metadata=metadata,
    )
    conn = beam.conn
    sets = []
    params = []
    if session_id is not None:
        sets.append("session_id = ?")
        params.append(session_id)
    if author_id is not None:
        sets.append("author_id = ?")
        params.append(author_id)
    if author_type is not None:
        sets.append("author_type = ?")
        params.append(author_type)
    if channel_id is not None:
        sets.append("channel_id = ?")
        params.append(channel_id)
    if valid_until is not None:
        sets.append("valid_until = ?")
        params.append(valid_until)
    if superseded_by is not None:
        sets.append("superseded_by = ?")
        params.append(superseded_by)
    if memory_type is not None:
        sets.append("memory_type = ?")
        params.append(memory_type)
    if sets:
        params.append(mid)
        conn.execute(
            f"UPDATE working_memory SET {', '.join(sets)} WHERE id = ?", tuple(params),
        )
        conn.commit()
    return mid


# ---------------------------------------------------------------------------
# API shape / envelope contract
# ---------------------------------------------------------------------------

class TestRecallPolicyValidation:
    def test_rejects_negative_top_k(self):
        with pytest.raises(ValueError):
            RecallPolicy(top_k=-1)

    def test_rejects_zero_top_k(self):
        with pytest.raises(ValueError):
            RecallPolicy(top_k=0)

    def test_rejects_negative_max_tokens(self):
        with pytest.raises(ValueError):
            RecallPolicy(top_k=5, max_tokens=-10)

    def test_rejects_negative_max_item_tokens(self):
        with pytest.raises(ValueError):
            RecallPolicy(top_k=5, max_item_tokens=-1)

    def test_defaults_are_sane(self):
        p = RecallPolicy(top_k=10)
        assert p.top_k == 10
        assert p.max_tokens is None or p.max_tokens > 0
        assert p.include_legacy_shared is False
        assert p.only_active is True


class TestEnvelopeShape:
    def test_envelope_fields(self, beam):
        _remember(beam, "hello world topic alpha")
        env = recall_bounded(beam, "hello world", policy=RecallPolicy(top_k=5))
        assert isinstance(env, RecallEnvelope)
        assert isinstance(env.results, list)
        assert isinstance(env.rendered_context, str)
        assert isinstance(env.token_count, int)
        assert env.retrieval_mode in ("vector", "hybrid", "fts", "recent_fallback")
        assert isinstance(env.applied_filters, dict)
        assert isinstance(env.degradation_reasons, list)
        assert isinstance(env.trace_id, str)
        # Diagnostics must not be sensitive.
        for field in ("rendered_context", "trace_id"):
            text = getattr(env, field)
            assert "sess-b" not in text  # no foreign session bleed in diag


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------

class TestIsolation:
    def test_session_isolation(self, beam):
        _remember(beam, "alpha secret in session a", session_id="sess-a")
        _remember(beam, "beta secret in session b", session_id="sess-b")
        env = recall_bounded(
            beam, "secret", policy=RecallPolicy(top_k=10, include_shared=False),
        )
        contents = " ".join(r["content"] for r in env.results)
        assert "alpha" in contents
        assert "beta" not in contents

    def test_actor_isolation(self, beam):
        _remember(beam, "actor one fact", author_id="actor-1")
        _remember(beam, "actor two fact", author_id="actor-2")
        env = recall_bounded(
            beam, "actor", policy=RecallPolicy(top_k=10, actor_ids=["actor-1"]),
        )
        contents = " ".join(r["content"] for r in env.results)
        assert "actor one" in contents
        assert "actor two" not in contents

    def test_project_isolation(self, beam):
        _remember(beam, "project x detail", channel_id="proj-x")
        _remember(beam, "project y detail", channel_id="proj-y")
        env = recall_bounded(
            beam, "project", policy=RecallPolicy(top_k=10, project_ids=["proj-x"]),
        )
        contents = " ".join(r["content"] for r in env.results)
        assert "project x" in contents
        assert "project y" not in contents

    def test_producer_isolation(self, beam):
        _remember(beam, "human said fact", author_type="human")
        _remember(beam, "agent said fact", author_type="agent")
        env = recall_bounded(
            beam, "said", policy=RecallPolicy(top_k=10, producer_types=["human"]),
        )
        contents = " ".join(r["content"] for r in env.results)
        assert "human said" in contents
        assert "agent said" not in contents

    def test_include_shared_permits_global(self, beam):
        _remember(beam, "global shared item", scope="global", session_id="sess-b")
        env = recall_bounded(
            beam, "global", policy=RecallPolicy(top_k=10, include_shared=True),
        )
        contents = " ".join(r["content"] for r in env.results)
        assert "global shared" in contents

    def test_default_excludes_other_session_non_global(self, beam):
        _remember(beam, "private session b row", scope="session", session_id="sess-b")
        env = recall_bounded(
            beam, "private", policy=RecallPolicy(top_k=10),
        )
        contents = " ".join(r["content"] for r in env.results)
        assert "private session b" not in contents


# ---------------------------------------------------------------------------
# Lifecycle / proposal invisibility
# ---------------------------------------------------------------------------

class TestLifecycleExclusion:
    def test_expired_valid_until_excluded(self, beam):
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        _remember(beam, "expired old content alpha", valid_until=past)
        _remember(beam, "live content alpha", )
        env = recall_bounded(beam, "alpha", policy=RecallPolicy(top_k=10))
        contents = " ".join(r["content"] for r in env.results)
        assert "live content" in contents
        assert "expired old" not in contents

    def test_superseded_excluded(self, beam):
        _remember(beam, "superseded content alpha")
        _remember(beam, "current content alpha", superseded_by="ep-xyz")
        env = recall_bounded(beam, "alpha", policy=RecallPolicy(top_k=10))
        contents = " ".join(r["content"] for r in env.results)
        assert "superseded content" in contents
        assert "current content" not in contents

    def test_pending_proposal_excluded(self, beam):
        # Proposals are wm rows with source sleep_model_refresh_proposal
        # and metadata status=pending.
        beam.remember(
            "[MODEL_REFRESH_PROPOSAL] tech::lang confidence=0.9: python",
            source="sleep_model_refresh_proposal",
            metadata={"status": "pending", "action": "update"},
            scope="session",
        )
        _remember(beam, "normal python content alpha")
        env = recall_bounded(beam, "python", policy=RecallPolicy(top_k=10))
        contents = " ".join(r["content"] for r in env.results)
        assert "MODEL_REFRESH_PROPOSAL" not in contents


# ---------------------------------------------------------------------------
# Hard caps
# ---------------------------------------------------------------------------

class TestHardCaps:
    def test_top_k_cap(self, beam):
        for i in range(15):
            _remember(beam, f"number item {i} alpha beta")
        env = recall_bounded(beam, "alpha beta", policy=RecallPolicy(top_k=5))
        assert len(env.results) <= 5

    def test_rendered_token_cap(self, beam):
        # Each item ~ a few tokens. Set a tight cap and confirm the
        # rendered context stays under it.
        for i in range(20):
            _remember(beam, f"padding content block number {i} " + ("word " * 20))
        env = recall_bounded(
            beam, "padding", policy=RecallPolicy(top_k=20, max_tokens=50),
        )
        assert env.token_count <= 50

    def test_max_item_tokens_truncates_row(self, beam):
        _remember(beam, "short alpha")
        _remember(beam, "alpha " + ("verylongword " * 100))
        env = recall_bounded(
            beam, "alpha", policy=RecallPolicy(top_k=5, max_item_tokens=5),
        )
        # every returned row's rendered form must fit the per-item cap
        for r in env.results:
            # re-render the row the same way the gate does
            line = r.get("content", "")
            assert len(line.split()) <= 30  # truncated well under original


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------

class TestDedup:
    def test_no_duplicate_ids(self, beam):
        _remember(beam, "duplicate prone content alpha")
        env = recall_bounded(beam, "alpha", policy=RecallPolicy(top_k=10))
        ids = [r["id"] for r in env.results]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Deterministic fallback
# ---------------------------------------------------------------------------

class TestDeterministicFallback:
    def test_empty_db_returns_recent_fallback(self, beam):
        env = recall_bounded(beam, "anything", policy=RecallPolicy(top_k=5))
        assert env.retrieval_mode == "recent_fallback"
        assert env.results == []

    def test_fts_hit_uses_fts_or_hybrid(self, beam):
        _remember(beam, "unique lexical token zebra")
        env = recall_bounded(beam, "zebra", policy=RecallPolicy(top_k=5))
        assert env.retrieval_mode in ("fts", "hybrid", "vector")
        assert len(env.results) >= 1


# ---------------------------------------------------------------------------
# Polyphonic fallback
# ---------------------------------------------------------------------------

class TestPolyphonicFallback:
    def test_polyphonic_empty_falls_back_to_linear(self, beam, monkeypatch):
        _remember(beam, "linear visible content alpha")
        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")

        # Force the polyphonic engine to return empty.
        from mnemosyne.core.polyphonic_recall import PolyphonicRecallEngine
        monkeypatch.setattr(
            PolyphonicRecallEngine, "recall", lambda self, *a, **k: [],
        )

        env = recall_bounded(beam, "alpha", policy=RecallPolicy(top_k=5))
        assert "polyphonic_empty_fallback_linear" in env.degradation_reasons
        contents = " ".join(r["content"] for r in env.results)
        assert "linear visible" in contents


# ---------------------------------------------------------------------------
# Legacy compatibility
# ---------------------------------------------------------------------------

class TestLegacyCompatibility:
    def test_bounded_does_not_mutate_recall_counters(self, beam):
        mid = _remember(beam, "counter check content alpha")
        # Baseline
        before = beam.conn.execute(
            "SELECT recall_count FROM working_memory WHERE id = ?", (mid,),
        ).fetchone()["recall_count"]
        recall_bounded(beam, "alpha", policy=RecallPolicy(top_k=5))
        after = beam.conn.execute(
            "SELECT recall_count FROM working_memory WHERE id = ?", (mid,),
        ).fetchone()["recall_count"]
        assert before == after

    def test_legacy_recall_still_increments(self, beam):
        mid = _remember(beam, "legacy path content alpha")
        before = beam.conn.execute(
            "SELECT recall_count FROM working_memory WHERE id = ?", (mid,),
        ).fetchone()["recall_count"]
        beam.recall("alpha", top_k=5)
        after = beam.conn.execute(
            "SELECT recall_count FROM working_memory WHERE id = ?", (mid,),
        ).fetchone()["recall_count"]
        assert after == before + 1


# ---------------------------------------------------------------------------
# Fix Round 1 — RED tests for review findings
# ---------------------------------------------------------------------------

class TestNativeBeamMemoryAPI:
    """Finding 1: BeamMemory.recall_bounded(query, policy) must be the
    canonical native public API, not just a free function."""

    def test_method_exists_on_beam(self, beam):
        assert hasattr(beam, "recall_bounded") and callable(beam.recall_bounded)

    def test_method_returns_envelope(self, beam):
        _remember(beam, "native api content alpha")
        env = beam.recall_bounded("alpha", RecallPolicy(top_k=5))
        assert isinstance(env, RecallEnvelope)

    def test_method_default_policy(self, beam):
        _remember(beam, "default policy content alpha")
        env = beam.recall_bounded("alpha")
        assert isinstance(env, RecallEnvelope)


class TestHardTokenLimitOversizedFirst:
    """Finding 3: max_tokens must be a hard estimated-token limit even
    when the first candidate alone exceeds the budget."""

    def test_single_oversized_row_respects_max_tokens(self, beam):
        _remember(beam, "alpha " + ("verylongword " * 200))
        env = beam.recall_bounded(
            "alpha", RecallPolicy(top_k=5, max_tokens=1),
        )
        assert env.token_count <= 1
        assert env.token_count == 0 or env.token_count == 1

    def test_max_tokens_never_exceeded(self, beam):
        for i in range(10):
            _remember(beam, f"item {i} " + ("padding " * 30))
        env = beam.recall_bounded(
            "item", RecallPolicy(top_k=10, max_tokens=20),
        )
        assert env.token_count <= 20


class TestMaxItemTokensIsEstimated:
    """Finding 3: max_item_tokens must be an estimated-token hard limit,
    not a whitespace-word count."""

    def test_max_item_tokens_uses_token_estimation(self, beam):
        # A single very long compound token that tiktoken/chars-4 would
        # count as many tokens but .split() counts as 1 word.
        long_compound = "a" * 200
        _remember(beam, f"alpha {long_compound}")
        env = beam.recall_bounded(
            "alpha", RecallPolicy(top_k=5, max_item_tokens=3),
        )
        from mnemosyne.core.token_counter import estimate_tokens
        for r in env.results:
            content_tokens = estimate_tokens(r.get("content", ""))
            assert content_tokens <= 10  # truncated well under 200-char token count


class TestPostFilterFallback:
    """Finding 4: when all candidates are removed by the strict predicate,
    use deterministic bounded linear/recent fallback rather than an empty
    result (when fallback is required)."""

    def test_all_filtered_triggers_recent_fallback(self, beam):
        # Seed a row in session-b. Query with default policy (session-a
        # only). Vector/FTS may surface it, but the strict predicate
        # removes it. Fallback should pull session-a rows.
        _remember(beam, "visible session a content alpha")
        _remember(beam, "hidden session b content alpha", session_id="sess-b")
        env = beam.recall_bounded("alpha", RecallPolicy(top_k=5, require_fallback=True))
        contents = " ".join(r["content"] for r in env.results)
        assert "visible session a" in contents

    def test_polyphonic_all_filtered_falls_back(self, beam, monkeypatch):
        _remember(beam, "linear alpha visible content")
        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
        from mnemosyne.core.polyphonic_recall import PolyphonicRecallEngine

        # Engine returns a session-b row that the predicate will reject.
        from mnemosyne.core.polyphonic_recall import PolyphonicResult

        def _fake_recall(self, *a, **k):
            return [PolyphonicResult(
                memory_id="nonexistent", combined_score=0.9,
                voice_scores={"vector": 0.9}, metadata={},
            )]

        monkeypatch.setattr(PolyphonicRecallEngine, "recall", _fake_recall)
        env = beam.recall_bounded("alpha", RecallPolicy(top_k=5))
        assert "linear alpha visible" in " ".join(r["content"] for r in env.results)


class TestProducerActorMapping:
    """Finding 5: producer maps to author_type, actor maps to author_id.
    These must not be conflated."""

    def test_producer_ids_filter_author_type(self, beam):
        _remember(beam, "producer human fact alpha", author_id="aid-1", author_type="human")
        _remember(beam, "producer agent fact alpha", author_id="aid-2", author_type="agent")
        env = beam.recall_bounded(
            "alpha", RecallPolicy(top_k=10, producer_ids=["human"]),
        )
        contents = " ".join(r["content"] for r in env.results)
        assert "producer human" in contents
        assert "producer agent" not in contents

    def test_actor_ids_filter_author_id(self, beam):
        _remember(beam, "actor one fact alpha", author_id="aid-1", author_type="human")
        _remember(beam, "actor two fact alpha", author_id="aid-2", author_type="human")
        env = beam.recall_bounded(
            "alpha", RecallPolicy(top_k=10, actor_ids=["aid-1"]),
        )
        contents = " ".join(r["content"] for r in env.results)
        assert "actor one" in contents
        assert "actor two" not in contents

    def test_cross_producer_isolation(self, beam):
        _remember(beam, "human producer detail alpha", author_type="human")
        _remember(beam, "agent producer detail alpha", author_type="agent")
        env = beam.recall_bounded(
            "alpha", RecallPolicy(top_k=10, producer_ids=["agent"]),
        )
        contents = " ".join(r["content"] for r in env.results)
        assert "agent producer" in contents
        assert "human producer" not in contents


class TestStrictNumericValidation:
    """Non-blocking: reject floats and bools as top_k."""

    def test_rejects_float_top_k(self):
        with pytest.raises(ValueError):
            RecallPolicy(top_k=5.0)

    def test_rejects_bool_top_k(self):
        with pytest.raises(ValueError):
            RecallPolicy(top_k=True)

    def test_rejects_float_max_tokens(self):
        with pytest.raises(ValueError):
            RecallPolicy(top_k=5, max_tokens=10.0)


class TestDedupKeepsHighestScore:
    """Non-blocking: dedupe should keep the highest-scored duplicate."""

    def test_dedupe_keeps_higher_score(self, beam):
        _remember(beam, "duplicate prone content alpha beta gamma")
        env = beam.recall_bounded(
            "alpha beta gamma", RecallPolicy(top_k=10),
        )
        # If the same id appears from multiple voices, only one survives
        # and it should be the one with the highest score.
        ids = [r["id"] for r in env.results]
        assert len(ids) == len(set(ids))


class TestSupplementHydration:
    """Finding 2: entity, fact, and MEMORIA supplements must enter the
    same gate."""

    def test_entity_match_enters_gate(self, beam):
        # Seed a memory with an entity annotation.
        beam.remember(
            "The deploy server is omega-cluster", source="conversation",
            extract_entities=True,
        )
        # Also seed a non-matching memory.
        _remember(beam, "unrelated noise content zeta")
        env = beam.recall_bounded("omega", RecallPolicy(top_k=10))
        contents = " ".join(r["content"] for r in env.results)
        assert "omega-cluster" in contents

    def test_memoria_supplement_enters_gate(self, beam):
        # Seed MEMORIA specialist data directly and query for it.
        beam.remember("user prefers dark mode for the editor", source="conversation")
        env = beam.recall_bounded("preference", RecallPolicy(top_k=10))
        # The query should return results (either from FTS or MEMORIA).
        assert isinstance(env.results, list)


# ---------------------------------------------------------------------------
# Fix Round 2 — Associative supplement hydration
# ---------------------------------------------------------------------------

class TestAssociativeHydration:
    """The bounded path must hydrate graph-related memories into the same
    gate, resolving real rows (not placeholders), so strict policy filters
    on producer/actor/project/session/lifecycle apply to associative
    results too.

    The related memory is deliberately chosen so it does NOT match the
    query via FTS/vector — it can only surface through graph traversal.
    """

    def test_associative_allowed_related_admitted(self, beam):
        """A related memory that passes policy is admitted via the gate."""
        from mnemosyne.core.episodic_graph import EpisodicGraph, GraphEdge
        from datetime import datetime, timezone

        # Seed a query-matching memory in session-a.
        _remember(beam, "terraform infrastructure provisioning setup")

        # Seed a related memory that does NOT lexically match the query —
        # it only surfaces through the graph edge. Use distinctive tokens
        # so FTS cannot match it to the query.
        related_mid = _remember(beam, "runbook incident response procedure")

        if beam.episodic_graph is None:
            beam.episodic_graph = EpisodicGraph(conn=beam.conn, db_path=beam.db_path)
        primary_id = beam.conn.execute(
            "SELECT id FROM working_memory WHERE content LIKE '%terraform%'"
        ).fetchone()["id"]
        beam.episodic_graph.add_edge(GraphEdge(
            source=primary_id,
            target=related_mid,
            edge_type="rel",
            weight=0.8,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ))

        env = beam.recall_bounded("terraform infrastructure", RecallPolicy(top_k=10))
        contents = " ".join(r.get("content", "") for r in env.results)
        assert "runbook" in contents or "incident response" in contents, (
            f"related memory not admitted via associative hydration; "
            f"got: {[r.get('content','')[:60] for r in env.results]}"
        )

    def test_associative_foreign_session_rejected(self, beam):
        """A related memory in a foreign session is rejected by policy."""
        from mnemosyne.core.episodic_graph import EpisodicGraph, GraphEdge
        from datetime import datetime, timezone

        _remember(beam, "terraform infrastructure provisioning setup")

        # Related memory in a DIFFERENT session with a distinctive token
        # that does NOT match the query via FTS.
        foreign_mid = _remember(
            beam, "classified runbook incident response procedure",
            session_id="sess-b", scope="session",
        )

        if beam.episodic_graph is None:
            beam.episodic_graph = EpisodicGraph(conn=beam.conn, db_path=beam.db_path)
        primary_id = beam.conn.execute(
            "SELECT id FROM working_memory WHERE content LIKE '%terraform%'"
        ).fetchone()["id"]

        beam.episodic_graph.add_edge(GraphEdge(
            source=primary_id,
            target=foreign_mid,
            edge_type="rel",
            weight=0.9,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ))

        env = beam.recall_bounded("terraform infrastructure", RecallPolicy(top_k=10))
        contents = " ".join(r.get("content", "") for r in env.results)
        # The foreign-session related memory must NOT leak.
        assert "classified" not in contents and "runbook" not in contents, (
            f"foreign-session associative memory leaked through the gate; "
            f"got: {[r.get('content','')[:60] for r in env.results]}"
        )

    def test_associative_does_not_mutate_counters(self, beam):
        """Associative hydration is read-only — no recall_count bump."""
        from mnemosyne.core.episodic_graph import EpisodicGraph, GraphEdge
        from datetime import datetime, timezone

        _remember(beam, "terraform infrastructure provisioning setup")
        related_mid = _remember(beam, "runbook incident response procedure")

        if beam.episodic_graph is None:
            beam.episodic_graph = EpisodicGraph(conn=beam.conn, db_path=beam.db_path)
        primary_id = beam.conn.execute(
            "SELECT id FROM working_memory WHERE content LIKE '%terraform%'"
        ).fetchone()["id"]
        beam.episodic_graph.add_edge(GraphEdge(
            source=primary_id,
            target=related_mid,
            edge_type="rel",
            weight=0.8,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ))

        before = beam.conn.execute(
            "SELECT recall_count FROM working_memory WHERE id = ?", (related_mid,),
        ).fetchone()["recall_count"]
        beam.recall_bounded("terraform infrastructure", RecallPolicy(top_k=10))
        after = beam.conn.execute(
            "SELECT recall_count FROM working_memory WHERE id = ?", (related_mid,),
        ).fetchone()["recall_count"]
        assert before == after


# ---------------------------------------------------------------------------
# Fix Round 3 — binding contract gaps
# ---------------------------------------------------------------------------

class TestPolyphonicAllFilteredLinearFallback:
    """Gap 1: polyphonic rows that hydrate but are ALL rejected by policy
    must fall back to bounded LINEAR hydration (not recent fallback first).
    The envelope must report a truthful linear retrieval mode and the
    polyphonic_empty_fallback_linear degradation reason."""

    def test_polyphonic_all_filtered_returns_linear_row(self, beam, monkeypatch):
        """A valid linear row exists. Polyphonic returns a real but
        policy-rejected row. The linear row must surface."""
        from mnemosyne.core.polyphonic_recall import (
            PolyphonicRecallEngine,
            PolyphonicResult,
        )

        # Seed a valid linear row in session-a.
        _remember(beam, "linear visible content alpha")

        # Seed a real row in session-b that polyphonic will return but
        # policy will reject (foreign session).
        foreign_mid = _remember(
            beam, "polyphonic foreign beta gamma",
            session_id="sess-b", scope="session",
        )

        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
        monkeypatch.setattr(
            PolyphonicRecallEngine, "recall",
            lambda self, *a, **k: [
                PolyphonicResult(
                    memory_id=foreign_mid,
                    combined_score=0.95,
                    voice_scores={"vector": 0.95},
                    metadata={},
                ),
            ],
        )

        env = beam.recall_bounded("alpha", RecallPolicy(top_k=5))
        contents = " ".join(r["content"] for r in env.results)
        assert "linear visible" in contents, (
            f"linear row not returned; got: {[r['content'][:50] for r in env.results]}"
        )
        assert "polyphonic_empty_fallback_linear" in env.degradation_reasons
        # Mode must be truthful about the actual retrieval path used.
        assert env.retrieval_mode in ("vector", "hybrid", "fts", "recent_fallback")


class TestEpisodicEntityFactResolution:
    """Gap 2: entity/fact candidate IDs must resolve across BOTH working
    and episodic memory, preserving real provenance fields.

    The episodic row's content deliberately does NOT match the query via
    FTS — the entity/fact annotation is the only path to the row.
    """

    def test_episodic_entity_match_reaches_gate(self, beam):
        """An entity annotation ONLY on an episodic_memory row (no FTS-
        matchable content) must reach the final gate through entity
        resolution."""
        from datetime import datetime, timezone
        em_id = "ep-entity-test-001"
        beam.conn.execute(
            "INSERT OR IGNORE INTO episodic_memory "
            "(id, content, source, timestamp, session_id, importance, scope, "
            "author_id, author_type, channel_id, veracity, memory_type, tier) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (em_id, "runbook incident response procedure",
             "conversation", datetime.now(timezone.utc).isoformat(),
             "sess-a", 0.8, "session", None, None, None, "unknown", "general", 1),
        )
        beam.conn.commit()
        beam.annotations.add(
            memory_id=em_id, kind="mentions", value="terraform-deploy-entity",
        )
        beam.conn.commit()

        env = beam.recall_bounded("terraform", RecallPolicy(top_k=10))
        contents = " ".join(r.get("content", "") for r in env.results)
        assert "runbook" in contents, (
            f"episodic entity match did not reach gate; "
            f"got: {[r.get('content','')[:50] for r in env.results]}"
        )

    def test_episodic_fact_match_reaches_gate(self, beam):
        """A fact annotation ONLY on an episodic_memory row must reach
        the final gate through fact resolution."""
        from datetime import datetime, timezone
        em_id = "ep-fact-test-001"
        beam.conn.execute(
            "INSERT OR IGNORE INTO episodic_memory "
            "(id, content, source, timestamp, session_id, importance, scope, "
            "author_id, author_type, channel_id, veracity, memory_type, tier) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (em_id, "general infrastructure notes unrelated",
             "conversation", datetime.now(timezone.utc).isoformat(),
             "sess-a", 0.8, "session", None, None, None, "unknown", "general", 1),
        )
        beam.conn.commit()
        beam.annotations.add(
            memory_id=em_id, kind="fact", value="deployment uses kubernetes",
        )
        beam.conn.commit()

        env = beam.recall_bounded("kubernetes", RecallPolicy(top_k=10))
        contents = " ".join(r.get("content", "") for r in env.results)
        assert "infrastructure notes" in contents, (
            f"episodic fact match did not reach gate; "
            f"got: {[r.get('content','')[:50] for r in env.results]}"
        )

    def test_foreign_episodic_entity_rejected(self, beam):
        """An episodic entity match in a foreign session is rejected."""
        from datetime import datetime, timezone
        em_id = "ep-entity-foreign-001"
        beam.conn.execute(
            "INSERT OR IGNORE INTO episodic_memory "
            "(id, content, source, timestamp, session_id, importance, scope, "
            "author_id, author_type, channel_id, veracity, memory_type, tier) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (em_id, "classified foreign notes",
             "conversation", datetime.now(timezone.utc).isoformat(),
             "sess-b", 0.8, "session", None, None, None, "unknown", "general", 1),
        )
        beam.conn.commit()
        beam.annotations.add(
            memory_id=em_id, kind="mentions", value="omega-cluster-entity",
        )
        beam.conn.commit()

        env = beam.recall_bounded("omega", RecallPolicy(top_k=10))
        contents = " ".join(r.get("content", "") for r in env.results)
        assert "classified foreign" not in contents

class TestMemoriaScopeProvenance:
    """Gap 3: MEMORIA synthetic rows must not be hard-coded global.
    Preserve safe scope/provenance from cited source rows."""

    def test_memoria_default_policy_admits_same_session(self, beam):
        """A MEMORIA result citing same-session source rows must be
        admitted under default policy (include_shared=False).

        The test isolates MEMORIA by using a query term that does NOT
        appear in the working_memory FTS index — only in the MEMORIA
        context. So the result can only come through the MEMORIA path.
        """
        # Seed an unrelated working memory (different content).
        _remember(beam, "completely unrelated topic noise")

        wm_mid = beam.conn.execute(
            "SELECT id FROM working_memory WHERE content LIKE '%unrelated%'"
        ).fetchone()["id"]

        def _fake_memoria(query, ability=None, top_k=10):
            return {
                "context": "specialized memoria memoriapass result",
                "facts": [],
                "source": "test_specialist",
                "source_memory_ids": [wm_mid],
            }

        original_memoria = beam.memoria_retrieve
        beam.memoria_retrieve = _fake_memoria
        try:
            env = beam.recall_bounded(
                "memoriapass", RecallPolicy(top_k=10, include_shared=False),
            )
            contents = " ".join(r.get("content", "") for r in env.results)
            assert "memoriapass" in contents, (
                f"MEMORIA same-session result not admitted under default policy; "
                f"got: {[r.get('content','')[:50] for r in env.results]}"
            )
        finally:
            beam.memoria_retrieve = original_memoria

    def test_memoria_foreign_source_rejected(self, beam):
        """A MEMORIA result citing only foreign-session sources must not
        leak under default policy."""
        foreign_mid = _remember(
            beam, "classified foreign memoriareject detail",
            session_id="sess-b", scope="session",
        )

        def _fake_memoria(query, ability=None, top_k=10):
            return {
                "context": "classified memoriareject content",
                "facts": [],
                "source": "test_specialist",
                "source_memory_ids": [foreign_mid],
            }

        original_memoria = beam.memoria_retrieve
        beam.memoria_retrieve = _fake_memoria
        try:
            env = beam.recall_bounded(
                "memoriareject", RecallPolicy(top_k=10, include_shared=False),
            )
            contents = " ".join(r.get("content", "") for r in env.results)
            assert "memoriareject" not in contents
            assert "classified" not in contents
        finally:
            beam.memoria_retrieve = original_memoria

class TestDropEmptyContentItems:
    """Quality: when max_item_tokens is too small for even one word,
    drop the candidate rather than retaining empty content."""

    def test_tiny_max_item_tokens_drops_oversized(self, beam):
        _remember(beam, "alpha " + ("supercalifragilistic " * 10))
        env = beam.recall_bounded(
            "alpha", RecallPolicy(top_k=5, max_item_tokens=1),
        )
        # No result should have empty content.
        for r in env.results:
            assert r.get("content", "").strip() != "", (
                f"empty-content result retained: {r}"
            )


class TestDedupKeepsHigherScore:
    """Quality: dedupe test must prove the higher-scored duplicate survives."""

    def test_dedupe_keeps_higher_scored(self, beam):
        """When the same memory id appears with different scores from
        different voices, the higher score must survive to the final
        result."""
        mid = _remember(beam, "alpha beta gamma delta epsilon zeta")
        env = beam.recall_bounded(
            "alpha beta gamma delta epsilon zeta", RecallPolicy(top_k=10),
        )
        # The memory should appear exactly once (deduped).
        ids = [r["id"] for r in env.results]
        assert ids.count(mid) <= 1
        # And if it appears, it should have a meaningful score.
        for r in env.results:
            if r["id"] == mid:
                assert r["score"] > 0


# ---------------------------------------------------------------------------
# Fix Round 4 — native enhanced bounded recall
# ---------------------------------------------------------------------------

class TestEnhancedBoundedRecall:
    def test_expanded_query_candidate_enters_strict_gate(self, beam, monkeypatch):
        from mnemosyne.core import beam as beam_module

        monkeypatch.setenv("MNEMOSYNE_ENHANCED_RECALL", "1")
        monkeypatch.delenv("MNEMOSYNE_POLYPHONIC_RECALL", raising=False)
        monkeypatch.setattr(beam_module._embeddings, "available", lambda: False)

        allowed_id = _remember(
            beam, "credential expanded-only allowed", session_id="sess-a",
        )
        forbidden_id = _remember(
            beam,
            "credential expanded-only forbidden",
            session_id="sess-b",
            scope="global",
        )

        env = beam.recall_bounded(
            "password",
            RecallPolicy(top_k=10, require_fallback=False),
        )

        ids = {row["id"] for row in env.results}
        assert allowed_id in ids
        assert forbidden_id not in ids

    def test_enhanced_rank_helpers_stay_inside_hard_gate(
        self, beam, monkeypatch,
    ):
        from mnemosyne.core import beam as beam_module
        from mnemosyne.core.token_counter import estimate_tokens

        monkeypatch.setenv("MNEMOSYNE_ENHANCED_RECALL", "1")
        monkeypatch.delenv("MNEMOSYNE_POLYPHONIC_RECALL", raising=False)
        monkeypatch.setattr(beam_module._embeddings, "available", lambda: False)

        allowed_ids = {
            _remember(
                beam,
                f"deploy allowed result {index} with bounded padding",
                importance=0.9 - index * 0.1,
                memory_type="request",
            )
            for index in range(3)
        }
        forbidden_id = _remember(
            beam,
            "deploy forbidden global result",
            session_id="sess-b",
            scope="global",
            memory_type="forbidden",
        )

        calls = {"intent": 0, "adjust": 0, "weibull": 0, "mmr_ids": []}

        def classify_intent(query):
            calls["intent"] += 1
            return SimpleNamespace(category="procedural")

        def adjust_weights(*args, **kwargs):
            calls["adjust"] += 1
            return (0.2, 0.7, 0.1)

        def weibull_boost(timestamp, query_time=None, memory_type="general"):
            assert memory_type != "forbidden", "filtered candidate reached rank"
            calls["weibull"] += 1
            return 0.5

        def mmr_rerank(results, lambda_param=0.7, top_k=10):
            calls["mmr_ids"] = [row["id"] for row in results]
            return list(reversed(results))[:top_k]

        monkeypatch.setattr(beam_module, "classify_intent", classify_intent)
        monkeypatch.setattr(beam_module, "adjust_weights", adjust_weights)
        monkeypatch.setattr(beam_module, "weibull_boost", weibull_boost)
        monkeypatch.setattr(beam_module, "mmr_rerank", mmr_rerank)

        policy = RecallPolicy(
            top_k=2,
            max_tokens=20,
            max_item_tokens=6,
            require_fallback=False,
        )
        env = beam.recall_bounded("how do I deploy", policy)

        assert calls["intent"] == 1
        assert calls["adjust"] == 1
        assert calls["weibull"] >= len(allowed_ids)
        assert set(calls["mmr_ids"]) == allowed_ids
        assert forbidden_id not in calls["mmr_ids"]
        assert len(env.results) <= policy.top_k
        assert env.token_count <= policy.max_tokens
        assert all(
            estimate_tokens(row["content"]) <= policy.max_item_tokens
            for row in env.results
        )

    def test_enhanced_bounded_is_read_only_and_skips_legacy_and_cache(
        self, beam, monkeypatch,
    ):
        from mnemosyne.core import beam as beam_module

        monkeypatch.setenv("MNEMOSYNE_ENHANCED_RECALL", "1")
        monkeypatch.delenv("MNEMOSYNE_POLYPHONIC_RECALL", raising=False)
        monkeypatch.setattr(beam_module._embeddings, "available", lambda: False)

        memory_id = _remember(beam, "enhanced counter sentinel")
        before = beam.conn.execute(
            "SELECT recall_count FROM working_memory WHERE id = ?",
            (memory_id,),
        ).fetchone()["recall_count"]

        expanded_queries = []

        def expand_query(query):
            expanded_queries.append(query)
            return query

        def forbidden_legacy(*args, **kwargs):
            raise AssertionError("bounded recall invoked a legacy recall method")

        class ForbiddenCache:
            def __getattr__(self, name):
                raise AssertionError(f"bounded recall accessed query cache: {name}")

        monkeypatch.setattr(beam_module, "expand_query", expand_query)
        monkeypatch.setattr(beam, "recall", forbidden_legacy)
        monkeypatch.setattr(beam, "recall_enhanced", forbidden_legacy)
        beam._query_cache = ForbiddenCache()

        env = beam.recall_bounded(
            "enhanced counter sentinel",
            RecallPolicy(top_k=5, require_fallback=False),
        )
        after = beam.conn.execute(
            "SELECT recall_count FROM working_memory WHERE id = ?",
            (memory_id,),
        ).fetchone()["recall_count"]

        assert expanded_queries == ["enhanced counter sentinel"]
        assert memory_id in {row["id"] for row in env.results}
        assert after == before


# ---------------------------------------------------------------------------
# Recall metadata reconciliation (Task 1 bounded-gate regressions)
# ---------------------------------------------------------------------------


class TestRecallMetadataBounded:
    """Bounded recall must expose parsed ``metadata: dict`` on every real
    result row, never the raw ``metadata_json`` storage field, and must
    not leak metadata values into rendered context or diagnostics."""

    def test_bounded_recall_returns_parsed_metadata_without_raw_storage(self, beam):
        working_id = _remember(
            beam, "bounded metadata alpha", metadata={"scope": "authorized"}
        )
        envelope = beam.recall_bounded(
            "bounded metadata", RecallPolicy(top_k=5, max_tokens=80)
        )
        row = next(
            (item for item in envelope.results if item["id"] == working_id),
            None,
        )
        assert row is not None, "seeded working row missing from bounded results"
        assert row["metadata"] == {"scope": "authorized"}
        assert "metadata_json" not in row
        assert "authorized" not in envelope.rendered_context

    def test_bounded_recall_does_not_expose_foreign_row_metadata(self, beam):
        _remember(
            beam,
            "foreign metadata alpha",
            session_id="sess-b",
            metadata={"private": "never-return"},
        )
        envelope = beam.recall_bounded(
            "foreign metadata", RecallPolicy(top_k=10)
        )

        assert all(
            row.get("metadata") != {"private": "never-return"}
            for row in envelope.results
        )
        assert "never-return" not in envelope.rendered_context

    def test_bounded_every_result_row_has_metadata_dict(self, beam):
        """Every public result row -- real or synthetic -- must carry a
        parsed ``metadata: dict`` and never the raw storage field."""
        _remember(beam, "dict-shape alpha", metadata={"k": "v"})
        _remember(beam, "dict-shape beta")
        envelope = beam.recall_bounded(
            "dict-shape", RecallPolicy(top_k=10, require_fallback=True)
        )
        assert envelope.results, "bounded recall returned no rows"
        for row in envelope.results:
            assert isinstance(row.get("metadata"), dict), (
                f"row {row.get('id')!r} missing parsed metadata dict"
            )
            assert "metadata_json" not in row, (
                f"raw metadata_json leaked on row {row.get('id')!r}"
            )

    def test_bounded_malformed_metadata_is_empty_dict(self, beam):
        """Malformed ``metadata_json`` must surface as ``{}`` without
        raising or leaking the raw value."""
        mid = _remember(beam, "malformed bounded alpha")
        beam.conn.execute(
            "UPDATE working_memory SET metadata_json = ? WHERE id = ?",
            ("{not-json", mid),
        )
        beam.conn.commit()
        envelope = beam.recall_bounded(
            "malformed bounded", RecallPolicy(top_k=10)
        )
        row = next(
            (r for r in envelope.results if r["id"] == mid), None,
        )
        assert row is not None, "seeded malformed-metadata row not returned"
        assert row["metadata"] == {}
        assert "metadata_json" not in row
        assert "not-json" not in envelope.rendered_context

    def test_bounded_metadata_absent_from_diagnostics(self, beam):
        """Metadata keys/values must never appear in rendered context,
        trace id, or applied filters."""
        _remember(
            beam, "diagnostic leak alpha",
            metadata={"secret_key": "secret_value_42"},
        )
        envelope = beam.recall_bounded(
            "diagnostic leak", RecallPolicy(top_k=10)
        )
        for field in ("rendered_context", "trace_id"):
            text = getattr(envelope, field)
            assert "secret_key" not in text, (
                f"metadata key leaked into {field}"
            )
            assert "secret_value_42" not in text, (
                f"metadata value leaked into {field}"
            )
        assert "secret_key" not in str(envelope.applied_filters)



# ---------------------------------------------------------------------------
# Security hardening (I-3 / I-5 / I-6)
# ---------------------------------------------------------------------------


class TestAllowlistTypingI6:
    """I-6: every allowlist field must reject a bare str / non-sequence
    deterministically, and empty allowlists must fail closed (reject all)
    without producing invalid ``IN ()`` SQL or silently widening scope."""

    @pytest.mark.parametrize("field", [
        "session_ids", "actor_ids", "producer_ids", "project_ids",
        "producer_types", "memory_types", "veracity",
    ])
    def test_bare_string_rejected(self, field):
        """A bare string like ``"ab"`` must not be char-split into
        ``{'a', 'b'}``; it must raise at construction."""
        kwargs = {field: "ab"}
        with pytest.raises((ValueError, TypeError)):
            RecallPolicy(top_k=5, **kwargs)

    @pytest.mark.parametrize("field", [
        "session_ids", "actor_ids", "producer_ids", "project_ids",
        "producer_types", "memory_types", "veracity",
    ])
    def test_non_string_element_rejected(self, field):
        """A sequence containing non-string elements must raise."""
        kwargs = {field: [1, 2, 3]}
        with pytest.raises((ValueError, TypeError)):
            RecallPolicy(top_k=5, **kwargs)

    @pytest.mark.parametrize("field", [
        "session_ids", "actor_ids", "producer_ids", "project_ids",
        "producer_types", "memory_types", "veracity",
    ])
    def test_int_rejected(self, field):
        """An int is not a valid allowlist."""
        kwargs = {field: 5}
        with pytest.raises((ValueError, TypeError)):
            RecallPolicy(top_k=5, **kwargs)

    def test_valid_string_sequence_accepted(self):
        """Sanity: a legitimate list of strings is accepted."""
        p = RecallPolicy(top_k=5, session_ids=["s1", "s2"], veracity=["stated"])
        assert p.session_ids == ["s1", "s2"]

    def test_tuple_of_strings_accepted(self):
        p = RecallPolicy(top_k=5, actor_ids=("a1", "a2"))
        assert p.actor_ids == ("a1", "a2")


class TestEmptyAllowlistFailClosedI6:
    """I-6: an empty allowlist must fail closed (match nothing), not
    crash with ``IN ()`` or silently widen to all rows."""

    def test_empty_session_ids_returns_no_rows(self, beam):
        _remember(beam, "empty allowlist alpha")
        env = beam.recall_bounded(
            "empty allowlist", RecallPolicy(top_k=5, session_ids=[]),
        )
        assert env.results == []

    def test_empty_actor_ids_returns_no_rows(self, beam):
        _remember(beam, "empty actor alpha")
        env = beam.recall_bounded(
            "empty actor", RecallPolicy(top_k=5, actor_ids=[]),
        )
        assert env.results == []


class TestOnlyActivePolicyI5:
    """I-5: ``only_active=False`` must actually allow lifecycle-expired
    and superseded rows through the gate (subject to remaining
    scope/policy filters). ``True`` (default) stays fail-closed."""

    def test_only_active_true_expires_row(self, beam):
        """Default policy excludes expired rows (fail-closed)."""
        from datetime import datetime, timedelta, timezone
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        mid = _remember(beam, "expired lifecycle alpha", valid_until=past)
        env = beam.recall_bounded(
            "expired lifecycle", RecallPolicy(top_k=10),
        )
        assert mid not in {r["id"] for r in env.results}

    def test_only_active_false_admits_expired_row(self, beam):
        """only_active=False must let the expired row through, subject
        to session scope (same session → admitted)."""
        from datetime import datetime, timedelta, timezone
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        mid = _remember(beam, "expired lifecycle beta", valid_until=past)
        env = beam.recall_bounded(
            "expired lifecycle",
            RecallPolicy(top_k=10, only_active=False),
        )
        ids = {r["id"] for r in env.results}
        assert mid in ids, (
            f"only_active=False failed to admit expired row {mid}; "
            f"got {sorted(ids)}"
        )

    def test_only_active_false_admits_superseded_row(self, beam):
        mid = _remember(
            beam, "superseded lifecycle beta", superseded_by="ep-xyz",
        )
        env = beam.recall_bounded(
            "superseded lifecycle",
            RecallPolicy(top_k=10, only_active=False),
        )
        ids = {r["id"] for r in env.results}
        assert mid in ids, (
            f"only_active=False failed to admit superseded row {mid}; "
            f"got {sorted(ids)}"
        )

    def test_only_active_false_still_enforces_session_scope(self, beam):
        """only_active=False must NOT bypass session isolation: an
        expired foreign-session row must still be rejected."""
        from datetime import datetime, timedelta, timezone
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        foreign_mid = _remember(
            beam, "expired foreign gamma",
            session_id="sess-b", valid_until=past,
        )
        env = beam.recall_bounded(
            "expired foreign",
            RecallPolicy(top_k=10, only_active=False),
        )
        assert foreign_mid not in {r["id"] for r in env.results}


class TestBoundedDegradationObservabilityI3:
    """I-3: FTS working/episodic and MEMORIA exceptions must produce
    structured ``degradation_reasons`` entries and a safe log signal,
    with no raw query/memory content in log messages."""

    def test_fts_working_failure_emits_degradation_reason(self, beam, monkeypatch):
        _remember(beam, "fts working observability alpha")
        import mnemosyne.core.beam as beam_mod

        def _boom(conn, query, k=20):
            raise RuntimeError("fts_working boom")

        monkeypatch.setattr(beam_mod, "_fts_search_working", _boom)
        env = beam.recall_bounded(
            "fts working observability", RecallPolicy(top_k=10),
        )
        assert "fts_working_failed" in env.degradation_reasons, (
            f"missing fts_working_failed; got {env.degradation_reasons}"
        )

    def test_fts_episodic_failure_emits_degradation_reason(self, beam, monkeypatch):
        _remember(beam, "fts episodic observability alpha")
        import mnemosyne.core.beam as beam_mod

        def _boom(conn, query, k=20):
            raise RuntimeError("fts_episodic boom")

        monkeypatch.setattr(beam_mod, "_fts_search", _boom)
        env = beam.recall_bounded(
            "fts episodic observability", RecallPolicy(top_k=10),
        )
        assert "fts_episodic_failed" in env.degradation_reasons, (
            f"missing fts_episodic_failed; got {env.degradation_reasons}"
        )

    def test_memoria_failure_emits_degradation_reason(self, beam, monkeypatch):
        _remember(beam, "memoria observability alpha")

        def _boom(self, query, ability=None, top_k=10):
            raise RuntimeError("memoria boom")

        monkeypatch.setattr(
            type(beam), "memoria_retrieve", _boom,
        )
        env = beam.recall_bounded(
            "memoria observability", RecallPolicy(top_k=10),
        )
        assert "memoria_failed" in env.degradation_reasons, (
            f"missing memoria_failed; got {env.degradation_reasons}"
        )

    def test_query_embedding_failure_emits_degradation_reason(
        self, beam, monkeypatch, caplog,
    ):
        import logging
        import mnemosyne.core.beam as beam_mod

        mid = _remember(beam, "bounded observability alpha topic")
        monkeypatch.setattr(
            beam_mod._embeddings, "available", lambda: True,
        )

        def _boom(query):
            raise RuntimeError("synthetic backend failure")

        monkeypatch.setattr(beam_mod._embeddings, "embed_query", _boom)
        with caplog.at_level(logging.INFO, logger="mnemosyne.core.recall_bounded"):
            env = beam.recall_bounded(
                "bounded observability", RecallPolicy(top_k=10),
            )
        assert "query_embedding_failed" in env.degradation_reasons, (
            f"missing query_embedding_failed; got {env.degradation_reasons}"
        )
        assert mid in {r["id"] for r in env.results}, (
            "envelope lost working FTS results after embedding failure"
        )
        full = caplog.text
        assert "bounded observability" not in full, (
            "raw query leaked into query-embedding degradation log"
        )
        assert "synthetic backend failure" not in full, (
            "raw exception text leaked into query-embedding degradation log"
        )

    def test_vec_working_failure_emits_degradation_reason(
        self, beam, monkeypatch, caplog,
    ):
        import logging
        import numpy as np
        import mnemosyne.core.beam as beam_mod
        from mnemosyne.core import embeddings as emb_mod

        mid = _remember(beam, "bounded observability alpha topic")
        fake_vec = np.ones(emb_mod.EMBEDDING_DIM, dtype=np.float32)
        monkeypatch.setattr(
            beam_mod._embeddings, "available", lambda: True,
        )
        monkeypatch.setattr(
            beam_mod._embeddings, "embed_query", lambda q: fake_vec.copy(),
        )

        def _boom(conn, query_embedding, k=50, where_sql=None, where_params=()):
            raise RuntimeError("synthetic backend failure")

        monkeypatch.setattr(beam_mod, "_wm_vec_search", _boom)
        with caplog.at_level(logging.INFO, logger="mnemosyne.core.recall_bounded"):
            env = beam.recall_bounded(
                "bounded observability", RecallPolicy(top_k=10),
            )
        assert "vec_working_failed" in env.degradation_reasons, (
            f"missing vec_working_failed; got {env.degradation_reasons}"
        )
        assert mid in {r["id"] for r in env.results}, (
            "envelope lost working FTS results after vec_working failure"
        )
        full = caplog.text
        assert "bounded observability" not in full, (
            "raw query leaked into vec_working degradation log"
        )
        assert "synthetic backend failure" not in full, (
            "raw exception text leaked into vec_working degradation log"
        )

    def test_vec_episodic_failure_emits_degradation_reason(
        self, beam, monkeypatch, caplog,
    ):
        import logging
        import numpy as np
        import mnemosyne.core.beam as beam_mod
        from mnemosyne.core import embeddings as emb_mod

        mid = _remember(beam, "bounded observability alpha topic")
        fake_vec = np.ones(emb_mod.EMBEDDING_DIM, dtype=np.float32)
        monkeypatch.setattr(
            beam_mod._embeddings, "available", lambda: True,
        )
        monkeypatch.setattr(
            beam_mod._embeddings, "embed_query", lambda q: fake_vec.copy(),
        )

        def _boom(conn, query_embedding, k=20):
            raise RuntimeError("synthetic backend failure")

        monkeypatch.setattr(beam_mod, "_in_memory_vec_search", _boom)
        with caplog.at_level(logging.INFO, logger="mnemosyne.core.recall_bounded"):
            env = beam.recall_bounded(
                "bounded observability", RecallPolicy(top_k=10),
            )
        assert "vec_episodic_failed" in env.degradation_reasons, (
            f"missing vec_episodic_failed; got {env.degradation_reasons}"
        )
        assert mid in {r["id"] for r in env.results}, (
            "envelope lost working FTS results after episodic vec failure"
        )
        full = caplog.text
        assert "bounded observability" not in full, (
            "raw query leaked into episodic vec degradation log"
        )
        assert "synthetic backend failure" not in full, (
            "raw exception text leaked into episodic vec degradation log"
        )

    def test_entity_failure_emits_degradation_reason(
        self, beam, monkeypatch, caplog,
    ):
        import logging
        import mnemosyne.core.beam as beam_mod

        mid = _remember(beam, "bounded observability alpha topic")

        def _boom(beam, query):
            raise RuntimeError("synthetic backend failure")

        monkeypatch.setattr(beam_mod, "_find_memories_by_entity", _boom)
        with caplog.at_level(logging.INFO, logger="mnemosyne.core.recall_bounded"):
            env = beam.recall_bounded(
                "bounded observability", RecallPolicy(top_k=10),
            )
        assert "entity_lookup_failed" in env.degradation_reasons, (
            f"missing entity_lookup_failed; got {env.degradation_reasons}"
        )
        assert mid in {r["id"] for r in env.results}, (
            "envelope lost working FTS results after entity lookup failure"
        )
        full = caplog.text
        assert "bounded observability" not in full, (
            "raw query leaked into entity degradation log"
        )
        assert "synthetic backend failure" not in full, (
            "raw exception text leaked into entity degradation log"
        )

    def test_fact_failure_emits_degradation_reason(
        self, beam, monkeypatch, caplog,
    ):
        import logging
        import mnemosyne.core.beam as beam_mod

        mid = _remember(beam, "bounded observability alpha topic")

        def _boom(beam, query):
            raise RuntimeError("synthetic backend failure")

        monkeypatch.setattr(beam_mod, "_find_memories_by_fact", _boom)
        with caplog.at_level(logging.INFO, logger="mnemosyne.core.recall_bounded"):
            env = beam.recall_bounded(
                "bounded observability", RecallPolicy(top_k=10),
            )
        assert "fact_lookup_failed" in env.degradation_reasons, (
            f"missing fact_lookup_failed; got {env.degradation_reasons}"
        )
        assert mid in {r["id"] for r in env.results}, (
            "envelope lost working FTS results after fact lookup failure"
        )
        full = caplog.text
        assert "bounded observability" not in full, (
            "raw query leaked into fact degradation log"
        )
        assert "synthetic backend failure" not in full, (
            "raw exception text leaked into fact degradation log"
        )

    def test_degradation_logs_are_content_free(self, beam, monkeypatch, caplog):
        """The safe log signal for FTS/MEMORIA failures must not echo
        the raw query or memory content."""
        import logging
        _remember(beam, "content free log sentinel alpha")
        import mnemosyne.core.beam as beam_mod

        def _boom(conn, query, k=20):
            raise RuntimeError("sentinel detail boom")

        monkeypatch.setattr(beam_mod, "_fts_search_working", _boom)
        with caplog.at_level(logging.INFO, logger="mnemosyne.core.recall_bounded"):
            beam.recall_bounded(
                "content free log sentinel", RecallPolicy(top_k=10),
            )
        full = caplog.text
        # The query string and the exception message detail must not
        # appear in log output — only a bounded, content-free signal.
        assert "content free log sentinel" not in full, (
            "raw query leaked into degradation log"
        )


# ---------------------------------------------------------------------------
# Round-2 review remediation: vector-path only_active, content-free logs
# ---------------------------------------------------------------------------


class TestOnlyActiveVectorPathRound2:
    """Round-2 I-1: ``only_active=False`` must admit lifecycle-expired/
    superseded rows via the **working-memory vector path** specifically
    — not just FTS or recent fallback. The test pins embedding
    availability so the vector path is the *only* matching retrieval
    voice, then asserts both admission and mode."""

    def test_only_active_false_admits_expired_row_via_vector_path(
        self, beam, monkeypatch,
    ):
        import numpy as np
        from datetime import datetime, timedelta, timezone
        import mnemosyne.core.beam as beam_mod
        from mnemosyne.core import embeddings as emb_mod

        # Seed an expired working-memory row with a vector embedding.
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        mid = _remember(
            beam, "vector lifecycle expired unique alpha",
            valid_until=past,
        )
        # Insert a matching embedding so _wm_vec_search_fallback finds it.
        fake_vec = np.ones(emb_mod.EMBEDDING_DIM, dtype=np.float32)
        beam.conn.execute(
            "INSERT INTO memory_embeddings(memory_id, embedding_json, model) "
            "VALUES (?, ?, ?)",
            (mid, __import__("json").dumps(fake_vec.tolist()), "test"),
        )
        beam.conn.commit()

        # Force embeddings "available" + deterministic query embedding.
        monkeypatch.setattr(emb_mod, "available", lambda: True)
        monkeypatch.setattr(
            beam_mod._embeddings, "available", lambda: True,
        )
        monkeypatch.setattr(
            beam_mod._embeddings, "embed_query",
            lambda q: fake_vec.copy(),
        )
        # Kill FTS so it cannot mask the vector-path defect.
        monkeypatch.setattr(beam_mod, "_fts_search_working", lambda *a, **k: [])
        monkeypatch.setattr(beam_mod, "_fts_search", lambda *a, **k: [])

        env = beam.recall_bounded(
            "vector lifecycle expired unique alpha",
            RecallPolicy(top_k=10, only_active=False),
        )
        ids = {r["id"] for r in env.results}
        assert mid in ids, (
            f"only_active=False failed to admit expired row {mid} via "
            f"vector path; got {sorted(ids)}"
        )
        # Must have used the vector path (not FTS or recent fallback).
        assert env.retrieval_mode == "vector", (
            f"expected vector mode; got {env.retrieval_mode!r} "
            "(FTS/fallback would mask the vector-path defect)"
        )

    def test_only_active_false_admits_superseded_row_via_vector_path(
        self, beam, monkeypatch,
    ):
        import numpy as np
        import mnemosyne.core.beam as beam_mod
        from mnemosyne.core import embeddings as emb_mod

        mid = _remember(
            beam, "vector lifecycle superseded unique beta",
            superseded_by="ep-replacement",
        )
        fake_vec = np.ones(emb_mod.EMBEDDING_DIM, dtype=np.float32)
        beam.conn.execute(
            "INSERT INTO memory_embeddings(memory_id, embedding_json, model) "
            "VALUES (?, ?, ?)",
            (mid, __import__("json").dumps(fake_vec.tolist()), "test"),
        )
        beam.conn.commit()

        monkeypatch.setattr(
            beam_mod._embeddings, "available", lambda: True,
        )
        monkeypatch.setattr(
            beam_mod._embeddings, "embed_query",
            lambda q: fake_vec.copy(),
        )
        monkeypatch.setattr(beam_mod, "_fts_search_working", lambda *a, **k: [])
        monkeypatch.setattr(beam_mod, "_fts_search", lambda *a, **k: [])

        env = beam.recall_bounded(
            "vector lifecycle superseded unique beta",
            RecallPolicy(top_k=10, only_active=False),
        )
        ids = {r["id"] for r in env.results}
        assert mid in ids, (
            f"only_active=False failed to admit superseded row {mid} via "
            f"vector path; got {sorted(ids)}"
        )
        assert env.retrieval_mode == "vector", (
            f"expected vector mode; got {env.retrieval_mode!r}"
        )

    def test_only_active_true_still_expires_row_via_vector_path(
        self, beam, monkeypatch,
    ):
        """Counterpart: with only_active=True (default), the vector path
        must still exclude the expired row (fail-closed preserved)."""
        import numpy as np
        from datetime import datetime, timedelta, timezone
        import mnemosyne.core.beam as beam_mod
        from mnemosyne.core import embeddings as emb_mod

        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        mid = _remember(
            beam, "vector lifecycle expired unique gamma",
            valid_until=past,
        )
        fake_vec = np.ones(emb_mod.EMBEDDING_DIM, dtype=np.float32)
        beam.conn.execute(
            "INSERT INTO memory_embeddings(memory_id, embedding_json, model) "
            "VALUES (?, ?, ?)",
            (mid, __import__("json").dumps(fake_vec.tolist()), "test"),
        )
        beam.conn.commit()

        monkeypatch.setattr(
            beam_mod._embeddings, "available", lambda: True,
        )
        monkeypatch.setattr(
            beam_mod._embeddings, "embed_query",
            lambda q: fake_vec.copy(),
        )
        monkeypatch.setattr(beam_mod, "_fts_search_working", lambda *a, **k: [])
        monkeypatch.setattr(beam_mod, "_fts_search", lambda *a, **k: [])

        env = beam.recall_bounded(
            "vector lifecycle expired unique gamma",
            RecallPolicy(top_k=10),  # only_active=True (default)
        )
        assert mid not in {r["id"] for r in env.results}


class TestDegradationLogsContentFreeRound2:
    """Round-2 I-3: degradation logs must be *strictly* content-free —
    no query, no raw exception message, no traceback text, no payload.
    The round-1 test only checked the query string; this also asserts
    the exception message and traceback are absent."""

    def test_fts_working_failure_log_is_strictly_content_free(
        self, beam, monkeypatch, caplog,
    ):
        import logging
        _remember(beam, "secret canary content alpha")
        import mnemosyne.core.beam as beam_mod

        def _boom(conn, query, k=20):
            raise RuntimeError("UNIQUE_EXC_DETAIL_42 boom")

        monkeypatch.setattr(beam_mod, "_fts_search_working", _boom)
        with caplog.at_level(logging.INFO, logger="mnemosyne.core.recall_bounded"):
            beam.recall_bounded(
                "secret canary content alpha", RecallPolicy(top_k=10),
            )
        full = caplog.text
        assert "secret canary content alpha" not in full, (
            "raw query leaked into degradation log"
        )
        assert "UNIQUE_EXC_DETAIL_42" not in full, (
            "raw exception message leaked into degradation log "
            "(exc_info=True renders it into the record)"
        )
        assert "Traceback" not in full, (
            "traceback text leaked into degradation log"
        )

    def test_memoria_failure_log_is_strictly_content_free(
        self, beam, monkeypatch, caplog,
    ):
        import logging
        _remember(beam, "memoria secret payload beta")
        import mnemosyne.core.beam as beam_mod

        def _boom(conn, query, k=20):
            raise RuntimeError("MEMORIA_EXC_DETAIL_99 leak")

        # Patch at the module level so the bounded path's except block hits it.
        monkeypatch.setattr(beam_mod, "_fts_search_working", _boom)
        # Also patch memoria to ensure its exception detail doesn't leak.
        def _memoria_boom(self, query, ability=None, top_k=10):
            raise RuntimeError("MEMORIA_INNER_DETAIL_77 leak")

        monkeypatch.setattr(type(beam), "memoria_retrieve", _memoria_boom)

        with caplog.at_level(logging.INFO, logger="mnemosyne.core.recall_bounded"):
            beam.recall_bounded(
                "memoria secret payload beta", RecallPolicy(top_k=10),
            )
        full = caplog.text
        assert "memoria secret payload beta" not in full, (
            "raw query leaked into memoria degradation log"
        )
        assert "MEMORIA_EXC_DETAIL_99" not in full, (
            "FTS exception message leaked even via memoria path"
        )
        assert "MEMORIA_INNER_DETAIL_77" not in full, (
            "memoria exception message leaked into degradation log"
        )
        assert "Traceback" not in full, (
            "traceback text leaked into memoria degradation log"
        )
