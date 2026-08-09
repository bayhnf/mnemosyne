"""Task 3 — Authoritative bounded recall (``recall_bounded``).

RED-first tests for the additive post-hydration gate that exposes a
``RecallEnvelope`` with hard result/token caps, strict isolation, and
deterministic fallback, while leaving legacy ``recall()`` untouched.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

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
