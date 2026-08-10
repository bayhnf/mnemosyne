"""Recall metadata contract tests (Task 1).

RED-first characterization of the public ``metadata: dict`` contract
that every authorized legacy recall result must expose. Pairs with the
bounded-gate assertions in ``tests/test_recall_bounded.py``.

Covers:
- persisted metadata round-trips for working + episodic tiers
- malformed / non-object JSON collapses to ``{}`` without raising
- polyphonic + synthetic rows expose a parsed dict, never raw storage
- raw ``metadata_json`` is never present on any public recall row
"""

from __future__ import annotations

from mnemosyne.core.beam import BeamMemory


# ---------------------------------------------------------------------------
# Persisted metadata round-trip
# ---------------------------------------------------------------------------


def test_recall_returns_persisted_metadata_for_both_tiers(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    beam = BeamMemory(db_path=tmp_path / "metadata.db", session_id="s1")
    working_id = beam.remember(
        "working metadata alpha", metadata={"kind": "working"}
    )
    episodic_id = beam.consolidate_to_episodic(
        "episodic metadata alpha", [], metadata={"kind": "episodic"}
    )

    rows = {row["id"]: row for row in beam.recall("metadata alpha", top_k=20)}

    assert working_id in rows, f"working row {working_id} missing from recall"
    assert episodic_id in rows, f"episodic row {episodic_id} missing from recall"
    assert rows[working_id]["metadata"] == {"kind": "working"}
    assert rows[episodic_id]["metadata"] == {"kind": "episodic"}
    assert "metadata_json" not in rows[working_id]
    assert "metadata_json" not in rows[episodic_id]


def test_recall_invalid_or_non_object_metadata_is_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    beam = BeamMemory(db_path=tmp_path / "invalid.db", session_id="s1")
    invalid_id = beam.remember("invalid metadata alpha")
    list_id = beam.remember("list metadata alpha")
    beam.conn.execute(
        "UPDATE working_memory SET metadata_json = ? WHERE id = ?",
        ("{broken", invalid_id),
    )
    beam.conn.execute(
        "UPDATE working_memory SET metadata_json = ? WHERE id = ?",
        ('["not", "an", "object"]', list_id),
    )
    beam.conn.commit()

    rows = {row["id"]: row for row in beam.recall("metadata alpha", top_k=20)}

    assert rows[invalid_id]["metadata"] == {}
    assert rows[list_id]["metadata"] == {}


def test_recall_null_metadata_is_empty(tmp_path, monkeypatch):
    """NULL metadata_json must not raise and must resolve to ``{}``."""
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    beam = BeamMemory(db_path=tmp_path / "null.db", session_id="s1")
    mid = beam.remember("null metadata alpha")
    beam.conn.execute(
        "UPDATE working_memory SET metadata_json = NULL WHERE id = ?", (mid,)
    )
    beam.conn.commit()

    rows = {row["id"]: row for row in beam.recall("metadata alpha", top_k=20)}
    assert rows[mid]["metadata"] == {}


# ---------------------------------------------------------------------------
# Polyphonic + synthetic rows
# ---------------------------------------------------------------------------


def test_polyphonic_rows_expose_parsed_metadata(tmp_path, monkeypatch):
    """A real memory_id returned by the polyphonic engine must expose
    parsed metadata from its underlying storage row, never the raw
    ``metadata_json`` field."""
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    beam = BeamMemory(db_path=tmp_path / "poly.db", session_id="s1")
    memory_id = beam.remember(
        "poly metadata alpha", metadata={"kind": "real"}
    )

    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
    # Force the polyphonic engine to surface only our seeded id. We
    # monkeypatch the engine factory so the real recall() polyphonic
    # branch resolves it through _fetch_polyphonic_row (the production
    # hydration path) without requiring sqlite-vec.
    from mnemosyne.core.polyphonic_recall import PolyphonicResult

    class _StubEngine:
        def __init__(self, *a, **kw):
            pass

        def recall(self, query, query_embedding=None, top_k=40):
            return [
                PolyphonicResult(
                    memory_id=memory_id,
                    combined_score=0.95,
                    voice_scores={"vector": 0.95},
                    metadata={},
                )
            ]

    import mnemosyne.core.beam as beam_mod
    monkeypatch.setattr(
        beam_mod.BeamMemory, "_get_polyphonic_engine", lambda self: _StubEngine()
    )

    results = beam.recall("poly metadata", top_k=20)
    target = next((r for r in results if r["id"] == memory_id), None)
    assert target is not None, "seeded polyphonic row missing from results"
    assert target["metadata"] == {"kind": "real"}
    assert "metadata_json" not in target


def test_fact_recall_synthetic_rows_have_empty_metadata(tmp_path, monkeypatch):
    """Synthetic fact-recall rows (``cf_*`` ids) must expose
    ``metadata == {}`` and never carry the raw storage field."""
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.setenv("MNEMOSYNE_FACT_RECALL_ENABLED", "1")
    beam = BeamMemory(db_path=tmp_path / "facts.db", session_id="s1")
    beam.conn.execute(
        "INSERT INTO facts(fact_id, session_id, subject, predicate, object, "
        "confidence) VALUES (?, ?, ?, ?, ?, ?)",
        ("fact-1", "s1", "alpha", "is", "real", 0.9),
    )
    beam.conn.commit()

    results = beam.recall("alpha", top_k=20)
    synthetic = [r for r in results if str(r.get("id", "")).startswith("cf_")]
    assert synthetic, "no synthetic cf_ row surfaced from fact recall"
    row = synthetic[0]
    assert row["metadata"] == {}
    assert "metadata_json" not in row


def test_memoria_synthetic_row_has_empty_metadata(tmp_path, monkeypatch):
    """MEMORIA synthetic rows must expose ``metadata == {}`` and must
    not leak any seeded secret."""
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    beam = BeamMemory(db_path=tmp_path / "memoria.db", session_id="s1")

    def _fake_memoria(self, query, ability=None, top_k=10):
        return {
            "source": "regex",
            "context": "memoria synthetic alpha",
            "source_memory_ids": [],
        }

    monkeypatch.setattr(beam, "memoria_retrieve", _fake_memoria.__get__(beam))

    results = beam.recall("memoria synthetic alpha", top_k=20)
    memoria_rows = [
        r for r in results if str(r.get("id", "")).startswith("memoria")
    ]
    assert memoria_rows, "no MEMORIA synthetic row surfaced"
    row = memoria_rows[0]
    assert row["metadata"] == {}
    assert "metadata_json" not in row


# ---------------------------------------------------------------------------
# Raw storage field never leaks
# ---------------------------------------------------------------------------


def test_no_recall_row_exposes_raw_metadata_json(tmp_path, monkeypatch):
    """Across all tiers, no public recall row may carry the raw
    ``metadata_json`` storage field, and every row must carry a parsed
    ``metadata: dict``."""
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    beam = BeamMemory(db_path=tmp_path / "leak.db", session_id="s1")
    beam.remember("leak guard alpha", metadata={"secret": "never-emit"})
    beam.consolidate_to_episodic(
        "episodic leak alpha", [], metadata={"secret": "never-emit"}
    )

    results = beam.recall("leak alpha", top_k=20)
    assert results, "recall returned nothing to assert against"
    for row in results:
        assert "metadata_json" not in row, (
            f"raw metadata_json leaked on row {row.get('id')!r}"
        )
        assert isinstance(row.get("metadata"), dict), (
            f"row {row.get('id')!r} missing parsed metadata dict"
        )
