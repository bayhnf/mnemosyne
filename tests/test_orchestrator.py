"""Tests for the recall orchestrator compatibility wrapper."""

import logging
import tempfile
from pathlib import Path

import pytest

CANARY = "TASK27_PRIVATE_CANARY"


def test_orchestrate_recall_with_beam_instance():
    from mnemosyne.core.beam import BeamMemory
    from mnemosyne.core.orchestrator import orchestrate_recall

    with tempfile.TemporaryDirectory() as tmpdir:
        beam = BeamMemory(session_id="test", db_path=Path(tmpdir) / "mnemosyne.db")
        beam.remember("Orchestrator test memory", importance=0.8)

        results = orchestrate_recall("orchestrator", beam=beam, top_k=3)
        assert results
        assert any("Orchestrator" in r.get("content", "") for r in results)


def test_orchestrate_recall_without_conn_uses_default_wrapper(monkeypatch, tmp_path):
    from mnemosyne.core.orchestrator import orchestrate_recall

    # Smoke test: no conn/beam should not raise even when no results exist.
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    results = orchestrate_recall("no matching memory", top_k=3)
    assert isinstance(results, list)


class _FailingBeam:
    def recall(self, query, top_k=20, **kwargs):
        raise RuntimeError("db exploded: " + CANARY)


def test_orchestrate_recall_propagates_failure_content_free(caplog):
    """Task 27: a failing recall must propagate the original exception and
    emit no canary/traceback/exc_info into log records (no silent fallback)."""
    from mnemosyne.core.orchestrator import orchestrate_recall

    with caplog.at_level(logging.WARNING, logger="mnemosyne.core.orchestrator"):
        with pytest.raises(RuntimeError, match="db exploded"):
            orchestrate_recall("q", beam=_FailingBeam())

    records = [r for r in caplog.records if r.name == "mnemosyne.core.orchestrator"]
    for record in records:
        assert CANARY not in record.getMessage()
        assert CANARY not in (record.exc_text or "")
        assert record.exc_info is None
