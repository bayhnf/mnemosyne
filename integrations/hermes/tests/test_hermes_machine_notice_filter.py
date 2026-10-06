"""Tests for Hermes machine-generated notice filtering in mnemosyne_hermes sync_turn (Issue #1102)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from mnemosyne.core.beam import BeamMemory
from mnemosyne_hermes import MnemosyneMemoryProvider


MACHINE_NOTICES = [
    "[IMPORTANT: Background process proc_1 completed normally (exit code 0).\nCommand: ls\nOutput:\nfoo]",
    "[IMPORTANT: Background process proc_2 exited (exit code 1).\nCommand: test.sh\nOutput:\nerror]",
    "[IMPORTANT: Background process proc_3 matched watch pattern \"done\".\nCommand: build.sh\nMatched output:\ndone]",
    "[IMPORTANT: 2 background processes completed.\n[IMPORTANT: Background process proc_a completed normally (exit code 0)]]",
    "[IMPORTANT: You are running as a scheduled cron job. DELIVERY: Your final response will be automatically delivered.]",
    "[IMPORTANT: MCP servers have been reloaded. Added 2 tools.]",
    "[System note: The user switched models from claude to gpt]",
    "[ASYNC DELEGATION COMPLETE — deleg_123]",
    "[ASYNC DELEGATION BATCH COMPLETE — deleg_456]",
    "[ASYNC DELEGATION TASK FAILED — deleg_789, task 2/3]",
    "[CONTEXT COMPACTION — REFERENCE ONLY]\nPrevious summary content here",
    "[CONTEXT SUMMARY]: previous context of discussion",
    "[PRIOR CONTEXT — for reference only; not a new message]",
    "[Your active task list was preserved across context compression]",
    "[Planning state preserved across model switch]",
    "A background fan-out of 3 subagent(s) you dispatched earlier has finished.",
    "A background subagent you dispatched earlier has finished. You may have moved on.",
]

HUMAN_CONTROLS = [
    "IMPORTANT: Background process — what does that mean?",
    "[IMPORTANT: Background process — what does that mean?]",
    "A background process I ran has finished — can you check the output?",
    "A background subagent you dispatched earlier has finished? no wait, I was asking about the report",
    "When you see PRIOR CONTEXT, treat it carefully",
    "the async delegation batch complete marker disappeared from my log",
    "I want to know about your task list",
]


@pytest.fixture
def provider():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        prov = MnemosyneMemoryProvider()
        prov._beam = BeamMemory(db_path=db_path)
        prov._sync_roles = {"user", "assistant"}
        prov._capture_identity_signals = MagicMock()
        prov._beam.remember = MagicMock(return_value="mem_123")
        yield prov


@pytest.mark.parametrize("notice", MACHINE_NOTICES)
def test_sync_turn_rejects_machine_notices(provider, notice):
    """Machine notice turns are rejected: neither user nor paired assistant autosaves."""
    provider.sync_turn(notice, "I see the process completed.")
    provider._beam.remember.assert_not_called()
    provider._capture_identity_signals.assert_not_called()


@pytest.mark.parametrize("notice", MACHINE_NOTICES)
def test_sync_turn_assistant_only_skips_machine_notices(provider, notice):
    """Even in assistant-only mode, paired assistant turns for machine notices are skipped."""
    provider._sync_roles = {"assistant"}
    provider.sync_turn(notice, "I see the process completed.")
    provider._beam.remember.assert_not_called()


@pytest.mark.parametrize("human_msg", HUMAN_CONTROLS)
def test_sync_turn_preserves_human_controls(provider, human_msg):
    """Genuine user prompts mentioning notice terms are preserved and autosaved."""
    provider.sync_turn(human_msg, "I can explain that.")

    calls = provider._beam.remember.call_args_list
    assert len(calls) == 2
    contents = [c.kwargs.get("content", "") for c in calls]
    assert contents[0] == f"[USER] {human_msg}"
    assert contents[1] == "[ASSISTANT] I can explain that."
    provider._capture_identity_signals.assert_called_once_with(human_msg)


def test_sync_turn_preserves_interrupted_human_prompt(provider):
    """Mixed turns with system note envelope prefix strip the prefix and save genuine input."""
    raw = (
        "[System note: Your previous turn was interrupted mid-run — the app or its backend process stopped. "
        "The interrupted request was:]\n\n"
        "please build the data ingestion pipeline"
    )
    provider.sync_turn(raw, "Working on the pipeline.")

    calls = provider._beam.remember.call_args_list
    assert len(calls) == 2
    contents = [c.kwargs.get("content", "") for c in calls]
    assert contents[0] == "[USER] please build the data ingestion pipeline"
    assert contents[1] == "[ASSISTANT] Working on the pipeline."
    provider._capture_identity_signals.assert_called_once_with("please build the data ingestion pipeline")


def test_sync_turn_with_real_sqlite_storage():
    """End-to-end test with real SQLite storage confirming zero rows for notices."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "real.db"
        prov = MnemosyneMemoryProvider()
        prov._beam = BeamMemory(db_path=db_path)
        prov._sync_roles = {"user", "assistant"}

        prov.sync_turn(
            "[IMPORTANT: Background process proc_99 completed normally (exit code 0).\nCommand: ls\nOutput:\nok]",
            "Acknowledged."
        )
        prov.sync_turn("My favorite programming language is Python", "Python is great!")

        results = prov._beam.recall("Background process", top_k=10)
        assert len(results) == 0

        py_results = prov._beam.recall("favorite programming language Python", top_k=5)
        assert len(py_results) > 0
        assert "favorite programming language is Python" in py_results[0]["content"]


@pytest.mark.parametrize("classifier_mode", ["off", "strict"])
def test_sync_turn_classifier_modes_reject_machine_notices(classifier_mode, monkeypatch):
    """Machine notice rejection is deterministic and independent of write_classifier mode."""
    monkeypatch.setenv("MNEMOSYNE_WRITE_CLASSIFIER", classifier_mode)
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "real.db"
        prov = MnemosyneMemoryProvider()
        prov._beam = BeamMemory(db_path=db_path)
        prov._sync_roles = {"user", "assistant"}

        prov.sync_turn(
            "[IMPORTANT: Background process proc_strict completed normally (exit code 0).\nCommand: ls\nOutput:\nok]",
            "Acknowledged."
        )
        assert len(prov._beam.recall("Background process proc_strict", top_k=5)) == 0


def test_sync_turn_custom_ignore_patterns_still_apply():
    """Custom ignore_patterns still filter configured user patterns alongside machine notices."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "real.db"
        prov = MnemosyneMemoryProvider()
        prov._beam = BeamMemory(db_path=db_path)
        prov._sync_roles = {"user", "assistant"}
        prov._write_policy_overrides = {"ignore_patterns": ["^SKIP_THIS_PATTERN"]}

        # Machine notice dropped
        prov.sync_turn("[IMPORTANT: 2 background processes completed]", "ack")
        # Custom ignore pattern dropped
        prov.sync_turn("SKIP_THIS_PATTERN: confidential token", "ack")
        # Genuine user prompt retained
        prov.sync_turn("KEEP_THIS_PATTERN: legitimate technical insight", "ack")

        assert len(prov._beam.recall("background processes completed", top_k=5)) == 0
        assert len(prov._beam.recall("confidential token", top_k=5)) == 0
        retained = prov._beam.recall("legitimate technical insight", top_k=5)
        assert len(retained) > 0
        assert "legitimate technical insight" in retained[0]["content"]


def test_sync_turn_no_verbatim_ledger_capture_on_machine_notice(provider):
    """Verbatim ledger records zero captures when turn is a machine notice envelope."""
    ledger = getattr(provider, "_verbatim_ledger", None)
    if ledger is not None:
        provider._active_session_id = "test-session"
        provider.sync_turn(
            "[IMPORTANT: Background process proc_ledger completed normally (exit code 0).\nCommand: ls\nOutput:\nok]",
            "Acknowledged.",
            session_id="test-session"
        )
        snapshot = ledger.snapshot_for("test-session")
        assert snapshot is None or len(snapshot.captures) == 0

