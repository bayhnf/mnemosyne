"""Task 17: MCP tool results must never expose raw exception text."""

from contextlib import contextmanager
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mnemosyne.mcp_tools import handle_tool_call


CANARY = "TASK17_PRIVATE_EXCEPTION_CANARY"


def _fresh_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))


def _seed_private_memory():
    result = handle_tool_call("mnemosyne_remember", {"content": "task17 seed"})
    return result["memory_id"]


def test_bounded_recall_policy_error_is_static(monkeypatch, tmp_path):
    _fresh_env(monkeypatch, tmp_path)

    result = handle_tool_call(
        "mnemosyne_recall",
        {"query": "x", "top_k_bounded": 0},
    )

    assert result == {"error": "invalid_recall_policy"}
    assert "positive int" not in json.dumps(result)
    assert "got 0" not in json.dumps(result)


def test_validate_transaction_failure_omits_exception_reason(monkeypatch, tmp_path):
    _fresh_env(monkeypatch, tmp_path)
    memory_id = _seed_private_memory()

    from mnemosyne import mcp_tools

    @contextmanager
    def _raising_transaction(_conn):
        raise RuntimeError(CANARY)
        yield

    monkeypatch.setattr(mcp_tools, "_guarded_transaction", _raising_transaction)

    result = handle_tool_call(
        "mnemosyne_validate",
        {"memory_id": memory_id, "action": "attest", "validator": "task17"},
    )

    assert result == {"error": "validation_failed", "memory_id": memory_id}
    assert CANARY not in json.dumps(result)
    assert "reason" not in result


def test_ingest_status_core_value_error_is_structured(monkeypatch, tmp_path):
    _fresh_env(monkeypatch, tmp_path)
    handle_tool_call(
        "mnemosyne_ingest",
        {
            "event_id": "task17-event",
            "producer": "test",
            "actor_id": "actor",
            "project_id": "project",
            "session_id": "session",
            "turn_id": "turn",
            "role": "user",
            "content": "seed only",
            "occurred_at": "2026-08-10T00:00:00Z",
        },
    )

    from mnemosyne.core import inhale

    def _raise_value_error(*_args, **_kwargs):
        raise ValueError(CANARY)

    monkeypatch.setattr(inhale, "ingest_status", _raise_value_error)

    # mnemosyne_ingest above uses the default bank, so use that same real
    # read-only bank to reach the core ingest_status() call.
    result = handle_tool_call("mnemosyne_ingest_status", {"bank": "default"})

    assert result == {
        "status": "error",
        "error": "invalid_request",
        "bank": "default",
    }
    assert CANARY not in json.dumps(result)
