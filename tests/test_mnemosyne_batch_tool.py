import json
import sqlite3
from pathlib import Path

import numpy as np
from hermes_memory_provider import MnemosyneMemoryProvider
from mnemosyne.core.beam import BeamMemory


def _beam(tmp_path):
    return BeamMemory(session_id="test_provider", db_path=Path(tmp_path) / "test.db")


def _provider(tmp_path):
    provider = MnemosyneMemoryProvider()
    provider._beam = _beam(tmp_path)
    provider._session_id = "test_provider"
    provider._agent_context = "primary"
    return provider


def _count_matching(beam, text):
    row = beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE content = ?",
        (text,),
    ).fetchone()
    return row[0]


def _enable_plain_vec_working(monkeypatch, beam):
    """Force the vector-write path without requiring sqlite-vec locally."""
    from mnemosyne.core import beam as beam_module

    beam.conn.execute("DROP TABLE IF EXISTS vec_working")
    beam.conn.execute(
        "CREATE TABLE vec_working (rowid INTEGER PRIMARY KEY, embedding TEXT)"
    )
    beam.conn.commit()
    monkeypatch.setattr(beam_module._embeddings, "available", lambda: True)
    monkeypatch.setattr(
        beam_module._embeddings,
        "embed",
        lambda contents: [
            np.full(beam_module.EMBEDDING_DIM, 0.1, dtype=np.float32)
            for _ in contents
        ],
    )
    monkeypatch.setattr(beam_module, "_wm_vec_available", lambda _conn: True)


def _vector_row_counts(beam):
    return {
        "working_memory": beam.conn.execute(
            "SELECT COUNT(*) FROM working_memory"
        ).fetchone()[0],
        "memory_embeddings": beam.conn.execute(
            "SELECT COUNT(*) FROM memory_embeddings"
        ).fetchone()[0],
        "vec_working": beam.conn.execute(
            "SELECT COUNT(*) FROM vec_working"
        ).fetchone()[0],
    }


def test_batch_schema_registered_and_dispatches(tmp_path):
    provider = _provider(tmp_path)
    names = {schema["name"] for schema in provider.get_tool_schemas()}
    assert "mnemosyne_batch" in names

    result = json.loads(provider.handle_tool_call("mnemosyne_batch", {
        "operations": [{"action": "remember", "content": "batch dispatch"}],
    }))
    assert result["status"] == "ok"
    assert result["results"][0]["status"] == "stored"


def test_batch_multiple_remember_returns_ids(tmp_path):
    provider = _provider(tmp_path)
    result = json.loads(provider.handle_tool_call("mnemosyne_batch", {
        "operations": [
            {"action": "remember", "content": "batch alpha"},
            {"action": "remember", "content": "batch beta"},
        ],
    }))

    assert result["status"] == "ok"
    ids = [item["memory_id"] for item in result["results"]]
    assert len(ids) == 2
    assert all(ids)
    assert provider._beam.get(ids[0])["content"] == "batch alpha"
    assert provider._beam.get(ids[1])["content"] == "batch beta"


def test_batch_update_and_invalidate(tmp_path):
    provider = _provider(tmp_path)
    update_id = provider._beam.remember("batch old", importance=0.3)
    invalidate_id = provider._beam.remember("batch expires", importance=0.3)

    result = json.loads(provider.handle_tool_call("mnemosyne_batch", {
        "operations": [
            {"action": "update", "memory_id": update_id, "content": "batch new", "importance": "0.8"},
            {"action": "invalidate", "memory_id": invalidate_id},
        ],
    }))

    assert result["status"] == "ok"
    assert [item["status"] for item in result["results"]] == ["updated", "invalidated"]
    updated = provider._beam.get(update_id)
    assert updated["content"] == "batch new"
    assert updated["importance"] == 0.8
    invalidated = provider._beam.conn.execute(
        "SELECT valid_until FROM working_memory WHERE id = ?",
        (invalidate_id,),
    ).fetchone()
    assert invalidated[0]


def test_batch_update_then_invalidate_with_replacement_preserves_outer_transaction(tmp_path):
    provider = _provider(tmp_path)
    target_id = provider._beam.remember("batch replacement target", importance=0.3)
    replacement_id = provider._beam.remember("batch replacement", importance=0.3)

    result = json.loads(provider.handle_tool_call("mnemosyne_batch", {
        "operations": [
            {
                "action": "update",
                "memory_id": target_id,
                "content": "batch replacement target updated",
            },
            {
                "action": "invalidate",
                "memory_id": target_id,
                "replacement_id": replacement_id,
            },
        ],
    }))

    assert result["status"] == "ok"
    assert [item["status"] for item in result["results"]] == ["updated", "invalidated"]
    target = provider._beam.get(target_id)
    assert target["content"] == "batch replacement target updated"
    row = provider._beam.conn.execute(
        "SELECT valid_until, superseded_by FROM working_memory WHERE id = ?", (target_id,)
    ).fetchone()
    assert row[0] is not None
    assert row[1] == replacement_id


def test_batch_extract_remember_uses_provider_default_scope(tmp_path):
    provider = _provider(tmp_path)
    provider._default_scope = "session"

    result = json.loads(provider.handle_tool_call("mnemosyne_batch", {
        "operations": [
            {"action": "remember", "content": "scope parity extract", "extract": True},
        ],
    }))

    assert result["status"] == "ok"
    memory_id = result["results"][0]["memory_id"]
    row = provider._beam.conn.execute(
        "SELECT scope FROM working_memory WHERE id = ?",
        (memory_id,),
    ).fetchone()
    assert row[0] == "session"


def test_batch_failure_rolls_back_earlier_remember(tmp_path):
    provider = _provider(tmp_path)
    result = json.loads(provider.handle_tool_call("mnemosyne_batch", {
        "operations": [
            {"action": "remember", "content": "rollback me"},
            {"action": "update", "memory_id": "missing", "content": "x"},
        ],
    }))

    assert result["status"] == "error"
    assert result["failed_index"] == 1
    assert result["action"] == "update"
    assert _count_matching(provider._beam, "rollback me") == 0


def test_batch_failure_rolls_back_vectorized_remember(tmp_path, monkeypatch):
    provider = _provider(tmp_path)
    _enable_plain_vec_working(monkeypatch, provider._beam)

    result = json.loads(provider.handle_tool_call("mnemosyne_batch", {
        "operations": [
            {"action": "remember", "content": "vector rollback"},
            {"action": "update", "memory_id": "missing", "content": "x"},
        ],
    }))

    assert result == {
        "status": "error",
        "error": "batch_failed",
        "failed_index": 1,
        "action": "update",
    }
    assert _vector_row_counts(provider._beam) == {
        "working_memory": 0,
        "memory_embeddings": 0,
        "vec_working": 0,
    }


def test_batch_success_commits_vectorized_remember(tmp_path, monkeypatch):
    provider = _provider(tmp_path)
    _enable_plain_vec_working(monkeypatch, provider._beam)

    result = json.loads(provider.handle_tool_call("mnemosyne_batch", {
        "operations": [{"action": "remember", "content": "vector commit"}],
    }))

    assert result["status"] == "ok"
    with sqlite3.connect(provider._beam.db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM working_memory"
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM memory_embeddings"
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM vec_working"
        ).fetchone()[0] == 1


def test_batch_failure_rolls_back_earlier_update(tmp_path):
    provider = _provider(tmp_path)
    memory_id = provider._beam.remember("before update", importance=0.3)

    result = json.loads(provider.handle_tool_call("mnemosyne_batch", {
        "operations": [
            {"action": "update", "memory_id": memory_id, "content": "after update"},
            {"action": "forget", "memory_id": "missing"},
        ],
    }))

    assert result["status"] == "error"
    assert result["failed_index"] == 1
    assert result["action"] == "forget"
    assert provider._beam.get(memory_id)["content"] == "before update"


def test_batch_audit_events_emit_only_after_successful_commit(tmp_path):
    provider = _provider(tmp_path)
    events = []
    provider._audit_event = lambda name, **kwargs: events.append((name, kwargs))

    failed = json.loads(provider.handle_tool_call("mnemosyne_batch", {
        "operations": [
            {"action": "remember", "content": "audit rollback"},
            {"action": "update", "memory_id": "missing", "content": "x"},
        ],
    }))
    assert failed["status"] == "error"
    assert events == []

    ok = json.loads(provider.handle_tool_call("mnemosyne_batch", {
        "operations": [
            {"action": "remember", "content": "audit commit"},
        ],
    }))
    assert ok["status"] == "ok"
    assert [event[0] for event in events] == ["remember"]


def test_batch_dry_run_writes_nothing(tmp_path):
    provider = _provider(tmp_path)
    existing_id = provider._beam.remember("dry existing", importance=0.3)
    result = json.loads(provider.handle_tool_call("mnemosyne_batch", {
        "dry_run": True,
        "operations": [
            {"action": "remember", "content": "dry new"},
            {"action": "update", "memory_id": existing_id, "content": "dry changed"},
        ],
    }))

    assert result["status"] == "dry_run"
    assert [item["status"] for item in result["results"]] == ["would_store", "would_update"]
    assert _count_matching(provider._beam, "dry new") == 0
    assert provider._beam.get(existing_id)["content"] == "dry existing"


def test_batch_unknown_action_rejected_before_mutation(tmp_path):
    provider = _provider(tmp_path)
    result = json.loads(provider.handle_tool_call("mnemosyne_batch", {
        "operations": [
            {"action": "remember", "content": "should not write"},
            {"action": "search", "query": "x"},
        ],
    }))

    assert result["status"] == "error"
    assert result["failed_index"] == 1
    assert _count_matching(provider._beam, "should not write") == 0


def test_batch_requires_exact_ids_for_destructive_ops(tmp_path):
    provider = _provider(tmp_path)
    for action in ("update", "forget", "invalidate"):
        op = {"action": action}
        if action == "update":
            op["content"] = "x"
        result = json.loads(provider.handle_tool_call("mnemosyne_batch", {
            "operations": [op],
        }))
        assert result["status"] == "error"
        assert result["failed_index"] == 0
        assert result["action"] == action


def test_batch_validation_error_hides_untrusted_action(tmp_path):
    provider = _provider(tmp_path)
    result = json.loads(provider.handle_tool_call("mnemosyne_batch", {
        "operations": [{"action": "synthetic-untrusted-action"}],
    }))

    assert result == {
        "status": "error",
        "error": "batch_validation_failed",
        "failed_index": 0,
    }
    assert "synthetic-untrusted-action" not in json.dumps(result)


def test_batch_execution_error_hides_internal_detail(tmp_path, monkeypatch, caplog):
    provider = _provider(tmp_path)

    def _explode(*args, **kwargs):
        raise RuntimeError("synthetic-internal-detail")

    monkeypatch.setattr("mnemosyne.batch_tool._apply_one", _explode)
    result = json.loads(provider.handle_tool_call("mnemosyne_batch", {
        "operations": [{"action": "remember", "content": "trigger failure"}],
    }))

    assert result["status"] == "error"
    assert result["error"] == "batch_failed"
    assert result["failed_index"] == 0
    assert result["action"] == "remember"
    assert "synthetic-internal-detail" not in json.dumps(result)
    assert "synthetic-internal-detail" not in caplog.text
    assert "Traceback" not in caplog.text
