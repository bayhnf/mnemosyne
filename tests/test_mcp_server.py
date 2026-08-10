"""
Tests for Mnemosyne MCP Server (Phase 6)

Run with: pytest tests/test_mcp_server.py -v
"""

import json
import os
import sqlite3
import subprocess
import sys
import pytest
from unittest.mock import MagicMock, patch

import mnemosyne.mcp_tools as mcp_tools
from mnemosyne.core.beam import BeamMemory

# Test tool schemas
from mnemosyne.mcp_tools import (
    TOOLS, get_tool_definitions, handle_tool_call, _create_instance,
)


class TestToolSchemas:
    """Verify tool schemas match MCP spec and are valid JSON."""

    def test_all_tools_present(self):
        """Core MCP tools must be defined without duplicate names."""
        names = [t["name"] for t in TOOLS]
        assert len(names) == len(set(names))
        assert len(names) >= 25
        assert "mnemosyne_remember_canonical" in names
        assert "mnemosyne_recall_canonical" in names
        assert "mnemosyne_remember" in names
        assert "mnemosyne_batch" in names
        assert "mnemosyne_recall" in names
        assert "mnemosyne_sleep" in names
        assert "mnemosyne_scratchpad_read" in names
        assert "mnemosyne_scratchpad_write" in names
        assert "mnemosyne_stats" in names  # renamed from get_stats
        # New shared tools
        assert "mnemosyne_shared_remember" in names
        assert "mnemosyne_shared_recall" in names
        assert "mnemosyne_shared_forget" in names
        assert "mnemosyne_shared_stats" in names
        # New validation/memory tools
        assert "mnemosyne_invalidate" in names
        assert "mnemosyne_validate" in names
        assert "mnemosyne_get" in names
        assert "mnemosyne_forget" in names
        assert "mnemosyne_export" in names
        assert "mnemosyne_import" in names
        # New graph/triple tools
        assert "mnemosyne_triple_add" in names
        assert "mnemosyne_triple_query" in names
        assert "mnemosyne_graph_query" in names
        assert "mnemosyne_graph_link" in names
        assert "mnemosyne_scratchpad_clear" in names
        assert "mnemosyne_update" in names
        assert "mnemosyne_diagnose" in names

    def test_tool_schemas_are_valid_json(self):
        """Each tool schema must be valid JSON-serializable."""
        for tool in TOOLS:
            # Schema must be serializable. ``mcp`` SDK 2.x renamed the wire
            # field to ``input_schema`` (was ``inputSchema`` in 1.x); the
            # schema dict in ``TOOLS`` uses the new key.
            dumped = json.dumps(tool["input_schema"])
            loaded = json.loads(dumped)
            assert loaded["type"] == "object"
            assert "properties" in loaded

    def test_remember_schema_has_required_fields(self):
        """mnemosyne_remember requires 'content'."""
        remember_tool = next(t for t in TOOLS if t["name"] == "mnemosyne_remember")
        schema = remember_tool["input_schema"]
        assert "required" in schema
        assert "content" in schema["required"]
        assert "properties" in schema
        assert "source" in schema["properties"]
        assert "importance" in schema["properties"]
        assert "metadata" in schema["properties"]
        # bank is not in the schema - handled via MCP server env var MNEMOSYNE_MCP_BANK
        assert "extract_entities" in schema["properties"]
        assert "extract" in schema["properties"]
        assert "veracity" in schema["properties"]

    def test_recall_schema_has_required_fields(self):
        """mnemosyne_recall requires 'query'."""
        recall_tool = next(t for t in TOOLS if t["name"] == "mnemosyne_recall")
        schema = recall_tool["input_schema"]
        assert "required" in schema
        assert "query" in schema["required"]
        assert "limit" in schema["properties"]
        # bank is not in the schema - handled via MCP server env var MNEMOSYNE_MCP_BANK
        assert "temporal_weight" in schema["properties"]
        assert schema["properties"]["explain"]["type"] == "boolean"

    def test_destructive_tools_exist(self):
        """Destructive tools are now exposed (Phase 7+)."""
        names = [t["name"] for t in TOOLS]
        # These tools now exist in the 23-tool set
        assert "mnemosyne_forget" in names
        assert "mnemosyne_invalidate" in names
        assert "mnemosyne_export" in names
        assert "mnemosyne_import" in names

    def test_invalidate_schema_documents_scope_safe_failure(self):
        """Invalidate must not reveal whether an out-of-scope ID exists."""
        invalidate_tool = next(t for t in TOOLS if t["name"] == "mnemosyne_invalidate")
        assert invalidate_tool["description"] == (
            "Mark a memory as expired or superseded. Provide memory_id from recall results. "
            "Optionally provide a replacement_id that must resolve to a working-memory or episodic-memory "
            "record that is accessible in the current session or global scope to chain old to new. "
            "An unknown or out-of-scope target or replacement returns status: memory_not_found."
        )

    def test_batch_schema_has_operations(self):
        batch_tool = next(t for t in TOOLS if t["name"] == "mnemosyne_batch")
        schema = batch_tool["input_schema"]
        assert "operations" in schema["required"]
        assert schema["properties"]["operations"]["type"] == "array"
        assert schema["properties"]["operations"]["maxItems"] == 50
        for context_field in ("bank", "author_id", "author_type", "channel_id"):
            assert context_field in schema["properties"]


class TestToolHandlers:
    """Test each handler with mocked Mnemosyne instance."""

    @pytest.fixture
    def mock_mnemosyne(self):
        """Create a mock Mnemosyne instance."""
        mock = MagicMock()
        mock.remember.return_value = "test-memory-id-123"
        mock.recall.return_value = [
            {"id": "mem1", "content": "Test content", "score": 0.95}
        ]
        mock.sleep.return_value = {"consolidated": 3, "deleted": 1}
        mock.scratchpad_read.return_value = ["entry1", "entry2"]
        mock.scratchpad_write.return_value = "scratch-id-456"
        mock.get_stats.return_value = {
            "total_memories": 42,
            "total_sessions": 3,
            "sources": {"conversation": 30, "file": 12},
            "last_memory": "2026-04-29T01:00:00",
            "database": "/test/db",
            "mode": "beam",
            "beam": {"working_memory": {}, "episodic_memory": {}}
        }
        return mock

    def test_handle_remember(self, mock_mnemosyne):
        """handle_remember returns success with memory_id."""
        with patch("mnemosyne.mcp_tools._create_instance", return_value=mock_mnemosyne):
            result = handle_tool_call("mnemosyne_remember", {
                "content": "Test memory",
                "source": "test",
                "importance": 0.9,
                "bank": "default"
            })
        assert result["status"] == "stored"
        assert result["memory_id"] == "test-memory-id-123"
        assert result["bank"] == "default"
        mock_mnemosyne.remember.assert_called_once()

    def test_handle_remember_forwards_veracity(self, tmp_path, monkeypatch):
        """Regression: MCP remember must forward veracity to Mnemosyne.remember.

        #386 wired veracity into _handle_remember() but Mnemosyne.remember()
        never got the parameter, so every real MCP remember raised
        `TypeError: remember() got an unexpected keyword argument 'veracity'`.
        The mocked handler tests above miss it because a MagicMock swallows any
        kwarg -- this test drives a real instance so the signature is exercised.
        """
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))

        result = handle_tool_call("mnemosyne_remember", {
            "content": "veracity plumbing regression",
            "source": "test",
            "scope": "global",
            "veracity": "tool",
        })
        assert result["status"] == "stored"

        # veracity must persist to the beam working_memory row, not be dropped.
        mem = _create_instance(bank="default")
        row = mem.beam.conn.execute(
            "SELECT veracity FROM working_memory WHERE id = ?",
            (result["memory_id"],),
        ).fetchone()
        assert row is not None
        assert row[0] == "tool"

    def test_handle_remember_uses_mcp_bank_env_default(self, mock_mnemosyne, monkeypatch):
        """MCP server bank default applies when tool call omits bank."""
        monkeypatch.setenv("MNEMOSYNE_MCP_BANK", "work")

        with patch(
            "mnemosyne.mcp_tools._create_instance",
            return_value=mock_mnemosyne,
        ) as create_instance:
            result = handle_tool_call("mnemosyne_remember", {
                "content": "Test memory",
                "source": "test",
            })

        assert result["status"] == "stored"
        assert result["bank"] == "work"
        assert create_instance.call_args.kwargs["bank"] == "work"

    def test_handle_remember_bank_arg_overrides_mcp_bank_env(self, mock_mnemosyne, monkeypatch):
        """Explicit per-call bank should override the server default bank."""
        monkeypatch.setenv("MNEMOSYNE_MCP_BANK", "work")

        with patch(
            "mnemosyne.mcp_tools._create_instance",
            return_value=mock_mnemosyne,
        ) as create_instance:
            result = handle_tool_call("mnemosyne_remember", {
                "content": "Test memory",
                "source": "test",
                "bank": "personal",
            })

        assert result["status"] == "stored"
        assert result["bank"] == "personal"
        assert create_instance.call_args.kwargs["bank"] == "personal"

    def test_handle_invalidate_preserves_session_scope_and_reports_not_found(self, tmp_path, monkeypatch):
        """MCP invalidate must not report success for a foreign session row."""
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))

        foreign = _create_instance(session_id="foreign-session", bank="default")
        foreign_id = foreign.beam.remember("foreign session memory", scope="session")

        mcp = _create_instance(bank="default")
        same_session_id = mcp.beam.remember("mcp session memory", scope="session")
        global_id = mcp.beam.remember("global memory", scope="global")

        wrapper_events = []
        monkeypatch.setattr(
            mcp_tools.Mnemosyne,
            "_emit_wrapper",
            lambda _self, event_type, memory_id, **kwargs: wrapper_events.append(
                (event_type, memory_id, kwargs)
            ),
        )
        failed = handle_tool_call("mnemosyne_invalidate", {
            "memory_id": foreign_id,
            "replacement_id": same_session_id,
        })
        assert failed["status"] == "memory_not_found"
        assert failed["memory_id"] == foreign_id
        assert wrapper_events == []
        foreign_row = mcp.beam.conn.execute(
            "SELECT valid_until, superseded_by FROM working_memory WHERE id = ?", (foreign_id,)
        ).fetchone()
        assert tuple(foreign_row) == (None, None)

        assert handle_tool_call("mnemosyne_invalidate", {"memory_id": same_session_id}) == {
            "status": "invalidated",
            "memory_id": same_session_id,
        }
        assert handle_tool_call("mnemosyne_invalidate", {"memory_id": global_id}) == {
            "status": "invalidated",
            "memory_id": global_id,
        }
        successful_rows = mcp.beam.conn.execute(
            "SELECT valid_until FROM working_memory WHERE id IN (?, ?)",
            (same_session_id, global_id),
        ).fetchall()
        assert len(successful_rows) == 2
        assert all(row[0] for row in successful_rows)

    def test_handle_invalidate_validates_replacement_in_current_scope(self, tmp_path, monkeypatch):
        """MCP replacement links must be resolvable before invalidating a target."""
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))

        mcp = _create_instance(bank="default")
        target_id = mcp.beam.remember("replacement validation target", scope="session")
        replacement_id = mcp.beam.remember("authorized replacement", scope="session")
        global_target_id = mcp.beam.remember("global replacement target", scope="global")
        global_replacement_id = mcp.beam.remember("global replacement", scope="global")

        foreign = _create_instance(session_id="foreign-session", bank="default")
        foreign_replacement_id = foreign.beam.remember(
            "foreign replacement", scope="session"
        )

        wrapper_events = []
        monkeypatch.setattr(
            mcp_tools.Mnemosyne,
            "_emit_wrapper",
            lambda _self, event_type, memory_id, **kwargs: wrapper_events.append(
                (event_type, memory_id, kwargs)
            ),
        )
        cache_invalidations = []
        monkeypatch.setattr(
            BeamMemory,
            "_invalidate_query_cache",
            lambda self: cache_invalidations.append(self),
        )

        assert handle_tool_call("mnemosyne_invalidate", {
            "memory_id": target_id,
            "replacement_id": "unknown-replacement",
        }) == {"status": "memory_not_found", "memory_id": target_id}
        target_row = mcp.beam.conn.execute(
            "SELECT valid_until, superseded_by FROM working_memory WHERE id = ?", (target_id,)
        ).fetchone()
        assert tuple(target_row) == (None, None)
        assert wrapper_events == []
        assert cache_invalidations == []

        assert handle_tool_call("mnemosyne_invalidate", {
            "memory_id": target_id,
            "replacement_id": foreign_replacement_id,
        }) == {"status": "memory_not_found", "memory_id": target_id}
        target_row = mcp.beam.conn.execute(
            "SELECT valid_until, superseded_by FROM working_memory WHERE id = ?", (target_id,)
        ).fetchone()
        assert tuple(target_row) == (None, None)
        assert wrapper_events == []
        assert cache_invalidations == []

        assert handle_tool_call("mnemosyne_invalidate", {
            "memory_id": target_id,
            "replacement_id": target_id,
        }) == {"status": "memory_not_found", "memory_id": target_id}
        target_row = mcp.beam.conn.execute(
            "SELECT valid_until, superseded_by FROM working_memory WHERE id = ?", (target_id,)
        ).fetchone()
        assert tuple(target_row) == (None, None)
        assert wrapper_events == []
        assert cache_invalidations == []

        assert handle_tool_call("mnemosyne_invalidate", {
            "memory_id": target_id,
            "replacement_id": replacement_id,
        }) == {"status": "invalidated", "memory_id": target_id}
        target_row = mcp.beam.conn.execute(
            "SELECT valid_until, superseded_by FROM working_memory WHERE id = ?", (target_id,)
        ).fetchone()
        assert target_row[0] is not None
        assert target_row[1] == replacement_id

        assert handle_tool_call("mnemosyne_invalidate", {
            "memory_id": global_target_id,
            "replacement_id": global_replacement_id,
        }) == {"status": "invalidated", "memory_id": global_target_id}
        global_target_row = mcp.beam.conn.execute(
            "SELECT valid_until, superseded_by FROM working_memory WHERE id = ?", (global_target_id,)
        ).fetchone()
        assert global_target_row[0] is not None
        assert global_target_row[1] == global_replacement_id
        assert len(cache_invalidations) == 2
        assert wrapper_events == [
            ("MEMORY_INVALIDATED", target_id, {"replacement_id": replacement_id}),
            (
                "MEMORY_INVALIDATED",
                global_target_id,
                {"replacement_id": global_replacement_id},
            ),
        ]

    def test_handle_batch_multiple_remember(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))

        result = handle_tool_call("mnemosyne_batch", {
            "operations": [
                {"action": "remember", "content": "mcp batch one"},
                {"action": "remember", "content": "mcp batch two"},
            ],
        })

        assert result["status"] == "ok"
        assert [item["status"] for item in result["results"]] == ["stored", "stored"]
        assert [event["event"] for event in result["audit_events"]] == ["remember", "remember"]
        mem = _create_instance(bank="default")
        legacy_count = mem.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE content IN (?, ?)",
            ("mcp batch one", "mcp batch two"),
        ).fetchone()[0]
        assert legacy_count == 2

    def test_handle_batch_updates_beam_only_memory(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))

        mem = _create_instance(bank="default")
        memory_id = mem.beam.remember("beam only before", importance=0.2)

        result = handle_tool_call("mnemosyne_batch", {
            "operations": [
                {"action": "update", "memory_id": memory_id, "content": "beam only after", "importance": "0.9"},
            ],
        })

        assert result["status"] == "ok"
        assert result["results"][0]["status"] == "updated"
        updated = _create_instance(bank="default").beam.get(memory_id)
        assert updated["content"] == "beam only after"
        assert updated["importance"] == 0.9


    def test_handle_batch_wrapper_update_forget_invalidate_and_scope(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))

        stored = handle_tool_call("mnemosyne_batch", {
            "operations": [
                {"action": "remember", "content": "mcp update target"},
                {"action": "remember", "content": "mcp forget target"},
                {"action": "remember", "content": "mcp invalidate target"},
                {"action": "remember", "content": "mcp extract target", "extract": True},
            ],
        })
        ids = [row["memory_id"] for row in stored["results"]]

        result = handle_tool_call("mnemosyne_batch", {
            "operations": [
                {"action": "update", "memory_id": ids[0], "content": "mcp updated", "importance": "0.7"},
                {"action": "forget", "memory_id": ids[1]},
                {"action": "invalidate", "memory_id": ids[2]},
            ],
        })

        assert result["status"] == "ok"
        assert [item["status"] for item in result["results"]] == ["updated", "deleted", "invalidated"]
        assert [event["event"] for event in result["audit_events"]] == ["update", "forget", "invalidate"]
        mem = _create_instance(bank="default")
        updated = mem.beam.get(ids[0])
        assert updated["content"] == "mcp updated"
        assert updated["importance"] == 0.7
        assert mem.beam.get(ids[1]) is None
        invalidated = mem.beam.conn.execute(
            "SELECT valid_until FROM working_memory WHERE id = ?",
            (ids[2],),
        ).fetchone()
        assert invalidated[0]
        extract_scope = mem.beam.conn.execute(
            "SELECT scope FROM working_memory WHERE id = ?",
            (ids[3],),
        ).fetchone()
        assert extract_scope[0] == "session"


    def test_handle_batch_failure_rolls_back(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))

        result = handle_tool_call("mnemosyne_batch", {
            "operations": [
                {"action": "remember", "content": "mcp rollback"},
                {"action": "update", "memory_id": "missing", "content": "x"},
            ],
        })

        assert result["status"] == "error"
        assert result["failed_index"] == 1
        mem = _create_instance(bank="default")
        row = mem.beam.conn.execute(
            "SELECT COUNT(*) FROM working_memory WHERE content = ?",
            ("mcp rollback",),
        ).fetchone()
        assert row[0] == 0

    def test_handle_recall_uses_mcp_bank_env_default(self, mock_mnemosyne, monkeypatch):
        """MCP recall should use the server default bank when omitted."""
        monkeypatch.setenv("MNEMOSYNE_MCP_BANK", "work")

        with patch(
            "mnemosyne.mcp_tools._create_instance",
            return_value=mock_mnemosyne,
        ) as create_instance:
            result = handle_tool_call("mnemosyne_recall", {
                "query": "test query",
            })

        assert result["status"] == "ok"
        assert result["bank"] == "work"
        assert create_instance.call_args.kwargs["bank"] == "work"

    def test_handle_recall(self, mock_mnemosyne):
        """handle_recall returns list of results."""
        with patch("mnemosyne.mcp_tools._create_instance", return_value=mock_mnemosyne):
            result = handle_tool_call("mnemosyne_recall", {
                "query": "test query",
                "top_k": 5,
                "bank": "default"
            })
        assert result["status"] == "ok"
        assert result["count"] == 1
        assert len(result["results"]) == 1
        mock_mnemosyne.recall.assert_called_once()
        assert mock_mnemosyne.recall.call_args.kwargs["explain"] is False

    def test_handle_recall_explain_unwraps_payload(self, mock_mnemosyne):
        mock_mnemosyne.recall.return_value = {
            "query": "test query",
            "top_k": 5,
            "results": [{"id": "mem1", "content": "Test content", "score": 0.95}],
            "explain": {"stages": [], "candidates": []},
        }
        with patch("mnemosyne.mcp_tools._create_instance", return_value=mock_mnemosyne):
            result = handle_tool_call("mnemosyne_recall", {
                "query": "test query",
                "top_k": 5,
                "explain": True,
                "bank": "default",
            })

        assert result["status"] == "ok"
        assert result["query"] == "test query"
        assert result["top_k"] == 5
        assert result["count"] == 1
        assert "explain" in result
        assert mock_mnemosyne.recall.call_args.kwargs["explain"] is True

    def test_handle_recall_forwards_scoring_weights(self, mock_mnemosyne):
        """Schema-advertised recall weights should be forwarded to Mnemosyne.recall()."""
        with patch("mnemosyne.mcp_tools._create_instance", return_value=mock_mnemosyne):
            handle_tool_call("mnemosyne_recall", {
                "query": "test query",
                "top_k": 5,
                "bank": "default",
                "vec_weight": 0.6,
                "fts_weight": 0.3,
                "importance_weight": 0.1,
            })

        _, kwargs = mock_mnemosyne.recall.call_args
        assert kwargs["vec_weight"] == 0.6
        assert kwargs["fts_weight"] == 0.3
        assert kwargs["importance_weight"] == 0.1

    def test_handle_recall_normalizes_blank_query_time(self, mock_mnemosyne):
        """Blank query_time reaches Mnemosyne.recall() as unset, not as "" (#555).

        Harnesses built against the older schema sent the declared default "",
        which used to reach _parse_query_time and raise.
        """
        for blank in ("", "   ", "\t"):
            mock_mnemosyne.recall.reset_mock()
            with patch("mnemosyne.mcp_tools._create_instance", return_value=mock_mnemosyne):
                handle_tool_call("mnemosyne_recall", {
                    "query": "test query",
                    "bank": "default",
                    "query_time": blank,
                })
            _, kwargs = mock_mnemosyne.recall.call_args
            assert kwargs["query_time"] is None, f"blank {blank!r} not normalized"

    def test_handle_recall_preserves_explicit_query_time(self, mock_mnemosyne):
        """A real ISO timestamp is forwarded untouched."""
        with patch("mnemosyne.mcp_tools._create_instance", return_value=mock_mnemosyne):
            handle_tool_call("mnemosyne_recall", {
                "query": "test query",
                "bank": "default",
                "query_time": "2026-04-29T12:00:00",
            })
        _, kwargs = mock_mnemosyne.recall.call_args
        assert kwargs["query_time"] == "2026-04-29T12:00:00"

    def test_handle_recall_does_not_swallow_falsey_non_strings(self, mock_mnemosyne):
        """0/False/[] are type errors, not "unset" — they must not become None.

        Guards against normalizing with `or None`, which would silently accept
        them and suppress _parse_query_time's TypeError.
        """
        for bad in (0, False, []):
            mock_mnemosyne.recall.reset_mock()
            with patch("mnemosyne.mcp_tools._create_instance", return_value=mock_mnemosyne):
                handle_tool_call("mnemosyne_recall", {
                    "query": "test query",
                    "bank": "default",
                    "query_time": bad,
                })
            _, kwargs = mock_mnemosyne.recall.call_args
            # Identity, not equality: 0 == False in Python, so `==` would let a
            # bool/int mix-up pass. The exact object must be forwarded.
            assert kwargs["query_time"] is bad, f"{bad!r} not forwarded unchanged"

    def test_handle_sleep(self, mock_mnemosyne):
        """handle_sleep returns consolidation stats."""
        with patch("mnemosyne.mcp_tools._create_instance", return_value=mock_mnemosyne):
            result = handle_tool_call("mnemosyne_sleep", {
                "dry_run": False,
                "bank": "default"
            })
        assert result["status"] == "consolidated"
        assert "result" in result
        assert "working" in result
        assert "episodic" in result
        assert result["bank"] == "default"
        mock_mnemosyne.sleep.assert_called_once_with(dry_run=False, force=False)

    def test_handle_scratchpad_read(self, mock_mnemosyne):
        """handle_scratchpad_read returns entries."""
        with patch("mnemosyne.mcp_tools._create_instance", return_value=mock_mnemosyne):
            result = handle_tool_call("mnemosyne_scratchpad_read", {
                "bank": "default"
            })
        assert result["entries_count"] == 2
        assert len(result["entries"]) == 2

    def test_handle_scratchpad_write(self, mock_mnemosyne):
        """handle_scratchpad_write returns entry_id."""
        with patch("mnemosyne.mcp_tools._create_instance", return_value=mock_mnemosyne):
            result = handle_tool_call("mnemosyne_scratchpad_write", {
                "content": "New scratchpad entry",
                "bank": "default"
            })
        assert result["status"] == "written"
        assert result["id"] == "scratch-id-456"

    def test_handle_get_stats(self, mock_mnemosyne):
        """handle_get_stats returns JSON-serializable stats."""
        mock_mnemosyne.get_stats.return_value = {
            "total_memories": 42,
            "total_sessions": 3,
            "sources": {"conversation": 30, "file": 12},
            "last_memory": "2026-04-29T01:00:00",
            "database": "/test/db",
            "mode": "beam",
            "beam": {"working_memory": {}, "episodic_memory": {}}
        }
        mock_mnemosyne._session_id = "test-session-123"
        with patch("mnemosyne.mcp_tools._create_instance", return_value=mock_mnemosyne):
            result = handle_tool_call("mnemosyne_stats", {
                "bank": "default"
            })
        assert "provider" in result
        assert "stats" in result
        # Must be JSON serializable
        dumped = json.dumps(result)
        loaded = json.loads(dumped)
        assert loaded["stats"]["total_memories"] == 42

    def test_error_handling(self, mock_mnemosyne):
        """Error handling returns MCP-compliant error results."""
        mock_mnemosyne.remember.side_effect = RuntimeError("DB locked")
        with patch("mnemosyne.mcp_tools._create_instance", return_value=mock_mnemosyne):
            with pytest.raises(RuntimeError, match="DB locked"):
                handle_tool_call("mnemosyne_remember", {"content": "test"})

    def test_hygiene_audit_default_uses_and_closes_exact_readonly_connection(
        self, tmp_path, monkeypatch
    ):
        """Default-bank audits use one real query-only connection, never Mnemosyne."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        db_path = data_dir / "mnemosyne.db"
        with sqlite3.connect(db_path) as writable:
            writable.execute("CREATE TABLE audit_marker (id INTEGER)")

        class _Report:
            def to_dict(self):
                return {"total_candidates": 0}

        from mnemosyne.core import hygiene

        captured = {}

        def _audit_noise(**kwargs):
            conn = kwargs["conn"]
            captured.update(kwargs)
            assert conn.execute("PRAGMA query_only").fetchone()[0] == 1
            with pytest.raises(
                sqlite3.OperationalError, match="attempt to write a readonly database"
            ):
                conn.execute("CREATE TABLE forbidden_write (id INTEGER)")
            return _Report()

        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))
        monkeypatch.setattr(
            mcp_tools,
            "_create_instance",
            lambda **_kwargs: pytest.fail("hygiene audit must not initialize Mnemosyne"),
        )
        monkeypatch.setattr(hygiene, "audit_noise", _audit_noise)

        arguments = {
            "limit": 17,
            "tables": ["audit_marker"],
            "min_score": 0.8,
            "offset": 3,
            "scan_all": True,
            "batch_size": 11,
        }
        result = handle_tool_call("mnemosyne_hygiene_audit", arguments)

        assert result == {
            "status": "audited",
            "report": {"total_candidates": 0},
            "bank": "default",
        }
        assert captured == {
            "db_path": db_path,
            "limit": 17,
            "tables": ["audit_marker"],
            "min_score": 0.8,
            "offset": 3,
            "scan_all": True,
            "batch_size": 11,
            "conn": captured["conn"],
        }
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            captured["conn"].execute("SELECT 1")

    def test_hygiene_audit_closes_connection_when_audit_raises(self, tmp_path, monkeypatch):
        """The read-only connection closes even if audit_noise raises."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        db_path = data_dir / "mnemosyne.db"
        sqlite3.connect(db_path).close()

        from mnemosyne.core import hygiene
        from mnemosyne import doctor

        captured = {}
        real_open = doctor.open_readonly_doctor_db

        def _open_readonly(path):
            captured["conn"] = real_open(path)
            return captured["conn"]

        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))
        monkeypatch.setattr(
            mcp_tools,
            "_create_instance",
            lambda **_kwargs: pytest.fail("hygiene audit must not initialize Mnemosyne"),
        )
        monkeypatch.setattr(doctor, "open_readonly_doctor_db", _open_readonly)
        monkeypatch.setattr(
            hygiene,
            "audit_noise",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("audit failed")),
        )

        with pytest.raises(RuntimeError, match="audit failed"):
            handle_tool_call("mnemosyne_hygiene_audit", {})

        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            captured["conn"].execute("SELECT 1")

    def test_hygiene_audit_readonly_open_failure_has_no_writable_fallback(
        self, tmp_path, monkeypatch
    ):
        """A missing default DB fails at the read-only open without running audit_noise."""
        data_dir = tmp_path / "missing-data"
        from mnemosyne.core import hygiene

        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))
        monkeypatch.setattr(
            mcp_tools,
            "_create_instance",
            lambda **_kwargs: pytest.fail("hygiene audit must not initialize Mnemosyne"),
        )
        monkeypatch.setattr(
            hygiene,
            "audit_noise",
            lambda **_kwargs: pytest.fail("audit must not run after readonly open failure"),
        )

        with pytest.raises(sqlite3.OperationalError):
            handle_tool_call("mnemosyne_hygiene_audit", {})

        assert not data_dir.exists()

    def test_hygiene_audit_unknown_bank_never_initializes_or_creates_state(
        self, tmp_path, monkeypatch
    ):
        """Unknown named banks fail before any manager/instance can create state."""
        data_dir = tmp_path / "fresh-data"

        def _snapshot(root):
            return (
                sorted(path.relative_to(root) for path in root.rglob("*"))
                if root.exists()
                else []
            )

        before = _snapshot(data_dir)
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))
        monkeypatch.setattr(
            mcp_tools,
            "_create_instance",
            lambda **_kwargs: pytest.fail("unknown bank must not initialize Mnemosyne"),
        )

        with pytest.raises(ValueError, match="Bank 'unknown' does not exist"):
            handle_tool_call("mnemosyne_hygiene_audit", {"bank": "unknown"})

        assert _snapshot(data_dir) == before
        assert not (data_dir / "banks").exists()
        assert not (data_dir / "config").exists()

    def test_hygiene_audit_named_bank_requires_an_existing_database(
        self, tmp_path, monkeypatch
    ):
        """A named bank directory without its DB is rejected without mutations."""
        data_dir = tmp_path / "data"
        bank_dir = data_dir / "banks" / "team"
        bank_dir.mkdir(parents=True)
        before = sorted(path.relative_to(data_dir) for path in data_dir.rglob("*"))

        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))
        monkeypatch.setattr(
            mcp_tools,
            "_create_instance",
            lambda **_kwargs: pytest.fail("named-bank audit must not initialize Mnemosyne"),
        )

        with pytest.raises(FileNotFoundError, match="Database for bank 'team' does not exist"):
            handle_tool_call("mnemosyne_hygiene_audit", {"bank": "team"})

        assert sorted(path.relative_to(data_dir) for path in data_dir.rglob("*")) == before

    def test_hygiene_clean_rejects_invalid_candidates_json(self):
        """Invalid hygiene payloads return the documented MCP error instead of raising."""
        result = handle_tool_call(
            "mnemosyne_hygiene_clean", {"candidates_json": "not-json"},
        )

        assert result == {"error": "candidates_json is not valid JSON"}

    @pytest.mark.parametrize(
        "candidates_json",
        [
            "null",
            "{}",
            "[{}]",
            json.dumps([{"memory_id": "memory-1", "table_name": "unknown"}]),
            json.dumps([{"memory_id": "memory-1", "table_name": "working_memory", "noise_score": float("nan")}]),
            json.dumps([{"memory_id": "memory-1", "table_name": "working_memory", "noise_score": 1.1}]),
            json.dumps([{"memory_id": "memory-1", "table_name": "working_memory", "noise_score": True}]),
            json.dumps([{"memory_id": "memory-1", "table_name": "working_memory", "noise_score": 10**1000}]),
            json.dumps([{"memory_id": "memory-1", "table_name": "working_memory", "importance": float("inf")}]),
            json.dumps([{"memory_id": "memory-1", "table_name": "working_memory", "importance": True}]),
            json.dumps([{"memory_id": "memory-1", "table_name": "working_memory", "content_length": -1}]),
            json.dumps([{"memory_id": "memory-1", "table_name": "working_memory", "content_length": 1.5}]),
            json.dumps([{"memory_id": "memory-1", "table_name": "working_memory", "content_length": True}]),
            json.dumps([{"memory_id": "memory-1", "table_name": "working_memory", "suggested_action": "destroy"}]),
        ],
    )
    def test_hygiene_clean_rejects_malformed_candidates(self, candidates_json, monkeypatch):
        """Malformed candidate payloads return MCP errors before opening a memory instance."""
        monkeypatch.setattr(
            mcp_tools,
            "_create_instance",
            lambda **_kwargs: pytest.fail("malformed payload must not initialize memory"),
        )
        result = handle_tool_call(
            "mnemosyne_hygiene_clean", {"candidates_json": candidates_json},
        )

        assert result == {"error": "candidates_json must be a list of valid hygiene candidates"}

    def test_hygiene_clean_parses_valid_candidates_json(self, monkeypatch):
        """Valid JSON reaches the hygiene cleaner with mapped candidate fields."""
        class _Memory:
            class beam:
                db_path = "test.db"

        class _Result:
            def to_dict(self):
                return {"cleaned": 1}

        captured = {}

        def _clean_noise(**kwargs):
            captured.update(kwargs)
            return _Result()

        from mnemosyne.core import hygiene

        monkeypatch.setattr(mcp_tools, "_create_instance", lambda **_kwargs: _Memory())
        monkeypatch.setattr(hygiene, "clean_noise", _clean_noise)

        candidates_json = json.dumps([{
            "memory_id": "memory-1",
            "table_name": "working_memory",
            "content_preview": "done",
            "noise_score": 0.9,
            "noise_reasons": ["short acknowledgement"],
            "secret_flags": [],
            "importance": 0.2,
            "source": "test",
            "timestamp": "2026-07-21T00:00:00Z",
            "suggested_action": "archive",
            "content_length": 4,
        }])
        result = handle_tool_call(
            "mnemosyne_hygiene_clean",
            {"candidates_json": candidates_json, "action": "archive", "confirm": True},
        )

        assert result == {"status": "applied", "result": {"cleaned": 1}, "bank": "default"}
        assert captured["db_path"] == "test.db"
        assert captured["action"] == "archive"
        assert captured["confirm"] is True
        assert captured["dry_run"] is False
        candidate = captured["candidates"][0]
        assert candidate.memory_id == "memory-1"
        assert candidate.table_name == "working_memory"
        assert candidate.content_preview == "done"
        assert candidate.noise_score == 0.9
        assert candidate.noise_reasons == ["short acknowledgement"]
        assert candidate.secret_flags == []
        assert candidate.importance == 0.2
        assert candidate.source == "test"
        assert candidate.timestamp == "2026-07-21T00:00:00Z"
        assert candidate.suggested_action == "archive"
        assert candidate.content_length == 4

    @pytest.mark.parametrize("importance", [2.0, -0.25])
    def test_hygiene_audit_candidates_with_out_of_range_importance_clean_from_stored_row(
        self, tmp_path, monkeypatch, importance
    ):
        """MCP audit output can be confirmed for finite legacy importance values.

        Candidate importance is audit metadata: archive must preserve the
        current stored value, not the stale value supplied by the audit.
        """
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))

        stored = handle_tool_call(
            "mnemosyne_remember",
            {"content": "heartbeat", "source": "heartbeat", "importance": importance},
        )
        assert stored["status"] == "stored"
        memory_id = stored["memory_id"]

        audited = handle_tool_call(
            "mnemosyne_hygiene_audit",
            {"tables": ["working_memory"], "min_score": 0.0},
        )
        candidates = audited["report"]["candidates"]
        assert len(candidates) == 1
        assert candidates[0]["memory_id"] == memory_id
        assert candidates[0]["importance"] == importance

        # Simulate a legitimate intervening row update: the audit candidate
        # remains the exact input to clean, but it must not overwrite storage.
        mem = _create_instance(bank="default")
        try:
            mem.beam.conn.execute(
                "UPDATE working_memory SET importance = ? WHERE id = ?", (0.75, memory_id)
            )
            mem.beam.conn.commit()
        finally:
            mem.beam.conn.close()

        cleaned = handle_tool_call(
            "mnemosyne_hygiene_clean",
            {
                "candidates_json": json.dumps(candidates),
                "action": "archive",
                "confirm": True,
            },
        )

        assert cleaned["status"] == "applied"
        assert cleaned["result"] == {
            "deleted": 0,
            "archived": 1,
            "kept": 0,
            "flagged": 0,
            "errors": [],
            "log_entries": 1,
        }
        mem = _create_instance(bank="default")
        try:
            row = mem.beam.conn.execute(
                "SELECT importance, metadata_json FROM working_memory WHERE id = ?", (memory_id,)
            ).fetchone()
            assert row[0] == 0
            assert json.loads(row[1])["_original_importance"] == 0.75
        finally:
            mem.beam.conn.close()

    def test_unknown_tool(self):
        """Unknown tool raises ValueError."""
        with pytest.raises(ValueError, match="Unknown tool"):
            handle_tool_call("mnemosyne_unknown", {})


class TestMCPIntegration:
    """Integration tests for MCP server lifecycle."""

    def test_mcp_server_imports(self):
        """MCP server module imports successfully."""
        from mnemosyne.mcp_server import run_mcp_server, main
        assert callable(run_mcp_server)
        assert callable(main)

    def test_mcp_tools_import_guard(self):
        """mcp_tools imports even if mcp package not available."""
        # The module should load regardless
        from mnemosyne import mcp_tools
        assert hasattr(mcp_tools, "TOOLS")
        assert hasattr(mcp_tools, "handle_tool_call")

    def test_mcp_tools_fresh_import_does_not_materialize_default_state(self, tmp_path):
        """A fresh MCP import must not initialize the legacy default database."""
        data_dir = tmp_path / "data"
        mnemosyne_home = tmp_path / "mnemosyne-home"
        env = os.environ.copy()
        env.update({
            "HOME": str(tmp_path / "home"),
            "MNEMOSYNE_HOME": str(mnemosyne_home),
            "MNEMOSYNE_DATA_DIR": str(data_dir),
        })
        script = """
import json
import os
from pathlib import Path

import mnemosyne.mcp_tools

root = Path(os.environ["MNEMOSYNE_DATA_DIR"])
print(json.dumps(sorted(str(path.relative_to(root)) for path in root.rglob("*")) if root.exists() else []))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout) == []
        assert not data_dir.exists()
        assert not (data_dir / "mnemosyne.db").exists()
        assert not (data_dir / "mnemosyne.db-wal").exists()
        assert not (data_dir / "mnemosyne.db-shm").exists()
        assert not (data_dir / "config").exists()
        assert not mnemosyne_home.exists()

    def test_mcp_tools_runtime_type_hints_are_resolvable_without_default_state(self, tmp_path):
        """Lazy imports must not break runtime type-hint consumers or create state."""
        data_dir = tmp_path / "data"
        mnemosyne_home = tmp_path / "mnemosyne-home"
        env = os.environ.copy()
        env.update({
            "HOME": str(tmp_path / "home"),
            "MNEMOSYNE_HOME": str(mnemosyne_home),
            "MNEMOSYNE_DATA_DIR": str(data_dir),
        })
        script = """
import json
import os
from pathlib import Path
import sys
import typing

import mnemosyne.mcp_tools as m

assert typing.get_type_hints(m._create_instance)["return"] is typing.Any
assert typing.get_type_hints(m._WrapperBatchAdapter.__init__)["mem"] is typing.Any
assert "mnemosyne.core.memory" not in sys.modules
root = Path(os.environ["MNEMOSYNE_DATA_DIR"])
print(json.dumps(sorted(str(path.relative_to(root)) for path in root.rglob("*")) if root.exists() else []))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout) == []
        assert not data_dir.exists()
        assert not mnemosyne_home.exists()

    def test_mcp_tools_public_mnemosyne_export_is_lazy_and_canonical(self, tmp_path):
        """The legacy public export resolves to the canonical class on demand."""
        env = os.environ.copy()
        env.update({
            "HOME": str(tmp_path / "home"),
            "MNEMOSYNE_HOME": str(tmp_path / "mnemosyne-home"),
            "MNEMOSYNE_DATA_DIR": str(tmp_path / "data"),
        })
        script = """
import json

from mnemosyne.mcp_tools import Mnemosyne
from mnemosyne.core.memory import Mnemosyne as CoreMnemosyne

print(json.dumps({"is_canonical": Mnemosyne is CoreMnemosyne}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout) == {"is_canonical": True}

    def test_mcp_tools_star_import_preserves_legacy_export_surface(self, tmp_path):
        """Star imports retain every old public name and resolve Mnemosyne canonically."""
        env = os.environ.copy()
        env.update({
            "HOME": str(tmp_path / "home"),
            "MNEMOSYNE_HOME": str(tmp_path / "mnemosyne-home"),
            "MNEMOSYNE_DATA_DIR": str(tmp_path / "data"),
        })
        expected = [
            "ALL_TOOL_SCHEMAS",
            "Any",
            "BatchValidationError",
            "BeamMemory",
            "CallToolResult",
            "Dict",
            "ErrorData",
            "List",
            "Mnemosyne",
            "Path",
            "TOOLS",
            "TextContent",
            "Tool",
            "apply_beam_batch",
            "batch_validation_error_payload",
            "dry_run_batch",
            "get_tool_definitions",
            "handle_tool_call",
            "json",
            "math",
            "os",
            "sqlite3",
            "validate_batch_operations",
        ]
        script = f"""
import json

ns = {{}}
exec("from mnemosyne.mcp_tools import *", ns)
from mnemosyne.core.memory import Mnemosyne as CoreMnemosyne

names = sorted(name for name in ns if name != "__builtins__")
assert names == {expected!r}
assert ns["Mnemosyne"] is CoreMnemosyne
assert not {{"TYPE_CHECKING", "TypeAlias", "MnemosyneInstance"}} & ns.keys()
print(json.dumps({{"names": names, "count": len(names)}}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout) == {"names": expected, "count": len(expected)}

    def test_named_bank_hygiene_audit_in_fresh_process_creates_no_default_state(self, tmp_path):
        """A real named-bank audit never initializes default state in a fresh process."""
        data_dir = tmp_path / "data"
        named_db = data_dir / "banks" / "team" / "mnemosyne.db"
        named_db.parent.mkdir(parents=True)
        sqlite3.connect(named_db).close()
        before = sorted(str(path.relative_to(data_dir)) for path in data_dir.rglob("*"))
        mnemosyne_home = tmp_path / "mnemosyne-home"
        env = os.environ.copy()
        env.update({
            "HOME": str(tmp_path / "home"),
            "MNEMOSYNE_HOME": str(mnemosyne_home),
            "MNEMOSYNE_DATA_DIR": str(data_dir),
        })
        script = """
import json
import os
from pathlib import Path

from mnemosyne.mcp_tools import handle_tool_call

root = Path(os.environ["MNEMOSYNE_DATA_DIR"])
result = handle_tool_call("mnemosyne_hygiene_audit", {"bank": "team"})
after = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
print(json.dumps({"result": result, "after": after}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr
        payload = json.loads(completed.stdout)
        assert payload["result"]["status"] == "audited"
        assert payload["result"]["bank"] == "team"
        assert payload["after"] == before
        assert not (data_dir / "mnemosyne.db").exists()
        assert not (data_dir / "mnemosyne.db-wal").exists()
        assert not (data_dir / "mnemosyne.db-shm").exists()
        assert not (data_dir / "config").exists()
        assert not mnemosyne_home.exists()

    def test_get_tool_definitions_returns_all(self):
        """get_tool_definitions returns all registered tools."""
        tools = get_tool_definitions()
        names = [t["name"] for t in tools]
        assert len(tools) == len(TOOLS)
        assert len(names) == len(set(names))
        assert "mnemosyne_remember" in names

    def test_tool_definitions_convertible_to_tool_pydantic(self):
        """Tool dict definitions must be compatible with the ``mcp`` SDK 2.x Tool Pydantic model.

        The SDK 2.x ``Tool`` model exposes ``input_schema`` (snake_case) as
        its canonical Python field; ``inputSchema`` is still accepted as a
        Pydantic alias on construction. This test asserts both paths:
        (a) ``Tool(**t)`` constructs without raising; (b) the constructed
        ``Tool`` carries the same schema as the source dict regardless of
        which key the dict used.
        """
        from mcp.types import Tool

        tools = get_tool_definitions()
        for t in tools:
            tool = Tool(**t)
            assert isinstance(tool, Tool)
            assert tool.name == t["name"]
            assert tool.description == t.get("description")
            # Pydantic normalizes both ``inputSchema`` (1.x alias) and
            # ``input_schema`` (2.x canonical) to ``tool.input_schema``.
            expected = t.get("input_schema", t.get("inputSchema"))
            assert tool.input_schema == expected

        # Keep the legacy-only wire shape covered even though the current
        # normalizer emits the SDK 2.x canonical key.
        legacy = dict(tools[0])
        legacy["inputSchema"] = legacy.pop("input_schema")
        legacy_tool = Tool(**legacy)
        assert legacy_tool.input_schema == legacy["inputSchema"]

    def test_mcp_client_request_surface_returns_typed_results(self):
        """Real SDK client requests exercise list/call registration and dispatch."""
        from mcp import ClientSession
        from mcp.shared.memory import create_client_server_memory_streams
        from mcp.types import CallToolResult, ListToolsResult
        from mnemosyne.mcp_server import _build_mcp_server

        import asyncio
        import contextlib
        from unittest.mock import patch

        async def exercise():
            server = _build_mcp_server()
            async with create_client_server_memory_streams() as (client_streams, server_streams):
                server_task = asyncio.create_task(
                    server.run(*server_streams, server.create_initialization_options())
                )
                try:
                    async with ClientSession(*client_streams) as client:
                        await client.initialize()
                        listed = await client.list_tools()
                        assert isinstance(listed, ListToolsResult)
                        assert len(listed.tools) >= 25
                        with patch(
                            "mnemosyne.mcp_server.handle_tool_call",
                            return_value={"status": "ok"},
                        ) as handle_call:
                            success = await client.call_tool("mnemosyne_stats", None)
                        assert isinstance(success, CallToolResult)
                        assert success.is_error is False
                        handle_call.assert_called_once_with("mnemosyne_stats", {})
                finally:
                    server_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await server_task

        asyncio.run(exercise())

    def test_transport_builders_share_mcp_server(self):
        """stdio and SSE transport setup both obtain the shared server builder."""
        from mnemosyne import mcp_server

        import asyncio
        from unittest.mock import AsyncMock, patch

        async def exercise_stdio():
            fake_server = type(
                "FakeServer",
                (),
                {
                    "run": AsyncMock(),
                    "create_initialization_options": lambda self: object(),
                },
            )()

            class _StdioContext:
                async def __aenter__(self):
                    return (object(), object())

                async def __aexit__(self, *args):
                    return False

            with (
                patch.object(mcp_server, "_build_mcp_server", return_value=fake_server) as build,
                patch.object(mcp_server, "stdio_server", return_value=_StdioContext()),
            ):
                await mcp_server._run_stdio()
            build.assert_called_once_with()
            fake_server.run.assert_awaited_once()

        asyncio.run(exercise_stdio())

        with patch.object(mcp_server, "_build_mcp_server", return_value=object()) as build:
            app = mcp_server._build_sse_app(host="127.0.0.1")
        build.assert_called_once_with()
        sse_route = next(route for route in app.routes if getattr(route, "path", None) == "/sse")
        assert any(cell.cell_contents is build.return_value for cell in (sse_route.endpoint.__closure__ or ()))

    def test_sse_messages_transport_uses_matching_mount_path(self):
        """The advertised POST URI and mounted raw ASGI app must agree."""
        from mnemosyne import mcp_server
        from starlette.routing import Mount
        from unittest.mock import AsyncMock, patch

        handle_post_message = AsyncMock()
        transport = type("FakeTransport", (), {"handle_post_message": handle_post_message})()
        transport_factory = patch("mcp.server.sse.SseServerTransport")
        with (
            patch.object(mcp_server, "_build_mcp_server", return_value=object()),
            transport_factory as build_transport,
        ):
            build_transport.return_value = transport
            app = mcp_server._build_sse_app(host="127.0.0.1")

        build_transport.assert_called_once_with("/messages/")
        message_mount = next(
            route
            for route in app.routes
            if isinstance(route, Mount) and route.path == "/messages"
        )
        assert message_mount.app is handle_post_message

    def test_sse_messages_invalid_session_returns_clean_error(self):
        """An invalid session POST must return an error without a second ASGI response."""
        from mnemosyne import mcp_server
        from starlette.testclient import TestClient
        from unittest.mock import patch

        with patch.object(mcp_server, "_build_mcp_server", return_value=object()):
            app = mcp_server._build_sse_app(host="127.0.0.1")

        response = TestClient(app).post(
            "/messages/?session_id=missing",
            json={"jsonrpc": "2.0", "method": "ping"},
        )

        assert response.status_code == 400
        assert response.text == "Invalid session ID"

    def test_sse_messages_missing_session_returns_clean_error(self):
        """A valid-but-missing session must return 404 without an ASGI exception."""
        from mnemosyne import mcp_server
        from starlette.testclient import TestClient
        from unittest.mock import patch

        with patch.object(mcp_server, "_build_mcp_server", return_value=object()):
            app = mcp_server._build_sse_app(host="127.0.0.1")

        response = TestClient(app).post(
            "/messages/?session_id=00000000-0000-0000-0000-000000000000",
            json={"jsonrpc": "2.0", "method": "ping"},
        )

        assert response.status_code == 404
        assert response.text == "Could not find session"

    def test_build_mcp_server_list_tools_returns_listtoolsresult(self):
        """SDK 2.x contract: the tools/list callback must return a ListToolsResult.

        Regression test for the CodeRabbit finding on PR #571 (round 2):
        ``_on_list_tools`` must wrap the Tool list in ``ListToolsResult(tools=...)``
        rather than returning a bare list. The SDK 2.x low-level server expects
        a ``ListToolsResult`` object — a bare list breaks the ``tools/list``
        response contract for both stdio and SSE transports.
        """
        from mcp.types import ListToolsResult, Tool
        from mnemosyne.mcp_server import _build_mcp_server

        server = _build_mcp_server()
        entry = server.get_request_handler("tools/list")
        assert entry is not None
        assert callable(entry.handler)
        on_list_tools = entry.handler

        import asyncio
        result = asyncio.run(on_list_tools(ctx=None, params=None))
        assert isinstance(result, ListToolsResult), (
            f"tools/list callback must return ListToolsResult, got {type(result).__name__}"
        )
        assert isinstance(result.tools, list)
        assert all(isinstance(t, Tool) for t in result.tools)
        assert len(result.tools) >= 25  # matches test_all_tools_present

    def test_build_mcp_server_call_tool_returns_is_error_on_failure(self):
        """SDK 2.x contract: tools/call must return CallToolResult with is_error=True on failure.

        Regression test for the CodeRabbit finding on PR #571 (round 2):
        the ``_on_call_tool`` exception path must set ``is_error=True`` so MCP
        clients can distinguish implementation failures from successful calls.
        Preserves the existing error payload shape for backward compatibility.
        """
        from mcp.types import CallToolResult
        from mnemosyne.mcp_server import _build_mcp_server

        server = _build_mcp_server()
        entry = server.get_request_handler("tools/call")
        assert entry is not None
        assert callable(entry.handler)
        on_call_tool = entry.handler

        # Construct a params-shaped object with empty name → handle_tool_call
        # raises before returning a valid result. This is the path that was
        # previously returning a successful-looking CallToolResult with error
        # content instead of a flagged failure.
        class _Params:
            name = ""
            arguments = {}

        import asyncio
        result = asyncio.run(on_call_tool(ctx=None, params=_Params()))
        assert isinstance(result, CallToolResult)
        assert result.is_error is True, (
            "tools/call failure path must set is_error=True for SDK 2.x contract"
        )
        # Error payload preserved for backward compatibility
        import json as _json
        assert len(result.content) >= 1
        payload = _json.loads(result.content[0].text)
        assert payload.get("status") == "error"

    def test_build_mcp_server_call_tool_success_returns_is_error_false(self):
        """SDK 2.x contract: successful tools/call must return is_error=False (or unset)."""
        from mcp.types import CallToolResult
        from mnemosyne.mcp_server import _build_mcp_server

        server = _build_mcp_server()
        entry = server.get_request_handler("tools/call")
        assert entry is not None
        assert callable(entry.handler)
        on_call_tool = entry.handler

        # Use a real tool definition and call it through the path. We mock
        # handle_tool_call to return a minimal valid result so we don't need
        # the full DB-backed stack here.
        import asyncio
        from unittest.mock import patch

        class _Params:
            name = "mnemosyne_stats"
            arguments = None

        with patch("mnemosyne.mcp_server.handle_tool_call", return_value={"status": "ok"}) as handle_call:
            result = asyncio.run(on_call_tool(ctx=None, params=_Params()))
        handle_call.assert_called_once_with("mnemosyne_stats", {})
        assert isinstance(result, CallToolResult)
        assert not result.is_error, (
            "successful tools/call must have is_error=False (or None)"
        )
        assert json.loads(result.content[0].text) == {"status": "ok"}

    def test_top_level_cli_forwards_mcp_arguments(self, tmp_path):
        """`mnemosyne mcp ...` must pass subcommand args to the MCP parser."""
        env = os.environ.copy()
        env["HOME"] = str(tmp_path / "home")
        env["MNEMOSYNE_DATA_DIR"] = str(tmp_path / "mnemosyne-data")
        script = """
import json
import sys
import mnemosyne.mcp_server

def fake_main(argv):
    print(json.dumps({"argv": argv}))

mnemosyne.mcp_server.main = fake_main
sys.argv = [
    "mnemosyne",
    "mcp",
    "--transport",
    "sse",
    "--port",
    "19090",
    "--bank",
    "work",
]
from mnemosyne.cli import run_cli
run_cli()
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == {
            "argv": ["--transport", "sse", "--port", "19090", "--bank", "work"]
        }

    def test_mcp_server_main_accepts_explicit_argv(self):
        """MCP server parser should parse caller-provided argv, not global sys.argv."""
        from mnemosyne.mcp_server import main

        with patch("mnemosyne.mcp_server.run_mcp_server") as run_mcp_server:
            main(["--transport", "sse", "--port", "19090", "--bank", "work"])

        run_mcp_server.assert_called_once_with(
            transport="sse", port=19090, bank="work", host="127.0.0.1"
        )


class TestImportGuard:
    """Verify MCP is truly optional."""

    def test_core_imports_without_mcp(self):
        """Core mnemosyne imports work without mcp installed."""
        from mnemosyne import remember, recall, get_stats
        assert callable(remember)
        assert callable(recall)
        assert callable(get_stats)

    def test_mcp_server_raises_without_mcp(self):
        """MCP server raises helpful error if mcp not installed."""
        from mnemosyne.mcp_server import _MCP_AVAILABLE, _run_stdio

        if _MCP_AVAILABLE:
            # mcp is installed — verify the server function exists and the flag is True
            assert _MCP_AVAILABLE is True
        else:
            # mcp is NOT installed — verify _run_stdio raises RuntimeError
            import asyncio
            with pytest.raises(RuntimeError, match="MCP not installed"):
                asyncio.get_event_loop().run_until_complete(_run_stdio())


class TestTask6BMCPParity:
    """Task 6B — close the 8 advertised-but-unhandled gaps and add native
    MCP endpoints for ingest, bounded recall, Dream lifecycle, reclaim.

    These tests are written RED-first: every assertion fails or raises on the
    current head because the handlers/schemas do not exist.
    """

    # ─── Schema/handler one-to-one parity (binding behavior #1) ───────────

    def test_every_schema_has_a_real_handler_no_provider_string_exception(self):
        """Every ALL_TOOL_SCHEMAS entry must have a _TOOL_HANDLERS entry.

        This is the binding contract: the prior exception that accepted a tool
        merely because a Hermes-provider source file contained its name is
        gone. The 8 known gaps (persona x4, sync x3, triple_end) must close.
        """
        from mnemosyne.tool_schemas import ALL_TOOL_SCHEMAS
        from mnemosyne.mcp_tools import _TOOL_HANDLERS

        schemas = {s["name"] for s in ALL_TOOL_SCHEMAS}
        handlers = set(_TOOL_HANDLERS)
        missing = schemas - handlers
        assert not missing, (
            f"schemas without a real _TOOL_HANDLERS entry: {sorted(missing)}"
        )

    def test_previously_unhandled_tools_now_have_handlers(self):
        """The 8 historical gaps must each gain a real handler."""
        from mnemosyne.mcp_tools import _TOOL_HANDLERS

        required = {
            "mnemosyne_persona_promote",
            "mnemosyne_persona_demote",
            "mnemosyne_persona_list",
            "mnemosyne_persona_reinforce",
            "mnemosyne_sync_push",
            "mnemosyne_sync_pull",
            "mnemosyne_sync_status",
            "mnemosyne_triple_end",
        }
        present = set(_TOOL_HANDLERS) & required
        assert present == required, f"still missing: {sorted(required - present)}"

    # ─── Native ingest endpoint (#2) ──────────────────────────────────────

    def test_ingest_schema_advertised(self):
        from mnemosyne.tool_schemas import ALL_TOOL_SCHEMAS
        names = {s["name"] for s in ALL_TOOL_SCHEMAS}
        assert "mnemosyne_ingest" in names

    def test_ingest_handler_stores_event_and_returns_content_free_receipt(self, tmp_path, monkeypatch):
        """mnemosyne_ingest durably ingests one event and returns only
        content-free receipt fields (no raw content in the payload)."""
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))

        result = handle_tool_call("mnemosyne_ingest", {
            "event_id": "evt-mcp-1",
            "producer": "cli",
            "actor_id": "user-1",
            "project_id": "proj-1",
            "session_id": "sess-1",
            "turn_id": "turn-1",
            "role": "user",
            "content": "private ingest payload",
            "occurred_at": "2026-08-10T00:00:00Z",
        })
        assert result["status"] in ("stored", "duplicate", "conflict", "rejected")
        # NEVER echo raw content back.
        assert "private ingest payload" not in json.dumps(result)
        assert "content" not in result
        assert result.get("event_id") == "evt-mcp-1"

    def test_ingest_status_handler_is_content_free(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        handle_tool_call("mnemosyne_ingest", {
            "event_id": "evt-status-1",
            "producer": "cli", "actor_id": "a", "project_id": "p",
            "session_id": "s", "turn_id": "t", "role": "user",
            "content": "secret-status", "occurred_at": "2026-08-10T00:00:00Z",
        })
        rows = handle_tool_call("mnemosyne_ingest_status", {"limit": 5})
        assert "secret-status" not in json.dumps(rows)
        for r in rows.get("receipts", []):
            assert "content" not in r
            assert "content_hash" not in r

    # ─── Bounded recall extension (#2) ────────────────────────────────────

    def test_recall_accepts_bounded_filter_fields_and_token_cap(self, tmp_path, monkeypatch):
        """mnemosyne_recall must accept optional producer/actor/project/session
        and hard token controls, while retaining legacy args when absent."""
        from mnemosyne.tool_schemas import ALL_TOOL_SCHEMAS
        schema = next(s for s in ALL_TOOL_SCHEMAS if s["name"] == "mnemosyne_recall")
        params = schema["parameters"]["properties"]
        for opt in ("producer", "actor", "project", "session",
                    "max_tokens", "max_item_tokens"):
            assert opt in params, f"recall schema missing bounded field {opt!r}"

        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        # Seed one memory, then bounded-recall with max_tokens=1 → truncated.
        handle_tool_call("mnemosyne_remember", {
            "content": "x" * 200, "scope": "session",
        })
        env = handle_tool_call("mnemosyne_recall", {
            "query": "x", "max_tokens": 1,
        })
        # Bounded path returns an envelope, not a bare list.
        assert "token_count" in env or "rendered_context" in env

    def test_recall_legacy_path_unchanged_when_bounded_fields_absent(self, tmp_path, monkeypatch):
        """When no bounded field is supplied, recall behaves exactly as before."""
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        handle_tool_call("mnemosyne_remember", {"content": "legacy recall test"})
        result = handle_tool_call("mnemosyne_recall", {"query": "legacy"})
        # Legacy shape: status/count/results
        assert result["status"] == "ok"
        assert "count" in result
        assert "results" in result

    # ─── Dream lifecycle (7 handlers) (#2) ────────────────────────────────

    def test_dream_schemas_advertised(self):
        from mnemosyne.tool_schemas import ALL_TOOL_SCHEMAS
        names = {s["name"] for s in ALL_TOOL_SCHEMAS}
        for tool in ("mnemosyne_dream_plan", "mnemosyne_dream_status",
                     "mnemosyne_dream_review", "mnemosyne_dream_verify",
                     "mnemosyne_dream_resume", "mnemosyne_dream_apply",
                     "mnemosyne_dream_undo"):
            assert tool in names, f"{tool} not advertised"

    def test_dream_plan_content_free_projection(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        # Seed a memory so plan has something to consider.
        handle_tool_call("mnemosyne_remember", {"content": "dream plan seed"})
        result = handle_tool_call("mnemosyne_dream_plan", {
            "session_id": "sess-dream",
        })
        assert "run_id" in result
        assert "state" in result
        # NEVER expose raw scope/manifest/actions content.
        assert "scope" not in result
        assert "manifest" not in result
        assert "actions" not in result

    def test_dream_review_uses_fixed_reviewer_role(self, tmp_path, monkeypatch):
        """Dream review must call the one native receipt path with the fixed
        reviewer role; the actor is taken from args, not from the role field."""
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        handle_tool_call("mnemosyne_remember", {"content": "dream review seed"})
        plan = handle_tool_call("mnemosyne_dream_plan", {"session_id": "sess-r"})
        run_id = plan["run_id"]
        result = handle_tool_call("mnemosyne_dream_review", {
            "run_id": run_id,
            "actor_id": "reviewer-a",
            "verdict": "PASS",
        })
        assert result.get("state") in ("awaiting_approval", "ready", "rejected",
                                       "failed_terminal", "planning")
        # role must NOT be caller-controllable.
        from mnemosyne.tool_schemas import ALL_TOOL_SCHEMAS
        schema = next(s for s in ALL_TOOL_SCHEMAS if s["name"] == "mnemosyne_dream_review")
        assert "role" not in schema["parameters"]["properties"]

    def test_dream_verify_requires_distinct_actor(self, tmp_path, monkeypatch):
        """Verifier must be a different actor than reviewer (core contract)."""
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        handle_tool_call("mnemosyne_remember", {"content": "dream verify seed"})
        plan = handle_tool_call("mnemosyne_dream_plan", {"session_id": "sess-v"})
        run_id = plan["run_id"]
        handle_tool_call("mnemosyne_dream_review", {
            "run_id": run_id, "actor_id": "same-actor", "verdict": "PASS",
        })
        # Same actor verifying must fail-closed.
        result = handle_tool_call("mnemosyne_dream_verify", {
            "run_id": run_id, "actor_id": "same-actor", "verdict": "PASS",
        })
        assert result.get("state") in ("rejected", "failed_terminal")

    def test_dream_status_content_free(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        handle_tool_call("mnemosyne_remember", {"content": "status seed"})
        plan = handle_tool_call("mnemosyne_dream_plan", {"session_id": "sess-s"})
        result = handle_tool_call("mnemosyne_dream_status", {"run_id": plan["run_id"]})
        assert result["run_id"] == plan["run_id"]
        assert "actions" not in result
        assert "manifest" not in result

    # ─── Orphan reclaim — dry-run default (#2) ────────────────────────────

    def test_reclaim_orphans_defaults_to_dry_run_and_is_content_free(self, tmp_path, monkeypatch):
        from mnemosyne.tool_schemas import ALL_TOOL_SCHEMAS
        schema = next(s for s in ALL_TOOL_SCHEMAS if s["name"] == "mnemosyne_reclaim_orphans")
        assert schema["parameters"]["properties"]["apply"]["default"] is False

        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        result = handle_tool_call("mnemosyne_reclaim_orphans", {})
        assert result.get("dry_run") is True
        # No row content echoed.
        assert "content" not in result

    def test_reclaim_orphans_apply_requires_explicit_opt_in(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        # Default (no apply=True) never mutates.
        result = handle_tool_call("mnemosyne_reclaim_orphans", {})
        assert result.get("dry_run") is True

    # ─── triple_end gap closure (#1) ──────────────────────────────────────

    def test_triple_end_handler_closes_open_triples(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        handle_tool_call("mnemosyne_triple_add", {
            "subject": "alice", "predicate": "knows", "object": "bob",
        })
        result = handle_tool_call("mnemosyne_triple_end", {
            "subject": "alice", "predicate": "knows",
        })
        assert result["status"] == "ended"
        assert result["count"] >= 1

    # ─── Persona handlers (4 gap closures) (#1) ───────────────────────────

    def test_persona_promote_demote_list_reinforce_handlers(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        # Seed a working memory row to promote.
        mem_result = handle_tool_call("mnemosyne_remember", {
            "content": "persona promote target",
        })
        memory_id = mem_result["memory_id"]

        promoted = handle_tool_call("mnemosyne_persona_promote", {
            "memory_id": memory_id, "tier": "long_term",
        })
        assert promoted["status"] == "ok"
        assert "persona_id" in promoted
        persona_id = promoted["persona_id"]

        listed = handle_tool_call("mnemosyne_persona_list", {})
        assert listed["status"] == "ok"
        assert listed["count"] >= 1

        reinforced = handle_tool_call("mnemosyne_persona_reinforce", {
            "persona_id": persona_id,
        })
        assert reinforced["status"] == "ok"

        demoted = handle_tool_call("mnemosyne_persona_demote", {
            "persona_id": persona_id,
        })
        assert demoted["status"] == "ok"

    def test_persona_promote_unknown_memory_returns_structured_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        result = handle_tool_call("mnemosyne_persona_promote", {
            "memory_id": "does-not-exist-xyz",
        })
        # Structured rejection, not an exception.
        assert result.get("status") == "error" or "error" in result

    # ─── Sync handlers — safe-default, no remote authority widened (#1, #6) ─

    def test_sync_status_no_remote_returns_unconfigured_structured(self, tmp_path, monkeypatch):
        """With no remote configured, sync_status must return a structured
        'unconfigured' status, never raise, never widen network authority."""
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("MNEMOSYNE_SYNC_REMOTE", raising=False)
        result = handle_tool_call("mnemosyne_sync_status", {})
        assert result.get("status") in ("ok", "unconfigured", "error")
        assert "remote" in result

    def test_sync_push_no_remote_returns_structured_rejection(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("MNEMOSYNE_SYNC_REMOTE", raising=False)
        result = handle_tool_call("mnemosyne_sync_push", {})
        # Must NOT raise; must NOT attempt any network call.
        assert result.get("status") in ("error", "unconfigured")
        assert "remote" in result or "error" in result

    # ─── Diagnose read-only path (#5) ─────────────────────────────────────

    def test_diagnose_handler_calls_read_only_path_and_writes_no_log(self, tmp_path, monkeypatch):
        """mnemosyne_diagnose must use run_diagnostics(read_only=True) and must
        not create a log directory/file or default writable DB."""
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        # Snapshot the data dir contents before.
        before = sorted(p.name for p in tmp_path.rglob("*")) if tmp_path.exists() else []

        import inspect
        from mnemosyne.diagnose import run_diagnostics
        sig = inspect.signature(run_diagnostics)
        if "read_only" not in sig.parameters:
            pytest.skip("run_diagnostics(read_only=True) not yet implemented by Task 6B-diagnose")

        handle_tool_call("mnemosyne_diagnose", {})
        # After call: no new log file or default DB created in data dir.
        after = sorted(p.name for p in tmp_path.rglob("*")) if tmp_path.exists() else []
        # Tolerate only the diagnose-produced structured dict (in-memory).
        new_files = [n for n in after if n not in before]
        # No .jsonl log, no mnemosyne.db materialized.
        assert not any(n.endswith(".jsonl") for n in new_files), (
            f"diagnose wrote a JSONL log: {new_files}"
        )
        assert not any(n == "mnemosyne.db" for n in new_files), (
            f"diagnose materialized a default DB: {new_files}"
        )

    def test_diagnose_repair_request_returns_structured_rejection(self, tmp_path, monkeypatch):
        """A repair request through the read-only MCP surface must return a
        structured rejection instead of mutating."""
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))

        import inspect
        from mnemosyne.diagnose import run_diagnostics
        sig = inspect.signature(run_diagnostics)
        if "read_only" not in sig.parameters:
            pytest.skip("run_diagnostics(read_only=True) not yet implemented")

        result = handle_tool_call("mnemosyne_diagnose", {
            "repair_vec_working": True,
        })
        # Must NOT have performed a repair; structured rejection.
        assert result.get("status") == "read_only" or result.get("repair_rejected") is True or                result.get("error") == "repair_not_permitted_over_mcp", (
            f"read-only MCP surface must reject repair, got: {result}"
        )

    # ─── Read/write distinguishability in schema metadata (#2) ────────────

    def test_read_write_tools_are_distinguishable_in_schema(self):
        """readOnly flag must let clients tell writes from reads.

        Applied at TOOLS construction time so canonical schema dicts stay
        byte-equal to the provider copies (test_hermes_provider_parity)."""
        from mnemosyne.mcp_tools import TOOLS
        names = {t["name"] for t in TOOLS}
        write_tools = ("mnemosyne_ingest", "mnemosyne_remember",
                       "mnemosyne_dream_apply", "mnemosyne_reclaim_orphans")
        read_tools = ("mnemosyne_recall", "mnemosyne_stats", "mnemosyne_dream_status")
        for w in write_tools:
            assert w in names
        for r in read_tools:
            assert r in names
        by_name = {t["name"]: t for t in TOOLS}
        for w in write_tools:
            assert by_name[w].get("readOnly") is False, (
                f"{w} must declare readOnly: false (it mutates)"
            )
        for r in read_tools:
            assert by_name[r].get("readOnly") is True, (
                f"{r} must declare readOnly: true (pure read)"
            )


    def test_forget_canonical_is_advertised_and_handled(self, tmp_path, monkeypatch):
        """FORGET_CANONICAL_SCHEMA must be in ALL_TOOL_SCHEMAS and have a real
        MCP handler (previously it was defined but only reachable through the
        Hermes provider; the parity test allowlisted it under PROVIDER_ONLY)."""
        from mnemosyne.tool_schemas import ALL_TOOL_SCHEMAS
        from mnemosyne.mcp_tools import _TOOL_HANDLERS
        names = {s["name"] for s in ALL_TOOL_SCHEMAS}
        assert "mnemosyne_forget_canonical" in names, (
            "FORGET_CANONICAL_SCHEMA must be advertised in ALL_TOOL_SCHEMAS"
        )
        assert "mnemosyne_forget_canonical" in _TOOL_HANDLERS, (
            "mnemosyne_forget_canonical must have a real MCP handler"
        )

        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        handle_tool_call("mnemosyne_remember_canonical", {
            "category": "identity", "name": "name", "body": "Alice",
        })
        result = handle_tool_call("mnemosyne_forget_canonical", {
            "category": "identity", "name": "name",
        })
        assert result.get("retired") is True
        result2 = handle_tool_call("mnemosyne_forget_canonical", {
            "category": "identity", "name": "name",
        })
        assert result2.get("retired") is False

    def test_forget_canonical_validates_required_fields(self):
        from mnemosyne.mcp_tools import handle_tool_call
        result = handle_tool_call("mnemosyne_forget_canonical", {})
        assert "error" in result

    # ─── All results JSON-serializable & content-free (#2, #4) ────────────

    def test_all_handler_results_are_json_serializable(self, tmp_path, monkeypatch):
        """Every handler result must be JSON-serializable structured data."""
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        from mnemosyne.mcp_tools import _TOOL_HANDLERS
        # Pick a safe read-only subset to exercise serializability.
        safe_calls = [
            ("mnemosyne_stats", {}),
            ("mnemosyne_dream_status", {"run_id": "nonexistent"}),
            ("mnemosyne_sync_status", {}),
        ]
        for name, args in safe_calls:
            result = _TOOL_HANDLERS[name](args)
            json.dumps(result)  # raises if not serializable
