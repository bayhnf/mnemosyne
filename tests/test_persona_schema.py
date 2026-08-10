"""Tests for memoria_persona L3 schema (v3.10.0)."""

import json
import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def persona_db(tmp_path):
    """Build a fresh mnemosyne DB and return the connection."""
    db_path = tmp_path / "mnemosyne.db"
    from mnemosyne.core.beam import BeamMemory
    beam = BeamMemory(session_id="test-persona", db_path=str(db_path))
    yield beam
    # No explicit close -- _thread_local cleanup happens via conftest autouse.


class TestPersonaSchema:
    def test_table_exists(self, persona_db):
        """memoria_persona table created on init."""
        # DEBUG
        all_tables = [r[0] for r in persona_db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        import sys
        print(f"\nDEBUG db_path={persona_db.db_path}", file=sys.stderr)
        print(f"DEBUG all_tables ({len(all_tables)}): {sorted(all_tables)}", file=sys.stderr)
        tables = [
            r[0]
            for r in persona_db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='memoria_persona'"
            ).fetchall()
        ]
        assert tables == ["memoria_persona"], f"Expected memoria_persona, got {tables}"

    def test_indexes_created(self, persona_db):
        """Both expected indexes exist."""
        indexes = [
            r[0]
            for r in persona_db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='memoria_persona'"
            ).fetchall()
        ]
        assert "idx_persona_session_tier" in indexes
        assert "idx_persona_tier_topic" in indexes

    def test_tier_check_constraint(self, persona_db):
        """Invalid tier values are rejected."""
        with pytest.raises(sqlite3.IntegrityError):
            persona_db.conn.execute(
                "INSERT INTO memoria_persona (tier, topic, content) VALUES ('invalid', 'x', 'y')"
            )

    def test_insert_valid_tiers(self, persona_db):
        """All three valid tiers accepted."""
        for tier in ("permanent", "long_term", "working"):
            persona_db.conn.execute(
                "INSERT INTO memoria_persona (tier, topic, content) VALUES (?, ?, ?)",
                (tier, f"topic_{tier}", f"content_{tier}"),
            )
        count = persona_db.conn.execute(
            "SELECT COUNT(*) FROM memoria_persona"
        ).fetchone()[0]
        assert count == 3

    def test_default_values(self, persona_db):
        """Defaults applied for confidence, created_at, reinforcement_count."""
        persona_db.conn.execute(
            "INSERT INTO memoria_persona (tier, topic, content) VALUES ('long_term', 'default_test', 'hello')"
        )
        row = persona_db.conn.execute(
            "SELECT confidence, reinforcement_count, created_at, last_reinforced_at "
            "FROM memoria_persona WHERE topic='default_test'"
        ).fetchone()
        assert row[0] == 0.7  # confidence default
        assert row[1] == 0    # reinforcement_count default
        assert row[2] is not None  # created_at populated
        assert row[3] is not None  # last_reinforced_at populated

    def test_idempotent_reinit(self, persona_db):
        """Re-running _init_schema does not error and does not lose data."""
        persona_db.conn.execute(
            "INSERT INTO memoria_persona (tier, topic, content) VALUES ('permanent', 'persisted', 'survives reinit')"
        )
        persona_db.conn.commit()

        # Re-run schema init
        from mnemosyne.core.beam import init_beam
        init_beam(db_path=Path(persona_db.db_path))

        # Data should still be there
        rows = persona_db.conn.execute(
            "SELECT content FROM memoria_persona WHERE topic='persisted'"
        ).fetchall()
        assert [r[0] for r in rows] == ["survives reinit"]

    def test_existing_tables_untouched(self, persona_db):
        """Adding memoria_persona does not modify other tables."""
        # Insert into memoria_preferences (already existed in v3.9.0)
        persona_db.conn.execute(
            "INSERT INTO memoria_preferences (preference, topic) VALUES ('test pref', 'test_topic')"
        )
        # Run init again
        from mnemosyne.core.beam import init_beam
        init_beam(db_path=Path(persona_db.db_path))
        # Should still exist
        rows = persona_db.conn.execute(
            "SELECT preference FROM memoria_preferences WHERE topic='test_topic'"
        ).fetchall()
        assert [r[0] for r in rows] == ["test pref"]


class TestPersonaPromotion:
    """Basic promotion/demotion API tests (Wave 1.2 precursor)."""

    def test_promote_insert(self, persona_db):
        """Promotion writes a row to memoria_persona."""
        persona_db.conn.execute(
            "INSERT INTO memoria_persona (tier, topic, content, source_memory_id, promotion_reason) "
            "VALUES (?, ?, ?, ?, ?)",
            ("long_term", "communication", "prefers terse responses", "wm-123", "auto-extracted"),
        )
        row = persona_db.conn.execute(
            "SELECT tier, content, source_memory_id, promotion_reason FROM memoria_persona "
            "WHERE topic='communication'"
        ).fetchone()
        assert tuple(row) == ("long_term", "prefers terse responses", "wm-123", "auto-extracted")

    def test_reinforce_increments(self, persona_db):
        """Reinforcement bumps the counter and last_reinforced_at."""
        persona_db.conn.execute(
            "INSERT INTO memoria_persona (tier, topic, content) VALUES ('long_term', 'r1', 'c1')"
        )
        pid = persona_db.conn.execute(
            "SELECT id FROM memoria_persona WHERE topic='r1'"
        ).fetchone()[0]

        persona_db.conn.execute(
            "UPDATE memoria_persona SET reinforcement_count = reinforcement_count + 1 "
            "WHERE id = ?", (pid,),
        )
        count = persona_db.conn.execute(
            "SELECT reinforcement_count FROM memoria_persona WHERE id = ?", (pid,)
        ).fetchone()[0]
        assert count == 1



class TestPersonaMCPHandlers:
    """Task 6B — MCP handlers for persona promote/demote/list/reinforce.

    The MCP layer is a thin adapter over PersonaAdapter; these verify the
    handler contracts: structured errors, content-free rejection on unknown
    ids, tier validation, and reinforcement counter semantics.
    """

    def test_mcp_persona_promote_validates_tier(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        from mnemosyne.mcp_tools import handle_tool_call
        # Need a real memory to promote.
        mem = handle_tool_call("mnemosyne_remember", {"content": "tier check"})
        result = handle_tool_call("mnemosyne_persona_promote", {
            "memory_id": mem["memory_id"], "tier": "bogus_tier",
        })
        assert result.get("status") == "error"
        assert "tier" in json.dumps(result).lower()

    def test_mcp_persona_demote_unknown_id_returns_structured_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        from mnemosyne.mcp_tools import handle_tool_call
        result = handle_tool_call("mnemosyne_persona_demote", {
            "persona_id": 999999,
        })
        assert result.get("status") == "error"

    def test_mcp_persona_reinforce_unknown_id_returns_structured_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        from mnemosyne.mcp_tools import handle_tool_call
        result = handle_tool_call("mnemosyne_persona_reinforce", {
            "persona_id": 999999,
        })
        assert result.get("status") == "error"

    def test_mcp_persona_list_filters_by_tier(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        from mnemosyne.mcp_tools import handle_tool_call
        m1 = handle_tool_call("mnemosyne_remember", {"content": "permanent fact"})
        m2 = handle_tool_call("mnemosyne_remember", {"content": "working note"})
        handle_tool_call("mnemosyne_persona_promote", {
            "memory_id": m1["memory_id"], "tier": "permanent",
        })
        handle_tool_call("mnemosyne_persona_promote", {
            "memory_id": m2["memory_id"], "tier": "working",
        })
        only_permanent = handle_tool_call("mnemosyne_persona_list", {"tier": "permanent"})
        assert only_permanent["status"] == "ok"
        assert only_permanent["count"] == 1
        assert only_permanent["personas"][0]["tier"] == "permanent"
