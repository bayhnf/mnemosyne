"""Task 6B round-2 re-review: precise failing tests for I1 and I2.

I1: config.yaml-only sync_remote must actually drive push/pull/status through
    the SyncAdapter with identical precedence/credentials/remote resolution.
I2: read-only ingest_status / persona_list must not crash or leak raw
    exception text on corrupt/malformed/missing-table databases.
"""
import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from mnemosyne.mcp_tools import handle_tool_call


def _fresh_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))


# ===========================================================================
# I1: config.yaml-only sync remote must drive the ACTUAL SyncAdapter push/pull
# path. Tests exercise the REAL hermes_memory_provider.sync_adapter.SyncAdapter;
# only its no-network transport boundary (_http_post) is monkeypatched to
# record self.remote / self.auth_token and return deterministic payloads.
# ===========================================================================

def _seed_sync_event(tmp_path):
    """Insert a row into memory_events so SyncEngine.pull_changes returns a
    non-empty event list, causing the push path to call _http_post.

    Builds a SyncAdapter once first so _ensure_tables runs and adds the
    migration columns (timestamp_epoch etc.) before we INSERT."""
    from mnemosyne.core.config import MnemosyneConfig
    from mnemosyne.mcp_tools import _create_instance, _sync_adapter_config_from_yaml
    MnemosyneConfig.reset_instance()
    mem = _create_instance(bank="default")
    cfg = _sync_adapter_config_from_yaml()
    if not cfg.get("remote"):
        cfg["remote"] = "https://seed-only.example.test:1"
    from hermes_memory_provider.sync_adapter import SyncAdapter
    SyncAdapter(mem.beam, config=cfg)
    import sqlite3
    conn = sqlite3.connect(tmp_path / "mnemosyne.db")
    has_epoch = "timestamp_epoch" in {
        r[1] for r in conn.execute("PRAGMA table_info(memory_events)").fetchall()
    }
    cols = "(event_id, memory_id, operation, timestamp, device_id, apply_state)"
    vals = "('evt-test-1', 'mem-test-1', 'CREATE', '2026-08-10T00:00:00', 'test-device', 'applied')"
    if has_epoch:
        cols = "(event_id, memory_id, operation, timestamp, timestamp_epoch, device_id, apply_state)"
        vals = "('evt-test-1', 'mem-test-1', 'CREATE', '2026-08-10T00:00:00', 1754764800.0, 'test-device', 'applied')"
    conn.execute(f"INSERT INTO memory_events {cols} VALUES {vals}")
    conn.commit()
    conn.close()
    MnemosyneConfig.reset_instance()


@pytest.fixture
def _record_transport(monkeypatch):
    """Monkeypatch the REAL SyncAdapter._http_post to record the instance's
    resolved remote/auth and return deterministic payloads. No fake class."""
    calls = []

    def _fake_http_post(self, path, payload):
        calls.append({
            "path": path,
            "remote": self.remote,
            "auth_token": getattr(self, "auth_token", ""),
            "encrypt_enabled": getattr(self, "encrypt_enabled", False),
            "encryption_key": getattr(self, "encryption_key", ""),
        })
        if path == "/sync/push":
            return {"status": "ok", "accepted": 0, "next_cursor": "c1"}
        if path == "/sync/pull":
            return {"status": "ok", "events": [], "next_cursor": "c1"}
        return {"status": "ok"}

    import hermes_memory_provider.sync_adapter as sa
    monkeypatch.setattr(sa.SyncAdapter, "_http_post", _fake_http_post)
    yield calls
    calls.clear()


def test_sync_push_config_yaml_only_reaches_real_adapter_transport(monkeypatch, tmp_path, _record_transport):
    """config.yaml sync_remote must drive the REAL SyncAdapter push path: the
    adapter must reach _http_post with self.remote set from config.yaml."""
    _fresh_env(monkeypatch, tmp_path)
    for v in ("MNEMOSYNE_SYNC_REMOTE", "MNEMOSYNE_SYNC_HOST", "MNEMOSYNE_SYNC_PORT"):
        monkeypatch.delenv(v, raising=False)
    (tmp_path / "config.yaml").write_text(
        "sync_remote: https://cfg-push.example.test:8765\n"
    )
    handle_tool_call("mnemosyne_remember", {"content": "seed"})
    _seed_sync_event(tmp_path)

    result = handle_tool_call("mnemosyne_sync_push", {})

    # The real adapter must NOT have returned "No remote configured".
    assert "No remote configured" not in result.get("error", ""), (
        f"config.yaml-only push did not reach the real adapter transport: {result}"
    )
    # The real adapter must have called _http_post with the config.yaml remote.
    push_calls = [c for c in _record_transport if c["path"] == "/sync/push"]
    assert push_calls, (
        "real SyncAdapter never called _http_post for push; it likely saw no remote"
    )
    assert push_calls[-1]["remote"] == "https://cfg-push.example.test:8765", (
        f"real adapter remote={push_calls[-1]['remote']!r}; "
        "config.yaml sync_remote did not reach the transport"
    )


def test_sync_pull_config_yaml_only_reaches_real_adapter_transport(monkeypatch, tmp_path, _record_transport):
    """config.yaml sync_remote must drive the REAL SyncAdapter pull path."""
    _fresh_env(monkeypatch, tmp_path)
    for v in ("MNEMOSYNE_SYNC_REMOTE", "MNEMOSYNE_SYNC_HOST", "MNEMOSYNE_SYNC_PORT"):
        monkeypatch.delenv(v, raising=False)
    (tmp_path / "config.yaml").write_text(
        "sync_remote: https://cfg-pull.example.test:8765\n"
    )
    handle_tool_call("mnemosyne_remember", {"content": "seed"})

    result = handle_tool_call("mnemosyne_sync_pull", {})

    assert "No remote configured" not in result.get("error", ""), (
        f"config.yaml-only pull did not reach the real adapter transport: {result}"
    )
    pull_calls = [c for c in _record_transport if c["path"] == "/sync/pull"]
    assert pull_calls, (
        "real SyncAdapter never called _http_post for pull; it likely saw no remote"
    )
    assert pull_calls[-1]["remote"] == "https://cfg-pull.example.test:8765", (
        f"real adapter remote={pull_calls[-1]['remote']!r}"
    )


def test_sync_status_config_yaml_only_emits_resolved_remote(monkeypatch, tmp_path, _record_transport):
    """config.yaml sync_remote must appear in the REAL SyncAdapter status output
    (status reads self.remote directly, no _http_post call)."""
    _fresh_env(monkeypatch, tmp_path)
    for v in ("MNEMOSYNE_SYNC_REMOTE", "MNEMOSYNE_SYNC_HOST", "MNEMOSYNE_SYNC_PORT"):
        monkeypatch.delenv(v, raising=False)
    (tmp_path / "config.yaml").write_text(
        "sync_remote: https://cfg-status.example.test:8765\n"
    )
    handle_tool_call("mnemosyne_remember", {"content": "seed"})

    result = handle_tool_call("mnemosyne_sync_status", {})
    assert "cfg-status.example.test" in result.get("remote", ""), (
        f"real adapter status did not emit config.yaml remote: {result}"
    )
    assert result.get("status") == "ok", (
        f"real adapter status should be ok with a configured remote: {result}"
    )


def test_sync_env_remote_wins_over_config_yaml(monkeypatch, tmp_path, _record_transport):
    """MNEMOSYNE_SYNC_REMOTE env must win over config.yaml sync_remote in the
    REAL adapter resolution (env > config dict via _string)."""
    _fresh_env(monkeypatch, tmp_path)
    (tmp_path / "config.yaml").write_text(
        "sync_remote: https://from-config.example.test:8765\n"
    )
    monkeypatch.setenv("MNEMOSYNE_SYNC_REMOTE", "https://from-env.example.test:9999")
    handle_tool_call("mnemosyne_remember", {"content": "seed"})
    _seed_sync_event(tmp_path)

    handle_tool_call("mnemosyne_sync_push", {})

    push_calls = [c for c in _record_transport if c["path"] == "/sync/push"]
    assert push_calls, "real adapter did not call transport"
    assert push_calls[-1]["remote"] == "https://from-env.example.test:9999", (
        f"env should override config.yaml in real adapter; got remote="
        f"{push_calls[-1]['remote']!r}"
    )


def test_sync_config_yaml_credentials_reach_real_adapter(monkeypatch, tmp_path, _record_transport):
    """config.yaml sync credentials (key/encrypt) must reach the REAL adapter."""
    _fresh_env(monkeypatch, tmp_path)
    for v in ("MNEMOSYNE_SYNC_REMOTE", "MNEMOSYNE_SYNC_KEY", "MNEMOSYNE_SYNC_TOKEN"):
        monkeypatch.delenv(v, raising=False)
    (tmp_path / "config.yaml").write_text(
        "sync_remote: https://cred.example.test:8765\n"
        "sync_key: cfg-secret-key\n"
        "sync_encrypt: true\n"
    )
    handle_tool_call("mnemosyne_remember", {"content": "seed"})
    _seed_sync_event(tmp_path)

    handle_tool_call("mnemosyne_sync_push", {})

    push_calls = [c for c in _record_transport if c["path"] == "/sync/push"]
    assert push_calls, "real adapter did not call transport"
    assert push_calls[-1]["encryption_key"] == "cfg-secret-key", (
        f"config.yaml sync_key did not reach real adapter: "
        f"key={push_calls[-1]['encryption_key']!r}"
    )
    assert push_calls[-1]["encrypt_enabled"] is True, (
        f"config.yaml sync_encrypt did not reach real adapter: "
        f"encrypt={push_calls[-1]['encrypt_enabled']!r}"
    )


# ===========================================================================
# I2: read-only handlers must not crash or leak on corrupt/malformed DBs
# ===========================================================================

def _seed_valid_then_corrupt(tmp_path):
    """Materialize a valid default DB, then overwrite it with garbage bytes."""
    from mnemosyne.core.beam import BeamMemory
    b = BeamMemory(session_id="s", db_path=tmp_path / "mnemosyne.db")
    b.conn.close()
    (tmp_path / "mnemosyne.db").write_bytes(b"not a sqlite database at all!!")


def _seed_db_missing_tables(tmp_path):
    """Create a valid SQLite file with no mnemosyne tables."""
    conn = sqlite3.connect(tmp_path / "mnemosyne.db")
    conn.execute("CREATE TABLE irrelevant (x INTEGER)")
    conn.commit()
    conn.close()


def test_ingest_status_corrupt_db_returns_structured_no_leak(monkeypatch, tmp_path):
    _fresh_env(monkeypatch, tmp_path)
    _seed_valid_then_corrupt(tmp_path)
    result = handle_tool_call("mnemosyne_ingest_status", {})
    assert isinstance(result, dict), (
        "corrupt DB: handler raised instead of returning structured result"
    )
    assert result.get("status") in ("error", "unavailable", "ok"), (
        f"corrupt DB: missing status field: {result}"
    )
    # No raw exception text / DB internals leak.
    blob = json.dumps(result)
    assert "not a database" not in blob.lower(), (
        f"corrupt DB: raw sqlite error text leaked: {blob}"
    )
    assert "file is not a database" not in blob.lower()


def test_ingest_status_missing_table_returns_structured_no_leak(monkeypatch, tmp_path):
    _fresh_env(monkeypatch, tmp_path)
    _seed_db_missing_tables(tmp_path)
    result = handle_tool_call("mnemosyne_ingest_status", {})
    assert isinstance(result, dict)
    assert result.get("status") in ("error", "unavailable", "ok")
    blob = json.dumps(result)
    assert "no such table" not in blob.lower(), (
        f"missing table: raw sqlite error text leaked: {blob}"
    )


def test_persona_list_corrupt_db_returns_structured_no_leak(monkeypatch, tmp_path):
    _fresh_env(monkeypatch, tmp_path)
    _seed_valid_then_corrupt(tmp_path)
    result = handle_tool_call("mnemosyne_persona_list", {})
    assert isinstance(result, dict), (
        "corrupt DB: handler raised instead of returning structured result"
    )
    blob = json.dumps(result)
    assert "not a database" not in blob.lower(), (
        f"corrupt DB: raw sqlite error text leaked: {blob}"
    )
    assert "file is not a database" not in blob.lower()


def test_persona_list_missing_table_returns_structured_no_leak(monkeypatch, tmp_path):
    _fresh_env(monkeypatch, tmp_path)
    _seed_db_missing_tables(tmp_path)
    result = handle_tool_call("mnemosyne_persona_list", {})
    assert isinstance(result, dict)
    blob = json.dumps(result)
    assert "no such table" not in blob.lower(), (
        f"missing table: raw sqlite error text leaked: {blob}"
    )


def test_readonly_handlers_missing_db_still_empty_no_materialization(monkeypatch, tmp_path):
    """Regression guard: the corrupt-DB fix must not regress the missing-DB
    case (still empty structured result, still no file created)."""
    _fresh_env(monkeypatch, tmp_path)
    for tool in ("mnemosyne_ingest_status", "mnemosyne_persona_list"):
        result = handle_tool_call(tool, {})
        assert result.get("status") == "ok", (
            f"{tool} on missing DB should be ok/empty: {result}"
        )
        assert not (tmp_path / "mnemosyne.db").exists(), (
            f"{tool} materialized mnemosyne.db on a fresh data dir"
        )


# ===========================================================================
# Minor (shares I2 error boundary): invalid bank name must be truthful
# ===========================================================================

def test_ingest_status_invalid_bank_returns_structured_error(monkeypatch, tmp_path):
    """An invalid bank name (path traversal / illegal chars) must return a
    structured invalid_bank error, not an empty OK that hides the rejection."""
    _fresh_env(monkeypatch, tmp_path)
    result = handle_tool_call("mnemosyne_ingest_status", {"bank": "bad/name"})
    assert result.get("status") == "error", (
        f"invalid bank should be a structured error, not empty ok: {result}"
    )
    assert "invalid" in result.get("error", "").lower() or "bank" in result.get("error", "").lower(), (
        f"invalid bank error should name the cause: {result}"
    )


def test_persona_list_invalid_bank_returns_structured_error(monkeypatch, tmp_path):
    _fresh_env(monkeypatch, tmp_path)
    result = handle_tool_call("mnemosyne_persona_list", {"bank": "bad/name"})
    assert result.get("status") == "error", (
        f"invalid bank should be a structured error, not empty ok: {result}"
    )
