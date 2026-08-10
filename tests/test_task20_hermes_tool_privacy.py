"""Task 20: Hermes provider tool JSON must never expose raw exception text."""

import importlib
import json
import sqlite3
import sys
import types

import pytest

from tests.test_hermes_provider_parity import (
    INTEGRATION_SRC,
    PROJECT_ROOT,
    _import_module,
)

CANARY = "SECRET-CANARY-9f3a-content-must-not-leak"
MIRRORS = ("hermes_memory_provider", "mnemosyne_hermes")


@pytest.fixture(scope="module")
def provider_modules():
    return {
        "hermes_memory_provider": _import_module(
            "hermes_memory_provider", PROJECT_ROOT
        ),
        "mnemosyne_hermes": _import_module("mnemosyne_hermes", INTEGRATION_SRC),
    }


def _raiser(message):
    def _raise(*_args, **_kwargs):
        raise RuntimeError(message)

    return _raise


def _bare_provider(module):
    provider = module.MnemosyneMemoryProvider.__new__(module.MnemosyneMemoryProvider)
    provider._beam = object()
    provider._agent_context = "test"
    provider._agent_identity = "test"
    provider._skip_contexts = set()
    provider._default_scope = "session"
    provider._retry_init_args = None
    provider._profile_isolation_enabled = False
    provider._audit_event = lambda *args, **kwargs: None
    provider._read_config_key = lambda key: None
    return provider


class _ValidateRow:
    def fetchone(self):
        return ("m1", "author", "content")


class _ValidateConn:
    def __init__(self, canary):
        self._canary = canary

    def execute(self, sql, *args):
        if sql.lstrip().upper().startswith("SELECT"):
            return _ValidateRow()
        raise RuntimeError(self._canary)


class _ValidateBeam:
    def __init__(self, canary):
        self.conn = _ValidateConn(canary)


def _fake_run_diagnostics(**kwargs):
    return {"checks_total": 0, "entries": [], "key_findings": []}


@pytest.mark.parametrize("mirror", MIRRORS)
def test_top_level_tool_catch_is_static(mirror, provider_modules, monkeypatch):
    module = provider_modules[mirror]
    provider = _bare_provider(module)
    monkeypatch.setattr(provider, "_handle_stats", _raiser(CANARY))
    out = json.loads(provider.handle_tool_call("mnemosyne_stats", {}))
    assert out == {"error": "tool_failed", "tool": "mnemosyne_stats"}
    assert CANARY not in json.dumps(out)


@pytest.mark.parametrize("mirror", MIRRORS)
def test_tool_config_value_error_is_static(mirror, provider_modules, monkeypatch):
    module = provider_modules[mirror]
    provider = _bare_provider(module)
    monkeypatch.setattr(
        provider,
        "_read_config_key",
        lambda key: [CANARY] if key == "tools" else None,
    )
    out = json.loads(provider.handle_tool_call("mnemosyne_remember", {}))
    assert out == {"error": "invalid_tool_name"}
    assert CANARY not in json.dumps(out)


@pytest.mark.parametrize("mirror", MIRRORS)
def test_sync_adapter_unavailable_is_static(mirror, provider_modules, monkeypatch):
    module = provider_modules[mirror]
    provider = _bare_provider(module)
    sub = importlib.import_module(f"{mirror}.sync_adapter")
    monkeypatch.setattr(sub, "SyncAdapter", _raiser(CANARY))
    out = json.loads(provider.handle_tool_call("mnemosyne_sync_status", {}))
    assert out == {"status": "error", "error": "sync_adapter_unavailable"}
    assert CANARY not in json.dumps(out)


@pytest.mark.parametrize("mirror", MIRRORS)
def test_persona_adapter_unavailable_is_static(mirror, provider_modules, monkeypatch):
    module = provider_modules[mirror]
    provider = _bare_provider(module)
    sub = importlib.import_module(f"{mirror}.persona_adapter")
    monkeypatch.setattr(sub, "PersonaAdapter", _raiser(CANARY))
    out = json.loads(provider.handle_tool_call("mnemosyne_persona_list", {}))
    assert out == {"status": "error", "error": "persona_adapter_unavailable"}
    assert CANARY not in json.dumps(out)


def test_mh_standalone_persona_handler_is_static(provider_modules, monkeypatch):
    module = provider_modules["mnemosyne_hermes"]
    monkeypatch.setattr(module, "_persona_adapter", None)
    sub = importlib.import_module("mnemosyne_hermes.persona_adapter")
    monkeypatch.setattr(sub, "PersonaAdapter", _raiser(CANARY))
    out = json.loads(module._get_persona_handler("mnemosyne_persona_list")({}))
    assert out == {"status": "error", "error": "persona_adapter_unavailable"}
    assert CANARY not in json.dumps(out)


@pytest.mark.parametrize("mirror", MIRRORS)
def test_validate_failure_reason_is_static(mirror, provider_modules):
    module = provider_modules[mirror]
    provider = _bare_provider(module)
    provider._beam = _ValidateBeam(CANARY)
    out = json.loads(
        provider.handle_tool_call(
            "mnemosyne_validate",
            {"memory_id": "m1", "action": "attest", "validator": "task20"},
        )
    )
    assert out == {
        "error": "validation_failed",
        "reason": "validation_error",
        "memory_id": "m1",
    }
    assert CANARY not in json.dumps(out)


@pytest.mark.parametrize("mirror", MIRRORS)
def test_apply_pending_failure_item_is_static(
    mirror, provider_modules, monkeypatch, tmp_path
):
    module = provider_modules[mirror]
    provider = _bare_provider(module)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    hermes_constants = types.ModuleType("hermes_constants")
    hermes_constants.get_hermes_home = lambda: tmp_path
    monkeypatch.setitem(sys.modules, "hermes_constants", hermes_constants)
    pending_dir = tmp_path / "pending" / "memory"
    pending_dir.mkdir(parents=True)
    (pending_dir / "p1.json").write_text(
        json.dumps({"id": "p1", "payload": {"content": "x"}})
    )
    provider._beam = types.SimpleNamespace(remember=_raiser(CANARY))
    out = json.loads(
        provider.handle_tool_call("mnemosyne_apply_pending", {"pending_ids": "p1"})
    )
    assert out["failed"] == [{"id": "p1", "error": "apply_failed"}]
    assert CANARY not in json.dumps(out)


@pytest.mark.parametrize("mirror", MIRRORS)
def test_diagnose_counts_error_is_static(
    mirror, provider_modules, monkeypatch, tmp_path
):
    module = provider_modules[mirror]
    provider = _bare_provider(module)
    provider._beam = types.SimpleNamespace(
        db_path=str(tmp_path / "active.db"), conn=object()
    )
    monkeypatch.setattr("mnemosyne.diagnose.run_diagnostics", _fake_run_diagnostics)
    monkeypatch.setattr("sqlite3.connect", _raiser(CANARY))
    out = json.loads(provider.handle_tool_call("mnemosyne_diagnose", {}))
    assert out["active_provider_counts_error"] == "diagnostic_unavailable"
    assert CANARY not in json.dumps(out)


@pytest.mark.parametrize("mirror", MIRRORS)
def test_diagnose_vec_working_error_is_static(
    mirror, provider_modules, monkeypatch, tmp_path
):
    module = provider_modules[mirror]
    db_path = tmp_path / "active.db"
    con = sqlite3.connect(str(db_path))
    try:
        con.executescript(
            "CREATE TABLE working_memory(id TEXT);"
            "CREATE TABLE episodic_memory(id TEXT);"
            "CREATE TABLE facts(id TEXT);"
        )
    finally:
        con.close()
    provider = _bare_provider(module)
    provider._beam = types.SimpleNamespace(db_path=str(db_path), conn=object())
    monkeypatch.setattr("mnemosyne.diagnose.run_diagnostics", _fake_run_diagnostics)
    monkeypatch.setattr("mnemosyne.core.beam.vec_working_coverage", _raiser(CANARY))
    out = json.loads(provider.handle_tool_call("mnemosyne_diagnose", {}))
    assert out["active_provider_vec_working_error"] == "diagnostic_unavailable"
    assert "active_provider_counts_error" not in out
    assert CANARY not in json.dumps(out)
