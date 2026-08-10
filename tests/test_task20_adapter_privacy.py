"""Task 20: Hermes adapter tool JSON must never expose raw exception text."""

import json

import pytest

from tests.test_hermes_provider_parity import (
    INTEGRATION_SRC,
    PROJECT_ROOT,
    _import_module,
)

CANARY = "SECRET-CANARY-adapter-9f3a"
PACKAGES = ("hermes_memory_provider", "mnemosyne_hermes")


def _raiser(message):
    def _raise(*_args, **_kwargs):
        raise RuntimeError(message)

    return _raise


def _adapter_module(package, submodule):
    root = INTEGRATION_SRC if package == "mnemosyne_hermes" else PROJECT_ROOT
    return _import_module(f"{package}.{submodule}", root)


@pytest.mark.parametrize("package", PACKAGES)
def test_sync_adapter_dispatch_catch_is_static(package, monkeypatch):
    mod = _adapter_module(package, "sync_adapter")
    adapter = mod.SyncAdapter.__new__(mod.SyncAdapter)
    adapter._engine = object()
    monkeypatch.setattr(adapter, "_handle_status", _raiser(CANARY))
    out = json.loads(adapter.handle_tool_call("mnemosyne_sync_status", {}))
    assert out == {
        "status": "error",
        "error": "sync_tool_failed",
        "tool": "mnemosyne_sync_status",
    }
    assert CANARY not in json.dumps(out)


@pytest.mark.parametrize("package", PACKAGES)
def test_sync_adapter_not_ready_is_static(package):
    mod = _adapter_module(package, "sync_adapter")
    adapter = mod.SyncAdapter.__new__(mod.SyncAdapter)
    adapter._engine = None
    adapter._error = CANARY
    out = json.loads(adapter.handle_tool_call("mnemosyne_sync_status", {}))
    assert out == {"status": "error", "error": "sync_adapter_unavailable"}
    assert CANARY not in json.dumps(out)


@pytest.mark.parametrize("package", PACKAGES)
def test_persona_adapter_dispatch_catch_is_static(package, monkeypatch):
    mod = _adapter_module(package, "persona_adapter")
    adapter = mod.PersonaAdapter.__new__(mod.PersonaAdapter)
    adapter._beam = object()
    adapter._local_beam = None
    monkeypatch.setattr(adapter, "_list", _raiser(CANARY))
    out = json.loads(adapter.handle_tool_call("mnemosyne_persona_list", {}))
    assert out == {
        "status": "error",
        "error": "persona_tool_failed",
        "tool": "mnemosyne_persona_list",
    }
    assert CANARY not in json.dumps(out)
