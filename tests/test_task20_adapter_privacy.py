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


@pytest.mark.parametrize("package", PACKAGES)
def test_persona_adapter_dispatch_catch_log_is_content_free(
    package, monkeypatch, caplog
):
    """Task 32: the dispatch catch log must be content-free, including exc_text.

    The backlog's "rendered log/traceback" canary bar covers the operator log,
    not just the LLM-visible JSON. ``logger.exception`` auto-attaches ``exc_info``
    so the RuntimeError(CANARY) traceback (and thus the canary marker) reaches
    the rendered log via ``rec.exc_text`` until the catch is switched to a
    static ``logger.error`` with no ``exc_info``.
    """
    import logging as _logging

    mod = _adapter_module(package, "persona_adapter")
    adapter = mod.PersonaAdapter.__new__(mod.PersonaAdapter)
    adapter._beam = object()
    adapter._local_beam = None
    monkeypatch.setattr(adapter, "_list", _raiser(CANARY))

    with caplog.at_level(_logging.ERROR, logger=mod.logger.name):
        out = json.loads(adapter.handle_tool_call("mnemosyne_persona_list", {}))

    # Static JSON envelope unchanged (failure must not become a success).
    assert out == {
        "status": "error",
        "error": "persona_tool_failed",
        "tool": "mnemosyne_persona_list",
    }
    assert CANARY not in json.dumps(out)

    # The error line naming the tool must still be present.
    assert "Persona tool mnemosyne_persona_list failed" in caplog.text

    # Content-free bar: canary must be absent from the full rendered log,
    # including any auto-attached traceback text (exc_text).
    rendered = caplog.text
    for rec in caplog.records:
        rendered += "\n" + (rec.exc_text or "")
    assert CANARY not in rendered, (
        f"persona adapter dispatch catch leaked canary into rendered log "
        f"(exc_text) for package={package!r}"
    )
