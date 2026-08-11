"""Task 20: Hermes adapter tool JSON must never expose raw exception text."""

import json
import logging
import urllib.error
import urllib.request
from io import BytesIO

import pytest

from tests.test_hermes_provider_parity import (
    INTEGRATION_SRC,
    PROJECT_ROOT,
    _import_module,
)

CANARY = "SECRET-CANARY-adapter-9f3a"
R6_CANARY = "SECRET-CANARY-r6"
PACKAGES = ("hermes_memory_provider", "mnemosyne_hermes")


def _raiser(message):
    def _raise(*_args, **_kwargs):
        raise RuntimeError(message)

    return _raise


def _adapter_module(package, submodule):
    root = INTEGRATION_SRC if package == "mnemosyne_hermes" else PROJECT_ROOT
    return _import_module(f"{package}.{submodule}", root)


def _all_log_fields(records):
    return "\n".join(
        f"{record.getMessage()}\n{record.args!r}\n{record.exc_text or ''}"
        for record in records
    )


@pytest.mark.parametrize("package", PACKAGES)
def test_sync_adapter_engine_init_failure_is_static(package, monkeypatch, caplog):
    mod = _adapter_module(package, "sync_adapter")
    from mnemosyne.core import sync

    adapter = mod.SyncAdapter.__new__(mod.SyncAdapter)
    adapter._beam = object()
    adapter._engine = None
    adapter._error = None
    adapter.encrypt_enabled = False
    adapter.encryption_key = ""
    monkeypatch.setattr(sync, "SyncEngine", _raiser(R6_CANARY))

    with caplog.at_level(logging.DEBUG, logger=mod.logger.name):
        adapter._build_engine()
        assert adapter.start() is False

    assert adapter._error == "sync_engine_init_failed"
    assert [record.getMessage() for record in caplog.records] == [
        "sync_adapter: engine_init_failed exception=RuntimeError",
        "sync_adapter: not_started",
    ]
    assert R6_CANARY not in _all_log_fields(caplog.records)


@pytest.mark.parametrize("package", PACKAGES)
def test_sync_adapter_unreadable_key_file_is_static(package, monkeypatch, caplog):
    mod = _adapter_module(package, "sync_adapter")
    adapter = mod.SyncAdapter.__new__(mod.SyncAdapter)
    adapter._config = {"key_source": f"file:{R6_CANARY}-path"}
    monkeypatch.setattr(mod.Path, "read_text", _raiser(R6_CANARY))

    with caplog.at_level(logging.WARNING, logger=mod.logger.name):
        assert adapter._resolve_key() == ""

    assert [record.getMessage() for record in caplog.records] == [
        "sync_adapter: key_file_unreadable exception=RuntimeError"
    ]
    assert R6_CANARY not in _all_log_fields(caplog.records)


@pytest.mark.parametrize("package", PACKAGES)
def test_sync_adapter_success_log_is_static(package, monkeypatch, caplog):
    mod = _adapter_module(package, "sync_adapter")
    from mnemosyne.core import sync

    class FakeEncryption:
        @classmethod
        def from_config(cls, **_kwargs):
            return cls()

    engine = type("Engine", (), {"device_id": R6_CANARY})()
    adapter = mod.SyncAdapter.__new__(mod.SyncAdapter)
    adapter._beam = object()
    adapter._engine = None
    adapter.remote = f"https://{R6_CANARY}-remote"
    adapter.encrypt_enabled = True
    adapter.encryption_key = f"{R6_CANARY}-key"
    monkeypatch.setattr(sync, "SyncEngine", lambda **_kwargs: engine)
    monkeypatch.setattr(sync, "SyncEncryption", FakeEncryption)

    with caplog.at_level(logging.INFO, logger=mod.logger.name):
        adapter._build_engine()

    assert [record.getMessage() for record in caplog.records] == [
        "sync_adapter: initialized"
    ]
    fields = _all_log_fields(caplog.records)
    assert R6_CANARY not in fields
    assert str(len(adapter.encryption_key)) not in fields


@pytest.mark.parametrize("package", PACKAGES)
def test_sync_adapter_http_error_is_static(package, monkeypatch, caplog):
    mod = _adapter_module(package, "sync_adapter")
    adapter = mod.SyncAdapter.__new__(mod.SyncAdapter)
    adapter._engine = object()
    adapter.remote = f"https://{R6_CANARY}-url"
    adapter.auth_token = R6_CANARY
    error = urllib.error.HTTPError(
        f"https://{R6_CANARY}-url",
        503,
        f"{R6_CANARY}-reason",
        {},
        BytesIO(f'{{"error":"{R6_CANARY}-body"}}'.encode()),
    )
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )

    with caplog.at_level(logging.DEBUG, logger=mod.logger.name):
        result = adapter._http_post("/sync/push", {"events": []})

    assert result == {
        "status": "error",
        "error": "sync_http_error",
        "http_status": 503,
    }
    assert R6_CANARY not in json.dumps(result)
    assert R6_CANARY not in _all_log_fields(caplog.records)


@pytest.mark.parametrize("package", PACKAGES)
def test_sync_adapter_request_error_is_static(package, monkeypatch, caplog):
    mod = _adapter_module(package, "sync_adapter")
    adapter = mod.SyncAdapter.__new__(mod.SyncAdapter)
    adapter._engine = object()
    adapter.remote = f"https://{R6_CANARY}-url"
    adapter.auth_token = R6_CANARY
    monkeypatch.setattr(urllib.request, "urlopen", _raiser(R6_CANARY))

    with caplog.at_level(logging.DEBUG, logger=mod.logger.name):
        result = adapter._http_post("/sync/push", {"events": []})

    assert result == {"status": "error", "error": "sync_request_failed"}
    assert R6_CANARY not in json.dumps(result)
    assert R6_CANARY not in _all_log_fields(caplog.records)


@pytest.mark.parametrize("package", PACKAGES)
def test_sync_adapter_unknown_tool_is_static(package):
    mod = _adapter_module(package, "sync_adapter")
    adapter = mod.SyncAdapter.__new__(mod.SyncAdapter)
    adapter._engine = object()

    out = json.loads(adapter.handle_tool_call(R6_CANARY, {"input": R6_CANARY}))

    assert out == {"status": "error", "error": "unknown_tool"}
    assert R6_CANARY not in json.dumps(out)


@pytest.mark.parametrize("package", PACKAGES)
def test_sync_adapter_dispatch_catch_is_static(package, monkeypatch, caplog):
    mod = _adapter_module(package, "sync_adapter")
    adapter = mod.SyncAdapter.__new__(mod.SyncAdapter)
    adapter._engine = object()
    monkeypatch.setattr(adapter, "_handle_status", _raiser(CANARY))
    with caplog.at_level(logging.DEBUG, logger=mod.logger.name):
        out = json.loads(adapter.handle_tool_call("mnemosyne_sync_status", {}))
    assert out == {
        "status": "error",
        "error": "sync_tool_failed",
        "tool": "mnemosyne_sync_status",
    }
    assert CANARY not in json.dumps(out)
    assert [record.getMessage() for record in caplog.records] == [
        "sync_adapter: tool_failed"
    ]
    assert CANARY not in _all_log_fields(caplog.records)


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
