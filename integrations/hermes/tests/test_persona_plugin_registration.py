"""Regression tests for standalone-plugin persona tool registration."""

import sys

import mnemosyne_hermes as plugin
from mnemosyne_hermes import persona_adapter


def test_parity_suite_keeps_collection_time_plugin_modules_live():
    assert sys.modules["mnemosyne_hermes"] is plugin
    assert sys.modules["mnemosyne_hermes.persona_adapter"] is persona_adapter


class _Context:
    def __init__(self):
        self.tools = {}

    def register_memory_provider(self, provider):
        self.provider = provider

    def register_cli_command(self, **_kwargs):
        pass

    def register_tool(self, *, name, handler, **_kwargs):
        self.tools[name] = handler


class _Provider:
    def __init__(self):
        self._beam = object()

    def handle_tool_call(self, _tool_name, _arguments):
        return '{"status": "ok"}'


class _PersonaAdapter:
    received_beam = None

    def __init__(self, *, beam_instance):
        type(self).received_beam = beam_instance

    def handle_tool_call(self, _tool_name, _arguments):
        return '{"status": "ok"}'


def test_persona_tool_uses_plugin_registered_provider_beam(monkeypatch):
    """Persona tools share the standalone plugin provider's BeamMemory instance."""
    provider = _Provider()
    context = _Context()

    monkeypatch.setattr(plugin, "MnemosyneMemoryProvider", lambda: provider)
    monkeypatch.setattr(persona_adapter, "PersonaAdapter", _PersonaAdapter)
    monkeypatch.delattr(plugin, "_provider", raising=False)
    monkeypatch.setattr(plugin, "_persona_adapter", None)
    _PersonaAdapter.received_beam = None

    plugin.register(context)
    context.tools["mnemosyne_persona_promote"]({"memory_id": "memory-1"})

    assert _PersonaAdapter.received_beam is provider._beam
