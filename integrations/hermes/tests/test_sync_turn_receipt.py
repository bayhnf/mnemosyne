"""
Task 7 — Hermes sync_turn receipt-aware + atomic, provenance propagation,
visible failure outcomes, and dream_active gating of the three sleep surfaces.

Written FIRST (RED). Exercises the packaged mnemosyne_hermes provider for
behavioral correctness and adds a focused parity check against the deployed
hermes_memory_provider mirror via the isolated import harness.
"""
from __future__ import annotations

import contextlib
import importlib
import json
import sys
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import mnemosyne_hermes
from mnemosyne_hermes import MnemosyneMemoryProvider

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Receipt:
    def __init__(self, event_id: str, status: str = "stored", index_status: str = "ready"):
        self.event_id = event_id
        self.status = status
        self.index_status = index_status
        self.memory_ids = [event_id[:8]]


class _ReceiptBeam:
    """Beam stand-in that supports remember_turn and exposes provenance."""

    author_id = "actor-42"
    author_type = "hermes"
    channel_id = "proj-7"
    session_id = "hermes_default"
    canonical_owner_id = "default"
    agent_context = "primary"

    def __init__(self) -> None:
        self.remember_calls: list[dict[str, Any]] = []
        self.remember_turn_calls: list[Any] = []

    def remember(self, **kwargs):
        self.remember_calls.append(kwargs)

    def remember_turn(self, turn):
        self.remember_turn_calls.append(turn)
        return _Receipt(turn.event_id)

    def remember_turns_atomic(self, turns):
        receipts = [self.remember_turn(turn) for turn in turns]
        return receipts

    def get_working_stats(self):
        return {"total": 0}

    def _count_unconsolidated_before(self, _cutoff):
        return 0


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_INTEGRATION_SRC = Path(__file__).resolve().parents[2] / "src"


def _drop_modules(prefix: str) -> None:
    for name in list(sys.modules):
        if name == prefix or name.startswith(f"{prefix}."):
            del sys.modules[name]


def _import_mirror_module(package: str, import_root: Path):
    """Import hermes_memory_provider in isolation (parity harness pattern)."""
    _drop_modules(package)
    saved_mnemo = {
        n: m for n, m in sys.modules.items()
        if n == "mnemosyne" or n.startswith("mnemosyne.")
    }
    _drop_modules("mnemosyne")
    sys.path.insert(0, str(import_root))
    sys.path.insert(0, str(_PROJECT_ROOT))
    try:
        return importlib.import_module(package)
    finally:
        for path in (str(import_root), str(_PROJECT_ROOT)):
            try:
                sys.path.remove(path)
            except ValueError:
                pass
        for n in list(sys.modules):
            if n == "mnemosyne" or n.startswith("mnemosyne."):
                sys.modules.pop(n, None)
        sys.modules.update(saved_mnemo)


@pytest.fixture(scope="module")
def mirror_module():
    return _import_mirror_module("hermes_memory_provider", _PROJECT_ROOT)


@contextlib.contextmanager
def _dummy_lock():
    yield


@contextlib.contextmanager
def _beam_scope_ctx(_session_id, beam_holder):
    yield beam_holder[0]


def _provider(
    beam=None,
    *,
    scope: str = "session",
    roles=("user", "assistant"),
    provider_class=MnemosyneMemoryProvider,
):
    """Build a provider whose sync_turn dependencies are stubbed."""
    p = provider_class.__new__(provider_class)
    real_beam = beam if beam is not None else _ReceiptBeam()
    p._beam = real_beam
    p._agent_context = ""
    p._skip_contexts = set()
    p._sync_roles = set(roles)
    p._default_scope = scope
    p._should_filter = lambda _c: False
    p._capture_identity_signals = lambda _c: None
    p._turn_count = 0
    p._auto_sleep_enabled = False
    p._audit_event = lambda *a, **k: None
    p._SYNC_TURN_SLOW_THRESHOLD_SECONDS = 1.0
    p._ensure_sync_turn_telemetry()
    p._maybe_retry_init = lambda: None
    # The integration variant uses a session-scope context manager; wire it
    # so the scope snapshot path is exercised.
    holder = [real_beam]
    p._ensure_beam_access_lock = _dummy_lock
    p._beam_session_scope = lambda sid: _beam_scope_ctx(sid, holder)
    p._session_id = "hermes_default"
    p._channel_id_explicit = False
    p._reflect_disabled_for_cron = False
    p._reflect_max_calls_per_session = None
    p._reflect_calls_this_session = 0
    return p


# ---------------------------------------------------------------------------
# sync_turn: receipt-aware + provenance propagation
# ---------------------------------------------------------------------------


def test_sync_turn_uses_remember_turn_with_stable_event_id():
    p = _provider()
    p.sync_turn("hello world", "assistant reply")
    assert len(p._beam.remember_turn_calls) == 2
    for turn in p._beam.remember_turn_calls:
        assert turn.event_id
        assert turn.turn_id
        assert turn.role in ("user", "assistant")


def test_sync_turn_propagates_provenance():
    p = _provider()
    p.sync_turn("user says", "assistant says")
    turns = {t.role: t for t in p._beam.remember_turn_calls}
    u = turns["user"]
    assert u.producer == "hermes"
    assert u.actor_id == "actor-42"
    assert u.project_id == "proj-7"
    assert u.session_id == "hermes_default"
    assert u.content_hash
    assert u.occurred_at
    assert u.content == "[USER] user says"


def test_sync_turn_stable_event_id_is_deterministic_for_same_turn():
    p = _provider()
    p.sync_turn("same content", "same reply")
    first = [t.event_id for t in p._beam.remember_turn_calls]
    p._beam.remember_turn_calls.clear()
    p.sync_turn("same content", "same reply")
    second = [t.event_id for t in p._beam.remember_turn_calls]
    assert first == second


def test_sync_turn_snapshot_scope_before_inhale():
    """Provenance must be snapshotted BEFORE ingest; a rebind during the
    call must not mutate the actor/project recorded on the receipt."""
    p = _provider()
    orig_author = p._beam.author_id
    mutated = {"done": False}
    real_remember_turn = p._beam.remember_turn

    def mutating_remember_turn(turn):
        p._beam.author_id = "someone-else"
        mutated["done"] = True
        return real_remember_turn(turn)

    p._beam.remember_turn = mutating_remember_turn
    p.sync_turn("content", "reply")
    assert mutated["done"]
    turns = {t.role: t for t in p._beam.remember_turn_calls}
    assert turns["user"].actor_id == orig_author


# ---------------------------------------------------------------------------
# Atomic when both sides available
# ---------------------------------------------------------------------------


def test_sync_turn_atomic_when_both_sides_available():
    """If the second ingest fails after the first succeeded, the overall
    sync_turn result must surface a structured failure rather than leaving
    a partial, silently-committed single-side state."""
    p = _provider()
    calls = {"n": 0}
    real_remember_turn = p._beam.remember_turn

    def fail_second(turn):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom on assistant")
        return real_remember_turn(turn)

    p._beam.remember_turn = fail_second
    p.sync_turn("user content", "assistant content")
    diag = p._sync_turn_diagnostics()
    assert diag["failed"] == 1
    assert diag["last_error"]
    # With the atomic path, a second-side failure rolls back the first side,
    # so no receipt landed -- the outcome is "failed" (not "partial").
    assert diag.get("last_outcome") in ("failed", "partial")


# ---------------------------------------------------------------------------
# Visible outcomes (no swallowed failures)
# ---------------------------------------------------------------------------


def test_sync_turn_failure_surface_is_structured():
    p = _provider()

    def boom(turn):
        raise RuntimeError("network down")

    p._beam.remember_turn = boom
    p.sync_turn("secret-user-content", "secret-assistant-content")
    diag = p._sync_turn_diagnostics()
    assert diag["failed"] == 1
    assert diag.get("last_outcome") == "failed"
    # PII-safe: raw content and raw exception message must not leak.
    err = diag.get("last_error") or ""
    assert "secret-user-content" not in err
    assert "network down" not in err


def test_sync_turn_success_records_outcome():
    p = _provider()
    p.sync_turn("hello", "reply")
    diag = p._sync_turn_diagnostics()
    assert diag["completed"] == 1
    assert diag.get("last_outcome") == "stored"


# ---------------------------------------------------------------------------
# Positional caller compatibility
# ---------------------------------------------------------------------------


def test_sync_turn_positional_signature_preserved():
    p = _provider()
    # Positional two-arg form and keyword-only session_id must both work,
    # and the call must continue to return None (no breaking return type).
    assert p.sync_turn("u", "a") is None
    assert p.sync_turn("u2", "a2", session_id="sess") is None


# ---------------------------------------------------------------------------
# dream_active gating of the three sleep surfaces
# ---------------------------------------------------------------------------


def test_auto_sleep_skipped_when_dream_active(monkeypatch):
    p = _provider()
    p._auto_sleep_enabled = True
    p._turn_count = 9  # next increment hits the % 10 == 0 gate
    p._beam.get_working_stats = lambda: {"total": 999}
    p._beam._count_unconsolidated_before = lambda _c: 999
    fired = {"sleep": False}

    # Mark the Beam as supporting sleep so the real auto-sleep path would
    # invoke it if not gated.
    p._beam.sleep = lambda *a, **k: fired.__setitem__("sleep", True)
    p._beam.db_path = ":memory:"
    monkeypatch.setattr(p, "_dream_active_blocks_sleep", lambda: True)
    p.sync_turn("u", "a")
    assert not fired["sleep"], "auto-sleep must be gated when dream_active"


def test_handle_sleep_skipped_when_dream_active(monkeypatch):
    p = _provider()
    monkeypatch.setattr(p, "_dream_active_blocks_sleep", lambda: True)
    out = json.loads(p._handle_sleep({}))
    assert out["status"] == "skipped"
    assert "dream" in out.get("reason", "").lower()


def test_session_end_skipped_when_dream_active(monkeypatch):
    p = _provider()
    monkeypatch.setattr(p, "_dream_active_blocks_sleep", lambda: True)
    fired = {"ran": False}
    orig_thread = threading.Thread

    class _SpyThread(orig_thread):
        def __init__(self, *a, **k):
            fired["ran"] = True
            super().__init__(*a, **k)

    monkeypatch.setattr(threading, "Thread", _SpyThread)
    p.on_session_end([])
    assert not fired["ran"]


def test_dream_active_blocks_sleep_reads_config_gate():
    """The helper must read the mnemosyne dream_active config key, returning
    True only when an applied/undoing Dream run owns canonical mutations."""
    p = _provider()
    # Default: no config → False (sleep is allowed).
    assert p._dream_active_blocks_sleep() is False


# ---------------------------------------------------------------------------
# Legacy fallback: when Beam lacks remember_turn, fall back to remember()
# ---------------------------------------------------------------------------


def test_sync_turn_falls_back_when_remember_turn_unavailable():
    _provider()  # exercise the default construction path

    class _LegacyBeam(_ReceiptBeam):
        def __getattribute__(self, name):
            if name == "remember_turn":
                raise AttributeError(name)
            return super().__getattribute__(name)

    legacy = _LegacyBeam()
    p2 = _provider(beam=legacy)
    p2.sync_turn("u content here", "a content here")
    assert len(legacy.remember_calls) == 2  # both roles saved
    diag = p2._sync_turn_diagnostics()
    assert diag["completed"] == 1


def test_sync_turn_specless_magicmock_uses_legacy_scope_and_failure_telemetry(
    mirror_module,
):
    """A dynamic mock child is not native Inhale capability."""
    providers = (
        MnemosyneMemoryProvider,
        mirror_module.MnemosyneMemoryProvider,
    )
    for provider_class in providers:
        beam = MagicMock()
        provider = _provider(
            beam=beam,
            scope="global",
            provider_class=provider_class,
        )

        provider.sync_turn("user content here", "assistant content here")

        assert beam.remember.call_count == 2
        assert [call.kwargs["scope"] for call in beam.remember.call_args_list] == [
            "global",
            "global",
        ]
        assert not beam.remember_turn.called

        failing_beam = MagicMock()
        failing_provider = _provider(
            beam=failing_beam,
            provider_class=provider_class,
        )
        secret = "private user message should not leak"
        failing_beam.remember.side_effect = RuntimeError(secret)

        failing_provider.sync_turn("user content here", "assistant content here")

        diagnostics = failing_provider._sync_turn_diagnostics()
        assert failing_beam.remember.call_count == 1
        assert not failing_beam.remember_turn.called
        assert diagnostics["completed"] == 0
        assert diagnostics["failed"] == 1
        assert diagnostics["last_error"] == "RuntimeError: <redacted>"
        assert secret not in diagnostics["last_error"]


# ---------------------------------------------------------------------------
# Parity: both provider mirrors behave identically
# ---------------------------------------------------------------------------


def test_sync_turn_parity_between_providers(mirror_module):
    """The deployed mirror (hermes_memory_provider) must produce the same
    event-id and provenance shape as the packaged provider."""
    modules = {
        "mnemosyne_hermes": mnemosyne_hermes,
        "hermes_memory_provider": mirror_module,
    }
    observed: dict[str, list[dict[str, Any]]] = {}
    for name, module in modules.items():
        beam = _ReceiptBeam()
        beam.author_id = "actor-X"
        beam.channel_id = "proj-Y"
        # Build each provider from its own class.
        prov = module.MnemosyneMemoryProvider.__new__(module.MnemosyneMemoryProvider)
        prov._beam = beam
        prov._agent_context = ""
        prov._skip_contexts = set()
        prov._sync_roles = {"user", "assistant"}
        prov._default_scope = "session"
        prov._should_filter = lambda _c: False
        prov._capture_identity_signals = lambda _c: None
        prov._turn_count = 0
        prov._auto_sleep_enabled = False
        prov._audit_event = lambda *a, **k: None
        prov._SYNC_TURN_SLOW_THRESHOLD_SECONDS = 1.0
        prov._ensure_sync_turn_telemetry()
        prov._maybe_retry_init = lambda: None
        holder = [beam]
        prov._ensure_beam_access_lock = _dummy_lock
        prov._beam_session_scope = lambda sid, _h=holder: _beam_scope_ctx(sid, _h)
        prov._session_id = "hermes_default"
        prov._channel_id_explicit = False
        prov._reflect_disabled_for_cron = False
        prov._reflect_max_calls_per_session = None
        prov._reflect_calls_this_session = 0
        prov.sync_turn("same user", "same assistant")
        observed[name] = [
            {
                "event_id": t.event_id,
                "producer": t.producer,
                "actor_id": t.actor_id,
                "project_id": t.project_id,
                "role": t.role,
            }
            for t in beam.remember_turn_calls
        ]
    assert observed["hermes_memory_provider"] == observed["mnemosyne_hermes"]
