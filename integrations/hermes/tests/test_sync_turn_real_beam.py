"""
Task 7 fix round 1 — real-BeamMemory regression coverage for the three
reviewer findings (C1, I1, I2).

These tests exercise the actual BeamMemory + native Inhale path, NOT a
specless MagicMock, so they catch the default-construction provenance bug
(C1) and prove atomicity rather than merely labeling a partial failure
(I1). They also test the real ``dream_active`` config key (I2).
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

import pytest

from mnemosyne_hermes import (
    MnemosyneMemoryProvider,
    _sync_turn_event_id,
    _sync_turn_turn_id,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_INTEGRATION_SRC = Path(__file__).resolve().parents[2] / "src"


def _drop_modules(prefix: str) -> None:
    for name in list(sys.modules):
        if name == prefix or name.startswith(f"{prefix}."):
            del sys.modules[name]


def _import_mirror_module(package: str, import_root: Path):
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


def _provider_with_beam(beam, *, provider_class=MnemosyneMemoryProvider):
    p = provider_class.__new__(provider_class)
    p._beam = beam
    p._agent_context = ""
    p._skip_contexts = set()
    p._sync_roles = {"user", "assistant"}
    p._default_scope = "session"
    p._should_filter = lambda _c: False
    p._capture_identity_signals = lambda _c: None
    p._turn_count = 0
    p._auto_sleep_enabled = False
    p._audit_event = lambda *a, **k: None
    p._SYNC_TURN_SLOW_THRESHOLD_SECONDS = 1.0
    p._ensure_sync_turn_telemetry()
    p._maybe_retry_init = lambda: None
    holder = [beam]
    p._ensure_beam_access_lock = _dummy_lock
    p._beam_session_scope = lambda sid: _beam_scope_ctx(sid, holder)
    p._session_id = "hermes_default"
    p._channel_id_explicit = False
    p._reflect_disabled_for_cron = False
    p._reflect_max_calls_per_session = None
    p._reflect_calls_this_session = 0
    return p


# ---------------------------------------------------------------------------
# C1: default-construction BeamMemory must not silently lose memory.
# ---------------------------------------------------------------------------

@pytest.fixture
def real_beam():
    from mnemosyne.core.beam import BeamMemory

    tmpdir = tempfile.mkdtemp()
    db_path = Path(tmpdir) / "real_beam.db"
    beam = BeamMemory(session_id="hermes_default", db_path=db_path)
    yield beam
    try:
        beam.conn.close()
    except Exception:
        pass


def test_real_beam_default_construction_stores_sync_turn(real_beam):
    """A default-constructed BeamMemory (author_id/author_type default to
    None) must NOT silently lose the turn memory to Inhale rejections.

    Before the fix, the provider snapshotted actor_id='' (from the default
    None author_id), and Inhale rejected every event -- the provider then
    reported last_outcome='stored' despite zero persisted rows. The fix adds
    an actor_id fallback (session id) so the events are actually stored."""
    # Confirm the default-construction precondition: author_id is None.
    assert real_beam.author_id is None

    p = _provider_with_beam(real_beam)
    p.sync_turn("user content here", "assistant content here")
    diag = p._sync_turn_diagnostics()

    # With the actor_id fallback, the default beam must actually store both
    # roles -- it must NOT silently reject them.
    assert diag.get("last_outcome") == "stored", (
        f"expected stored with actor_id fallback; "
        f"outcome={diag.get('last_outcome')} "
        f"receipts={diag.get('last_receipts')}"
    )
    # Memory must be persisted, not silently lost.
    rows = real_beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE content LIKE '%content here%'"
    ).fetchone()[0]
    assert rows == 2, f"expected 2 stored rows, got {rows}"
    receipt_rows = real_beam.conn.execute(
        "SELECT status FROM ingest_receipts"
    ).fetchall()
    assert len(receipt_rows) == 2
    assert all(r["status"] in ("stored", "duplicate") for r in receipt_rows)


def test_real_beam_with_provenance_stores_sync_turn(real_beam):
    """When provenance is populated, a real BeamMemory must store both roles
    and report last_outcome='stored' with real receipt rows."""
    from mnemosyne.core import beam as beam_module
    from mnemosyne.core import inhale as inhale_module

    beam_module._embeddings.available = lambda: False

    real_beam.author_id = "actor-real"
    real_beam.author_type = "hermes"
    p = _provider_with_beam(real_beam)
    p.sync_turn("hello real beam", "reply from assistant")
    diag = p._sync_turn_diagnostics()

    assert diag.get("last_outcome") == "stored", (
        f"expected stored with provenance; got outcome={diag.get('last_outcome')} "
        f"receipts={diag.get('last_receipts')}"
    )
    receipt_rows = real_beam.conn.execute(
        "SELECT status FROM ingest_receipts"
    ).fetchall()
    assert len(receipt_rows) == 2
    statuses = [r["status"] for r in receipt_rows]
    assert all(s in ("stored", "duplicate") for s in statuses), statuses


# ---------------------------------------------------------------------------
# C1 (cont): a rejected receipt must be surfaced as a failure, never stored.
# ---------------------------------------------------------------------------

class _RejectingReceipt:
    def __init__(self, event_id: str):
        self.event_id = event_id
        self.status = "rejected"
        self.index_status = "failed_terminal"
        self.memory_ids = []


class _RejectingBeam:
    """A beam whose remember_turn always returns a rejected receipt (simulates
    a validation/security failure without a real DB)."""

    author_id = "actor-x"
    author_type = "hermes"
    channel_id = "proj-x"
    session_id = "hermes_default"
    canonical_owner_id = "default"
    agent_context = "primary"

    def __init__(self):
        self.remember_turn_calls = []

    def remember_turn(self, turn):
        self.remember_turn_calls.append(turn)
        return _RejectingReceipt(turn.event_id)

    def remember(self, **kwargs):
        pass


def test_rejected_receipt_is_not_reported_as_stored():
    p = _provider_with_beam(_RejectingBeam())
    p.sync_turn("user says something", "assistant says something")
    diag = p._sync_turn_diagnostics()
    # A rejected receipt must NEVER be reported as stored.
    assert diag.get("last_outcome") != "stored"
    assert diag.get("last_outcome") in ("failed", "partial")
    assert diag["failed"] == 1
    # PII safety: no user content in the error string.
    err = diag.get("last_error") or ""
    assert "user says something" not in err
    assert "assistant says something" not in err


def test_rejected_receipt_surfaces_error_telemetry(mirror_module):
    """Both mirrors must surface a structured error for rejected receipts."""
    for provider_class in (MnemosyneMemoryProvider, mirror_module.MnemosyneMemoryProvider):
        p = _provider_with_beam(_RejectingBeam(), provider_class=provider_class)
        p.sync_turn("private user message", "private assistant message")
        diag = p._sync_turn_diagnostics()
        assert diag.get("last_outcome") != "stored"
        assert diag["failed"] == 1
        assert "private user message" not in (diag.get("last_error") or "")
        assert "private assistant message" not in (diag.get("last_error") or "")


# ---------------------------------------------------------------------------
# I1/I2: atomicity when both sides are available — no first-side persistence
# on second-side failure.
# ---------------------------------------------------------------------------

class _FailSecondReceipt:
    def __init__(self, event_id: str):
        self.event_id = event_id
        self.status = "stored"
        self.index_status = "ready"
        self.memory_ids = [event_id[:8]]


def test_sync_turn_atomic_no_first_side_persistence_on_failure(monkeypatch):
    """When the second role fails, the first role's ingest must be rolled
    back (no durable first-side event). This is the atomicity proof.

    Uses a real BeamMemory and forces the second event's memory_id to collide
    with a pre-existing row, so the atomic transaction fails on commit and
    rolls back the first event's insert.
    """
    import tempfile as _tf
    from mnemosyne.core import beam as beam_module
    from mnemosyne.core.beam import BeamMemory
    from mnemosyne.core.inhale import _memory_id_for_event

    beam_module._embeddings.available = lambda: False

    db_path = Path(_tf.mkdtemp()) / "atom_fail.db"
    beam = BeamMemory(
        session_id="hermes_default", db_path=db_path,
        author_id="actor-atom", author_type="hermes",
    )

    # Pre-insert a working_memory row whose id collides with the id that
    # remember_turns_atomic will compute for the assistant event. This forces
    # an IntegrityError inside the atomic transaction, rolling back the user
    # event that was inserted first in the same transaction.
    assistant_event_id = _sync_turn_event_id(
        "hermes", "hermes_default",
        _sync_turn_turn_id("hermes_default", "user content", "assistant content"),
        "assistant",
    )
    colliding_id = _memory_id_for_event(assistant_event_id)
    beam.conn.execute(
        "INSERT INTO working_memory "
        "(id, content, source, timestamp, session_id, importance, "
        "metadata_json, veracity, trust_tier, author_id, author_type, "
        "channel_id, scope) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (colliding_id, "pre-existing", "x", "2026-01-01T00:00:00Z",
         "hermes_default", 0.5, "{}", "unknown", "IMPORTED",
         "a", "b", "c", "session"),
    )
    beam.conn.commit()

    p = _provider_with_beam(beam)
    p.sync_turn("user content", "assistant content")
    diag = p._sync_turn_diagnostics()

    # The second-side collision must surface as a structured failure.
    assert diag["failed"] == 1
    assert diag.get("last_outcome") in ("failed", "partial")

    # ATOMICITY PROOF: the user event was part of the same transaction as
    # the assistant event. The transaction rolled back, so the user content
    # must NOT be persisted.
    user_rows = beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE content = '[USER] user content'"
    ).fetchone()[0]
    assert user_rows == 0, (
        f"first-side user event was persisted despite second-side failure "
        f"(atomicity violated): {user_rows} rows"
    )
    user_receipts = beam.conn.execute(
        "SELECT COUNT(*) FROM ingest_receipts WHERE event_id LIKE '%'"
    ).fetchone()[0]
    # No NEW receipts should have been committed for this turn.
    assert user_receipts == 0, (
        f"receipts were committed despite rollback: {user_receipts} rows"
    )


# ---------------------------------------------------------------------------
# I2: real dream_active=True config gating.
# ---------------------------------------------------------------------------

def test_dream_active_true_blocks_sleep_via_real_config(monkeypatch):
    """The _dream_active_blocks_sleep helper must return True when the real
    mnemosyne.core.config dream_active key is set to a truthy value."""
    from mnemosyne.core.config import get_config

    cfg = get_config()
    monkeypatch.setattr(cfg, "get_bool", lambda key, default=False: key == "dream_active")
    p = _provider_with_beam(_RejectingBeam())
    assert p._dream_active_blocks_sleep() is True


def test_dream_active_false_allows_sleep_via_real_config(monkeypatch):
    from mnemosyne.core.config import get_config

    cfg = get_config()
    monkeypatch.setattr(cfg, "get_bool", lambda key, default=False: False)
    p = _provider_with_beam(_RejectingBeam())
    assert p._dream_active_blocks_sleep() is False
