"""Task 34a: Hermes provider residual log privacy (P1 sites S1 + S8).

Covers the two P1 cutover-blocker log sites identified in
``task-34-hermes-residual-log-root-cause-glm.md`` for BOTH provider mirrors:

* S1 -- ``Mnemosyne init failed: %s`` (WARNING) renders the raw init exception,
  which for SQLite/IO/schema errors embeds the private DB path. Reachable on any
  provider init failure.
* S8 -- ``Mnemosyne shared surface initialized: db=%s`` (INFO) renders the
  private shared-surface DB path unconditionally on every surface init.

The remaining P2 sites (S2-S7) are deliberately out of scope for Task 34a.
"""

import logging

import pytest

from tests.test_hermes_provider_parity import (
    INTEGRATION_SRC,
    PROJECT_ROOT,
    _import_module,
)

# Unique exception/path canary. Injected through the real init / surface path so
# the assertion exercises the actual log call, not a mocked one.
CANARY = "SECRET-CANARY-task34a-7d2e-path-or-content"
MIRRORS = ("hermes_memory_provider", "mnemosyne_hermes")


def _module(mirror):
    root = INTEGRATION_SRC if mirror == "mnemosyne_hermes" else PROJECT_ROOT
    return _import_module(mirror, root)


class _FakeBeam:
    """Stand-in BeamMemory for the S8 surface-init path (succeeds)."""

    def __init__(self, *args, **kwargs):
        pass


def _raise_canary(*_args, **_kwargs):
    raise RuntimeError(CANARY)


@pytest.mark.parametrize("mirror", MIRRORS)
def test_init_failure_log_is_content_free(mirror, monkeypatch, caplog):
    """S1: the init-failed WARNING must not render the raw exception.

    Drives a real init failure by making ``_get_beam_class`` raise a
    canary-bearing RuntimeError, then asserts the rendered WARNING keeps the
    static prefix and the exception class name (operationally useful) while the
    canary marker and any traceback text are absent. Failure semantics
    (``_init_error`` set, ``_beam`` cleared) are preserved.
    """
    module = _module(mirror)
    provider = module.MnemosyneMemoryProvider()
    monkeypatch.setattr(module, "_get_beam_class", _raise_canary)

    with caplog.at_level(logging.WARNING, logger=module.logger.name):
        provider.initialize("test")

    assert provider._beam is None
    assert provider._init_error is not None
    assert "Mnemosyne init failed" in caplog.text
    # Operationally useful class name is preserved.
    assert "RuntimeError" in caplog.text

    rendered = caplog.text
    for rec in caplog.records:
        rendered += "\n" + (rec.exc_text or "")
    assert CANARY not in rendered, (
        f"S1 init-failed log leaked canary into rendered log for mirror={mirror!r}"
    )


@pytest.mark.parametrize("mirror", MIRRORS)
def test_shared_surface_init_log_drops_path(mirror, monkeypatch, tmp_path, caplog):
    """S8: the shared-surface initialized INFO must not render the DB path.

    Drives the real ``_ensure_surface_beam`` with a canary-bearing synthetic
    private path (under a tmp dir so ``mkdir`` succeeds) and a fake BeamMemory
    that succeeds, so the unconditional INFO log fires. Asserts the static
    prefix is present while the path canary is absent.
    """
    module = _module(mirror)
    provider = module.MnemosyneMemoryProvider()
    monkeypatch.setattr(module, "_get_beam_class", lambda: _FakeBeam)
    # Inject a synthetic private path so the INFO log would render it pre-fix.
    # tmp_path keeps mkdir happy; the canary in the filename is what must not leak.
    provider._shared_surface_path = tmp_path / f"{CANARY}.db"

    with caplog.at_level(logging.INFO, logger=module.logger.name):
        provider._ensure_surface_beam()

    assert provider._surface_beam is not None
    assert "Mnemosyne shared surface initialized" in caplog.text
    assert CANARY not in caplog.text, (
        f"S8 surface-initialized log leaked private path for mirror={mirror!r}"
    )
