"""Task 34a/35: Hermes provider residual + success-path log privacy.

Covers the two P1 cutover-blocker log sites identified in
``task-34-hermes-residual-log-root-cause-glm.md`` for BOTH provider mirrors:

* S1 -- ``Mnemosyne init failed: %s`` (WARNING) renders the raw init exception,
  which for SQLite/IO/schema errors embeds the private DB path. Reachable on any
  provider init failure.
* S8 -- ``Mnemosyne shared surface initialized: db=%s`` (INFO) renders the
  private shared-surface DB path unconditionally on every surface init.

Task 35 adds the success-path init INFO sites:

* S-PROFILE -- ``Mnemosyne initialized (profile isolation ON): ... db=%s``
  (INFO) renders ``mem.db_path`` on every successful bank-isolated init in
  BOTH provider mirrors.
* S-NONPROFILE-INT -- the integration-only nonprofile init INFO renders the
  derived ``db_path``. The primary mirror's nonprofile counterpart already
  logs session only and is deliberately left untouched (mirror parity is
  achieved by making the integration side match it).

The remaining P2 sites (S2-S7) are deliberately out of scope for these tasks.
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
# Distinct Task 35 marker so a leak pinpoints the success-path site.
DB_PATH_CANARY = "SECRET-CANARY-task35-9b3f-db-path"
MIRRORS = ("hermes_memory_provider", "mnemosyne_hermes")


def _module(mirror):
    root = INTEGRATION_SRC if mirror == "mnemosyne_hermes" else PROJECT_ROOT
    return _import_module(mirror, root)


class _FakeBeam:
    """Stand-in BeamMemory for the S8 surface-init path (succeeds)."""

    def __init__(self, *args, **kwargs):
        pass


class _FakeMem:
    """Stand-in Mnemosyne wrapper for the profile-isolation init path."""

    def __init__(self, db_path):
        self.db_path = db_path
        self.beam = _FakeBeam()


def _raise_canary(*_args, **_kwargs):
    raise RuntimeError(CANARY)


def _patch_mnemosyne(mirror, monkeypatch, mem):
    """Route the provider's call-time Mnemosyne import to a fake wrapper."""
    if mirror == "mnemosyne_hermes":
        from mnemosyne.core import memory

        monkeypatch.setattr(memory, "Mnemosyne", lambda **kwargs: mem)
    else:
        import mnemosyne

        monkeypatch.setattr(mnemosyne, "Mnemosyne", lambda **kwargs: mem)


def _assert_marker_absent(caplog, marker, site):
    rendered = caplog.text
    for rec in caplog.records:
        rendered += "\n" + (rec.exc_text or "")
    assert marker not in rendered, (
        f"{site} success log leaked marker into rendered output"
    )
    for rec in caplog.records:
        assert marker not in rec.getMessage(), (
            f"{site} success log leaked marker into getMessage()"
        )
        assert marker not in str(rec.args), (
            f"{site} success log leaked marker into record args={rec.args!r}"
        )
        for value in rec.__dict__.values():
            assert marker not in str(value), (
                f"{site} success log leaked marker into record extras"
            )


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


@pytest.mark.parametrize("mirror", MIRRORS)
def test_profile_isolation_init_log_drops_db_path(
    mirror, monkeypatch, tmp_path, caplog
):
    """Task 35: the profile-isolation success INFO must not render db_path.

    Drives the real ``initialize()`` profile branch with a fake Mnemosyne
    wrapper whose ``db_path`` carries the canary, so the real logger call
    would render it pre-fix. Asserts the static prefix and the session/bank
    render survive (success behavior preserved) while the marker is absent
    from both the rendered message and every record's args.
    """
    module = _module(mirror)
    provider = module.MnemosyneMemoryProvider()
    mem = _FakeMem(str(tmp_path / f"{DB_PATH_CANARY}.db"))
    _patch_mnemosyne(mirror, monkeypatch, mem)

    with caplog.at_level(logging.INFO, logger=module.logger.name):
        provider.initialize("test", agent_identity="profile35", profile_isolation=True)

    assert provider._memory is mem
    assert provider._beam is mem.beam
    assert "Mnemosyne initialized (profile isolation ON)" in caplog.text
    assert "session=hermes_test" in caplog.text
    assert "bank=profile35" in caplog.text
    _assert_marker_absent(caplog, DB_PATH_CANARY, f"profile-isolation({mirror})")


def test_integration_nonprofile_init_log_drops_db_path(monkeypatch, tmp_path, caplog):
    """Task 35: integration-only nonprofile success INFO must not render db_path.

    Drives the real ``initialize()`` nonprofile branch with a canary-bearing
    ``hermes_home`` so the derived db_path would render pre-fix. The primary
    mirror's nonprofile counterpart already logs session only; this asserts
    the integration side matches that parity without touching the primary.
    """
    module = _module("mnemosyne_hermes")
    provider = module.MnemosyneMemoryProvider()
    monkeypatch.setattr(module, "_get_beam_class", lambda: _FakeBeam)
    hermes_home = str(tmp_path / f"{DB_PATH_CANARY}-hermes")

    with caplog.at_level(logging.INFO, logger=module.logger.name):
        provider.initialize("test", hermes_home=hermes_home)

    assert provider._beam is not None
    assert "Mnemosyne initialized" in caplog.text
    assert "session=hermes_test" in caplog.text
    _assert_marker_absent(caplog, DB_PATH_CANARY, "nonprofile(mnemosyne_hermes)")
