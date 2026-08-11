"""Tests for memory audit log integration."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pytest

from mnemosyne.core.beam import BeamMemory
from hermes_memory_provider import MnemosyneMemoryProvider
from hermes_memory_provider.audit import AuditLog
from tests.test_hermes_provider_parity import (
    INTEGRATION_SRC,
    PROJECT_ROOT,
    _import_module,
)


MIRROR_PACKAGES = ("hermes_memory_provider", "mnemosyne_hermes")


def _audit_module(package: str):
    root = INTEGRATION_SRC if package == "mnemosyne_hermes" else PROJECT_ROOT
    return _import_module(f"{package}.audit", root)


def _all_log_fields(records) -> str:
    return "\n".join(
        str(value)
        for record in records
        for value in (record.getMessage(), record.args, *vars(record).values())
    )


def _provider(tmp_path: Path) -> MnemosyneMemoryProvider:
    db_path = tmp_path / "banks" / "test" / "mnemosyne.db"
    beam = BeamMemory(session_id="audit-test", db_path=db_path)
    provider = MnemosyneMemoryProvider()
    provider._beam = beam
    provider._session_id = "audit-test"
    provider._agent_context = "primary"
    provider._profile_isolation_enabled = True
    provider._init_audit_log()
    return provider


def _call(provider: MnemosyneMemoryProvider, name: str, args: dict) -> dict:
    return json.loads(provider.handle_tool_call(name, args))


@pytest.mark.parametrize("package", MIRROR_PACKAGES)
def test_audit_log_health_and_init_failure_are_content_free(
    package, monkeypatch, caplog, tmp_path
):
    audit = _audit_module(package)
    exception_canary = "SECRET-CANARY-r5-init"
    path_canary = "SECRET-CANARY-r5-path"

    class _FailingInitConnection:
        def __init__(self):
            self.closed = False

        def __bool__(self):
            return False

        def execute(self, *_args, **_kwargs):
            raise RuntimeError(exception_canary)

        def close(self):
            self.closed = True

    connection = _FailingInitConnection()
    monkeypatch.setattr(
        audit.sqlite3, "connect", lambda *_args, **_kwargs: connection
    )

    with caplog.at_level(logging.WARNING, logger=audit.logger.name):
        log = audit.AuditLog(tmp_path / f"{path_canary}.db")

    fields = _all_log_fields(caplog.records)
    assert log.healthy is False
    assert connection.closed is True
    assert "audit: failed to create table" in fields
    assert "RuntimeError" in fields
    assert exception_canary not in fields
    assert path_canary not in fields


@pytest.mark.parametrize("package", MIRROR_PACKAGES)
def test_audit_log_record_failure_is_unhealthy_and_content_free(
    package, monkeypatch, caplog, tmp_path
):
    audit = _audit_module(package)
    exception_canary = "SECRET-CANARY-r5-record"
    path_canary = "SECRET-CANARY-r5-record-path"

    class _FailingRecordConnection:
        def __init__(self):
            self.closed = False

        def __bool__(self):
            return False

        def execute(self, _sql, parameters=None):
            if parameters is not None:
                raise RuntimeError(exception_canary)

        def commit(self):
            pass

        def close(self):
            self.closed = True

    connection = _FailingRecordConnection()
    monkeypatch.setattr(
        audit.sqlite3, "connect", lambda *_args, **_kwargs: connection
    )
    log = audit.AuditLog(tmp_path / f"{path_canary}.db")

    with caplog.at_level(logging.DEBUG, logger=audit.logger.name):
        log.record("remember")

    fields = _all_log_fields(caplog.records)
    assert log.healthy is False
    assert connection.closed is True
    assert "audit: failed to record event" in fields
    assert "RuntimeError" in fields
    assert exception_canary not in fields
    assert path_canary not in fields


@pytest.mark.parametrize("package", MIRROR_PACKAGES)
def test_audit_log_health_is_true_for_successful_db(package, tmp_path):
    log = _audit_module(package).AuditLog(tmp_path / "audit.db")
    assert log.healthy is True
    log.close()


class TestAuditLogModule:
    def test_creates_table(self, tmp_path):
        db_path = tmp_path / "audit.db"
        log = AuditLog(db_path)
        assert log.count() == 0
        log.close()

    def test_record_and_query(self, tmp_path):
        db_path = tmp_path / "audit.db"
        log = AuditLog(db_path)
        log.record("remember", memory_id="m1", bank="private", scope="global")
        log.record("forget", memory_id="m2", bank="private")
        assert log.count() == 2
        events = log.query(limit=10)
        assert events[0]["action"] == "forget"
        assert events[1]["action"] == "remember"
        log.close()

    def test_never_raises_on_bad_path(self, tmp_path):
        log = AuditLog(Path("/nonexistent/dir/audit.db"))
        log.record("remember", memory_id="x")
        assert log.count() == 0


class TestAuditIntegration:
    def test_remember_creates_audit_event(self, tmp_path):
        provider = _provider(tmp_path)
        result = _call(provider, "mnemosyne_remember", {
            "content": "audit test fact",
            "source": "fact",
            "importance": 0.7,
        })
        assert result["status"] == "stored"
        events = provider._audit.query(limit=10)
        assert len(events) == 1
        assert events[0]["action"] == "remember"
        assert events[0]["memory_id"] == result["memory_id"]
        assert events[0]["bank"] == "private"
        assert events[0]["source_tool"] == "mnemosyne_remember"

    def test_forget_creates_audit_event(self, tmp_path):
        provider = _provider(tmp_path)
        stored = _call(provider, "mnemosyne_remember", {
            "content": "to be forgotten",
            "source": "fact",
        })
        _call(provider, "mnemosyne_forget", {"memory_id": stored["memory_id"]})
        events = provider._audit.query(limit=10)
        assert len(events) == 2
        assert events[0]["action"] == "forget"
        assert events[0]["memory_id"] == stored["memory_id"]

    def test_forget_not_found_no_audit(self, tmp_path):
        provider = _provider(tmp_path)
        _call(provider, "mnemosyne_forget", {"memory_id": "nonexistent"})
        events = provider._audit.query(limit=10)
        assert len(events) == 0

    def test_invalidate_creates_audit_event(self, tmp_path):
        provider = _provider(tmp_path)
        stored = _call(provider, "mnemosyne_remember", {
            "content": "will be invalidated",
            "source": "fact",
        })
        _call(provider, "mnemosyne_invalidate", {"memory_id": stored["memory_id"]})
        events = provider._audit.query(limit=10)
        assert len(events) == 2
        assert events[0]["action"] == "invalidate"
        assert events[0]["memory_id"] == stored["memory_id"]

    def test_invalidate_not_found_returns_status(self, tmp_path):
        provider = _provider(tmp_path)
        result = _call(provider, "mnemosyne_invalidate", {"memory_id": "nonexistent-id"})
        assert result["status"] == "memory_not_found"
        assert result["memory_id"] == "nonexistent-id"
        events = provider._audit.query(limit=10)
        assert len(events) == 1
        assert json.loads(events[0]["metadata_json"]) == {"invalidated": False}

    def test_invalidate_success_returns_status(self, tmp_path):
        provider = _provider(tmp_path)
        stored = _call(provider, "mnemosyne_remember", {
            "content": "will be invalidated",
            "source": "fact",
        })
        before_invalidate = datetime.now()
        result = _call(provider, "mnemosyne_invalidate", {"memory_id": stored["memory_id"]})
        after_invalidate = datetime.now()
        assert result["status"] == "invalidated"
        assert result["memory_id"] == stored["memory_id"]
        row = provider._beam.conn.execute(
            "SELECT valid_until FROM working_memory WHERE id = ?",
            (stored["memory_id"],),
        ).fetchone()
        assert row is not None
        valid_until = datetime.fromisoformat(row[0])
        assert before_invalidate <= valid_until <= after_invalidate
        context_ids = {memory["id"] for memory in provider._beam.get_context(limit=10)}
        assert stored["memory_id"] not in context_ids
        events = provider._audit.query(limit=10)
        assert json.loads(events[0]["metadata_json"]) == {"invalidated": True}

    def test_invalidate_with_replacement_persists_link_and_audit_metadata(self, tmp_path):
        provider = _provider(tmp_path)
        original = _call(provider, "mnemosyne_remember", {
            "content": "original fact",
            "source": "fact",
        })
        replacement = _call(provider, "mnemosyne_remember", {
            "content": "replacement fact",
            "source": "fact",
        })
        before_invalidate = datetime.now()
        result = _call(provider, "mnemosyne_invalidate", {
            "memory_id": original["memory_id"],
            "replacement_id": replacement["memory_id"],
        })
        after_invalidate = datetime.now()
        assert result["status"] == "invalidated"
        row = provider._beam.conn.execute(
            "SELECT valid_until, superseded_by FROM working_memory WHERE id = ?",
            (original["memory_id"],),
        ).fetchone()
        assert row is not None
        valid_until = datetime.fromisoformat(row[0])
        assert before_invalidate <= valid_until <= after_invalidate
        assert row[1] == replacement["memory_id"]
        context_ids = {memory["id"] for memory in provider._beam.get_context(limit=10)}
        assert original["memory_id"] not in context_ids
        assert replacement["memory_id"] in context_ids
        events = provider._audit.query(limit=10)
        assert json.loads(events[0]["metadata_json"]) == {
            "replacement_id": replacement["memory_id"],
            "invalidated": True,
        }

    def test_sleep_creates_audit_event(self, tmp_path):
        provider = _provider(tmp_path)
        _call(provider, "mnemosyne_remember", {
            "content": "sleep test",
            "source": "fact",
        })
        _call(provider, "mnemosyne_sleep", {})
        events = provider._audit.query(limit=10)
        sleep_events = [e for e in events if e["action"] == "sleep"]
        assert len(sleep_events) == 1
        assert sleep_events[0]["source_tool"] == "mnemosyne_sleep"

    def test_sleep_dry_run_no_audit(self, tmp_path):
        provider = _provider(tmp_path)
        _call(provider, "mnemosyne_sleep", {"dry_run": True})
        events = provider._audit.query(limit=10)
        sleep_events = [e for e in events if e["action"] == "sleep"]
        assert len(sleep_events) == 0

    def test_shared_remember_creates_audit_event(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("MNEMOSYNE_HOST_LLM_ENABLED", "0")
        provider = MnemosyneMemoryProvider()
        provider.initialize(
            session_id="audit-shared-test",
            hermes_home=str(tmp_path / "profiles" / "test"),
            agent_identity="test",
            shared_surface_path=str(tmp_path / "shared" / "mnemosyne.db"),
        )
        provider._init_audit_log()
        result = _call(provider, "mnemosyne_shared_remember", {
            "content": "shared audit fact",
            "kind": "meta",
        })
        assert result["status"] == "stored_shared"
        events = provider._audit.query(limit=10)
        shared_events = [e for e in events if e["action"] == "shared_remember"]
        assert len(shared_events) == 1
        assert shared_events[0]["bank"] == "surface"

    def test_shared_forget_creates_audit_event(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("MNEMOSYNE_HOST_LLM_ENABLED", "0")
        provider = MnemosyneMemoryProvider()
        provider.initialize(
            session_id="audit-shared-test",
            hermes_home=str(tmp_path / "profiles" / "test"),
            agent_identity="test",
            shared_surface_path=str(tmp_path / "shared" / "mnemosyne.db"),
        )
        provider._init_audit_log()
        stored = _call(provider, "mnemosyne_shared_remember", {
            "content": "shared to delete",
            "kind": "meta",
        })
        _call(provider, "mnemosyne_shared_forget", {"memory_id": stored["memory_id"]})
        events = provider._audit.query(limit=10)
        forget_events = [e for e in events if e["action"] == "shared_forget"]
        assert len(forget_events) == 1
        assert forget_events[0]["memory_id"] == stored["memory_id"]

    def test_no_audit_when_beam_missing(self, tmp_path):
        provider = MnemosyneMemoryProvider()
        provider._session_id = "no-beam"
        provider._agent_context = "primary"
        # _audit is None, should not crash
        provider._audit_event("remember", memory_id="x")
