"""Task 19: Beam operational diagnostics must stay content-free.

The eight Beam/SHMR failure paths must emit only static codes and messages.
Synthetic canary strings injected into exceptions and filesystem paths must
never reach the returned result dict, serialized JSON, or captured log
records (including exc_text tracebacks).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from mnemosyne.core.beam import BeamMemory, _deferred_commits

CANARY = "TASK19_PRIVATE_CANARY"
BEAM_LOGGER = "mnemosyne.core.beam"


class TestSleepAllSessionsContentFree:
    def test_consolidation_error_is_static_code(self, tmp_path, monkeypatch):
        # Seed a working_memory row for an alien session so sleep_all_sessions
        # iterates over it.
        host = BeamMemory(session_id="host", db_path=tmp_path / "host.db")
        host.conn.execute(
            "INSERT INTO working_memory "
            "(id, session_id, content, source, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            ("wm1", "alien", "seed", "test", "2020-01-01T00:00:00Z"),
        )
        host.conn.commit()

        def raising_sleep(self, *a, **kw):
            raise RuntimeError(CANARY + ":: /secret/db/path")

        monkeypatch.setattr(BeamMemory, "sleep", raising_sleep)

        result = host.sleep_all_sessions(dry_run=False)

        assert result["errors"] == 1
        details = result["error_details"]
        assert len(details) == 1
        assert details[0]["error"] == "consolidation_failed"
        blob = json.dumps(result, default=str)
        assert CANARY not in blob
        assert "/secret/db/path" not in blob

    def test_consolidation_error_log_is_content_free(
        self, tmp_path, monkeypatch, caplog
    ):
        host = BeamMemory(session_id="host", db_path=tmp_path / "host.db")
        host.conn.execute(
            "INSERT INTO working_memory "
            "(id, session_id, content, source, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            ("wm1", "alien", "seed", "test", "2020-01-01T00:00:00Z"),
        )
        host.conn.commit()

        def raising_sleep(self, *a, **kw):
            raise RuntimeError(CANARY)

        monkeypatch.setattr(BeamMemory, "sleep", raising_sleep)

        with caplog.at_level(logging.ERROR, logger=BEAM_LOGGER):
            host.sleep_all_sessions(dry_run=False)

        error_records = [
            r
            for r in caplog.records
            if r.name == BEAM_LOGGER
            and r.levelno == logging.ERROR
            and "consolidation failed" in r.getMessage()
        ]
        assert error_records, "expected at least one consolidation-failed ERROR"
        for r in error_records:
            assert CANARY not in r.getMessage()
            assert (r.exc_text or "") == "" or CANARY not in (r.exc_text or "")
            assert r.exc_info is None or not r.exc_info


class TestPolyphonicRecallContentFree:
    def test_engine_failure_log_omits_exception(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
        beam = BeamMemory(session_id="poly", db_path=tmp_path / "poly.db")

        class _ExplodingEngine:
            def recall(self, *a, **kw):
                raise RuntimeError(CANARY + "_POLY")

        monkeypatch.setattr(beam, "_get_polyphonic_engine", lambda: _ExplodingEngine())

        with caplog.at_level(logging.ERROR, logger=BEAM_LOGGER):
            results = beam.recall("query", top_k=5)

        assert results == []
        error_records = [
            r
            for r in caplog.records
            if r.name == BEAM_LOGGER
            and r.levelno == logging.ERROR
            and "polyphonic recall engine failed" in r.getMessage()
        ]
        assert error_records, "expected polyphonic-failed ERROR"
        for r in error_records:
            assert CANARY not in r.getMessage()
            assert CANARY not in (r.exc_text or "")
            assert r.exc_info is None or not r.exc_info


class TestDeferredCommitContentFree:
    def test_final_commit_error_log_omits_exception(
        self, tmp_path, monkeypatch, caplog
    ):
        beam = BeamMemory(session_id="dc", db_path=tmp_path / "dc.db")

        def raising_commit():
            raise sqlite3.OperationalError(CANARY + ":: /secret/db/path")

        monkeypatch.setattr(beam.conn, "_real_commit", raising_commit)

        with caplog.at_level(logging.ERROR, logger=BEAM_LOGGER):
            with pytest.raises(sqlite3.OperationalError):
                with _deferred_commits(beam.conn):
                    pass

        error_records = [
            r
            for r in caplog.records
            if r.name == BEAM_LOGGER
            and r.levelno == logging.ERROR
            and "final commit failed" in r.getMessage()
        ]
        assert error_records, "expected final-commit-failed ERROR"
        for r in error_records:
            assert CANARY not in r.getMessage()
            assert "/secret/db/path" not in r.getMessage()


class TestE6AnnotationsContentFree:
    def test_init_schema_error_log_omits_exception(self, tmp_path, monkeypatch, caplog):
        from mnemosyne.core import annotations as ann_module

        def raising_init(_db_path):
            raise sqlite3.OperationalError(CANARY + ":: /opt/secret/db")

        monkeypatch.setattr(ann_module, "init_annotations", raising_init)

        with caplog.at_level(logging.ERROR, logger=BEAM_LOGGER):
            BeamMemory(db_path=tmp_path / "init.db")

        error_records = [
            r
            for r in caplog.records
            if r.name == BEAM_LOGGER
            and r.levelno == logging.ERROR
            and "failed to initialize annotations schema" in r.getMessage()
        ]
        assert error_records, "expected annotations-init-failed ERROR"
        for r in error_records:
            assert CANARY not in r.getMessage()
            assert "/opt/secret/db" not in r.getMessage()

    def test_auto_migration_failure_omits_path_and_exception(
        self, tmp_path, monkeypatch, caplog
    ):
        from mnemosyne.migrations import e6_triplestore_split as e6mod

        # Seed a legacy triples table so has_pending_migration has something
        # to find. Reuse the helper pattern from test_beam_e6_auto_migrate.py.
        from mnemosyne.core.triples import init_triples

        db_path = tmp_path / "secret_path_migrate.db"
        init_triples(db_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT INTO triples (subject, predicate, object, valid_from, source, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("s", "mentions", "o", "2020-01-01T00:00:00Z", "test", 0.9),
        )
        conn.commit()
        conn.close()

        monkeypatch.setattr(e6mod, "has_pending_migration", lambda _conn: True)

        def raising_migrate(*a, **kw):
            raise RuntimeError(CANARY + "_MIGRATE")

        monkeypatch.setattr(e6mod, "migrate", raising_migrate)

        with caplog.at_level(logging.ERROR, logger=BEAM_LOGGER):
            BeamMemory(db_path=db_path)

        error_records = [
            r
            for r in caplog.records
            if r.name == BEAM_LOGGER
            and r.levelno == logging.ERROR
            and "auto-migration failed" in r.getMessage()
        ]
        assert error_records, "expected auto-migration-failed ERROR"
        for r in error_records:
            assert CANARY not in r.getMessage()
            assert str(db_path) not in r.getMessage()

    def test_auto_migrate_success_warning_omits_db_path(
        self, tmp_path, monkeypatch, caplog
    ):
        from mnemosyne.migrations import e6_triplestore_split as e6mod
        from mnemosyne.core.triples import init_triples

        db_path = tmp_path / "canary_path_warn.db"
        init_triples(db_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT INTO triples (subject, predicate, object, valid_from, source, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("s", "mentions", "o", "2020-01-01T00:00:00Z", "test", 0.9),
        )
        conn.commit()
        conn.close()

        monkeypatch.setattr(e6mod, "has_pending_migration", lambda _conn: True)
        monkeypatch.setattr(e6mod, "migrate", lambda *a, **kw: 5)

        with caplog.at_level(logging.WARNING, logger=BEAM_LOGGER):
            BeamMemory(db_path=db_path)

        warning_records = [
            r
            for r in caplog.records
            if r.name == BEAM_LOGGER
            and r.levelno == logging.WARNING
            and "auto-migrated" in r.getMessage()
        ]
        assert warning_records, "expected auto-migrated WARNING"
        for r in warning_records:
            assert "canary_path_warn" not in r.getMessage()
            assert "5" in r.getMessage()

    def test_opt_out_warning_omits_db_path(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setenv("MNEMOSYNE_AUTO_MIGRATE", "0")
        from mnemosyne.core.triples import init_triples

        db_path = tmp_path / "canary_optout.db"
        init_triples(db_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT INTO triples (subject, predicate, object, valid_from, source, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("s", "mentions", "o", "2020-01-01T00:00:00Z", "test", 0.9),
        )
        conn.commit()
        conn.close()

        with caplog.at_level(logging.WARNING, logger=BEAM_LOGGER):
            BeamMemory(db_path=db_path)

        warning_records = [
            r
            for r in caplog.records
            if r.name == BEAM_LOGGER
            and r.levelno == logging.WARNING
            and "MNEMOSYNE_AUTO_MIGRATE=0" in r.getMessage()
        ]
        assert warning_records, "expected opt-out WARNING"
        for r in warning_records:
            assert "canary_optout" not in r.getMessage()
