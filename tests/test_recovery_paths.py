"""Tests for recovery.get_default_paths() honoring the same path configuration
as the live store (mnemosyne.core.beam).

The disaster-recovery helpers (backup/restore, and `mnemosyne reindex`'s
auto-backup) must resolve the database to the same location the store actually
uses. Previously they hardcoded ``~/.mnemosyne/data`` and ignored
MNEMOSYNE_DATA_DIR / HERMES_HOME, so they operated on (or failed to find) the
wrong database.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mnemosyne.dr import recovery


def test_get_default_paths_honors_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("MNEMOSYNE_BACKUP_DIR", raising=False)
    data_dir, backup_dir, db_path = recovery.get_default_paths()
    assert data_dir == tmp_path / "data"
    assert db_path == tmp_path / "data" / "mnemosyne.db"
    # backups land alongside the data dir, not under ~/.mnemosyne
    assert backup_dir == tmp_path / "backups"


def test_get_default_paths_honors_hermes_home(monkeypatch, tmp_path):
    monkeypatch.delenv("MNEMOSYNE_DATA_DIR", raising=False)
    monkeypatch.delenv("MNEMOSYNE_BACKUP_DIR", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    data_dir, backup_dir, db_path = recovery.get_default_paths()
    assert data_dir == tmp_path / "home" / "mnemosyne" / "data"
    assert db_path == data_dir / "mnemosyne.db"
    assert backup_dir == tmp_path / "home" / "mnemosyne" / "backups"


def test_get_default_paths_backup_dir_override(monkeypatch, tmp_path):
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MNEMOSYNE_BACKUP_DIR", str(tmp_path / "custom_backups"))
    _, backup_dir, _ = recovery.get_default_paths()
    assert backup_dir == tmp_path / "custom_backups"


def test_get_default_paths_data_dir_takes_precedence_over_hermes_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "explicit"))
    data_dir, _, db_path = recovery.get_default_paths()
    assert data_dir == tmp_path / "explicit"
    assert db_path == tmp_path / "explicit" / "mnemosyne.db"


def test_create_backup_succeeds_with_sqlite_vec_tables(tmp_path):
    """Regression: create_backup() must load sqlite-vec on the source AND
    destination connections, otherwise sqlite3.Connection.backup() and
    Connection.iterdump() both fail with ``no such module: vec0`` on
    databases that use vec0 virtual tables.

    Pre-fix: this test fails with ``sqlite3.OperationalError: no such
    module: vec0`` raised from inside the backup serialization path.
    """
    pytest.importorskip("sqlite_vec")

    db_path = tmp_path / "vec_test.db"
    backup_dir = tmp_path / "backups"

    # Build a tiny DB that has a vec0 virtual table — the exact schema
    # shape that triggered the original bug in 3.10.x.
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    import sqlite_vec
    sqlite_vec.load(conn)
    conn.execute(
        "CREATE VIRTUAL TABLE vec_items USING vec0("
        "embedding float[4] distance_metric=cosine)"
    )
    conn.execute("CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT)")
    conn.executemany("INSERT INTO meta VALUES (?, ?)", [("a", "1"), ("b", "2")])
    conn.commit()
    conn.close()

    # Act: this is the call path `mnemosyne backup` uses. Pre-fix it
    # raised sqlite3.OperationalError: no such module: vec0.
    result = recovery.create_backup(db_path=db_path, backup_dir=backup_dir)

    # Assert: backup file exists, is non-empty, gzipped, and the gz
    # contents contain the vec0 table definition.
    assert Path(result["backup_path"]).exists()
    assert result["backup_size"] > 0
    import gzip
    with gzip.open(result["backup_path"], "rt") as f:
        dump = f.read()
    assert "vec_items" in dump
    assert "CREATE VIRTUAL TABLE" in dump


# ---------------------------------------------------------------------------
# Task 1 / Wave 1 P0: backup unique filename + fail-closed restore
# ---------------------------------------------------------------------------

import gzip as _gzip


def _make_simple_db(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    conn.executemany("INSERT INTO t VALUES (?, ?)", [(1, "a"), (2, "b")])
    conn.commit()
    conn.close()


def test_two_rapid_backups_get_unique_filenames(tmp_path):
    """Two backups created within the same second must not overwrite each
    other."""
    db_path = tmp_path / "src.db"
    _make_simple_db(db_path)
    bdir = tmp_path / "backups"

    r1 = recovery.create_backup(db_path=db_path, backup_dir=bdir)
    r2 = recovery.create_backup(db_path=db_path, backup_dir=bdir)

    assert Path(r1["backup_path"]).name != Path(r2["backup_path"]).name, (
        "two backups in the same second collided on filename"
    )
    assert Path(r1["backup_path"]).exists()
    assert Path(r2["backup_path"]).exists()


def test_restore_rejects_active_wal_sidecar(tmp_path):
    """A stale -wal sidecar means uncommitted frames would be silently
    dropped by a main-file replace; restore must refuse."""
    db_path = tmp_path / "target.db"
    bdir = tmp_path / "backups"
    _make_simple_db(db_path)
    backup = recovery.create_backup(db_path=db_path, backup_dir=bdir)

    (db_path.parent / (db_path.name + "-wal")).write_bytes(b"\x00" * 64)
    with pytest.raises(RuntimeError, match="sidecar"):
        recovery.restore_backup(Path(backup["backup_path"]), db_path)


def test_restore_rejects_active_shm_sidecar(tmp_path):
    db_path = tmp_path / "target.db"
    bdir = tmp_path / "backups"
    _make_simple_db(db_path)
    backup = recovery.create_backup(db_path=db_path, backup_dir=bdir)

    (db_path.parent / (db_path.name + "-shm")).write_bytes(b"\x00" * 64)
    with pytest.raises(RuntimeError, match="sidecar"):
        recovery.restore_backup(Path(backup["backup_path"]), db_path)


def test_restore_payload_checksum_mismatch_rejected_and_target_preserved(tmp_path):
    """Corruption that still decompresses as gzip but changes the dump payload
    must be caught by the payload checksum and rejected; the target is
    preserved."""
    db_path = tmp_path / "target.db"
    bdir = tmp_path / "backups"
    _make_simple_db(db_path)
    backup = recovery.create_backup(db_path=db_path, backup_dir=bdir)
    backup_path = Path(backup["backup_path"])

    raw = _gzip.decompress(backup_path.read_bytes())
    corrupted = raw.replace(b"VALUES", b"VALOOS")
    if corrupted == raw:
        corrupted = raw.replace(b"CREATE", b"CREAT")
    backup_path.write_bytes(_gzip.compress(corrupted))

    conn = sqlite3.connect(str(db_path))
    conn.execute("INSERT INTO t VALUES (99, 'preserved')")
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match="checksum"):
        recovery.restore_backup(backup_path, db_path)

    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT v FROM t WHERE id = 99").fetchone()
    conn.close()
    assert row is not None and row[0] == "preserved", (
        "target was corrupted by a failed restore"
    )


def test_restore_failed_integrity_preserves_original(tmp_path):
    """A dump that rebuilds but fails integrity_check must not replace the
    target."""
    db_path = tmp_path / "target.db"
    bdir = tmp_path / "backups"
    _make_simple_db(db_path)
    backup = recovery.create_backup(db_path=db_path, backup_dir=bdir)
    backup_path = Path(backup["backup_path"])

    conn = sqlite3.connect(str(db_path))
    conn.execute("INSERT INTO t VALUES (99, 'keepme')")
    conn.commit()
    conn.close()

    # Break the dump so it produces an invalid DB but still parses as SQL.
    raw = _gzip.decompress(backup_path.read_bytes())
    backup_path.write_bytes(_gzip.compress(raw.replace(b"CREATE TABLE", b"BREAK TABLE")))

    with pytest.raises(Exception):
        recovery.restore_backup(backup_path, db_path)

    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT v FROM t WHERE id = 99").fetchone()
    conn.close()
    assert row is not None and row[0] == "keepme"


def test_successful_restore_replaces_target_and_preserves_original(tmp_path):
    db_path = tmp_path / "target.db"
    bdir = tmp_path / "backups"
    _make_simple_db(db_path)
    backup = recovery.create_backup(db_path=db_path, backup_dir=bdir)

    # Mutate the live target after the backup.
    conn = sqlite3.connect(str(db_path))
    conn.execute("INSERT INTO t VALUES (42, 'post-backup')")
    conn.commit()
    conn.close()

    result = recovery.restore_backup(Path(backup["backup_path"]), db_path)

    assert result["integrity_check"] is True
    conn = sqlite3.connect(str(db_path))
    total = conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    post_backup_gone = conn.execute(
        "SELECT COUNT(*) FROM t WHERE id = 42"
    ).fetchone()[0]
    conn.close()
    assert total == 2, "target not restored to backup contents"
    assert post_backup_gone == 0
    preserved = Path(result["preserved_original"])
    assert preserved.exists(), "original target must be preserved as a sidecar"
