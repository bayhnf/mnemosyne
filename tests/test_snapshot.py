"""Tests for mnemosyne.dr.snapshot — isolated read-only snapshot
and raw restore.

Task 5 (supplemental plan): a raw page-level snapshot helper that opens the
source strictly read-only (``mode=ro`` + ``PRAGMA query_only=ON``), copies
pages via ``sqlite3.Connection.backup()``, forces ``journal_mode=DELETE`` so
fresh consumers never inherit WAL/SHM sidecars, and writes a ``0600`` SHA-256
sidecar that contains the hash only. Restore reuses recovery's staged
atomic-replace safeguards (writer lock, staged rebuild, fsync, atomic
replace, post-replace integrity check with rollback).
"""

from __future__ import annotations

import hashlib
import sqlite3
import os
import shutil
import stat
from pathlib import Path

import pytest

import mnemosyne.dr.snapshot as snapshot
from mnemosyne.dr import snapshot as snapshot_mod


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _seed_database(path: Path) -> Path:
    """Create a small ordinary (rollback-journal) SQLite database."""
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    conn.executemany("INSERT INTO t VALUES (?, ?)", [(1, "a"), (2, "b")])
    conn.commit()
    conn.close()
    return path


def _seed_wal_database(path: Path):
    """Create a database running in WAL mode with one committed frame in the
    WAL that has NOT been checkpointed back to the main file.

    Returns (path, writer_conn). SQLite checkpoints AND removes the WAL file
    on the close of the LAST connection that held the write lock, so the
    caller MUST keep ``writer_conn`` open across the snapshot and close it in
    a ``finally`` — otherwise the WAL is quiesced before the snapshot runs and
    the test would not exercise the WAL-source path.
    """
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    conn.executemany("INSERT INTO t VALUES (?, ?)", [(1, "a"), (2, "b")])
    conn.commit()
    assert path.with_name(path.name + "-wal").exists(), "fixture did not produce a WAL"
    return path, conn


def _source_fingerprint(path: Path) -> tuple:
    """Return (size, mtime_ns, sha256-of-bytes) for a database path including
    its WAL/SHM sidecars if present."""
    parts = [
        path,
        path.with_name(path.name + "-wal"),
        path.with_name(path.name + "-shm"),
    ]
    fp = []
    for p in parts:
        if p.exists():
            st = p.stat()
            fp.append(
                (
                    str(p),
                    st.st_size,
                    st.st_mtime_ns,
                    hashlib.sha256(p.read_bytes()).hexdigest(),
                )
            )
    return tuple(fp)


def _integrity(path: Path) -> str:
    conn = sqlite3.connect(str(path))
    try:
        return conn.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        conn.close()


def _journal_mode(path: Path) -> str:
    conn = sqlite3.connect(str(path))
    try:
        return conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# create_isolated_snapshot — happy paths
# ---------------------------------------------------------------------------


def test_isolated_snapshot_uses_read_only_source_and_is_restore_compatible(tmp_path):
    source = _seed_database(tmp_path / "source.db")
    source_before = _source_fingerprint(source)
    result = snapshot.create_isolated_snapshot(source, tmp_path / "snapshots")

    snap = Path(result["snapshot_path"])
    assert result["integrity_check"] is True
    assert stat.S_IMODE(snap.stat().st_mode) == 0o600
    sidecar = Path(result["checksum_path"])
    assert stat.S_IMODE(sidecar.stat().st_mode) == 0o600
    sidecar_text = sidecar.read_text()
    assert sidecar_text.split()[0] == result["sha256"]
    assert str(source) not in sidecar_text, "sidecar must not leak source path"
    assert len(sidecar_text.split()) == 1, "sidecar must contain hash only"
    assert len(result["sha256"]) == 64
    assert _integrity(snap) == "ok"
    assert _journal_mode(snap) == "delete"
    assert not snap.with_name(snap.name + "-wal").exists()
    assert not snap.with_name(snap.name + "-shm").exists()
    assert _source_fingerprint(source) == source_before


def test_isolated_snapshot_copies_wal_source_and_leaves_no_wal_sidecar(tmp_path):
    source, holder = _seed_wal_database(tmp_path / "source.db")
    try:
        assert source.with_name(source.name + "-wal").exists(), (
            "WAL not alive at snapshot time"
        )
        result = snapshot.create_isolated_snapshot(source, tmp_path / "snapshots")
    finally:
        holder.close()
    snap = Path(result["snapshot_path"])

    conn = sqlite3.connect(str(snap))
    rows = conn.execute("SELECT id FROM t ORDER BY id").fetchall()
    conn.close()
    assert rows == [(1,), (2,)], "WAL-committed frames not captured by backup"

    assert _journal_mode(snap) == "delete"
    assert not snap.with_name(snap.name + "-wal").exists()
    assert not snap.with_name(snap.name + "-shm").exists()


def test_isolated_snapshot_creates_0700_destination_dir(tmp_path):
    source = _seed_database(tmp_path / "source.db")
    dest_dir = tmp_path / "snapshots"
    snapshot.create_isolated_snapshot(source, dest_dir)
    assert stat.S_IMODE(dest_dir.stat().st_mode) == 0o700


def test_isolated_snapshot_destination_collision_is_safe(tmp_path):
    source = _seed_database(tmp_path / "source.db")
    dest_dir = tmp_path / "snapshots"
    dest_dir.mkdir()
    pre_existing = dest_dir / "source.pristine.sqlite"
    pre_existing.write_bytes(b"not to be overwritten")

    result = snapshot.create_isolated_snapshot(source, dest_dir)
    snap = Path(result["snapshot_path"])
    assert snap != pre_existing
    assert pre_existing.read_bytes() == b"not to be overwritten"
    assert _integrity(snap) == "ok"


# ---------------------------------------------------------------------------
# create_isolated_snapshot — failure paths
# ---------------------------------------------------------------------------


def test_isolated_snapshot_missing_source_raises(tmp_path):
    with pytest.raises(snapshot.SnapshotError):
        snapshot.create_isolated_snapshot(
            tmp_path / "does_not_exist.db", tmp_path / "snapshots"
        )
    if (tmp_path / "snapshots").exists():
        assert list((tmp_path / "snapshots").iterdir()) == []


def test_isolated_snapshot_corrupt_source_raises_and_leaves_no_artifact(tmp_path):
    source = tmp_path / "corrupt.db"
    source.write_bytes(b"not a sqlite database")

    with pytest.raises(snapshot.SnapshotError):
        snapshot.create_isolated_snapshot(source, tmp_path / "snapshots")

    if (tmp_path / "snapshots").exists():
        assert list((tmp_path / "snapshots").iterdir()) == []


# ---------------------------------------------------------------------------
# restore_isolated_snapshot — happy path
# ---------------------------------------------------------------------------


def test_restore_isolated_snapshot_restores_into_target(tmp_path):
    source = _seed_database(tmp_path / "source.db")
    snap_result = snapshot.create_isolated_snapshot(source, tmp_path / "snapshots")
    snap = Path(snap_result["snapshot_path"])

    target = tmp_path / "target.db"
    result = snapshot.restore_isolated_snapshot(snap, target)

    assert result["integrity_check"] is True
    assert target.exists()
    assert _integrity(target) == "ok"
    conn = sqlite3.connect(str(target))
    rows = conn.execute("SELECT id FROM t ORDER BY id").fetchall()
    conn.close()
    assert rows == [(1,), (2,)]


def test_restore_isolated_snapshot_preserves_existing_target(tmp_path):
    source = _seed_database(tmp_path / "source.db")
    snap_result = snapshot.create_isolated_snapshot(source, tmp_path / "snapshots")
    snap = Path(snap_result["snapshot_path"])

    target = tmp_path / "target.db"
    _seed_database(target)
    conn = sqlite3.connect(str(target))
    conn.execute("INSERT INTO t VALUES (99, 'post-backup')")
    conn.commit()
    conn.close()

    result = snapshot.restore_isolated_snapshot(snap, target)

    conn = sqlite3.connect(str(target))
    total = conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    row99 = conn.execute("SELECT COUNT(*) FROM t WHERE id = 99").fetchone()[0]
    conn.close()
    assert total == 2
    assert row99 == 0
    assert result["preserved_original"] is not None
    assert Path(result["preserved_original"]).exists()


# ---------------------------------------------------------------------------
# restore_isolated_snapshot — failure paths
# ---------------------------------------------------------------------------


def test_restore_rejects_missing_checksum_sidecar_and_preserves_target(tmp_path):
    source = _seed_database(tmp_path / "source.db")
    snap_result = snapshot.create_isolated_snapshot(source, tmp_path / "snapshots")
    snap = Path(snap_result["snapshot_path"])
    Path(snap_result["checksum_path"]).unlink()

    target = tmp_path / "target.db"
    _seed_database(target)
    conn = sqlite3.connect(str(target))
    conn.execute("INSERT INTO t VALUES (99, 'keep')")
    conn.commit()
    conn.close()

    with pytest.raises(snapshot.SnapshotError):
        snapshot.restore_isolated_snapshot(snap, target)

    conn = sqlite3.connect(str(target))
    row = conn.execute("SELECT v FROM t WHERE id = 99").fetchone()
    conn.close()
    assert row and row[0] == "keep"


def test_restore_rejects_tampered_checksum_sidecar_and_preserves_target(tmp_path):
    source = _seed_database(tmp_path / "source.db")
    snap_result = snapshot.create_isolated_snapshot(source, tmp_path / "snapshots")
    snap = Path(snap_result["snapshot_path"])

    checksum_path = Path(snap_result["checksum_path"])
    checksum_path.write_text("a" * 64)

    target = tmp_path / "target.db"
    _seed_database(target)
    conn = sqlite3.connect(str(target))
    conn.execute("INSERT INTO t VALUES (99, 'keep')")
    conn.commit()
    conn.close()

    with pytest.raises(snapshot.SnapshotError, match="checksum"):
        snapshot.restore_isolated_snapshot(snap, target)

    conn = sqlite3.connect(str(target))
    row = conn.execute("SELECT v FROM t WHERE id = 99").fetchone()
    conn.close()
    assert row and row[0] == "keep"


def test_restore_rejects_tampered_snapshot_bytes_and_preserves_target(tmp_path):
    source = _seed_database(tmp_path / "source.db")
    snap_result = snapshot.create_isolated_snapshot(source, tmp_path / "snapshots")
    snap = Path(snap_result["snapshot_path"])

    data = bytearray(snap.read_bytes())
    data[-1] ^= 0xFF
    snap.write_bytes(bytes(data))

    target = tmp_path / "target.db"
    _seed_database(target)
    conn = sqlite3.connect(str(target))
    conn.execute("INSERT INTO t VALUES (99, 'keep')")
    conn.commit()
    conn.close()

    with pytest.raises(snapshot.SnapshotError, match="checksum"):
        snapshot.restore_isolated_snapshot(snap, target)

    conn = sqlite3.connect(str(target))
    row = conn.execute("SELECT v FROM t WHERE id = 99").fetchone()
    conn.close()
    assert row and row[0] == "keep"


def test_restore_rejects_active_target_wal_sidecar(tmp_path):
    """Active -wal sidecar must be rejected as a SnapshotError with no raw
    target path in the message (not a recovery RuntimeError leaking paths).
    """
    source = _seed_database(tmp_path / "source.db")
    snap_result = snapshot.create_isolated_snapshot(source, tmp_path / "snapshots")
    snap = Path(snap_result["snapshot_path"])

    target = tmp_path / "target.db"
    _seed_database(target)
    target.with_name(target.name + "-wal").write_bytes(b"\x00" * 64)

    with pytest.raises(snapshot.SnapshotError) as ei:
        snapshot.restore_isolated_snapshot(snap, target)

    msg = str(ei.value)
    assert str(target) not in msg, f"raw path leaked: {msg!r}"


def test_restore_rejects_active_target_shm_sidecar(tmp_path):
    """Active -shm sidecar must be rejected as a SnapshotError with no raw
    target path in the message.
    """
    source = _seed_database(tmp_path / "source.db")
    snap_result = snapshot.create_isolated_snapshot(source, tmp_path / "snapshots")
    snap = Path(snap_result["snapshot_path"])

    target = tmp_path / "target.db"
    _seed_database(target)
    target.with_name(target.name + "-shm").write_bytes(b"\x00" * 64)

    with pytest.raises(snapshot.SnapshotError) as ei:
        snapshot.restore_isolated_snapshot(snap, target)

    msg = str(ei.value)
    assert str(target) not in msg, f"raw path leaked: {msg!r}"


def test_restore_missing_snapshot_raises(tmp_path):
    target = tmp_path / "target.db"
    with pytest.raises(snapshot.SnapshotError):
        snapshot.restore_isolated_snapshot(tmp_path / "nope.sqlite", target)


def test_restore_post_replace_integrity_failure_preserves_target(tmp_path, monkeypatch):
    source = _seed_database(tmp_path / "source.db")
    snap_result = snapshot.create_isolated_snapshot(source, tmp_path / "snapshots")
    snap = Path(snap_result["snapshot_path"])

    target = tmp_path / "target.db"
    _seed_database(target)
    conn = sqlite3.connect(str(target))
    conn.execute("INSERT INTO t VALUES (77, 'preserved-me')")
    conn.commit()
    conn.close()

    monkeypatch.setattr(snapshot_mod.recovery, "verify_integrity", lambda p: False)

    with pytest.raises(snapshot.SnapshotError, match="integrity|post-replace"):
        snapshot.restore_isolated_snapshot(snap, target)

    conn = sqlite3.connect(str(target))
    row = conn.execute("SELECT v FROM t WHERE id = 77").fetchone()
    ok = conn.execute("PRAGMA integrity_check").fetchone()[0]
    conn.close()
    assert row and row[0] == "preserved-me"
    assert ok == "ok"


def test_restore_failure_leaves_no_staged_artifact(tmp_path, monkeypatch):
    source = _seed_database(tmp_path / "source.db")
    snap_result = snapshot.create_isolated_snapshot(source, tmp_path / "snapshots")
    snap = Path(snap_result["snapshot_path"])

    target = tmp_path / "target.db"
    _seed_database(target)

    real_connect = sqlite3.connect

    def patched_connect(target_or_uri, *args, **kwargs):
        conn = real_connect(target_or_uri, *args, **kwargs)
        if ".restore_staged" in str(target_or_uri):

            def broken(dst):
                raise sqlite3.OperationalError("forced staging failure")

            conn.backup = broken
        return conn

    monkeypatch.setattr(snapshot_mod.sqlite3, "connect", patched_connect)

    with pytest.raises(snapshot.SnapshotError):
        snapshot.restore_isolated_snapshot(snap, target)

    monkeypatch.setattr(snapshot_mod.sqlite3, "connect", real_connect)

    leftovers = [p for p in target.parent.iterdir() if "restore_staged" in p.name]
    assert leftovers == [], f"staged artifact left behind: {leftovers}"


# ---------------------------------------------------------------------------
# Task 5 pre-review hardening: cleanup + source-mutation regressions
# ---------------------------------------------------------------------------


def test_restore_post_replace_integrity_failure_cleans_preserved_artifact(
    tmp_path, monkeypatch
):
    """Regression: a post-replace integrity failure must NOT orphan the
    ``.restore_preserved`` copy that this invocation created.

    Before the fix, ``preserved_path`` was created by ``shutil.copy2`` but
    never tracked in the ``created`` list, so the failure-path ``_cleanup``
    could not remove it. The original target was rolled back correctly, but a
    stale ``.restore_preserved`` sidecar was left on disk.
    """
    source = _seed_database(tmp_path / "source.db")
    snap_result = snapshot.create_isolated_snapshot(source, tmp_path / "snapshots")
    snap = Path(snap_result["snapshot_path"])

    target = tmp_path / "target.db"
    _seed_database(target)
    conn = sqlite3.connect(str(target))
    conn.execute("INSERT INTO t VALUES (77, 'preserved-me')")
    conn.commit()
    conn.close()

    monkeypatch.setattr(snapshot_mod.recovery, "verify_integrity", lambda p: False)

    with pytest.raises(snapshot.SnapshotError):
        snapshot.restore_isolated_snapshot(snap, target)

    # The preserved artifact created by THIS call must be cleaned up.
    preserved = target.with_name(target.name + ".restore_preserved")
    assert not preserved.exists(), (
        f"cleanup violation: {preserved} orphaned after failed restore"
    )

    # And the original target must still be intact (rollback worked).
    conn = sqlite3.connect(str(target))
    row = conn.execute("SELECT v FROM t WHERE id = 77").fetchone()
    conn.close()
    assert row and row[0] == "preserved-me"


def test_restore_does_not_mutate_snapshot_source(tmp_path):
    """Safety invariant: restoring a snapshot must not mutate the snapshot
    file (no byte/mtime change). The restore source must be opened read-only
    so a checkpoint or journal-mode flip can never write back to it.
    """
    source = _seed_database(tmp_path / "source.db")
    snap_result = snapshot.create_isolated_snapshot(source, tmp_path / "snapshots")
    snap = Path(snap_result["snapshot_path"])

    snap_bytes_before = snap.read_bytes()
    snap_mtime_before = snap.stat().st_mtime_ns

    target = tmp_path / "target.db"
    snapshot.restore_isolated_snapshot(snap, target)

    assert snap.read_bytes() == snap_bytes_before, (
        "snapshot source was mutated during restore"
    )
    assert snap.stat().st_mtime_ns == snap_mtime_before, (
        "snapshot source mtime changed during restore"
    )
    # No sidecars should be created beside the snapshot either.
    assert not snap.with_name(snap.name + "-wal").exists()
    assert not snap.with_name(snap.name + "-shm").exists()


# ---------------------------------------------------------------------------
# Task 5 fix round 1: reviewer-driven regressions
# ---------------------------------------------------------------------------


def test_restore_active_sidecar_failure_is_content_free_snapshoterror(tmp_path):
    """Important 1: recovery._reject_active_sidecars raises RuntimeError with
    the full target path; snapshot must convert every such failure to a
    content-free SnapshotError whose message contains no raw path.
    """
    source = _seed_database(tmp_path / "source.db")
    snap_result = snapshot.create_isolated_snapshot(source, tmp_path / "snapshots")
    snap = Path(snap_result["snapshot_path"])

    target = tmp_path / "target.db"
    _seed_database(target)
    target.with_name(target.name + "-wal").write_bytes(b"\x00" * 64)

    with pytest.raises(snapshot.SnapshotError) as ei:
        snapshot.restore_isolated_snapshot(snap, target)

    msg = str(ei.value)
    assert str(target) not in msg, (
        f"raw target path leaked into SnapshotError message: {msg!r}"
    )
    assert target.name not in msg, (
        f"target name leaked into SnapshotError message: {msg!r}"
    )
    # -wal sidecar path components must not appear either.
    assert "wal" not in msg.lower() or "sidecar" in msg.lower()


def test_restore_writer_lock_failure_is_content_free_snapshoterror(
    tmp_path, monkeypatch
):
    """Important 1: recovery._acquire_writer_lock raises RuntimeError with the
    full target path when a writer holds the lock; snapshot must surface this
    as a content-free SnapshotError with no raw path.
    """
    source = _seed_database(tmp_path / "source.db")
    snap_result = snapshot.create_isolated_snapshot(source, tmp_path / "snapshots")
    snap = Path(snap_result["snapshot_path"])

    target = tmp_path / "target.db"
    _seed_database(target)

    def _always_locked(db_path):
        raise RuntimeError(
            f"Refusing to restore: a live writer appears to hold {db_path} "
            f"(forced). Stop all Mnemosyne processes before restoring."
        )

    monkeypatch.setattr(snapshot_mod.recovery, "_acquire_writer_lock", _always_locked)

    with pytest.raises(snapshot.SnapshotError) as ei:
        snapshot.restore_isolated_snapshot(snap, target)

    msg = str(ei.value)
    assert str(target) not in msg, (
        f"raw target path leaked into SnapshotError message: {msg!r}"
    )
    assert target.name not in msg, (
        f"target name leaked into SnapshotError message: {msg!r}"
    )


def test_restore_rollback_copy_failure_preserves_original_bytes(tmp_path, monkeypatch):
    """Important 2: when the post-replace integrity check fails AND the
    rollback copy of the preserved original also fails, the preserved
    `.restore_preserved` file must NOT be deleted/overwritten. It is the only
    remaining copy of the original.
    """
    source = _seed_database(tmp_path / "source.db")
    snap_result = snapshot.create_isolated_snapshot(source, tmp_path / "snapshots")
    snap = Path(snap_result["snapshot_path"])

    target = tmp_path / "target.db"
    _seed_database(target)
    conn = sqlite3.connect(str(target))
    conn.execute("INSERT INTO t VALUES (77, 'preserved-me')")
    conn.commit()
    conn.close()
    original_bytes = target.read_bytes()

    # Force post-replace integrity failure.
    monkeypatch.setattr(snapshot_mod.recovery, "verify_integrity", lambda p: False)
    # Break only the ROLLBACK copy (preserved -> target direction). The
    # preserve copy (target -> preserved) must succeed so that the preserved
    # file exists to test the protection on the rollback-failure path.
    real_copy2 = shutil.copy2

    def rollback_broken(src, dst, *a, **kw):
        if str(src).endswith(".restore_preserved"):
            raise OSError("forced rollback copy failure")
        return real_copy2(src, dst, *a, **kw)

    monkeypatch.setattr(snapshot_mod.shutil, "copy2", rollback_broken)

    with pytest.raises(snapshot.SnapshotError):
        snapshot.restore_isolated_snapshot(snap, target)

    monkeypatch.setattr(snapshot_mod.shutil, "copy2", real_copy2)

    preserved = target.with_name(target.name + ".restore_preserved")
    assert preserved.exists(), (
        "preserved original was deleted when rollback copy failed; the only "
        "copy of the original is gone"
    )
    assert preserved.read_bytes() == original_bytes, (
        "preserved original bytes changed; rollback-failure path corrupted it"
    )


def test_restore_rejects_sidecar_with_trailing_content(tmp_path):
    """Minor 1: the sidecar must contain exactly one whitespace-separated
    token (the 64-char hex hash). A sidecar with the correct hash plus extra
    junk must be rejected.
    """
    source = _seed_database(tmp_path / "source.db")
    snap_result = snapshot.create_isolated_snapshot(source, tmp_path / "snapshots")
    snap = Path(snap_result["snapshot_path"])

    checksum_path = Path(snap_result["checksum_path"])
    correct_hash = snap_result["sha256"]
    checksum_path.write_text(correct_hash + " extra-junk-token")

    target = tmp_path / "target.db"
    _seed_database(target)

    with pytest.raises(snapshot.SnapshotError, match="sidecar"):
        snapshot.restore_isolated_snapshot(snap, target)


def test_restore_rejects_non_hex_sidecar_as_malformed(tmp_path):
    """A 64-char token that is not hexadecimal must be a malformed sidecar,
    not a checksum mismatch.
    """
    source = _seed_database(tmp_path / "source.db")
    snap_result = snapshot.create_isolated_snapshot(source, tmp_path / "snapshots")
    snap = Path(snap_result["snapshot_path"])

    Path(snap_result["checksum_path"]).write_text("x" * 64)

    target = tmp_path / "target.db"
    _seed_database(target)

    with pytest.raises(snapshot.SnapshotError, match="sidecar"):
        snapshot.restore_isolated_snapshot(snap, target)


def test_mode_check_raises_content_free_error_without_assert(tmp_path, monkeypatch):
    """Minor 2: mode enforcement must not rely on `assert` (stripped under
    -O) and must emit a content-free error (no path fragment).
    """
    source = _seed_database(tmp_path / "source.db")

    # Force _assert_mode to detect a mismatch by chmod'ing the snapshot to a
    # wrong mode right after creation. We patch os.chmod applied to the
    # snapshot_path so the on-disk mode disagrees with the expected 0600.
    real_chmod = os.chmod

    def weakening_chmod(path, mode, *a, **kw):
        real_chmod(path, mode, *a, **kw)
        # If this looks like our snapshot data file, weaken it.
        if str(path).endswith(".pristine.sqlite"):
            real_chmod(path, 0o644)

    monkeypatch.setattr(snapshot_mod.os, "chmod", weakening_chmod)

    with pytest.raises(snapshot.SnapshotError) as ei:
        snapshot.create_isolated_snapshot(source, tmp_path / "snapshots")

    msg = str(ei.value)
    assert str(source) not in msg
    # No path basename should leak either.
    assert ".pristine.sqlite" not in msg
    assert "source.db" not in msg


# ---------------------------------------------------------------------------
# Task 9: snapshot restore must also re-acquire the writer lock on the
# replacement inode after its own os.replace(). Snapshot does NOT inherit a
# recovery-only fix — it has its own replace/verify/rollback/finally.
# ---------------------------------------------------------------------------

import multiprocessing as _mp
import threading as _threading


def _snapshot_worker_enter(target_str, enter_event, allow_event, result_queue):
    """Separate-process competitor for snapshot restore: waits for
    enter_event, then tries BEGIN IMMEDIATE + insert with busy_timeout=0."""
    import sqlite3

    try:
        enter_event.wait(timeout=10)
        conn = sqlite3.connect(target_str, timeout=0)
        conn.execute("PRAGMA busy_timeout=0")
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT INTO t VALUES (8888, 'competitor')")
        conn.commit()
        conn.close()
        result_queue.put((True, None))
    except sqlite3.OperationalError as exc:
        result_queue.put((False, str(exc)))
    except Exception as exc:  # pragma: no cover - defensive
        result_queue.put((False, repr(exc)))


def test_snapshot_restore_blocks_writer_in_post_replace_verify_window(
    tmp_path, monkeypatch
):
    """Through restore_isolated_snapshot, a separate-process writer must NOT
    be able to BEGIN IMMEDIATE on the target during post-replace verify.
    Normal restore succeeds after the release event, and no public
    SnapshotError contains the target path."""
    source = _seed_database(tmp_path / "source.db")
    snap_result = snapshot.create_isolated_snapshot(source, tmp_path / "snapshots")
    snap = Path(snap_result["snapshot_path"])

    target = tmp_path / "target.db"
    _seed_database(target)

    verify_entered = _threading.Event()
    allow_verify = _threading.Event()

    ctx = _mp.get_context("spawn")
    m_enter = ctx.Event()
    m_allow = ctx.Event()
    m_queue = ctx.Queue()
    proc = ctx.Process(
        target=_snapshot_worker_enter,
        args=(str(target), m_enter, m_allow, m_queue),
    )
    proc.start()

    def gated_verify(path):
        verify_entered.set()
        m_enter.set()
        allow_verify.wait(timeout=10)
        return True

    monkeypatch.setattr(snapshot_mod.recovery, "verify_integrity", gated_verify)

    try:
        result_holder = {"exc": None}

        def do_restore():
            try:
                snapshot.restore_isolated_snapshot(snap, target)
            except Exception as exc:
                result_holder["exc"] = exc

        t = _threading.Thread(target=do_restore)
        t.start()
        verify_entered.wait(timeout=10)
        competitor_result = m_queue.get(timeout=5)
        allow_verify.set()
        t.join(timeout=10)

        assert result_holder["exc"] is None, (
            f"restore raised unexpectedly: {result_holder['exc']!r}"
        )
        entered, err = competitor_result
        assert entered is False, (
            "competing writer entered the snapshot post-replace VERIFY window "
            f"(err={err})"
        )
        assert err is not None and "locked" in err.lower(), (
            f"expected 'database is locked', got: {err}"
        )
    finally:
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)


def test_snapshot_restore_reacquire_failure_is_content_free_and_preserves_original(
    tmp_path, monkeypatch
):
    """If post-replace lock re-acquisition fails, restore_isolated_snapshot
    must surface a content-free SnapshotError (no raw target path or file
    name) and must retain the .restore_preserved original (data-loss guard).
    """
    source = _seed_database(tmp_path / "source.db")
    snap_result = snapshot.create_isolated_snapshot(source, tmp_path / "snapshots")
    snap = Path(snap_result["snapshot_path"])

    target = tmp_path / "target.db"
    _seed_database(target)
    conn = sqlite3.connect(str(target))
    conn.execute("INSERT INTO t VALUES (77, 'preserved-me')")
    conn.commit()
    conn.close()

    original_acquire = snapshot_mod.recovery._acquire_writer_lock
    call_count = {"n": 0}

    def fail_second_acquire(db_path):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise sqlite3.OperationalError("database is locked (forced)")
        return original_acquire(db_path)

    monkeypatch.setattr(
        snapshot_mod.recovery, "_acquire_writer_lock", fail_second_acquire
    )

    with pytest.raises(snapshot.SnapshotError) as ei:
        snapshot.restore_isolated_snapshot(snap, target)

    msg = str(ei.value)
    assert str(target) not in msg, f"raw target path leaked into SnapshotError: {msg!r}"
    assert target.name not in msg, (
        f"target file name leaked into SnapshotError: {msg!r}"
    )

    # Data-loss guard: the preserved original must still exist.
    preserved = target.with_name(target.name + ".restore_preserved")
    assert preserved.exists(), (
        "preserved original was deleted on re-acquire failure (data-loss)"
    )
