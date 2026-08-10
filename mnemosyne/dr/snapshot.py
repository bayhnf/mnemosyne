"""Isolated, read-only SQLite snapshot and restore.

Task 5 (supplemental plan): a page-level snapshot helper that opens the
source database in read-only + query-only mode, copies it via
``sqlite3.Connection.backup()``, forces ``journal_mode=DELETE`` so fresh
consumers never inherit WAL/SHM sidecars, and writes a SHA-256 sidecar
restricted to ``0600``. Restore reuses ``mnemosyne.dr.recovery``'s staged
atomic-replace safeguards: writer lock, staged rebuild, fsync, atomic
replace, post-replace integrity check with rollback.

This module deliberately does NOT call ``recovery.create_backup()`` /
``recovery.restore_backup()`` (those serialize via ``iterdump()`` and load
sqlite-vec); the snapshot path is a raw page copy that must work in the
sqlite-vec-unavailable trial lane.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Dict

from mnemosyne.dr import recovery

# Each data/checksum artifact is created with these restrictive bits and the
# mode is verified immediately after the chmod/fchmod call. Asserting (rather
# than just setting) catches a filesystem/umask that silently weakens perms.
_DATA_FILE_MODE = 0o600
_DIR_MODE = 0o700


class SnapshotError(RuntimeError):
    """Structured, content-free failure of a snapshot operation.

    Raised for every validation/IO failure so callers never see file paths,
    SQL fragments, or byte data from the source database in the public
    message. ``__cause__`` is preserved for diagnostics.
    """


def _assert_mode(path: Path, expected: int) -> None:
    """Verify on-disk mode bits match ``expected``; fail closed otherwise.

    Uses an explicit raise rather than ``assert`` (which ``python -O`` strips,
    silently disabling the fail-closed check). The message is content-free
    (no path fragment) so private paths never leak via this error path.
    """
    import stat

    actual = stat.S_IMODE(path.stat().st_mode)
    if actual != expected:
        raise SnapshotError(
            f"file mode check failed: expected {oct(expected)} got {oct(actual)}"
        )


def _open_ro_source(db_path: Path) -> sqlite3.Connection:
    """Open the source DB strictly read-only and forbid any SQL text writes.

    ``mode=ro`` rejects writes at the SQLite VFS layer; ``query_only=ON``
    additionally forbids schema/DML on this connection even if a code path
    later tries to execute SQL text. The only PRAGMA executed on the source
    is ``query_only`` (a no-op configuration PRAGMA, not a data mutation).
    """
    uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.execute("PRAGMA query_only=ON")
    return conn


def _allocate_unique_snapshot(dest_dir: Path, stem: str) -> Path:
    """Reserve ``{stem}.pristine.sqlite`` (retry on collision) via
    ``O_CREAT | O_EXCL`` so two concurrent snapshots cannot pick the same
    destination and clobber each other. Mirrors recovery's allocation.
    """
    import secrets

    candidate = dest_dir / f"{stem}.pristine.sqlite"
    for _ in range(64):
        try:
            fd = os.open(
                str(candidate), os.O_WRONLY | os.O_CREAT | os.O_EXCL, _DATA_FILE_MODE
            )
            os.close(fd)
            return candidate
        except FileExistsError:
            candidate = dest_dir / f"{stem}.{secrets.token_hex(3)}.pristine.sqlite"
    raise SnapshotError("could not allocate a unique snapshot filename")


def _write_sidecar(checksum_path: Path, sha256: str) -> None:
    """Write the SHA-256 sidecar with mode ``0600``. Contains the hash only —
    never the source path — so a leaked sidecar doesn't leak the originating
    database path.
    """
    fd = os.open(
        str(checksum_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _DATA_FILE_MODE
    )
    with os.fdopen(fd, "w") as f:
        f.write(sha256)
        f.flush()
        os.fsync(f.fileno())
    os.chmod(checksum_path, _DATA_FILE_MODE)
    _assert_mode(checksum_path, _DATA_FILE_MODE)


def _cleanup(paths):
    """Remove only the paths created by the current call."""
    for p in paths:
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass


def create_isolated_snapshot(db_path: Path, dest_dir: Path) -> Dict[str, "str | int"]:
    """Create a read-only page-level snapshot of ``db_path`` under ``dest_dir``.

    Security invariants:

      * The source is opened ``mode=ro`` + ``PRAGMA query_only=ON``; no SQL
        text is executed against it beyond the ``query_only`` PRAGMA.
      * The destination dir is created ``0700``; the snapshot DB and SHA-256
        sidecar are created ``0600`` and mode-verified after chmod.
      * Before closing the destination, full ``PRAGMA integrity_check`` must
        be ``ok`` and then ``PRAGMA journal_mode=DELETE`` is forced. A
        page-level backup of a WAL source otherwise preserves the WAL header
        and a later read can create ``-wal``/``-shm`` sidecars.
      * The SHA-256 sidecar contains the hash only (no source path).
      * On any failure, only paths created by THIS call are removed and a
        content-free ``SnapshotError`` is raised.
    """
    db_path = Path(db_path)
    dest_dir = Path(dest_dir)

    if not db_path.exists():
        raise SnapshotError("source database not found")

    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(dest_dir, _DIR_MODE)
        _assert_mode(dest_dir, _DIR_MODE)
    except OSError as exc:
        raise SnapshotError("could not prepare destination directory") from exc

    snapshot_path = _allocate_unique_snapshot(dest_dir, db_path.stem)
    checksum_path = snapshot_path.with_name(snapshot_path.name + ".sha256")
    created = [snapshot_path, checksum_path]

    src_conn = None
    dst_conn = None
    try:
        src_conn = _open_ro_source(db_path)
        dst_conn = sqlite3.connect(str(snapshot_path))
        os.chmod(snapshot_path, _DATA_FILE_MODE)
        _assert_mode(snapshot_path, _DATA_FILE_MODE)

        # Copy pages. A corrupt source (not a real SQLite file) raises here.
        src_conn.backup(dst_conn)

        # Full integrity check BEFORE flipping journal mode, so a corrupt
        # copy is caught before any further mutation of the destination.
        integrity = dst_conn.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise SnapshotError("snapshot failed integrity_check")

        # Mandatory: flip journal mode to DELETE on the OPEN connection so the
        # WAL header does not persist in the page copy. Without this, a
        # snapshot of a WAL-mode source leaves the destination in WAL mode
        # and the next consumer recreates -wal/-shm sidecars.
        dst_conn.execute("PRAGMA journal_mode=DELETE")

        dst_conn.commit()
        dst_conn.close()
        dst_conn = None
        src_conn.close()
        src_conn = None

        # fsync data file before hashing so the checksum covers durable bytes.
        recovery._fsync_path(snapshot_path)

        sha256 = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
        _write_sidecar(checksum_path, sha256)

        recovery._fsync_path(checksum_path)
        recovery._fsync_dir(dest_dir)

        return {
            "snapshot_path": str(snapshot_path),
            "checksum_path": str(checksum_path),
            "sha256": sha256,
            "integrity_check": True,
        }
    except Exception as exc:
        _cleanup(created)
        if isinstance(exc, SnapshotError):
            raise
        raise SnapshotError("snapshot creation failed") from exc
    finally:
        for conn in (dst_conn, src_conn):
            if conn is not None:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass


def restore_isolated_snapshot(
    snapshot_path: Path, target_path: Path
) -> Dict[str, "str | int"]:
    """Restore a snapshot onto ``target_path`` with staged atomic replace.

    Sequence:

      1. Verify the sidecar checksum BEFORE any target mutation. Missing,
         unreadable, or tampered sidecar -> refuse.
      2. Reject active ``-wal``/``-shm`` sidecars at the target.
      3. Acquire and HOLD an exclusive writer lock on the target through
         ``os.replace`` (reuses recovery's held-lock safeguard).
      4. Rebuild the snapshot into a uniquely-named staged DB via
         ``Connection.backup()``, force ``journal_mode=DELETE``, require
         ``integrity_check == "ok"`` before the atomic replace.
      5. Preserve the original target, ``os.replace`` the staged file onto
         the target, then release the old-inode writer lock and re-acquire
         ``BEGIN IMMEDIATE`` on the replacement inode (the held lock does not
         follow the path across replace). Fsync the parent dir after
         re-acquisition. On re-acquisition failure, surface a content-free
         ``SnapshotError`` and retain the preserved original.
      6. Post-replace integrity check under the new-inode lock; on failure
         restore the preserved original in place and raise.

    Target is preserved on any post-check failure; pre-check failures leave
    the target untouched. Only paths created by this call are cleaned up.
    """
    snapshot_path = Path(snapshot_path)
    target_path = Path(target_path)

    if not snapshot_path.exists():
        raise SnapshotError("snapshot not found")

    checksum_path = snapshot_path.with_name(snapshot_path.name + ".sha256")
    if not checksum_path.exists():
        raise SnapshotError("checksum sidecar not found")

    # --- 1. Verify sidecar BEFORE any target mutation --------------------
    # The sidecar must contain exactly one whitespace-separated token: the
    # 64-char lowercase hex SHA-256. Extra tokens (trailing junk) are rejected
    # so a tampered sidecar cannot pass merely by prefixing the correct hash.
    try:
        raw = checksum_path.read_text()
    except OSError as exc:
        raise SnapshotError("checksum sidecar unreadable") from exc
    tokens = raw.split()
    if len(tokens) != 1:
        raise SnapshotError("checksum sidecar malformed")
    expected = tokens[0].strip()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise SnapshotError("checksum sidecar malformed")
    actual = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    if actual != expected:
        raise SnapshotError("checksum mismatch")

    # --- 2. Reject active target sidecars --------------------------------
    # recovery's helpers raise RuntimeError with the full target path embedded;
    # convert to a content-free SnapshotError so no raw private path escapes.
    try:
        recovery._reject_active_sidecars(target_path)
    except Exception as exc:
        raise SnapshotError(
            "active target sidecar present; quiesce writers first"
        ) from exc

    # --- 3. Acquire + hold the writer lock through replace ---------------
    try:
        lock_conn = recovery._acquire_writer_lock(target_path)
    except Exception as exc:
        raise SnapshotError("could not acquire exclusive writer lock") from exc
    staged_path = recovery._unique_staged_path(target_path)
    preserved_path = target_path.with_name(target_path.name + ".restore_preserved")
    preserved_existed = target_path.exists()
    created = [staged_path]

    try:
        # --- 4. Rebuild into staged DB, force DELETE, integrity check ----
        # Open the snapshot source read-only (mode=ro + query_only=ON) so a
        # restore can never mutate the snapshot file (no checkpoint, no
        # journal-mode flip writing back to the source).
        src = _open_ro_source(snapshot_path)
        staged = sqlite3.connect(str(staged_path))
        try:
            src.backup(staged)
            staged.execute("PRAGMA journal_mode=DELETE")
            integrity = staged.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise SnapshotError("staged restore failed integrity_check")
            staged.commit()
        finally:
            try:
                staged.close()
            except sqlite3.Error:
                pass
            try:
                src.close()
            except sqlite3.Error:
                pass

        recovery._fsync_path(staged_path)

        # --- 5. Preserve original, atomic replace -------------------------
        # ponytail: rename changes SQLite's locked inode; re-acquire the native
        # lock immediately after replace. A zero-window design needs an in-place
        # restore or a separately governed sentinel lock.
        if preserved_existed:
            shutil.copy2(target_path, preserved_path)
            # The preserved copy is an artifact of THIS call, so it must be
            # tracked for failure-path cleanup (post-replace integrity check
            # can still fail after os.replace succeeds).
            created.append(preserved_path)
        os.replace(staged_path, target_path)
        created.remove(staged_path)  # consumed by replace
        # The old-inode lock no longer covers the path; release it and
        # re-acquire on the replacement inode BEFORE any fsync or verify, so
        # the vulnerable interval is just the release+acquire syscall pair.
        # The staged bytes were fsynced pre-replace; dir fd is a different
        # inode and safe to fsync under the held file lock.
        lock_conn = recovery._release_writer_lock(lock_conn)
        try:
            lock_conn = recovery._reacquire_writer_lock(target_path)
        except Exception as exc:
            # Content-free: no raw target path escapes the public exception.
            # Retain the preserved original: exception cleanup would otherwise
            # delete the only remaining copy of the original while the target
            # holds an image a competitor may have touched.
            if preserved_path in created:
                created.remove(preserved_path)
            raise SnapshotError("could not reacquire exclusive writer lock") from exc
        recovery._fsync_dir(target_path.parent)

        # --- 6. Post-replace integrity; rollback on failure --------------
        if not recovery.verify_integrity(target_path):
            # Rollback: copy the preserved original back over the target. If
            # THIS copy fails (disk full, I/O error), we must NOT delete the
            # preserved copy --- it is the only remaining original. Remove it
            # from the cleanup list so the failure handler cannot unlink it.
            if preserved_existed and preserved_path.exists():
                try:
                    shutil.copy2(preserved_path, target_path)
                    recovery._fsync_path(target_path)
                    recovery._fsync_dir(target_path.parent)
                except OSError:
                    # Protect the only surviving copy of the original.
                    if preserved_path in created:
                        created.remove(preserved_path)
            raise SnapshotError("post-replace integrity_check failed")
    except Exception as exc:
        _cleanup(created)
        if isinstance(exc, SnapshotError):
            raise
        raise SnapshotError("restore failed") from exc
    finally:
        lock_conn = recovery._release_writer_lock(lock_conn)

    return {
        "restored": True,
        "snapshot_used": str(snapshot_path),
        "database_path": str(target_path),
        "integrity_check": True,
        "sha256": expected,
        "preserved_original": str(preserved_path) if preserved_existed else None,
    }
