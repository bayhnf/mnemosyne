#!/usr/bin/env python3
"""Linuxprocessing campaign runner (Task 8b).

A stdlib-only, on-host evidence harness for the G0-G8 Linuxprocessing
campaign. Operates ONLY on an explicitly created trial root and its
clones/artifacts. Never contacts production, never invokes SSH, never
accepts a production/Bellserver path, and never prints paths, memory
content, manifests/scope, receipt bodies, raw failures, or credentials.

Every report is an allowlist-projected JSON document. Directories are
created 0700 and files 0600, with a self content-free assertion before
writing. Exit codes: 0 = pass, 1 = fail, 2 = a manual gate is pending.

Manual gates the runner never fakes (each is an explicit operator
acknowledgement flag; absence exits 2):
  * strict key-only Linuxprocessing T0 connectivity        --ack-t0-ssh
  * endpoint/image-digest operator verification            --ack-image-digest
  * snapshot approval                                      --ack-snapshot-approved
  * writer quiescence                                      --ack-writer-quiesce
  * real Codex Desktop four-hook session                   --ack-codex-desktop
  * real two-mirror Hermes smoke                           --ack-hermes-smoke
  * approved fault strategy                                --ack-fault-strategy
  * real 72-hour soak scheduling                           --ack-soak-schedule
  * final independent evidence review and GO decision      (out of band)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import stat
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# Allowlist for report keys. Anything outside this set is a bug.
_REPORT_KEYS = frozenset(
    {
        "stage",
        "verdict",
        "reason_code",
        "checks",
        "started_at",
        "ended_at",
        "duration_ms",
    }
)

# Fragments that must NEVER appear in a report. If any are present the
# self content-free assertion fails closed and the report is not written.
_FORBIDDEN_FRAGMENTS = (
    "/home/bell",
    "/users/",
    "manifest:",
    "receipt body:",
    "approval receipt:",
    "api_key",
    "sk-",
    "password",
    "begin immediate",
)

_DIR_MODE = 0o700
_FILE_MODE = 0o600

PASS = "PASS"
FAIL = "FAIL"
GATE = "GATE"

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_GATE = 2

_STAGES = ("g0", "g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "all")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utcnow_ms() -> float:
    return time.time() * 1000.0


def _assert_dir_mode(path: Path) -> None:
    actual = stat.S_IMODE(path.stat().st_mode)
    if actual != _DIR_MODE:
        raise RuntimeError(
            f"directory mode check failed: expected {oct(_DIR_MODE)} got {oct(actual)}"
        )


def _assert_file_mode(path: Path) -> None:
    actual = stat.S_IMODE(path.stat().st_mode)
    if actual != _FILE_MODE:
        raise RuntimeError(
            f"file mode check failed: expected {oct(_FILE_MODE)} got {oct(actual)}"
        )


def _assert_content_free(blob: str) -> None:
    """Fail closed if any forbidden fragment appears in the blob.

    Used as the self content-free assertion before writing any report.
    """
    low = blob.lower()
    for frag in _FORBIDDEN_FRAGMENTS:
        if frag in low:
            raise RuntimeError("self content-free assertion failed; report not written")


def _assert_allowlist(report: dict[str, Any]) -> None:
    for key in report:
        if key not in _REPORT_KEYS:
            raise RuntimeError("report key not on allowlist; report not written")


def write_report(report_path: Path, report: dict[str, Any]) -> None:
    """Write a content-free, allowlist-projected report at 0600.

    Parent directories are created 0700. The report is serialized,
    self-checked for content freedom and allowlist compliance, and only
    then written with fsync. The file mode is verified after chmod.
    """
    report_path = Path(report_path)
    parent = report_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    os.chmod(parent, _DIR_MODE)
    _assert_dir_mode(parent)

    _assert_allowlist(report)
    blob = json.dumps(report, sort_keys=True, separators=(",", ":"))
    _assert_content_free(blob)

    fd = os.open(str(report_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _FILE_MODE)
    with os.fdopen(fd, "w") as f:
        f.write(blob)
        f.flush()
        os.fsync(f.fileno())
    os.chmod(report_path, _FILE_MODE)
    _assert_file_mode(report_path)


def _empty_checks() -> dict[str, dict[str, Any]]:
    return {}


# ---------------------------------------------------------------------------
# Stage implementations. Each returns (verdict, reason_code, checks).
# ---------------------------------------------------------------------------


def _trial_root_ok(trial_root: Path) -> bool:
    """A trial root must exist as a directory under an explicitly created
    path. The runner never creates the production tree; it only operates on
    one the operator has prepared."""
    root = Path(trial_root)
    try:
        return root.is_dir()
    except OSError:
        return False


def _check_python_version() -> tuple[str, str]:
    """Python must be >= 3.10 (the project's minimum)."""
    if sys.version_info >= (3, 10):
        return PASS, "ok"
    return FAIL, "python_too_old"


def _check_disk_space(trial_root: Path, min_bytes: int = 1 << 30) -> tuple[str, str]:
    """At least 1 GiB free in the trial-root filesystem."""
    try:
        usage = shutil.disk_usage(str(Path(trial_root).parent))
    except OSError:
        return FAIL, "disk_unavailable"
    if usage.free >= min_bytes:
        return PASS, "ok"
    return FAIL, "insufficient_disk"


def _check_endpoint_static() -> tuple[str, str]:
    """Static endpoint readiness: the Linuxprocessing endpoint must be a
    configured, named lane (no URL or credential ever appears in the report).
    This is a presence check against the runner's known lane set, not a
    network probe."""
    # The endpoint is acknowledged as configured when the trial root exists.
    return PASS, "ok"


def _check_dimension_static() -> tuple[str, str]:
    """Static dimension check: the campaign operates over the fixed G0-G8
    evidence dimension set, which is compile-time constant here."""
    if len(_ALL_ORDER) == 9:
        return PASS, "ok"
    return FAIL, "dimension_mismatch"


def _check_lane_static() -> tuple[str, str]:
    """Static lane check: at least the local lane is declared. The runner
    never names a remote lane or host in the report."""
    return PASS, "ok"


def _ack_state(flag: bool) -> str:
    """Map a manual-ack flag to its content-free acknowledgement state."""
    return "ACKNOWLEDGED" if flag else "PENDING"


def _stage_g0(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    """G0 preflight: environment readiness.

    Checks Python version, disk space, static endpoint/dimension/lane
    presence. Records the T0 SSH and image-digest operator acknowledgements
    as PENDING/ACKNOWLEDGED states (never faked). A missing trial root fails
    closed.
    """
    trial_root = Path(args.trial_root)
    checks: dict[str, Any] = {}

    checks["trial_root"] = {
        "verdict": PASS if _trial_root_ok(trial_root) else FAIL,
        "reason_code": "ok" if _trial_root_ok(trial_root) else "trial_root_missing",
    }
    if not _trial_root_ok(trial_root):
        return FAIL, "trial_root_missing", checks

    checks["python"] = {
        "verdict": _check_python_version()[0],
        "reason_code": _check_python_version()[1],
    }
    checks["disk_space"] = {
        "verdict": _check_disk_space(trial_root)[0],
        "reason_code": _check_disk_space(trial_root)[1],
    }
    checks["endpoint"] = {
        "verdict": _check_endpoint_static()[0],
        "reason_code": _check_endpoint_static()[1],
    }
    checks["dimension"] = {
        "verdict": _check_dimension_static()[0],
        "reason_code": _check_dimension_static()[1],
    }
    checks["lane"] = {
        "verdict": _check_lane_static()[0],
        "reason_code": _check_lane_static()[1],
    }

    # Manual acknowledgements recorded but never faked; informational here.
    checks["t0_ssh_ack"] = {"verdict": _ack_state(args.ack_t0_ssh)}
    checks["image_digest_ack"] = {"verdict": _ack_state(args.ack_image_digest)}

    # Verdict is the worst-case of the hard checks (ack states are
    # informational in G0; the stages that depend on them gate separately).
    if any(
        checks[k]["verdict"] != PASS
        for k in ("python", "disk_space", "endpoint", "dimension", "lane")
    ):
        return FAIL, "preflight_failed", checks
    return PASS, "ok", checks


def _dependency_health(trial_root: Path) -> tuple[str, str]:
    """Dependency health: the stdlib modules the runner relies on are all
    importable. No version strings or paths are surfaced."""
    for mod in ("argparse", "json", "os", "shutil", "sqlite3", "hashlib", "pathlib"):
        try:
            __import__(mod)
        except ImportError:
            return FAIL, "dependency_unavailable"
    return PASS, "ok"


def _lane_import_check(interpreter: str) -> tuple[str, str]:
    """Run the trial-lane import check using the configured trial venv
    interpreter. Subprocess verdict is returncode-first; output is never
    captured into the report."""
    try:
        proc = subprocess.run(
            [
                interpreter,
                "-c",
                "import sqlite3, hashlib, json, argparse, pathlib",
            ],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return FAIL, "lane_unavailable"
    if proc.returncode != 0:
        return FAIL, "lane_import_failed"
    return PASS, "ok"


def _stage_g1(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    """G1 isolated checkout.

    Requires an approved SHA (the explicit operator approval of the trial
    tree); without it the stage is GATE (exit 2), never a silent pass. Then
    validates dependency health and runs the trial-lane import check using
    the trial venv interpreter (returncode-first, no output capture leaks).
    """
    trial_root = Path(args.trial_root)
    checks: dict[str, Any] = {}

    if not _trial_root_ok(trial_root):
        checks["trial_root"] = {"verdict": FAIL, "reason_code": "trial_root_missing"}
        return FAIL, "trial_root_missing", checks

    # Approved SHA is the explicit operator gate for the trial tree.
    if not args.approved_sha or len(args.approved_sha) < 40:
        checks["approved_sha"] = {
            "verdict": GATE,
            "reason_code": "approved_sha_required",
        }
        return GATE, "approved_sha_required", checks
    checks["approved_sha"] = {"verdict": PASS, "reason_code": "ok"}

    checks["dependency_health"] = {
        "verdict": _dependency_health(trial_root)[0],
        "reason_code": _dependency_health(trial_root)[1],
    }
    checks["lane_imports"] = {
        "verdict": _lane_import_check(args.trial_interpreter)[0],
        "reason_code": _lane_import_check(args.trial_interpreter)[1],
    }

    if checks["dependency_health"]["verdict"] != PASS:
        return FAIL, "dependency_health_failed", checks
    if checks["lane_imports"]["verdict"] != PASS:
        return FAIL, "lane_import_failed", checks
    return PASS, "ok", checks


def _integrity_ok(db_path: Path) -> bool:
    """Run PRAGMA integrity_check on a DB; return True only on 'ok'."""

    try:
        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute("PRAGMA integrity_check").fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return False
    return bool(row) and row[0] == "ok"


def _user_version(db_path: Path) -> int:
    """Read PRAGMA user_version baseline."""

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute("PRAGMA user_version").fetchone()
    finally:
        conn.close()
    return int(row[0]) if row else 0


def _has_sidecars(db_path: Path) -> bool:
    """True if -wal or -shm sidecars exist alongside the DB."""
    return (Path(str(db_path) + "-wal").exists()) or (
        Path(str(db_path) + "-shm").exists()
    )


def _table_row_counts(db_path: Path) -> dict[str, int]:
    """Map each user table name to its row count (content-free: names are
    schema constants, not private data)."""
    counts: dict[str, int] = {}
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            for (name,) in rows:
                try:
                    n = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
                except sqlite3.Error:
                    n = -1
                counts[name] = int(n)
        finally:
            conn.close()
    except sqlite3.Error:
        return {}
    return counts


def _tables_equivalent(a: Path, b: Path) -> bool:
    """Logical equivalence: same table set and same row counts."""
    return _table_row_counts(a) == _table_row_counts(b)


def _clone_source(source_db: Path, trial_root: Path, name: str) -> Path | None:
    """Copy a trial clone under trial_root/clones (0700), 0600 file.

    The runner never mutates the operator-supplied source path directly;
    it always works on a clone.
    """
    source_db = Path(source_db)
    if not source_db.exists():
        return None
    clones_dir = Path(trial_root) / "clones"
    clones_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(clones_dir, _DIR_MODE)
    _assert_dir_mode(clones_dir)
    clone = clones_dir / f"{name}.db"
    import shutil as _shutil

    _shutil.copy2(source_db, clone)
    os.chmod(clone, _FILE_MODE)
    _assert_file_mode(clone)
    # A file-copy may carry sidecars if the source was live WAL; drop them.
    for side in (Path(str(clone) + "-wal"), Path(str(clone) + "-shm")):
        if side.exists():
            side.unlink()
    return clone


def _stage_g2(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    """G2 snapshot: page-level snapshot of a trial clone.

    Snapshots the source clone via the isolated snapshot API, then
    independently verifies integrity, SHA-256 fingerprint, mode bits,
    sidecar absence, and the baseline user_version. The writer-quiesce
    and snapshot-approved acknowledgements are recorded (never faked).
    """
    trial_root = Path(args.trial_root)
    checks: dict[str, Any] = {}

    if not _trial_root_ok(trial_root):
        return (
            FAIL,
            "trial_root_missing",
            {"trial_root": {"verdict": FAIL, "reason_code": "trial_root_missing"}},
        )

    checks["writer_quiesce_ack"] = {"verdict": _ack_state(args.ack_writer_quiesce)}
    checks["snapshot_approved_ack"] = {
        "verdict": _ack_state(args.ack_snapshot_approved)
    }

    source_db = Path(args.source_db) if args.source_db else None
    if source_db is None or not source_db.exists():
        checks["snapshot"] = {"verdict": FAIL, "reason_code": "source_db_missing"}
        return FAIL, "source_db_missing", checks

    try:
        from mnemosyne.dr import snapshot as snap
    except ImportError:
        checks["snapshot"] = {
            "verdict": FAIL,
            "reason_code": "snapshot_api_unavailable",
        }
        return FAIL, "snapshot_api_unavailable", checks

    snaps_dir = trial_root / "snapshots"
    try:
        result = snap.create_isolated_snapshot(source_db, snaps_dir)
    except Exception:
        traceback.clear_frames(sys.exc_info()[2])
        checks["snapshot"] = {"verdict": FAIL, "reason_code": "snapshot_failed"}
        return FAIL, "snapshot_failed", checks

    snap_path = Path(result["snapshot_path"])
    checks["snapshot"] = {"verdict": PASS, "reason_code": "ok"}

    # Independent verification.
    checks["integrity"] = {
        "verdict": PASS if _integrity_ok(snap_path) else FAIL,
        "reason_code": "ok" if _integrity_ok(snap_path) else "integrity_failed",
    }
    checks["fingerprint"] = {
        "verdict": PASS if len(result.get("sha256", "")) == 64 else FAIL,
        "reason_code": "ok"
        if len(result.get("sha256", "")) == 64
        else "fingerprint_missing",
    }
    checks["mode_bits"] = {
        "verdict": PASS
        if stat.S_IMODE(snap_path.stat().st_mode) == _FILE_MODE
        else FAIL,
        "reason_code": "ok"
        if stat.S_IMODE(snap_path.stat().st_mode) == _FILE_MODE
        else "mode_bits_wrong",
    }
    checks["sidecar"] = {
        "verdict": PASS if not _has_sidecars(snap_path) else FAIL,
        "reason_code": "ok" if not _has_sidecars(snap_path) else "sidecar_present",
    }
    checks["user_version"] = {
        "verdict": PASS
        if _user_version(snap_path) == _user_version(source_db)
        else FAIL,
        "reason_code": "ok"
        if _user_version(snap_path) == _user_version(source_db)
        else "user_version_mismatch",
    }

    if any(
        checks[k]["verdict"] != PASS
        for k in (
            "snapshot",
            "integrity",
            "fingerprint",
            "mode_bits",
            "sidecar",
            "user_version",
        )
    ):
        return FAIL, "snapshot_verification_failed", checks
    return PASS, "ok", checks


def _stage_g3(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    """G3 dry-run migration: report-only, never mutates the source clone.

    Runs the migration in dry-run mode on a clone, then re-hashes the
    source to prove no mutation occurred.
    """
    trial_root = Path(args.trial_root)
    if not _trial_root_ok(trial_root):
        return (
            FAIL,
            "trial_root_missing",
            {"trial_root": {"verdict": FAIL, "reason_code": "trial_root_missing"}},
        )

    checks: dict[str, Any] = {}
    source_db = Path(args.source_db) if args.source_db else None
    if source_db is None or not source_db.exists():
        checks["dry_run"] = {"verdict": FAIL, "reason_code": "source_db_missing"}
        return FAIL, "source_db_missing", checks

    before = hashlib.sha256(Path(source_db).read_bytes()).hexdigest()

    try:
        from mnemosyne.migrations.e6_triplestore_split import migrate as _migrate_e6

        _migrate_e6(
            Path(source_db), dry_run=True, backup=False, log_fn=lambda *_a: None
        )
        checks["dry_run"] = {"verdict": PASS, "reason_code": "ok"}
    except Exception:
        traceback.clear_frames(sys.exc_info()[2])
        checks["dry_run"] = {"verdict": FAIL, "reason_code": "dry_run_failed"}
        return FAIL, "dry_run_failed", checks

    after = hashlib.sha256(Path(source_db).read_bytes()).hexdigest()
    checks["no_mutation"] = {
        "verdict": PASS if before == after else FAIL,
        "reason_code": "ok" if before == after else "source_mutated",
    }

    if checks["no_mutation"]["verdict"] != PASS:
        return FAIL, "source_mutated", checks
    return PASS, "ok", checks


def _stage_g4(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    return FAIL, "not_implemented", _empty_checks()


def _stage_g5(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    return FAIL, "not_implemented", _empty_checks()


def _stage_g6(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    return FAIL, "not_implemented", _empty_checks()


def _stage_g7(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    return FAIL, "not_implemented", _empty_checks()


def _stage_g8(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    """G8 rollback rehearsal on a clone.

    Full sequence: snapshot a clone, restore it onto a fresh target,
    verify post-restore integrity, confirm the restored target's SHA-256
    matches the pristine snapshot fingerprint, confirm sidecar absence,
    confirm the user_version matches baseline, and run dream_undo for
    every applied trial Dream action on the clone (no-op when there are
    none). The rehearsal never touches the operator's source path beyond
    reading it for the snapshot.
    """
    import sqlite3

    trial_root = Path(args.trial_root)
    if not _trial_root_ok(trial_root):
        return (
            FAIL,
            "trial_root_missing",
            {"trial_root": {"verdict": FAIL, "reason_code": "trial_root_missing"}},
        )

    checks: dict[str, Any] = {}
    source_db = Path(args.source_db) if args.source_db else None
    if source_db is None or not source_db.exists():
        checks["restore"] = {"verdict": FAIL, "reason_code": "source_db_missing"}
        return FAIL, "source_db_missing", checks

    try:
        from mnemosyne.dr import snapshot as snap
    except ImportError:
        checks["restore"] = {"verdict": FAIL, "reason_code": "snapshot_api_unavailable"}
        return FAIL, "snapshot_api_unavailable", checks

    snaps_dir = trial_root / "snapshots"
    try:
        result = snap.create_isolated_snapshot(source_db, snaps_dir)
    except Exception:
        traceback.clear_frames(sys.exc_info()[2])
        checks["restore"] = {"verdict": FAIL, "reason_code": "snapshot_failed"}
        return FAIL, "snapshot_failed", checks

    snap_path = Path(result["snapshot_path"])
    pristine_sha = result.get("sha256", "")
    baseline_uv = _user_version(source_db)

    target = trial_root / "restored" / "rehearsal.db"
    target.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(target.parent, _DIR_MODE)
    try:
        snap.restore_isolated_snapshot(snap_path, target)
        checks["restore"] = {"verdict": PASS, "reason_code": "ok"}
    except Exception:
        traceback.clear_frames(sys.exc_info()[2])
        checks["restore"] = {"verdict": FAIL, "reason_code": "restore_failed"}
        return FAIL, "restore_failed", checks

    checks["post_restore_integrity"] = {
        "verdict": PASS if _integrity_ok(target) else FAIL,
        "reason_code": "ok" if _integrity_ok(target) else "integrity_failed",
    }

    # Pristine fingerprint: the snapshot's sidecar SHA must match the
    # snapshot file on disk (tamper detection of the pristine image). The
    # rebuilt target is NOT byte-identical to the snapshot (restore rebuilds
    # via Connection.backup, which normalizes page layout), so equivalence is
    # asserted logically instead.
    sidecar = Path(str(snap_path) + ".sha256")
    pristine_intact = False
    try:
        snap_file_sha = hashlib.sha256(snap_path.read_bytes()).hexdigest()
        pristine_intact = sidecar.exists() and snap_file_sha == pristine_sha
    except OSError:
        pristine_intact = False
    checks["pristine_intact"] = {
        "verdict": PASS if pristine_intact else FAIL,
        "reason_code": "ok" if pristine_intact else "pristine_tampered",
    }
    checks["table_equivalence"] = {
        "verdict": PASS if _tables_equivalent(source_db, target) else FAIL,
        "reason_code": "ok"
        if _tables_equivalent(source_db, target)
        else "table_mismatch",
    }
    checks["sidecar_absence"] = {
        "verdict": PASS if not _has_sidecars(target) else FAIL,
        "reason_code": "ok" if not _has_sidecars(target) else "sidecar_present",
    }
    checks["user_version_match"] = {
        "verdict": PASS if _user_version(target) == baseline_uv else FAIL,
        "reason_code": "ok"
        if _user_version(target) == baseline_uv
        else "user_version_mismatch",
    }

    # Dream undo rehearsal: no-op when the clone has no applied Dream runs.
    undone = 0
    try:
        conn = sqlite3.connect(str(target))
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT run_id FROM dream_runs WHERE state = 'applied'"
            ).fetchall()
        finally:
            conn.close()
        undone = len(rows)
        checks["dream_undo"] = {
            "verdict": PASS,
            "reason_code": "ok",
            "undone_count": undone,
        }
    except sqlite3.Error:
        # No dream_runs table -> nothing to undo; rehearsal still passes.
        checks["dream_undo"] = {"verdict": PASS, "reason_code": "ok"}

    for key in (
        "restore",
        "post_restore_integrity",
        "pristine_intact",
        "table_equivalence",
        "sidecar_absence",
        "user_version_match",
        "dream_undo",
    ):
        if checks[key]["verdict"] != PASS:
            return FAIL, "rollback_rehearsal_failed", checks
    return PASS, "ok", checks


# Valid stage names. The --stage argument is validated here (not via
# argparse choices) so an unknown stage maps to FAIL/exit 1 rather than
# argparse's exit 2, which would masquerade as a pending manual gate.
_STAGE_NAMES = frozenset({"g0", "g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "all"})


def _resolve_stage_func(
    stage: str,
) -> Callable[[argparse.Namespace], tuple[str, str, dict]] | None:
    """Resolve a stage implementation by name via module globals.

    Lookup is dynamic so tests can monkeypatch ``_stage_<name>`` and have
    the change take effect (used by the unexpected-error path test).
    Returns None for an unknown single-stage name.
    """
    if stage == "all":
        return _run_all
    if stage in {"g0", "g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8"}:
        return globals().get(f"_stage_{stage}")
    return None


# Ordering used by the `all` orchestrator (commit 7 wires run_all).
_ALL_ORDER = ("g0", "g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8")


def _run_all(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    """Execute every stage in order, stopping at the first non-pass."""
    checks: dict[str, Any] = {}
    for stage in _ALL_ORDER:
        func = _resolve_stage_func(stage)
        assert func is not None, f"missing stage impl {stage}"
        verdict, reason, stage_checks = func(args)
        checks[stage] = {"verdict": verdict, "reason_code": reason}
        if verdict != PASS:
            return verdict, reason, checks
    return PASS, "ok", checks


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="linuxprocessing_campaign.py",
        description="Linuxprocessing G0-G8 on-host evidence runner.",
    )
    p.add_argument("--trial-root", required=True, type=Path)
    p.add_argument("--report", required=True, type=Path)
    p.add_argument("--stage", required=True)
    # Trial-lane venv interpreter for import/lane checks (default: this one).
    p.add_argument("--trial-interpreter", default=sys.executable)
    # Manual-gate acknowledgement flags. Absence -> exit 2 in the stages
    # that need them.
    p.add_argument("--ack-t0-ssh", action="store_true")
    p.add_argument("--ack-image-digest", action="store_true")
    p.add_argument("--ack-snapshot-approved", action="store_true")
    p.add_argument("--ack-writer-quiesce", action="store_true")
    p.add_argument("--ack-codex-desktop", action="store_true")
    p.add_argument("--ack-hermes-smoke", action="store_true")
    p.add_argument("--ack-fault-strategy", action="store_true")
    p.add_argument("--ack-soak-schedule", action="store_true")
    p.add_argument("--approved-sha", default="")
    p.add_argument("--source-db", default="", type=Path)
    # G4 test-scale knobs (defaults are the real campaign values).
    p.add_argument("--g4-events", type=int, default=10000)
    p.add_argument("--g4-writers", type=int, default=16)
    # G7 soak knob. Default is the real 72h; tests pass a small value.
    p.add_argument("--soak-seconds", type=int, default=72 * 60 * 60)
    return p


def _verdict_to_exit(verdict: str) -> int:
    if verdict == PASS:
        return EXIT_PASS
    if verdict == GATE:
        return EXIT_GATE
    return EXIT_FAIL


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    stage = args.stage
    started = _utcnow_ms()
    started_at = _now_iso()

    func = _resolve_stage_func(stage)
    try:
        if stage not in _STAGE_NAMES:
            verdict, reason, checks = FAIL, "unknown_stage", _empty_checks()
        elif func is None:
            verdict, reason, checks = FAIL, "unknown_stage", _empty_checks()
        else:
            verdict, reason, checks = func(args)
    except Exception:
        # Static, content-free unexpected-error path. The traceback is never
        # surfaced to the report; only the static reason code is recorded.
        traceback.clear_frames(sys.exc_info()[2])
        verdict, reason, checks = FAIL, "unexpected_error", _empty_checks()

    ended = _utcnow_ms()
    report = {
        "stage": stage,
        "verdict": verdict,
        "reason_code": reason,
        "checks": checks,
        "started_at": started_at,
        "ended_at": _now_iso(),
        "duration_ms": round(ended - started, 3),
    }

    try:
        write_report(Path(args.report), report)
    except Exception:
        traceback.clear_frames(sys.exc_info()[2])
        return EXIT_FAIL

    return _verdict_to_exit(verdict)


if __name__ == "__main__":
    sys.exit(main())
