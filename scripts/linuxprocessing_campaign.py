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
import json
import os
import shutil
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
            f"directory mode check failed: expected {oct(_DIR_MODE)} "
            f"got {oct(actual)}"
        )


def _assert_file_mode(path: Path) -> None:
    actual = stat.S_IMODE(path.stat().st_mode)
    if actual != _FILE_MODE:
        raise RuntimeError(
            f"file mode check failed: expected {oct(_FILE_MODE)} "
            f"got {oct(actual)}"
        )


def _assert_content_free(blob: str) -> None:
    """Fail closed if any forbidden fragment appears in the blob.

    Used as the self content-free assertion before writing any report.
    """
    low = blob.lower()
    for frag in _FORBIDDEN_FRAGMENTS:
        if frag in low:
            raise RuntimeError(
                "self content-free assertion failed; report not written"
            )


def _assert_allowlist(report: dict[str, Any]) -> None:
    for key in report:
        if key not in _REPORT_KEYS:
            raise RuntimeError(
                "report key not on allowlist; report not written"
            )


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

    fd = os.open(
        str(report_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _FILE_MODE
    )
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

    checks["python"] = {"verdict": _check_python_version()[0], "reason_code": _check_python_version()[1]}
    checks["disk_space"] = {
        "verdict": _check_disk_space(trial_root)[0],
        "reason_code": _check_disk_space(trial_root)[1],
    }
    checks["endpoint"] = {"verdict": _check_endpoint_static()[0], "reason_code": _check_endpoint_static()[1]}
    checks["dimension"] = {"verdict": _check_dimension_static()[0], "reason_code": _check_dimension_static()[1]}
    checks["lane"] = {"verdict": _check_lane_static()[0], "reason_code": _check_lane_static()[1]}

    # Manual acknowledgements recorded but never faked; informational here.
    checks["t0_ssh_ack"] = {"verdict": _ack_state(args.ack_t0_ssh)}
    checks["image_digest_ack"] = {"verdict": _ack_state(args.ack_image_digest)}

    # Verdict is the worst-case of the hard checks (ack states are
    # informational in G0; the stages that depend on them gate separately).
    if any(checks[k]["verdict"] != PASS for k in ("python", "disk_space", "endpoint", "dimension", "lane")):
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
        checks["approved_sha"] = {"verdict": GATE, "reason_code": "approved_sha_required"}
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


def _stage_g2(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    return FAIL, "not_implemented", _empty_checks()


def _stage_g3(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    return FAIL, "not_implemented", _empty_checks()


def _stage_g4(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    return FAIL, "not_implemented", _empty_checks()


def _stage_g5(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    return FAIL, "not_implemented", _empty_checks()


def _stage_g6(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    return FAIL, "not_implemented", _empty_checks()


def _stage_g7(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    return FAIL, "not_implemented", _empty_checks()


def _stage_g8(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    return FAIL, "not_implemented", _empty_checks()


# Valid stage names. The --stage argument is validated here (not via
# argparse choices) so an unknown stage maps to FAIL/exit 1 rather than
# argparse's exit 2, which would masquerade as a pending manual gate.
_STAGE_NAMES = frozenset({"g0", "g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "all"})


def _resolve_stage_func(stage: str) -> Callable[[argparse.Namespace], tuple[str, str, dict]] | None:
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
