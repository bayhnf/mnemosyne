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


def _stage_g0(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    return FAIL, "not_implemented", _empty_checks()


def _stage_g1(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    return FAIL, "not_implemented", _empty_checks()


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
