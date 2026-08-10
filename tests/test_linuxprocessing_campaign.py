"""Task 8b: Linuxprocessing campaign runner tests.

The runner is a stdlib-only, on-host evidence harness for G0-G8 that never
contacts production, never prints private paths/manifests/contents/credentials,
and uses content-free allowlist-projected JSON reports with 0700 dirs / 0600
files and a self content-free assertion before writing.

Tests cover the full RED->GREEN ladder mandated by the Task 8b brief across
seven commits. Helpers are inlined per the brief's helper policy.
"""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path
from typing import Any


import scripts.linuxprocessing_campaign as lpc


# ---------------------------------------------------------------------------
# Allowlist used across every content-freedom assertion in this file.
# A report may only contain keys drawn from this set.
# ---------------------------------------------------------------------------
REPORT_KEYS = frozenset(
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

FORBIDDEN_FRAGMENTS = (
    "/home/bell",
    "/Users/",
    "MANIFEST:",
    "RECEIPT BODY:",
    "APPROVAL RECEIPT:",
    "api_key",
    "sk-",
    "password",
    "BEGIN IMMEDIATE",
)


def _run_stage(
    stage: str,
    trial_root: Path,
    monkeypatch,
    *extra: str,
) -> tuple[int, Path]:
    """Invoke a stage in-process; return (exit_code, report_path)."""
    report_path = trial_root / "report.json"
    argv = [
        "linuxprocessing_campaign.py",
        "--trial-root",
        str(trial_root),
        "--report",
        str(report_path),
        "--stage",
        stage,
    ]
    argv.extend(extra)
    monkeypatch.setattr(sys, "argv", argv)
    code = lpc.main()
    return code, report_path


def _read_report(path: Path) -> dict[str, Any]:
    assert path.exists(), f"report not written at {path}"
    # Mode must be 0600.
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with path.open() as f:
        return json.load(f)


def _assert_content_free(blob: str) -> None:
    low = blob.lower()
    for forbidden in FORBIDDEN_FRAGMENTS:
        assert forbidden.lower() not in low, f"report leaked {forbidden!r}"


def _assert_allowlist(report: dict[str, Any]) -> None:
    for key in report:
        assert key in REPORT_KEYS, f"non-allowlist key {key!r}"


# ===========================================================================
# Commit 1: skeleton, report writer, exit mapping, mode enforcement
# ===========================================================================


class TestSkeleton:
    def test_report_dir_is_0700_file_is_0600(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        code, report_path = _run_stage("g0", trial, monkeypatch)
        # Even on a non-pass, the report must be written with the right modes.
        assert report_path.exists()
        # Parent dir created by runner must be 0700.
        assert stat.S_IMODE(report_path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(report_path.stat().st_mode) == 0o600

    def test_report_is_allowlist_json_and_content_free(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        _, report_path = _run_stage("g0", trial, monkeypatch)
        report = _read_report(report_path)
        _assert_allowlist(report)
        _assert_content_free(json.dumps(report))

    def test_exit_code_pass_is_zero_fail_is_one_gate_is_two(self):
        # The verdict->exit mapping is the skeleton's contract.
        assert lpc._verdict_to_exit(lpc.PASS) == 0
        assert lpc._verdict_to_exit(lpc.FAIL) == 1
        assert lpc._verdict_to_exit(lpc.GATE) == 2

    def test_invalid_stage_exits_one(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        report_path = trial / "r.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "lpc.py",
                "--trial-root",
                str(trial),
                "--report",
                str(report_path),
                "--stage",
                "g99",
            ],
        )
        assert lpc.main() == 1

    def test_unknown_error_is_static_and_content_free(self, tmp_path, monkeypatch):
        """The unexpected-error path must emit a static reason code only."""
        trial = tmp_path / "trial"
        trial.mkdir()
        report_path = trial / "r.json"

        def _boom(*_a, **_kw):
            raise RuntimeError("secret path /home/bell leaked")

        monkeypatch.setattr(lpc, "_stage_g0", _boom)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "lpc.py",
                "--trial-root",
                str(trial),
                "--report",
                str(report_path),
                "--stage",
                "g0",
            ],
        )
        code = lpc.main()
        assert code == 1
        report = _read_report(report_path)
        assert report["reason_code"] == "unexpected_error"
        _assert_content_free(json.dumps(report))
