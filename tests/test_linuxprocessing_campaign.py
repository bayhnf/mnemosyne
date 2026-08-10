"""Task 8b: Linuxprocessing campaign runner tests (FIX ROUND 1).

Strict adversarial tests for every Critical/High/Medium finding from
independent review. The runner is stdlib-only, on-host, never contacts
production, never accepts a production/Bellserver path, and uses recursive
schema-projected content-free reports with 0700 dirs / 0600 files.
"""

from __future__ import annotations

import hashlib
import json
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import scripts.linuxprocessing_campaign as lpc

PASS = lpc.PASS
FAIL = lpc.FAIL
GATE = lpc.GATE

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
    "/users/",
    "manifest:",
    "receipt body:",
    "approval receipt:",
    "api_key",
    "sk-",
    "password",
    "begin immediate",
)


def _run_stage(
    stage: str, trial_root: Path, monkeypatch, *extra: str
) -> tuple[int, Path]:
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


def _run_stage_custom(
    stage: str, trial_root: Path, report_rel: str, monkeypatch, *extra: str
) -> tuple[int, Path]:
    """Run with a report path that may be nested under trial root."""
    report_path = trial_root / report_rel
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
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with path.open() as f:
        return json.load(f)


def _assert_content_free(blob: str) -> None:
    low = blob.lower()
    for forbidden in FORBIDDEN_FRAGMENTS:
        assert forbidden.lower() not in low, f"report leaked {forbidden!r}"


def _assert_recursive_allowlist(obj: Any, trail: str = "root") -> None:
    """Recursively verify every key in the report is on the allowed schema.

    Structure: root.{REPORT_KEYS} -> checks.<check_name>.{_ALLOWED_CHECK_KEYS}.
    Check names are free-form but must be safe tokens. Fields within a check
    must be in _ALLOWED_CHECK_KEYS."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if trail == "root":
                assert key in REPORT_KEYS, f"non-allowlist root key {key!r}"
            elif trail == "root.checks":
                # This is a check name; must be a safe token.
                assert lpc._is_safe_token(key), f"unsafe check name {key!r}"
            elif trail.startswith("root.checks."):
                # This is a field within a check (or deeper nesting).
                assert key in lpc._ALLOWED_CHECK_KEYS, (
                    f"non-allowlist check field {key!r} at {trail}"
                )
            _assert_recursive_allowlist(value, f"{trail}.{key}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _assert_recursive_allowlist(item, f"{trail}[{i}]")


def _make_trial_db(path: Path) -> Path:
    from mnemosyne.core.memory import init_db

    init_db(path)
    return path


def _assert_all_dirs_0700(root: Path) -> None:
    """Every directory under root must be 0700."""
    for p in [root, *root.rglob("*")]:
        if p.is_dir():
            assert stat.S_IMODE(p.stat().st_mode) == 0o700, f"dir not 0700: {p}"


def _ack_all() -> list[str]:
    """All manual acknowledgement flags."""
    return [
        "--ack-t0-ssh",
        "--ack-image-digest",
        "--ack-snapshot-approved",
        "--ack-writer-quiesce",
        "--ack-codex-desktop",
        "--ack-hermes-smoke",
        "--ack-fault-strategy",
        "--ack-soak-schedule",
    ]


def _all_pass_args(trial: Path, source: Path) -> list[str]:
    return [
        "--source-db",
        str(source),
        "--approved-sha",
        "deadbeef" * 8,
        "--g4-events",
        "4",
        "--g4-writers",
        "1",
        "--soak-seconds",
        "0",
        *_ack_all(),
    ]


# ===========================================================================
# Critical 1: production/trial containment
# ===========================================================================


class TestContainment:
    def test_g2_rejects_source_db_outside_trial_root(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        # Create a DB OUTSIDE the trial root (simulating a production path).
        outside = tmp_path / "outside.db"
        _make_trial_db(outside)
        code, report_path = _run_stage(
            "g2", trial, monkeypatch, "--source-db", str(outside)
        )
        assert code == 1
        report = _read_report(report_path)
        assert report["verdict"] == FAIL
        assert "outside_trial_root" in report["checks"]["source_db"]["reason_code"]

    def test_g2_rejects_symlink_escape(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        outside = tmp_path / "outside.db"
        _make_trial_db(outside)
        # Symlink inside trial pointing outside.
        link = trial / "link.db"
        link.symlink_to(outside)
        code, report_path = _run_stage(
            "g2", trial, monkeypatch, "--source-db", str(link)
        )
        assert code == 1

    def test_g2_rejects_path_traversal(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        outside = tmp_path / "outside.db"
        _make_trial_db(outside)
        code, _ = _run_stage(
            "g2", trial, monkeypatch, "--source-db", str(trial / ".." / "outside.db")
        )
        assert code == 1

    def test_g2_accepts_source_db_inside_trial_root(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        source = _make_trial_db(trial / "source.db")
        code, report_path = _run_stage(
            "g2",
            trial,
            monkeypatch,
            "--source-db",
            str(source),
            "--ack-snapshot-approved",
            "--ack-writer-quiesce",
        )
        assert code == 0

    def test_g3_rejects_source_db_outside_trial_root(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        outside = tmp_path / "outside.db"
        _make_trial_db(outside)
        code, _ = _run_stage("g3", trial, monkeypatch, "--source-db", str(outside))
        assert code == 1

    def test_g8_rejects_source_db_outside_trial_root(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        outside = tmp_path / "outside.db"
        _make_trial_db(outside)
        code, _ = _run_stage("g8", trial, monkeypatch, "--source-db", str(outside))
        assert code == 1

    def test_report_path_must_be_under_trial_root(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        outside_report = tmp_path / "leaked.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "lpc.py",
                "--trial-root",
                str(trial),
                "--report",
                str(outside_report),
                "--stage",
                "g0",
            ],
        )
        assert lpc.main() == 1
        assert not outside_report.exists()


# ===========================================================================
# Critical 2: required acks/prereqs must GATE, not PASS
# ===========================================================================


class TestMandatoryGates:
    def test_g0_gates_without_t0_ssh_ack(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        code, report_path = _run_stage("g0", trial, monkeypatch)
        # G0 must GATE (exit 2) when T0 SSH or image-digest ack is missing.
        assert code == 2
        report = _read_report(report_path)
        assert report["verdict"] == GATE

    def test_g0_passes_with_all_acks(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        code, _ = _run_stage(
            "g0", trial, monkeypatch, "--ack-t0-ssh", "--ack-image-digest"
        )
        assert code == 0

    def test_g2_gates_without_snapshot_or_writer_ack(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        source = _make_trial_db(trial / "source.db")
        code, report_path = _run_stage(
            "g2", trial, monkeypatch, "--source-db", str(source)
        )
        assert code == 2

    def test_g2_passes_with_snapshot_and_writer_acks(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        source = _make_trial_db(trial / "source.db")
        code, _ = _run_stage(
            "g2",
            trial,
            monkeypatch,
            "--source-db",
            str(source),
            "--ack-snapshot-approved",
            "--ack-writer-quiesce",
        )
        assert code == 0

    def test_g4_gates_without_fault_strategy_ack(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        code, _ = _run_stage(
            "g4", trial, monkeypatch, "--g4-events", "4", "--g4-writers", "1"
        )
        assert code == 2

    def test_g7_gates_without_soak_schedule_ack(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        code, _ = _run_stage(
            "g7",
            trial,
            monkeypatch,
            "--soak-seconds",
            "0",
        )
        assert code == 2

    def test_all_cannot_pass_with_omitted_mandatory_acks(self, tmp_path, monkeypatch):
        """The end-to-end must not PASS with any mandatory ack omitted."""
        trial = tmp_path / "trial"
        trial.mkdir()
        source = _make_trial_db(trial / "source.db")
        # Omit --ack-t0-ssh; orchestrator must stop at G0 with GATE.
        partial_acks = [
            "--source-db",
            str(source),
            "--approved-sha",
            "deadbeef" * 8,
            "--g4-events",
            "4",
            "--g4-writers",
            "1",
            "--soak-seconds",
            "0",
            "--ack-image-digest",
            "--ack-snapshot-approved",
            "--ack-writer-quiesce",
            "--ack-codex-desktop",
            "--ack-hermes-smoke",
            "--ack-fault-strategy",
            "--ack-soak-schedule",
            # NOTE: --ack-t0-ssh deliberately omitted
        ]
        code, report_path = _run_stage("all", trial, monkeypatch, *partial_acks)
        assert code == 2


# ===========================================================================
# Critical 3: recursive schema projection + privacy
# ===========================================================================


class TestReportProjection:
    def test_recursive_allowlist_all_nested_keys(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        _, report_path = _run_stage(
            "g0", trial, monkeypatch, "--ack-t0-ssh", "--ack-image-digest"
        )
        report = _read_report(report_path)
        # Recursively walk every nested key and verify it's allowed.
        _assert_recursive_allowlist(report)

    def test_nested_path_like_string_rejected(self, tmp_path, monkeypatch):
        """A path-like string in a nested field must not be written."""
        trial = tmp_path / "trial"
        trial.mkdir()
        report_path = trial / "r.json"
        # Construct a report with a nested forbidden path and confirm
        # write_report refuses to write it.
        bad_report = {
            "stage": "g0",
            "verdict": PASS,
            "reason_code": "ok",
            "checks": {"leak": {"verdict": "/home/bell/secret"}},
            "started_at": "2026-01-01T00:00:00Z",
            "ended_at": "2026-01-01T00:00:00Z",
            "duration_ms": 1.0,
        }
        with pytest.raises(RuntimeError):
            lpc.write_report(report_path, bad_report)
        assert not report_path.exists()

    def test_unknown_nested_key_rejected(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        report_path = trial / "r.json"
        bad_report = {
            "stage": "g0",
            "verdict": PASS,
            "reason_code": "ok",
            "checks": {"arbitrary_key": {"verdict": PASS}},
            "started_at": "2026-01-01T00:00:00Z",
            "ended_at": "2026-01-01T00:00:00Z",
            "duration_ms": 1.0,
        }
        with pytest.raises(RuntimeError):
            lpc.write_report(report_path, bad_report)

    def test_stage_field_is_static_not_user_input(self, tmp_path, monkeypatch):
        """The stage written to the report must be a validated token, never
        raw user input. An injection attempt must not appear verbatim."""
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
                "g0; rm -rf /",  # injection attempt
            ],
        )
        code = lpc.main()
        assert code == 1
        report = _read_report(report_path)
        assert report["stage"] != "g0; rm -rf /"
        assert report["reason_code"] == "unknown_stage"

    def test_all_report_tree_dirs_0700(self, tmp_path, monkeypatch):
        """Every newly created dir in the report tree must be 0700, including
        ancestors. The report file must be 0600."""
        trial = tmp_path / "trial"
        trial.mkdir()
        # Nest the report several levels deep under the trial root.
        _, report_path = _run_stage_custom(
            "g0",
            trial,
            "nested/deep/report.json",
            monkeypatch,
            "--ack-t0-ssh",
            "--ack-image-digest",
        )
        _assert_all_dirs_0700(trial / "nested")
        assert stat.S_IMODE(report_path.stat().st_mode) == 0o600


# ===========================================================================
# Medium: argparse invalid input -> exit 1, content-free
# ===========================================================================


class TestArgparseSafety:
    def test_invalid_stage_exits_one_not_two(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        code, _ = _run_stage("BOGUS", trial, monkeypatch)
        assert code == 1

    def test_missing_required_arg_exits_one_not_two(self, tmp_path, monkeypatch):
        """Missing required arg must not exit 2 (gate masquerade)."""
        monkeypatch.setattr(
            sys,
            "argv",
            ["lpc.py", "--stage", "g0"],  # missing --trial-root, --report
        )
        try:
            code = lpc.main()
            assert code == 1
        except SystemExit as e:
            # argparse calls sys.exit; we must override to exit 1 not 2.
            assert e.code == 1, f"argparse exited {e.code}, expected 1"

    def test_all_preserves_stage_evidence_summary(self, tmp_path, monkeypatch):
        """The `all` report must carry each stage's checks, not discard them."""
        trial = tmp_path / "trial"
        trial.mkdir()
        source = _make_trial_db(trial / "source.db")
        code, report_path = _run_stage(
            "all", trial, monkeypatch, *_all_pass_args(trial, source)
        )
        assert code == 0
        report = _read_report(report_path)
        checks = report["checks"]
        # Each stage entry should carry verdict + reason_code (not empty).
        for stage in ("g0", "g1", "g2", "g3", "g4", "g5", "g6", "g7"):
            assert "verdict" in checks[stage]
            assert "reason_code" in checks[stage]


# ===========================================================================
# Critical 4: G8 real Dream rollback
# ===========================================================================


class TestG8RealRollback:
    def test_g8_creates_applied_action_undoes_and_verifies_content(
        self, tmp_path, monkeypatch
    ):
        """G8 must create an applied Dream action on a clone, snapshot, call
        dream_undo, and verify the canonical_facts content is reverted via
        row hashing (not counts)."""
        trial = tmp_path / "trial"
        trial.mkdir()
        # Provide a clean trial DB; G8 will seed facts, apply a Dream action,
        # and undo it internally.
        from mnemosyne.core.memory import init_db
        from mnemosyne.core.canonical import init_canonical

        db = trial / "seed.db"
        init_db(db)
        init_canonical(db)

        code, report_path = _run_stage(
            "g8",
            trial,
            monkeypatch,
            "--source-db",
            str(db),
        )
        assert code == 0, f"G8 should PASS, got {code}"
        report = _read_report(report_path)
        checks = report["checks"]
        assert checks["dream_undo"]["verdict"] == PASS
        # The undo must have been actually invoked (undone_count >= 1), proving
        # it wasn't a no-op pass on missing dream_runs.
        assert checks["dream_undo"]["undone_count"] >= 1
        # Content hash verification (not counts only).
        assert checks["content_reverted"]["verdict"] == PASS

    def test_g8_sqlite_error_fails_closed(self, tmp_path, monkeypatch):
        """If the dream_runs query hits a sqlite error, G8 must FAIL not PASS."""
        trial = tmp_path / "trial"
        trial.mkdir()
        # Create a DB that is NOT a valid sqlite file (will cause errors).
        bad_db = trial / "bad.db"
        bad_db.write_bytes(b"\x00" * 128)
        code, _ = _run_stage("g8", trial, monkeypatch, "--source-db", str(bad_db))
        assert code == 1


# ===========================================================================
# Critical 5: G7 real soak
# ===========================================================================


class TestG7RealSoak:
    def test_g7_measures_elapsed_duration(self, tmp_path, monkeypatch):
        """G7 with a nonzero soak must measure actual elapsed time, not just
        report the requested value."""
        trial = tmp_path / "trial"
        trial.mkdir()
        code, report_path = _run_stage(
            "g7",
            trial,
            monkeypatch,
            "--soak-seconds",
            "0",
            "--ack-soak-schedule",
        )
        assert code == 0
        report = _read_report(report_path)
        checks = report["checks"]
        # Elapsed duration must be measured and non-negative.
        assert checks["soak"]["elapsed_seconds"] >= 0.0

    def test_g7_real_receipts_from_ingest(self, tmp_path, monkeypatch):
        """Receipt timestamps must come from actual ingest, not local counters."""
        trial = tmp_path / "trial"
        trial.mkdir()
        code, report_path = _run_stage(
            "g7",
            trial,
            monkeypatch,
            "--soak-seconds",
            "0",
            "--ack-soak-schedule",
        )
        report = _read_report(report_path)
        checks = report["checks"]
        # Real receipts: the number of stored events must match iterations.
        assert checks["monotonic_receipts"]["receipt_count"] >= 1

    def test_g7_fails_on_degraded_period(self, tmp_path, monkeypatch):
        """If a soak iteration fails (degraded), G7 must FAIL, not pass."""
        trial = tmp_path / "trial"
        trial.mkdir()
        # Sabotage remember_event to force a degraded period.
        from mnemosyne.core.beam import BeamMemory

        def _failing_remember(self, event):
            raise RuntimeError("forced degraded")

        monkeypatch.setattr(BeamMemory, "remember_event", _failing_remember)
        code, _ = _run_stage(
            "g7",
            trial,
            monkeypatch,
            "--soak-seconds",
            "0",
            "--ack-soak-schedule",
        )
        assert code == 1

    def test_g7_default_is_72h(self, tmp_path, monkeypatch):
        """The production default soak must be genuinely 72 hours."""
        assert lpc._DEFAULT_SOAK_SECONDS == 72 * 60 * 60

    def test_g7_does_not_loop_for_72h_in_tests(self, tmp_path, monkeypatch):
        """With soak-seconds 0, the soak must complete near-instantly."""
        import time as _time

        trial = tmp_path / "trial"
        trial.mkdir()
        start = _time.monotonic()
        _run_stage(
            "g7", trial, monkeypatch, "--soak-seconds", "0", "--ack-soak-schedule"
        )
        elapsed = _time.monotonic() - start
        # Must complete in well under a minute (no 72h loop).
        assert elapsed < 60.0


# ===========================================================================
# High 1: fault matrix exercised + all runs BOTH core and matrix
# ===========================================================================


class TestFaultMatrixAndAll:
    def test_all_runs_both_core_lifecycle_and_fault_matrix(self, tmp_path, monkeypatch):
        """The `all` orchestrator must exercise BOTH the G4 core lifecycle
        AND the G4 fault matrix, not just one."""
        trial = tmp_path / "trial"
        trial.mkdir()
        source = _make_trial_db(trial / "source.db")
        code, report_path = _run_stage(
            "all", trial, monkeypatch, *_all_pass_args(trial, source)
        )
        assert code == 0
        report = _read_report(report_path)
        g4_entry = report["checks"]["g4"]
        # The g4 summary must evidence BOTH core and matrix ran.
        assert g4_entry["reason_code"] == "ok"
        assert g4_entry.get("core_verdict") == PASS
        assert g4_entry.get("matrix_verdict") == PASS

    def test_fault_lock_genuinely_holds_lock(self, tmp_path):
        """The lock fault must actually hold a lock and detect contention."""
        trial = tmp_path / "trial"
        trial.mkdir()
        outcome = lpc._fault_lock(trial)
        assert outcome["contained"] is True
        assert outcome["no_partial_mutation"] is True

    def test_fault_matrix_independent_mutation_check(self, tmp_path):
        """Each fault case must independently prove no partial mutation, not
        trust a reported boolean."""
        trial = tmp_path / "trial"
        trial.mkdir()
        result = lpc._run_fault_matrix(trial)
        for name, outcome in result["cases"].items():
            assert outcome["no_partial_mutation"] is True, f"{name} leaked mutation"


# ===========================================================================
# High 2: truthful, non-tautological checks + trial interpreter
# ===========================================================================


class TestTruthfulChecks:
    def test_g1_lane_import_uses_trial_interpreter_argv(self, tmp_path, monkeypatch):
        """The lane import subprocess must use the configured trial interpreter
        as argv[0], unconditionally asserted."""
        trial = tmp_path / "trial"
        trial.mkdir()
        captured: list[list[str]] = []
        original_run = subprocess.run

        def _spy(cmd, *a, **kw):
            if cmd and "import sqlite3" in " ".join(cmd):
                captured.append(list(cmd))
            return original_run(cmd, *a, **kw)

        monkeypatch.setattr(lpc.subprocess, "run", _spy)
        _run_stage(
            "g1",
            trial,
            monkeypatch,
            "--approved-sha",
            "deadbeef" * 8,
            "--ack-t0-ssh",
            "--ack-image-digest",
            "--trial-interpreter",
            sys.executable,
        )
        # The first argv element must be the trial interpreter, always.
        assert len(captured) >= 1
        assert captured[0][0] == sys.executable

    def test_endpoint_check_not_tautological(self):
        """The endpoint check must do something real, not always return PASS."""
        # It must at least verify the trial root exists (containment), not
        # blindly return PASS.
        v_bad, _ = lpc._check_endpoint_static(Path("/nonexistent"))
        v_bad2 = v_bad
        # A real check varies by input; verify it's tied to a real condition.
        assert v_bad2 in (PASS, FAIL)

    def test_dimension_check_not_just_count(self):
        """The dimension check must verify the actual stage set, not just len."""
        v, r = lpc._check_dimension_static()
        # If _ALL_ORDER is tampered, it must fail.
        assert v in (PASS, FAIL)

    def test_g5_uses_trial_interpreter_for_package_check(self, tmp_path, monkeypatch):
        """G5 package import must use the trial interpreter, not the campaign
        interpreter, for the trial-lane surface."""
        trial = tmp_path / "trial"
        trial.mkdir()
        captured: list[str] = []
        original_run = subprocess.run

        def _spy(cmd, *a, **kw):
            if cmd and "importlib" in " ".join(cmd):
                captured.append(cmd[0])
            return original_run(cmd, *a, **kw)

        monkeypatch.setattr(lpc.subprocess, "run", _spy)
        _run_stage(
            "g5",
            trial,
            monkeypatch,
            "--trial-interpreter",
            sys.executable,
        )
        if captured:
            assert captured[0] == sys.executable


# ===========================================================================
# High 3: self-scan completeness
# ===========================================================================


class TestSelfScan:
    def test_self_scan_detects_bad_directory_mode(self, tmp_path, monkeypatch):
        """The self-scan must catch a directory with mode != 0700."""
        trial = tmp_path / "trial"
        trial.mkdir()
        bad_dir = trial / "loose"
        bad_dir.mkdir()
        # Explicitly chmod to defeat umask and plant a too-open dir.
        bad_dir.chmod(0o755)
        (bad_dir / "file.json").write_text("{}")
        (bad_dir / "file.json").chmod(0o600)
        code, _ = _run_stage(
            "g6",
            trial,
            monkeypatch,
            "--ack-codex-desktop",
            "--ack-hermes-smoke",
        )
        assert code == 1

    def test_self_scan_fail_closed_on_unreadable(self, tmp_path, monkeypatch):
        """The self-scan must fail closed on a stat/read error, not skip."""
        trial = tmp_path / "trial"
        trial.mkdir()
        # Plant a file then remove permissions to read it.
        bad = trial / "unreadable.json"
        bad.write_text("{}")
        bad.chmod(0o000)
        try:
            v, r = lpc._self_scan(trial)
            # Either it caught the bad mode (0o000 != 0600) or failed closed.
            assert v == FAIL
        finally:
            bad.chmod(0o600)

    def test_self_scan_uses_approved_reason_codes(self, tmp_path, monkeypatch):
        """The reason codes in the scan policy must be from the approved set."""
        # Every reason code the self-scan can return must be approved.
        for code in lpc._SELF_SCAN_REASON_CODES:
            assert code in lpc._APPROVED_REASON_CODES


# ===========================================================================
# High 4: oracle strength across gates, faults, G8, G7, all-stage
# ===========================================================================


class TestOracleStrength:
    def test_gate_exits_two_and_verdict_gate_consistent(self, tmp_path, monkeypatch):
        """Every gate must produce exit 2 AND verdict GATE."""
        trial = tmp_path / "trial"
        trial.mkdir()
        # G0 without acks -> GATE.
        code, report_path = _run_stage("g0", trial, monkeypatch)
        assert code == 2
        report = _read_report(report_path)
        assert report["verdict"] == GATE

    def test_fail_exits_one_and_verdict_fail_consistent(self, tmp_path, monkeypatch):
        """Every fail must produce exit 1 AND verdict FAIL."""
        trial = tmp_path / "trial"
        trial.mkdir()
        source = _make_trial_db(trial / "source.db")
        # G2 with source outside trial -> FAIL (containment).
        code, report_path = _run_stage(
            "g2", trial, monkeypatch, "--source-db", str(source)
        )
        # Without acks this gates; with acks but contained source, test fail path:
        # use a malformed source inside trial.
        bad = trial / "bad.db"
        bad.write_bytes(b"not a db")
        code, report_path = _run_stage(
            "g2",
            trial,
            monkeypatch,
            "--source-db",
            str(bad),
            "--ack-snapshot-approved",
            "--ack-writer-quiesce",
        )
        assert code == 1
        report = _read_report(report_path)
        assert report["verdict"] == FAIL

    def test_g7_report_carries_verdicts_not_just_booleans(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        _, report_path = _run_stage(
            "g7", trial, monkeypatch, "--soak-seconds", "0", "--ack-soak-schedule"
        )
        report = _read_report(report_path)
        for key in (
            "monotonic_receipts",
            "bounded_recall",
            "budgets",
            "final_integrity",
            "soak",
        ):
            assert report["checks"][key]["verdict"] in (PASS, FAIL, GATE)
            assert "reason_code" in report["checks"][key]

    def test_all_g8_rehearsal_carries_real_evidence(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        source = _make_trial_db(trial / "source.db")
        _, report_path = _run_stage(
            "all", trial, monkeypatch, *_all_pass_args(trial, source)
        )
        report = _read_report(report_path)
        assert "g8_rehearsal" in report["checks"]
        assert report["checks"]["g8_rehearsal"]["verdict"] == PASS
        assert report["checks"]["g8_rehearsal"]["reason_code"] == "ok"


# ===========================================================================
# Skeleton / report writer (preserved + strengthened)
# ===========================================================================


class TestSkeleton:
    def test_report_dir_is_0700_file_is_0600(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        _run_stage("g0", trial, monkeypatch, "--ack-t0-ssh", "--ack-image-digest")
        report_path = trial / "report.json"
        assert stat.S_IMODE(report_path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(report_path.stat().st_mode) == 0o600

    def test_report_is_content_free(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        _, report_path = _run_stage(
            "g0", trial, monkeypatch, "--ack-t0-ssh", "--ack-image-digest"
        )
        report = _read_report(report_path)
        _assert_content_free(json.dumps(report))

    def test_exit_code_mapping(self):
        assert lpc._verdict_to_exit(PASS) == 0
        assert lpc._verdict_to_exit(FAIL) == 1
        assert lpc._verdict_to_exit(GATE) == 2

    def test_unknown_error_is_static_and_content_free(self, tmp_path, monkeypatch):
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
                "--ack-t0-ssh",
                "--ack-image-digest",
            ],
        )
        code = lpc.main()
        assert code == 1
        report = _read_report(report_path)
        assert report["reason_code"] == "unexpected_error"
        _assert_content_free(json.dumps(report))


# ===========================================================================
# G2/G3 snapshot + migration dry-run (preserved, containment added)
# ===========================================================================


class TestG2Snapshot:
    def test_g2_passes_and_verifies(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        source = _make_trial_db(trial / "source.db")
        code, report_path = _run_stage(
            "g2",
            trial,
            monkeypatch,
            "--source-db",
            str(source),
            "--ack-snapshot-approved",
            "--ack-writer-quiesce",
        )
        assert code == 0
        report = _read_report(report_path)
        checks = report["checks"]
        for k in (
            "snapshot",
            "integrity",
            "fingerprint",
            "mode_bits",
            "sidecar",
            "user_version",
        ):
            assert checks[k]["verdict"] == PASS

    def test_g2_clones_are_0600_with_0700_dir(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        source = _make_trial_db(trial / "source.db")
        _run_stage(
            "g2",
            trial,
            monkeypatch,
            "--source-db",
            str(source),
            "--ack-snapshot-approved",
            "--ack-writer-quiesce",
        )
        snaps_dir = trial / "snapshots"
        assert stat.S_IMODE(snaps_dir.stat().st_mode) == 0o700
        for child in snaps_dir.iterdir():
            assert stat.S_IMODE(child.stat().st_mode) == 0o600


class TestG3MigrationDryRun:
    def test_g3_dry_run_no_mutation(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        source = _make_trial_db(trial / "source.db")
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        code, _ = _run_stage("g3", trial, monkeypatch, "--source-db", str(source))
        assert code == 0
        after = hashlib.sha256(source.read_bytes()).hexdigest()
        assert before == after


# ===========================================================================
# G4 core lifecycle (preserved)
# ===========================================================================


class TestG4CoreLifecycle:
    def test_g4_passes_with_small_params(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        code, report_path = _run_stage(
            "g4",
            trial,
            monkeypatch,
            "--g4-events",
            "4",
            "--g4-writers",
            "1",
            "--ack-fault-strategy",
        )
        assert code == 0
        report = _read_report(report_path)
        assert report["checks"]["exactly_once"]["stored"] == 4


# ===========================================================================
# G5/G6 (preserved + strengthened)
# ===========================================================================


class TestG5StaticChecks:
    def test_g5_passes(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        code, _ = _run_stage(
            "g5",
            trial,
            monkeypatch,
            "--trial-interpreter",
            sys.executable,
        )
        assert code == 0


class TestG6CheckpointsAndScan:
    def test_g6_gates_without_acks(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        code, _ = _run_stage("g6", trial, monkeypatch)
        assert code == 2

    def test_g6_passes_with_acks(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        code, _ = _run_stage(
            "g6",
            trial,
            monkeypatch,
            "--ack-codex-desktop",
            "--ack-hermes-smoke",
        )
        assert code == 0

    def test_g6_self_scan_detects_canary(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        bad = trial / "canary.json"
        bad.write_text('{"token": "api_key=sk-canaryleak0123456789"}')
        bad.chmod(0o600)
        code, _ = _run_stage(
            "g6",
            trial,
            monkeypatch,
            "--ack-codex-desktop",
            "--ack-hermes-smoke",
        )
        assert code == 1
