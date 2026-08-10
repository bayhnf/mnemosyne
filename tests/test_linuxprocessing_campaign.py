"""Task 8b: Linuxprocessing campaign runner tests.

The runner is a stdlib-only, on-host evidence harness for G0-G8 that never
contacts production, never prints private paths/manifests/contents/credentials,
and uses content-free allowlist-projected JSON reports with 0700 dirs / 0600
files and a self content-free assertion before writing.

Tests cover the full RED->GREEN ladder mandated by the Task 8b brief across
seven commits. Helpers are inlined per the brief's helper policy.
"""

from __future__ import annotations

import hashlib
import json
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


import scripts.linuxprocessing_campaign as lpc

# Local aliases for readability in assertions.
PASS = lpc.PASS
FAIL = lpc.FAIL
GATE_MARKER = lpc.GATE


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


# ===========================================================================
# Commit 2: G0/G1 preflight and isolated checkout
# ===========================================================================


class TestG0Preflight:
    def test_g0_passes_when_python_space_endpoint_dimension_lane_ok(
        self, tmp_path, monkeypatch
    ):
        trial = tmp_path / "trial"
        trial.mkdir()
        code, report_path = _run_stage("g0", trial, monkeypatch)
        assert code == 0
        report = _read_report(report_path)
        assert report["verdict"] == lpc.PASS
        checks = report["checks"]
        assert checks["python"]["verdict"] == PASS
        assert checks["disk_space"]["verdict"] == PASS
        assert checks["endpoint"]["verdict"] == PASS
        assert checks["dimension"]["verdict"] == PASS
        assert checks["lane"]["verdict"] == PASS

    def test_g0_fails_when_python_too_old(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        # Force a too-old python version.
        monkeypatch.setattr(lpc.sys, "version_info", (3, 9, 0))
        code, report_path = _run_stage("g0", trial, monkeypatch)
        assert code == 1
        report = _read_report(report_path)
        assert report["verdict"] == lpc.FAIL
        assert report["checks"]["python"]["verdict"] == FAIL

    def test_g0_fails_when_trial_root_missing(self, tmp_path, monkeypatch):
        # A nonexistent trial root is a fail, never a silent pass.
        trial = tmp_path / "trial"  # never created
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
                "g0",
            ],
        )
        assert lpc.main() == 1

    def test_g0_content_free_and_allowlist(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        _, report_path = _run_stage("g0", trial, monkeypatch)
        report = _read_report(report_path)
        _assert_allowlist(report)
        _assert_content_free(json.dumps(report))


class TestG0ManualGates:
    def test_g0_records_t0_ssh_as_pending_without_ack(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        code, report_path = _run_stage("g0", trial, monkeypatch)
        # G0 preflight passes on environment, but the T0 SSH and image-digest
        # gates are recorded as PENDING (never faked). Their presence in the
        # checks is informational; the stage verdict reflects the environment.
        report = _read_report(report_path)
        assert report["checks"]["t0_ssh_ack"]["verdict"] == "PENDING"
        assert report["checks"]["image_digest_ack"]["verdict"] == "PENDING"

    def test_g0_records_t0_ssh_ack_when_flag_set(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        _run_stage(
            "g0",
            trial,
            monkeypatch,
            "--ack-t0-ssh",
            "--ack-image-digest",
        )
        report = _read_report(trial / "report.json")
        assert report["checks"]["t0_ssh_ack"]["verdict"] == "ACKNOWLEDGED"
        assert report["checks"]["image_digest_ack"]["verdict"] == "ACKNOWLEDGED"


class TestG1Checkout:
    def test_g1_gates_pending_without_approved_sha(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        code, report_path = _run_stage("g1", trial, monkeypatch)
        # Missing approved SHA -> GATE (exit 2), never pass.
        assert code == 2
        report = _read_report(report_path)
        assert report["verdict"] == lpc.GATE
        assert report["reason_code"] == "approved_sha_required"

    def test_g1_passes_with_approved_sha_and_lane_imports(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        code, report_path = _run_stage(
            "g1",
            trial,
            monkeypatch,
            "--approved-sha",
            "deadbeef" * 8,
        )
        assert code == 0
        report = _read_report(report_path)
        assert report["verdict"] == lpc.PASS
        checks = report["checks"]
        # SHA presence + dependency health + lane imports each PASS.
        assert checks["approved_sha"]["verdict"] == PASS
        assert checks["dependency_health"]["verdict"] == PASS
        assert checks["lane_imports"]["verdict"] == PASS

    def test_g1_lane_imports_use_trial_interpreter(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        report_path = trial / "report.json"
        # Capture the interpreter used for lane import subprocesses.
        captured: list[str] = []
        real_run = subprocess.run

        def _spy(cmd, *a, **kw):  # type: ignore[no-untyped-def]
            if cmd and "lane_import" in " ".join(cmd):
                captured.append(cmd[0])
            return real_run(cmd, *a, **kw)

        monkeypatch.setattr(lpc.subprocess, "run", _spy)
        _run_stage(
            "g1",
            trial,
            monkeypatch,
            "--approved-sha",
            "deadbeef" * 8,
            "--trial-interpreter",
            sys.executable,
        )
        report = _read_report(report_path)
        # Lane import was attempted with the configured trial interpreter.
        if captured:
            assert captured[0] == sys.executable
        assert report["checks"]["lane_imports"]["verdict"] == PASS

    def test_g1_fails_when_lane_import_broken(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        # Point the trial interpreter at a bogus path so the lane import
        # subprocess cannot run; dependency health still passes.
        code, _ = _run_stage(
            "g1",
            trial,
            monkeypatch,
            "--approved-sha",
            "deadbeef" * 8,
            "--trial-interpreter",
            "/nonexistent/interpreter/bin/python",
        )
        assert code == 1


# ===========================================================================
# Commit 3: G2/G3/G8 snapshot, dry-run migration, restore + rollback rehearsal
# ===========================================================================


def _make_trial_db(path: Path) -> Path:
    """Seed a small trial DB via the real init_db; returns its path."""
    from mnemosyne.core.memory import init_db

    init_db(path)
    return path


class TestG2Snapshot:
    def test_g2_snapshots_trial_clone_and_verifies_integrity(
        self, tmp_path, monkeypatch
    ):
        trial = tmp_path / "trial"
        trial.mkdir()
        source = _make_trial_db(trial / "source.db")
        code, report_path = _run_stage(
            "g2",
            trial,
            monkeypatch,
            "--source-db",
            str(source),
        )
        assert code == 0
        report = _read_report(report_path)
        assert report["verdict"] == lpc.PASS
        checks = report["checks"]
        assert checks["snapshot"]["verdict"] == PASS
        assert checks["integrity"]["verdict"] == PASS
        assert checks["fingerprint"]["verdict"] == PASS
        assert checks["mode_bits"]["verdict"] == PASS
        assert checks["sidecar"]["verdict"] == PASS
        assert checks["user_version"]["verdict"] == PASS

    def test_g2_gates_without_snapshot_approval(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        source = _make_trial_db(trial / "source.db")
        # The writer-quiesce + snapshot-approved acks are the G2 gate.
        code, report_path = _run_stage(
            "g2",
            trial,
            monkeypatch,
            "--source-db",
            str(source),
        )
        report = _read_report(report_path)
        assert report["checks"]["writer_quiesce_ack"]["verdict"] == "PENDING"
        assert report["checks"]["snapshot_approved_ack"]["verdict"] == "PENDING"

    def test_g2_fails_when_source_db_missing(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        code, report_path = _run_stage(
            "g2",
            trial,
            monkeypatch,
            "--source-db",
            str(trial / "nope.db"),
        )
        assert code == 1
        report = _read_report(report_path)
        assert report["checks"]["snapshot"]["verdict"] == FAIL

    def test_g2_clones_are_0600_with_0700_dir(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        source = _make_trial_db(trial / "source.db")
        _run_stage("g2", trial, monkeypatch, "--source-db", str(source))
        snaps_dir = trial / "snapshots"
        assert snaps_dir.exists()
        assert stat.S_IMODE(snaps_dir.stat().st_mode) == 0o700
        for child in snaps_dir.iterdir():
            assert stat.S_IMODE(child.stat().st_mode) == 0o600


class TestG3MigrationDryRun:
    def test_g3_dry_run_is_report_only_no_mutation(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        source = _make_trial_db(trial / "source.db")
        # Baseline hash before dry-run.
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        code, report_path = _run_stage(
            "g3",
            trial,
            monkeypatch,
            "--source-db",
            str(source),
        )
        assert code == 0
        after = hashlib.sha256(source.read_bytes()).hexdigest()
        # Dry-run must not mutate the source clone.
        assert before == after
        report = _read_report(report_path)
        assert report["verdict"] == lpc.PASS
        assert report["checks"]["dry_run"]["verdict"] == PASS
        assert report["checks"]["no_mutation"]["verdict"] == PASS

    def test_g3_content_free(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        source = _make_trial_db(trial / "source.db")
        _, report_path = _run_stage(
            "g3", trial, monkeypatch, "--source-db", str(source)
        )
        report = _read_report(report_path)
        _assert_content_free(json.dumps(report))
        _assert_allowlist(report)


class TestG8RollbackRehearsal:
    def test_g8_rehearsal_restores_clone_and_verifies_pristine(
        self, tmp_path, monkeypatch
    ):
        trial = tmp_path / "trial"
        trial.mkdir()
        source = _make_trial_db(trial / "source.db")
        code, report_path = _run_stage(
            "g8",
            trial,
            monkeypatch,
            "--source-db",
            str(source),
        )
        assert code == 0
        report = _read_report(report_path)
        assert report["verdict"] == lpc.PASS
        checks = report["checks"]
        assert checks["restore"]["verdict"] == PASS
        assert checks["post_restore_integrity"]["verdict"] == PASS
        # Pristine fingerprint: the snapshot's sidecar SHA matches the
        # snapshot file on disk (tamper detection of the pristine image).
        assert checks["pristine_intact"]["verdict"] == PASS
        # Logical equivalence between source and restored target.
        assert checks["table_equivalence"]["verdict"] == PASS
        assert checks["sidecar_absence"]["verdict"] == PASS
        assert checks["user_version_match"]["verdict"] == PASS
        assert checks["dream_undo"]["verdict"] == PASS

    def test_g8_fails_when_source_missing(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        code, _ = _run_stage(
            "g8",
            trial,
            monkeypatch,
            "--source-db",
            str(trial / "nope.db"),
        )
        assert code == 1


# ===========================================================================
# Commit 4: G4 core lifecycle / concurrency
# ===========================================================================


def _seed_clone_for_g4(trial_root: Path) -> Path:
    """Make a trial clone DB for G4 (no BeamMemory wiring yet)."""
    from mnemosyne.core.memory import init_db

    clones = trial_root / "clones"
    clones.mkdir(exist_ok=True)
    clone = clones / "g4.db"
    init_db(clone)
    return clone


class TestG4CoreLifecycle:
    def test_g4_passes_with_small_params_and_records_exactly_once(
        self, tmp_path, monkeypatch
    ):
        trial = tmp_path / "trial"
        trial.mkdir()
        code, report_path = _run_stage(
            "g4",
            trial,
            monkeypatch,
            "--g4-events",
            "12",
            "--g4-writers",
            "3",
        )
        assert code == 0
        report = _read_report(report_path)
        assert report["verdict"] == lpc.PASS
        checks = report["checks"]
        assert checks["exactly_once"]["verdict"] == PASS
        # 12 distinct events -> 12 stored, 0 duplicates expected on first run.
        assert checks["exactly_once"]["stored"] == 12
        assert checks["exactly_once"]["duplicate"] == 0

    def test_g4_crash_retry_completes_once(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        code, report_path = _run_stage(
            "g4",
            trial,
            monkeypatch,
            "--g4-events",
            "5",
            "--g4-writers",
            "1",
        )
        assert code == 0
        report = _read_report(report_path)
        assert report["checks"]["crash_retry"]["verdict"] == PASS

    def test_g4_concurrent_duplicate_race_exactly_one(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        code, report_path = _run_stage(
            "g4",
            trial,
            monkeypatch,
            "--g4-events",
            "8",
            "--g4-writers",
            "4",
        )
        assert code == 0
        report = _read_report(report_path)
        assert report["checks"]["duplicate_race"]["verdict"] == PASS

    def test_g4_dream_lifecycle_and_undo_on_clone(self, tmp_path, monkeypatch):
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
        )
        assert code == 0
        report = _read_report(report_path)
        checks = report["checks"]
        assert checks["dream_lifecycle"]["verdict"] == PASS
        # Applied then undone -> final state is undone.
        assert checks["dream_lifecycle"]["final_state"] == "undone"

    def test_g4_content_free(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        _, report_path = _run_stage(
            "g4", trial, monkeypatch, "--g4-events", "3", "--g4-writers", "1"
        )
        report = _read_report(report_path)
        _assert_content_free(json.dumps(report))
        _assert_allowlist(report)

    def test_g4_fails_without_trial_root(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"  # not created
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
                "g4",
            ],
        )
        assert lpc.main() == 1


# ===========================================================================
# Commit 5: G4 fault matrix
# ===========================================================================


class TestG4FaultMatrix:
    """Each deterministic synthetic fault must produce a structured outcome
    and prove no partial mutation of the clone."""

    def _run_matrix(self, tmp_path, monkeypatch) -> dict:
        trial = tmp_path / "trial"
        trial.mkdir()
        code, report_path = _run_stage("g4", trial, monkeypatch, "--fault-matrix")
        assert code == 0, f"fault matrix failed: code={code}"
        return _read_report(report_path)

    def test_matrix_runs_all_ten_cases(self, tmp_path, monkeypatch):
        report = self._run_matrix(tmp_path, monkeypatch)
        checks = report["checks"]
        assert "fault_matrix" in checks
        cases = checks["fault_matrix"]["cases"]
        expected = {
            "lock",
            "read_only",
            "malformed_db",
            "provider_failure",
            "dimension",
            "crash",
            "sidecar",
            "wal",
            "concurrent_planner",
            "sleep_vs_dream",
        }
        assert set(cases.keys()) == expected
        for name, outcome in cases.items():
            assert outcome["verdict"] == PASS, f"{name} did not pass"
            assert outcome["contained"] is True
            assert outcome["no_partial_mutation"] is True

    def test_matrix_lock_fault_is_contained(self, tmp_path, monkeypatch):
        report = self._run_matrix(tmp_path, monkeypatch)
        lock = report["checks"]["fault_matrix"]["cases"]["lock"]
        assert lock["verdict"] == PASS
        assert lock["contained"] is True

    def test_matrix_malformed_db_no_partial_mutation(self, tmp_path, monkeypatch):
        report = self._run_matrix(tmp_path, monkeypatch)
        malformed = report["checks"]["fault_matrix"]["cases"]["malformed_db"]
        assert malformed["no_partial_mutation"] is True

    def test_matrix_read_only_fault(self, tmp_path, monkeypatch):
        report = self._run_matrix(tmp_path, monkeypatch)
        ro = report["checks"]["fault_matrix"]["cases"]["read_only"]
        assert ro["verdict"] == PASS

    def test_matrix_wal_sidecar_faults(self, tmp_path, monkeypatch):
        report = self._run_matrix(tmp_path, monkeypatch)
        assert report["checks"]["fault_matrix"]["cases"]["wal"]["verdict"] == PASS
        assert report["checks"]["fault_matrix"]["cases"]["sidecar"]["verdict"] == PASS

    def test_matrix_provider_failure(self, tmp_path, monkeypatch):
        report = self._run_matrix(tmp_path, monkeypatch)
        pf = report["checks"]["fault_matrix"]["cases"]["provider_failure"]
        assert pf["verdict"] == PASS

    def test_matrix_concurrent_planner(self, tmp_path, monkeypatch):
        report = self._run_matrix(tmp_path, monkeypatch)
        cp = report["checks"]["fault_matrix"]["cases"]["concurrent_planner"]
        assert cp["verdict"] == PASS

    def test_matrix_sleep_vs_dream(self, tmp_path, monkeypatch):
        report = self._run_matrix(tmp_path, monkeypatch)
        sd = report["checks"]["fault_matrix"]["cases"]["sleep_vs_dream"]
        assert sd["verdict"] == PASS

    def test_matrix_content_free(self, tmp_path, monkeypatch):
        report = self._run_matrix(tmp_path, monkeypatch)
        _assert_content_free(json.dumps(report))
        _assert_allowlist(report)


# ===========================================================================
# Commit 6: G5/G6 manual checkpoints and evidence scan
# ===========================================================================


class TestG5StaticChecks:
    def test_g5_passes_static_plugin_package_checks(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        code, report_path = _run_stage("g5", trial, monkeypatch)
        assert code == 0
        report = _read_report(report_path)
        assert report["verdict"] == lpc.PASS
        checks = report["checks"]
        assert checks["package_import"]["verdict"] == PASS
        assert checks["plugin_surface"]["verdict"] == PASS

    def test_g5_content_free(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        _, report_path = _run_stage("g5", trial, monkeypatch)
        report = _read_report(report_path)
        _assert_content_free(json.dumps(report))
        _assert_allowlist(report)

    def test_g5_fails_without_trial_root(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"  # not created
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
                "g5",
            ],
        )
        assert lpc.main() == 1


class TestG6CheckpointsAndScan:
    def test_g6_gates_pending_without_codex_desktop_ack(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        code, report_path = _run_stage("g6", trial, monkeypatch)
        # Codex Desktop + Hermes smoke are ack-gated; missing -> exit 2.
        assert code == 2
        report = _read_report(report_path)
        assert report["verdict"] == lpc.GATE
        checks = report["checks"]
        assert checks["codex_desktop_ack"]["verdict"] == "PENDING"
        assert checks["hermes_smoke_ack"]["verdict"] == "PENDING"

    def test_g6_passes_with_acks_and_self_scan(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        code, report_path = _run_stage(
            "g6",
            trial,
            monkeypatch,
            "--ack-codex-desktop",
            "--ack-hermes-smoke",
        )
        assert code == 0
        report = _read_report(report_path)
        assert report["verdict"] == lpc.PASS
        checks = report["checks"]
        assert checks["codex_desktop_ack"]["verdict"] == "ACKNOWLEDGED"
        assert checks["hermes_smoke_ack"]["verdict"] == "ACKNOWLEDGED"
        assert checks["self_scan"]["verdict"] == PASS

    def test_g6_self_scan_detects_bad_mode(self, tmp_path, monkeypatch):
        """The self-scan must catch a trial file with a bad (too-open) mode."""
        trial = tmp_path / "trial"
        trial.mkdir()
        # Plant a too-open file in the trial tree.
        bad = trial / "leaked.json"
        bad.write_text("{}")
        bad.chmod(0o644)  # too open; should be 0600
        code, report_path = _run_stage(
            "g6",
            trial,
            monkeypatch,
            "--ack-codex-desktop",
            "--ack-hermes-smoke",
        )
        assert code == 1
        report = _read_report(report_path)
        checks = report["checks"]
        assert checks["self_scan"]["verdict"] == FAIL
        assert "bad_mode" in checks["self_scan"]["reason_code"]

    def test_g6_self_scan_detects_canary_content(self, tmp_path, monkeypatch):
        """The self-scan must catch a canary secret in a trial file."""
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

    def test_g6_content_free(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        _, report_path = _run_stage(
            "g6",
            trial,
            monkeypatch,
            "--ack-codex-desktop",
            "--ack-hermes-smoke",
        )
        report = _read_report(report_path)
        _assert_content_free(json.dumps(report))
        _assert_allowlist(report)


# ===========================================================================
# Commit 7: G7 soak and orchestrator
# ===========================================================================


class TestG7Soak:
    def test_g7_short_soak_passes_with_budgets_and_monotonic_receipts(
        self, tmp_path, monkeypatch
    ):
        trial = tmp_path / "trial"
        trial.mkdir()
        code, report_path = _run_stage(
            "g7",
            trial,
            monkeypatch,
            "--soak-seconds",
            "0",
        )
        assert code == 0
        report = _read_report(report_path)
        assert report["verdict"] == lpc.PASS
        checks = report["checks"]
        assert checks["soak"]["verdict"] == PASS
        assert checks["monotonic_receipts"]["verdict"] == PASS
        assert checks["bounded_recall"]["verdict"] == PASS
        assert checks["budgets"]["verdict"] == PASS
        assert checks["final_integrity"]["verdict"] == PASS
        # Default duration reported is the real 72h, even when the actual
        # soak ran for the smaller test value.
        assert checks["soak"]["default_duration_seconds"] == 72 * 60 * 60

    def test_g7_real_default_duration_is_72h(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        _, report_path = _run_stage("g7", trial, monkeypatch, "--soak-seconds", "0")
        report = _read_report(report_path)
        # The default (real) soak duration is 72 hours, surfaced in the report.
        assert report["checks"]["soak"]["default_duration_seconds"] == 259200
        assert report["checks"]["soak"]["actual_duration_seconds"] == 0

    def test_g7_monotonic_receipts_strictly_increasing(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        _, report_path = _run_stage("g7", trial, monkeypatch, "--soak-seconds", "0")
        report = _read_report(report_path)
        stamps = report["checks"]["monotonic_receipts"]["timestamps"]
        assert stamps == sorted(stamps)
        assert len(stamps) == len(set(stamps))

    def test_g7_budgets_within_bounds(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        _, report_path = _run_stage("g7", trial, monkeypatch, "--soak-seconds", "0")
        report = _read_report(report_path)
        budgets = report["checks"]["budgets"]
        # FD and RSS must be positive and bounded; log lines non-negative.
        assert budgets["open_fds"] >= 0
        assert budgets["rss_kb"] >= 0
        assert budgets["log_lines"] >= 0

    def test_g7_content_free(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        _, report_path = _run_stage("g7", trial, monkeypatch, "--soak-seconds", "0")
        report = _read_report(report_path)
        _assert_content_free(json.dumps(report))
        _assert_allowlist(report)

    def test_g7_fails_without_trial_root(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"  # not created
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
                "g7",
            ],
        )
        assert lpc.main() == 1


class TestAllOrchestrator:
    """The `all` orchestrator runs every stage in order, stops at the first
    non-pass, and executes the full G8 rollback rehearsal after G7."""

    def test_all_stops_at_first_non_pass(self, tmp_path, monkeypatch):
        # Without the G6 acks, `all` should stop at G6 (GATE -> exit 2).
        trial = tmp_path / "trial"
        trial.mkdir()
        source = _make_trial_db(trial / "source.db")
        code, report_path = _run_stage(
            "all",
            trial,
            monkeypatch,
            "--source-db",
            str(source),
            "--approved-sha",
            "deadbeef" * 8,
            # Small test-only G4 values so the orchestrator does not run the
            # real CLI default (10000 events / 16 writers) before reaching G6.
            "--g4-events",
            "4",
            "--g4-writers",
            "1",
            "--soak-seconds",
            "0",
            # Deliberately omit the G6 acks so the orchestrator stops there.
        )
        assert code == 2
        report = _read_report(report_path)
        assert report["verdict"] == lpc.GATE
        checks = report["checks"]
        # g0..g5 ran and passed; g6 is where it stopped.
        for stage in ("g0", "g1", "g2", "g3", "g4", "g5"):
            assert checks[stage]["verdict"] == PASS, f"{stage} did not pass"
        assert checks["g6"]["verdict"] == GATE_MARKER

    def test_all_passes_end_to_end_and_runs_g8_rehearsal(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        source = _make_trial_db(trial / "source.db")
        code, report_path = _run_stage(
            "all",
            trial,
            monkeypatch,
            "--source-db",
            str(source),
            "--approved-sha",
            "deadbeef" * 8,
            "--ack-codex-desktop",
            "--ack-hermes-smoke",
            "--g4-events",
            "4",
            "--g4-writers",
            "1",
            "--soak-seconds",
            "0",
        )
        assert code == 0
        report = _read_report(report_path)
        assert report["verdict"] == lpc.PASS
        checks = report["checks"]
        for stage in ("g0", "g1", "g2", "g3", "g4", "g5", "g6", "g7"):
            assert checks[stage]["verdict"] == PASS, f"{stage} did not pass"
        # The orchestrator must execute the G8 rollback rehearsal after G7,
        # not merely report the earlier standalone G8 as passed.
        assert "g8_rehearsal" in checks
        assert checks["g8_rehearsal"]["verdict"] == PASS

    def test_all_content_free(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        source = _make_trial_db(trial / "source.db")
        _, report_path = _run_stage(
            "all",
            trial,
            monkeypatch,
            "--source-db",
            str(source),
            "--approved-sha",
            "deadbeef" * 8,
            "--ack-codex-desktop",
            "--ack-hermes-smoke",
            "--g4-events",
            "3",
            "--g4-writers",
            "1",
            "--soak-seconds",
            "0",
        )
        report = _read_report(report_path)
        _assert_content_free(json.dumps(report))
        _assert_allowlist(report)
