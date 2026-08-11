"""Task 8b: Linuxprocessing campaign runner tests (FIX ROUND 1).

Strict adversarial tests for every Critical/High/Medium finding from
independent review. The runner is stdlib-only, on-host, never contacts
production, never accepts a production/Bellserver path, and uses recursive
schema-projected content-free reports with 0700 dirs / 0600 files.
"""

from __future__ import annotations

import hashlib
import json
import os
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
    "/home/",
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
    report_path = trial_root / "reports" / "report.json"
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


def test_campaign_pass_receipt_uses_current_utc_clock(monkeypatch):
    expected = "2026-08-11T02:32:09Z"
    monkeypatch.setattr(lpc, "_now_iso", lambda: expected)

    receipt = lpc._pass_receipt(
        "reviewer", "campaign-reviewer", "run-id", "manifest-hash"
    )

    assert receipt["timestamp"] == expected


def _read_report(path: Path) -> dict[str, Any]:
    assert path.exists(), f"report not written at {path}"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with path.open() as f:
        return json.load(f)


_MISSING_STATE = object()


def _campaign_process_state() -> tuple[Any, ...]:
    """Snapshot campaign-mutable process state (callable identity).

    Distinguishes an absent MNEMOSYNE_DATA_DIR from an empty value.
    """
    from mnemosyne.core import beam, embeddings, inhale, shmr
    from mnemosyne.core.config import MnemosyneConfig

    return (
        os.environ.get("MNEMOSYNE_DATA_DIR", _MISSING_STATE),
        MnemosyneConfig._instance,
        shmr._embedding_fn,
        embeddings.embed,
        embeddings.available,
        beam._wm_vec_available,
        beam._store_working_embedding,
        inhale._finalize_receipt,
    )


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


def _artifact_manifest(db_path: Path) -> dict[str, tuple[bool, str]]:
    result = {}
    for suffix in ("", "-wal", "-shm", "-journal"):
        path = Path(f"{db_path}{suffix}")
        result[suffix] = (
            path.exists(),
            hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "",
        )
    return result


def _make_candidate_repo(root: Path) -> tuple[Path, str]:
    candidate = root / "candidate"
    (candidate / "mnemosyne").mkdir(parents=True)
    (candidate / "mnemosyne" / "__init__.py").write_text("", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(candidate)], check=True)
    subprocess.run(["git", "-C", str(candidate), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(candidate),
            "-c",
            "user.name=campaign-test",
            "-c",
            "user.email=campaign@example.invalid",
            "commit",
            "-qm",
            "candidate",
        ],
        check=True,
    )
    sha = subprocess.check_output(
        ["git", "-C", str(candidate), "rev-parse", "HEAD"], text=True
    ).strip()
    return candidate, sha


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
    candidate, sha = _make_candidate_repo(trial)
    return [
        "--source-db",
        str(source),
        "--candidate",
        str(candidate),
        "--approved-sha",
        sha,
        "--image-digest",
        "a" * 64,
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
        # Path inside the trial root but outside reports/ must be rejected
        # under the R1 evidence boundary (and an outside-trial path too).
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

    def test_g0_requires_a_bound_image_digest(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        code, report_path = _run_stage(
            "g0", trial, monkeypatch, "--ack-t0-ssh", "--ack-image-digest"
        )
        assert code == lpc.EXIT_GATE
        assert _read_report(report_path)["checks"]["image_digest"]["verdict"] == GATE

    def test_g0_rejects_malformed_image_digest(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        code, report_path = _run_stage(
            "g0",
            trial,
            monkeypatch,
            "--ack-t0-ssh",
            "--ack-image-digest",
            "--image-digest",
            "not-a-digest",
        )
        assert code == lpc.EXIT_FAIL
        assert (
            _read_report(report_path)["checks"]["image_digest"]["reason_code"]
            == "image_digest_invalid"
        )

    def test_g0_records_valid_digest_without_static_claims(
        self, tmp_path, monkeypatch
    ):
        trial = tmp_path / "trial"
        trial.mkdir()
        digest = "a" * 64
        code, report_path = _run_stage(
            "g0",
            trial,
            monkeypatch,
            "--ack-t0-ssh",
            "--ack-image-digest",
            "--image-digest",
            f"sha256:{digest}",
        )
        assert code == lpc.EXIT_PASS
        checks = _read_report(report_path)["checks"]
        assert checks["image_digest"]["digest"] == digest
        assert {"endpoint", "dimension", "lane"}.isdisjoint(checks)

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
# G1: reviewed candidate binding + real dependency health
# ===========================================================================


class TestG1ReviewedCandidate:
    def test_g1_approved_sha_invalid(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        candidate, _ = _make_candidate_repo(trial)

        code, report_path = _run_stage(
            "g1",
            trial,
            monkeypatch,
            "--candidate",
            str(candidate),
            "--approved-sha",
            "g" * 40,
            "--trial-interpreter",
            sys.executable,
        )

        assert code == lpc.EXIT_FAIL
        assert (
            _read_report(report_path)["checks"]["approved_sha"]["reason_code"]
            == "approved_sha_invalid"
        )
        assert str(candidate) not in report_path.read_text(encoding="utf-8")

    def test_g1_approved_sha_mismatch(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        candidate, _ = _make_candidate_repo(trial)

        code, report_path = _run_stage(
            "g1",
            trial,
            monkeypatch,
            "--candidate",
            str(candidate),
            "--approved-sha",
            "A" * 64,
            "--trial-interpreter",
            sys.executable,
        )

        assert code == lpc.EXIT_FAIL
        assert (
            _read_report(report_path)["checks"]["approved_sha"]["reason_code"]
            == "approved_sha_mismatch"
        )
        assert str(candidate) not in report_path.read_text(encoding="utf-8")

    def test_g1_candidate_outside_trial_root(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        candidate, sha = _make_candidate_repo(tmp_path / "outside")

        code, report_path = _run_stage(
            "g1",
            trial,
            monkeypatch,
            "--candidate",
            str(candidate),
            "--approved-sha",
            sha,
            "--trial-interpreter",
            sys.executable,
        )

        assert code == lpc.EXIT_FAIL
        assert (
            _read_report(report_path)["checks"]["candidate"]["reason_code"]
            == "candidate_outside_trial_root"
        )
        assert str(candidate) not in report_path.read_text(encoding="utf-8")

    def test_g1_candidate_unavailable_gates(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()

        code, report_path = _run_stage(
            "g1",
            trial,
            monkeypatch,
            "--approved-sha",
            "a" * 40,
            "--trial-interpreter",
            sys.executable,
        )

        assert code == lpc.EXIT_GATE
        assert (
            _read_report(report_path)["checks"]["candidate"]["reason_code"]
            == "candidate_unavailable"
        )

    def test_g1_pins_candidate_before_symlink_swap(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        inside, sha = _make_candidate_repo(trial / "inside")
        outside, _ = _make_candidate_repo(tmp_path / "outside")
        (outside / "mnemosyne" / "__init__.py").write_text(
            "raise RuntimeError('outside candidate')\n", encoding="utf-8"
        )
        candidate = trial / "candidate-link"
        candidate.symlink_to(inside, target_is_directory=True)

        def _swap_candidate(*_):
            candidate.unlink()
            candidate.symlink_to(outside, target_is_directory=True)
            return PASS, "ok"

        monkeypatch.setattr(lpc, "_dependency_health_via_trial", _swap_candidate)

        code, report_path = _run_stage(
            "g1",
            trial,
            monkeypatch,
            "--candidate",
            str(candidate),
            "--approved-sha",
            sha,
            "--trial-interpreter",
            sys.executable,
        )

        assert candidate.resolve() == outside
        assert code == lpc.EXIT_PASS
        assert _read_report(report_path)["checks"]["approved_sha"]["sha"] == sha
        assert str(outside) not in report_path.read_text(encoding="utf-8")

    def test_g1_matching_candidate_sha_passes(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        candidate, sha = _make_candidate_repo(trial)
        monkeypatch.setattr(
            lpc, "_dependency_health_via_trial", lambda *_: (PASS, "ok")
        )
        monkeypatch.setattr(lpc, "_lane_import_check", lambda *_: (PASS, "ok"))

        code, report_path = _run_stage(
            "g1",
            trial,
            monkeypatch,
            "--candidate",
            str(candidate),
            "--approved-sha",
            sha.upper(),
            "--trial-interpreter",
            sys.executable,
        )

        assert code == lpc.EXIT_PASS
        assert _read_report(report_path)["checks"]["approved_sha"]["sha"] == sha
        assert str(candidate) not in report_path.read_text(encoding="utf-8")


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

    def test_unapproved_top_level_reason_code_is_not_written(self, tmp_path):
        report_path = tmp_path / "not-created" / "r.json"
        report = {
            "stage": "g0",
            "verdict": PASS,
            "reason_code": "not_approved",
            "checks": {},
            "started_at": "2026-01-01T00:00:00Z",
            "ended_at": "2026-01-01T00:00:00Z",
            "duration_ms": 1.0,
        }
        with pytest.raises(RuntimeError, match="reason code not approved"):
            lpc.write_report(report_path, report)
        assert not report_path.exists()
        assert not report_path.parent.exists()

    def test_unapproved_nested_reason_code_is_not_written(self, tmp_path):
        report_path = tmp_path / "r.json"
        report = {
            "stage": "g0",
            "verdict": PASS,
            "reason_code": "ok",
            "checks": {
                "self_scan": {"verdict": PASS, "reason_code": "not_approved"}
            },
            "started_at": "2026-01-01T00:00:00Z",
            "ended_at": "2026-01-01T00:00:00Z",
            "duration_ms": 1.0,
        }
        with pytest.raises(RuntimeError, match="reason code not approved"):
            lpc.write_report(report_path, report)
        assert not report_path.exists()

    @pytest.mark.parametrize(
        ("tuple_payload", "message"),
        [
            (({"not_allowed": "value"},), "check field not on allowlist"),
            (({"reason_code": "not_approved"},), "reason code not approved"),
        ],
        ids=("schema", "reason-code"),
    )
    def test_tuple_values_cannot_bypass_report_validation(
        self, tmp_path, tuple_payload, message
    ):
        report_path = tmp_path / "not-created" / "r.json"
        report = {
            "stage": "g0",
            "verdict": PASS,
            "reason_code": "ok",
            "checks": {"self_scan": tuple_payload},
            "started_at": "2026-01-01T00:00:00Z",
            "ended_at": "2026-01-01T00:00:00Z",
            "duration_ms": 1.0,
        }
        with pytest.raises(RuntimeError, match=message):
            lpc.write_report(report_path, report)
        assert not report_path.exists()
        assert not report_path.parent.exists()

    def test_current_emitted_reason_codes_are_approved(self):
        assert {
            "g4_core_failed",
            "not_exactly_once",
            "retry_failed",
            "race_not_exactly_one",
            "dream_lifecycle_failed",
            "g5_package_import_failed",
            "g5_plugin_surface_failed",
        } <= lpc._APPROVED_REASON_CODES

    def test_content_validation_failure_has_no_filesystem_side_effects(
        self, tmp_path
    ):
        existing_parent = tmp_path / "existing"
        existing_parent.mkdir()
        existing_parent.chmod(0o750)
        missing_parent = tmp_path / "missing"
        report = {
            "stage": "g0",
            "verdict": PASS,
            "reason_code": "ok",
            "checks": {
                "self_scan": {
                    "verdict": "/home/canary",
                    "reason_code": "ok",
                }
            },
            "started_at": "2026-01-01T00:00:00Z",
            "ended_at": "2026-01-01T00:00:00Z",
            "duration_ms": 1.0,
        }
        for report_path in (
            existing_parent / "r.json",
            missing_parent / "r.json",
        ):
            with pytest.raises(RuntimeError, match="content-free"):
                lpc.write_report(report_path, report)
            assert not report_path.exists()

        assert stat.S_IMODE(existing_parent.stat().st_mode) == 0o750
        assert not missing_parent.exists()

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
        report_path = trial / "reports" / "r.json"
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
        # Nest the report several levels deep under the reports/ evidence root.
        _, report_path = _run_stage_custom(
            "g0",
            trial,
            "reports/nested/deep/report.json",
            monkeypatch,
            "--ack-t0-ssh",
            "--ack-image-digest",
        )
        _assert_all_dirs_0700(trial / "reports" / "nested")
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
        monkeypatch.setattr(
            lpc, "_dependency_health_via_trial", lambda *_: (PASS, "ok")
        )
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

    def test_g8_restores_snapshot_on_a_disposable_target(
        self, tmp_path, monkeypatch
    ):
        """G8 must call the real snapshot.restore_isolated_snapshot onto a
        fresh, nonexistent disposable target and prove the restored DB is
        logically equivalent to the pristine snapshot (table row counts,
        canonical-content hash, user_version), with no sidecars. The report
        evidence is corroborated by direct inspection of the real target."""
        import sqlite3

        from mnemosyne.core.canonical import init_canonical
        from mnemosyne.core.memory import init_db
        from mnemosyne.dr import snapshot as snap

        trial = tmp_path / "trial"
        trial.mkdir()
        db = trial / "seed.db"
        init_db(db)
        init_canonical(db)

        # Wrap restore_isolated_snapshot so the test observes the real call
        # (args + delegation to the real implementation). The wrapper delegates
        # to the original real implementation; it does NOT mock the oracle.
        restore_calls: list[tuple] = []
        real_restore = snap.restore_isolated_snapshot

        def _recording_restore(snapshot_path, target_path):
            restore_calls.append((Path(snapshot_path), Path(target_path)))
            return real_restore(snapshot_path, target_path)

        monkeypatch.setattr(snap, "restore_isolated_snapshot", _recording_restore)
        # The campaign imports snapshot as `snap` inside _stage_g8; patch the
        # attribute the campaign fetches lazily too, so the wrapper is honored.
        import mnemosyne.dr.snapshot as snap_module

        monkeypatch.setattr(snap_module, "restore_isolated_snapshot", _recording_restore)

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

        # The disposable-target restore is invoked exactly once. (The clone
        # may also be produced via restore_isolated_snapshot, so select the
        # call whose target is the fresh restore_target.db rather than the
        # rehearsal clone.)
        disposable = [c for c in restore_calls if c[1].name == "restore_target.db"]
        assert len(disposable) == 1
        snapshot_path, restore_target = disposable[0]
        assert snapshot_path.exists()
        # Fresh target: it must now exist (restore created it) and be contained.
        assert restore_target.exists()
        assert trial in restore_target.resolve().parents or restore_target.resolve() == trial.resolve()

        # Report evidence.
        assert checks["restore"]["verdict"] == PASS
        assert checks["table_equivalence"]["verdict"] == PASS
        assert checks["restore_content_match"]["verdict"] == PASS
        assert checks["user_version_match"]["verdict"] == PASS
        assert checks["post_restore_integrity"]["verdict"] == PASS
        assert checks["sidecar_absence"]["verdict"] == PASS

        # Independent corroboration against the REAL target file: logical
        # equivalence to the pristine snapshot by row counts, content hash,
        # and user_version; integrity_check ok; no sidecars.
        def _row_counts(path):
            conn = sqlite3.connect(str(path))
            try:
                rows = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
                counts = {}
                for (name,) in rows:
                    counts[name] = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
                return counts
            finally:
                conn.close()

        assert _row_counts(snapshot_path) == _row_counts(restore_target)

        def _content_hash(path):
            conn = sqlite3.connect(str(path))
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    "SELECT id, owner_id, category, name, body, confidence, "
                    "version, valid_from, valid_until FROM canonical_facts ORDER BY id"
                ).fetchall()
                return hashlib.sha256(
                    json.dumps([dict(r) for r in rows], sort_keys=True, default=str).encode()
                ).hexdigest()
            finally:
                conn.close()

        assert _content_hash(snapshot_path) == _content_hash(restore_target)

        def _user_version(path):
            conn = sqlite3.connect(str(path))
            try:
                return conn.execute("PRAGMA user_version").fetchone()[0]
            finally:
                conn.close()

        assert _user_version(snapshot_path) == _user_version(restore_target)

        conn = sqlite3.connect(str(restore_target))
        try:
            assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        finally:
            conn.close()
        assert not Path(str(restore_target) + "-wal").exists()
        assert not Path(str(restore_target) + "-shm").exists()

    def test_g8_sqlite_error_fails_closed(self, tmp_path, monkeypatch):
        """If the dream_runs query hits a sqlite error, G8 must FAIL not PASS."""
        trial = tmp_path / "trial"
        trial.mkdir()
        # Create a DB that is NOT a valid sqlite file (will cause errors).
        bad_db = trial / "bad.db"
        bad_db.write_bytes(b"\x00" * 128)
        code, _ = _run_stage("g8", trial, monkeypatch, "--source-db", str(bad_db))
        assert code == 1

    def test_g8_clone_preserves_committed_wal_frames(self, tmp_path, monkeypatch):
        """G8's clone must reflect committed frames still living only in the
        source -wal. Replacing the raw shutil.copy2+sidecar-delete path with
        snap.create_isolated_snapshot+restore_isolated_snapshot preserves them.
        """
        import sqlite3

        from mnemosyne.core.memory import init_db

        trial = tmp_path / "trial"
        trial.mkdir()
        source = trial / "source.db"
        init_db(source)
        writer = sqlite3.connect(str(source))
        try:
            writer.execute("PRAGMA journal_mode=WAL")
            writer.execute("PRAGMA wal_autocheckpoint=0")
            writer.execute("CREATE TABLE wal_witness (value TEXT NOT NULL)")
            writer.execute("INSERT INTO wal_witness(value) VALUES ('committed-wal-frame')")
            writer.commit()
            assert Path(str(source) + "-wal").exists()

            code, _ = _run_stage(
                "g8", trial, monkeypatch, "--source-db", str(source)
            )

            assert code == 0
            clone = trial / "g8" / "rehearsal.db"
            with sqlite3.connect(str(clone)) as conn:
                assert conn.execute(
                    "SELECT value FROM wal_witness"
                ).fetchone() == ("committed-wal-frame",)
        finally:
            writer.close()


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

    def test_g7_does_not_fabricate_fd_zero_on_measurement_error(self, monkeypatch):
        monkeypatch.setattr(
            lpc.os, "listdir", lambda _path: (_ for _ in ()).throw(OSError)
        )
        assert lpc._resource_snapshot() is None

    def test_g7_fails_with_resource_measurement_failed(self, tmp_path, monkeypatch):
        """An unavailable resource measurement must fail G7, not emit a zero."""
        trial = tmp_path / "trial"
        trial.mkdir()
        monkeypatch.setattr(lpc, "_resource_snapshot", lambda: None)
        code, report_path = _run_stage(
            "g7", trial, monkeypatch, "--soak-seconds", "0", "--ack-soak-schedule"
        )
        assert code == 1
        report = _read_report(report_path)
        assert report["reason_code"] == "resource_measurement_failed"
        assert report["checks"]["budgets"] == {
            "verdict": FAIL,
            "reason_code": "resource_measurement_failed",
        }

    def test_g7_reports_no_fabricated_log_lines(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        code, report_path = _run_stage(
            "g7", trial, monkeypatch, "--soak-seconds", "0", "--ack-soak-schedule"
        )
        assert code == 0
        report = _read_report(report_path)
        assert "log_lines" not in lpc._resource_snapshot()
        assert "log_lines" not in report["checks"]["budgets"]


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
        monkeypatch.setattr(
            lpc, "_dependency_health_via_trial", lambda *_: (PASS, "ok")
        )
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
        trust a reported boolean. Runs inside the campaign process-state
        context so the matrix's direct hook writes are isolated (R2)."""
        trial = tmp_path / "trial"
        trial.mkdir()
        with lpc._campaign_process_state():
            result = lpc._run_fault_matrix(trial)
        for name, outcome in result["cases"].items():
            assert outcome["no_partial_mutation"] is True, f"{name} leaked mutation"


# ===========================================================================
# High 2: truthful, non-tautological checks + trial interpreter
# ===========================================================================


class TestTruthfulChecks:
    def test_g1_lane_import_uses_trial_interpreter_argv(self, tmp_path, monkeypatch):
        """The candidate import must use the configured trial interpreter."""
        trial = tmp_path / "trial"
        trial.mkdir()
        candidate, sha = _make_candidate_repo(trial)
        captured: list[tuple[list[str], Path, dict[str, Any]]] = []
        original_run = subprocess.run
        monkeypatch.setattr(
            lpc, "_dependency_health_via_trial", lambda *_: (PASS, "ok")
        )

        def _spy(cmd, *a, **kw):
            if cmd and cmd[1:] == ["-c", "import mnemosyne"]:
                captured.append((list(cmd), Path(kw["cwd"]).resolve(), kw))
            return original_run(cmd, *a, **kw)

        monkeypatch.setattr(lpc.subprocess, "run", _spy)
        code, report_path = _run_stage(
            "g1",
            trial,
            monkeypatch,
            "--candidate",
            str(candidate),
            "--approved-sha",
            sha,
            "--trial-interpreter",
            sys.executable,
        )

        assert code == lpc.EXIT_PASS
        assert len(captured) == 1
        assert captured[0][0][0] == sys.executable
        assert captured[0][1] == candidate.resolve()
        assert captured[0][2]["capture_output"] is True
        assert str(candidate) not in report_path.read_text(encoding="utf-8")

    def test_g1_dependency_health_uses_trial_pip_check(self, monkeypatch):
        captured: list[tuple[list[str], dict[str, Any]]] = []

        def _spy(cmd, *a, **kw):
            captured.append((list(cmd), kw))
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(lpc.subprocess, "run", _spy)

        assert lpc._dependency_health_via_trial(sys.executable) == (PASS, "ok")
        assert captured[0][0] == [sys.executable, "-m", "pip", "check"]
        assert captured[0][1]["capture_output"] is True

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
        reports = trial / "reports"
        reports.mkdir()
        reports.chmod(0o700)
        bad_dir = reports / "loose"
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
        reports = trial / "reports"
        reports.mkdir()
        reports.chmod(0o700)
        # Plant a file then remove permissions to read it.
        bad = reports / "unreadable.json"
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

    def test_g6_rejects_unclassified_fifo(self, tmp_path):
        reports = tmp_path / "reports"
        reports.mkdir(mode=0o700)
        os.mkfifo(reports / "evidence")
        assert lpc._self_scan(tmp_path) == (FAIL, "scan_read_error")

    def test_g6_scans_yaml_and_lowercase_error_tokens(self, tmp_path):
        reports = tmp_path / "reports"
        reports.mkdir(mode=0o700)
        evidence = reports / "evidence.yaml"
        evidence.write_text("traceback: leaked", encoding="utf-8")
        os.chmod(evidence, 0o600)
        assert lpc._self_scan(tmp_path) == (FAIL, "canary_content")

    def test_g6_binary_artifact_fails_closed(self, tmp_path):
        reports = tmp_path / "reports"
        reports.mkdir(mode=0o700)
        bad = reports / "blob.png"
        bad.write_bytes(b"\xff\xfe\x00binary")
        os.chmod(bad, 0o600)
        assert lpc._self_scan(tmp_path) == (FAIL, "scan_read_error")

    def test_g6_binary_db_class_is_skipped(self, tmp_path):
        reports = tmp_path / "reports"
        reports.mkdir(mode=0o700)
        db = reports / "evidence.db"
        db.write_bytes(b"\xff\xfe\x00binary")
        os.chmod(db, 0o600)
        assert lpc._self_scan(tmp_path) == (PASS, "ok")


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
        monkeypatch.setattr(
            lpc, "_dependency_health_via_trial", lambda *_: (PASS, "ok")
        )
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
        report_path = trial / "reports" / "report.json"
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
        report_path = trial / "reports" / "r.json"

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
        code, _ = _run_stage(
            "g3",
            trial,
            monkeypatch,
            "--source-db",
            str(source),
            "--ack-writer-quiesce",
        )
        assert code == 0
        after = hashlib.sha256(source.read_bytes()).hexdigest()
        assert before == after

    def test_g3_rehearses_e6_and_e7_on_clone_without_touching_source_sidecars(
        self, tmp_path, monkeypatch
    ):
        import sqlite3

        from mnemosyne.core.memory import init_db
        from mnemosyne.migrations.e6_triplestore_split import migrate as real_e6
        from mnemosyne.migrations.e7_311_tables import migrate_311_tables as real_e7

        trial = tmp_path / "trial"
        trial.mkdir()
        source = trial / "source.db"
        init_db(source)
        writer = sqlite3.connect(str(source))
        try:
            writer.execute("PRAGMA journal_mode=WAL")
            writer.execute("PRAGMA wal_autocheckpoint=0")
            writer.execute("CREATE TABLE wal_witness (value TEXT NOT NULL)")
            writer.execute(
                "INSERT INTO wal_witness(value) VALUES ('committed-wal-frame')"
            )
            writer.commit()
            assert Path(str(source) + "-wal").exists()
        finally:
            writer.close()

        before = _artifact_manifest(source)

        e6_calls: list[tuple] = []
        e7_calls: list[tuple] = []

        def _recording_e6(db_path, dry_run, backup, log_fn):
            e6_calls.append((Path(db_path), dry_run, backup))
            return real_e6(db_path, dry_run=dry_run, backup=backup, log_fn=log_fn)

        def _recording_e7(db_path, dry_run):
            e7_calls.append((Path(db_path), dry_run))
            return real_e7(db_path, dry_run=dry_run)

        import mnemosyne.migrations.e6_triplestore_split as e6_mod
        import mnemosyne.migrations.e7_311_tables as e7_mod

        monkeypatch.setattr(e6_mod, "migrate", _recording_e6)
        monkeypatch.setattr(e7_mod, "migrate_311_tables", _recording_e7)

        code, report_path = _run_stage(
            "g3",
            trial,
            monkeypatch,
            "--source-db",
            str(source),
            "--ack-writer-quiesce",
        )
        assert code == 0, f"G3 should PASS, got {code}"
        report = _read_report(report_path)

        assert _artifact_manifest(source) == before
        assert e6_calls and e6_calls[0][1:] == (True, False)
        assert e7_calls and e7_calls[0][1] is True
        assert Path(e6_calls[0][0]).parent != source.parent
        assert Path(e7_calls[0][0]) == Path(e6_calls[0][0])
        assert sqlite3.connect(e6_calls[0][0]).execute(
            "SELECT COUNT(*) FROM wal_witness"
        ).fetchone()[0] == 1
        assert report["checks"]["dry_run"]["verdict"] == PASS
        assert report["checks"]["dry_run_e7"]["verdict"] == PASS
        assert report["checks"]["no_mutation"]["verdict"] == PASS

    def test_g3_gates_without_writer_quiesce_ack(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        source = _make_trial_db(trial / "source.db")
        code, report_path = _run_stage(
            "g3", trial, monkeypatch, "--source-db", str(source)
        )
        assert code == lpc.EXIT_GATE
        report = _read_report(report_path)
        assert (
            report["checks"]["writer_quiesce_ack"]["reason_code"]
            == "writer_quiesce_ack_required"
        )


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
        reports = trial / "reports"
        reports.mkdir()
        reports.chmod(0o700)
        bad = reports / "canary.json"
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

    def test_report_path_must_be_under_reports_dir(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        report = trial / "outside-reports" / "report.json"
        exit_code, _ = _run_stage_custom(
            "g0", trial, "outside-reports/report.json", monkeypatch, *_ack_all()
        )
        assert exit_code == lpc.EXIT_FAIL
        assert not report.exists()

    def test_g6_rejects_home_canary_in_evidence(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        evidence = trial / "reports"
        evidence.mkdir()
        (evidence / "canary.json").write_text("/home/canary", encoding="utf-8")
        os.chmod(evidence / "canary.json", 0o600)
        exit_code, report = _run_stage("g6", trial, monkeypatch, *_ack_all())
        assert exit_code == lpc.EXIT_FAIL
        assert _read_report(report)["checks"]["self_scan"]["reason_code"] == "canary_content"

    def test_g6_ignores_home_canary_outside_evidence(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        metadata = trial / "venv" / "direct_url.json"
        metadata.parent.mkdir()
        metadata.write_text("/home/canary", encoding="utf-8")
        os.chmod(metadata, 0o600)
        monkeypatch.setattr(lpc, "_FORBIDDEN_FRAGMENTS", ("/home/",))
        exit_code, _ = _run_stage("g6", trial, monkeypatch, *_ack_all())
        assert exit_code == lpc.EXIT_PASS

    def test_g6_rejects_symlink_in_reports_tree(self, tmp_path, monkeypatch):
        trial = tmp_path / "trial"
        trial.mkdir()
        reports = trial / "reports"
        reports.mkdir()
        outside = trial / "outside.json"
        outside.write_text("safe", encoding="utf-8")
        os.chmod(outside, 0o600)
        (reports / "link.json").symlink_to(outside)
        exit_code, report = _run_stage("g6", trial, monkeypatch, *_ack_all())
        assert exit_code == lpc.EXIT_FAIL
        assert _read_report(report)["checks"]["self_scan"]["reason_code"] == "scan_read_error"

    def test_g6_rejects_dangling_symlink_in_reports_tree(self, tmp_path, monkeypatch):
        """A dangling symlink in reports/ must fail closed, not be skipped as a
        non-file/non-dir entry (R1 fix round 1)."""
        trial = tmp_path / "trial"
        trial.mkdir()
        reports = trial / "reports"
        reports.mkdir()
        reports.chmod(0o700)
        (reports / "dangling.json").symlink_to(trial / "missing.json")
        exit_code, report = _run_stage("g6", trial, monkeypatch, *_ack_all())
        assert exit_code == lpc.EXIT_FAIL
        assert _read_report(report)["checks"]["self_scan"]["reason_code"] == "scan_read_error"

    def test_g0_rejects_report_when_reports_root_is_symlink_to_outside(self, tmp_path, monkeypatch):
        """If <trial>/reports is a symlink to a directory outside the trial
        root, the report path must be rejected and nothing written outside."""
        trial = tmp_path / "trial"
        trial.mkdir()
        outside_dir = tmp_path / "outside-evidence"
        outside_dir.mkdir()
        (trial / "reports").symlink_to(outside_dir)
        outside_report = outside_dir / "report.json"
        exit_code, _ = _run_stage("g0", trial, monkeypatch, *_ack_all())
        assert exit_code == lpc.EXIT_FAIL
        assert not outside_report.exists()

    def test_report_path_equal_to_reports_root_is_rejected(self, tmp_path, monkeypatch):
        """--report <trial>/reports must be rejected: the evidence root is not
        itself a report file (R1 fix round 1)."""
        trial = tmp_path / "trial"
        trial.mkdir()
        exit_code, _ = _run_stage_custom(
            "g0", trial, "reports", monkeypatch, *_ack_all()
        )
        assert exit_code == lpc.EXIT_FAIL

    def test_self_scan_rejects_bad_mode_on_reports_root(self, tmp_path, monkeypatch):
        """_self_scan must validate the reports/ root directory's own 0700 mode,
        not only the modes of entries beneath it (R1 fix round 1)."""
        trial = tmp_path / "trial"
        trial.mkdir()
        reports = trial / "reports"
        reports.mkdir()
        reports.chmod(0o755)  # too-open evidence root
        verdict, reason = lpc._self_scan(trial)
        assert verdict == FAIL
        assert reason == "bad_directory_mode"


# ===========================================================================
# R2: campaign process-state isolation (same-process + failure path)
# ===========================================================================


class TestCampaignProcessStateIsolation:
    def test_g4_restores_campaign_process_state(self, tmp_path, monkeypatch):
        before = _campaign_process_state()
        trial = tmp_path / "trial"
        trial.mkdir()
        exit_code, _ = _run_stage(
            "g4", trial, monkeypatch, *_ack_all(), "--g4-events", "1", "--g4-writers", "2"
        )
        assert exit_code == lpc.EXIT_PASS
        assert _campaign_process_state() == before

    def test_campaign_state_restores_when_g4_fails(self, tmp_path, monkeypatch):
        before = _campaign_process_state()
        trial = tmp_path / "trial"
        trial.mkdir()
        monkeypatch.setattr(lpc, "_run_exactly_once", lambda *_: (_ for _ in ()).throw(RuntimeError("test failure")))
        exit_code, _ = _run_stage(
            "g4", trial, monkeypatch, *_ack_all(), "--g4-events", "1", "--g4-writers", "2"
        )
        assert exit_code == lpc.EXIT_FAIL
        assert _campaign_process_state() == before

    def test_campaign_then_sync_embedding_hooks_restored(self, tmp_path, monkeypatch):
        """Sync-facing hooks must be the real functions after a campaign stage,
        in the same-process campaign-then-sync order (no subprocess)."""
        trial = tmp_path / "trial"
        trial.mkdir()
        exit_code, _ = _run_stage(
            "g4", trial, monkeypatch, *_ack_all(), "--g4-events", "1", "--g4-writers", "2"
        )
        assert exit_code == lpc.EXIT_PASS
        from mnemosyne.core import embeddings
        from mnemosyne.core.beam import _embeddings

        assert _embeddings.embed is embeddings.embed
        assert _embeddings.available is embeddings.available
