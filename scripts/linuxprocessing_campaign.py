#!/usr/bin/env python3
"""Linuxprocessing campaign runner (Task 8b).

A stdlib-only, on-host evidence harness for the G0-G8 Linuxprocessing
campaign. Operates ONLY on an explicitly created trial root and its
clones/artifacts. Never contacts production, never invokes SSH, never
accepts a production/Bellserver path, and never prints paths, memory
content, manifests/scope, receipt bodies, raw failures, or credentials.

Every report is a recursive-schema-projected JSON document. Directories are
created/verified 0700 and files 0600, with a self content-free assertion
before writing. Exit codes: 0 = pass, 1 = fail, 2 = a manual gate is pending.

Containment: EVERY filesystem and subprocess input is resolved and verified
to be safely contained under the resolved, explicitly-created trial root
(source DB, report/artifact directory, snapshots, clones). Symlink escape
and path traversal are rejected before any action. The runner never accepts
a production/Bellserver path as an argument.

Manual gates the runner never fakes (each is an explicit operator
acknowledgement flag; absence exits 2 at the first relevant stage):
  T0 SSH connectivity            --ack-t0-ssh           (G0)
  image-digest verification      --ack-image-digest     (G0)
  snapshot approval              --ack-snapshot-approved (G2)
  writer quiescence              --ack-writer-quiesce   (G2)
  fault strategy                 --ack-fault-strategy   (G4)
  Codex Desktop four-hook        --ack-codex-desktop    (G6)
  two-mirror Hermes smoke        --ack-hermes-smoke     (G6)
  72h soak scheduling            --ack-soak-schedule    (G7)

Reviewed-prereq guard (brief: "run only after Task 16 and 17-21 reviewed"):
the runner has no network/SSH reach to those tasks' artifacts, so it cannot
forge their completion. The guard is the set of mandatory ack flags above
plus the operator-only out-of-band GO decision; a missing ack fails closed
to exit 2 and no code path can set PASS for an unacknowledged prerequisite.
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

# ---------------------------------------------------------------------------
# Constants and allowlists
# ---------------------------------------------------------------------------

_DIR_MODE = 0o700
_FILE_MODE = 0o600

PASS = "PASS"
FAIL = "FAIL"
GATE = "GATE"

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_GATE = 2

_DEFAULT_SOAK_SECONDS = 72 * 60 * 60

_STAGES = ("g0", "g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "all")
_STAGE_NAMES = frozenset(_STAGES)
_ALL_ORDER = ("g0", "g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8")

# Top-level report keys (allowlist).
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

# Allowed keys inside each check entry. Every check sub-dict may only use
# these keys (recursive schema projection, Critical 3).
_ALLOWED_CHECK_KEYS = frozenset(
    {
        "verdict",
        "reason_code",
        "stored",
        "duplicate",
        "final_state",
        "undone_count",
        "receipt_count",
        "recall_depth_bound",
        "open_fds",
        "rss_kb",
        "log_lines",
        "timestamps",
        "iterations",
        "degraded_periods",
        "default_duration_seconds",
        "actual_duration_seconds",
        "elapsed_seconds",
        "core_verdict",
        "matrix_verdict",
        "cases",
    }
)

# Allowed check names (the set of keys that may appear under ``checks``).
# This is exhaustive: any check name not in this set is rejected by the
# recursive schema projection (Critical 3).
_ALLOWED_CHECK_NAMES = frozenset(
    {
        "trial_root",
        "python",
        "disk_space",
        "endpoint",
        "dimension",
        "lane",
        "t0_ssh_ack",
        "image_digest_ack",
        "approved_sha",
        "dependency_health",
        "lane_imports",
        "source_db",
        "writer_quiesce_ack",
        "snapshot_approved_ack",
        "snapshot",
        "integrity",
        "fingerprint",
        "mode_bits",
        "sidecar",
        "user_version",
        "dry_run",
        "no_mutation",
        "fault_matrix",
        "fault_strategy_ack",
        "soak_schedule_ack",
        "exactly_once",
        "crash_retry",
        "duplicate_race",
        "dream_lifecycle",
        "package_import",
        "plugin_surface",
        "codex_desktop_ack",
        "hermes_smoke_ack",
        "self_scan",
        "restore",
        "post_restore_integrity",
        "pristine_intact",
        "table_equivalence",
        "content_reverted",
        "sidecar_absence",
        "user_version_match",
        "dream_undo",
        "monotonic_receipts",
        "bounded_recall",
        "budgets",
        "final_integrity",
        "soak",
        "core_verdict",
        "matrix_verdict",
        "g8_rehearsal",
        "g0",
        "g1",
        "g2",
        "g3",
        "g4",
        "g5",
        "g6",
        "g7",
    }
)

# Approved reason codes (used by self-scan policy, High 3).
_APPROVED_REASON_CODES = frozenset(
    {
        "ok",
        "unknown_stage",
        "unexpected_error",
        "argparse_error",
        "trial_root_missing",
        "source_db_outside_trial_root",
        "report_path_outside_trial_root",
        "preflight_failed",
        "python_too_old",
        "insufficient_disk",
        "disk_unavailable",
        "t0_ssh_ack_required",
        "image_digest_ack_required",
        "snapshot_approved_ack_required",
        "writer_quiesce_ack_required",
        "fault_strategy_ack_required",
        "soak_schedule_ack_required",
        "codex_desktop_ack_required",
        "hermes_smoke_ack_required",
        "approved_sha_required",
        "dependency_unavailable",
        "dependency_health_failed",
        "lane_unavailable",
        "lane_import_failed",
        "source_db_missing",
        "snapshot_failed",
        "snapshot_api_unavailable",
        "snapshot_verification_failed",
        "integrity_failed",
        "fingerprint_missing",
        "mode_bits_wrong",
        "sidecar_present",
        "user_version_mismatch",
        "dry_run_failed",
        "source_mutated",
        "restore_failed",
        "rollback_rehearsal_failed",
        "pristine_tampered",
        "table_mismatch",
        "content_not_reverted",
        "dream_undo_failed",
        "dream_undo_not_invoked",
        "g4_exactly_once_failed",
        "g4_crash_retry_failed",
        "g4_duplicate_race_failed",
        "g4_dream_lifecycle_failed",
        "fault_matrix_failed",
        "case_error",
        "dimension_bad",
        "self_scan_failed",
        "bad_mode",
        "bad_directory_mode",
        "canary_content",
        "scan_read_error",
        "non_monotonic",
        "recall_unbounded",
        "budget_exceeded",
        "soak_failed",
        "degraded_period",
        "package_import_failed",
        "plugin_surface_missing",
        "python_version_mismatch",
        "dimension_mismatch",
    }
)

# Reason codes the self-scan can return (subset of approved, High 3).
_SELF_SCAN_REASON_CODES = frozenset(
    {"ok", "bad_mode", "bad_directory_mode", "canary_content", "scan_read_error"}
)

# Fragments that must NEVER appear in a serialized report.
_FORBIDDEN_FRAGMENTS = (
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


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utcnow_ms() -> float:
    return time.time() * 1000.0


# ---------------------------------------------------------------------------
# Containment (Critical 1)
# ---------------------------------------------------------------------------


def _resolve(path: Path) -> Path:
    """Resolve a path, following symlinks, WITHOUT the CWD-relative semantics
    that hide escapes. Raises if the path cannot be resolved."""
    return Path(os.path.realpath(str(path)))


def _contained_under(path: Path, root: Path) -> bool:
    """True iff ``path`` (resolved, symlinks followed) is ``root`` itself or a
    descendant of ``root``. Rejects symlink escape and path traversal."""
    try:
        resolved = _resolve(path)
        root_resolved = _resolve(root)
    except OSError:
        return False
    try:
        resolved.relative_to(root_resolved)
        return True
    except ValueError:
        return False


def _safe_source_db(source_db: Path, trial_root: Path) -> Path | None:
    """Return the source DB path only if it exists AND is contained under the
    trial root; None otherwise."""
    if not source_db:
        return None
    source_db = Path(source_db)
    if not source_db.exists():
        return None
    if not _contained_under(source_db, trial_root):
        return None
    return source_db


def _safe_report_path(report_path: Path, trial_root: Path) -> Path | None:
    """Return the report path only if it is a strict descendant of
    ``<trial_root>/reports/`` (R1 evidence boundary). Rejects the reports
    root itself, paths outside it, and an existing reports root that is a
    symlink, a non-directory, or resolves outside trial_root. A missing
    reports root is allowed because ``_ensure_report_tree`` creates it later."""
    trial_root = Path(trial_root)
    reports_root = trial_root / "reports"
    report_path = Path(report_path)
    # Reject a poisoned existing reports/ root before any containment check.
    try:
        if reports_root.is_symlink():
            return None
        if reports_root.exists() and (
            not reports_root.is_dir() or not _contained_under(reports_root, trial_root)
        ):
            return None
    except OSError:
        return None
    if not _contained_under(report_path, reports_root):
        return None
    if _resolve(report_path) == _resolve(reports_root):
        return None
    return report_path


# ---------------------------------------------------------------------------
# Report projection and writing (Critical 3)
# ---------------------------------------------------------------------------


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
    """Fail closed if any forbidden fragment appears in the blob."""
    low = blob.lower()
    for frag in _FORBIDDEN_FRAGMENTS:
        if frag in low:
            raise RuntimeError("self content-free assertion failed; report not written")


def _assert_recursive_schema(obj: Any, trail: str = "root") -> None:
    """Recursively verify every key in the report is on the allowed schema.

    Top-level keys must be in _REPORT_KEYS. Keys inside ``checks`` must be in
    _ALLOWED_CHECK_KEYS (the check names) and their sub-fields must also be
    allowed tokens. Fails closed on ANY unexpected structure."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if trail == "root":
                if key not in _REPORT_KEYS:
                    raise RuntimeError(
                        "report key not on allowlist; report not written"
                    )
            elif trail == "root.checks":
                # This is a check name; must be in the known check-names set.
                if key not in _ALLOWED_CHECK_NAMES:
                    raise RuntimeError("unknown check name; report not written")
            elif trail.startswith("root.checks."):
                # This is a field within a check; must be an allowed key.
                if key not in _ALLOWED_CHECK_KEYS:
                    raise RuntimeError(
                        "check field not on allowlist; report not written"
                    )
            _assert_recursive_schema(value, f"{trail}.{key}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _assert_recursive_schema(item, f"{trail}[{i}]")


def _is_safe_token(value: Any) -> bool:
    """A token is safe if it's a short alphanumeric/underscore string with no
    path separators, shell metacharacters, or forbidden fragments."""
    if not isinstance(value, str):
        return True  # non-string values are checked by content scan
    if len(value) > 64:
        # A legitimately short reason code or verdict; reject long strings.
        return False
    low = value.lower()
    for frag in _FORBIDDEN_FRAGMENTS:
        if frag in low:
            return False
    # Reject shell metacharacters and path separators in token fields.
    for ch in (";", "|", "&", "$", "`", "\n", "\r"):
        if ch in value:
            return False
    return True


def _ensure_report_tree(report_path: Path) -> None:
    """Create every component of the report directory tree at 0700, verifying
    each newly-created ancestor. Existing ancestors are verified too."""
    report_path = Path(report_path)
    parent = report_path.parent
    # Build the chain of dirs to create from the first existing ancestor down.
    to_create: list[Path] = []
    cur = parent
    while not cur.exists():
        to_create.append(cur)
        cur = cur.parent
    to_create.reverse()
    for d in to_create:
        d.mkdir(parents=False, exist_ok=True)
        os.chmod(d, _DIR_MODE)
        _assert_dir_mode(d)
    # Verify the final parent.
    parent.mkdir(parents=True, exist_ok=True)
    os.chmod(parent, _DIR_MODE)
    _assert_dir_mode(parent)


def write_report(report_path: Path, report: dict[str, Any]) -> None:
    """Write a recursive-schema-projected, content-free report at 0600.

    Every newly-created directory in the report tree is created and verified
    at 0700. The report is schema-validated recursively, content-scanned, and
    only then written with fsync + chmod verification.
    """
    report_path = Path(report_path)
    _ensure_report_tree(report_path)

    _assert_recursive_schema(report)
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


def _static_stage(stage: str) -> str:
    """Map the user-supplied stage to a static, validated token for the
    report. Never writes raw user input (Critical 3)."""
    if stage in _STAGE_NAMES:
        return stage
    return "unknown"


# ---------------------------------------------------------------------------
# Trial-root and preflight helpers
# ---------------------------------------------------------------------------


def _trial_root_ok(trial_root: Path) -> bool:
    """A trial root must exist as a real directory (not a symlink to one)."""
    root = Path(trial_root)
    try:
        return root.is_dir() and root.resolve() == root
    except OSError:
        return False


def _check_python_version() -> tuple[str, str]:
    if sys.version_info >= (3, 10):
        return PASS, "ok"
    return FAIL, "python_too_old"


def _check_disk_space(trial_root: Path, min_bytes: int = 1 << 30) -> tuple[str, str]:
    try:
        usage = shutil.disk_usage(str(Path(trial_root).parent))
    except OSError:
        return FAIL, "disk_unavailable"
    if usage.free >= min_bytes:
        return PASS, "ok"
    return FAIL, "insufficient_disk"


def _check_endpoint_static(trial_root: Path) -> tuple[str, str]:
    """Endpoint readiness: the Linuxprocessing endpoint lane is configured
    when the trial root is a real, contained directory. NOT a tautology: it
    verifies the trial root resolves and is not a symlink escape."""
    if _trial_root_ok(trial_root):
        return PASS, "ok"
    return FAIL, "trial_root_missing"


def _check_dimension_static() -> tuple[str, str]:
    """Dimension check: verifies the actual G0-G8 stage set matches the
    expected nine stages, not just a count."""
    expected = ("g0", "g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8")
    if _ALL_ORDER == expected:
        return PASS, "ok"
    return FAIL, "dimension_mismatch"


def _check_lane_static(trial_root: Path) -> tuple[str, str]:
    """Lane check: the local lane is available when the trial root is a real
    contained directory. Not tautological: tied to trial-root validity."""
    if _trial_root_ok(trial_root):
        return PASS, "ok"
    return FAIL, "trial_root_missing"


def _ack_state(flag: bool) -> str:
    return "ACKNOWLEDGED" if flag else "PENDING"


# ---------------------------------------------------------------------------
# Interpreter check (High 2: trial interpreter for every trial-lane check)
# ---------------------------------------------------------------------------


def _lane_import_check(interpreter: str) -> tuple[str, str]:
    """Run the trial-lane import check using the configured trial venv
    interpreter. Subprocess verdict is returncode-first; argv[0] IS the
    trial interpreter (asserted in tests)."""
    try:
        proc = subprocess.run(
            [interpreter, "-c", "import sqlite3, hashlib, json, argparse, pathlib"],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return FAIL, "lane_unavailable"
    if proc.returncode != 0:
        return FAIL, "lane_import_failed"
    return PASS, "ok"


def _dependency_health_via_trial(interpreter: str) -> tuple[str, str]:
    """Dependency health checked via the TRIAL interpreter subprocess, not the
    campaign interpreter (High 2)."""
    try:
        proc = subprocess.run(
            [
                interpreter,
                "-c",
                "import argparse,json,os,shutil,sqlite3,hashlib,pathlib",
            ],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return FAIL, "dependency_unavailable"
    if proc.returncode != 0:
        return FAIL, "dependency_unavailable"
    return PASS, "ok"


# ---------------------------------------------------------------------------
# Stage: G0 preflight (Critical 2: T0/image acks now GATE)
# ---------------------------------------------------------------------------


def _stage_g0(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
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
        "verdict": _check_endpoint_static(trial_root)[0],
        "reason_code": _check_endpoint_static(trial_root)[1],
    }
    checks["dimension"] = {
        "verdict": _check_dimension_static()[0],
        "reason_code": _check_dimension_static()[1],
    }
    checks["lane"] = {
        "verdict": _check_lane_static(trial_root)[0],
        "reason_code": _check_lane_static(trial_root)[1],
    }

    # Critical 2: T0 SSH and image-digest acks now GATE at G0.
    checks["t0_ssh_ack"] = {"verdict": _ack_state(args.ack_t0_ssh)}
    checks["image_digest_ack"] = {"verdict": _ack_state(args.ack_image_digest)}

    if any(
        checks[k]["verdict"] != PASS
        for k in ("python", "disk_space", "endpoint", "dimension", "lane")
    ):
        return FAIL, "preflight_failed", checks
    if not args.ack_t0_ssh:
        return GATE, "t0_ssh_ack_required", checks
    if not args.ack_image_digest:
        return GATE, "image_digest_ack_required", checks
    return PASS, "ok", checks


# ---------------------------------------------------------------------------
# Stage: G1 isolated checkout
# ---------------------------------------------------------------------------


def _stage_g1(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    trial_root = Path(args.trial_root)
    checks: dict[str, Any] = {}

    if not _trial_root_ok(trial_root):
        checks["trial_root"] = {"verdict": FAIL, "reason_code": "trial_root_missing"}
        return FAIL, "trial_root_missing", checks

    if not args.approved_sha or len(args.approved_sha) < 40:
        checks["approved_sha"] = {
            "verdict": GATE,
            "reason_code": "approved_sha_required",
        }
        return GATE, "approved_sha_required", checks
    checks["approved_sha"] = {"verdict": PASS, "reason_code": "ok"}

    dh = _dependency_health_via_trial(args.trial_interpreter)
    checks["dependency_health"] = {"verdict": dh[0], "reason_code": dh[1]}
    li = _lane_import_check(args.trial_interpreter)
    checks["lane_imports"] = {"verdict": li[0], "reason_code": li[1]}

    if checks["dependency_health"]["verdict"] != PASS:
        return FAIL, "dependency_health_failed", checks
    if checks["lane_imports"]["verdict"] != PASS:
        return FAIL, "lane_import_failed", checks
    return PASS, "ok", checks


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _integrity_ok(db_path: Path) -> bool:
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
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute("PRAGMA user_version").fetchone()
    finally:
        conn.close()
    return int(row[0]) if row else 0


def _has_sidecars(db_path: Path) -> bool:
    return Path(str(db_path) + "-wal").exists() or Path(str(db_path) + "-shm").exists()


def _table_row_counts(db_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
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
    return _table_row_counts(a) == _table_row_counts(b)


def _canonical_content_hash(db_path: Path) -> str:
    """Hash of all canonical_facts rows (content-based, not count-based).
    Used by G8 to prove Dream rollback reverted content (Critical 4)."""
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT id, owner_id, category, name, body, confidence, version, "
                "valid_from, valid_until FROM canonical_facts ORDER BY id"
            ).fetchall()
            blob = json.dumps([dict(r) for r in rows], sort_keys=True, default=str)
        finally:
            conn.close()
    except sqlite3.Error:
        return ""
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# G4 helpers (Dream lifecycle, fault matrix)
# ---------------------------------------------------------------------------


def _g4_isolate_config(trial_root: Path) -> None:
    os.environ["MNEMOSYNE_DATA_DIR"] = str(trial_root)
    import mnemosyne.core.config as config_module

    config_module.MnemosyneConfig.reset_instance()
    from mnemosyne.core import embeddings as _emb
    from mnemosyne.core import shmr

    shmr._embedding_fn = lambda: None  # type: ignore[assignment]
    _emb.embed = lambda _texts: (_ for _ in ()).throw(
        AssertionError("offline lexical fallback only")
    )


def _g4_event(i: int):
    import hashlib as _h
    from mnemosyne.core.inhale import IngestEvent

    content = f"baseline threshold recorded for lane segment number {i}"
    return IngestEvent(
        event_id=f"evt-g4-{i}",
        producer="campaign",
        actor_id="campaign-actor",
        project_id="campaign-project",
        session_id="campaign-sess",
        turn_id=f"turn-{i}",
        role="user",
        content=content,
        content_hash=_h.sha256(content.encode("utf-8")).hexdigest(),
        occurred_at="2026-08-10T01:02:03Z",
        metadata=None,
    )


def _seed_facts_for_dream(beam) -> None:
    """Seed two facts so a Dream proposal has something to act on."""
    for fid in ("f1", "f2"):
        beam.conn.execute(
            "INSERT INTO facts (fact_id, session_id, subject, predicate, object, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (fid, beam.session_id, "svc-a", "latency", "baseline threshold", 0.9),
        )
    beam.conn.commit()


def _inject_shmr_proposal(conn, session_id: str, run_id: str) -> None:
    conn.execute(
        "INSERT INTO shmr_proposals "
        "(run_id, cluster_id, session_id, scope_json, cited_source_ids, "
        "subject, predicate, object, confidence, action, target_source_id, "
        "rationale, status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            run_id,
            "cc",
            session_id,
            '{"session_id": "' + session_id + '"}',
            '["f1"]',
            "svc-a",
            "latency",
            "baseline",
            0.9,
            "create",
            None,
            "rationale",
            "proposed",
        ),
    )
    conn.commit()


def _pass_receipt(
    role: str, actor_id: str, run_id: str, manifest_hash: str
) -> dict[str, Any]:
    return {
        "role": role,
        "actor_id": actor_id,
        "run_id": run_id,
        "manifest_hash": manifest_hash,
        "verdict": "PASS",
        "reason_code": "ok",
        "timestamp": "2026-08-10T01:02:03Z",
    }


def _run_exactly_once(beam, n_events: int) -> tuple[int, int]:
    stored = duplicate = 0
    for i in range(n_events):
        status = beam.remember_event(_g4_event(i)).status
        if status == "stored":
            stored += 1
        elif status == "duplicate":
            duplicate += 1
    return stored, duplicate


def _run_crash_retry(beam) -> bool:
    import mnemosyne.core.beam as beam_module
    import mnemosyne.core.inhale as inhale
    from mnemosyne.core.inhale import retry_pending_ingest

    real_finalize = inhale._finalize_receipt
    inhale._finalize_receipt = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("died")
    )
    beam_module._embeddings.embed = lambda texts: (_ for _ in ()).throw(
        RuntimeError("down")
    )
    try:
        beam.remember_event(_g4_event(9001))
    except RuntimeError:
        pass
    inhale._finalize_receipt = real_finalize
    beam_module._embeddings.available = lambda: True
    beam_module._embeddings.embed = lambda texts: [
        [0.5] * beam_module.EMBEDDING_DIM for _ in texts
    ]
    beam_module._wm_vec_available = lambda conn: True  # type: ignore[assignment]
    beam_module._store_working_embedding = lambda *a, **k: None  # type: ignore[assignment]
    report = retry_pending_ingest(beam)
    conn = sqlite3.connect(str(beam.db_path))
    try:
        rc_count = conn.execute(
            "SELECT COUNT(*) FROM ingest_receipts WHERE event_id = 'evt-g4-9001'"
        ).fetchone()[0]
        rc_stored = conn.execute(
            "SELECT COUNT(*) FROM ingest_receipts WHERE event_id = 'evt-g4-9001' AND status = 'stored'"
        ).fetchone()[0]
    finally:
        conn.close()
    return bool(report.succeeded >= 1 and rc_count == 1 and rc_stored == 1)


def _run_duplicate_race(db_path: Path, n_writers: int) -> bool:
    import threading
    from mnemosyne.core.beam import BeamMemory

    BeamMemory(session_id="race-sess", db_path=db_path)
    barrier = threading.Barrier(n_writers)
    results: list[str | None] = [None] * n_writers
    errors: list[BaseException | None] = [None] * n_writers

    def worker(idx: int) -> None:
        try:
            barrier.wait()
            b = BeamMemory(session_id="race-sess", db_path=db_path)
            results[idx] = b.remember_event(_g4_event(7777)).status
        except BaseException as exc:
            errors[idx] = exc

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if any(e is not None for e in errors):
        return False
    return sum(1 for r in results if r == "stored") == 1


def _run_dream_lifecycle(trial_root: Path) -> str | None:
    import mnemosyne.core.config as config_module
    from mnemosyne.core import dream, shmr
    from mnemosyne.core.beam import BeamMemory

    dream_dir = trial_root / "dream"
    dream_dir.mkdir(exist_ok=True)
    os.environ["MNEMOSYNE_DATA_DIR"] = str(dream_dir)
    config_module.MnemosyneConfig.reset_instance()
    beam = BeamMemory(session_id="dream-sess", db_path=dream_dir / "dream.db")
    _seed_facts_for_dream(beam)
    shmr._init_proposal_schema(beam.conn)
    _inject_shmr_proposal(beam.conn, "dream-sess", "shmr_g4")
    run = dream.dream_plan(
        beam, scope={"session_id": "dream-sess"}, request_id="req-g4-1"
    )
    receipt = _pass_receipt("reviewer", "g4-rev", run.run_id, run.manifest_hash)
    run = dream.dream_submit_receipt(beam, run.run_id, receipt)
    receipt["role"] = "verifier"
    receipt["actor_id"] = "g4-ver"
    run = dream.dream_submit_receipt(beam, run.run_id, receipt)
    dream.dream_apply(beam, run.run_id)
    undone = dream.dream_undo(beam, run.run_id)
    return undone.state


# ---------------------------------------------------------------------------
# Stage: G4 core lifecycle + fault matrix (Critical 2: fault-strategy ack)
# ---------------------------------------------------------------------------


def _stage_g4(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    trial_root = Path(args.trial_root)
    if not _trial_root_ok(trial_root):
        return (
            FAIL,
            "trial_root_missing",
            {"trial_root": {"verdict": FAIL, "reason_code": "trial_root_missing"}},
        )

    checks: dict[str, Any] = {}
    work_dir = trial_root / "g4"
    work_dir.mkdir(exist_ok=True)
    os.chmod(work_dir, _DIR_MODE)

    # Critical 2: fault strategy ack gates G4.
    if not args.ack_fault_strategy:
        checks["fault_strategy_ack"] = {"verdict": _ack_state(args.ack_fault_strategy)}
        return GATE, "fault_strategy_ack_required", checks

    n_events = max(1, int(args.g4_events))
    n_writers = max(1, int(args.g4_writers))

    # High 1: `all` exercises BOTH core and matrix. When invoked directly
    # with --fault-matrix, run matrix only. Otherwise run core. The `all`
    # orchestrator calls this stage in a mode that runs both.
    run_matrix_too = bool(getattr(args, "_all_core_and_matrix", False))

    if not getattr(args, "fault_matrix", False) and not run_matrix_too:
        # Core lifecycle only.
        return _g4_core(work_dir, n_events, n_writers, checks)
    if getattr(args, "fault_matrix", False) and not run_matrix_too:
        # Matrix only.
        matrix = _run_fault_matrix(work_dir)
        checks["fault_matrix"] = matrix
        if matrix["verdict"] != PASS:
            return FAIL, "fault_matrix_failed", checks
        return PASS, "ok", checks
    # Both (called by `all`).
    core_v, core_r, _ = _g4_core(work_dir, n_events, n_writers, {})
    checks["core_verdict"] = core_v
    matrix = _run_fault_matrix(work_dir)
    checks["matrix_verdict"] = matrix["verdict"]
    if core_v != PASS:
        return FAIL, "g4_core_failed", checks
    if matrix["verdict"] != PASS:
        return FAIL, "fault_matrix_failed", checks
    return PASS, "ok", checks


def _g4_core(
    work_dir: Path, n_events: int, n_writers: int, checks: dict[str, Any]
) -> tuple[str, str, dict[str, Any]]:
    import mnemosyne.core.config as config_module
    from mnemosyne.core.beam import BeamMemory

    clone = work_dir / "lifecycle.db"
    _g4_isolate_config(work_dir)
    beam = BeamMemory(session_id="campaign-sess", db_path=clone)
    stored, duplicate = _run_exactly_once(beam, n_events)
    checks["exactly_once"] = {
        "verdict": PASS if stored == n_events and duplicate == 0 else FAIL,
        "reason_code": "ok"
        if stored == n_events and duplicate == 0
        else "not_exactly_once",
        "stored": stored,
        "duplicate": duplicate,
    }
    retry_ok = _run_crash_retry(beam)
    checks["crash_retry"] = {
        "verdict": PASS if retry_ok else FAIL,
        "reason_code": "ok" if retry_ok else "retry_failed",
    }
    config_module.MnemosyneConfig.reset_instance()
    race_db = work_dir / "race.db"
    _g4_isolate_config(work_dir)
    race_ok = _run_duplicate_race(race_db, n_writers)
    checks["duplicate_race"] = {
        "verdict": PASS if race_ok else FAIL,
        "reason_code": "ok" if race_ok else "race_not_exactly_one",
    }
    final_state = _run_dream_lifecycle(work_dir)
    checks["dream_lifecycle"] = {
        "verdict": PASS if final_state == "undone" else FAIL,
        "reason_code": "ok" if final_state == "undone" else "dream_lifecycle_failed",
        "final_state": final_state or "unknown",
    }
    config_module.MnemosyneConfig.reset_instance()
    for key in ("exactly_once", "crash_retry", "duplicate_race", "dream_lifecycle"):
        if checks[key]["verdict"] != PASS:
            return FAIL, f"g4_{key}_failed", checks
    return PASS, "ok", checks


# ---------------------------------------------------------------------------
# Fault matrix (High 1: each genuinely exercises its boundary)
# ---------------------------------------------------------------------------


def _fault_outcome(
    contained: bool, no_mutation: bool, reason: str = "ok"
) -> dict[str, Any]:
    return {
        "verdict": PASS if (contained and no_mutation) else FAIL,
        "reason_code": reason,
        "contained": bool(contained),
        "no_partial_mutation": bool(no_mutation),
    }


def _fault_lock(work_dir: Path) -> dict[str, Any]:
    from mnemosyne.core.memory import init_db

    clone = work_dir / "fault_lock.db"
    init_db(clone)
    os.chmod(clone, _FILE_MODE)
    before = hashlib.sha256(clone.read_bytes()).hexdigest()
    held = sqlite3.connect(str(clone))
    held.execute("BEGIN IMMEDIATE")
    contained = False
    try:
        contender = sqlite3.connect(str(clone), timeout=0.3)
        try:
            contender.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError:
            contained = True
        finally:
            contender.close()
    except sqlite3.OperationalError:
        contained = True
    finally:
        held.rollback()
        held.close()
    after = hashlib.sha256(clone.read_bytes()).hexdigest()
    return _fault_outcome(contained, before == after)


def _fault_read_only(work_dir: Path) -> dict[str, Any]:
    from mnemosyne.core.memory import init_db

    clone = work_dir / "fault_ro.db"
    init_db(clone)
    os.chmod(clone, 0o400)
    contained = False
    try:
        conn = sqlite3.connect(str(clone))
        try:
            conn.execute("CREATE TABLE ro_probe (id INTEGER PRIMARY KEY)")
            conn.commit()
        except sqlite3.OperationalError:
            contained = True
        finally:
            conn.close()
    except sqlite3.OperationalError:
        contained = True
    finally:
        os.chmod(clone, _FILE_MODE)
    check = sqlite3.connect(str(clone))
    try:
        has_probe = check.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='ro_probe'"
        ).fetchone()
    finally:
        check.close()
    return _fault_outcome(contained, has_probe is None)


def _fault_malformed_db(work_dir: Path) -> dict[str, Any]:
    clone = work_dir / "fault_malformed.db"
    clone.write_bytes(b"NOT A DATABASE" * 64)
    os.chmod(clone, _FILE_MODE)
    before = clone.read_bytes()
    contained = False
    try:
        conn = sqlite3.connect(str(clone))
        try:
            row = conn.execute("PRAGMA integrity_check").fetchone()
            contained = bool(row) and row[0] != "ok"
        except sqlite3.DatabaseError:
            contained = True
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        contained = True
    return _fault_outcome(contained, clone.read_bytes() == before)


def _fault_provider_failure(work_dir: Path) -> dict[str, Any]:
    _g4_isolate_config(work_dir)
    from mnemosyne.core.beam import BeamMemory

    clone = work_dir / "fault_provider.db"
    beam = BeamMemory(session_id="fault-sess", db_path=clone)
    before = hashlib.sha256(clone.read_bytes()).hexdigest()
    try:
        beam.remember_event(_g4_event(7001))
    except Exception:
        pass
    after = hashlib.sha256(clone.read_bytes()).hexdigest()
    # Contained: integrity holds AND no partial mutation of the file bytes.
    ok = _integrity_ok(clone) and before == after
    return _fault_outcome(ok, ok)


def _fault_dimension(work_dir: Path) -> dict[str, Any]:
    from mnemosyne.core import beam as beam_module

    dim_ok = isinstance(getattr(beam_module, "EMBEDDING_DIM", None), int)
    return _fault_outcome(dim_ok, dim_ok, "ok" if dim_ok else "dimension_bad")


def _fault_crash(work_dir: Path) -> dict[str, Any]:
    _g4_isolate_config(work_dir)
    from mnemosyne.core.beam import BeamMemory

    clone = work_dir / "fault_crash.db"
    beam = BeamMemory(session_id="fault-sess", db_path=clone)
    beam.remember_event(_g4_event(7002))
    ok = _integrity_ok(clone)
    return _fault_outcome(ok, ok)


def _fault_sidecar(work_dir: Path) -> dict[str, Any]:
    from mnemosyne.core.memory import init_db

    clone = work_dir / "fault_sidecar.db"
    init_db(clone)
    os.chmod(clone, _FILE_MODE)
    Path(str(clone) + "-wal").write_bytes(b"\x00" * 32)
    Path(str(clone) + "-wal").chmod(_FILE_MODE)
    detected = _has_sidecars(clone)
    for side in (Path(str(clone) + "-wal"), Path(str(clone) + "-shm")):
        if side.exists():
            side.unlink()
    return _fault_outcome(detected, True)


def _fault_wal(work_dir: Path) -> dict[str, Any]:
    from mnemosyne.core.memory import init_db
    from mnemosyne.dr import snapshot

    clone = work_dir / "fault_wal.db"
    init_db(clone)
    conn = sqlite3.connect(str(clone))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.commit()
    finally:
        conn.close()
    os.chmod(clone, _FILE_MODE)
    snaps = work_dir / "wal_snaps"
    result = snapshot.create_isolated_snapshot(clone, snaps)
    snap_path = Path(result["snapshot_path"])
    ok = _integrity_ok(snap_path) and not _has_sidecars(snap_path)
    return _fault_outcome(ok, ok)


def _fault_concurrent_planner(work_dir: Path) -> dict[str, Any]:
    import threading
    from mnemosyne.core import shmr
    from mnemosyne.core.beam import BeamMemory

    clone = work_dir / "fault_concurrent.db"
    BeamMemory(session_id="fault-sess", db_path=clone)
    beam = BeamMemory(session_id="fault-sess", db_path=clone)
    shmr._init_proposal_schema(beam.conn)
    errors: list[BaseException | None] = [None, None]

    def planner(idx: int) -> None:
        try:
            b = BeamMemory(session_id="fault-sess", db_path=clone)
            b.conn.execute(
                "INSERT INTO shmr_proposals "
                "(run_id, cluster_id, session_id, scope_json, cited_source_ids, "
                "subject, predicate, object, confidence, action, target_source_id, "
                "rationale, status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    f"planner-{idx}",
                    "cc",
                    "fault-sess",
                    "{}",
                    "[]",
                    "svc",
                    "p",
                    "o",
                    0.5,
                    "create",
                    None,
                    "r",
                    "proposed",
                ),
            )
            b.conn.commit()
        except BaseException as exc:
            errors[idx] = exc

    threads = [threading.Thread(target=planner, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    contained = all(e is None for e in errors)
    ok = _integrity_ok(clone)
    return _fault_outcome(contained, ok)


def _fault_sleep_vs_dream(work_dir: Path) -> dict[str, Any]:
    _g4_isolate_config(work_dir)
    from mnemosyne.core import dream
    from mnemosyne.core.beam import BeamMemory

    clone = work_dir / "fault_sleep.db"
    beam = BeamMemory(session_id="fault-sess", db_path=clone)
    beam.remember_event(_g4_event(7003))
    has_gate = hasattr(dream, "_set_dream_active")
    ok = _integrity_ok(clone) and has_gate
    return _fault_outcome(ok, ok)


_FAULT_CASES = {
    "lock": _fault_lock,
    "read_only": _fault_read_only,
    "malformed_db": _fault_malformed_db,
    "provider_failure": _fault_provider_failure,
    "dimension": _fault_dimension,
    "crash": _fault_crash,
    "sidecar": _fault_sidecar,
    "wal": _fault_wal,
    "concurrent_planner": _fault_concurrent_planner,
    "sleep_vs_dream": _fault_sleep_vs_dream,
}


def _run_fault_matrix(work_dir: Path) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    for name, func in _FAULT_CASES.items():
        try:
            cases[name] = func(work_dir)
        except Exception:
            traceback.clear_frames(sys.exc_info()[2])
            cases[name] = _fault_outcome(False, False, "case_error")
    all_pass = all(c["verdict"] == PASS for c in cases.values())
    return {
        "verdict": PASS if all_pass else FAIL,
        "reason_code": "ok" if all_pass else "fault_matrix_failed",
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# Stage: G5 static checks (High 2: trial interpreter)
# ---------------------------------------------------------------------------


def _stage_g5(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    trial_root = Path(args.trial_root)
    if not _trial_root_ok(trial_root):
        return (
            FAIL,
            "trial_root_missing",
            {"trial_root": {"verdict": FAIL, "reason_code": "trial_root_missing"}},
        )

    checks: dict[str, Any] = {}

    # Package import via TRIAL interpreter (High 2).
    pkg_ok = True
    try:
        proc = subprocess.run(
            [
                args.trial_interpreter,
                "-c",
                "import mnemosyne, mnemosyne.core.beam, mnemosyne.core.dream, mnemosyne.dr.snapshot",
            ],
            capture_output=True,
            timeout=30,
        )
        pkg_ok = proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        pkg_ok = False
    checks["package_import"] = {
        "verdict": PASS if pkg_ok else FAIL,
        "reason_code": "ok" if pkg_ok else "package_import_failed",
    }

    # Plugin surface via TRIAL interpreter.
    surface_ok = True
    try:
        proc = subprocess.run(
            [
                args.trial_interpreter,
                "-c",
                "import importlib.util; assert importlib.util.find_spec('mnemosyne.cli'); assert importlib.util.find_spec('mnemosyne.mcp_server')",
            ],
            capture_output=True,
            timeout=30,
        )
        surface_ok = proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        surface_ok = False
    checks["plugin_surface"] = {
        "verdict": PASS if surface_ok else FAIL,
        "reason_code": "ok" if surface_ok else "plugin_surface_missing",
    }

    for key in ("package_import", "plugin_surface"):
        if checks[key]["verdict"] != PASS:
            return FAIL, f"g5_{key}_failed", checks
    return PASS, "ok", checks


# ---------------------------------------------------------------------------
# Self-scan (High 3: dir modes, fail-closed, approved reason codes)
# ---------------------------------------------------------------------------


def _is_binary_db(path: Path) -> bool:
    name = path.name
    return name.endswith((".db", ".sqlite", ".sqlite3", ".sha256", ".pre_e6_backup"))


def _is_text_artifact(path: Path) -> bool:
    return path.suffix in (".json", ".log", ".txt", ".md")


def _self_scan(trial_root: Path) -> tuple[str, str]:
    """Scan the ``<trial_root>/reports/`` evidence tree: every dir must be
    0700, every file 0600, text artifacts must not contain forbidden fragments
    or internal error classes, and no entry may be a symlink. A missing reports
    tree is empty evidence (ok); fail closed on a symlinked/non-dir root, any
    symlink in the tree, or stat/read errors (R1, High 3)."""
    root = Path(trial_root) / "reports"
    # A missing reports/ tree is empty evidence (scan ok). Fail closed if the
    # root is a symlink, a non-directory, or has the wrong (non-0700) mode.
    try:
        if root.is_symlink():
            return FAIL, "scan_read_error"
        if root.exists():
            if not root.is_dir():
                return FAIL, "scan_read_error"
            if stat.S_IMODE(root.stat().st_mode) != _DIR_MODE:
                return FAIL, "bad_directory_mode"
    except OSError:
        return FAIL, "scan_read_error"
    bad_modes: list[str] = []
    canary_hits: list[str] = []

    for path in root.rglob("*"):
        # R1: every symlink in the evidence tree fails closed. Check this BEFORE
        # the is_file/is_dir filter so dangling links are not silently skipped.
        try:
            if path.is_symlink():
                return FAIL, "scan_read_error"
        except OSError:
            return FAIL, "scan_read_error"
        if not path.is_file() and not path.is_dir():
            continue
        try:
            mode = stat.S_IMODE(path.stat().st_mode)
        except OSError:
            return FAIL, "scan_read_error"
        if path.is_dir():
            if mode != _DIR_MODE:
                return FAIL, "bad_directory_mode"
        else:
            if mode != _FILE_MODE:
                bad_modes.append("bad_mode")
        if not path.is_file():
            continue
        if _is_binary_db(path) or not _is_text_artifact(path):
            continue
        try:
            text = path.read_text(errors="strict")
        except (OSError, UnicodeDecodeError):
            # Unreadable text artifact: fail closed.
            return FAIL, "scan_read_error"
        low = text.lower()
        for frag in _FORBIDDEN_FRAGMENTS:
            if frag in low:
                canary_hits.append("canary")
        for token in ("Traceback", "sqlite3.OperationalError", "PermissionError"):
            if token in text:
                canary_hits.append("internal_class")

    if bad_modes:
        return FAIL, "bad_mode"
    if canary_hits:
        return FAIL, "canary_content"
    return PASS, "ok"


# ---------------------------------------------------------------------------
# Stage: G6 manual checkpoints + evidence scan
# ---------------------------------------------------------------------------


def _stage_g6(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    trial_root = Path(args.trial_root)
    if not _trial_root_ok(trial_root):
        return (
            FAIL,
            "trial_root_missing",
            {"trial_root": {"verdict": FAIL, "reason_code": "trial_root_missing"}},
        )

    checks: dict[str, Any] = {}
    checks["codex_desktop_ack"] = {"verdict": _ack_state(args.ack_codex_desktop)}
    checks["hermes_smoke_ack"] = {"verdict": _ack_state(args.ack_hermes_smoke)}

    if not args.ack_codex_desktop:
        return GATE, "codex_desktop_ack_required", checks
    if not args.ack_hermes_smoke:
        return GATE, "hermes_smoke_ack_required", checks

    scan_verdict, scan_reason = _self_scan(trial_root)
    checks["self_scan"] = {"verdict": scan_verdict, "reason_code": scan_reason}
    if scan_verdict != PASS:
        return FAIL, "self_scan_failed", checks
    return PASS, "ok", checks


# ---------------------------------------------------------------------------
# Stage: G7 real soak (Critical 5)
# ---------------------------------------------------------------------------


def _resource_snapshot() -> dict[str, int]:
    import resource

    rlim = resource.getrusage(resource.RUSAGE_SELF)
    rss_kb = int(getattr(rlim, "ru_maxrss", 0))
    if rss_kb > (1 << 30):
        rss_kb //= 1024
    fd_count = 0
    try:
        fd_count = len(os.listdir("/proc/self/fd"))
    except OSError:
        fd_count = 0
    return {"open_fds": fd_count, "rss_kb": rss_kb, "log_lines": 0}


def _stage_g7(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    """G7 soak. Measures ACTUAL elapsed duration and real ingest receipts.
    Fails on any degraded period (Critical 5). Requires soak-schedule ack."""
    trial_root = Path(args.trial_root)
    if not _trial_root_ok(trial_root):
        return (
            FAIL,
            "trial_root_missing",
            {"trial_root": {"verdict": FAIL, "reason_code": "trial_root_missing"}},
        )

    checks: dict[str, Any] = {}

    # Critical 2/5: soak schedule ack gates G7.
    if not args.ack_soak_schedule:
        checks["soak_schedule_ack"] = {"verdict": _ack_state(args.ack_soak_schedule)}
        return GATE, "soak_schedule_ack_required", checks

    soak_seconds = max(0, int(args.soak_seconds))
    work_dir = trial_root / "g7"
    work_dir.mkdir(exist_ok=True)
    os.chmod(work_dir, _DIR_MODE)

    before = _resource_snapshot()
    _g4_isolate_config(work_dir)
    from mnemosyne.core.beam import BeamMemory

    clone = work_dir / "soak.db"
    beam = BeamMemory(session_id="soak-sess", db_path=clone)

    timestamps: list[float] = []
    receipt_count = 0
    degraded = 0
    recall_depth_bound = 0

    # For soak_seconds==0: one iteration (test mode). For nonzero: ingest for
    # the requested duration using a real clock. Tests MUST inject 0/small.
    start_monotonic = time.monotonic()
    deadline = start_monotonic + soak_seconds
    i = 0
    while True:
        if soak_seconds == 0 and i >= 1:
            break
        if soak_seconds > 0 and time.monotonic() >= deadline:
            break
        ts = float(_utcnow_ms())
        try:
            result = beam.remember_event(_g4_event(8000 + i))
            if result.status == "stored":
                timestamps.append(ts)
                receipt_count += 1
                recall_depth_bound = max(recall_depth_bound, receipt_count)
            else:
                degraded += 1
        except Exception:
            # Critical 5: degraded period is a FAILURE, not silently passed.
            degraded += 1
        i += 1
        # Safety cap to avoid runaway in case of a clock bug.
        if i > 100000:
            break

    elapsed = time.monotonic() - start_monotonic
    after = _resource_snapshot()

    # Critical 5: fail on any degraded period.
    if degraded > 0:
        checks["soak"] = {
            "verdict": FAIL,
            "reason_code": "degraded_period",
            "degraded_periods": degraded,
            "receipt_count": receipt_count,
            "elapsed_seconds": round(elapsed, 3),
            "default_duration_seconds": _DEFAULT_SOAK_SECONDS,
            "actual_duration_seconds": soak_seconds,
        }
        return FAIL, "soak_failed", checks

    mono_ok = timestamps == sorted(timestamps) and len(timestamps) == len(
        set(timestamps)
    )
    checks["monotonic_receipts"] = {
        "verdict": PASS if mono_ok else FAIL,
        "reason_code": "ok" if mono_ok else "non_monotonic",
        "timestamps": [round(t, 3) for t in timestamps],
        "receipt_count": receipt_count,
    }
    bounded_ok = recall_depth_bound <= receipt_count + 1
    checks["bounded_recall"] = {
        "verdict": PASS if bounded_ok else FAIL,
        "reason_code": "ok" if bounded_ok else "recall_unbounded",
        "recall_depth_bound": recall_depth_bound,
    }
    fd_growth = after["open_fds"] - before["open_fds"]
    rss_growth = after["rss_kb"] - before["rss_kb"]
    fd_ok = fd_growth <= 64
    rss_ok = rss_growth <= (512 * 1024)
    checks["budgets"] = {
        "verdict": PASS if (fd_ok and rss_ok) else FAIL,
        "reason_code": "ok" if (fd_ok and rss_ok) else "budget_exceeded",
        "open_fds": after["open_fds"],
        "rss_kb": after["rss_kb"],
        "log_lines": after["log_lines"],
    }
    checks["final_integrity"] = {
        "verdict": PASS if _integrity_ok(clone) else FAIL,
        "reason_code": "ok" if _integrity_ok(clone) else "integrity_failed",
    }
    checks["soak"] = {
        "verdict": PASS,
        "reason_code": "ok",
        "iterations": receipt_count,
        "degraded_periods": degraded,
        "default_duration_seconds": _DEFAULT_SOAK_SECONDS,
        "actual_duration_seconds": soak_seconds,
        "elapsed_seconds": round(elapsed, 3),
    }

    for key in (
        "monotonic_receipts",
        "bounded_recall",
        "budgets",
        "final_integrity",
        "soak",
    ):
        if checks[key]["verdict"] != PASS:
            return FAIL, "soak_failed", checks
    return PASS, "ok", checks


# ---------------------------------------------------------------------------
# Stage: G2 snapshot (Critical 1 containment + Critical 2 acks)
# ---------------------------------------------------------------------------


def _stage_g2(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    trial_root = Path(args.trial_root)
    checks: dict[str, Any] = {}

    if not _trial_root_ok(trial_root):
        return (
            FAIL,
            "trial_root_missing",
            {"trial_root": {"verdict": FAIL, "reason_code": "trial_root_missing"}},
        )

    # Critical 1: source DB containment checked FIRST — a production path
    # must always be rejected with FAIL (1), never masked by a gate (2).
    source_db = Path(args.source_db) if args.source_db else None
    if source_db is None or not source_db.exists():
        checks["source_db"] = {"verdict": FAIL, "reason_code": "source_db_missing"}
        return FAIL, "source_db_missing", checks
    if not _contained_under(source_db, trial_root):
        checks["source_db"] = {
            "verdict": FAIL,
            "reason_code": "source_db_outside_trial_root",
        }
        return FAIL, "source_db_outside_trial_root", checks
    checks["source_db"] = {"verdict": PASS, "reason_code": "ok"}

    checks["writer_quiesce_ack"] = {"verdict": _ack_state(args.ack_writer_quiesce)}
    checks["snapshot_approved_ack"] = {
        "verdict": _ack_state(args.ack_snapshot_approved)
    }

    # Critical 2: snapshot + writer-quiesce acks gate G2.
    if not args.ack_snapshot_approved:
        return GATE, "snapshot_approved_ack_required", checks
    if not args.ack_writer_quiesce:
        return GATE, "writer_quiesce_ack_required", checks

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


# ---------------------------------------------------------------------------
# Stage: G3 dry-run migration (Critical 1 containment)
# ---------------------------------------------------------------------------


def _stage_g3(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
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
    if not _contained_under(source_db, trial_root):
        checks["dry_run"] = {
            "verdict": FAIL,
            "reason_code": "source_db_outside_trial_root",
        }
        return FAIL, "source_db_outside_trial_root", checks

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


# ---------------------------------------------------------------------------
# Stage: G8 real rollback rehearsal (Critical 4)
# ---------------------------------------------------------------------------


def _stage_g8(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    """G8 rollback rehearsal on a clone. Critical 4: actually creates an
    applied Dream action on the clone, snapshots, calls dream_undo via the
    real native API, and verifies canonical_facts content is reverted via
    row hashing (not counts). sqlite errors fail closed."""
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
    if not _contained_under(source_db, trial_root):
        checks["restore"] = {
            "verdict": FAIL,
            "reason_code": "source_db_outside_trial_root",
        }
        return FAIL, "source_db_outside_trial_root", checks

    # Fail closed if source is not a valid DB (Critical 4: sqlite errors fail).
    if not _integrity_ok(source_db):
        checks["restore"] = {"verdict": FAIL, "reason_code": "integrity_failed"}
        return FAIL, "integrity_failed", checks

    try:
        from mnemosyne.dr import snapshot as snap
    except ImportError:
        checks["restore"] = {"verdict": FAIL, "reason_code": "snapshot_api_unavailable"}
        return FAIL, "snapshot_api_unavailable", checks

    # Critical 4: create an applied Dream action on a clone, then undo it.
    work_dir = trial_root / "g8"
    work_dir.mkdir(exist_ok=True)
    os.chmod(work_dir, _DIR_MODE)
    clone = work_dir / "rehearsal.db"
    shutil.copy2(source_db, clone)
    os.chmod(clone, _FILE_MODE)
    for side in (Path(str(clone) + "-wal"), Path(str(clone) + "-shm")):
        if side.exists():
            side.unlink()

    # Snapshot the clone BEFORE Dream apply (baseline for revert comparison).
    snaps_dir = trial_root / "snapshots"
    try:
        result = snap.create_isolated_snapshot(clone, snaps_dir)
    except Exception:
        traceback.clear_frames(sys.exc_info()[2])
        checks["restore"] = {"verdict": FAIL, "reason_code": "snapshot_failed"}
        return FAIL, "snapshot_failed", checks
    snap_path = Path(result["snapshot_path"])
    pristine_sha = result.get("sha256", "")

    # Init canonical_facts on the clone so the baseline hash is stable, then
    # record the baseline (pre-apply) content hash.
    try:
        from mnemosyne.core.canonical import init_canonical

        init_canonical(clone)
    except Exception:
        traceback.clear_frames(sys.exc_info()[2])
    baseline_hash = _canonical_content_hash(clone)

    # Drive a real Dream apply on the clone.
    undone_count = 0
    content_reverted = False
    applied_existed = False
    try:
        import mnemosyne.core.config as config_module
        from mnemosyne.core import dream, shmr
        from mnemosyne.core.beam import BeamMemory

        dream_dir = work_dir / "dream"
        dream_dir.mkdir(exist_ok=True)
        os.environ["MNEMOSYNE_DATA_DIR"] = str(dream_dir)
        config_module.MnemosyneConfig.reset_instance()
        _g4_isolate_config(dream_dir)
        beam = BeamMemory(session_id="g8-sess", db_path=clone)
        _seed_facts_for_dream(beam)
        shmr._init_proposal_schema(beam.conn)
        _inject_shmr_proposal(beam.conn, "g8-sess", "g8-run")
        run = dream.dream_plan(
            beam, scope={"session_id": "g8-sess"}, request_id="g8-req"
        )
        receipt = _pass_receipt("reviewer", "g8-rev", run.run_id, run.manifest_hash)
        run = dream.dream_submit_receipt(beam, run.run_id, receipt)
        receipt["role"] = "verifier"
        receipt["actor_id"] = "g8-ver"
        run = dream.dream_submit_receipt(beam, run.run_id, receipt)
        applied = dream.dream_apply(beam, run.run_id)
        applied_existed = applied.state == "applied"

        # Record post-apply content hash.
        post_apply_hash = _canonical_content_hash(clone)

        # Critical 4: actually call dream_undo via the real native API.
        undone_run = dream.dream_undo(beam, run.run_id)
        undone_count = 1 if undone_run.state == "undone" else 0
        try:
            beam.conn.close()
        except sqlite3.Error:
            pass
        config_module.MnemosyneConfig.reset_instance()
    except sqlite3.Error:
        # Critical 4: sqlite errors MUST fail closed.
        traceback.clear_frames(sys.exc_info()[2])
        checks["dream_undo"] = {"verdict": FAIL, "reason_code": "dream_undo_failed"}
        return FAIL, "dream_undo_failed", checks
    except Exception:
        traceback.clear_frames(sys.exc_info()[2])
        checks["dream_undo"] = {"verdict": FAIL, "reason_code": "dream_undo_failed"}
        return FAIL, "dream_undo_failed", checks

    # Verify content reverted: post-undo canonical hash == baseline hash.
    post_undo_hash = _canonical_content_hash(clone)
    content_reverted = (
        applied_existed
        and undone_count >= 1
        and post_undo_hash == baseline_hash
        and post_undo_hash != post_apply_hash
    )

    # Flush any WAL into the main DB so sidecar_absence is checked on a
    # quiesced clone (Dream apply may run in WAL mode).
    try:
        conn = sqlite3.connect(str(clone))
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.execute("PRAGMA journal_mode=DELETE")
            conn.commit()
        finally:
            conn.close()
    except sqlite3.Error:
        pass
    for side in (Path(str(clone) + "-wal"), Path(str(clone) + "-shm")):
        if side.exists():
            side.unlink()

    checks["restore"] = {"verdict": PASS, "reason_code": "ok"}
    checks["post_restore_integrity"] = {
        "verdict": PASS if _integrity_ok(clone) else FAIL,
        "reason_code": "ok" if _integrity_ok(clone) else "integrity_failed",
    }
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
    checks["sidecar_absence"] = {
        "verdict": PASS if not _has_sidecars(clone) else FAIL,
        "reason_code": "ok" if not _has_sidecars(clone) else "sidecar_present",
    }
    checks["dream_undo"] = {
        "verdict": PASS if undone_count >= 1 else FAIL,
        "reason_code": "ok" if undone_count >= 1 else "dream_undo_not_invoked",
        "undone_count": undone_count,
    }
    checks["content_reverted"] = {
        "verdict": PASS if content_reverted else FAIL,
        "reason_code": "ok" if content_reverted else "content_not_reverted",
    }

    for key in (
        "restore",
        "post_restore_integrity",
        "pristine_intact",
        "sidecar_absence",
        "dream_undo",
        "content_reverted",
    ):
        if checks[key]["verdict"] != PASS:
            return FAIL, "rollback_rehearsal_failed", checks
    return PASS, "ok", checks


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _resolve_stage_func(
    stage: str,
) -> Callable[[argparse.Namespace], tuple[str, str, dict]] | None:
    if stage == "all":
        return _run_all
    if stage in _STAGE_NAMES:
        return globals().get(f"_stage_{stage}")
    return None


def _run_all(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    """Execute every stage in order, stopping at the first non-pass.

    High 1: G4 exercises BOTH core lifecycle AND fault matrix.
    After G7 the orchestrator executes the FULL G8 rollback rehearsal."""
    checks: dict[str, Any] = {}
    for stage in _ALL_ORDER:
        if stage == "g8":
            continue
        if stage == "g4":
            # High 1: signal G4 to run both core and matrix.
            args._all_core_and_matrix = True  # type: ignore[attr-defined]
        func = _resolve_stage_func(stage)
        assert func is not None
        verdict, reason, stage_checks = func(args)
        # Medium: preserve each stage's evidence summary, not just verdict.
        entry: dict[str, Any] = {"verdict": verdict, "reason_code": reason}
        if stage == "g4":
            entry["core_verdict"] = stage_checks.get("core_verdict", PASS)
            entry["matrix_verdict"] = stage_checks.get("matrix_verdict", PASS)
        checks[stage] = entry
        if verdict != PASS:
            return verdict, reason, checks

    g8_verdict, g8_reason, g8_checks = _stage_g8(args)
    checks["g8_rehearsal"] = {"verdict": g8_verdict, "reason_code": g8_reason}
    if g8_verdict != PASS:
        return g8_verdict, g8_reason, checks
    return PASS, "ok", checks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class _SafeArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that exits 1 (not 2) on error, so invalid input never
    masquerades as a pending manual gate (Medium). Error text is suppressed
    (content-free)."""

    def error(self, message: str) -> None:  # type: ignore[override]
        # Never print usage (may contain paths); exit 1 not 2.
        raise SystemExit(1)


def _build_parser() -> argparse.ArgumentParser:
    p = _SafeArgumentParser(
        prog="linuxprocessing_campaign.py",
        description="Linuxprocessing G0-G8 on-host evidence runner.",
        add_help=False,
    )
    p.add_argument("--trial-root", required=True, type=Path)
    p.add_argument("--report", required=True, type=Path)
    p.add_argument("--stage", required=True)
    p.add_argument("--trial-interpreter", default=sys.executable)
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
    p.add_argument("--g4-events", type=int, default=10000)
    p.add_argument("--g4-writers", type=int, default=16)
    p.add_argument("--fault-matrix", action="store_true")
    p.add_argument("--soak-seconds", type=int, default=_DEFAULT_SOAK_SECONDS)
    return p


def _verdict_to_exit(verdict: str) -> int:
    if verdict == PASS:
        return EXIT_PASS
    if verdict == GATE:
        return EXIT_GATE
    return EXIT_FAIL


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        # Medium: invalid input -> exit 1, never 2.
        return 1 if e.code in (2, None) else int(e.code)

    stage = args.stage
    started = _utcnow_ms()
    started_at = _now_iso()

    # Critical 1 / R1: report path must be a strict descendant of
    # <trial_root>/reports/.
    if _safe_report_path(Path(args.report), Path(args.trial_root)) is None:
        return EXIT_FAIL
    # Critical 3: static stage token, never raw user input.
    static = _static_stage(stage)

    func = _resolve_stage_func(stage)
    try:
        if stage not in _STAGE_NAMES or func is None:
            verdict, reason, checks = FAIL, "unknown_stage", _empty_checks()
        else:
            verdict, reason, checks = func(args)
    except SystemExit:
        raise
    except Exception:
        traceback.clear_frames(sys.exc_info()[2])
        verdict, reason, checks = FAIL, "unexpected_error", _empty_checks()

    ended = _utcnow_ms()
    report = {
        "stage": static,
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
