"""
Mnemosyne Diagnostics
=====================
PII-safe debug logging for troubleshooting installation and runtime issues.

Logs to $HERMES_HOME/mnemosyne/logs/diagnose_YYYY-MM-DD_HHMMSS.jsonl,
or ~/.hermes/mnemosyne/logs when HERMES_HOME is unset.
Never includes memory content, user queries, or API keys.

Supports --fix mode: auto-installs missing dependencies.
"""

import importlib.metadata  # noqa: F401  (monkeypatched by tests; runtime_diagnostics calls .version)
import json
import os
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path

from mnemosyne.runtime_diagnostics import collect_runtime_diagnostics

def _default_log_dir() -> Path:
    """Resolve diagnostics beside the active Hermes home."""
    hermes_home = os.environ.get("HERMES_HOME")
    base = Path(hermes_home).expanduser() if hermes_home else Path.home() / ".hermes"
    return base / "mnemosyne" / "logs"


LOG_DIR = _default_log_dir()

# Map of missing dependency checks to pip install commands
FIX_MAP = {
    "fastembed": {
        "check": lambda e: e["check"] == "fastembed" and e["status"] == "MISSING",
        "install": ["pip", "install", "mnemosyne-memory[embeddings]"],
        "label": "fastembed (embeddings engine)",
    },
    "sqlite_vec": {
        "check": lambda e: e["check"] == "sqlite_vec" and e["status"] == "MISSING",
        "install": ["pip", "install", "sqlite-vec"],
        "label": "sqlite-vec (vector search)",
    },
    "numpy": {
        "check": lambda e: e["check"] == "numpy" and e["status"] == "MISSING",
        "install": ["pip", "install", "numpy"],
        "label": "numpy",
    },
    "huggingface_hub": {
        "check": lambda e: e["check"] == "huggingface_hub" and e["status"] == "MISSING",
        "install": ["pip", "install", "huggingface_hub"],
        "label": "huggingface_hub",
    },
}


def _ensure_log_dir():
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _log_path() -> Path:
    _ensure_log_dir()
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return LOG_DIR / f"diagnose_{ts}.jsonl"


def _safe_env(name: str) -> str:
    """Return env var presence indicator, never the value."""
    val = os.environ.get(name, "")
    return "set" if val else "unset"


def _memory_orphan_diagnostics(conn) -> dict[str, int]:
    """Return read-only memory reference integrity diagnostics."""
    foreign_keys_enabled = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    live_id_tables = [
        table
        for table in ("working_memory", "memories", "episodic_memory")
        if table in tables
    ]
    live_ids = set()
    for table in live_id_tables:
        live_ids.update(
            row[0]
            for row in conn.execute(f"SELECT id FROM {table} WHERE id IS NOT NULL")
        )

    diagnostics = {
        "gists_total": 0,
        "gists_with_memory_id": 0,
        "gists_orphan_memory_id": 0,
        "memory_embeddings_total": 0,
        "memory_embeddings_orphan_memory_id": 0,
        "orphan_memory_id_overlap": 0,
    }

    orphan_gist_ids = set()
    if "gists" in tables:
        diagnostics["gists_total"] = int(
            conn.execute("SELECT COUNT(*) FROM gists").fetchone()[0]
        )
        gist_memory_ids = [
            row[0]
            for row in conn.execute(
                "SELECT memory_id FROM gists WHERE memory_id IS NOT NULL"
            )
        ]
        diagnostics["gists_with_memory_id"] = len(gist_memory_ids)
        orphan_gist_ids = {mid for mid in gist_memory_ids if mid not in live_ids}
        diagnostics["gists_orphan_memory_id"] = sum(
            1 for mid in gist_memory_ids if mid in orphan_gist_ids
        )

    orphan_embedding_ids = set()
    if "memory_embeddings" in tables:
        diagnostics["memory_embeddings_total"] = int(
            conn.execute("SELECT COUNT(*) FROM memory_embeddings").fetchone()[0]
        )
        embedding_memory_ids = [
            row[0]
            for row in conn.execute(
                "SELECT memory_id FROM memory_embeddings WHERE memory_id IS NOT NULL"
            )
        ]
        orphan_embedding_ids = {mid for mid in embedding_memory_ids if mid not in live_ids}
        diagnostics["memory_embeddings_orphan_memory_id"] = sum(
            1 for mid in embedding_memory_ids if mid in orphan_embedding_ids
        )

    diagnostics["orphan_memory_id_overlap"] = len(
        orphan_gist_ids.intersection(orphan_embedding_ids)
    )
    diagnostics["foreign_keys_enabled"] = int(foreign_keys_enabled)
    return diagnostics




def _sqlite_integrity_diagnostics(conn) -> dict[str, str]:
    """Return PII-safe SQLite integrity diagnostics."""
    try:
        rows = conn.execute("PRAGMA quick_check").fetchall()
        result = "; ".join(str(row[0]) for row in rows) if rows else "unknown"
    except Exception as exc:
        return {"quick_check": "ERROR", "detail": str(exc)[:200]}
    return {"quick_check": result, "detail": ""}


def run_diagnostics(
    *,
    repair_vec_working: bool = False,
    dry_run: bool = False,
    bank: str | None = None,
    read_only: bool = False,
) -> dict:
    """
    Run full diagnostic scan and write PII-safe log.
    Returns summary dict for display.

    Args:
        repair_vec_working: If true, idempotently backfill missing rows in the
            dedicated working-memory sqlite-vec table from memory_embeddings.
        dry_run: With repair_vec_working, report what would be repaired without
            writing.
        bank: Optional named bank to diagnose. When provided, diagnostics run
            against the bank's own SQLite DB (data/banks/<bank>/mnemosyne.db).
            When None, the default/profile-root DB is used.
        read_only: If true, never writes a log file, never constructs a default
            writable database, never runs repair, and never modifies SQLite.
            Repairs requested through the read-only path are rejected with a
            structured status instead of being executed.
    """
    if read_only:
        return _run_read_only_diagnostics(
            repair_vec_working=repair_vec_working, bank=bank
        )
    log_path = _log_path()
    entries: list[dict] = []
    resolved_bank: str | None = None
    resolved_db: str | None = None

    def log(category: str, check: str, status: str, detail: str = ""):
        entry = {
            "ts": datetime.now().isoformat(),
            "category": category,
            "check": check,
            "status": status,
            "detail": detail
        }
        entries.append(entry)
        return entry

    # --- Pure runtime/dependency/capability checks (no provider construction) ---
    for check in collect_runtime_diagnostics()["checks"]:
        log(check["category"], check["check"], check["status"], check["detail"])

    # --- Database state ---
    try:
        from mnemosyne.core.memory import Mnemosyne
        if bank:
            mem = Mnemosyne(session_id="hermes_default", bank=bank)
        else:
            mem = Mnemosyne()
        stats = mem.get_stats()

        # PII-safe: counts and config only, never content
        log("db", "legacy_total", str(stats.get("total_memories", 0)))
        log("db", "total_sessions", str(stats.get("total_sessions", 0)))

        beam = stats.get("beam", {})
        wm = beam.get("working_memory", {})
        ep = beam.get("episodic_memory", {})

        log("db", "working_total", str(wm.get("total", 0)))
        log("db", "episodic_total", str(ep.get("total", 0)))
        log("db", "episodic_vectors", str(ep.get("vectors", 0)))
        log("db", "episodic_vec_type", ep.get("vec_type", "none"))
        log("db", "db_path", stats.get("database", "unknown"))

        if bank:
            log("db", "resolved_bank", bank)
            log("db", "resolved_db", stats.get("database", "unknown"))
            resolved_bank = bank
            resolved_db = stats.get("database", "unknown")
        else:
            resolved_bank = "default"
            resolved_db = stats.get("database", "unknown")

        try:
            integrity = _sqlite_integrity_diagnostics(mem.beam.conn)
            quick_check = integrity["quick_check"]
            log(
                "db",
                "sqlite_quick_check",
                "OK" if quick_check == "ok" else quick_check,
                integrity.get("detail", ""),
            )
        except Exception as exc:
            log("db", "sqlite_quick_check", "ERROR", str(exc))

        try:
            orphan_diag = _memory_orphan_diagnostics(mem.beam.conn)
            log("db", "foreign_keys_enabled", "YES" if orphan_diag["foreign_keys_enabled"] else "NO")
            log("db", "gists_total", str(orphan_diag["gists_total"]))
            log("db", "gists_with_memory_id", str(orphan_diag["gists_with_memory_id"]))
            log("db", "gists_orphan_memory_id", str(orphan_diag["gists_orphan_memory_id"]))
            log("db", "memory_embeddings_total", str(orphan_diag["memory_embeddings_total"]))
            log("db", "memory_embeddings_orphan_memory_id", str(orphan_diag["memory_embeddings_orphan_memory_id"]))
            log("db", "orphan_memory_id_overlap", str(orphan_diag["orphan_memory_id_overlap"]))
        except Exception as exc:
            log("db", "memory_orphan_diagnostics", "ERROR", str(exc))

        try:
            from mnemosyne.core.hygiene import noise_summary as _noise_summary
            db_path_str = stats.get("database")
            if not db_path_str or db_path_str == "unknown":
                log("db", "hygiene_noise_summary", "SKIPPED", "database path unavailable")
            else:
                hygiene = _noise_summary(Path(db_path_str), limit=200)
                log("db", "hygiene_noise_scanned", str(hygiene.get("total_scanned", 0)))
                log("db", "hygiene_noise_candidates", str(hygiene.get("total_candidates", 0)))
                log("db", "hygiene_noise_ratio", str(hygiene.get("candidate_ratio", 0.0)))
                log("db", "hygiene_noise_with_secrets", str(hygiene.get("with_secrets", 0)))
        except Exception as exc:
            log("db", "hygiene_noise_summary", "ERROR", str(exc))

        try:
            from mnemosyne.core.beam import repair_vec_working as _repair_vec_working, vec_working_coverage
            if repair_vec_working:
                vec_working = _repair_vec_working(mem.beam.conn, dry_run=dry_run)
                after = vec_working.get("after", {})
                log("db", "vec_working_repair_status", vec_working.get("status", "unknown"))
                log("db", "vec_working_repair_inserted", str(vec_working.get("inserted", 0)))
            else:
                after = vec_working_coverage(mem.beam.conn)
                vec_working = None
            log("db", "vec_working_status", after.get("status", "unknown"))
            log("db", "vec_working_available", "YES" if after.get("vec_working_available") else "NO")
            log("db", "vec_working_rows", str(after.get("vec_working_rows", 0)))
            log("db", "vec_working_missing", str(after.get("missing_vec_working_rows", 0)))
            log("db", "vec_working_orphans", str(after.get("orphan_vec_working_rows", 0)))
            log("db", "working_embedding_rows", str(after.get("working_embedding_rows", 0)))
        except Exception as exc:
            if repair_vec_working:
                log("db", "vec_working_repair_status", "ERROR", str(exc))
            else:
                log("db", "vec_working_coverage", "ERROR", str(exc))
    except Exception as e:
        log("db", "stats", "ERROR", str(e))

    # --- Environment variables (presence only, never values) ---
    env_vars = [
        "MNEMOSYNE_DATA_DIR",
        "MNEMOSYNE_LLM_ENABLED",
        "MNEMOSYNE_LLM_BASE_URL",
        "MNEMOSYNE_VEC_TYPE",
        "MNEMOSYNE_WM_MAX_ITEMS",
        "HERMES_HOME",
    ]
    for var in env_vars:
        log("env", var, _safe_env(var))

    # --- Write log file ---
    with open(log_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    # --- Build summary ---
    non_failure_statuses = ("OK", "YES", "set", "OPTIONAL")
    summary = {
        "log_path": str(log_path),
        "checks_total": len(entries),
        "checks_passed": sum(1 for e in entries if e["status"] in non_failure_statuses),
        "checks_failed": sum(
            1 for e in entries if str(e["status"]).upper() in ("MISSING", "NO", "ERROR")
        ),
        "key_findings": [],
        "fixable": [],
        "entries": entries,
        "resolved_bank": resolved_bank,
        "resolved_db": resolved_db,
    }

    # Auto-detect common problems
    embed_ok = any(e["check"] == "embeddings_available" and e["status"] == "YES" for e in entries)
    vec_ok = any(e["check"] == "sqlite_vec_available" and e["status"] == "YES" for e in entries)
    ep_vec = next((e for e in entries if e["check"] == "episodic_vectors"), None)
    ep_vec_type = next((e for e in entries if e["check"] == "episodic_vec_type"), None)
    vec_working_status = next((e for e in entries if e["check"] == "vec_working_status"), None)
    vec_working_missing = next((e for e in entries if e["check"] == "vec_working_missing"), None)
    vec_working_rows = next((e for e in entries if e["check"] == "vec_working_rows"), None)
    working_embedding_rows = next((e for e in entries if e["check"] == "working_embedding_rows"), None)
    vec_working_repair_status = next((e for e in entries if e["check"] == "vec_working_repair_status"), None)
    vec_working_repair_inserted = next((e for e in entries if e["check"] == "vec_working_repair_inserted"), None)
    sqlite_quick_check = next((e for e in entries if e["check"] == "sqlite_quick_check"), None)
    hygiene_noise_candidates = next((e for e in entries if e["check"] == "hygiene_noise_candidates"), None)
    hygiene_noise_scanned = next((e for e in entries if e["check"] == "hygiene_noise_scanned"), None)
    hygiene_noise_with_secrets = next((e for e in entries if e["check"] == "hygiene_noise_with_secrets"), None)

    if sqlite_quick_check and sqlite_quick_check["status"] != "OK":
        summary["key_findings"].append(
            f"SQLite quick_check reported: {sqlite_quick_check['status']}"
        )
    if hygiene_noise_candidates and hygiene_noise_scanned:
        candidates = int(hygiene_noise_candidates["status"])
        scanned = int(hygiene_noise_scanned["status"])
        if scanned:
            summary["key_findings"].append(
                f"Hygiene noise summary: {candidates} candidates across {scanned} scanned rows (read-only sample)"
            )
            if hygiene_noise_with_secrets and int(hygiene_noise_with_secrets["status"]) > 0:
                summary["key_findings"].append(
                    f"Hygiene scan flagged {hygiene_noise_with_secrets['status']} rows with possible secrets - review before sharing/export"
                )

    if not embed_ok:
        summary["key_findings"].append("fastembed not available - install with: pip install mnemosyne-memory[embeddings]")
        summary["fixable"].append("fastembed")
    if not vec_ok:
        summary["key_findings"].append("sqlite-vec not available - install with: pip install sqlite-vec")
        summary["fixable"].append("sqlite_vec")
    if embed_ok and vec_ok and ep_vec and ep_vec["status"] == "0":
        summary["key_findings"].append(
            "Both fastembed and sqlite-vec are available but episodic vectors=0 - "
            "memories may not have been consolidated yet. Use the mnemosyne_sleep "
            "tool or call BeamMemory.sleep()."
        )
    if embed_ok and vec_ok and ep_vec and int(ep_vec["status"]) > 0:
        vtype = ep_vec_type["status"] if ep_vec_type else "unknown"
        msg = f"Semantic search is active with {ep_vec['status']} vectors in episodic memory (backend: {vtype})"
        if vtype in ("binary", "json"):
            # vec_episodes (the sqlite-vec ANN table) is absent -- usually
            # because this Python's sqlite3 can't load the sqlite-vec
            # extension. Recall still works via the binary/JSON fallback;
            # the ANN index only matters at much larger scale.
            msg += " - the sqlite-vec ANN index is not in use (extension not loadable); the fallback is fine at small/medium scale"
        summary["key_findings"].append(msg)

    if vec_working_repair_status:
        inserted = vec_working_repair_inserted["status"] if vec_working_repair_inserted else "0"
        action = "would insert" if dry_run else "inserted"
        summary["key_findings"].append(
            f"vec_working repair {vec_working_repair_status['status']}: {action} {inserted} rows"
        )
    if vec_working_status:
        missing = int(vec_working_missing["status"]) if vec_working_missing else 0
        rows = vec_working_rows["status"] if vec_working_rows else "0"
        fallback_rows = working_embedding_rows["status"] if working_embedding_rows else "0"
        if vec_working_status["status"] == "complete":
            summary["key_findings"].append(
                f"Working-memory sqlite-vec coverage complete: vec_working rows={rows}, fallback embeddings={fallback_rows}"
            )
        elif missing > 0:
            summary["key_findings"].append(
                f"vec_working is missing {missing} backfillable working-memory vectors - run: mnemosyne diagnose --repair-vec-working"
            )
        elif vec_working_status["status"] == "fallback_only":
            summary["key_findings"].append(
                "Working-memory vector recall is using memory_embeddings fallback; sqlite-vec vec_working is unavailable"
            )

    return summary


def _read_only_doctor_status(payload: dict) -> str:
    """Collapse the safe doctor payload into one fixed health status enum."""

    if not isinstance(payload, dict) or not payload:
        return "unavailable"
    sqlite_health = payload.get("sqlite_health")
    if isinstance(sqlite_health, dict) and sqlite_health.get("status") == "unavailable":
        return "unavailable"
    recovery = payload.get("recovery_integrity")
    recovery_status = recovery.get("status") if isinstance(recovery, dict) else None
    severities = {
        finding.get("severity")
        for finding in payload.get("findings", [])
        if isinstance(finding, dict)
    }
    if severities & {"critical", "error"} or recovery_status == "error":
        return "error"
    if severities & {"warning"} or recovery_status == "warning":
        return "warning"
    return "ok"


def _read_only_key_findings(payload: dict) -> list[str]:
    """Surface only the fixed, content-free doctor finding messages."""

    if not isinstance(payload, dict):
        return []
    return [
        str(finding.get("message"))
        for finding in payload.get("findings", [])
        if isinstance(finding, dict)
        and finding.get("severity") in {"warning", "error", "critical"}
    ]


def _log_doctor_metrics(payload: dict, log) -> None:
    """Flatten bounded doctor metrics into the existing entries contract."""

    sqlite = payload.get("sqlite_health") or {}
    quick = sqlite.get("quick_check") or {}
    fk = sqlite.get("foreign_key_check") or {}
    log("db", "sqlite_quick_check", "OK" if quick.get("status") == "ok" else quick.get("status", "UNKNOWN"))
    log("db", "foreign_key_check", "OK" if fk.get("status") == "ok" else str(fk.get("status", "UNKNOWN")))

    ingest = payload.get("ingest_receipts") or {}
    if ingest.get("status") in {"checked", "scan_limited"}:
        log("db", "ingest_pending_or_failed", str(ingest.get("pending_or_failed", 0)))
        log("db", "ingest_conflicts", str(ingest.get("conflicts", 0)))
        log("db", "ingest_stale_receipt_claims", str(ingest.get("stale_receipt_claims", 0)))
    else:
        log("db", "ingest_receipts", str(ingest.get("status", "unavailable")))

    sleep_claims = payload.get("sleep_claims") or {}
    if sleep_claims.get("status") in {"checked", "scan_limited"}:
        log("db", "sleep_stale_claims", str(sleep_claims.get("stale_orphan_claim_candidates", 0)))
    else:
        log("db", "sleep_claims", str(sleep_claims.get("status", "unavailable")))

    dream = payload.get("dream_health") or {}
    if dream.get("status") in {"checked", "scan_limited"}:
        log("db", "dream_non_terminal_runs", str(dream.get("non_terminal_runs", 0)))
    else:
        log("db", "dream_health", str(dream.get("status", "unavailable")))

    proposals = payload.get("proposal_containment") or {}
    if proposals.get("status") in {"checked", "scan_limited"}:
        log("db", "proposal_pending", str(proposals.get("pending_proposals", 0)))
        log("db", "proposal_leakage_candidates", str(proposals.get("leakage_candidates", 0)))
    else:
        log("db", "proposal_containment", str(proposals.get("status", "unavailable")))

    vector_coverage = payload.get("vector_coverage") or {}
    working = vector_coverage.get("working") or {}
    episodic = vector_coverage.get("episodic") or {}
    log("db", "working_total", str(working.get("active_source_rows", 0)))
    log("db", "vec_working_status", str(working.get("status", "unavailable")))
    log("db", "episodic_total", str(episodic.get("source_rows", 0)))
    log("db", "episodic_vectors", str(episodic.get("binary_vector_rows", 0)))
    log("db", "doctor_status", _read_only_doctor_status(payload))


def _run_read_only_diagnostics(
    *, repair_vec_working: bool = False, bank: str | None = None
) -> dict:
    """Non-writing diagnostics: no log file, no default DB, no repair, no writes.

    The returned summary keeps the existing ``run_diagnostics`` shape
    (entries/checks/key_findings) plus ``read_only``, ``repair_rejected``,
    ``log_path=None``, and a structured ``doctor`` payload for the MCP surface.
    """

    entries: list[dict] = []
    resolved_bank = (bank or "default").strip() or "default"
    resolved_db: str | None = None
    payload: dict = {}

    def log(category: str, check: str, status: str, detail: str = ""):
        entry = {
            "ts": datetime.now().isoformat(),
            "category": category,
            "check": check,
            "status": status,
            "detail": detail,
        }
        entries.append(entry)
        return entry

    # Pure runtime/dependency/capability checks (no provider construction).
    for check in collect_runtime_diagnostics()["checks"]:
        log(check["category"], check["check"], check["status"], check["detail"])

    try:
        from mnemosyne.core.banks import get_bank_db_path_read_only
        from mnemosyne.doctor import build_doctor_report, doctor_report_payload

        db_path = get_bank_db_path_read_only(resolved_bank)
        resolved_db = str(db_path)
        report = build_doctor_report(resolved_bank, db_path)
        payload = doctor_report_payload(report)
        _log_doctor_metrics(payload, log)
    except (FileNotFoundError, ValueError, OSError):
        log("db", "doctor_status", "unavailable", "database unavailable")
    except sqlite3.Error:
        log("db", "sqlite_quick_check", "ERROR", "sqlite error")

    if repair_vec_working:
        log(
            "db",
            "vec_working_repair_status",
            "rejected_read_only",
            "repair is not available in read-only diagnostics",
        )

    non_failure_statuses = ("OK", "YES", "set", "OPTIONAL")
    summary = {
        "log_path": None,
        "read_only": True,
        "repair_rejected": bool(repair_vec_working),
        "checks_total": len(entries),
        "checks_passed": sum(1 for e in entries if e["status"] in non_failure_statuses),
        "checks_failed": sum(
            1
            for e in entries
            if str(e["status"]).upper() in ("MISSING", "NO", "ERROR")
        ),
        "key_findings": _read_only_key_findings(payload),
        "fixable": [],
        "entries": entries,
        "resolved_bank": resolved_bank,
        "resolved_db": resolved_db,
        "doctor": payload,
    }
    return summary


def auto_fix(entries: list[dict] | None = None, dry_run: bool = False) -> dict:
    """
    Auto-install missing dependencies detected by diagnostics.

    Args:
        entries: Optional list of diagnostic entries. If None, runs diagnostics first.
        dry_run: If True, report what would be fixed without installing.

    Returns:
        Dict with 'fixed', 'failed', 'skipped' lists and 'ran' bool.
    """
    if entries is None:
        summary = run_diagnostics()
        entries = summary.get("entries", [])

    result = {"fixed": [], "failed": [], "skipped": [], "ran": True}

    for fix_key, fix_info in FIX_MAP.items():
        # Check if this dependency is MISSING
        is_missing = any(fix_info["check"](e) for e in entries)
        if not is_missing:
            continue

        label = fix_info["label"]
        cmd = fix_info["install"]

        if dry_run:
            result["fixed"].append(f"WOULD install: {label} ({' '.join(cmd)})")
            continue

        print(f"🔧 Installing {label}...")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            result["fixed"].append(label)
            print(f"   ✅ {label} installed")
        except subprocess.CalledProcessError as e:
            result["failed"].append({"label": label, "error": e.stderr.strip()})
            print(f"   ❌ Failed: {e.stderr.strip()[:200]}")
        except FileNotFoundError:
            result["failed"].append({"label": label, "error": "pip not found"})
            print("   ❌ pip not found in PATH")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mnemosyne diagnostics")
    parser.add_argument("--fix", action="store_true", help="Auto-install missing dependencies")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fixed/repaired without writing")
    parser.add_argument("--repair-vec-working", action="store_true", help="Backfill missing vec_working rows from memory_embeddings")
    parser.add_argument("--bank", type=str, default=None, help="Mnemosyme bank to diagnose (default: profile-root DB)")
    args = parser.parse_args()

    result = run_diagnostics(repair_vec_working=args.repair_vec_working, dry_run=args.dry_run, bank=args.bank)
    print(json.dumps(result, indent=2))

    if args.fix or (args.dry_run and not args.repair_vec_working):
        fix_result = auto_fix(result.get("entries", []), dry_run=args.dry_run)
        print("\n--- Auto-fix ---")
        print(json.dumps(fix_result, indent=2))
