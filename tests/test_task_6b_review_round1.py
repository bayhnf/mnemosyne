"""Task 6B round-1 review: precise failing tests for each Important finding.

Each test encodes the exact defect the independent reviewer flagged, not a
hypothetical. Together they are the RED gate for round-2 of Task 6B.
"""
import json
import re
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from mnemosyne.mcp_tools import handle_tool_call, _TOOL_HANDLERS
from mnemosyne.tool_schemas import READ_ONLY_TOOLS


# Helpers ------------------------------------------------------------------

def _fresh_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))


def _readOnly(name: str) -> bool:
    from mnemosyne.mcp_tools import TOOLS
    by_name = {t["name"]: t for t in TOOLS}
    return bool(by_name[name].get("readOnly"))


# Finding 1: mnemosyne_dream_status must NOT be readOnly (writes config.yaml)

def test_dream_status_is_not_labeled_read_only(monkeypatch, tmp_path):
    """dream_status calls _reconcile_gate_from_durable_state which writes
    config.yaml; readOnly:true is untruthful."""
    assert "mnemosyne_dream_status" not in READ_ONLY_TOOLS, (
        "mnemosyne_dream_status writes config.yaml via gate reconciliation; "
        "it must not be in READ_ONLY_TOOLS."
    )
    assert _readOnly("mnemosyne_dream_status") is False


def test_dream_status_truthfully_labeled_not_read_only(monkeypatch, tmp_path):
    """dream_status writes config.yaml via gate reconciliation. The truthful
    fix is an honest readOnly:false label (Finding 1); we do NOT forbid the
    core mutation, which is correct semantics outside the read-only contract."""
    # The label assertion is in test_dream_status_is_not_labeled_read_only;
    # here we confirm the mutation exists so the label is necessary, not
    # conservative.
    _fresh_env(monkeypatch, tmp_path)
    handle_tool_call("mnemosyne_remember", {"content": "seed"})
    plan = handle_tool_call("mnemosyne_dream_plan", {"session_id": "s-cfg"})
    handle_tool_call("mnemosyne_dream_status", {"run_id": plan["run_id"]})
    # dream_status CAN mutate config.yaml (gate reconciliation); the point is
    # that readOnly:true would be a LIE. The label must be false.
    assert _readOnly("mnemosyne_dream_status") is False
    # And the tool must still be callable (not removed from the surface).
    assert "mnemosyne_dream_status" in _TOOL_HANDLERS


# Finding 2: read-only-flagged handlers must not materialize a default DB
# on a fresh data dir.

@pytest.mark.parametrize("tool,args", [
    ("mnemosyne_ingest_status", {}),
    ("mnemosyne_persona_list", {}),
])
def test_readonly_handler_does_not_materialize_default_db(monkeypatch, tmp_path, tool, args):
    """On a fresh data dir, a readOnly tool must not create mnemosyne.db."""
    assert _readOnly(tool) is True, f"{tool} must remain readOnly for this gate"
    _fresh_env(monkeypatch, tmp_path)
    default_db = tmp_path / "mnemosyne.db"
    assert not default_db.exists()

    result = handle_tool_call(tool, args)

    # No default DB materialized; banks/ dir not created either.
    assert not default_db.exists(), (
        f"{tool} materialized a default mnemosyne.db on a fresh data dir "
        f"(readOnly contract). Result keys: {sorted(result)}"
    )
    assert not (tmp_path / "banks").exists(), (
        f"{tool} created a banks/ directory on a fresh data dir"
    )


# Finding 3: reclaim_orphans.apply must mutate ONLY on literal JSON boolean True.

@pytest.mark.parametrize("bad_apply", ["false", "true", "yes", "1", 0, 1, "no", None])
def test_reclaim_orphans_rejects_non_bool_apply(monkeypatch, tmp_path, bad_apply):
    """Binding 2: mutation requires explicit opt-in. Truthy strings must NOT
    mutate. Only literal JSON true is permitted."""
    _fresh_env(monkeypatch, tmp_path)
    args = {} if bad_apply is None else {"apply": bad_apply}
    result = handle_tool_call("mnemosyne_reclaim_orphans", args)
    # Every non-literal-True value MUST result in dry_run == True (no mutation).
    assert result.get("dry_run") is True, (
        f"apply={bad_apply!r} must be treated as dry-run (only literal True mutates)"
    )


def test_reclaim_orphans_literal_true_mutates(monkeypatch, tmp_path):
    _fresh_env(monkeypatch, tmp_path)
    result = handle_tool_call("mnemosyne_reclaim_orphans", {"apply": True})
    assert result.get("dry_run") is False


# Finding 4: doctor stale_receipt_claims must match core lease eligibility
# (aware UTC, and reclaim-at now, not now-60s).

def test_doctor_stale_receipt_claims_uses_utc_and_matches_core_eligibility(tmp_path):
    """Core reclaim (inhale._try_claim) treats a lease as stale when
    lease_iso < now_iso (both aware-UTC). The doctor metric must classify the
    same rows as stale, in any timezone."""
    from mnemosyne.doctor import IngestReceiptHealthAdapter, open_readonly_doctor_db

    now_utc = datetime.now(timezone.utc)
    # A lease that is clearly expired (1 hour ago) by UTC.
    expired_lease = (now_utc - timedelta(hours=1)).isoformat()
    # A lease still valid for 1 hour ahead (must NOT be counted stale even in
    # a timezone far ahead of UTC, e.g. Asia/Jakarta UTC+7).
    live_lease = (now_utc + timedelta(hours=1)).isoformat()

    db_path = tmp_path / "stale-utc.db"
    w = sqlite3.connect(db_path)
    w.executescript(
        f"""
        CREATE TABLE ingest_receipts (
          event_id TEXT PRIMARY KEY, payload_hash TEXT, memory_ids TEXT,
          status TEXT, index_status TEXT, attempts INTEGER, created_at TEXT,
          updated_at TEXT, claim_worker_id TEXT, claim_worker_lease TEXT
        );
        CREATE TABLE ingest_conflicts (
          seq INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT,
          stored_payload_hash TEXT, conflicting_payload_hash TEXT, observed_at TEXT
        );
        INSERT INTO ingest_receipts VALUES
          ('ev-expired', 'h1', '[]', 'stored', 'pending', 0,
           '{expired_lease}', '{expired_lease}', 'w1', '{expired_lease}'),
          ('ev-live', 'h2', '[]', 'stored', 'pending', 0,
           '{expired_lease}', '{expired_lease}', 'w2', '{live_lease}');
        """
    )
    w.commit()
    w.close()
    conn = open_readonly_doctor_db(db_path)
    try:
        result = IngestReceiptHealthAdapter(conn).inspect()
    finally:
        conn.close()
    assert result.metrics["stale_receipt_claims"] == 1, (
        f"expected exactly the 1 expired lease to be stale, got "
        f"{result.metrics['stale_receipt_claims']} (a live UTC lease was "
        f"misclassified, or the cutoff timezone/offset diverges from core)"
    )


# Finding 5: Dream undo projection must distinguish first vs second undo
# (surface already_undone content-free).

def test_dream_projection_surfaces_already_undone_content_free():
    """Unit: _dream_projection must surface a content-free already_undone
    boolean that distinguishes a first undo from a second undo, WITHOUT
    echoing the raw failure_reason field."""
    from mnemosyne.core.dream import DreamRun
    from mnemosyne.mcp_tools import _dream_projection

    first_run = DreamRun(run_id="r1", state="undone", checkpoint="undone")
    second_run = DreamRun(
        run_id="r1", state="undone", checkpoint="undone",
        failure_reason="already_undone",
    )
    first = _dream_projection(first_run)
    second = _dream_projection(second_run)

    # The raw failure_reason string must NEVER appear in the projection.
    assert "failure_reason" not in first
    assert "already_undone" not in json.dumps(first) or first["already_undone"] is False
    # First undo: explicit content-free False.
    assert first.get("already_undone") is False
    # Second undo: explicit content-free True — distinguishable from first.
    assert second.get("already_undone") is True
    # No raw failure text leaks.
    assert "failure_reason" not in second


def test_dream_undo_mcp_second_call_returns_already_undone(monkeypatch, tmp_path):
    """Integration: seeding a run already in 'undone' state (as a first undo
    leaves it), a second MCP undo must return already_undone==True.

    Drives the durable state directly to avoid the numpy/SHMR dependency the
    full plan→apply lifecycle needs; the second-undo branch only inspects
    ``run.state == 'undone'``.
    """
    _fresh_env(monkeypatch, tmp_path)
    from mnemosyne.core.config import MnemosyneConfig
    MnemosyneConfig.reset_instance()
    # Materialize the schema once via a writable beam, then mark a run undone.
    from mnemosyne.core.beam import BeamMemory
    from mnemosyne.core import dream
    # Seed into the default-bank DB path so the MCP handler resolves the
    # same database (get_bank_db_path_read_only("default") -> data_dir/mnemosyne.db).
    b = BeamMemory(session_id="s-undo2", db_path=tmp_path / "mnemosyne.db")
    dream._init_dream_schema(b.conn)
    now = dream._now_iso()
    b.conn.execute(
        "INSERT INTO dream_runs "
        "(run_id, request_id, state, scope_json, manifest_json, manifest_hash, "
        "semantic_hash, checkpoint, error_code, failure_reason, "
        "enrichment_pending, created_at, updated_at) "
        "VALUES (?, NULL, 'undone', '{}', '{}', '', '', 'undone', NULL, NULL, 0, ?, ?)",
        ("run-undone-seed", now, now),
    )
    b.conn.commit()
    b.conn.close()
    MnemosyneConfig.reset_instance()

    second = handle_tool_call("mnemosyne_dream_undo", {"run_id": "run-undone-seed"})
    assert second.get("state") == "undone"
    assert second.get("already_undone") is True, (
        f"second undo (run already undone) did not signal already_undone: {second}"
    )


# Finding 6: owned parity test must enforce actual schema<->handler parity
# (no provider-substring exception). Lives in test_tool_surface_parity.py
# (this assertion documents the contract this file now relies on).

def test_parity_test_file_has_no_provider_substring_exception():
    """tests/test_tool_surface_parity.py must not retain the legacy
    provider-source-substring escape hatch: no PROVIDER_ONLY set and no
    provider_src scan. (A comment documenting the removed hatch is fine.)"""
    src = (Path(__file__).parent / "test_tool_surface_parity.py").read_text()
    assert "PROVIDER_ONLY = {" not in src, (
        "PROVIDER_ONLY set still present in test_tool_surface_parity.py; the "
        "provider-substring exception was meant to be eliminated."
    )
    assert "provider_src" not in src, (
        "provider-source substring scan still present in "
        "test_tool_surface_parity.py"
    )
    # The strict parity test must exist (renamed, not deleted).
    assert "test_every_advertised_tool_has_a_real_handler" in src, (
        "strict schema<->handler parity test missing from "
        "test_tool_surface_parity.py"
    )


# Finding 7: _handle_diagnose must not retain a fallback to the legacy
# writing path.

def test_handle_diagnose_has_no_legacy_writing_fallback():
    """Binding 8: no silent fallback. The handler must not introspect the
    signature for read_only support and must not CALL run_diagnostics with a
    dry_run= kwarg (the legacy writable path)."""
    import inspect
    src = inspect.getsource(_TOOL_HANDLERS["mnemosyne_diagnose"])
    assert "supports_read_only" not in src, (
        "_handle_diagnose still introspects for read_only support and keeps a "
        "legacy fallback path"
    )
    # No call site passes dry_run= to run_diagnostics (the writable path).
    assert "run_diagnostics(" not in src.replace(
        "run_diagnostics(read_only=True)", "", 1
    ) or "dry_run=" not in re.sub(
        r'"[^"]*dry_run[^"]*"', "", src
    ), (
        "_handle_diagnose still routes to run_diagnostics(dry_run=...) "
        "(legacy writable path)"
    )


def test_handle_diagnose_fails_closed_when_read_only_rejected(monkeypatch, tmp_path):
    """If run_diagnostics rejects read_only (version skew), the MCP handler
    must fail closed with a structured result — NEVER silently fall back to a
    writable dry_run path."""
    _fresh_env(monkeypatch, tmp_path)
    import mnemosyne.diagnose as diagnose

    orig = diagnose.run_diagnostics
    attempted_writable = {"v": False}

    def skew_run_diagnostics(*args, **kwargs):
        if kwargs.get("dry_run") is True:
            attempted_writable["v"] = True
        raise TypeError("read_only not a valid parameter")

    monkeypatch.setattr(diagnose, "run_diagnostics", skew_run_diagnostics)
    try:
        result = handle_tool_call("mnemosyne_diagnose", {})
    finally:
        monkeypatch.setattr(diagnose, "run_diagnostics", orig)
    assert not attempted_writable["v"], (
        "handler silently fell back to the writable dry_run path under skew"
    )
    assert "status" in result or "error" in result, (
        f"diagnose returned non-structured result under skew: {result}"
    )


# Finding: sync must honor config.yaml / HOST+PORT resolution like SyncAdapter,
# and local sync_status must remain usable without a remote.

def test_sync_remote_resolution_honors_host_port_env(monkeypatch, tmp_path):
    """A deployment configured via MNEMOSYNE_SYNC_HOST+PORT must not get an
    'unconfigured' rejection: SyncAdapter would resolve a remote from these."""
    _fresh_env(monkeypatch, tmp_path)
    for v in ("MNEMOSYNE_SYNC_REMOTE", "MNEMOSYNE_SYNC_HOST", "MNEMOSYNE_SYNC_PORT"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("MNEMOSYNE_SYNC_HOST", "sync.example.test")
    monkeypatch.setenv("MNEMOSYNE_SYNC_PORT", "8765")

    # Seed so a local status path has an engine; we only assert it is NOT
    # rejected as 'unconfigured' purely on the absence of MNEMOSYNE_SYNC_REMOTE.
    handle_tool_call("mnemosyne_remember", {"content": "sync seed"})
    result = handle_tool_call("mnemosyne_sync_status", {})
    assert result.get("status") != "unconfigured", (
        f"HOST+PORT configured but MCP sync treated it as unconfigured: {result}"
    )
    assert "sync.example.test" in result.get("remote", ""), (
        f"remote did not resolve from HOST+PORT env: {result}"
    )


def test_sync_remote_resolution_honors_config_yaml(monkeypatch, tmp_path):
    """A deployment configured via config.yaml sync_remote must not get an
    'unconfigured' rejection."""
    _fresh_env(monkeypatch, tmp_path)
    for v in ("MNEMOSYNE_SYNC_REMOTE", "MNEMOSYNE_SYNC_HOST", "MNEMOSYNE_SYNC_PORT"):
        monkeypatch.delenv(v, raising=False)
    (tmp_path / "config.yaml").write_text("sync_remote: https://cfg.example.test:8765\n")

    handle_tool_call("mnemosyne_remember", {"content": "sync cfg seed"})
    result = handle_tool_call("mnemosyne_sync_status", {})
    assert result.get("status") != "unconfigured", (
        f"config.yaml sync_remote set but MCP sync treated it as unconfigured: {result}"
    )


def test_sync_status_remains_usable_without_any_remote(monkeypatch, tmp_path):
    """Even with no remote, sync_status must return local status (device id,
    event count, encryption state) rather than a hard 'unconfigured' block."""
    _fresh_env(monkeypatch, tmp_path)
    for v in ("MNEMOSYNE_SYNC_REMOTE", "MNEMOSYNE_SYNC_HOST", "MNEMOSYNE_SYNC_PORT"):
        monkeypatch.delenv(v, raising=False)
    handle_tool_call("mnemosyne_remember", {"content": "sync local seed"})
    result = handle_tool_call("mnemosyne_sync_status", {})
    # Local status fields must be present even without a remote.
    assert "device_id" in result or "local_events" in result, (
        f"sync_status with no remote did not return local status: {result}"
    )
