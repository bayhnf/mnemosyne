"""
Task 16: Contain all public Dream / CLI / MCP unexpected failures.

Every public Dream API must catch unexpected Exception and return a static
``DreamRun`` (``failed_terminal`` / ``integrity_failure``) without leaking the
exception text. A busy/locked ``sqlite3.OperationalError`` is the only
retryable path. The Dream CLI must not leak a traceback, and the MCP
``tools/call`` envelope must serialize only the static ``tool_call_failed``
message. A distinctive non-secret CANARY is injected into every failure
path; it must never reach any returned result, persisted row, or output.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timezone

import pytest

from mnemosyne.core import config as config_module
from mnemosyne.core import dream
from mnemosyne.core.beam import BeamMemory

CANARY = "ZXCV_TASK16_CANARY_BOUNDARY_5130"


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    config_module.MnemosyneConfig.reset_instance()
    yield
    config_module.MnemosyneConfig.reset_instance()


@pytest.fixture
def beam(tmp_path):
    b = BeamMemory(session_id="boundary-sess", db_path=tmp_path / "boundary.db")
    dream._init_dream_schema(b.conn)
    # Seed a healthy dream_runs row so status/apply/undo have a target.
    now = datetime.now(timezone.utc).isoformat()
    b.conn.execute(
        "INSERT INTO dream_runs "
        "(run_id, request_id, state, scope_json, manifest_json, "
        "manifest_hash, semantic_hash, checkpoint, error_code, "
        "failure_reason, enrichment_pending, created_at, updated_at) "
        "VALUES (?, NULL, 'awaiting_approval', '{}', '{}', 'mh-boundary', "
        "'', '', NULL, NULL, 0, ?, ?)",
        ("boundary-run", now, now),
    )
    b.conn.commit()
    return b


def _pass_receipt(run_id, manifest_hash):
    return {
        "role": "reviewer",
        "actor_id": "boundary-reviewer",
        "run_id": run_id,
        "manifest_hash": manifest_hash,
        "verdict": "PASS",
        "reason_code": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Core API: dream_status
# ---------------------------------------------------------------------------


def test_status_corrupt_json_is_static_and_does_not_mutate_run(beam, monkeypatch):
    """A corrupt manifest_json row is terminal-integrity and never rewritten."""
    beam.conn.execute(
        "UPDATE dream_runs SET manifest_json = ? WHERE run_id = ?",
        ("{this is not valid json " + CANARY, "boundary-run"),
    )
    beam.conn.commit()

    before = tuple(
        beam.conn.execute(
            "SELECT state, error_code, failure_reason FROM dream_runs WHERE run_id = ?",
            ("boundary-run",),
        ).fetchone()
    )

    run = dream.dream_status(beam, "boundary-run")

    after = tuple(
        beam.conn.execute(
            "SELECT state, error_code, failure_reason FROM dream_runs WHERE run_id = ?",
            ("boundary-run",),
        ).fetchone()
    )

    assert (run.state, run.error_code, run.failure_reason) == (
        "failed_terminal",
        "integrity_failure",
        "integrity_failure",
    )
    assert before == after
    assert CANARY not in json.dumps(asdict(run), default=str)


def test_status_locked_is_retryable_and_does_not_mutate_run(beam, monkeypatch):
    """A busy/locked OperationalError during _load_run is retryable, static."""

    def _raising_load_run(beam_arg, run_id):
        # Force the SELECT to raise a locked error so the wrapper classifies
        # it as database_busy without touching the durable row.
        raise __import__("sqlite3").OperationalError("database is locked " + CANARY)

    monkeypatch.setattr(dream, "_load_run", _raising_load_run)

    before = tuple(
        beam.conn.execute(
            "SELECT state, error_code, failure_reason FROM dream_runs WHERE run_id = ?",
            ("boundary-run",),
        ).fetchone()
    )

    run = dream.dream_status(beam, "boundary-run")

    after = tuple(
        beam.conn.execute(
            "SELECT state, error_code, failure_reason FROM dream_runs WHERE run_id = ?",
            ("boundary-run",),
        ).fetchone()
    )

    assert (run.state, run.error_code, run.failure_reason) == (
        "failed_retryable",
        "database_busy",
        "database_busy",
    )
    assert before == after
    assert CANARY not in json.dumps(asdict(run), default=str)


# ---------------------------------------------------------------------------
# Core API: dream_submit_receipt
# ---------------------------------------------------------------------------


def test_submit_receipt_rolls_back_partial_insert(beam, monkeypatch):
    """A mid-submit failure rolls back the receipt insert (no partial row)."""
    valid_receipt = _pass_receipt("boundary-run", "mh-boundary")
    original_execute = beam.conn.execute
    receipt_canary = CANARY + "_RECEIPT"

    def _flaky_execute(sql, *params):
        if isinstance(sql, str) and sql.startswith("INSERT INTO dream_receipts"):
            original_execute(sql, *params)
            raise RuntimeError(receipt_canary)
        return original_execute(sql, *params)

    monkeypatch.setattr(beam.conn, "execute", _flaky_execute)

    run = dream.dream_submit_receipt(beam, "boundary-run", valid_receipt)

    assert run.error_code == "integrity_failure"
    assert (
        beam.conn.execute(
            "SELECT COUNT(*) FROM dream_receipts WHERE run_id = ?",
            ("boundary-run",),
        ).fetchone()[0]
        == 0
    )
    assert CANARY not in json.dumps(asdict(run), default=str)


# ---------------------------------------------------------------------------
# Core API: dream_plan
# ---------------------------------------------------------------------------


def test_plan_unexpected_failure_has_a_real_run_id(beam, monkeypatch):
    """A pre-persistence plan failure returns a real UUID + static terminal."""

    def _raising_gather_actions(beam_arg, scope):
        raise RuntimeError(CANARY + "_PLAN")

    monkeypatch.setattr(dream, "_gather_actions", _raising_gather_actions)

    run = dream.dream_plan(beam, {"session_id": "boundary-session"})

    uuid.UUID(run.run_id)
    assert (run.state, run.error_code, run.failure_reason) == (
        "failed_terminal",
        "integrity_failure",
        "integrity_failure",
    )
    assert CANARY not in json.dumps(asdict(run), default=str)


def test_plan_transaction_rejection_has_a_real_run_id(beam):
    """Caller-held transaction rejection still carries a real UUID."""
    beam.conn.execute("BEGIN")
    try:
        run = dream.dream_plan(beam, {"session_id": "boundary-session"})
        uuid.UUID(run.run_id)
        assert run.error_code == "validation_failed"
    finally:
        beam.conn.rollback()


# ---------------------------------------------------------------------------
# Core APIs: dream_apply / dream_undo preflight failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("operation", [dream.dream_apply, dream.dream_undo])
def test_apply_and_undo_preflight_failures_are_contained(beam, monkeypatch, operation):
    """A schema-init failure before the body is contained as integrity_failure."""

    def _raising_init(conn):
        raise RuntimeError(CANARY + "_INIT")

    monkeypatch.setattr(dream, "_init_dream_schema", _raising_init)

    run = operation(beam, "boundary-run")

    assert run.error_code == "integrity_failure"
    assert CANARY not in json.dumps(asdict(run), default=str)


# ---------------------------------------------------------------------------
# Core API: dream_resume
# ---------------------------------------------------------------------------


def test_resume_update_failure_is_static_terminal(beam, monkeypatch):
    """An unexpected failure during resume's status/apply chain is contained."""

    def _raising_load_run(beam_arg, run_id):
        raise RuntimeError(CANARY + "_RESUME")

    monkeypatch.setattr(dream, "_load_run", _raising_load_run)

    run = dream.dream_resume(beam, "boundary-run")

    assert (run.state, run.error_code, run.failure_reason) == (
        "failed_terminal",
        "integrity_failure",
        "integrity_failure",
    )
    assert CANARY not in json.dumps(asdict(run), default=str)


# ---------------------------------------------------------------------------
# CLI boundary: memory construction + projection failures
# ---------------------------------------------------------------------------


def _run_cli_script(script_body, tmp_path):
    """Run an inline python script that invokes cmd_dream in-process.

    Returns (returncode, stdout, stderr). The script monkeypatches an internal
    seam to raise RuntimeError(CANARY) and then calls cmd_dream directly.
    """
    env = os.environ.copy()
    env["MNEMOSYNE_DATA_DIR"] = str(tmp_path / "mnemosyne-data")
    env["MNEMOSYNE_NO_EMBEDDINGS"] = "1"
    env.pop("MNEMOSYNE_BANK", None)
    script = (
        "import sys\n"
        "from mnemosyne.cli import cmd_dream\n"
        f"{script_body}\n"
        "try:\n"
        "    cmd_dream(['status', '--run-id', 'boundary-run', '--json'])\n"
        "except SystemExit as exc:\n"
        "    sys.exit(exc.code if isinstance(exc.code, int) else 1)\n"
        "sys.exit(0)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_cli_memory_construction_failure_is_static(tmp_path):
    """_get_memory raising -> exit 1, static projection, no canary/traceback."""
    script_body = (
        "import mnemosyne.cli as _cli\n"
        "def _broken_get_memory():\n"
        f"    raise RuntimeError('{CANARY}_MEM')\n"
        "_cli._get_memory = _broken_get_memory\n"
    )
    rc, out, err = _run_cli_script(script_body, tmp_path)
    combined = out + err

    assert rc == 1
    assert CANARY not in combined
    assert "Traceback" not in combined
    # The JSON projection key must be present with the static error code.
    assert '"error_code": "integrity_failure"' in out
    assert '"state": "failed_terminal"' in out


def test_cli_projection_failure_is_static(tmp_path):
    """_dream_run_projection raising after a healthy status -> static failure."""
    script_body = (
        "import mnemosyne.cli as _cli\n"
        "from mnemosyne.core.beam import BeamMemory\n"
        "from mnemosyne.core import dream\n"
        "import tempfile\n"
        "_beam = BeamMemory(session_id='s', db_path=tempfile.mktemp(suffix='.db'))\n"
        "dream._init_dream_schema(_beam.conn)\n"
        "from datetime import datetime, timezone\n"
        "now = datetime.now(timezone.utc).isoformat()\n"
        "_beam.conn.execute(\n"
        '    "INSERT INTO dream_runs (run_id, request_id, state, scope_json, '
        "manifest_json, manifest_hash, semantic_hash, checkpoint, error_code, "
        "failure_reason, enrichment_pending, created_at, updated_at) "
        "VALUES (?, NULL, 'awaiting_approval', '{}', '{}', 'mh', '', '', "
        'NULL, NULL, 0, ?, ?)",\n'
        "    ('boundary-run', now, now))\n"
        "_beam.conn.commit()\n"
        "def _healthy_get_memory():\n"
        "    return _beam\n"
        "_cli._get_memory = _healthy_get_memory\n"
        "def _broken_projection(run):\n"
        f"    raise RuntimeError('{CANARY}_PROJ')\n"
        "_cli._dream_run_projection = _broken_projection\n"
    )
    rc, out, err = _run_cli_script(script_body, tmp_path)
    combined = out + err

    assert rc == 1
    assert CANARY not in combined
    assert "Traceback" not in combined
    assert '"error_code": "integrity_failure"' in out
    assert '"state": "failed_terminal"' in out


# ---------------------------------------------------------------------------
# MCP boundary: tools/call error envelope must be static
# ---------------------------------------------------------------------------


def test_mcp_call_tool_error_envelope_is_static(monkeypatch):
    """handle_tool_call raising -> CallToolResult(is_error=True) with static msg."""
    pytest.importorskip("mcp.types")
    import asyncio
    from unittest.mock import patch

    from mnemosyne.mcp_server import _build_mcp_server

    server = _build_mcp_server()
    entry = server.get_request_handler("tools/call")
    on_call_tool = entry.handler

    class _Params:
        name = "mnemosyne_stats"
        arguments = {}

    with patch(
        "mnemosyne.mcp_server.handle_tool_call",
        side_effect=RuntimeError(CANARY + "_MCP"),
    ):
        result = asyncio.run(on_call_tool(ctx=None, params=_Params()))

    assert result.is_error is True
    assert json.loads(result.content[0].text) == {
        "status": "error",
        "message": "tool_call_failed",
    }
    assert CANARY not in result.content[0].text
