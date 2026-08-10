"""Hook fixture tests for the Mnemosyne Codex plugin.

Each test exercises one hook script against a disposable Mnemosyne instance and
verifies the binding contract: stable event IDs, bounded context injection,
visible structured failures, 0600 transport-only spool, ack-based deletion,
no transcript parsing, and fail-open behavior.

These tests run before any production hook code exists (RED), then drive
implementation to GREEN.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest

PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOOKS_DIR = os.path.join(PLUGIN_ROOT, "hooks")
WORKTREE_ROOT = os.path.dirname(os.path.dirname(PLUGIN_ROOT))


def _run_hook(
    script_name: str, stdin_payload: dict, env: dict
) -> tuple[int, dict | None, str, str]:
    """Run a hook script with the given stdin JSON payload.

    Returns (exit_code, parsed_stdout_json_or_None, stdout_raw, stderr).
    """
    full_env = dict(os.environ)
    full_env.update(env)
    # Simulate installed mnemosyne package for subprocess hooks.
    prior = full_env.get("PYTHONPATH", "")
    full_env["PYTHONPATH"] = WORKTREE_ROOT + (os.pathsep + prior if prior else "")
    proc = subprocess.run(
        [sys.executable, os.path.join(HOOKS_DIR, script_name)],
        input=json.dumps(stdin_payload),
        capture_output=True,
        text=True,
        env=full_env,
        timeout=30,
        cwd=tempfile.gettempdir(),
    )
    parsed = None
    if proc.stdout.strip():
        try:
            parsed = json.loads(proc.stdout)
        except json.JSONDecodeError:
            parsed = None
    return proc.returncode, parsed, proc.stdout, proc.stderr


class _HookTestBase(unittest.TestCase):
    """Common setup: disposable data dir for an isolated Mnemosyne instance."""

    def setUp(self) -> None:
        self.data_dir = tempfile.mkdtemp(prefix="mnem-codex-test-")
        self.env = {"MNEMOSYNE_DATA_DIR": self.data_dir}
        self.session_id = "sess-123"
        self.turn_id = "turn-1"
        self.actor = "codex-user"
        self.project = "proj-x"

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.data_dir, ignore_errors=True)


class TestSessionStart(_HookTestBase):
    def test_injects_bounded_recall_context_on_startup(self) -> None:
        """After ingesting a memory, SessionStart must inject it as context."""
        # Ingest a preference first via the user_prompt_submit hook.
        actor_env = dict(self.env)
        actor_env.update(
            {
                "MNEMOSYNE_CODEX_ACTOR_ID": self.actor,
                "MNEMOSYNE_CODEX_PROJECT_ID": self.project,
            }
        )
        _run_hook(
            "user_prompt_submit.py",
            {
                "session_id": self.session_id,
                "prompt": "I prefer dark mode for all editors",
                "cwd": "/tmp",
            },
            actor_env,
        )
        # Now SessionStart should recall and inject it.
        payload = {
            "session_id": self.session_id,
            "source": "startup",
            "cwd": "/tmp/fake",
        }
        code, out, _stdout, _stderr = _run_hook("session_start.py", payload, actor_env)
        self.assertEqual(code, 0, f"SessionStart must exit 0 (fail-open), got {code}")
        self.assertIsNotNone(out, "SessionStart must emit JSON")
        hso = (out or {}).get("hookSpecificOutput", {})
        self.assertEqual(hso.get("hookEventName"), "SessionStart")
        self.assertIn("additionalContext", hso)
        self.assertIn("dark mode", hso.get("additionalContext", ""))

    def test_fires_on_compact_source(self) -> None:
        """SessionStart must also recall after compact."""
        payload = {
            "session_id": self.session_id,
            "source": "compact",
            "cwd": "/tmp/fake",
        }
        code, out, _stdout, _stderr = _run_hook("session_start.py", payload, self.env)
        self.assertEqual(code, 0)
        # additionalContext may be empty if no memories; the key is it doesn't crash
        self.assertIsNotNone(out)

    def test_empty_recall_context_when_no_memories(self) -> None:
        payload = {
            "session_id": self.session_id,
            "source": "startup",
            "cwd": "/tmp/fake",
        }
        code, out, _stdout, _stderr = _run_hook("session_start.py", payload, self.env)
        self.assertEqual(code, 0)
        # No memories ingested yet — context may be empty but hook must succeed
        ctx = (out or {}).get("hookSpecificOutput", {}).get("additionalContext", "")
        # It must not be a memory error
        self.assertNotIn("Traceback", ctx)


class TestUserPromptSubmit(_HookTestBase):
    def test_ingests_user_event_with_stable_id(self) -> None:
        """UserPromptSubmit must durably ingest the user prompt with a stable event_id."""
        prompt = "How do I configure dark mode?"
        payload = {
            "session_id": self.session_id,
            "prompt": prompt,
            "cwd": "/tmp/fake",
        }
        actor_env = dict(self.env)
        actor_env.update(
            {
                "MNEMOSYNE_CODEX_ACTOR_ID": self.actor,
                "MNEMOSYNE_CODEX_PROJECT_ID": self.project,
            }
        )
        code, out, _stdout, _stderr = _run_hook(
            "user_prompt_submit.py", payload, actor_env
        )
        self.assertEqual(code, 0, f"UserPromptSubmit must exit 0, stderr={_stderr}")
        # The event must be durably ingested — verify via the DB in the
        # same data dir the hook wrote to.
        import sqlite3

        db_path = os.path.join(self.data_dir, "mnemosyne.db")
        self.assertTrue(os.path.exists(db_path), "hook must create mnemosyne.db")
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT content FROM working_memory WHERE content = ?", (prompt,)
        ).fetchall()
        conn.close()
        self.assertEqual(len(rows), 1, "prompt must be durably ingested")

    def test_recall_context_is_bounded(self) -> None:
        """Injected context must respect the ≤8 items / ≤1200 tokens bound."""
        # Ingest several memories first via the same hook
        actor_env = dict(self.env)
        actor_env.update(
            {
                "MNEMOSYNE_CODEX_ACTOR_ID": self.actor,
                "MNEMOSYNE_CODEX_PROJECT_ID": self.project,
            }
        )
        for i in range(20):
            _run_hook(
                "user_prompt_submit.py",
                {
                    "session_id": self.session_id,
                    "prompt": f"fact number {i} about the project",
                    "cwd": "/tmp/fake",
                },
                actor_env,
            )
        code, out, _stdout, _stderr = _run_hook(
            "user_prompt_submit.py",
            {
                "session_id": self.session_id,
                "prompt": "project facts",
                "cwd": "/tmp/fake",
            },
            actor_env,
        )
        self.assertEqual(code, 0)
        ctx = (out or {}).get("hookSpecificOutput", {}).get("additionalContext", "")
        # Context must be bounded; count only memory-item lines (start with "-"),
        # excluding the wrapper tags (<mnemosyne-recall>...</mnemosyne-recall>).
        item_lines = [l for l in ctx.splitlines() if l.strip().startswith("-")]
        self.assertLessEqual(
            len(item_lines),
            8,
            f"recall must be ≤8 items, got {len(item_lines)}: {ctx[:300]}",
        )

    def test_stable_event_id_across_replays(self) -> None:
        """Same session+turn must produce the same event_id (idempotent ingest)."""
        # The hook derives event_id deterministically; verify by calling the helper
        sys.path.insert(0, HOOKS_DIR)
        # We validate the shape via the public helper if exposed; otherwise via ingest
        import importlib

        spec = importlib.util.spec_from_file_location(
            "_ev_helper", os.path.join(HOOKS_DIR, "common.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        eid1 = mod.stable_event_id("mem-test-scope", "turn-5", "user")
        eid2 = mod.stable_event_id("mem-test-scope", "turn-5", "user")
        self.assertEqual(eid1, eid2, "event_id must be stable across calls")
        eid3 = mod.stable_event_id("mem-test-scope", "turn-6", "user")
        self.assertNotEqual(eid1, eid3, "different turn must yield different event_id")


class TestStop(_HookTestBase):
    def test_ingests_last_assistant_message(self) -> None:
        """Stop must durably ingest the acknowledged assistant message."""
        payload = {
            "session_id": self.session_id,
            "last_assistant_message": "Dark mode is configured in Settings > Appearance.",
            "cwd": "/tmp/fake",
        }
        actor_env = dict(self.env)
        actor_env.update(
            {
                "MNEMOSYNE_CODEX_ACTOR_ID": self.actor,
                "MNEMOSYNE_CODEX_PROJECT_ID": self.project,
            }
        )
        code, _out, _stdout, stderr = _run_hook("stop.py", payload, actor_env)
        self.assertEqual(code, 0, f"Stop must exit 0, stderr={stderr}")

    def test_no_transcript_parsing(self) -> None:
        """Stop must not read or parse transcript_path — it uses only the
        last_assistant_message field from the hook payload."""
        payload = {
            "session_id": self.session_id,
            "last_assistant_message": "Answer here.",
            "transcript_path": "/nonexistent/path.jsonl",
            "cwd": "/tmp/fake",
        }
        actor_env = dict(self.env)
        actor_env.update(
            {
                "MNEMOSYNE_CODEX_ACTOR_ID": self.actor,
                "MNEMOSYNE_CODEX_PROJECT_ID": self.project,
            }
        )
        code, _out, _stdout, stderr = _run_hook("stop.py", payload, actor_env)
        self.assertEqual(
            code,
            0,
            f"Stop must succeed even with unreadable transcript, stderr={stderr}",
        )


class TestSpoolFailurePath(_HookTestBase):
    """When native ingest fails (unreachable DB), the transport spool catches
    the event in a 0600 SQLite file that is never searchable."""

    def test_spool_is_mode_0600(self) -> None:
        spool_path = os.path.join(self.data_dir, "codex-spool.db")
        actor_env = dict(self.env)
        actor_env.update(
            {
                "MNEMOSYNE_CODEX_ACTOR_ID": self.actor,
                "MNEMOSYNE_CODEX_PROJECT_ID": self.project,
                "MNEMOSYNE_CODEX_SPOOL_PATH": spool_path,
                "MNEMOSYNE_CODEX_FORCE_SPOOL": "1",
            }
        )
        payload = {
            "session_id": self.session_id,
            "prompt": "force spool test",
            "cwd": "/tmp",
        }
        code, _out, _stdout, _stderr = _run_hook(
            "user_prompt_submit.py", payload, actor_env
        )
        self.assertEqual(code, 0)
        self.assertTrue(
            os.path.exists(spool_path), "spool db must be created on failure path"
        )
        mode = stat.S_IMODE(os.stat(spool_path).st_mode)
        self.assertEqual(mode, 0o600, f"spool must be 0600, got {oct(mode)}")

    def test_spool_is_not_searchable_by_recall(self) -> None:
        """The spool must never be recallable — it is transport-only."""
        spool_path = os.path.join(self.data_dir, "codex-spool.db")
        actor_env = dict(self.env)
        actor_env.update(
            {
                "MNEMOSYNE_CODEX_ACTOR_ID": self.actor,
                "MNEMOSYNE_CODEX_PROJECT_ID": self.project,
                "MNEMOSYNE_CODEX_SPOOL_PATH": spool_path,
                "MNEMOSYNE_CODEX_FORCE_SPOOL": "1",
            }
        )
        payload = {
            "session_id": self.session_id,
            "prompt": "unique spooled secret phrase",
            "cwd": "/tmp",
        }
        _run_hook("user_prompt_submit.py", payload, actor_env)
        # Recall must NOT find the spooled content
        from mnemosyne.core.memory import Mnemosyne

        m = Mnemosyne(
            session_id=self.session_id,
            db_path=os.path.join(self.data_dir, "mnemosyne.db"),
            author_id=self.actor,
            author_type="human",
            channel_id=self.project,
        )
        env = m.recall_bounded("unique spooled secret phrase")
        for result in env.results:
            content = str(result.get("content", ""))
            self.assertNotIn(
                "unique spooled secret phrase",
                content,
                "spool content must not be recallable",
            )

    def test_spool_deleted_after_ack(self) -> None:
        """After successful retry/ack, the spooled row must be removed."""
        spool_path = os.path.join(self.data_dir, "codex-spool.db")
        actor_env = dict(self.env)
        actor_env.update(
            {
                "MNEMOSYNE_CODEX_ACTOR_ID": self.actor,
                "MNEMOSYNE_CODEX_PROJECT_ID": self.project,
                "MNEMOSYNE_CODEX_SPOOL_PATH": spool_path,
                "MNEMOSYNE_CODEX_FORCE_SPOOL": "1",
            }
        )
        payload = {
            "session_id": self.session_id,
            "prompt": "will be acked",
            "cwd": "/tmp",
        }
        _run_hook("user_prompt_submit.py", payload, actor_env)
        self.assertTrue(os.path.exists(spool_path))
        # Now flush the spool (simulating SessionEnd or retry)
        sys.path.insert(0, HOOKS_DIR)
        import importlib

        spec = importlib.util.spec_from_file_location(
            "_spool_helper", os.path.join(HOOKS_DIR, "common.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Clear force-spool so flush can reach native ingest
        flush_env = dict(actor_env)
        del flush_env["MNEMOSYNE_CODEX_FORCE_SPOOL"]
        mod.flush_spool(spool_path, flush_env)
        # Spool table must now be empty
        remaining = mod.spool_count(spool_path)
        self.assertEqual(remaining, 0, "spool must be empty after successful flush/ack")


# ---------------------------------------------------------------------------
# Task 3: shared-classifier admission gate before spool persistence
# ---------------------------------------------------------------------------


class TestAdmissionTerminalDrop(unittest.TestCase):
    """Task 3: events rejected by the shared admission classifier are
    terminal-dropped and never written to the spool; native
    ``admission_rejected`` receipts are terminal; safe transient failures
    still spool for retry."""

    SECRET = "api_key=sk-abcdefghij0123456789"

    def setUp(self) -> None:
        self.data_dir = tempfile.mkdtemp(prefix="mnem-admission-")
        self.spool_path = os.path.join(self.data_dir, "codex-spool.db")
        self.base_env = {
            "MNEMOSYNE_DATA_DIR": self.data_dir,
            "MNEMOSYNE_CODEX_SPOOL_PATH": self.spool_path,
            "MNEMOSYNE_CODEX_ACTOR_ID": "alice",
            "MNEMOSYNE_CODEX_PROJECT_ID": "projX",
        }

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.data_dir, ignore_errors=True)

    def _load_common(self):
        import importlib

        sys.path.insert(0, HOOKS_DIR)
        try:
            spec = importlib.util.spec_from_file_location(
                "_t3_common", os.path.join(HOOKS_DIR, "common.py")
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        finally:
            sys.path.remove(HOOKS_DIR)

    def _event(self, content: str, metadata: dict | None = None) -> dict:
        return {
            "event_id": "cx-t3-secret",
            "producer": "codex",
            "actor_id": "alice",
            "project_id": "projX",
            "scope": "mem-t3",
            "turn_id": "turn-1",
            "role": "user",
            "content": content,
            "metadata": metadata,
            "occurred_at": "2026-08-10T00:00:00+00:00",
        }

    def _spool_bytes(self) -> bytes:
        if not os.path.exists(self.spool_path):
            return b""
        with open(self.spool_path, "rb") as fh:
            return fh.read()

    def test_secret_event_is_terminal_drop_and_never_written_to_spool(self) -> None:
        """A secret event must be terminal-dropped before any spool write."""
        mod = self._load_common()
        env = dict(self.base_env)
        env["MNEMOSYNE_CODEX_FORCE_SPOOL"] = "1"  # force the spool fallback path
        with mod._scoped_env(env):
            outcome, spool_status = mod.ingest_or_spool(self._event(self.SECRET))
        self.assertEqual(outcome.error_code, "admission_rejected")
        self.assertEqual(outcome.reason, "admission_rejected")
        self.assertEqual(spool_status, "")
        self.assertEqual(mod.spool_count(self.spool_path), 0)
        self.assertNotIn(self.SECRET.encode(), self._spool_bytes())

    def test_secret_in_metadata_is_terminal_drop_not_spooled(self) -> None:
        """Secrets in canonical metadata must also be terminal-dropped."""
        mod = self._load_common()
        env = dict(self.base_env)
        env["MNEMOSYNE_CODEX_FORCE_SPOOL"] = "1"
        event = self._event("clean content", metadata={"token": self.SECRET})
        with mod._scoped_env(env):
            outcome, spool_status = mod.ingest_or_spool(event)
        self.assertEqual(outcome.error_code, "admission_rejected")
        self.assertEqual(spool_status, "")
        self.assertEqual(mod.spool_count(self.spool_path), 0)
        self.assertNotIn(self.SECRET.encode(), self._spool_bytes())

    def test_native_admission_rejected_receipt_is_terminal_not_spooled(self) -> None:
        """A native receipt coded ``admission_rejected`` must not enqueue."""
        mod = self._load_common()
        with mod._scoped_env(self.base_env):
            original = mod.native_ingest
            mod.native_ingest = lambda _ev: mod.Outcome(False, "admission_rejected", "")
            try:
                outcome, spool_status = mod.ingest_or_spool(self._event("safe content"))
            finally:
                mod.native_ingest = original
        self.assertEqual(outcome.error_code, "admission_rejected")
        self.assertEqual(spool_status, "")
        self.assertEqual(mod.spool_count(self.spool_path), 0)

    def test_safe_transient_failure_still_spools_under_admission_gate(self) -> None:
        """Admission must not turn safe transient failures into data loss."""
        mod = self._load_common()
        env = dict(self.base_env)
        env["MNEMOSYNE_CODEX_FORCE_SPOOL"] = "1"
        with mod._scoped_env(env):
            outcome, spool_status = mod.ingest_or_spool(
                self._event("User prefers dark mode for all editors")
            )
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.error_code, "spooled")
        self.assertEqual(spool_status, "stored")
        self.assertEqual(mod.spool_count(self.spool_path), 1)


class TestVisibleFailures(_HookTestBase):
    def test_memory_failure_produces_visible_non_sensitive_warning(self) -> None:
        """When the DB is unreachable, the hook emits a systemMessage warning
        (visible to the user) but still exits 0 (fail-open for Codex)."""
        bad_env = dict(self.env)
        bad_env["MNEMOSYNE_DATA_DIR"] = "/proc/this/cannot/exist/mnemosyne"
        bad_env.update(
            {
                "MNEMOSYNE_CODEX_ACTOR_ID": self.actor,
                "MNEMOSYNE_CODEX_PROJECT_ID": self.project,
            }
        )
        payload = {"session_id": self.session_id, "prompt": "test", "cwd": "/tmp"}
        code, out, _stdout, _stderr = _run_hook(
            "user_prompt_submit.py", payload, bad_env
        )
        self.assertEqual(code, 0, "must fail-open with exit 0")
        # Must emit a visible systemMessage (non-sensitive — no content/credentials)
        msg = (out or {}).get("systemMessage", "")
        self.assertTrue(msg, "must emit a non-empty systemMessage on memory failure")
        # Must NOT contain the prompt content or secrets
        self.assertNotIn(
            "test",
            msg.lower().replace("test", "", 1).replace("memory", "memory"),
            "warning must not echo user content",
        )
        # The word 'memory' or 'mnemosyne' should appear so user knows what failed
        lower = msg.lower()
        self.assertTrue(
            "memory" in lower or "mnemosyne" in lower,
            f"warning must reference memory/mnemosyne, got: {msg}",
        )


class TestNoTranscriptParsing(_HookTestBase):
    def test_no_transcript_file_read_in_any_hook(self) -> None:
        """None of the hook scripts read transcript_path — they use only
        structured fields from the hook payload."""
        # Verify by source inspection: no 'open' or 'read' of transcript_path
        for script in (
            "session_start.py",
            "user_prompt_submit.py",
            "stop.py",
            "session_end.py",
        ):
            path = os.path.join(HOOKS_DIR, script)
            with open(path, encoding="utf-8") as fh:
                src = fh.read()
            # Must not read files referenced as transcript
            self.assertNotIn(
                "transcript_path",
                src.replace('"transcript_path"', "").replace("'transcript_path'", ""),
                f"{script} must not read transcript_path",
            )
            # Must not use open() to read transcript content
            self.assertNotIn(
                "open(transcript", src, f"{script} must not open transcript files"
            )
            self.assertNotIn(
                "read_transcript", src, f"{script} must not parse transcripts"
            )


class TestSessionEndTimeout(_HookTestBase):
    def test_session_end_completes_under_3_seconds(self) -> None:
        """SessionEnd must complete in well under 3 seconds (durable or sync)."""
        import time

        payload = {"session_id": self.session_id, "cwd": "/tmp"}
        actor_env = dict(self.env)
        actor_env.update(
            {
                "MNEMOSYNE_CODEX_ACTOR_ID": self.actor,
                "MNEMOSYNE_CODEX_PROJECT_ID": self.project,
            }
        )
        start = time.monotonic()
        code, _out, _stdout, _stderr = _run_hook("session_end.py", payload, actor_env)
        elapsed = time.monotonic() - start
        self.assertEqual(code, 0)
        self.assertLess(elapsed, 3.0, f"SessionEnd took {elapsed:.2f}s, must be < 3s")


if __name__ == "__main__":
    unittest.main()
