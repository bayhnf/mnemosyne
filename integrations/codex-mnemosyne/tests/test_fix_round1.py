"""Round-1 fix tests: binding review findings for Task 8.

Each test maps to a blocking finding from task-8-review-round1.md:

  C1 / req 1: persistent cross-session memory, isolated per actor+project
  req 2: stable host IDs (turn_id) + non-object payload fail-open
  req 3: honest, content-free visible messages (no false "queued")
  req 4: durable bounded spool: idempotent, finite capacity, terminal state,
         no silent deletion of corrupt rows, additive schema
  req 5: SessionEnd returns before 3s even if ingest is slow/hung
  req 6: installed plugin uses PLUGIN_DATA; absent package warning

These tests exercise actual hook replays (subprocess), not helper-only calls.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import unittest

PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOOKS_DIR = os.path.join(PLUGIN_ROOT, "hooks")
# The worktree root holds the mnemosyne package; in an installed deployment
# this resolves via pip install. For subprocess hook tests we put it on
# PYTHONPATH to simulate the installed-package case.
WORKTREE_ROOT = os.path.dirname(os.path.dirname(PLUGIN_ROOT))


def _run(
    script: str,
    payload,
    env: dict,
    cwd: str = "/tmp",
    *,
    inject_worktree: bool = True,
    no_site: bool = False,
) -> tuple[int, dict | None, str, str]:
    """Run a hook with raw stdin (may be non-object JSON). Returns (rc, parsed, out, err).

    By default PYTHONPATH is prefixed with the worktree root to simulate the
    installed-package case. With `no_site`, the hook runs under `python -S`
    with no PYTHONPATH, so a pip-installed mnemosyne is genuinely unimportable
    in that interpreter even when pytest itself runs inside an installed env.
    """
    full = dict(os.environ)
    full.update(env)
    if inject_worktree and not no_site:
        prior = full.get("PYTHONPATH", "")
        full["PYTHONPATH"] = WORKTREE_ROOT + (os.pathsep + prior if prior else "")
    if no_site:
        full.pop("PYTHONPATH", None)
    if isinstance(payload, (dict, list)):
        raw = json.dumps(payload)
    else:
        raw = payload  # pre-serialized string (e.g. a JSON array or string)
    command = [sys.executable]
    if no_site:
        command.append("-S")
    command.append(os.path.join(HOOKS_DIR, script))
    proc = subprocess.run(
        command,
        input=raw,
        capture_output=True,
        text=True,
        env=full,
        timeout=30,
        cwd=cwd,
    )
    parsed = None
    if proc.stdout.strip():
        try:
            parsed = json.loads(proc.stdout)
        except json.JSONDecodeError:
            parsed = None
    return proc.returncode, parsed, proc.stdout, proc.stderr


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self.data_dir = tempfile.mkdtemp(prefix="mnem-fix1-")
        self.spool_path = os.path.join(self.data_dir, "codex-spool.db")

    def tearDown(self) -> None:
        shutil.rmtree(self.data_dir, ignore_errors=True)

    def _env(self, **over) -> dict:
        env = {
            "MNEMOSYNE_DATA_DIR": self.data_dir,
            "MNEMOSYNE_CODEX_SPOOL_PATH": self.spool_path,
        }
        env.update(over)
        return env


# ---------------------------------------------------------------------------
# C1 / req 1: persistent cross-session memory, isolated per actor + project
# ---------------------------------------------------------------------------


class TestPersistentCrossSessionMemory(_Base):
    """Same actor+project session-A -> session-B must recall."""

    def test_cross_session_recall_same_actor_project(self) -> None:
        env = self._env(
            MNEMOSYNE_CODEX_ACTOR_ID="alice", MNEMOSYNE_CODEX_PROJECT_ID="projX"
        )
        _run(
            "user_prompt_submit.py",
            {
                "session_id": "sess-A",
                "turn_id": "tA1",
                "prompt": "I prefer dark mode for editors",
                "cwd": "/tmp",
            },
            env,
        )
        code, out, _o, err = _run(
            "user_prompt_submit.py",
            {
                "session_id": "sess-B",
                "turn_id": "tB1",
                "prompt": "what do I prefer?",
                "cwd": "/tmp",
            },
            env,
        )
        self.assertEqual(code, 0, f"exit 0 fail-open, err={err}")
        ctx = (out or {}).get("hookSpecificOutput", {}).get("additionalContext", "")
        self.assertIn(
            "dark mode", ctx, f"session-B must recall session-A memory: {ctx[:300]}"
        )

    def test_different_actor_must_not_recall(self) -> None:
        env_a = self._env(
            MNEMOSYNE_CODEX_ACTOR_ID="alice", MNEMOSYNE_CODEX_PROJECT_ID="projX"
        )
        _run(
            "user_prompt_submit.py",
            {
                "session_id": "sess-A",
                "turn_id": "tA1",
                "prompt": "alice secret alpha",
                "cwd": "/tmp",
            },
            env_a,
        )
        env_b = self._env(
            MNEMOSYNE_CODEX_ACTOR_ID="bob", MNEMOSYNE_CODEX_PROJECT_ID="projX"
        )
        # Bob asks a question (different content) that should NOT surface alice's secret.
        code, out, _o, err = _run(
            "user_prompt_submit.py",
            {
                "session_id": "sess-B",
                "turn_id": "tB1",
                "prompt": "what do you recall about secrets?",
                "cwd": "/tmp",
            },
            env_b,
        )
        self.assertEqual(code, 0)
        ctx = (out or {}).get("hookSpecificOutput", {}).get("additionalContext", "")
        self.assertNotIn(
            "alice secret alpha", ctx, f"cross-actor must NOT recall: {ctx[:300]}"
        )

    def test_different_project_must_not_recall(self) -> None:
        env_a = self._env(
            MNEMOSYNE_CODEX_ACTOR_ID="alice", MNEMOSYNE_CODEX_PROJECT_ID="projX"
        )
        _run(
            "user_prompt_submit.py",
            {
                "session_id": "sess-A",
                "turn_id": "tA1",
                "prompt": "projX confidential detail",
                "cwd": "/tmp",
            },
            env_a,
        )
        env_b = self._env(
            MNEMOSYNE_CODEX_ACTOR_ID="alice", MNEMOSYNE_CODEX_PROJECT_ID="projY"
        )
        # Project-Y asks a question (different content) that should NOT surface projX's secret.
        code, out, _o, err = _run(
            "user_prompt_submit.py",
            {
                "session_id": "sess-B",
                "turn_id": "tB1",
                "prompt": "what confidential details are known?",
                "cwd": "/tmp",
            },
            env_b,
        )
        self.assertEqual(code, 0)
        ctx = (out or {}).get("hookSpecificOutput", {}).get("additionalContext", "")
        self.assertNotIn(
            "projX confidential detail",
            ctx,
            f"cross-project must NOT recall: {ctx[:300]}",
        )

    def test_session_start_recalls_across_sessions(self) -> None:
        env = self._env(
            MNEMOSYNE_CODEX_ACTOR_ID="alice", MNEMOSYNE_CODEX_PROJECT_ID="projX"
        )
        _run(
            "user_prompt_submit.py",
            {
                "session_id": "sess-A",
                "turn_id": "tA1",
                "prompt": "I like terse answers",
                "cwd": "/tmp",
            },
            env,
        )
        code, out, _o, err = _run(
            "session_start.py",
            {"session_id": "sess-B", "source": "startup", "cwd": "/tmp"},
            env,
        )
        self.assertEqual(code, 0)
        ctx = (out or {}).get("hookSpecificOutput", {}).get("additionalContext", "")
        self.assertIn(
            "terse",
            ctx,
            f"SessionStart in session-B must recall session-A: {ctx[:300]}",
        )


# ---------------------------------------------------------------------------
# req 2: stable host IDs + non-object payload fail-open
# ---------------------------------------------------------------------------


class TestStableHostIdsAndPayloadSafety(_Base):
    def test_uses_payload_turn_id_when_present(self) -> None:
        """The host turn_id must be used for event identity when supplied."""
        env = self._env(
            MNEMOSYNE_CODEX_ACTOR_ID="alice", MNEMOSYNE_CODEX_PROJECT_ID="projX"
        )
        # Same turn_id, replayed => idempotent (same event_id, deduped)
        payload = {
            "session_id": "s1",
            "turn_id": "host-turn-42",
            "prompt": "hello world",
            "cwd": "/tmp",
        }
        _run("user_prompt_submit.py", payload, env)
        _run("user_prompt_submit.py", payload, env)
        # Verify only ONE row for this content in the DB
        import sqlite3

        db = os.path.join(self.data_dir, "mnemosyne.db")
        conn = sqlite3.connect(db)
        rows = conn.execute(
            "SELECT content FROM working_memory WHERE content = ?", ("hello world",)
        ).fetchall()
        conn.close()
        self.assertEqual(len(rows), 1, "replay with same turn_id must be idempotent")

    def test_non_object_json_string_fail_open(self) -> None:
        env = self._env()
        code, out, _o, err = _run("user_prompt_submit.py", '"just a string"', env)
        self.assertEqual(code, 0, f"non-object JSON must exit 0, err={err}")
        self.assertIsNotNone(out, "must emit valid JSON, not traceback")
        self.assertNotIn("Traceback", err)
        self.assertNotIn("Traceback", out.get("systemMessage", "") if out else "")

    def test_non_object_json_array_fail_open(self) -> None:
        env = self._env()
        code, out, _o, err = _run("stop.py", "[1, 2, 3]", env)
        self.assertEqual(code, 0, f"non-object JSON must exit 0, err={err}")
        self.assertIsNotNone(out)

    def test_non_object_json_number_session_start(self) -> None:
        env = self._env()
        code, out, _o, err = _run("session_start.py", "42", env)
        self.assertEqual(code, 0)
        self.assertIsNotNone(out, "must emit JSON not traceback")
        self.assertNotIn("Traceback", err)

    def test_non_object_json_session_end(self) -> None:
        env = self._env()
        code, out, _o, err = _run("session_end.py", "true", env)
        self.assertEqual(code, 0)
        self.assertIsNotNone(out)


# ---------------------------------------------------------------------------
# req 3: honest, content-free visible messages
# ---------------------------------------------------------------------------


class TestHonestMessages(_Base):
    _SENSITIVE = [
        "secret",
        "confidential",
        "prompt",
        "/tmp",
        "sess-",
        "cx-",
        "mem-",
        "dark mode",
        "alpha",
        "projX",
        "alice",
    ]

    def _assert_content_free(self, text: str, raw_stdout: str = "") -> None:
        combined = f"{text} {raw_stdout}".lower()
        for s in self._SENSITIVE:
            self.assertNotIn(
                s.lower(),
                combined,
                f"output must not leak sensitive token '{s}': {text[:200]}",
            )

    def test_spool_write_failure_does_not_claim_queued(self) -> None:
        """If the durable spool write FAILS, the message must NOT say 'queued'."""
        env = self._env(
            MNEMOSYNE_CODEX_ACTOR_ID="alice",
            MNEMOSYNE_CODEX_PROJECT_ID="projX",
            MNEMOSYNE_CODEX_FORCE_SPOOL="1",
        )
        # Make spool directory unwritable so the spool write fails.
        ro_dir = os.path.join(self.data_dir, "ro")
        os.makedirs(ro_dir, mode=0o500)
        env["MNEMOSYNE_CODEX_SPOOL_PATH"] = os.path.join(ro_dir, "codex-spool.db")
        code, out, _o, err = _run(
            "user_prompt_submit.py",
            {
                "session_id": "s1",
                "turn_id": "t1",
                "prompt": "secret project alpha",
                "cwd": "/tmp",
            },
            env,
        )
        self.assertEqual(code, 0, "fail-open")
        msg = (out or {}).get("systemMessage", "")
        self.assertTrue(msg, "must emit a visible systemMessage when spool write fails")
        # Must not affirmatively claim the event was queued/spooled.
        self.assertNotIn("was queued", msg.lower())
        self.assertNotIn("was spooled", msg.lower())
        self.assertNotIn("has been queued", msg.lower())
        self.assertNotIn("has been spooled", msg.lower())
        self.assertNotIn("durably queued", msg.lower())
        self._assert_content_free(msg)

    def test_recall_error_message_is_content_free(self) -> None:
        env = self._env(
            MNEMOSYNE_DATA_DIR="/proc/cannot/exist/mnemosyne",
            MNEMOSYNE_CODEX_ACTOR_ID="alice",
            MNEMOSYNE_CODEX_PROJECT_ID="projX",
        )
        code, out, _o, err = _run(
            "session_start.py",
            {"session_id": "s1", "source": "startup", "cwd": "/tmp"},
            env,
        )
        self.assertEqual(code, 0)
        msg = (out or {}).get("systemMessage", "")
        self.assertTrue(msg, "recall error must produce a visible systemMessage")
        self._assert_content_free(msg)

    def test_session_start_never_claims_queued(self) -> None:
        """SessionStart does no ingest; it must never say an event was queued."""
        env = self._env(
            MNEMOSYNE_CODEX_ACTOR_ID="alice", MNEMOSYNE_CODEX_PROJECT_ID="projX"
        )
        code, out, _o, err = _run(
            "session_start.py",
            {"session_id": "s1", "source": "startup", "cwd": "/tmp"},
            env,
        )
        self.assertEqual(code, 0)
        blob = json.dumps(out or {})
        self.assertNotIn("queued", blob.lower(), "SessionStart must not mention queued")
        self.assertNotIn(
            "spooled", blob.lower(), "SessionStart must not mention spooled"
        )

    def test_ingest_error_does_not_leak_exception(self) -> None:
        env = self._env(
            MNEMOSYNE_DATA_DIR="/proc/cannot/exist/mnemosyne",
            MNEMOSYNE_CODEX_ACTOR_ID="alice",
            MNEMOSYNE_CODEX_PROJECT_ID="projX",
            MNEMOSYNE_CODEX_FORCE_SPOOL="1",
        )
        code, out, _o, err = _run(
            "user_prompt_submit.py",
            {
                "session_id": "s1",
                "turn_id": "t1",
                "prompt": "confidential secret data",
                "cwd": "/tmp",
            },
            env,
        )
        self.assertEqual(code, 0)
        blob = json.dumps(out or {})
        self.assertNotIn("Exception", blob)
        self.assertNotIn("Traceback", blob)
        self._assert_content_free(blob)


# ---------------------------------------------------------------------------
# req 4: durable bounded spool
# ---------------------------------------------------------------------------


class TestBoundedSpool(_Base):
    def test_spool_is_mode_0600(self) -> None:
        env = self._env(
            MNEMOSYNE_CODEX_ACTOR_ID="alice",
            MNEMOSYNE_CODEX_PROJECT_ID="projX",
            MNEMOSYNE_CODEX_FORCE_SPOOL="1",
        )
        _run(
            "user_prompt_submit.py",
            {"session_id": "s1", "turn_id": "t1", "prompt": "x", "cwd": "/tmp"},
            env,
        )
        self.assertTrue(os.path.exists(self.spool_path))
        mode = stat.S_IMODE(os.stat(self.spool_path).st_mode)
        self.assertEqual(mode, 0o600)

    def test_duplicate_event_id_is_idempotent(self) -> None:
        env = self._env(
            MNEMOSYNE_CODEX_ACTOR_ID="alice",
            MNEMOSYNE_CODEX_PROJECT_ID="projX",
            MNEMOSYNE_CODEX_FORCE_SPOOL="1",
        )
        payload = {
            "session_id": "s1",
            "turn_id": "t1",
            "prompt": "dup content",
            "cwd": "/tmp",
        }
        _run("user_prompt_submit.py", payload, env)
        _run("user_prompt_submit.py", payload, env)
        import sqlite3

        conn = sqlite3.connect(self.spool_path)
        # idempotent: duplicate stable event_id must not create two spool rows
        rows = conn.execute(
            "SELECT event_id, COUNT(*) FROM spooled_events GROUP BY event_id"
        ).fetchall()
        conn.close()
        for eid, cnt in rows:
            self.assertEqual(
                cnt, 1, f"event_id {eid} spooled {cnt} times; must be idempotent"
            )

    def test_corrupt_row_not_silently_deleted(self) -> None:
        env = self._env(
            MNEMOSYNE_CODEX_ACTOR_ID="alice",
            MNEMOSYNE_CODEX_PROJECT_ID="projX",
            MNEMOSYNE_CODEX_FORCE_SPOOL="1",
        )
        _run(
            "user_prompt_submit.py",
            {
                "session_id": "s1",
                "turn_id": "t1",
                "prompt": "good event",
                "cwd": "/tmp",
            },
            env,
        )
        # Corrupt one row's payload
        import sqlite3

        conn = sqlite3.connect(self.spool_path)
        conn.execute(
            "UPDATE spooled_events SET payload_json = 'NOT JSON{' WHERE rowid > 0"
        )
        conn.commit()
        conn.close()
        # Flush: corrupt row must NOT be silently deleted
        _run(
            "session_end.py",
            {"session_id": "s1", "cwd": "/tmp"},
            self._env(
                MNEMOSYNE_CODEX_ACTOR_ID="alice", MNEMOSYNE_CODEX_PROJECT_ID="projX"
            ),
        )
        conn = sqlite3.connect(self.spool_path)
        remaining = conn.execute("SELECT COUNT(*) FROM spooled_events").fetchone()[0]
        conn.close()
        self.assertGreaterEqual(
            remaining, 1, "corrupt rows must not be silently deleted"
        )

    def test_finite_capacity(self) -> None:
        """The spool must have a finite row capacity; excess does not grow unbounded."""
        env = self._env(
            MNEMOSYNE_CODEX_ACTOR_ID="alice",
            MNEMOSYNE_CODEX_PROJECT_ID="projX",
            MNEMOSYNE_CODEX_FORCE_SPOOL="1",
        )
        for i in range(50):
            _run(
                "user_prompt_submit.py",
                {
                    "session_id": "s1",
                    "turn_id": f"t{i}",
                    "prompt": f"event {i}",
                    "cwd": "/tmp",
                },
                env,
            )
        import sqlite3

        conn = sqlite3.connect(self.spool_path)
        count = conn.execute("SELECT COUNT(*) FROM spooled_events").fetchone()[0]
        conn.close()
        self.assertLess(count, 50, f"spool must cap capacity, got {count}")
        self.assertGreater(count, 0)

    def test_terminal_state_stops_retry(self) -> None:
        """A row that has reached the terminal retry ceiling must not be retried
        again and must be retained (not deleted). We simulate this by marking a
        row terminal directly, then flushing with a WORKING native DB: a
        terminal row must NOT be selected for retry."""
        env = self._env(
            MNEMOSYNE_CODEX_ACTOR_ID="alice",
            MNEMOSYNE_CODEX_PROJECT_ID="projX",
            MNEMOSYNE_CODEX_FORCE_SPOOL="1",
        )
        _run(
            "user_prompt_submit.py",
            {
                "session_id": "s1",
                "turn_id": "t1",
                "prompt": "doomed event",
                "cwd": "/tmp",
            },
            env,
        )
        import sqlite3

        # Mark the row terminal (attempts at ceiling, terminal flag set).
        conn = sqlite3.connect(self.spool_path)
        conn.execute(
            "UPDATE spooled_events SET attempts = 999, terminal = 1 WHERE rowid > 0"
        )
        conn.commit()
        rows_before = conn.execute("SELECT COUNT(*) FROM spooled_events").fetchone()[0]
        conn.close()
        self.assertGreater(rows_before, 0)
        # Flush with a WORKING native DB: terminal rows must be skipped + retained.
        _run(
            "session_end.py",
            {"session_id": "s1", "cwd": "/tmp"},
            self._env(
                MNEMOSYNE_CODEX_ACTOR_ID="alice", MNEMOSYNE_CODEX_PROJECT_ID="projX"
            ),
        )
        conn = sqlite3.connect(self.spool_path)
        rows_after = conn.execute("SELECT COUNT(*) FROM spooled_events").fetchone()[0]
        conn.close()
        self.assertEqual(
            rows_after,
            rows_before,
            "terminal rows must be retained, not retried/deleted",
        )


# ---------------------------------------------------------------------------
# req 5: SessionEnd returns before 3s even if ingest is slow/hung
# ---------------------------------------------------------------------------


class TestSessionEndBoundedTime(_Base):
    def test_session_end_retains_unacked_events(self) -> None:
        """SessionEnd must not delete events it cannot deliver. We seed the
        spool, then make native ingest fail (bad data dir) during SessionEnd:
        the row must be retained, not deleted."""
        env = self._env(
            MNEMOSYNE_CODEX_ACTOR_ID="alice",
            MNEMOSYNE_CODEX_PROJECT_ID="projX",
            MNEMOSYNE_CODEX_FORCE_SPOOL="1",
        )
        _run(
            "user_prompt_submit.py",
            {"session_id": "s1", "turn_id": "t1", "prompt": "unacked", "cwd": "/tmp"},
            env,
        )
        import sqlite3

        conn = sqlite3.connect(self.spool_path)
        before = conn.execute("SELECT COUNT(*) FROM spooled_events").fetchone()[0]
        conn.close()
        self.assertGreater(before, 0)
        # SessionEnd with an unreachable native DB: ingest fails, row retained.
        bad_env = self._env(
            MNEMOSYNE_DATA_DIR="/proc/cannot/exist/mnemosyne",
            MNEMOSYNE_CODEX_ACTOR_ID="alice",
            MNEMOSYNE_CODEX_PROJECT_ID="projX",
        )
        _run("session_end.py", {"session_id": "s1", "cwd": "/tmp"}, bad_env)
        conn = sqlite3.connect(self.spool_path)
        after = conn.execute("SELECT COUNT(*) FROM spooled_events").fetchone()[0]
        conn.close()
        self.assertEqual(after, before, "unacked events must be retained")


# ---------------------------------------------------------------------------
# req 6: installed plugin (PLUGIN_DATA) + absent package warning
# ---------------------------------------------------------------------------


class TestInstalledPlugin(_Base):
    def test_plugin_data_used_for_default_state(self) -> None:
        """Default writable plugin state must come from PLUGIN_DATA, not repo root."""
        pd = os.path.join(self.data_dir, "plugin-data")
        os.makedirs(pd, exist_ok=True)
        env = self._env(
            PLUGIN_DATA=pd,
            MNEMOSYNE_CODEX_ACTOR_ID="alice",
            MNEMOSYNE_CODEX_PROJECT_ID="projX",
            MNEMOSYNE_CODEX_FORCE_SPOOL="1",
        )
        # Clear the explicit spool path so default resolution kicks in
        env.pop("MNEMOSYNE_CODEX_SPOOL_PATH")
        _run(
            "user_prompt_submit.py",
            {
                "session_id": "s1",
                "turn_id": "t1",
                "prompt": "plugin data test",
                "cwd": "/tmp",
            },
            env,
        )
        # Spool must land under PLUGIN_DATA, not the source repo
        self.assertTrue(
            os.path.exists(os.path.join(pd, "codex-spool.db")),
            "spool must default to PLUGIN_DATA",
        )

    def test_absent_mnemosyne_package_emits_actionable_warning(self) -> None:
        """When the mnemosyne package is not importable, emit a safe actionable warning."""
        env = self._env(
            PLUGIN_DATA=os.path.join(self.data_dir, "pd"),
        )
        code, out, _out, err = _run(
            "user_prompt_submit.py",
            {"session_id": "s1", "turn_id": "t1", "prompt": "hello", "cwd": "/tmp"},
            env,
            no_site=True,
        )
        self.assertEqual(code, 0, "must fail-open even if package absent")
        msg = (out or {}).get("systemMessage", "")
        self.assertTrue(msg, "must emit a visible warning when package absent")
        lower = msg.lower()
        self.assertIn("mnemosyne", lower, "warning must name mnemosyne")
        self.assertIn("install", lower, "warning must be actionable (mention install)")


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Slow/hung ingest: SessionEnd must return before the 3s ceiling even when
# a native ingest attempt blocks. Uses a stdlib-only slow-ingest shim.
# ---------------------------------------------------------------------------


class TestSessionEndSlowIngest(_Base):
    def test_slow_ingest_does_not_block_session_end(self) -> None:
        """If native ingest sleeps for 10s, SessionEnd must still return < 3s
        and retain the unacknowledged event. Uses a stdlib subprocess deadline,
        not a reliance on Codex killing the hook."""
        env = self._env(
            MNEMOSYNE_CODEX_ACTOR_ID="alice",
            MNEMOSYNE_CODEX_PROJECT_ID="projX",
            MNEMOSYNE_CODEX_FORCE_SPOOL="1",
        )
        # Seed the spool with one pending event.
        _run(
            "user_prompt_submit.py",
            {
                "session_id": "s1",
                "turn_id": "t1",
                "prompt": "will hang on flush",
                "cwd": "/tmp",
            },
            env,
        )
        import sqlite3

        conn = sqlite3.connect(self.spool_path)
        before = conn.execute("SELECT COUNT(*) FROM spooled_events").fetchone()[0]
        conn.close()
        self.assertGreater(before, 0)

        # Install a slow-ingest shim: a fake mnemosyne package on PYTHONPATH
        # whose remember_event sleeps 10s. The hook's deadline must cut it off.
        fake_root = tempfile.mkdtemp(prefix="mnem-fake-pkg-")
        pkg_dir = os.path.join(fake_root, "mnemosyne", "core")
        os.makedirs(pkg_dir, exist_ok=True)
        # mnemosyne/__init__.py
        open(os.path.join(fake_root, "mnemosyne", "__init__.py"), "w").write("")
        # core/__init__.py
        open(os.path.join(pkg_dir, "__init__.py"), "w").write("")
        # core/memory.py with a slow Mnemosyne
        open(os.path.join(pkg_dir, "memory.py"), "w").write(
            "class Mnemosyne:\n"
            "    def __init__(self, **kw):\n"
            "        self.__dict__.update(kw)\n"
            "    def remember_event(self, event):\n"
            "        import time; time.sleep(10)\n"
            "        raise RuntimeError('slow ingest timed out')\n"
            "    def recall_bounded(self, *a, **kw):\n"
            "        raise RuntimeError('unavailable')\n"
            "    def retry_pending_ingest(self, *a, **kw):\n"
            "        pass\n"
        )
        # core/inhale.py — minimal IngestEvent + receipt
        open(os.path.join(pkg_dir, "inhale.py"), "w").write(
            "from dataclasses import dataclass, field\n"
            "from typing import Dict, Any, List, Optional\n"
            "@dataclass(frozen=True)\n"
            "class IngestEvent:\n"
            "    event_id: str\n"
            "    producer: str\n"
            "    actor_id: str\n"
            "    project_id: str\n"
            "    session_id: str\n"
            "    turn_id: str\n"
            "    role: str\n"
            "    content: str\n"
            "    content_hash: str\n"
            "    occurred_at: str\n"
            "    metadata: Optional[Dict[str, Any]] = None\n"
        )
        # core/recall_bounded.py — minimal RecallPolicy
        open(os.path.join(pkg_dir, "recall_bounded.py"), "w").write(
            "from dataclasses import dataclass\n"
            "from typing import Optional, Sequence\n"
            "@dataclass(frozen=True)\n"
            "class RecallPolicy:\n"
            "    top_k: int = 20\n"
            "    max_tokens: Optional[int] = None\n"
            "    actor_ids: Optional[Sequence[str]] = None\n"
            "    project_ids: Optional[Sequence[str]] = None\n"
            "    session_ids: Optional[Sequence[str]] = None\n"
        )
        try:
            slow_env = self._env(
                MNEMOSYNE_CODEX_ACTOR_ID="alice", MNEMOSYNE_CODEX_PROJECT_ID="projX"
            )
            slow_env["PYTHONPATH"] = fake_root
            start = time.monotonic()
            code, out, _o, err = _run(
                "session_end.py",
                {"session_id": "s1", "cwd": "/tmp"},
                slow_env,
                inject_worktree=False,
            )
            elapsed = time.monotonic() - start
            # Task 8 official hook contract: retained rows => nonzero exit.
            self.assertNotEqual(code, 0, "retained rows must exit nonzero")
            self.assertLess(
                elapsed,
                3.0,
                f"SessionEnd took {elapsed:.2f}s with slow ingest; must be < 3s",
            )
            # The unacknowledged event must be retained (not deleted).
            conn = sqlite3.connect(self.spool_path)
            after = conn.execute("SELECT COUNT(*) FROM spooled_events").fetchone()[0]
            conn.close()
            self.assertEqual(
                after, before, "unacked event must be retained after slow ingest"
            )
        finally:
            shutil.rmtree(fake_root, ignore_errors=True)


# ---------------------------------------------------------------------------
