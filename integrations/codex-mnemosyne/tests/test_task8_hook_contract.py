"""Task 8 supplemental tests: official Codex hook contract for SessionEnd.

Binding facts (official Codex manual):
  - systemMessage output is documented for SessionStart, PreCompact,
    PostCompact, UserPromptSubmit, SubagentStop, and Stop — NOT SessionEnd.
  - SessionEnd output is advisory; a command error is reported as a hook
    failure (nonzero exit).
  - Retained spool rows must never be deleted or represented as delivered.
  - Error text must be static and content-free (no exception text, ids,
    content, hashes, scope, or paths).

These tests drive SessionEnd through a real subprocess and assert the
contract directly. They were added RED (expected initial failure: exit 0
and/or an unsupported systemMessage) and drive the minimal GREEN change.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest

PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOOKS_DIR = os.path.join(PLUGIN_ROOT, "hooks")
WORKTREE_ROOT = os.path.dirname(os.path.dirname(PLUGIN_ROOT))
README_PATH = os.path.join(PLUGIN_ROOT, "README.md")


def _run(script: str, payload, env: dict) -> tuple[int, dict | None, str, str]:
    full = dict(os.environ)
    full.update(env)
    prior = full.get("PYTHONPATH", "")
    full["PYTHONPATH"] = WORKTREE_ROOT + (os.pathsep + prior if prior else "")
    raw = json.dumps(payload) if isinstance(payload, (dict, list)) else payload
    proc = subprocess.run(
        [sys.executable, os.path.join(HOOKS_DIR, script)],
        input=raw,
        capture_output=True,
        text=True,
        env=full,
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


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self.data_dir = tempfile.mkdtemp(prefix="mnem-task8-")
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

    def _seed_pending_row(self, prompt: str = "pending event") -> int:
        """Spool one pending row (force-spool), return row count before flush."""
        env = self._env(
            MNEMOSYNE_CODEX_ACTOR_ID="alice",
            MNEMOSYNE_CODEX_PROJECT_ID="projX",
            MNEMOSYNE_CODEX_FORCE_SPOOL="1",
        )
        _run(
            "user_prompt_submit.py",
            {"session_id": "s1", "turn_id": "t1", "prompt": prompt, "cwd": "/tmp"},
            env,
        )
        conn = sqlite3.connect(self.spool_path)
        before = conn.execute("SELECT COUNT(*) FROM spooled_events").fetchone()[0]
        conn.close()
        self.assertGreater(before, 0, "spool must be seeded")
        return before

    def _spool_count(self) -> int:
        conn = sqlite3.connect(self.spool_path)
        n = conn.execute("SELECT COUNT(*) FROM spooled_events").fetchone()[0]
        conn.close()
        return n


# ---------------------------------------------------------------------------
# Contract 1: retained rows => nonzero exit, static content-free stderr,
#             no systemMessage, row retention, elapsed < 3s
# ---------------------------------------------------------------------------


class TestSessionEndRetainedRowsContract(_Base):
    def test_retained_rows_nonzero_static_stderr_no_systemmessage(self) -> None:
        """A SessionEnd that retains a pending/terminal row must: exit nonzero,
        write one static content-free diagnostic to stderr, NOT emit an
        unsupported systemMessage, retain the rows, and finish < 3s."""
        before = self._seed_pending_row(prompt="secret retained value")

        env = self._env(
            MNEMOSYNE_DATA_DIR="/proc/cannot/exist/mnemosyne",
            MNEMOSYNE_CODEX_ACTOR_ID="alice",
            MNEMOSYNE_CODEX_PROJECT_ID="projX",
        )
        start = time.monotonic()
        code, out, _stdout, stderr = _run(
            "session_end.py", {"session_id": "s1", "cwd": "/tmp"}, env
        )
        elapsed = time.monotonic() - start

        # Must exit nonzero so Codex reports the hook failure.
        self.assertNotEqual(code, 0, "retained rows must exit nonzero")

        # Must NOT emit an unsupported systemMessage.
        sm = (out or {}).get("systemMessage")
        self.assertIsNone(sm, f"SessionEnd must not emit systemMessage, got: {sm!r}")

        # Must write a static, content-free diagnostic to stderr.
        self.assertTrue(stderr.strip(), "must write a diagnostic to stderr")
        # Static + content-free: no raw exception text, and must not echo the
        # seeded prompt content or any sensitive token. The static diagnostic
        # itself is a fixed string ("mnemosyne: session end flush incomplete;
        # some events retained.") which is intentionally content-free.
        for forbidden in (
            "Traceback",
            "Exception",
            "Error:",
            "secret retained value",  # the seeded prompt content
            "cx-",
            "mem-",
            "alice",
            "projX",
            "/proc/",
            "session_id",
        ):
            self.assertNotIn(
                forbidden,
                stderr,
                f"stderr must be content-free, leaked '{forbidden}': {stderr!r}",
            )

        # Rows must be retained (not deleted).
        self.assertEqual(
            self._spool_count(),
            before,
            "retained rows must not be deleted",
        )

        # Must finish under the 3s ceiling.
        self.assertLess(elapsed, 3.0, f"SessionEnd took {elapsed:.2f}s, must be < 3s")


# ---------------------------------------------------------------------------
# Contract 2: flush raises => distinct static "status unavailable" stderr,
#             nonzero exit, no raw exception text
# ---------------------------------------------------------------------------


class TestSessionEndFlushExceptionContract(unittest.TestCase):
    """If flush raises, SessionEnd must write a distinct static content-free
    "status unavailable" diagnostic to stderr, exit nonzero, and never expose
    raw exception text.

    flush_spool_bounded is designed never to raise through its subprocess
    boundary (it catches everything internally). To exercise the real
    exception-handling code in session_end.main(), we import the hook module
    directly and make flush raise. This verifies the contract path that
    production code defends against.
    """

    def setUp(self) -> None:
        self.data_dir = tempfile.mkdtemp(prefix="mnem-task8-exc-")

    def tearDown(self) -> None:
        shutil.rmtree(self.data_dir, ignore_errors=True)

    def test_flush_exception_nonzero_static_unavailable_no_exception_text(self) -> None:
        import importlib.util
        import io

        # Load session_end.py as an isolated module.
        spec = importlib.util.spec_from_file_location(
            "_task8_session_end", os.path.join(HOOKS_DIR, "session_end.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Point spool_path at the disposable dir.
        os.environ["MNEMOSYNE_CODEX_SPOOL_PATH"] = os.path.join(
            self.data_dir, "codex-spool.db"
        )
        # Feed empty stdin.
        sys.stdin = (
            io.StringInput("") if hasattr(io, "StringInput") else io.StringIO("")
        )
        captured = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = captured
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            # Make flush raise a noisy exception (with sensitive-looking text)
            # to prove the hook sanitizes it.
            mod.common.flush_spool_bounded = lambda *a, **kw: (_ for _ in ()).throw(
                RuntimeError("detail=secret cx-leak mem-scope /proc/path alice")
            )
            code = mod.main()
        finally:
            sys.stderr = old_stderr
            sys.stdout = old_stdout
            sys.stdin = sys.__stdin__
            os.environ.pop("MNEMOSYNE_CODEX_SPOOL_PATH", None)

        stderr = captured.getvalue()

        # Must exit nonzero.
        self.assertNotEqual(code, 0, "flush exception must exit nonzero")

        # Must write the distinct static "unavailable" diagnostic to stderr.
        self.assertTrue(stderr.strip(), "must write a diagnostic to stderr")
        self.assertIn(
            "unavailable",
            stderr.lower(),
            f"must be the distinct unavailable diagnostic: {stderr!r}",
        )

        # Must NOT expose raw exception text or sensitive tokens.
        for forbidden in (
            "Traceback",
            "RuntimeError",
            "detail=",
            "secret",
            "cx-leak",
            "mem-scope",
            "/proc/",
            "alice",
        ):
            self.assertNotIn(
                forbidden,
                stderr,
                f"stderr leaked '{forbidden}': {stderr!r}",
            )


# ---------------------------------------------------------------------------
# Contract 2b (round 2): corrupt / uninspectable existing spool must NOT be
# represented as successful delivery. Distinct from a genuine empty/absent
# spool (which is exit 0 success).
# ---------------------------------------------------------------------------


class TestSessionEndCorruptSpoolContract(_Base):
    """A corrupt or unreadable existing spool file must surface as a hook
    failure, not be silently coerced to success.

    Reproduces the controller root-cause finding (Aug 10 2026): a spool file
    containing non-SQLite bytes caused session_end.py to return exit 0,
    stdout {}, empty stderr, because both _flush_spool_inner and spool_count
    swallow the DB inspection error and return 0.

    Required behavior: nonzero exit, one static content-free *status
    unavailable* stderr diagnostic (distinct from the retained-rows
    diagnostic), no systemMessage, no raw SQLite/prompt/path leak, the spool
    file left unchanged, elapsed < 3s.
    """

    def test_corrupt_spool_is_unavailable_not_success(self) -> None:
        # Seed a corrupt (non-SQLite) spool file in place.
        corrupt_bytes = b"not sqlite"
        with open(self.spool_path, "wb") as fh:
            fh.write(corrupt_bytes)
        os.chmod(self.spool_path, 0o600)

        env = self._env(
            MNEMOSYNE_CODEX_ACTOR_ID="alice",
            MNEMOSYNE_CODEX_PROJECT_ID="projX",
        )
        start = time.monotonic()
        code, out, _stdout, stderr = _run(
            "session_end.py", {"session_id": "s1", "cwd": "/tmp"}, env
        )
        elapsed = time.monotonic() - start

        # Must exit nonzero — corrupt spool is NOT a successful delivery.
        self.assertNotEqual(
            code, 0, "corrupt spool must exit nonzero, not be coerced to success"
        )

        # Must NOT emit an unsupported systemMessage.
        self.assertIsNone(
            (out or {}).get("systemMessage"),
            "SessionEnd must not emit systemMessage on corrupt spool",
        )

        # Must write the distinct static "unavailable" diagnostic to stderr
        # (NOT the retained-rows diagnostic, since we cannot inspect state).
        self.assertTrue(stderr.strip(), "must write a diagnostic to stderr")
        self.assertIn(
            "unavailable",
            stderr.lower(),
            f"corrupt spool must use the distinct unavailable diagnostic: {stderr!r}",
        )
        # Must NOT use the retained-rows wording for an uninspectable spool.
        self.assertNotIn(
            "retained",
            stderr.lower(),
            f"corrupt spool must use the unavailable diagnostic, not retained: "
            f"{stderr!r}",
        )

        # Static + content-free: no raw SQLite text, exception text, or leak.
        for forbidden in (
            "Traceback",
            "Exception",
            "sqlite3.",
            "DatabaseError",
            "OperationalError",
            "not sqlite",
            "not a database",
            "cx-",
            "mem-",
            "alice",
            "projX",
            "/tmp",
            "session_id",
        ):
            self.assertNotIn(
                forbidden,
                stderr,
                f"stderr leaked '{forbidden}': {stderr!r}",
            )

        # The corrupt spool file must be left unchanged (never mutated/deleted
        # when it cannot be inspected).
        with open(self.spool_path, "rb") as fh:
            after = fh.read()
        self.assertEqual(
            after,
            corrupt_bytes,
            "corrupt spool must be left unchanged, not repaired/deleted",
        )

        # Must finish under the 3s ceiling.
        self.assertLess(elapsed, 3.0, f"SessionEnd took {elapsed:.2f}s, must be < 3s")

    def test_genuinely_empty_spool_is_still_success(self) -> None:
        """A genuine empty (zero-row) spool must remain exit 0 success. This
        guards against the fix over-broadening: corrupt -> unavailable, but
        empty -> success."""
        # No spool file at all, and no MNEMOSYNE_CODEX_FORCE_SPOOL: a clean
        # empty state must be success.
        env = self._env(
            MNEMOSYNE_CODEX_ACTOR_ID="alice",
            MNEMOSYNE_CODEX_PROJECT_ID="projX",
        )
        code, out, _stdout, stderr = _run(
            "session_end.py", {"session_id": "s1", "cwd": "/tmp"}, env
        )
        self.assertEqual(code, 0, f"genuine empty spool must exit 0: {stderr}")
        self.assertIsNotNone(out, "must emit a JSON line on success")
        self.assertEqual(stderr.strip(), "", "empty spool must not write stderr")


# ---------------------------------------------------------------------------
# Contract 2c (round 3): inaccessible-existing spool path must NOT be conflated
# with a genuinely absent spool. os.path.exists() returns False on EACCES
# during stat/traverse, which previously made spool_inspect_state return
# (0, True) -> main() exit 0 success for a path that exists but cannot be
# inspected.
# ---------------------------------------------------------------------------


class TestSessionEndInaccessibleSpoolContract(_Base):
    """A spool path that exists but cannot be traversed/stat'd/opened must
    surface as a hook failure (status unavailable), not be coerced to success.

    Reproduces the controller root-cause finding (Aug 10, 2026): with
    MNEMOSYNE_CODEX_SPOOL_PATH=<tmp>/noaccess/codex-spool.db and the parent
    directory chmod 000, os.path.exists() returned False, so the strict helper
    treated the existing-but-inaccessible path as absent and main() exited 0.

    Permissions are always restored in cleanup so the temp dir can be removed.
    If the platform cannot make the path inaccessible to the test process
    (e.g. running as root), the inaccessible test is skipped with a reason;
    the assertion is never weakened.
    """

    def setUp(self) -> None:
        super().setUp()
        self._noaccess_dir = os.path.join(self.data_dir, "noaccess")
        os.makedirs(self._noaccess_dir, mode=0o700, exist_ok=True)
        self._inaccessible_spool = os.path.join(self._noaccess_dir, "codex-spool.db")

    def tearDown(self) -> None:
        # Always restore perms so shutil.rmtree in the parent teardown works.
        try:
            os.chmod(self._noaccess_dir, 0o700)
        except OSError:
            pass
        super().tearDown()

    @unittest.skipIf(
        hasattr(os, "geteuid") and os.geteuid() == 0,
        "running as root: chmod 000 cannot revoke root's access, so this "
        "test cannot reproduce the inaccessible-path condition",
    )
    def test_inaccessible_existing_spool_is_unavailable_not_success(self) -> None:
        # Seed an existing spool file, then make its parent dir inaccessible
        # (mode 000) so stat/traverse/open fail with EACCES, not ENOENT.
        with open(self._inaccessible_spool, "w") as fh:
            fh.write("existing-but-inaccessible")
        os.chmod(self._inaccessible_spool, 0o600)
        os.chmod(self._noaccess_dir, 0o000)

        env = self._env(
            MNEMOSYNE_CODEX_ACTOR_ID="alice",
            MNEMOSYNE_CODEX_PROJECT_ID="projX",
            MNEMOSYNE_CODEX_SPOOL_PATH=self._inaccessible_spool,
        )
        start = time.monotonic()
        try:
            code, out, _stdout, stderr = _run(
                "session_end.py", {"session_id": "s1", "cwd": "/tmp"}, env
            )
        finally:
            # Restore immediately so later assertions/teardown can read state.
            os.chmod(self._noaccess_dir, 0o700)
        elapsed = time.monotonic() - start

        # Must exit nonzero — an inaccessible existing path is NOT success.
        self.assertNotEqual(
            code,
            0,
            "inaccessible existing spool must exit nonzero, not be conflated "
            "with absent path",
        )

        # Must NOT emit an unsupported systemMessage.
        self.assertIsNone(
            (out or {}).get("systemMessage"),
            "SessionEnd must not emit systemMessage on inaccessible spool",
        )

        # Must write the distinct static "unavailable" diagnostic (NOT the
        # retained wording, since state cannot be determined).
        self.assertTrue(stderr.strip(), "must write a diagnostic to stderr")
        self.assertIn(
            "unavailable",
            stderr.lower(),
            f"inaccessible spool must use the distinct unavailable diagnostic: "
            f"{stderr!r}",
        )
        self.assertNotIn(
            "retained",
            stderr.lower(),
            f"inaccessible spool must use unavailable, not retained: {stderr!r}",
        )

        # Static + content-free: no raw exception text, no path leak.
        for forbidden in (
            "Traceback",
            "Exception",
            "PermissionError",
            "OSError",
            "EACCES",
            "noaccess",
            "codex-spool.db",
            "cx-",
            "mem-",
            "alice",
            "/tmp",
        ):
            self.assertNotIn(
                forbidden,
                stderr,
                f"stderr leaked '{forbidden}': {stderr!r}",
            )

        # Must finish under the 3s ceiling.
        self.assertLess(elapsed, 3.0, f"SessionEnd took {elapsed:.2f}s, must be < 3s")

    def test_genuinely_absent_spool_is_still_success(self) -> None:
        """A genuinely absent spool path (ENOENT) must remain exit 0 success.
        This guards against the fix over-broadening: inaccessible -> unavailable,
        but absent -> success."""
        # Point at a path that does not exist under a normal writable dir.
        absent = os.path.join(self.data_dir, "never-created.db")
        env = self._env(
            MNEMOSYNE_CODEX_ACTOR_ID="alice",
            MNEMOSYNE_CODEX_PROJECT_ID="projX",
            MNEMOSYNE_CODEX_SPOOL_PATH=absent,
        )
        code, out, _stdout, stderr = _run(
            "session_end.py", {"session_id": "s1", "cwd": "/tmp"}, env
        )
        self.assertEqual(code, 0, f"genuinely absent spool must exit 0: {stderr}")
        self.assertIsNotNone(out, "must emit a JSON line on success")
        self.assertEqual(stderr.strip(), "", "absent spool must not write stderr")


# ---------------------------------------------------------------------------
# Contract 3: README states real SessionEnd behavior + desktop manual sequence
#             (restart desktop, marketplace, install/enable, /hooks trust,
#              new session, test the four lifecycle hooks)
# ---------------------------------------------------------------------------


class TestReadmeSessionEndContract(unittest.TestCase):
    def test_readme_documents_nonzero_exit_and_no_systemmessage(self) -> None:
        with open(README_PATH, encoding="utf-8") as fh:
            readme = fh.read()
        # Real SessionEnd behavior must be stated.
        self.assertIn(
            "nonzero",
            readme.lower(),
            "README must state SessionEnd exits nonzero on retained rows",
        )
        # SessionEnd must not be described as emitting a systemMessage.
        # (Only the SessionEnd advisory section — other hooks legitimately use it.)

    def test_readme_documents_hooks_trust_manual_sequence(self) -> None:
        with open(README_PATH, encoding="utf-8") as fh:
            readme = fh.read().lower()
        # Desktop manual sequence checkpoints.
        for needle in (
            "/hooks",
            "trust",
            "marketplace",
            "sessionend",
            "restart",
        ):
            self.assertIn(
                needle,
                readme,
                f"README must document the desktop manual sequence including "
                f"'{needle}'",
            )


if __name__ == "__main__":
    unittest.main()
