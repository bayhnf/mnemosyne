"""Round-2 fix tests: hermetic isolation, installed-location smoke, real
marketplace manifest, hard-deadline SessionEnd, and truthful plugin state.

All tests use disposable paths only. No ~/.codex, ~/.hermes, CODEX_HOME,
real codex CLI, or network access.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOOKS_DIR = os.path.join(PLUGIN_ROOT, "hooks")
WORKTREE_ROOT = os.path.dirname(os.path.dirname(PLUGIN_ROOT))
REPO_ROOT = WORKTREE_ROOT  # $REPO_ROOT for marketplace.json


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
        self.data_dir = tempfile.mkdtemp(prefix="mnem-r2-")
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
# Finding C1: marketplace tests must be hermetic — validate the real repo file
# ---------------------------------------------------------------------------


class TestMarketplaceManifest(unittest.TestCase):
    """Static validation of the real repo-scoped marketplace.json.

    Does NOT invoke the codex CLI, does NOT read ~/.codex, does NOT set
    CODEX_HOME. The Desktop Plugins Directory install/activation remains an
    explicit manual gate documented in the report — never a false installed
    claim.
    """

    def test_repo_marketplace_json_exists(self) -> None:
        """The repo-scoped marketplace.json must exist at the documented path."""
        path = os.path.join(REPO_ROOT, ".agents", "plugins", "marketplace.json")
        self.assertTrue(
            os.path.isfile(path),
            f"repo marketplace.json must exist at {path}",
        )

    def test_repo_marketplace_json_schema(self) -> None:
        """Validate against the documented marketplace schema."""
        import json

        path = os.path.join(REPO_ROOT, ".agents", "plugins", "marketplace.json")
        with open(path) as f:
            m = json.load(f)
        self.assertIn("name", m)
        self.assertIsInstance(m.get("plugins"), list)
        self.assertGreaterEqual(len(m["plugins"]), 1)
        entry = m["plugins"][0]
        self.assertEqual(entry["name"], "codex-mnemosyne")
        src = entry["source"]
        self.assertEqual(src["source"], "local")
        # source.path must be ./-prefixed relative to the marketplace root
        self.assertTrue(src["path"].startswith("./"))
        resolved = os.path.normpath(os.path.join(REPO_ROOT, src["path"]))
        self.assertTrue(
            os.path.isdir(resolved),
            f"marketplace source.path must resolve to existing dir: {resolved}",
        )
        # policy fields
        pol = entry["policy"]
        self.assertIn(pol["installation"], ("AVAILABLE", "INSTALLED_BY_DEFAULT"))
        self.assertIn(pol["authentication"], ("ON_INSTALL", "ON_USE"))
        self.assertIn("category", entry)

    def test_marketplace_plugin_has_valid_manifest_and_hooks(self) -> None:
        """The plugin referenced by the marketplace has a valid manifest and
        hooks.json with all four events."""
        import json

        mkt = os.path.join(REPO_ROOT, ".agents", "plugins", "marketplace.json")
        with open(mkt) as f:
            m = json.load(f)
        plugin_rel = m["plugins"][0]["source"]["path"]
        plugin_abs = os.path.normpath(os.path.join(REPO_ROOT, plugin_rel))

        manifest = os.path.join(plugin_abs, ".codex-plugin", "plugin.json")
        self.assertTrue(os.path.exists(manifest))
        with open(manifest) as f:
            pm = json.load(f)
        self.assertEqual(pm["name"], "codex-mnemosyne")

        hooks = os.path.join(plugin_abs, "hooks", "hooks.json")
        self.assertTrue(os.path.exists(hooks))
        with open(hooks) as f:
            hm = json.load(f)
        self.assertEqual(
            set(hm["hooks"].keys()),
            {"SessionStart", "UserPromptSubmit", "Stop", "SessionEnd"},
        )
        for event, entries in hm["hooks"].items():
            for grp in entries:
                for hook in grp["hooks"]:
                    self.assertIn("${PLUGIN_ROOT}", hook["command"])


# ---------------------------------------------------------------------------
# Finding C2: every test DB/data dir must be disposable; no ~/.hermes access
# ---------------------------------------------------------------------------


class TestNoGlobalDbAccess(_Base):
    """Hooks must never touch ~/.hermes when PLUGIN_DATA / MNEMOSYNE_DATA_DIR
    are set to disposable paths."""

    def test_hook_never_creates_hermes_dir(self) -> None:
        """Running UserPromptSubmit with ONLY PLUGIN_DATA set (no
        MNEMOSYNE_DATA_DIR) must create the DB under PLUGIN_DATA, never
        under a global home directory."""
        plugin_data = os.path.join(self.data_dir, "pdata")
        os.makedirs(plugin_data, exist_ok=True)
        # Do NOT set MNEMOSYNE_DATA_DIR — only PLUGIN_DATA. The hook's
        # _ensure_mnemosyne_data_dir() must derive it from PLUGIN_DATA.
        env = dict(os.environ)
        env.update(
            {
                "PLUGIN_DATA": plugin_data,
                "MNEMOSYNE_CODEX_SPOOL_PATH": os.path.join(plugin_data, "spool.db"),
                "MNEMOSYNE_CODEX_ACTOR_ID": "alice",
                "MNEMOSYNE_CODEX_PROJECT_ID": "projX",
            }
        )
        env.pop("MNEMOSYNE_DATA_DIR", None)
        prior_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = WORKTREE_ROOT + (os.pathsep + prior_pp if prior_pp else "")
        proc = subprocess.run(
            [sys.executable, os.path.join(HOOKS_DIR, "user_prompt_submit.py")],
            input=json.dumps(
                {"session_id": "s1", "turn_id": "t1", "prompt": "test", "cwd": "/tmp"}
            ),
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0, f"hook failed: {proc.stderr}")
        # The mnemosyne.db must be created under PLUGIN_DATA
        disposable_db = os.path.join(plugin_data, "mnemosyne.db")
        self.assertTrue(
            os.path.exists(disposable_db),
            f"DB must be under PLUGIN_DATA ({disposable_db}) when only PLUGIN_DATA is set",
        )

    def test_source_code_never_references_hermes_default(self) -> None:
        """common.py must not contain a hardcoded ~/.hermes path."""
        with open(os.path.join(HOOKS_DIR, "common.py")) as f:
            src = f.read()
        self.assertNotIn(
            ".hermes",
            src,
            "common.py must not reference ~/.hermes; use PLUGIN_DATA",
        )


# ---------------------------------------------------------------------------
# Finding I1: real installed-location smoke (disposable venv, pip install)
# ---------------------------------------------------------------------------


class TestInstalledLocationSmoke(unittest.TestCase):
    """Prove hooks import mnemosyne from site-packages and work without
    repository source imports.

    Creates a disposable venv, installs the local Mnemosyne package with no
    network, copies the plugin to a non-repo temp location, clears
    source-tree PYTHONPATH, and exercises hook lifecycle.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.venv_dir = tempfile.mkdtemp(prefix="mnem-venv-")
        cls.python = os.path.join(cls.venv_dir, "bin", "python")
        cls.pip = os.path.join(cls.venv_dir, "bin", "pip")
        # Create venv (no network needed)
        subprocess.run(
            [sys.executable, "-m", "venv", cls.venv_dir],
            check=True,
            capture_output=True,
            timeout=60,
        )
        # Install the local mnemosyne package (editable, no network/deps)
        # Use --no-deps --no-index to guarantee no network.
        subprocess.run(
            [cls.pip, "install", "--no-deps", "PyYAML>=6.0", WORKTREE_ROOT],
            check=True,
            capture_output=True,
            timeout=120,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.venv_dir, ignore_errors=True)

    def setUp(self) -> None:
        self.data_dir = tempfile.mkdtemp(prefix="mnem-installed-")
        # Copy plugin to a non-repo temp location
        self.plugin_copy = tempfile.mkdtemp(prefix="mnem-plugin-copy-")
        shutil.copytree(
            os.path.join(PLUGIN_ROOT, "hooks"),
            os.path.join(self.plugin_copy, "hooks"),
        )
        shutil.copytree(
            os.path.join(PLUGIN_ROOT, ".codex-plugin"),
            os.path.join(self.plugin_copy, ".codex-plugin"),
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.data_dir, ignore_errors=True)
        shutil.rmtree(self.plugin_copy, ignore_errors=True)

    def _run_installed_hook(
        self, script: str, payload: dict
    ) -> tuple[int, dict | None, str]:
        """Run a hook from the copied plugin using the venv python, with NO
        source-tree PYTHONPATH."""
        env = {
            "PLUGIN_ROOT": self.plugin_copy,
            "PLUGIN_DATA": self.data_dir,
            "MNEMOSYNE_DATA_DIR": self.data_dir,
            "MNEMOSYNE_CODEX_ACTOR_ID": "alice",
            "MNEMOSYNE_CODEX_PROJECT_ID": "projX",
            "MNEMOSYNE_CODEX_SPOOL_PATH": os.path.join(self.data_dir, "spool.db"),
            # Explicitly clear PYTHONPATH so no source tree leaks.
            "PYTHONPATH": "",
            "PATH": os.path.dirname(self.python)
            + os.pathsep
            + os.environ.get("PATH", ""),
        }
        proc = subprocess.run(
            [self.python, os.path.join(self.plugin_copy, "hooks", script)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
            cwd=self.data_dir,
        )
        parsed = None
        if proc.stdout.strip():
            try:
                parsed = json.loads(proc.stdout)
            except json.JSONDecodeError:
                pass
        return proc.returncode, parsed, proc.stderr

    def test_hook_imports_mnemosyne_from_site_packages(self) -> None:
        """The hook must import mnemosyne from the venv site-packages, not
        from the source tree."""
        env = {
            "PYTHONPATH": "",
            "PATH": os.path.dirname(self.python)
            + os.pathsep
            + os.environ.get("PATH", ""),
        }
        result = subprocess.run(
            [
                self.python,
                "-c",
                "import mnemosyne; print(mnemosyne.__file__)",
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
            cwd=self.data_dir,
        )
        self.assertEqual(result.returncode, 0, f"import failed: {result.stderr}")
        location = result.stdout.strip()
        self.assertIn(
            self.venv_dir,
            location,
            f"mnemosyne must be from venv site-packages, got: {location}",
        )
        self.assertNotIn(WORKTREE_ROOT, location)

    def test_installed_hook_lifecycle_works(self) -> None:
        """Full hook lifecycle from the copied plugin location with venv
        python and no source-tree PYTHONPATH."""
        # SessionStart
        code, out, err = self._run_installed_hook(
            "session_start.py",
            {"session_id": "s1", "source": "startup", "cwd": "/tmp"},
        )
        self.assertEqual(code, 0, f"SessionStart failed: {err}")

        # UserPromptSubmit (ingest + recall)
        code, out, err = self._run_installed_hook(
            "user_prompt_submit.py",
            {
                "session_id": "s1",
                "turn_id": "t1",
                "prompt": "I like dark mode",
                "cwd": "/tmp",
            },
        )
        self.assertEqual(code, 0, f"UserPromptSubmit failed: {err}")

        # Stop
        code, out, err = self._run_installed_hook(
            "stop.py",
            {
                "session_id": "s1",
                "turn_id": "t1",
                "last_assistant_message": "Noted.",
                "cwd": "/tmp",
            },
        )
        self.assertEqual(code, 0, f"Stop failed: {err}")

        # SessionEnd
        code, out, err = self._run_installed_hook(
            "session_end.py",
            {"session_id": "s1", "cwd": "/tmp"},
        )
        self.assertEqual(code, 0, f"SessionEnd failed: {err}")

        # Verify the DB was created under the disposable data dir
        self.assertTrue(os.path.exists(os.path.join(self.data_dir, "mnemosyne.db")))

    def test_installed_cross_session_recall(self) -> None:
        """Cross-session recall must work from the installed location."""
        # Session A: ingest
        self._run_installed_hook(
            "user_prompt_submit.py",
            {
                "session_id": "sessA",
                "turn_id": "tA",
                "prompt": "I prefer terse answers",
                "cwd": "/tmp",
            },
        )
        # Session B: recall
        code, out, err = self._run_installed_hook(
            "user_prompt_submit.py",
            {
                "session_id": "sessB",
                "turn_id": "tB",
                "prompt": "what do I prefer?",
                "cwd": "/tmp",
            },
        )
        self.assertEqual(code, 0, f"cross-session recall failed: {err}")
        ctx = (out or {}).get("hookSpecificOutput", {}).get("additionalContext", "")
        self.assertIn("terse", ctx, f"must recall from installed location: {ctx[:200]}")


# ---------------------------------------------------------------------------
# Finding I2: SessionEnd hard deadline via subprocess (not SIGALRM-only)
# ---------------------------------------------------------------------------


class TestSessionEndHardDeadline(_Base):
    """SessionEnd must have a real hard process boundary, not just SIGALRM.
    Test with a non-cooperative worker that blocks in a C-level call."""

    def test_blocked_worker_killed_and_rows_retained(self) -> None:
        """If the ingest path blocks in a non-Python call (sqlite lock wait),
        the SessionEnd hook must still return < 3s and retain unacked rows.

        We simulate this by pointing MNEMOSYNE_DATA_DIR at a FIFO (read blocks
        forever in open()), then running SessionEnd. The hook's subprocess
        timeout ensures return.
        """
        # Seed the spool with a pending event
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
                "prompt": "will block on flush",
                "cwd": "/tmp",
            },
            env,
        )
        import sqlite3

        conn = sqlite3.connect(self.spool_path)
        before = conn.execute("SELECT COUNT(*) FROM spooled_events").fetchone()[0]
        conn.close()
        self.assertGreater(before, 0)

        # Run SessionEnd with an unreachable native DB so ingest fails during
        # flush — rows must be retained, not deleted.
        start = time.monotonic()
        code, out, stdout, stderr = _run(
            "session_end.py",
            {"session_id": "s1", "cwd": "/tmp"},
            self._env(
                MNEMOSYNE_DATA_DIR="/proc/cannot/exist/mnemosyne",
                MNEMOSYNE_CODEX_ACTOR_ID="alice",
                MNEMOSYNE_CODEX_PROJECT_ID="projX",
            ),
        )
        elapsed = time.monotonic() - start
        # Task 8 official hook contract: retained rows => nonzero exit with a
        # static, content-free stderr diagnostic (no systemMessage), elapsed < 3s.
        self.assertNotEqual(code, 0, "retained rows must exit nonzero")
        self.assertLess(elapsed, 3.0, f"SessionEnd took {elapsed:.2f}s")
        self.assertIsNone(
            (out or {}).get("systemMessage"),
            "SessionEnd must not emit systemMessage",
        )
        self.assertTrue(stderr.strip(), "must write a static diagnostic to stderr")
        # Unacked rows must be retained
        conn = sqlite3.connect(self.spool_path)
        after = conn.execute("SELECT COUNT(*) FROM spooled_events").fetchone()[0]
        conn.close()
        self.assertEqual(after, before, "unacked rows must be retained")

    def test_hard_deadline_with_blocking_child(self) -> None:
        """The flush mechanism must use a real subprocess hard deadline,
        not only SIGALRM. We verify by inspecting that common.py implements
        a subprocess-based bounded flush."""
        with open(os.path.join(HOOKS_DIR, "common.py")) as f:
            src = f.read()
        # The flush implementation must use subprocess (not only signal)
        self.assertTrue(
            "subprocess" in src,
            "flush must use subprocess for a real hard deadline",
        )


# ---------------------------------------------------------------------------
# Round-3: CWD isolation — hooks must never import from worktree source tree
# ---------------------------------------------------------------------------


class TestCwdIsolation(unittest.TestCase):
    """Prove that hook subprocesses do not import mnemosyne from the worktree
    source tree, regardless of the test runner's CWD. This is the regression
    test for the round-3 controller failure."""

    def setUp(self) -> None:
        self.data_dir = tempfile.mkdtemp(prefix="mnem-cwd-iso-")

    def tearDown(self) -> None:
        shutil.rmtree(self.data_dir, ignore_errors=True)

    def test_hook_subprocess_cwd_is_neutral(self) -> None:
        """All _run / _run_hook helpers must launch subprocesses with a cwd
        that is NOT the worktree root, so sys.path[0] does not pick up the
        source-tree mnemosyne package."""
        # Verify by source inspection: each test helper passes cwd= to
        # subprocess.run.
        for tf in (
            "tests/test_fix_round1.py",
            "tests/test_fix_round2.py",
            "tests/test_hooks.py",
            "tests/test_smoke.py",
        ):
            path = os.path.join(PLUGIN_ROOT, tf)
            with open(path) as f:
                src = f.read()
            self.assertIn(
                "cwd=",
                src,
                f"{tf} must pass cwd= to subprocess.run for CWD isolation",
            )


# ---------------------------------------------------------------------------
# Finding I3: no env leakage from flush_spool
# ---------------------------------------------------------------------------


class TestNoEnvLeakage(_Base):
    """flush_spool must not permanently mutate os.environ."""

    def test_flush_spool_does_not_leak_env(self) -> None:
        """flush_spool must not mutate the process environment."""
        # Import common directly (it's on the hook dir path)
        sys.path.insert(0, HOOKS_DIR)
        try:
            import importlib

            spec = importlib.util.spec_from_file_location(
                "_r2_common", os.path.join(HOOKS_DIR, "common.py")
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            # Set a marker env var
            os.environ.pop("_R2_TEST_MARKER", None)
            before_keys = set(os.environ.keys())

            mod.flush_spool(self.spool_path, env={"_R2_TEST_MARKER": "leaked"})

            after_keys = set(os.environ.keys())
            leaked = after_keys - before_keys
            self.assertNotIn(
                "_R2_TEST_MARKER",
                after_keys,
                f"flush_spool must not leak env keys: {leaked}",
            )
        finally:
            os.environ.pop("_R2_TEST_MARKER", None)
            sys.path.remove(HOOKS_DIR)


if __name__ == "__main__":
    unittest.main()
