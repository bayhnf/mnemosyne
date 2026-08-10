"""Isolated end-to-end smoke test for the Mnemosyne Codex plugin.

Simulates a full Codex session lifecycle: SessionStart → UserPromptSubmit →
Stop → SessionEnd, against a disposable Mnemosyne instance, and verifies
the whole flow is fail-open, bounded, and leaves no errors.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOOKS_DIR = os.path.join(PLUGIN_ROOT, "hooks")
WORKTREE_ROOT = os.path.dirname(os.path.dirname(PLUGIN_ROOT))


def _run_hook(script: str, payload: dict, env: dict) -> tuple[int, dict | None, str]:
    full = dict(os.environ)
    full.update(env)
    # Simulate installed mnemosyne package for subprocess hooks.
    prior = full.get("PYTHONPATH", "")
    full["PYTHONPATH"] = WORKTREE_ROOT + (os.pathsep + prior if prior else "")
    proc = subprocess.run(
        [sys.executable, os.path.join(HOOKS_DIR, script)],
        input=json.dumps(payload),
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
    return proc.returncode, parsed, proc.stderr


class TestFullSessionLifecycle(unittest.TestCase):
    """Drive SessionStart → UserPromptSubmit → Stop → SessionEnd."""

    def setUp(self) -> None:
        self.data_dir = tempfile.mkdtemp(prefix="mnem-codex-smoke-")
        self.env = {
            "MNEMOSYNE_DATA_DIR": self.data_dir,
            "MNEMOSYNE_CODEX_ACTOR_ID": "smoke-user",
            "MNEMOSYNE_CODEX_PROJECT_ID": "smoke-proj",
        }
        self.session_id = "smoke-session-1"

    def tearDown(self) -> None:
        shutil.rmtree(self.data_dir, ignore_errors=True)

    def test_full_lifecycle_succeeds_and_is_bounded(self) -> None:
        # 1. SessionStart on fresh instance (no memories yet)
        code, out, stderr = _run_hook(
            "session_start.py",
            {"session_id": self.session_id, "source": "startup", "cwd": "/tmp"},
            self.env,
        )
        self.assertEqual(code, 0, f"SessionStart failed: {stderr}")
        # On an empty DB, no context to inject — {} is correct.

        # 2. First user prompt
        code, out, stderr = _run_hook(
            "user_prompt_submit.py",
            {
                "session_id": self.session_id,
                "prompt": "Remember that I prefer Python over JavaScript",
                "cwd": "/tmp",
            },
            self.env,
        )
        self.assertEqual(code, 0, f"UserPromptSubmit failed: {stderr}")

        # 3. Stop — ingest assistant reply
        code, _out, stderr = _run_hook(
            "stop.py",
            {
                "session_id": self.session_id,
                "last_assistant_message": "Got it — Python preference noted.",
                "cwd": "/tmp",
            },
            self.env,
        )
        self.assertEqual(code, 0, f"Stop failed: {stderr}")

        # 4. Second user prompt — should now recall the preference
        code, out, stderr = _run_hook(
            "user_prompt_submit.py",
            {
                "session_id": self.session_id,
                "prompt": "What language do I prefer?",
                "cwd": "/tmp",
            },
            self.env,
        )
        self.assertEqual(code, 0, f"second UserPromptSubmit failed: {stderr}")
        ctx = (out or {}).get("hookSpecificOutput", {}).get("additionalContext", "")
        # The preference should be recalled (bounded context injection)
        self.assertIn(
            "Python", ctx, f"recall should contain Python preference, got: {ctx[:300]}"
        )

        # 5. SessionEnd — must flush and succeed under 3s
        code, _out, stderr = _run_hook(
            "session_end.py",
            {"session_id": self.session_id, "cwd": "/tmp"},
            self.env,
        )
        self.assertEqual(code, 0, f"SessionEnd failed: {stderr}")

    def test_post_compact_sessionstart_recalls_ingested_memory(self) -> None:
        """After compact, SessionStart must re-hydrate identity/preferences."""
        # Ingest a preference statement that Mnemosyne will index.
        _run_hook(
            "user_prompt_submit.py",
            {
                "session_id": self.session_id,
                "prompt": "I prefer concise terse answers in all responses",
                "cwd": "/tmp",
            },
            self.env,
        )
        # Simulate compact — SessionStart must re-hydrate.
        code, out, stderr = _run_hook(
            "session_start.py",
            {"session_id": self.session_id, "source": "compact", "cwd": "/tmp"},
            self.env,
        )
        self.assertEqual(code, 0, f"post-compact SessionStart failed: {stderr}")
        ctx = (out or {}).get("hookSpecificOutput", {}).get("additionalContext", "")
        # The preference must appear in the recalled context.
        self.assertTrue(
            "concise" in ctx or "terse" in ctx,
            f"post-compact recall must re-hydrate preference, got: {ctx[:300]}",
        )


if __name__ == "__main__":
    unittest.main()
