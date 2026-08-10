"""Contract tests for the Mnemosyne Codex plugin manifest and hooks manifest.

These verify the plugin.json and hooks.json are well-formed and declare exactly
the lifecycle hooks the brief requires — before any production code exists.
"""

from __future__ import annotations

import json
import os
import unittest

PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_json(rel: str) -> dict | list:
    with open(os.path.join(PLUGIN_ROOT, rel), encoding="utf-8") as fh:
        return json.load(fh)


class TestPluginManifest(unittest.TestCase):
    """The .codex-plugin/plugin.json manifest is upstream-ready."""

    def setUp(self) -> None:
        self.manifest = _load_json(os.path.join(".codex-plugin", "plugin.json"))

    def test_name_is_codex_mnemosyne(self) -> None:
        self.assertEqual(self.manifest["name"], "codex-mnemosyne")

    def test_required_top_level_fields(self) -> None:
        for key in ("name", "version", "description", "license"):
            self.assertIn(key, self.manifest, f"missing required field: {key}")

    def test_hooks_are_discovered_not_declared_in_manifest(self) -> None:
        """Codex discovers hooks/hooks.json without a manifest hooks field."""
        self.assertNotIn("hooks", self.manifest)

    def test_default_prompt_is_present_and_concise(self) -> None:
        """Plugin ingestion requires one to three short starter prompts."""
        prompts = self.manifest["interface"]["defaultPrompt"]
        self.assertIsInstance(prompts, list)
        self.assertGreaterEqual(len(prompts), 1)
        self.assertLessEqual(len(prompts), 3)
        for prompt in prompts:
            self.assertIsInstance(prompt, str)
            self.assertTrue(prompt.strip())
            self.assertLessEqual(len(prompt), 128)

    def test_no_tools_block(self) -> None:
        """The plugin must not declare MCP tools — Mnemosyne is the sole memory
        provider and hooks are the only integration surface."""
        self.assertNotIn("tools", self.manifest)

    def test_no_mcp_servers(self) -> None:
        """No MCP server block; hooks-only integration per the brief."""
        self.assertNotIn("mcpServers", self.manifest)


class TestHooksManifest(unittest.TestCase):
    """hooks.json declares exactly the four required lifecycle hooks."""

    def setUp(self) -> None:
        self.hooks = _load_json(os.path.join("hooks", "hooks.json"))["hooks"]

    def test_required_hook_events_present(self) -> None:
        for event in ("SessionStart", "UserPromptSubmit", "Stop", "SessionEnd"):
            self.assertIn(event, self.hooks, f"missing hook event: {event}")

    def test_session_start_covers_compact(self) -> None:
        """SessionStart must fire after compact, not just startup."""
        matchers = self.hooks["SessionStart"][0].get("matcher", "")
        self.assertIn("compact", matchers, "SessionStart matcher must include compact")

    def test_all_hooks_use_command_type(self) -> None:
        for event_name, entries in self.hooks.items():
            for entry in entries:
                for hook in entry["hooks"]:
                    self.assertEqual(
                        hook["type"],
                        "command",
                        f"{event_name} hook must be command type",
                    )

    def test_session_end_has_timeout_under_3s(self) -> None:
        """SessionEnd must complete under 3 seconds."""
        entry = self.hooks["SessionEnd"][0]
        timeout = entry["hooks"][0].get("timeout")
        self.assertIsNotNone(timeout, "SessionEnd must declare an explicit timeout")
        self.assertLessEqual(timeout, 3, "SessionEnd timeout must be <= 3 seconds")

    def test_all_hook_commands_reference_python(self) -> None:
        """All hooks invoke python3 scripts — no node/jq dependency."""
        for event_name, entries in self.hooks.items():
            for entry in entries:
                for hook in entry["hooks"]:
                    cmd = hook["command"]
                    self.assertIn(
                        "python3", cmd, f"{event_name} must use python3, got: {cmd}"
                    )


if __name__ == "__main__":
    unittest.main()
