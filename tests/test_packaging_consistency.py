"""Static packaging-consistency coverage (Python 3.10-compatible, stdlib only).

Pins the mcp>=2.0.0 lower bound across pyproject.toml / setup.py / uv.lock,
the Hermes base dependency without the embeddings extra, and native Windows
commands for every Codex hook. Parsing is deliberately static (string/regex
plus json) so this suite runs on Python 3.10 without tomllib.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERMES_PYPROJECT = ROOT / "integrations" / "hermes" / "pyproject.toml"
HOOKS_JSON = ROOT / "integrations" / "codex-mnemosyne" / "hooks" / "hooks.json"


def _all_hook_definitions(manifest):
    for entries in manifest["hooks"].values():
        for entry in entries:
            yield from entry["hooks"]


def test_hermes_base_dependency_does_not_require_embeddings_extra():
    metadata = HERMES_PYPROJECT.read_text(encoding="utf-8")
    assert "mnemosyne-memory[embeddings]" not in metadata
    assert '"mnemosyne-memory>=' in metadata


def test_codex_plugin_declares_windows_commands_for_every_hook():
    manifest = json.loads(HOOKS_JSON.read_text(encoding="utf-8"))
    commands = list(_all_hook_definitions(manifest))
    assert commands
    assert all(command["command"].startswith("python3 ") for command in commands)
    assert all(
        command.get("commandWindows", "").startswith("python ") for command in commands
    )


def test_mcp_lower_bound_matches_every_packaging_surface():
    pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    setup_text = (ROOT / "setup.py").read_text(encoding="utf-8")
    lock_text = (ROOT / "uv.lock").read_text(encoding="utf-8")

    assert "mcp>=2.0.0; python_version >= '3.10'" in pyproject_text
    assert setup_text.count("mcp>=2.0.0") >= 2
    assert re.search(
        r'(?ms)^\[\[package\]\]\nname = "mcp"\nversion = "2\.',
        lock_text,
    )
