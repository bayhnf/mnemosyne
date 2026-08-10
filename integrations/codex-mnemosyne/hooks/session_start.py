#!/usr/bin/env python3
"""SessionStart hook for Mnemosyne Codex integration.

Fires on startup|resume|clear|compact. Recalls identity/preferences/project
context via bounded recall using the deterministic memory scope (actor+project)
and injects it as additionalContext. Fail-open: any error produces a visible
non-sensitive warning but never blocks the session.

SessionStart performs NO ingest, so it never claims an event was queued.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def _identity_query(payload: dict) -> str:
    cwd = payload.get("cwd", "")
    parts = ["user identity preferences project context"]
    if cwd:
        parts.append(cwd)
    return " ".join(parts)


def main() -> int:
    payload = common.read_stdin()
    actor = common.actor_id()
    project = common.project_id(payload.get("cwd", ""))
    scope = common.memory_scope(actor, project)

    importable, err = common._import_mnemosyne()
    if not importable:
        common.emit_system_message(common.message_package_absent())
        return 0

    results, mode, _deg = common.native_recall(
        _identity_query(payload),
        top_k=6,
        max_tokens=800,
        scope=scope,
        actor=actor,
        project=project,
    )

    if mode == "error":
        common.emit_system_message(common.message_recall_unavailable())
        return 0

    context = common.format_recall_context(results, mode)
    common.emit_context("SessionStart", context)
    return 0


if __name__ == "__main__":
    sys.exit(main())
