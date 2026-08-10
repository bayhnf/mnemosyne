#!/usr/bin/env python3
"""SessionStart hook for Mnemosyne Codex integration.

Fires on startup|resume|clear|compact.  Recalls identity/preferences/project
context via bounded recall and injects it as additionalContext.  Fail-open:
any error produces a visible non-sensitive warning but never blocks the session.
"""

from __future__ import annotations

import os
import sys

# Support both `python3 hooks/session_start.py` and direct execution.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def _identity_query(payload: dict) -> str:
    """Build a recall query biased toward identity/preferences for this session."""
    cwd = payload.get("cwd", "")
    parts = ["user identity preferences project context"]
    if cwd:
        parts.append(cwd)
    return " ".join(parts)


def main() -> int:
    payload = common.read_stdin()
    session_id = str(payload.get("session_id", "unknown"))

    # Bounded recall for identity/preferences: <=6 items, <=800 tokens.
    results, mode, degradation = common.native_recall(
        _identity_query(payload),
        top_k=6,
        max_tokens=800,
        session_id=session_id,
    )

    if mode == "error":
        # Memory is unreachable.  Fail-open with a visible non-sensitive warning.
        common.emit_system_message(common.memory_down_message())
        return 0

    context = common.format_recall_context(results, mode, degradation)
    common.emit_context("SessionStart", context)
    return 0


if __name__ == "__main__":
    sys.exit(main())
