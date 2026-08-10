#!/usr/bin/env python3
"""UserPromptSubmit hook for Mnemosyne Codex integration.

Fires on every user prompt.  Does three things, in order:
  1. Durably ingests the user prompt as a stable event (idempotent by event_id).
  2. Calls bounded recall for relevant context.
  3. Injects the context as additionalContext.

Fail-open: on memory failure, the event is spooled and a visible non-sensitive
warning is emitted.  Never blocks the prompt.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def main() -> int:
    payload = common.read_stdin()
    session_id = str(payload.get("session_id", "unknown"))
    prompt = str(payload.get("prompt", ""))
    cwd = payload.get("cwd", "")

    if not prompt.strip():
        common.emit_noop()
        return 0

    turn_id = common.stable_turn_id(session_id, prompt)
    event_id = common.stable_event_id(session_id, turn_id, "user")

    event = {
        "event_id": event_id,
        "producer": common.PRODUCER,
        "actor_id": common.actor_id(),
        "project_id": common.project_id(cwd),
        "session_id": session_id,
        "turn_id": turn_id,
        "role": "user",
        "content": prompt,
    }

    ingest_outcome = common.ingest_or_spool(event)

    # Bounded recall for context relevant to this prompt: <=8 items, <=1200 tokens.
    results, mode, degradation = common.native_recall(
        prompt,
        top_k=8,
        max_tokens=1200,
        session_id=session_id,
    )

    if mode == "error" and not ingest_outcome.ok:
        # Both ingest (spooled) and recall failed — memory is down.
        common.emit_system_message(common.memory_down_message())
        return 0

    context = common.format_recall_context(results, mode, degradation)
    common.emit_context("UserPromptSubmit", context)
    return 0


if __name__ == "__main__":
    sys.exit(main())
