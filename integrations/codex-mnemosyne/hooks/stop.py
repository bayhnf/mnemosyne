#!/usr/bin/env python3
"""Stop hook for Mnemosyne Codex integration.

Fires at the end of each assistant turn.  Durably ingests the acknowledged
last assistant message as a stable event.  Does NOT parse the transcript —
it uses only the last_assistant_message field from the hook payload.

Fail-open: on memory failure, the event is spooled and the hook exits 0.
Stop hooks are advisory and must never block.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def main() -> int:
    payload = common.read_stdin()
    session_id = str(payload.get("session_id", "unknown"))
    message = str(payload.get("last_assistant_message", ""))
    cwd = payload.get("cwd", "")

    if not message.strip():
        common.emit_noop()
        return 0

    turn_id = common.stable_turn_id(session_id, message)
    event_id = common.stable_event_id(session_id, turn_id, "assistant")

    event = {
        "event_id": event_id,
        "producer": common.PRODUCER,
        "actor_id": common.actor_id(),
        "project_id": common.project_id(cwd),
        "session_id": session_id,
        "turn_id": turn_id,
        "role": "assistant",
        "content": message,
    }

    common.ingest_or_spool(event)
    # Stop hooks are advisory — emit nothing and exit 0.
    common.emit_noop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
