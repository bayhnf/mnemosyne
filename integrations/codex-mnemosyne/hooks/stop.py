#!/usr/bin/env python3
"""Stop hook for Mnemosyne Codex integration.

Fires at the end of each assistant turn. Durably ingests the acknowledged
last assistant message as a stable event, scoped to the actor+project memory
scope so it persists across Codex sessions. Does NOT parse the transcript —
it uses only the last_assistant_message field from the hook payload.

Uses payload.turn_id when nonempty, with a deterministic content fallback
only if absent. Fail-open: on memory failure, the event is spooled and the
hook exits 0. Stop hooks are advisory and must never block.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def main() -> int:
    payload = common.read_stdin()
    actor = common.actor_id()
    project = common.project_id(payload.get("cwd", ""))
    scope = common.memory_scope(actor, project)
    message = str(payload.get("last_assistant_message", ""))

    if not message.strip():
        common.emit_noop()
        return 0

    host_turn_id = str(payload.get("turn_id") or "").strip()
    turn_id = host_turn_id if host_turn_id else common.fallback_turn_id(scope, message)

    importable, _err = common._import_mnemosyne()
    if not importable:
        common.emit_system_message(common.message_package_absent())
        return 0

    event_id = common.stable_event_id(scope, turn_id, "assistant")
    event = {
        "event_id": event_id,
        "producer": common.PRODUCER,
        "actor_id": actor,
        "project_id": project,
        "scope": scope,
        "turn_id": turn_id,
        "role": "assistant",
        "content": message,
        "metadata": {"codex_session_id": str(payload.get("session_id", ""))},
    }
    _outcome, spool_status = common.ingest_or_spool(event)

    if spool_status == "stored":
        common.emit_system_message(common.message_ingest_queued())
    elif spool_status == "full":
        common.emit_system_message(common.message_queue_full())
    elif spool_status == "error":
        common.emit_system_message(common.message_ingest_not_queued())
    else:
        common.emit_noop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
