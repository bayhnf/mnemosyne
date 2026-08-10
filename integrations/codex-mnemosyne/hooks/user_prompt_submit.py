#!/usr/bin/env python3
"""UserPromptSubmit hook for Mnemosyne Codex integration.

Fires on every user prompt. Does three things, in order:
  1. Durably ingests the user prompt as a stable event (idempotent by
     event_id, scoped to the actor+project memory scope so it persists
     across Codex sessions).
  2. Calls bounded recall for relevant context using the same scope.
  3. Injects the context as additionalContext.

Uses payload.turn_id (host turn id) when nonempty, with a deterministic
content fallback only if absent. Fail-open: on memory failure the event is
spooled and a visible non-sensitive warning is emitted. Never blocks.
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
    prompt = str(payload.get("prompt", ""))

    # Host turn id is the stable identity for this turn when supplied.
    host_turn_id = str(payload.get("turn_id") or "").strip()
    turn_id = host_turn_id if host_turn_id else common.fallback_turn_id(scope, prompt)

    importable, _err = common._import_mnemosyne()

    if prompt.strip() and importable:
        event_id = common.stable_event_id(scope, turn_id, "user")
        event = {
            "event_id": event_id,
            "producer": common.PRODUCER,
            "actor_id": actor,
            "project_id": project,
            "scope": scope,
            "turn_id": turn_id,
            "role": "user",
            "content": prompt,
            # Non-recall provenance only: the ephemeral Codex session id is
            # recorded in metadata, never used as the recall key.
            "metadata": {"codex_session_id": str(payload.get("session_id", ""))},
        }
        _outcome, spool_status = common.ingest_or_spool(event)

        if spool_status == "stored":
            common.emit_system_message(common.message_ingest_queued())
            return 0
        if spool_status in ("full",):
            common.emit_system_message(common.message_queue_full())
            return 0
        if spool_status == "error":
            common.emit_system_message(common.message_ingest_not_queued())
            return 0
    elif prompt.strip() and not importable:
        common.emit_system_message(common.message_package_absent())
        return 0

    if not importable:
        common.emit_system_message(common.message_package_absent())
        return 0

    results, mode, _deg = common.native_recall(
        prompt or _identity_query(payload),
        top_k=8,
        max_tokens=1200,
        scope=scope,
        actor=actor,
        project=project,
    )

    if mode == "error":
        common.emit_system_message(common.message_recall_unavailable())
        return 0

    context = common.format_recall_context(results, mode)
    common.emit_context("UserPromptSubmit", context)
    return 0


def _identity_query(payload: dict) -> str:
    cwd = payload.get("cwd", "")
    parts = ["user identity preferences project context"]
    if cwd:
        parts.append(cwd)
    return " ".join(parts)


if __name__ == "__main__":
    sys.exit(main())
