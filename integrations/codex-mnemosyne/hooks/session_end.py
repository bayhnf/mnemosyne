#!/usr/bin/env python3
"""SessionEnd hook for Mnemosyne Codex integration.

Flushes the transport-only spool: attempts to deliver any spooled events via
native ingest and ack-deletes successful ones, under a hard wall-clock deadline.

Must return before the 3-second Codex ceiling even if native ingest is slow or
hung. Uses a SIGALRM-based bounded flush (falling back to a between-row
deadline) so the hook does NOT rely solely on Codex forcibly killing it.
Unacknowledged rows are always retained.

Fail-open: if flush leaves pending events, they remain for the next session.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

# Bounded budget: comfortably under the 3s Codex SessionEnd ceiling.
_SESSION_END_BUDGET_S = 2.0


def main() -> int:
    _payload = common.read_stdin()

    path = common.spool_path()
    try:
        flushed, retained_pending, retained_terminal = common.flush_spool_bounded(
            path, budget_s=_SESSION_END_BUDGET_S
        )
    except Exception:
        # Fail-open: spooled events remain for next session.
        common.emit_noop()
        return 0

    # Honest, content-free status. SessionEnd output is advisory (does not
    # steer Codex), but we surface the condition visibly.
    if retained_pending > 0 or retained_terminal > 0:
        common.emit_system_message(common.message_session_end_pending())
    else:
        common.emit_noop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
