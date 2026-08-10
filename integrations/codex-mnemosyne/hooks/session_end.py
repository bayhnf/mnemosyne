#!/usr/bin/env python3
"""SessionEnd hook for Mnemosyne Codex integration.

Flushes the transport-only spool: attempts to deliver any spooled events via
native ingest and ack-deletes successful ones, under a hard wall-clock deadline.

Official Codex hook contract (Task 8):
  - systemMessage is NOT supported for SessionEnd. Output is advisory only.
  - A command error (nonzero exit) is reported by Codex as a hook failure.
  - Retained spool rows must never be deleted or represented as delivered.
  - Error text must be static and content-free (no exception text, ids,
    content, hashes, scope, or paths).

Behavior:
  - Successful empty/fully-acknowledged flush: exit 0.
  - Bounded flush retains a pending or terminal row: finish before 3s, retain
    rows, write one static content-free diagnostic to stderr, exit nonzero.
    Never emit an unsupported systemMessage.
  - Flush raises: write a distinct static content-free "status unavailable"
    diagnostic to stderr, exit nonzero. Never expose exception text.

Must return before the 3-second Codex ceiling even if native ingest is slow or
hung. Uses a subprocess hard deadline. Unacknowledged rows are always retained.
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
        _flushed, retained_pending, retained_terminal = common.flush_spool_bounded(
            path, budget_s=_SESSION_END_BUDGET_S
        )
    except Exception:
        # Flush itself raised. Static, content-free diagnostic; nonzero exit
        # so Codex reports the hook failure. Never expose exception text.
        # Rows are untouched (whatever the child already committed stands).
        sys.stderr.write(common.diag_session_end_unavailable())
        return 1

    # Trust the actual spool state over the child's reported counts: if the
    # child was killed mid-flush (subprocess timeout), it may report (0, 0, 0)
    # even though rows remain. Any retained row — pending or terminal — must
    # be surfaced, never represented as delivered.
    #
    # Distinguish three outcomes via strict inspection (spool_inspect_state):
    #   - flush reported retained rows, OR inspectable spool still has rows
    #     => retained diagnostic + nonzero (rows not delivered);
    #   - spool is absent/empty and inspectable (count == 0, inspectable)
    #     => genuine success, exit 0;
    #   - spool exists but is corrupt/uninspectable (inspectable == False)
    #     => distinct "unavailable" diagnostic + nonzero. A corrupt spool must
    #     NEVER be coerced to success.
    count, inspectable = common.spool_inspect_state(path)
    if retained_pending > 0 or retained_terminal > 0 or count > 0:
        # Rows were retained (not delivered). Static, content-free stderr;
        # nonzero exit. No unsupported systemMessage.
        sys.stderr.write(common.diag_session_end_retained())
        return 1
    if not inspectable:
        # Spool exists but cannot be inspected (corrupt / not SQLite). Distinct
        # static "status unavailable" diagnostic; nonzero exit. Never expose
        # raw SQLite text, paths, or content.
        sys.stderr.write(common.diag_session_end_unavailable())
        return 1

    # Successful empty/fully-acknowledged flush. Emit an empty JSON object so
    # callers expecting a JSON line on stdout still parse cleanly; this is
    # advisory and contains no systemMessage.
    common.emit_noop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
