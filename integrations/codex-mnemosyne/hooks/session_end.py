#!/usr/bin/env python3
"""SessionEnd hook for Mnemosyne Codex integration.

Flushes the transport-only spool: attempts to deliver any spooled events via
native ingest and ack-deletes successful ones.  Must complete under 3 seconds
(the hook timeout in hooks.json enforces this).

Fail-open: if flush fails, spooled events remain for the next session's
SessionStart/SessionEnd to retry.  Never blocks session teardown.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def main() -> int:
    # SessionEnd payload may include session_id; we only need to flush.
    _payload = common.read_stdin()

    path = common.spool_path()
    # Best-effort flush; never raises.  Completes in well under 3s because
    # native ingest is a local SQLite write.
    try:
        common.flush_spool(path)
    except Exception:
        # Fail-open: spooled events remain for next session.
        pass

    common.emit_noop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
