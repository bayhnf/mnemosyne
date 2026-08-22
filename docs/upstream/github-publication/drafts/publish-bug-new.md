## Newly verified findings

The sections are collapsed to keep this tracking issue scannable.

<details>
<summary><strong>B01 — Concurrent fresh-database initialization can race schema guards</strong></summary>

**Status:** `VERIFIED_NEW` on upstream `4c6b280`; no semantic duplicate found across 315 issues and 500 PRs.

**Discovery:** Our Hermes-integrated setup can initialize the same Mnemosyne database from multiple provider processes. We added a per-database initialization lock after the setup exposed this first-open risk.

**Reproduction:** `_add_column_if_missing()` performs `PRAGMA table_info` followed by `ALTER TABLE`. A sanitized two-connection barrier forced both workers to observe the missing `author_id` column before either ALTER. Across 20 fresh databases:

```text
worker A: ok
worker B: OperationalError: duplicate column name: author_id
author_id_count=1
integrity=ok
REPRODUCED=20/20
```

**Impact:** one concurrently starting provider, MCP server, or worker can fail initialization and continue without memory; readiness can disagree across processes.

**Root cause:** non-atomic check-before-ALTER schema reconciliation.

**Suggested fix:** serialize migrations at the database boundary, re-read schema after acquiring the lock, apply only still-missing changes, and treat duplicate-column as benign only after validating the final column contract. Propagate unrelated SQLite errors.

**Acceptance:** two-thread and two-process first-open tests; one final column; integrity `ok`; idempotent existing-DB startup; unrelated failures remain visible.

We are willing to submit one focused concurrency PR.
</details>

<details>
<summary><strong>B15 — Batch mutation failures leak exception text and tracebacks</strong></summary>

**Status:** `VERIFIED_NEW` on upstream `4c6b280`; no semantic duplicate found.

**Reproduction:** a synthetic batch operation raised `RuntimeError("PRIVATE_CANARY")`. `apply_beam_batch()` returned the exception class/message in `payload.error`; `logger.exception()` also retained the canary and traceback.

**Impact:** MCP/tool callers and logs can receive private backend details, paths, or content; this violates the static content-free contract already merged for CLI boundaries.

**Suggested fix:** project validation/execution failures to `batch_validation_failed` / `batch_failed`; retain only safe index and allowlisted action; replace traceback logging with content-free structured logging; preserve rollback and post-commit audit behavior.

**Acceptance:** canary absent from payload/log; no traceback; unknown actions not reflected; whole batch rolls back; audit events publish only after commit.

We are willing to submit one focused shared-boundary PR.
</details>

<details>
<summary><strong>B16 — SHMR import violates the supported no-NumPy base-install contract</strong></summary>

**Status:** `VERIFIED_NEW` on upstream `4c6b280`. PR #31 established core base-install support without NumPy but did not cover SHMR.

**Reproduction:** blocking NumPy imports and importing `mnemosyne.core.shmr` with embeddings disabled raises `ModuleNotFoundError`; `shmr.py` imports NumPy unconditionally.

**Impact:** a supported minimal install can fail merely by importing optional SHMR; degraded/non-dense behavior is unreachable.

**Scope correction:** dense clustering/vector APIs may keep NumPy as a runtime requirement. The bug is unconditional module import, not a claim that all dense SHMR must become NumPy-free.

**Suggested fix:** guard the NumPy import, keep annotations unevaluated, support scalar/list math only where degradation needs it, and raise one explicit capability error when a dense-only API is invoked.

**Acceptance:** module import succeeds without NumPy; degraded path is reachable; dense-only calls fail clearly; normal NumPy behavior remains unchanged; fresh base-install subprocess test passes.

We are willing to submit one focused optional-dependency PR.
</details>
