# Privacy, Security, and Evidence Posture

**Baseline:** `85a4b88`. **Merge base:** `6363a4b`. **Branch HEAD:** `770602a`.

This document states the content-free surface contract, the security posture,
and — conservatively — the test/evidence status. No claim here is stated as
current upstream CI proof unless explicitly labeled historical.

---

## Content-free surface contract

Every new public API that returns status or failure does so with a structured
`error_code` / state field, never raw exception text, file paths, SQL
fragments, or memory content. This is enforced at each boundary:

- **Inhale receipts:** `ingest_status()` returns content-free receipts only.
- **Dream projections:** `dream_status()` returns state + structured
  `error_code` from a closed `ERROR_CODES` frozenset.
- **Snapshot errors:** `SnapshotError` messages never contain file paths or
  byte data; `__cause__` is preserved for diagnostics.
- **Sync diagnostics:** client/server/apply/embedding failures produce
  content-free structured diagnostics.
- **MCP error handlers:** project to a closed vocabulary
  (e.g. `tool_call_failed`, `validation_error`).
- **Batch tool:** stable codes (`batch_validation_failed`, `batch_failed`)
  with an `_ALLOWED_BATCH_ACTIONS` allowlist.

The privacy boundary is independently testable: canary tests inject synthetic
detail and assert it is absent from serialized results and rendered log
records (including `exc_text`).

## Security posture

**A. Write-path secret/noise filtering.** `filters.py` adds
`classify_memory_write` with curated `SECRET_PATTERNS` (API keys, tokens,
passwords, private keys) and `DEFAULT_NOISE_PATTERNS`. In `strict` mode a
`secret_detected` decision rejects the write; in `warn` mode it proceeds with
decision metadata. `MNEMOSYNE_WRITE_CLASSIFIER` controls the mode (`off`
default, preserving legacy behavior). `redact_memory_artifact` is the
redaction primitive used by the content-free paths.

**B. Fail-closed file permissions.** Snapshot artifacts are created `0600` /
directories `0700`, and the mode is *asserted* immediately after the
chmod/fchmod call (using an explicit `raise`, not `assert`, so `python -O`
cannot strip it).

**C. No new network egress.** The branch adds no outbound network calls. The
Codex integration is a local hook plugin that calls the in-process Mnemosyne
SDK. No new network dependencies are introduced.

**D. Additive-only schema.** No existing table is altered destructively. New
tables use `CREATE TABLE IF NOT EXISTS`. `PRAGMA foreign_keys=ON` remains off
by deliberate upstream decision (issue #503); Dream's own tables are the
exception that declare `FOREIGN KEY (run_id) REFERENCES dream_runs(run_id)`.

**E. Transactions are always owned by the mutating subsystem.** Inhale,
Dream, and the snapshot restore path each reject caller-open or
deferred-transaction contexts *before* any mutation. This prevents the
"cannot start a transaction within a transaction" class (#489) on the native
paths.

## Privacy note (Codex integration)

The Codex integration hooks the Codex lifecycle (SessionStart /
UserPromptSubmit / Stop / SessionEnd) to Mnemosyne ingest/recall. SessionStart
performs **no ingest** (it only recalls), so it never claims an event was
queued. The failed-delivery spool is `0600` transport-only with ack-based
deletion, and the hooks never parse transcripts.

---

## Evidence status (conservative)

**No claim of current upstream CI pass is made.** Upstream CI must re-run
independently for every slice.

**Historical local test evidence (labeled, not current proof):** a local test
run on this branch produced a large number of passes with a small number of
skips and a small number of environment-only failures (e.g. an
`EMBEDDING_DIM` mismatch between the test fixture and the host, and an MCP
API-version mismatch in tooling, neither of which is a code defect in this
branch). This is *historical local evidence* and is **not** a representation
of current upstream CI status. It is referenced only to indicate the branch
was exercised locally before handoff.

**Soak evidence:** no soak completion is claimed. The on-host soak stage is
gate-gated in the (excluded) trial harness and has not been acked. Any
performance characteristics of the native API should be considered
**proposed** until a reviewed soak completes. No performance numbers, memory
counts, or operational snapshots are included in this package, by design.

**Structural performance expectations (from code inspection, not
measurement):**

- Inhale's atomic transaction is the same write `remember()` already performs,
  plus one `ingest_receipts` insert in the same transaction. Enrichment runs
  outside the transaction, so p99 write latency should be comparable to legacy
  `remember()` plus a single indexed insert.
- Exhale is read-only and hydrates through the same read helpers as legacy
  recall; the added cost is the post-hydration gate (predicate → dedup → rank
  → hard cap → token budget), O(n) in candidate count.
- Dream planning is report-only and sources from `propose_harmony`, bounded by
  `shmr_batch_size` and `shmr_max_iterations`. Apply is a single transaction
  over DELETE+INSERT per action.

These expectations are **not measurements** and must not be cited as such.

## Rollback

Every subsystem is recoverable:

- **Inhale/Exhale:** disabling the native path reverts to legacy; the
  `ingest_*` tables are inert without the native API and can be dropped with
  `DROP TABLE IF EXISTS`.
- **Dream:** `dream_active` defaults to `False`. An applied run can be undone
  with `dream_undo(run_id)` (before-image restore, idempotent, scoped to the
  run). Dream tables drop in FK order.
- **Snapshot:** `dr/snapshot.py` is a new module; removing it has no effect on
  existing `recovery.create_backup()` backups.
- **Codex integration:** removing `integrations/codex-mnemosyne/` and
  uninstalling the plugin reverts Codex to its built-in memory. No Mnemosyne
  data is lost.
- **Config:** removing the `dream_active` key restores the default (`False`).
- **Full branch revert:** `git revert` the PR range. Because schema additions
  are `CREATE TABLE IF NOT EXISTS`, a reverted binary simply stops creating
  the new tables; existing new tables remain inert and can be dropped manually.

**No destructive migration is performed at any point**, so every rollback
path is recoverable.

---

## Correction note on evidence claims

A prior internal audit cited specific live database row counts, an "error
count since restart: 0" figure, specific memory IDs, and a specific SHA match
against a deployment candidate. Those are **operational details from one host
at one time** and are deliberately **not** carried into this package. They
are not reliable upstream evidence and are excluded by the content-free
contract. See [work-report.md](work-report.md) for the full list of
exclusions.
