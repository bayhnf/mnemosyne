# Mnemosyne Primary-Memory Contribution Report

**Maintainer-facing engineering report for the `codex/mnemosyne-primary-memory` branch.**
**Upstream baseline:** `mnemosyne-oss/mnemosyne` `main` @ `85a4b88` ("Merge pull request #702 from dplush/fix/690-bank-aware-export"). The prior internal baseline was `f6ba5c8` (#686); `f6ba5c8` is an ancestor of `85a4b88` (17 commits between them).
**Branch HEAD:** `770602a` ("fix(test): resolve cross-test module pollution in hermes_memory_provider tests").
**Divergence base:** `6363a4b` ("fix(cli): normalize MSYS backup destinations (#683)").

This report is derived solely from local source and history in this worktree and from the upstream `main` commit identified above. It contains **no hostnames, usernames, account paths, IP addresses, credentials, real memory content, operational counts/snapshots, or private operational paths**. All evidence labels distinguish *verified-from-source* facts from *proposed* work.

---

## 1. Executive Summary

This branch introduces a coherent **primary-memory subsystem** for Mnemosyne: durable receipt-backed ingest ("Inhale"), hard-bounded read-only recall ("Exhale"), and a reviewer/verifier-gated canonical-mutation lifecycle ("Dream"), plus the supporting infrastructure needed to make them production-credible — disaster-recovery snapshots, content-free diagnostics, a Codex lifecycle integration, and a large hardening/test campaign.

The work is **additive and backward-compatible by construction**. The legacy `remember()` / `recall()` / `sleep()` contract is untouched. Every new subsystem writes to new tables (`ingest_receipts`, `ingest_conflicts`, `dream_runs`, `dream_actions`, `dream_receipts`) created with `CREATE TABLE IF NOT EXISTS` on open, so existing databases acquire the schema without a migration step. No existing table column is widened, narrowed, or dropped.

**Scale of the change:** 112 files changed, 37,271 insertions, 1,114 deletions across 135 commits since the divergence base. The change set introduces four new core modules (`inhale.py`, `recall_bounded.py`, `dream.py`, `dr/snapshot.py`), extends the CLI and MCP tool surface by ~15 native tools, adds a full Codex hook integration (`integrations/codex-mnemosyne/`), and lands ~19,200 new test lines across 51 test files.

**Maturity:** the core lifecycle code is well-instrumented and defensively implemented (content-free failure paths, fail-closed validation, dual-actor receipts, deterministic manifest hashing). The branch carries one on-host evidence harness (`scripts/linuxprocessing_campaign.py`) whose 72-hour soak stage is **explicitly gate-gated and not claimed complete**. A historical local test run of 3180 passed / 12 skipped with one environment-only failure is referenced as *historical local evidence*, **not** as current upstream CI proof.

**Recommendation:** the work decomposes cleanly into independently reviewable PRs along the dependency axis described in §13. The Inhale/Exhale pair and the DR snapshot module are the lowest-risk, highest-value slices and can land first.

---

## 2. Current Upstream Baseline

The current official upstream `main` baseline at the time of this report is `85a4b88` ("Merge pull request #702 from dplush/fix/690-bank-aware-export"), which is 17 commits ahead of the prior internal reference point `f6ba5c8` (#686). The relevant recent upstream history:

- `85a4b88` Merge pull request #702 from dplush/fix/690-bank-aware-export
- `5cd43d3` fix(hermes): preserve explicit default exports
- `2884c45` fix(hermes): export the resolved memory bank
- `f9dc629` Merge pull request #704 from dplush/fix/703-local-model-download-notice
- `f6ba5c8` docs: clarify LLM consolidation configuration (#686)
- `68ec73c` fix(beam): preserve degradation savepoint during vec refresh (#694)
- `6363a4b` fix(cli): normalize MSYS backup destinations (#683)

The divergence base (merge base) between this branch and upstream is `6363a4b`, i.e. the branch forked from upstream shortly after the MSYS backup-destination fix. A rebase/merge before any PR will be required; the conflicts are expected to be minimal because the branch's changes are concentrated in *new files* and *additive functions* rather than edits to the lines touched by the commits above, but the rebase must be verified (see §11 and §13). Each slice should rebase onto the then-current `upstream/main` at PR-open time.

Upstream `main` at `85a4b88` ships version `3.16.0`. This branch does **not** bump `__version__` beyond `3.16.0`; a version bump and `CHANGELOG.md` `[Unreleased]` entry are deferred to the per-PR slicing plan (§13) per the project's Simple Versioning convention.

---

## 3. Inhale / Exhale / Dream Definitions

These three terms are the branch's vocabulary for a primary-memory lifecycle layered on top of the existing BEAM engine. They are implemented as additive modules; none replaces `remember()` / `recall()` / `sleep()`.

### 3.1 Inhale — durable receipt-backed ingest

**Module:** `mnemosyne/core/inhale.py` (1,289 lines, new).
**SQLite tables:** `ingest_receipts`, `ingest_conflicts` (created idempotently in `mnemosyne/core/beam.py` near line 1241).

Inhale wraps the existing `BeamMemory.remember()` write path in a durable, idempotent envelope. The contract:

1. Every ingest event carries a stable `event_id` that serves as an idempotency key.
2. The raw memory row(s), the `ingest_receipts` row, and the sync event commit are written in **one SQLite transaction**.
3. Indexing/enrichment (embeddings, annotations) runs *outside* that transaction; the receipt transitions `pending → ready | degraded | failed_retryable | failed_terminal`.
4. A crash after the raw commit leaves a durable `pending` receipt that `retry_pending_ingest()` can re-run without duplicating memory.
5. `ingest_conflicts` is an append-only audit trail; conflicts are never recorded on the original receipt row, so a conflict does not terminalize a still-retryable original.

Public API (beam-first; `beam` is the dependency-injection seam):
- `remember_event(event)` / `remember_turn(turn)` — single-event / single-turn durable ingest.
- `retry_pending_ingest(limit=100)` — re-runs indexing for non-ready receipts.
- `ingest_status(event_id=None, limit=100)` — returns **content-free** receipts only.

**Invariants:**
- Every `event_id` is processed exactly once (deduplication on `payload_hash`).
- `MAX_ATTEMPTS = 5` bounds retries; after that the receipt terminalizes.
- `MAX_CONTENT_CHARS = 1_000_000`, `MAX_FIELD_CHARS = 255`, `MAX_METADATA_BYTES = 128 KiB` enforce input limits at the trust boundary.
- An `_InhaleTransactionError` is raised before any mutation if a caller-owned or deferred-transaction context is detected, so Inhale never rolls back a transaction it did not open.

### 3.2 Exhale — hard-bounded read-only recall

**Module:** `mnemosyne/core/recall_bounded.py` (1,059 lines, new).
**SQLite tables:** none (read-only over existing tiers).

Exhale is an additive post-hydration gate returning a `RecallEnvelope` with hard result/token caps, strict isolation, and deterministic fallback. It does **not** touch the legacy `recall()` path. Every retrieval mode (linear, enhanced, associative, polyphonic, entity, fact, MEMORIA, episodic supplements) hydrates into one flat candidate list and passes through a single gate:

```
hydrate → strict predicate → lifecycle → deduplicate
        → rank → hard top_k → rendered-token budget
```

Public API:
- `recall_bounded(query, policy=None)` returns a `RecallEnvelope`.

**`RecallPolicy` (frozen dataclass) is the authoritative specification:**
- Hard caps: `top_k` (default 20, must be a positive strict int), `max_tokens`, `max_item_tokens`.
- Identity allowlists (each `None` = unconstrained): `producer_ids`, `actor_ids`, `project_ids`, `session_ids`, `producer_types`, `memory_types`, `veracity`.
- `include_shared`, `include_legacy_shared`, `only_active`, `require_fallback` booleans.
- `from_date` / `to_date` temporal window, `source` filter.

**Invariants:**
- The bounded path is **read-only**: it never bumps `recall_count` / `last_recalled`. It calls only read-only local helpers (`_fts_search`, `_wm_vec_search`, `_find_memories_by_entity`, `_find_memories_by_fact`, `memoria_retrieve`, `fact_recall`, `_fetch_polyphonic_row`), never the side-effecting legacy `recall()` / `recall_enhanced()` / `_recall_polyphonic()`.
- Proposal/Dream sources (`_PROPOSAL_SOURCES = {"sleep_model_refresh_proposal"}`) are always excluded — pending proposals are never surfaced by Exhale.
- `__post_init__` validates every field: bare strings are rejected for sequence allowlists (so `"ab"` is not char-split into `{'a','b'}`), non-positive ints are rejected, bools are type-checked.
- `_is_strict_int` rejects `bool` (since `bool` is an `int` subclass) and `float`.

### 3.3 Dream — reviewer/verifier-gated canonical mutations

**Module:** `mnemosyne/core/dream.py` (1,708 lines, new).
**SQLite tables:** `dream_runs`, `dream_actions`, `dream_receipts` (created idempotently at lines 151–203).

Dream is a small orchestrator over existing Mnemosyne functions; it is **not** a second memory engine. It sources proposals from Task 4's `shmr.propose_harmony` (a read-only candidate generator), builds a deterministic canonical-JSON manifest, persists reviewer/verifier receipts, and — only after a valid dual-PASS receipt — applies the proposed actions atomically inside a single Dream-owned transaction.

**Lifecycle:** `planning → awaiting_approval → ready → applying → applied → undoing → undone`; terminals: `rejected / failed_retryable / failed_terminal`.

Public API (beam-first):
- `dream_plan(beam, scope, limits=None, request_id=None) -> DreamRun`
- `dream_submit_receipt(beam, run_id, receipt) -> DreamRun`
- `dream_apply(beam, run_id) -> DreamRun`
- `dream_resume(beam, run_id) -> DreamRun`
- `dream_undo(beam, run_id) -> DreamRun`
- `dream_status(beam, run_id) -> DreamRun`

**Receipt model:** `RECEIPT_ROLES = ("reviewer", "verifier")`. Transition to `ready` requires a PASS reviewer followed by a PASS verifier with a **different** `actor_id` (no self-approval). Any validation failure or non-PASS receipt transitions the run to a terminal/retryable state with a structured `error_code`.

**Invariants:**
- **Additive tables only.** Planning is deterministic + report-only; sources/proposals stay invisible to recall before apply (Dream never writes `working_memory` / `episodic_memory` / `canonical_facts` before the apply transaction).
- **Deterministic manifest.** A UUID4 `run_id` is persisted on the run, but `manifest_hash` is a SHA-256 over a canonical *semantic* projection that excludes volatile run/timestamp fields, so equivalent inputs hash deterministically. Receipts bind **both** the exact `run_id` AND `manifest_hash`.
- **Apply only from `ready`**, revalidating every persisted source hash immediately before one transaction. Dream owns that transaction: it rejects caller-open / deferred contexts up front via `_deferred_commits`, then uses Beam's seam. Facts are mutated via `DELETE+INSERT` (the FTS triggers have no UPDATE row). Before/after images + audit + run state are committed together; enrichment is left explicitly pending.
- **Undo restores from THIS run's exact before-images only**; a second undo returns `already_undone`; apply/undo are idempotent.
- **`dream_active`** is set before verified apply and cleared on normal ownership end; a stale `true` is reconciled from durable run state. Crash leaves it fail-safe `true` (no auto-apply).
- **Structured error codes, never raw exceptions.** `ERROR_CODES = frozenset({"provider_unavailable", "provider_empty_response", "provider_invalid_output", "embedding_unavailable", "dimension_mismatch", "no_candidates", "no_convergence", "budget_exhausted", "stale_manifest", "validation_failed", "database_busy", "integrity_failure"})`. Expected validation failures never log at ERROR/CRITICAL.

---

## 4. Capability Matrix

| Capability | Legacy path (unchanged) | Native path (this branch) | Surface |
|---|---|---|---|
| Write a memory | `remember()` | `remember_event()` / `remember_turn()` (Inhale) | Python SDK, CLI, MCP, Hermes provider, Codex hooks |
| Recall memories | `recall()` (linear/enhanced/polyphonic) | `recall_bounded()` (Exhale) | Python SDK, CLI, MCP, Codex SessionStart/UserPromptSubmit |
| Retry failed ingest | — | `retry_pending_ingest()` | Python SDK, CLI (`ingest-status`), MCP (`mnemosyne_ingest_retry`) |
| Inspect ingest state | — | `ingest_status()` (content-free) | CLI (`ingest-status`), MCP (`mnemosyne_ingest_status`) |
| Plan canonical mutations | `sleep()` + model-refresh auto-apply (ungated) | `dream_plan()` (report-only) | Python SDK, CLI, MCP (`mnemosyne_dream_plan`) |
| Review/verify plan | — | `dream_submit_receipt()` (dual-actor) | MCP (`mnemosyne_dream_review` / `mnemosyne_dream_verify`) |
| Apply verified plan | model-refresh auto-apply | `dream_apply()` (Dream-owned txn) | Python SDK, CLI, MCP (`mnemosyne_dream_apply`) |
| Undo applied plan | — | `dream_undo()` (before-image restore) | Python SDK, CLI, MCP (`mnemosyne_dream_undo`) |
| Reclaim stale claims | — | `reclaim_orphans()` | CLI (`reclaim-orphans`), MCP (`mnemosyne_reclaim_orphans`) |
| Consolidate | `sleep()` | `sleep()` (unchanged) | unchanged |
| Sync | `SyncEngine` | `SyncEngine` (content-free failure hardening) | unchanged |
| Snapshot/restore | `dr/recovery.create_backup()` | `dr/snapshot.restore_isolated_snapshot()` (page-level) | Python SDK, campaign harness |

---

## 5. Architecture & Invariants

The branch preserves every documented upstream invariant and adds new ones. The key architectural decisions:

**A. Beam is the dependency-injection seam.** Every new native API is beam-first: it accepts the `beam` instance rather than opening its own connection or constructing its own `BeamMemory`. The module-level convenience functions in `mnemosyne/core/memory.py` (lines 1219–1290) are thin wrappers that construct a default beam and delegate. This keeps the new code out of the connection-management business and makes it testable with injected fakes.

**B. Transactions are always owned by the subsystem that mutates.** Inhale, Dream, and the snapshot restore path each reject caller-open or deferred-transaction contexts *before* any mutation ( Inhale: `_InhaleTransactionError`; Dream: explicit reject of caller-open/`_defer_commit`-flagged contexts; snapshot: writer-lock + staged rebuild). This prevents the "cannot start a transaction within a transaction" class of failure reported upstream in #489.

**C. Additive-only schema.** No existing table is altered destructively. New tables use `CREATE TABLE IF NOT EXISTS`. Where columns are added to a new table for older databases (e.g. `claim_worker_id` / `claim_worker_lease` on `ingest_receipts`), the code uses `PRAGMA table_info` detection plus `ALTER TABLE ADD COLUMN`, matching the project's documented "idempotent `init_*` functions run on every open" pattern from `docs/architecture.md`. `PRAGMA foreign_keys=ON` remains off by deliberate upstream decision (issue #503); Dream's two `dream_*` tables are the exception that do declare `FOREIGN KEY (run_id) REFERENCES dream_runs(run_id)`.

**D. Content-free public surfaces.** Every new public API that returns status (ingest receipts, Dream projections, sync diagnostics, snapshot errors) is **content-free**: it returns structured `error_code` / state fields, never raw exception text, file paths, memory content, or SQL fragments. This is enforced at the boundary (`SnapshotError`, `_assert_mode`, the `redact_memory_artifact` filter, and the content-free projection in `mcp_tools.py` error handlers).

**E. Fail-closed validation.** Validation failures fail closed rather than degrading silently: snapshot mode bits are asserted (not just set) with an explicit raise that `python -O` cannot strip; campaign evidence files self-assert content-free before writing; Dream validation failures terminalize the run rather than leaving it applyable.

**F. Deterministic, auditable mutation.** Dream's `manifest_hash` is a SHA-256 over a semantic projection that excludes volatile fields, so the same inputs produce the same hash. Receipts bind both `run_id` and `manifest_hash`, so a receipt cannot be replayed against a different manifest. Undo restores only from the current run's before-images, preventing cross-run contamination.

---

## 6. Detailed Change Inventory

Grouped by subsystem. File paths are relative to the repo root. Line counts are approximate and taken from `git diff --stat 6363a4b...HEAD` (merge base to branch HEAD).

### 6.1 New core modules

| File | LOC | Purpose |
|---|---|---|
| `mnemosyne/core/inhale.py` | +1,289 | Durable receipt-backed ingest (Inhale). |
| `mnemosyne/core/recall_bounded.py` | +1,059 | Hard-bounded read-only recall (Exhale). |
| `mnemosyne/core/dream.py` | +1,708 | Reviewer/verifier-gated canonical-mutation lifecycle (Dream). |
| `mnemosyne/dr/snapshot.py` | +374 | Page-level isolated snapshot + atomic restore. |

### 6.2 Modified core modules

| File | Δ | Purpose |
|---|---|---|
| `mnemosyne/core/beam.py` | +299 | `ingest_receipts` / `ingest_conflicts` schema; `remember_event`/`remember_turn`/`retry_pending_ingest`/`ingest_status`/`recall_bounded` delegation. |
| `mnemosyne/core/shmr.py` | +730/−250 | `propose_harmony` read-only candidate generator for Dream; fact/source alias canonicalization. |
| `mnemosyne/core/memory.py` | +115 | Thin module-level wrappers for the native API. |
| `mnemosyne/core/filters.py` | +157 | Secret/noise detection; `classify_memory_write`; `redact_memory_artifact`. |
| `mnemosyne/core/sync.py` | +107 | Content-free failure paths; empty-embedding rejection; degraded-operation reporting. |
| `mnemosyne/core/sync_server.py` | +42 | Content-free server/apply diagnostics. |
| `mnemosyne/core/config.py` | +5 | `dream_active` config seam (`MNEMOSYNE_DREAM_ACTIVE`, default `False`). |
| `mnemosyne/core/orchestrator.py` | +33 | Compatibility entry point clarification. |
| `mnemosyne/core/plugins.py` | +55 | Module isolation on rediscovery; loose plugin module name handling. |
| `mnemosyne/core/hygiene.py` | +73 | Audit/cleanup hardening. |
| `mnemosyne/core/model_refresh.py` | +12 | NaN-confidence hardening (split from #546). |
| `mnemosyne/migrations/e7_311_tables.py` | +63 | Idempotent 3.11.1 sync-table migration (canonical DDL copied from `sync.py`). |

### 6.3 CLI / MCP / tool surface

| File | Δ | Purpose |
|---|---|---|---|
| `mnemosyne/cli.py` | +895 | `ingest-status`, `reclaim-orphans`, Dream subcommands, content-free error boundaries. |
| `mnemosyne/mcp_tools.py` | +801 | 15 new native tool handlers; content-free error projection. |
| `mnemosyne/tool_schemas.py` | +281 | Schemas for `mnemosyne_ingest`, `mnemosyne_ingest_status`, `mnemosyne_ingest_retry`, `mnemosyne_dream_*`, `mnemosyne_reclaim_orphans`. |
| `mnemosyne/mcp_server.py` | +14 | Tool registration glue. |
| `mnemosyne/batch_tool.py` | +20 | Vector rollback atomicity. |
| `mnemosyne/diagnose.py` | +172 | `ingest_receipts` / `dream_health` projection in `diagnose` output. |
| `mnemosyne/doctor.py` | +636 | `IngestHealthAdapter`, `DreamHealthAdapter`; content-free findings. |

### 6.4 Hermes provider

| File | Δ | Purpose |
|---|---|---|
| `hermes_memory_provider/__init__.py` | +390 | Native API exposure; content-free audit/diagnostics; sync tool error schema. |
| `hermes_memory_provider/audit.py` | +13 | Content-free audit health. |
| `hermes_memory_provider/sync_adapter.py` | +51 | Content-free sync diagnostics. |
| `hermes_memory_provider/persona_adapter.py` | +10 | Persona alignment. |
| `integrations/hermes/src/mnemosyne_hermes/__init__.py` | +409 | Mirror of provider changes for the packaged Hermes integration. |
| `integrations/hermes/src/mnemosyne_hermes/audit.py` | +13 | Mirror. |
| `integrations/hermes/src/mnemosyne_hermes/sync_adapter.py` | +51 | Mirror. |
| `integrations/hermes/tests/test_sync_turn_real_beam.py` | +481 | Real-beam sync-turn receipt tests. |
| `integrations/hermes/tests/test_sync_turn_receipt.py` | +457 | Sync-turn receipt contract tests. |

### 6.5 Codex integration (new)

| File | Purpose |
|---|---|
| `integrations/codex-mnemosyne/.codex-plugin/plugin.json` | Plugin manifest. |
| `integrations/codex-mnemosyne/hooks/common.py` (868 LOC) | Shared import/scope/recall/ingest helpers; 0600 spool. |
| `integrations/codex-mnemosyne/hooks/session_start.py` | Bounded recall on startup/resume/clear/compact; **no ingest**. |
| `integrations/codex-mnemosyne/hooks/user_prompt_submit.py` | Bounded recall before each prompt. |
| `integrations/codex-mnemosyne/hooks/stop.py` | Durable ingest of the turn via Inhale. |
| `integrations/codex-mnemosyne/hooks/session_end.py` | Durable ingest of the session tail. |
| `integrations/codex-mnemosyne/hooks/hooks.json` | Hook registration. |
| `integrations/codex-mnemosyne/README.md` | Setup guide. |
| `integrations/codex-mnemosyne/tests/*` | ~2,900 LOC of hook contract, fix-round, smoke, and task-8 tests. |

### 6.6 Campaign harness (new)

| File | LOC | Purpose |
|---|---|---|
| `scripts/linuxprocessing_campaign.py` | +2,497 | Stdlib-only on-host evidence harness for the G0–G8 campaign. Operates only on an explicitly created trial root; never contacts production; never prints paths/content/credentials. |

### 6.7 Tests

51 test files changed, +19,231 / −133 lines. Highlights:

| File | LOC | Coverage |
|---|---|---|
| `tests/test_recall_bounded.py` | +1,784 | Exhale policy/envelope/fallback/isolation. |
| `tests/test_shmr_dream_proposals.py` | +1,752 | Dream candidate sourcing; manifest determinism. |
| `tests/test_linuxprocessing_campaign.py` | +2,160 | Campaign harness containment and evidence shape. |
| `tests/test_recovery_paths.py` | +844 | DR recovery + snapshot restore. |
| `tests/test_snapshot.py` | +810 | Page-level snapshot + atomic restore + mode assertions. |
| `tests/test_task6_trial_driver.py` | +814 | Trial driver integration. |
| `tests/test_mcp_server.py` | +482 | MCP native tool surface lifecycle. |
| `tests/test_inhale.py` | — | Inhale idempotency, retry, conflict audit. |
| `tests/test_dream_lifecycle.py` / `test_dream_boundary.py` / `test_dream_content_free.py` | — | Dream lifecycle, dual-actor receipts, content-free projection. |
| `tests/test_filters.py` | — | Secret/noise classification. |
| `tests/test_migration_dry_run_fingerprint.py` | +188 | Proves E6/E7/CLI dry runs never write (schema fingerprint + byte-size identical). |

### 6.8 Other

| File | Δ | Purpose |
|---|---|---|---|
| `conftest.py` | +34 | Repo-root Hermes module-cache restore (cross-test isolation). |
| `.github/workflows/ci.yml` | +40 | CI adjustments. |
| `.agents/plugins/marketplace.json` | +20 | Plugin marketplace entry. |
| `docs/api/tool-schema.mdx` | +327 | Native tool documentation. |
| `docs/api/configuration.mdx` | +7 | `dream_active` config doc. |
| `uv.lock` | +138 | Lockfile updates. |
| `setup.py` | +4 | Packaging metadata. |
| `scripts/generate-docs.py` | +2 | Doc generation tweak. |

---

## 7. Overlap Audit with Current Issues / PRs

The branch's work was cross-checked against the 16 open upstream issues summarized in `open_issues_summary.md` (generated 2026-07-19). Issue/PR numbers below refer to the upstream tracker.

| Upstream issue | Title (abbreviated) | Branch relationship |
|---|---|---|
| **#489** | "cannot start a transaction within a transaction" (`task_progress` BEGIN IMMEDIATE) | **Directly addresses.** Inhale and Dream both reject caller-open / deferred-transaction contexts before mutating, eliminating this failure class for the native paths. |
| **#491** | Trim-before-embedding can trigger a foreign-key failure | **Related.** The Inhale path defers enrichment outside the atomic write transaction and records a retryable receipt on enrichment failure, so a trim/concurrent-delete no longer silently orphans an embedding. Does not touch the legacy `remember()` trim ordering. |
| **#482** | 50 of 106 `config.yaml` keys silently ignored | **Adjacent.** The branch does not attempt the module-level-constants refactor, but every *new* config key it adds (`dream_active`) is resolved at request time via `MnemosyneConfig`, not captured at import, so it does not worsen #482. The CHANGELOG `[Unreleased]` entries for `degrade_batch` and BEAM recall weights (already on `main`) are the down-payment on the fix. |
| **#474** | `embeddings.available()` reports True while `vec_episodes` is never created | **Adjacent.** The branch's content-free diagnostics and `doctor` ingest/dream adapters make this class of silent fallback observable, but do not change `embeddings.available()` semantics. |
| **#408** | Referential integrity: missing `PRAGMA foreign_keys=ON` + missing FK on `gists.memory_id` | **Does not address.** The branch respects the existing decision (FK enforcement off; issue #503). Dream's own tables declare FKs but the global pragma stays off. |
| **#434 / #435** | Canonical memory overuse / canonical deletion | **Partially addresses.** Dream provides the gated, reviewable canonical-mutation channel that #434's "canonical should be a curated, durable layer" asks for; the extractor aggressiveness itself is not changed. |
| **#449** | Semantic dirtying and validation-aware recall | **Partially addresses.** Exhale's `veracity` and `only_active` policy fields give recall access to the validation dimension; full dirty-flag propagation is not implemented. |
| **#450** | Session-driven memory activation with meaningful-use reinforcement | **Does not address.** No session-driven recency model. |
| **#446** | Versioned, provenance-aware `config.yaml` migrations | **Does not address.** |
| **#403** | Survive Hermes Update | **Adjacent.** The Codex integration and the native API's idempotent schema-acquire reduce update friction but are not a full "survive update" solution. |
| **#372** | Health check endpoint | **Partially addresses.** `doctor` and `diagnose` now project ingest/dream health content-free; a standalone HTTP health endpoint is not added. |
| **#370** | Bootstrap command to import pre-Mnemosyne history | **Does not address.** |
| **#329** | Memory provider tools sometimes not injected | **Does not address** (upstream-tracked as a Hermes bug). |
| **#327** | Map Hermes gateway identity into provider scoping | **Adjacent.** Exhale's `RecallPolicy` identity allowlists (`producer_ids`, `actor_ids`, `project_ids`, `session_ids`) are the recall-side consumer of gateway identity; the provider-side mapping is not changed. |
| **#326** | Use Hermes `queue_prefetch`/`prefetch` as a real background recall cache | **Does not address** (blocked on a Hermes-side RFC). |
| **#487** | Large Hermes gateway RSS increase on first turn | **Does not address.** |

**No open PR numbers in the upstream range observed on this branch conflict with work already under review.** The branch's commits reference internal task codes (g0–g8, t4, e6, e7) rather than upstream PR numbers, except for the `[Unreleased]` CHANGELOG entries that document fixes already merged into `main` (e.g. #482, #521, #542, #556, #601, #603, #606, #608, #621, #625, #649, #660, #666, #676) — those are not part of the diff and are listed only for context.

---

## 8. Non-Redundant Proposal Set

The work the branch does that is **not** available upstream, grouped by review boundary. Each item is a candidate PR.

**P1 — Inhale (durable receipt-backed ingest).** New module + beam schema + CLI/MCP surface + tests. Dependency: none. Lowest risk, highest value. Addresses the transaction-nesting class of #489 and the silent-embedding-orphan class of #491 for the native path.

**P2 — Exhale (bounded read-only recall).** New module + CLI/MCP surface + tests. Dependency: none (read-only). Can land in parallel with P1.

**P3 — DR isolated snapshot.** `dr/snapshot.py` + recovery enhancements + tests. Dependency: none. Pure addition to the DR subsystem; the snapshot path is explicitly separate from `recovery.create_backup()` (which uses `iterdump()` and loads sqlite-vec).

**P4 — Dream lifecycle.** New module + beam/Dream schema + CLI/MCP surface + tests. Dependency: P1's `ingest_receipts` seam is not required, but Dream's value is reduced without Exhale's bounded recall to preview proposals. Recommend landing after P1+P2.

**P5 — SHMR `propose_harmony` (read-only candidate generator).** Dependency: required by P4 (Dream sources proposals from it). Can land with P4 or just before.

**P6 — Content-free diagnostics + doctor/diagnose adapters.** `doctor.py` / `diagnose.py` ingest/dream health projection. Dependency: P1 + P4 tables exist. Observability backbone for the above.

**P7 — Filters: secret/noise classification + `redact_memory_artifact`.** Dependency: none. Improves the write path for both legacy and native ingest.

**P8 — Sync content-free failure hardening.** `sync.py` / `sync_server.py` / provider adapters. Dependency: none. Small, focused security/observability improvement.

**P9 — Codex integration.** `integrations/codex-mnemosyne/`. Dependency: P1 + P2 (the hooks call Inhale and Exhale). Self-contained plugin with its own tests.

**P10 — Campaign evidence harness.** `scripts/linuxprocessing_campaign.py` + tests. Dependency: P3 (uses snapshot). Operator tooling; not required for runtime.

**P11 — Cross-test isolation conftest.** `conftest.py` repo-root Hermes module-cache restore + the long sequence of `test(isolation)` commits. Dependency: none. Should land early to stabilize CI for all other PRs.

---

## 9. Deliberately Held / Rejected Items

Items the branch **chose not** to do, with reasoning:

- **No `PRAGMA foreign_keys=ON` flip.** Upstream issue #503 documents that it broke tests that intentionally create orphan rows and that an FK on `memory_embeddings` previously caused every embedding insert to fail silently. Dream's own tables declare FKs, but the global pragma is left off by design. *Upgrade path:* revisit after the application-layer integrity checks (`doctor`) are exhaustive.
- **No replacement of `remember()` / `recall()` / `sleep()`.** The native API is additive. The legacy contract is the default. *Upgrade path:* the native API can become the default in a future MAJOR bump after soak evidence.
- **No auto-apply of Dream proposals.** `dream_active` defaults to `False`; apply requires a dual-actor PASS receipt. The model-refresh auto-apply path is unchanged in this branch. *Upgrade path:* model-refresh can be migrated to gate through Dream once Dream has soak evidence.
- **No HTTP health endpoint (#372).** `doctor`/`diagnose` expose the same content-free health, but a network endpoint is out of scope for a local-first project. *Upgrade path:* if added, it belongs behind the existing opt-in sync server.
- **No config.yaml versioned migrations (#446).** The new `dream_active` key is resolved at request time and does not worsen #482, but the branch does not attempt the full module-level-constants refactor. *Upgrade path:* tracked in #446.
- **No session-driven recency (#450) or full semantic dirtying (#449).** Exhale exposes the policy hooks (`veracity`, `only_active`) but does not implement the reinforcement model. *Upgrade path:* behind the Exhale policy seam.
- **No Codex built-in memory disable automation.** The Codex plugin manifest states Mnemosyne is the sole persistent provider, but disabling Codex's built-in memory is an operator setup step documented in the README, not an automated action.

---

## 10. Schema / Migration / Backward Compatibility

**New tables (all `CREATE TABLE IF NOT EXISTS`, idempotent on every open):**

| Table | Created in | Purpose |
|---|---|---|
| `ingest_receipts` | `beam.py` (~L1241) | Inhale durable receipt per event_id. |
| `ingest_conflicts` | `beam.py` (immediately after) | Append-only conflict audit trail. |
| `dream_runs` | `dream.py` (~L151) | Dream run state + manifest hash. |
| `dream_actions` | `dream.py` (~L170) | Proposed actions per run (FK to `dream_runs`). |
| `dream_receipts` | `dream.py` (~L188) | Reviewer/verifier receipts (FK to `dream_runs`, unique on `(run_id, role)`). |

**Column additions to new tables for older databases:** `ingest_receipts.claim_worker_id` and `claim_worker_lease` are added via `PRAGMA table_info` detection + `ALTER TABLE ADD COLUMN` so a pre-existing `ingest_receipts` row set acquires the columns without a separate migration step.

**E7 migration (`mnemosyne/migrations/e7_311_tables.py`):** idempotent `CREATE TABLE IF NOT EXISTS` for `memory_events` and `sync_meta`, with DDL copied verbatim from `sync.py`'s `_init_events_table`. The migration is safe to re-run and does not delete data. `tests/test_migration_dry_run_fingerprint.py` proves the dry-run path leaves the schema fingerprint (`sqlite_master` rows + `PRAGMA user_version`) and on-disk byte size identical, including for a WAL-mode bank with a committed seed write.

**Backward compatibility:**
- An existing database opened by this branch acquires the five new tables on open with no operator action.
- The legacy `remember()` / `recall()` / `sleep()` / sync paths are byte-for-byte unchanged in behavior; the content-free sync hardening only changes *log/error* shape, not wire protocol.
- The `MnemosyneConfig` schema gains one key (`dream_active`, default `False`); no existing key is removed or renamed.
- Export/import schema version is unchanged at `1.3`.

**Upgrade note:** because the branch does not bump `__version__` past `3.16.0`, a consumer updating to a build containing this work will not see a version signal. Each sliced PR (§13) should bump the MINOR per Simple Versioning and add a CHANGELOG entry.

---

## 11. Failure Semantics / Observability

**Failure semantics by subsystem:**

- **Inhale:** atomic write-or-not. A crash after the raw commit leaves a `pending` receipt; `retry_pending_ingest()` is idempotent. After `MAX_ATTEMPTS = 5` the receipt terminalizes to `failed_terminal`. Conflicts are appended to `ingest_conflicts` and never terminalize the original. Input limits (`MAX_CONTENT_CHARS`, `MAX_FIELD_CHARS`, `MAX_METADATA_BYTES`) are enforced at the trust boundary.
- **Exhale:** read-only and deterministic. On any internal failure it degrades to lexical FTS5-only if `require_fallback=True`, otherwise raises. It never partially mutates state (it never mutates state at all).
- **Dream:** every failure path persists a structured `error_code` from `ERROR_CODES`. Apply fails closed if any source hash changed since planning (`stale_manifest`). Undo is idempotent and scoped to the current run. `dream_active` is fail-safe `true` on crash (no auto-apply).
- **Snapshot:** `SnapshotError` is raised for every validation/IO failure with a content-free message; `__cause__` is preserved for diagnostics. Mode bits are asserted (not just set). Restore reuses `recovery`'s staged atomic-replace (writer lock → staged rebuild → fsync → atomic replace → post-replace integrity check with rollback).
- **Sync:** client transport, TLS config, server, apply, and per-item embedding failures all produce content-free structured diagnostics. Empty per-item embeddings are rejected. Degraded embedding operations are reported, not hidden.

**Observability surface:**
- `mnemosyne doctor` gains `IngestHealthAdapter` and `DreamHealthAdapter` projecting content-free metrics: ingest `index_status` distribution, non-terminal Dream run count, terminal run counts by `error_code`.
- `mnemosyne diagnose` projects `ingest_receipts.status` and `dream_health.non_terminal_runs`.
- The MCP `mnemosyne_ingest_status` and `mnemosyne_dream_status` tools return content-free structured results.
- All new public failure surfaces return `error_code` strings, never raw exception text.

---

## 12. Security / Privacy

The branch strengthens Mnemosyne's security posture in three ways and introduces no new network egress.

**A. Content-free public surfaces.** Ingest receipts, Dream projections, snapshot errors, sync diagnostics, and MCP error handlers all return structured `error_code` / state fields. `SnapshotError` messages explicitly never contain file paths, SQL fragments, or byte data. `_assert_mode` raises with a content-free message. This closes a class of information-leakage via error messages identified in `SECURITY.md` ("Credential exposure in logs or error messages").

**B. Write-path secret/noise filtering.** `mnemosyne/core/filters.py` adds `classify_memory_write` with curated `SECRET_PATTERNS` (API keys, tokens, passwords, private keys) and `DEFAULT_NOISE_PATTERNS`. In `strict` mode a `secret_detected` decision rejects the write; in `warn` mode it proceeds with decision metadata. `MNEMOSYNE_WRITE_CLASSIFIER` controls the mode (`off` default, preserving legacy behavior). `redact_memory_artifact` provides the redaction primitive used by the content-free paths.

**C. Fail-closed file permissions.** Snapshot artifacts are created `0600` / directories `0700`, and the mode is *asserted* immediately after the chmod/fchmod call (using an explicit `raise`, not `assert`, so `python -O` cannot strip it). The campaign harness creates/verifies directories `0700` and files `0600` with a self content-free assertion before writing.

**D. No new network egress.** The branch adds no outbound network calls. The Codex integration is a local hook plugin that calls the in-process Mnemosyne SDK. The campaign harness is explicitly network-free and SSH-free.

**E. Containment of the campaign harness.** `scripts/linuxprocessing_campaign.py` resolves and verifies every filesystem/subprocess input to be contained under an explicitly created trial root; symlink escape and path traversal are rejected before any action. It never accepts a production path as an argument, never prints paths/content/credentials, and exits `2` (manual gate pending) rather than `0` for any unacknowledged prerequisite.

**Privacy note:** the Codex integration hooks Codex's lifecycle (SessionStart/UserPromptSubmit/Stop/SessionEnd) to Mnemosyne ingest/recall. SessionStart performs **no ingest** (it only recalls), so it never claims an event was queued. The failed-delivery spool is `0600` transport-only with ack-based deletion, and the hooks never parse transcripts.

---

## 13. Testing Matrix and Sanitized Deployment Evidence

**Test matrix (verified from `git diff --stat`):** 51 test files changed, +19,231 / −133 lines. The new tests cover:

- Inhale idempotency, retry, conflict audit, content limits, transaction-context rejection.
- Exhale policy validation (strict-int, bool, sequence-of-strings), hard caps, read-only invariant, deterministic fallback, proposal-source exclusion.
- Dream lifecycle transitions, dual-actor receipt enforcement, manifest determinism, apply/undo idempotency, stale-manifest detection, content-free error projection.
- SHMR `propose_harmony` candidate generation and alias canonicalization.
- DR snapshot page-level copy, atomic restore, mode-bit assertion, WAL sidecar handling.
- Migration dry-run fingerprint invariance (schema + byte size).
- MCP native tool surface lifecycle and content-free error handlers.
- Codex hook contracts, smoke, and fix-round tests.
- Campaign harness containment and evidence shape.
- Cross-test isolation (Hermes module-cache restore, embeddings/plugin teardown).

**Historical local test evidence (clearly labeled, NOT current upstream proof):** a local test run on this branch produced **3180 passed, 12 skipped, with one environment-only failure**. This is *historical local evidence* from the contributor's environment and is **not** a representation of current upstream CI status. It is referenced here only to indicate the branch was exercised locally before handoff. Upstream CI must re-run independently.

**Deployment / soak evidence (interim, not complete):**

- The branch includes a stdlib-only on-host evidence harness (`scripts/linuxprocessing_campaign.py`) for a G0–G8 campaign. The harness is gate-gated: every stage requires an explicit operator acknowledgement flag (`--ack-t0-ssh`, `--ack-image-digest`, `--ack-snapshot-approved`, `--ack-writer-quiesce`, `--ack-fault-strategy`, `--ack-codex-desktop`, `--ack-hermes-smoke`, `--ack-soak-schedule`). A missing ack fails closed to exit `2`.
- **The 72-hour soak stage (G7) is explicitly not claimed complete.** The harness enforces `--ack-soak-schedule` as a manual gate; absence exits `2` and no code path can set PASS for an unacknowledged soak. Performance/soak numbers must therefore be treated as **proposed/interim** until the soak ack is recorded and the evidence is reviewed.
- No performance numbers, memory-content counts, operational snapshots, or operational reports are included in this document, by design.

---

## 14. Performance / Soak Evidence (Interim)

**No soak completion is claimed.** The soak stage of the campaign harness is gate-gated and has not been acked. Any performance characteristics of the native API should be considered **proposed** until:

1. The 72-hour soak is run with `--ack-soak-schedule` recorded.
2. Resource-measurement evidence (truthful `getrusage` reporting, enforced by the `fix(g7)` commits) is reviewed.
3. The G8 comparison evidence (snapshot-restore-based clone source, enforced fail-closed) is reviewed.

**Structural performance expectations (from code inspection, not measurement):**
- Inhale's atomic transaction is the same write that `remember()` already performs, plus one `ingest_receipts` insert in the same transaction. The enrichment runs outside the transaction, so p99 write latency should be comparable to legacy `remember()` plus a single indexed insert.
- Exhale is read-only and hydrates through the same read helpers as legacy recall; the added cost is the post-hydration gate (predicate → dedup → rank → hard cap → token budget), which is O(n) in the candidate count.
- Dream planning is report-only and sources from `shmr.propose_harmony`, whose cost is bounded by `shmr_batch_size` (default 50) and `shmr_max_iterations` (default 10). Apply is a single transaction over DELETE+INSERT per action.

These expectations are **not measurements** and must not be cited as such.

---

## 15. Rollback

**Per-subsystem rollback:**

- **Inhale:** disabling the native ingest path and reverting to legacy `remember()` is a no-op for existing data — legacy recall reads the same `working_memory` / `episodic_memory` rows. The `ingest_receipts` / `ingest_conflicts` tables can be left in place (they are inert without the native API). To drop them: `DROP TABLE IF EXISTS ingest_conflicts; DROP TABLE IF EXISTS ingest_receipts;` (order matters for the Dream FK if Dream is also installed; Dream does not reference these tables, so order is safe).
- **Exhale:** read-only; rolling back means callers stop calling `recall_bounded()` and use `recall()`. No data change.
- **Dream:** `dream_active` defaults to `False`; setting it false (or removing the config key) disables the apply gate. Dream terminals (`rejected`, `failed_terminal`) are self-healing. To roll back an *applied* Dream run, use `dream_undo(run_id)` which restores from the run's before-images. To drop the tables: `DROP TABLE IF EXISTS dream_receipts; DROP TABLE IF EXISTS dream_actions; DROP TABLE IF EXISTS dream_runs;` (FK order).
- **Snapshot:** `dr/snapshot.py` is a new module; removing it has no effect on existing backups created by `recovery.create_backup()`.
- **Codex integration:** removing `integrations/codex-mnemosyne/` and uninstalling the plugin reverts Codex to its built-in memory. No Mnemosyne data is lost.
- **Config:** removing the `dream_active` key from `config.yaml` restores the default (`False`).
- **Full branch revert:** `git revert` the PR range, or `git checkout 6363a4b -- .` on the affected paths (merge base). Because the schema additions are `CREATE TABLE IF NOT EXISTS`, a reverted binary will simply stop creating the new tables; existing new tables remain inert and can be dropped manually.

**No destructive migration is performed at any point**, so every rollback path is recoverable.

---

## 16. Dependency-Ordered Issue / PR Plan

Recommended landing order, earliest first. Each item is a PR; dependencies are to *earlier* PRs in the list.

1. **P11 — Cross-test isolation conftest.** No runtime dependency. Stabilizes CI for everything below. *Files:* `conftest.py`, the `test(isolation)` commits.
2. **P7 — Filters: secret/noise classification + redaction.** No runtime dependency. Improves both legacy and native write paths. *Files:* `mnemosyne/core/filters.py`, `tests/test_filters.py`.
3. **P8 — Sync content-free failure hardening.** No runtime dependency. *Files:* `mnemosyne/core/sync.py`, `mnemosyne/core/sync_server.py`, provider adapters, `tests/test_sync*.py`.
4. **P1 — Inhale (durable receipt-backed ingest).** Depends on: nothing. *Files:* `mnemosyne/core/inhale.py`, `beam.py` changes, CLI/MCP/tool-schema, `tests/test_inhale.py`.
5. **P2 — Exhale (bounded read-only recall).** Depends on: nothing. *Files:* `mnemosyne/core/recall_bounded.py`, CLI/MCP/tool-schema, `tests/test_recall_bounded.py`.
6. **P3 — DR isolated snapshot.** Depends on: nothing. *Files:* `mnemosyne/dr/snapshot.py`, `mnemosyne/dr/recovery.py`, `tests/test_snapshot.py`, `tests/test_recovery_paths.py`.
7. **P6 — Content-free diagnostics + doctor/diagnose adapters.** Depends on: P1 tables, P4 tables (can stub Dream projections until P4 lands). *Files:* `doctor.py`, `diagnose.py`.
8. **P5 — SHMR `propose_harmony`.** Depends on: nothing (read-only). *Files:* `mnemosyne/core/shmr.py`, `tests/test_shmr_dream_proposals.py`.
9. **P4 — Dream lifecycle.** Depends on: P5 (candidate source). Recommend P1+P2 landed first for end-to-end coherence. *Files:* `mnemosyne/core/dream.py`, CLI/MCP/tool-schema, `tests/test_dream_*.py`.
10. **P9 — Codex integration.** Depends on: P1 (Inhale), P2 (Exhale). *Files:* `integrations/codex-mnemosyne/**`.
11. **P10 — Campaign evidence harness.** Depends on: P3 (snapshot). Operator tooling. *Files:* `scripts/linuxprocessing_campaign.py`, `tests/test_linuxprocessing_campaign.py`.

**Versioning:** each PR bumps the MINOR per Simple Versioning and adds a CHANGELOG `[Unreleased]` entry. Breaking-change call-outs (none currently expected) get a MAJOR bump.

---

## 17. Contributor License Agreement (CLA)

All contributions represented by this branch are submitted under the project's [Individual Contributor License Agreement](CLA.md) (adapted from the Apache Software Foundation's Individual CLA, last updated 2026-07-13).

By submitting any PR derived from this branch, the contributor confirms:

1. Each contribution is the contributor's original creation.
2. The contributor is legally entitled to grant the license.
3. If the contributor's employer has rights to intellectual property that the contributor creates, the contributor has received permission to make contributions on behalf of that employer, or the employer has waived such rights.
4. Contributions include complete details of any third-party license or other restriction of which the contributor is personally aware.

The CLA grants the project a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable license to reproduce, prepare derivative works of, publicly display, publicly perform, sublicense, and distribute the contributions, and to re-license them under any license chosen by the project (currently MIT). The contributor retains all ownership rights; the CLA is a license, not an assignment.

Past contributions made before 2026-07-13 remain under the MIT License and are not affected.

---

*End of report. Generated from local source and history in the `codex/mnemosyne-primary-memory` worktree against upstream `main` baseline `85a4b88` (merge base `6363a4b`). No hostnames, usernames, account paths, IP addresses, credentials, real memory content, operational counts, snapshots, reports, or private operational paths are included. Historical local test evidence and soak status are clearly labeled as interim/proposed.*
