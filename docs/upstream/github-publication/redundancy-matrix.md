# Redundancy Matrix — Upstream Issue / PR Overlap

**Baseline:** `85a4b88`. **Merge base:** `6363a4b`. **Branch HEAD:** `770602a`.

This matrix states, for each relevant open upstream issue, whether this branch
**addresses**, is **adjacent to**, or **does not address** it. "Addresses"
means the branch's code prevents the failure class on the *native* path;
"adjacent" means the branch improves observability of the class but does not
change the legacy path that triggers it. No open upstream PR in the observed
range conflicts with work on this branch.

Issue numbers refer to the upstream tracker. The overlap assessment is
evidence-backed from source; where a relationship is inferential, it is marked.

---

## Addresses (native path prevents the failure class)

| Issue | Title (abbreviated) | Relationship | Evidence |
|---|---|---|---|
| **#489** | "cannot start a transaction within a transaction" | **Addresses (native path).** Inhale and Dream both reject caller-open / deferred-transaction contexts *before* any mutation (`_InhaleTransactionError`; Dream's explicit `_deferred_commits` reject), eliminating this failure class for the native write/apply paths. | `inhale.py` transaction-context guard; `dream.py` ownership reject. |
| **#491** | Trim-before-embedding can trigger a foreign-key failure | **Related (native path).** Inhale defers enrichment outside the atomic write transaction and records a retryable receipt on enrichment failure, so a trim/concurrent-delete no longer silently orphans an embedding *on the native path*. The legacy `remember()` trim ordering is untouched. | `inhale.py` receipt state machine; enrichment-outside-transaction. |

## Adjacent (improves observability, does not change the trigger)

| Issue | Title (abbreviated) | Relationship | Evidence |
|---|---|---|---|
| **#482** | ~50 of 106 `config.yaml` keys silently ignored | **Adjacent.** The branch does not attempt the module-level-constants refactor. Every *new* config key it adds (`dream_active`) is resolved at request time via `MnemosyneConfig`, not captured at import, so it does not worsen #482. | `config.py` `dream_active` seam. |
| **#474** | `embeddings.available()` reports True while `vec_episodes` is never created | **Adjacent.** Content-free diagnostics and `doctor` ingest/dream adapters make the silent-fallback class observable; `embeddings.available()` semantics are unchanged. | `doctor.py` adapters; `diagnose.py` projections. |
| **#434 / #435** | Canonical memory overuse / canonical deletion | **Partially addresses.** Dream provides the gated, reviewable canonical-mutation channel #434 asks for; extractor aggressiveness itself is not changed. | `dream.py` lifecycle. |
| **#449** | Semantic dirtying and validation-aware recall | **Partially addresses.** Exhale's `veracity` and `only_active` policy fields give recall access to the validation dimension; full dirty-flag propagation is not implemented. | `recall_bounded.py` `RecallPolicy`. |
| **#327** | Map gateway identity into provider scoping | **Adjacent.** Exhale's `RecallPolicy` identity allowlists (`producer_ids`, `actor_ids`, `project_ids`, `session_ids`) are the recall-side consumer of gateway identity; provider-side mapping is unchanged. | `recall_bounded.py` allowlists. |
| **#372** | Health check endpoint | **Partially addresses.** `doctor`/`diagnose` now project ingest/dream health content-free; a standalone HTTP health endpoint is not added. | `doctor.py`, `diagnose.py`. |
| **#403** | Survive Hermes Update | **Adjacent.** The Codex integration and the native API's idempotent schema-acquire reduce update friction; not a full "survive update" solution. | `integrations/codex-mnemosyne/`. |

## Does not address

| Issue | Title (abbreviated) | Why not |
|---|---|---|
| **#408** | Referential integrity: missing `PRAGMA foreign_keys=ON` + missing FK on `gists.memory_id` | The branch respects the existing decision (FK enforcement off; issue #503). Dream's own tables declare FKs, but the global pragma stays off. |
| **#450** | Session-driven memory activation with meaningful-use reinforcement | No session-driven recency model. Exhale exposes hooks but no reinforcement. |
| **#446** | Versioned, provenance-aware `config.yaml` migrations | Out of scope. |
| **#370** | Bootstrap command to import pre-Mnemosyne history | Out of scope. |
| **#329** | Memory provider tools sometimes not injected | Upstream-tracked Hermes bug; not touched. |
| **#326** | Use Hermes `queue_prefetch` as a real background recall cache | Blocked on a Hermes-side RFC. |
| **#487** | Large Hermes gateway RSS increase on first turn | Not touched. |

---

## No conflict with open PRs

No open upstream PR numbers in the observed range conflict with work already
under review on this branch. The branch's commits reference internal task
codes, not upstream PR numbers, except for `[Unreleased]` CHANGELOG entries
that document fixes already merged into `main` — those are not part of the
diff and are listed only for context.

## Correction note

A prior internal report listed the upstream baseline as `f6ba5c8`. The current
official baseline is `85a4b88`, which is ahead of `f6ba5c8`. The merge base
between this branch and upstream remains `6363a4b`. Slicing should rebase onto
the then-current `upstream/main` at PR-open time. See
[work-report.md](work-report.md) for the baseline correction detail.
