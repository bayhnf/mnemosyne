## Existing, fixed, and in-progress findings

These are not duplicate bug reports. They connect our sanitized deployment evidence to the official tracker and show where we can contribute focused fixes.

<details>
<summary><strong>Recovery, hygiene, and transaction correctness — B03/B04/B05/B17</strong></summary>

- **B03:** online SQLite backup was required because SQL dumps cannot faithfully restore FTS/vec tables. Multi-store JSON import can also commit partial state before a later failure. Official: #640, #727, PR #815. Suggested shape: online backup; staged checksum/integrity-checked atomic restore; one transaction or staged-clone publication for JSON import.
- **B04:** one malformed hygiene candidate must not roll back successful siblings or corrupt counters. PR #816 uses per-candidate savepoints, content-free errors, and idempotent restore; it needs current-main rebase/review.
- **B05:** successful consolidation summaries containing `fail` can be miscounted as errors. Official: #717 / PR #719. Suggested shape: combine error-like text with the zero-item failure predicate and cover boundary rows.
- **B17:** caller-open/shared transactions can trigger nested `BEGIN IMMEDIATE` failures. #489 was fixed by merged PR #501. Future mutation paths should own their transaction or reject caller-open contexts before mutation.

</details>

<details>
<summary><strong>Embeddings, SHMR, and recall signals — B06/B07/B08/B09/B10</strong></summary>

- **B06:** embedding API may report available but return no/wrong-count vectors, leaving text without vector coverage. Official: #718/#735; PRs #720/#797. Validate at the embedding boundary and surface bounded degradation/retry state.
- **B07:** episodic vec0 dimension mismatch aborts all recall. Official: #753 / PR #754. Degrade only the vector voice and preserve FTS/keyword/importance recall.
- **B08:** SHMR passed unsupported arguments to the local helper. Official: #716 / PR #721. Use the supported prompt/system contract and no-network fallback tests.
- **B09:** polyphonic results omit signals consumed by Hermes prefetch. Official: #700 / PR #701. Normalize the provider-facing score contract or consume `voice_scores` explicitly.
- **B10:** configured remote LLM can still trigger local model loading/download. Official: #688. Bypass local discovery/download when remote intent is explicit.

</details>

<details>
<summary><strong>Consolidation, lifecycle age, scope, and multilingual hygiene — B11/B12/B13/B14</strong></summary>

- **B11:** deployment concurrency exercised all-session timeout, shared SQLite/vec races, and unrelated-session auto-sleep. Official history: #342/#498/#687 and PRs #349/#520/#772. Existing focused serialization/session-scoping is the correct solution; no duplicate lease issue is proposed.
- **B12:** a newly written summary of old sources appears fresh. Official: #564. Preserve source-time separately from derivation-time and use source-time for recency.
- **B13:** session/project identity mapping can make valid indexed memories invisible to the intended reader. Official: #327/#601/#653; PR #604. Map identity once, persist explicit scope, and fail closed when project context is absent.
- **B14:** CJK acknowledgements/prose/secret labels exposed hygiene gaps. Official: #806; merged PR #810 fixed secret labels. Recheck only the remaining noise/value and Unicode punctuation lane before any focused PR.

</details>

### Contribution plan

We will add useful evidence to the original issues, not open duplicates. Focused PRs should reference the official issue and `Addresses Bxx in this tracker`; they must not close this tracker.
