## B06 — Embedding API failures can silently omit vectors

**Status:** `EXISTING` #718/#735; `OPEN_PR` #720/#797.

### Evidence and impact

An isolated API-path probe showed `embeddings.available() == True` while `embed()` returned `None`. Because callers treated `None` as ordinary unavailability, text persisted without vectors and no exception reached the existing warning path. Wrong-count single-item results created the same silent degradation.

Impact: recall silently becomes lexical-only, vector coverage drifts, and operators see successful writes with reduced retrieval quality.

### Suggested solution

Validate API results at the embedding boundary. API mode should raise a content-free typed/runtime error for missing, empty, or wrong-count vectors. Best-effort write/import callers may catch it and persist text, but they must emit a bounded degradation diagnostic or durable retry receipt. PR #720 covers operator warning/persistence; PR #797 covers fail-loud API semantics.

---

## B07 — Query-side vector dimension mismatch crashes recall

**Status:** `EXISTING` #753 / `OPEN_PR` #754.

An isolated connection raised the upstream vec0 dimension-mismatch `OperationalError` through `_vec_search()`. The equivalent write and working-memory query paths already degrade, so the episodic query path is asymmetric.

Impact: one incompatible query embedding aborts the entire recall instead of serving FTS/keyword/importance voices.

Suggested solution: catch the vec0 query `OperationalError` at the episodic vector voice, emit safe reindex/dimension guidance, return an empty vector result, and preserve non-vector recall. PR #754 implements this focused shape.

---

## B08 — SHMR local-LLM call contract mismatch

**Status:** `EXISTING` #716 / `OPEN_PR` #721.

SHMR passed unsupported keyword arguments to a prompt-only local helper, making local inference unreachable. This forced fallback behavior and obscured the configured local-first path.

Suggested solution: call only the supported prompt/system contract, isolate remote fallback in tests, verify the no-system branch used by harmonization, and preserve diagnostic visibility. PR #721 needs rebase and current-head review.

---

## B09 — Polyphonic recall lacks signals consumed by Hermes prefetch

**Status:** `EXISTING` #700 / `OPEN_PR` #701.

Polyphonic results carry RRF `score` and `voice_scores` but omit linear `keyword_score`, `fts_score`, and `dense_score`. Hermes automatic prefetch uses those fields; the signal becomes zero and matching raw transcripts can be dropped.

Impact: explicit recall may find content while silent cross-session prefetch misses it.

Suggested solution: define one normalized provider-facing result contract. Either emit comparable per-signal fields from polyphonic recall or make prefetch consume `voice_scores` without changing core RRF ordering. Add an end-to-end polyphonic-result-through-prefetch regression. PR #701 is the existing implementation path.
