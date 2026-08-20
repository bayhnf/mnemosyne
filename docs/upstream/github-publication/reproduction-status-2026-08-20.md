# Reproduction Status

**Baseline:** `upstream/main` @ `7ec6f4c`
**Environment:** isolated worktree; `TZ=UTC`; `MNEMOSYNE*` and `NVIDIA_EMBEDDING*` unset; live Mnemosyne untouched.

## Verified

### #806 — CJK hygiene detection

Current behavior:
- `好的`, `收到，已处理`, `知道了`, `没问题`, and `已完成` score `(0.0, [])` and are kept as non-noise.
- A Chinese-labelled secret-shaped value returns no detected secret.
- A long Chinese log block scores `(0.65, ['likely_dump'])`, a false positive from ASCII-only sentence counting.

Smallest solution: add deterministic CJK noise/value markers, count CJK sentence punctuation (`。！？；`), and recognize Chinese secret labels plus full-width separators. Add regression tests.

### #727 — Partial JSON import restore

Source audit confirms `Mnemosyne.import_from_file()` commits BEAM data, legacy memories, and legacy embeddings separately before importing other stores. An injected failure in a later store can leave earlier rows committed. The public import boundary has no single rollback transaction spanning all stores.

Smallest safe solution: stage the import into a temporary SQLite database and atomically publish only after every store import succeeds; alternatively, require one caller-owned connection for every store and one transaction. Do not add compensating delete logic. Inject failures at each store boundary and assert the target remains unchanged.

### #735 — Embedding API returns `None` silently

With a custom non-OpenAI endpoint configured, `embeddings.available()` returns `True`, while an API transport failure causes `embeddings.embed(['probe'])` to return `None` without raising. Reproduced with an isolated mocked transport; no network or live database used.

Smallest solution: preserve the `None` compatibility path but emit one content-free warning/diagnostic when availability is true and an embed attempt returns `None`. Better: return a typed result/error code and persist an observable degraded receipt. Add a regression test.

### #753 — Vector dimension mismatch crashes recall

The current `_vec_search()` path executes the sqlite-vec KNN query without catching `sqlite3.OperationalError`. An isolated connection stub raising the documented dimension-mismatch error propagates `OperationalError` to the caller.

Smallest solution: catch the dimension-mismatch error at `_vec_search()`, log existing content-free dimension guidance, and return `[]` so FTS/keyword/importance recall continues. Add a regression test.

## Verified reproduction: #688

Source inspection confirms `mnemosyne/core/shmr.py:_call_llm()` calls `_call_local_llm()` before the remote path. `_load_llm()` checks `LLM_ENABLED` but does not short-circuit when `LLM_BASE_URL` is configured, so a missing local GGUF can trigger `_download_model()` even when a remote backend is the configured intent.

Smallest solution: if `LLM_BASE_URL` is set, skip local model loading and let the remote path run. Add a test asserting `_download_model()` is not called when a remote base URL is configured.

## Verified reproduction: #700

Source inspection confirms `_recall_polyphonic()` is a separate result path, while Hermes prefetch reads `keyword_score`, `fts_score`, and `dense_score`. If polyphonic results omit those fields, the prefetch topical signal defaults to zero and raw transcript rows can be filtered out despite a matching `score`/`voice_scores` result.

Smallest solution: emit the same per-signal fields from the polyphonic path or centralize result normalization before provider filtering. Add a regression test that passes a polyphonic result through the prefetch gate.

## Verified reproduction: #687

Source inspection confirms the Hermes auto-sleep worker selects `sleep_all_sessions()` when the capability exists. That broad operation is unsafe for a per-session lifecycle trigger in a shared database.

Smallest solution: auto-sleep must calculate eligibility for the triggering session and call `sleep()`; retain `sleep_all_sessions()` for explicit/manual maintenance. Add a deterministic two-session regression test.

## Existing local proof

The primary-memory focused slice passed `378 passed, 1 skipped`.

## Candidate inventory still pending

`#813 #783 #718 #707 #700 #688 #687 #682 #656 #640 #635 #602 #578 #573 #560 #559 #552 #548 #537 #523 #506 #487 #434`

Enhancements and proposals still pending: `#790 #789 #784 #766 #761 #732 #724 #715 #712 #695 #661 #651 #598 #586 #543 #514 #450 #449 #446 #403 #372 #370 #327 #326`, plus roadmap slices S2–S9.

## Next step

Reproduce remaining candidates in small deterministic batches. Each item must be marked `verified`, `not reproduced`, `not feasible`, or `needs evidence`, with a tested solution/approach before the two aggregate GitHub issues are published.

The first parallel batch was incomplete because the model provider hit HTTP 429; missing rows remain unverified.

## Safety

Do not include databases, memory content, credentials, host details, private paths, or the operator-specific `linuxprocessing_campaign` harness in upstream material.
