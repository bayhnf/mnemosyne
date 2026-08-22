## B10 — Remote LLM configuration can still activate local model loading

**Status:** `EXISTING` #688.

Source tracing showed the SHMR path calling the local helper before remote fallback. The local loader gated on LLM enablement but not on a configured remote base URL, so a missing local GGUF could trigger a large unsolicited download.

Impact: unexpected disk/network use, startup latency, and violation of configured remote-backend intent.

Suggested solution: when a remote base URL is explicitly configured, bypass local model discovery/download. Add a regression that patches the downloader and proves it is never called while remote mode is selected.

---

## B11 — Automatic consolidation concurrency and scope

**Status:** `EXISTING`; fixes include #349, #520, #772; remaining related issue #687.

Our real integration exercised overlapping per-turn writes, prefetch, and background consolidation. Official history confirms three failure classes: all-session timeout, shared SQLite/vec access races, and unrelated-session consolidation.

Impact ranges from silent failed episodic writes to process crashes and cross-session maintenance mutations.

Existing focused solutions are the correct shape: isolated worker connections, provider-level serialization around Beam access, session-scoped automatic `sleep()`, and explicit-only `sleep_all_sessions()`. Do not file a new lease issue unless a new overlapping-worker failure reproduces after these merged fixes.

---

## B12 — Derived summaries claim write time as content age

**Status:** `EXISTING` #564.

Shadow policy uses source time for recency after observing that a freshly written summary of old memories can outrank its sources. Upstream #564 contains the detailed reproduction and ranking crossover.

Suggested solution: carry a source-time range/representative timestamp into derived rows and use it for recency scoring while preserving derivation time separately for audit. Add ranking tests across aged sources. Contribute evidence to #564 rather than opening a duplicate.

---

## B13 — Identity and scope visibility gaps

**Status:** `EXISTING` / partly fixed: #327, #601, #653; merged PR #604.

Our deployment found session writes bound to the wrong identity subject and project recall missing `reader_project`; the local adapter fix proved same-session isolation, cross-session user recall, and exact project isolation with synthetic fixtures. Official overlap already covers gateway identity mapping, session rebind, and MCP fixed-session invisibility.

Impact: writes succeed and remain indexed but are invisible to the intended reader, or memory from unrelated identities can be mixed if mapping is too broad.

Suggested direction: one explicit mapping from Hermes runtime identity to persisted scope/metadata and reader filters; fail closed for missing project identity; expose on-demand project recall. Automatic project prefetch should wait for Hermes to pass project context—guessing it risks leakage. Add evidence to #327/#653 instead of filing a duplicate.

---

## Tracker note

Items with existing official issues remain here only to connect sanitized deployment evidence, current status, and focused contribution plans. Their detailed reproduction should be posted to the original issue when useful.