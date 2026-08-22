## B03 — Recovery and import atomicity

**Status:** `EXISTING` / `OPEN_PR`  
**Official overlap:** #640, #727, PR #815.

### Evidence and impact

Our deployment required online SQLite backup because SQL `iterdump()` cannot faithfully restore FTS/vec virtual tables (#640). Source audit and failure injection also confirmed that `Mnemosyne.import_from_file()` commits BEAM, legacy memory, embeddings, and optional stores in separate stages. A later store failure can leave earlier stages durable (#727).

Impact includes partial restore state, non-repeatable recovery, stale/duplicated rows on retry, and a false sense of backup safety.

### Suggested solution

- Use SQLite online backup for physical database backup.
- Validate checksum and metadata before replacement.
- Restore through a staged DB, integrity-check it, fsync, then atomically publish with rollback.
- For JSON import, either use one caller-owned transaction across all stores or import into a staged clone and publish only after every section succeeds.
- Never use compensating deletes as rollback.

### Acceptance evidence

Inject failures at every store/replacement boundary and assert the original target is unchanged, temporary artifacts are removed, integrity is `ok`, and retries are idempotent. PR #815 owns focused recovery hardening; #727 remains the JSON-import atomicity tracker.

---

## B04 — Hygiene transaction isolation

**Status:** `OPEN_PR` #816.

Our cleanup trials showed that a malformed candidate must not roll back successful siblings or leak backend details. The focused approach uses one savepoint per candidate, updates counters only after release, records content-free error codes, and makes archive restore idempotent.

Impact: without isolation, one bad candidate can corrupt cleanup accounting, stop the batch, or leave a partial mutation. Suggested completion: rebase #816, prove row-not-found and outer-commit paths, remove duplicate tests, and run hygiene/doctor suites against current main.

---

## B05 — Beam health predicate miscounts successful summaries

**Status:** `EXISTING` #717 / `OPEN_PR` #719.

A successful consolidation summary containing the word `fail` could be counted as an error because the zero-item guard was applied inconsistently. This produces false health alarms and can trigger unnecessary operator recovery.

Suggested solution: count only rows satisfying both an error-like summary predicate and the corresponding zero-item failure condition. Boundary tests must include successful rows containing `error`/`fail`, zero-item rows without either word, and true error rows. PR #719 contains the focused fix but needs rebase/current-head evidence.

---

## B14 — CJK hygiene classification gaps

**Status:** `EXISTING` #806; partially fixed by merged PR #810.

Isolated reproduction found trivial CJK acknowledgements scoring as valuable, long CJK prose false-flagged as dumps because only ASCII sentence punctuation was counted, and CJK-labelled secrets missed. PR #810 fixed secret-label detection across write filters, hygiene, and doctor redaction.

Remaining work must be rechecked before any PR: add curated CJK noise/value tokens and count `。！？；` for structural heuristics without introducing broad multilingual false positives. Use individual regression fixtures; do not reopen the already-fixed secret lane.
