## B01 — Concurrent fresh-database initialization can race schema guards

**Status:** `VERIFIED_NEW`  
**Upstream baseline:** `4c6b280`  
**Duplicate audit:** no semantic duplicate found across 315 issues and 500 PRs.

### Discovery source

Our Hermes-integrated Mnemosyne setup can initialize the same database from more than one provider process. We added a per-database initialization lock after the setup exposed concurrent first-open risk in upstream schema initialization.

### Independent reproduction

`_add_column_if_missing()` performs:

1. `PRAGMA table_info(table)`
2. checks whether the column is absent
3. `ALTER TABLE ... ADD COLUMN`

A sanitized test used two independent SQLite connections. A barrier forced both workers to complete step 1 before either executed step 3. Across 20 fresh databases:

```text
worker A: ok
worker B: OperationalError: duplicate column name: author_id
author_id_count=1
integrity=ok
REPRODUCED=20/20
```

The database remains valid, but one initializer fails.

### Impact

- One concurrently starting provider/MCP/worker can lose memory initialization for that process or session.
- Readiness can disagree between processes opening the same new or newly migrated database.
- Provider retry can hide the root schema race rather than fix it.
- Any `_add_column_if_missing()` call has the same check-before-ALTER window.

### Root cause

Schema reconciliation is a non-atomic check-then-write sequence. SQLite serializes the ALTER itself, but not the preceding decision that the column is missing.

### Suggested solution

Use one shared migration boundary:

1. acquire a database-level migration transaction/lock;
2. re-read `PRAGMA table_info` after acquisition;
3. apply only still-missing changes;
4. on a duplicate-column race, verify the final column contract before treating it as success;
5. propagate unrelated SQLite errors.

### Acceptance tests

- [ ] Two threads initialize one fresh DB successfully.
- [ ] Two processes initialize one fresh DB successfully.
- [ ] The column exists exactly once.
- [ ] `PRAGMA integrity_check` returns `ok`.
- [ ] Existing DB initialization remains idempotent.
- [ ] Lock, readonly, disk-I/O, and unrelated migration errors remain visible.

We are willing to submit this as a focused concurrency PR after maintainer confirmation.

---

## B02 — CLI failure-boundary hardening

**Status:** `FIXED/MERGED` via #814.

Our deployment exposed wrong success exits, raw exception/path leakage, side effects on help/version, swallowed hygiene failures, and locale-dependent candidate decoding. Each boundary was reproduced with synthetic canaries. The focused PR added exact static stderr, empty stdout on failures, command-specific codes, UTF-8 decoding, side-effect-free informational commands, and portable tests. Full CI and CLA passed; `dplush` approved; #814 merged.

This item remains in the tracker as proven provenance and as the review standard for future PRs, not as open work.
