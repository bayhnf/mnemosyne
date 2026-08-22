# GLM Handoff — Mnemosyne Upstream Tracking Issues

**Date:** 2026-08-22  
**Repository:** `mnemosyne-oss/mnemosyne`  
**GitHub account:** `bayhnf`  
**Language for all GitHub communication:** English

## Objective

Continue managing two deployment-informed tracking issues. Do not recreate, duplicate, close, or republish them.

- Bug tracker: https://github.com/mnemosyne-oss/mnemosyne/issues/827
- Improvements tracker: https://github.com/mnemosyne-oss/mnemosyne/issues/828

Each tracker has one index body and two collapsible detail comments so readers are not overwhelmed.

## Published GitHub structure

### Issue #827 — Bugs

Title:

`[TRACKING][BUG] Deployment-informed correctness and reliability findings`

Comments:

- New verified findings: https://github.com/mnemosyne-oss/mnemosyne/issues/827#issuecomment-5378533682
- Existing/fixed/in-progress findings: https://github.com/mnemosyne-oss/mnemosyne/issues/827#issuecomment-5378534117

Content:

- 17 indexed bug/failure classes.
- Three newly verified non-duplicate findings:
  - B01: concurrent fresh-database schema initialization race.
  - B15: batch mutation errors expose backend exception details.
  - B16: SHMR import breaks the supported no-NumPy base-install contract.
- Remaining items link to existing official issues/PRs instead of duplicating them.

### Issue #828 — Improvements

Title:

`[TRACKING][FEATURE] Deployment-informed memory lifecycle and integration improvements`

Comments:

- New proposals: https://github.com/mnemosyne-oss/mnemosyne/issues/828#issuecomment-5378586315
- Existing/partially covered directions: https://github.com/mnemosyne-oss/mnemosyne/issues/828#issuecomment-5378586528

Content:

- 11 indexed improvements.
- Two proposals with no semantic equivalent found:
  - E01: instrumented provider-migration governance.
  - E02: safe recall metadata projection for policy-aware consumers.
- Other improvements explicitly reference existing issues and PRs.

## Provenance wording

Use this framing consistently:

> These findings came from a Mnemosyne setup that we personally operate and have extended for a real Hermes integration. Running the system across real lifecycle and failure paths exposed the items below. We are not presenting private deployment observations as upstream proof: confirmed bugs are independently reproduced against current upstream code with sanitized fixtures, and proposed improvements include sanitized evidence, examples, expected impact, compatibility considerations, and an implementation approach.

Private deployment observations are discovery signals, not sole proof.

## Evidence completed

### B01 — Fresh database initialization race

Baseline: upstream `4c6b280`.

A deterministic two-connection barrier forced both workers to complete `PRAGMA table_info` before either called `ALTER TABLE ADD COLUMN`.

Result across 20 fresh databases:

```text
one worker: ok
one worker: OperationalError: duplicate column name: author_id
author_id_count=1
integrity=ok
REPRODUCED=20/20
```

Suggested solution: database-level migration serialization, schema re-read after lock acquisition, final-contract verification, and propagation of unrelated SQLite errors.

### B15 — Batch error disclosure

Baseline: upstream `4c6b280`.

A synthetic batch operation raised a canary-bearing `RuntimeError`. The current shared batch function:

- serialized exception class/message in the returned error field;
- logged the canary;
- logged a traceback.

Suggested solution: closed error vocabulary (`batch_validation_failed`, `batch_failed`), safe index and allowlisted action only, content-free logging, unchanged transaction rollback.

### B16 — SHMR no-NumPy import

Baseline: upstream `4c6b280`.

Blocking NumPy imports while importing `mnemosyne.core.shmr` produced `ModuleNotFoundError`. `shmr.py` imports NumPy unconditionally even though PR #31 established a supported no-NumPy base installation.

Scope correction: dense SHMR operations may still require NumPy. Only module import and non-dense degradation must remain available.

### E02 — Recall metadata projection

Baseline: upstream `4c6b280`.

A memory stored custom metadata successfully, but recall returned selected fields without `metadata_json`. Policy-aware consumers must execute extra table lookups.

Treat this as an enhancement/API decision, not a bug. Preferred options are allowlisted projection or bulk hydration—not unconditional raw metadata.

## Dedup audit completed

Corpus reviewed:

- 315 official issues: open and closed.
- 500 pull requests: open, closed, and merged.
- Titles and bodies were searched semantically.

Do not open duplicate child issues for these existing areas:

- Session switching: #601 / merged PR #604.
- Transient SQLite initialization retry: #654 / merged PR #496.
- Consolidation concurrency/session scope: #342, #498, #687 / PRs #349, #520, #772.
- Virtual-table backup: #640 / PR #815.
- Vector dimension mismatch: #753 / PR #754.
- Silent embedding failure: #718, #735 / PRs #720, #797.
- Polyphonic prefetch fields: #700 / PR #701.
- Source-time recency: #564.
- MCP/session identity visibility: #327, #653, #761.
- Background prefetch: #326 and prior PR #541/merged relevance hardening #616.
- Canonical lifecycle: #434, #435, #449 and merged retirement PR #723.
- MCP SDK compatibility: merged PR #571.
- CJK secret detection: #806 / merged PR #810.
- Write admission/config: #789, #821.
- Hygiene remediation: #428 / merged PR #431.

## Live PR state at last refresh

- #814: approved and merged as upstream `4c6b280`.
- #815: open, recovery hardening.
- #816: open, hygiene transaction behavior; required rebase at last check.
- #817: open, model-refresh/Dream config foundation.
- #719: open; health predicate fix; prior feedback addressed; stale/conflicting at last check.
- #720: open; embedding warning; prior feedback addressed.
- #721: open; SHMR local path; stale/conflicting at last check.
- #774: open; migration dry-run; stale/conflicting at last check.

Always refresh live state before acting.

Latest refresh after publication:

- upstream `main`: `cbbfc2a` (PRs #823–#825 merged after #814; no overlap with B01/B15/B16 paths).
- #827 and #828: open, only author comments, no maintainer reply yet.
- #829: open focused E02 PR (`contrib/recall-metadata-projection`), mergeable; docs/lint/build/CLA green and test matrix running.
- #564: `AxDSan` approved inherited source timestamps (`timestamp=max(source timestamps)`, `created_at=now`) and requested sequencing after PR #563; this is owned by the existing reporter, not us.

## Local files

Canonical package:

`docs/upstream/`

Current status and queues:

- `docs/upstream/github-publication/publication-status-2026-08-22.md`
- `docs/upstream/github-publication/active-upstream-queue-2026-08-22.md`
- `docs/upstream/github-publication/reproduction-status-2026-08-20.md`
- `docs/upstream/github-publication/two-issue-submission-plan-2026-08-20.md`

Published body/comment sources:

- `docs/upstream/github-publication/drafts/bug-tracker-body.md`
- `docs/upstream/github-publication/drafts/publish-bug-new.md`
- `docs/upstream/github-publication/drafts/publish-bug-existing.md`
- `docs/upstream/github-publication/drafts/enhancement-tracker-body.md`
- `docs/upstream/github-publication/drafts/publish-enh-new.md`
- `docs/upstream/github-publication/drafts/publish-enh-existing.md`

Long-form source drafts are also under the same `drafts/` directory.

Deployment evidence sources, for local review only:

- `/home/bell/mnemosyne-shadow/`
- `/home/bell/mnemosyne-sidecar-2b5e510/candidate/`
- `/home/bell/mnemosyne-sidecar-2b5e510/stage-live-shadow-8HW7wXFm/canary-report.json`
- `/home/bell/.hermes/reports/mnemosyne-scope-improvement-20260814.md`
- `/home/bell/.hermes/reports/kyo-soul-capability-upgrade-20260813.md`

Never paste private paths or operational identifiers into GitHub.

## Publication verification already passed

Read-back checks confirmed:

- #827: open; two comments; no unresolved placeholders; collapsible sections balanced.
- #828: open; two comments; no unresolved placeholders; collapsible sections balanced.
- No private home paths, GitHub tokens, internal IP markers, or draft URL placeholders in published content.
- Combined published content remains comfortably below GitHub limits.

Labels were not applied because `bayhnf` lacks repository label permissions. Titles already contain `[BUG]` and `[FEATURE]`; do not repeatedly retry label mutation.

## Required next actions

1. Read new maintainer replies on #827 and #828.
2. Reply in English only.
3. Do not immediately implement every tracker item.
4. Ask/observe which new item maintainers want first.
5. For existing issues, place additional evidence on the original issue rather than creating duplicates.
6. Create one focused PR per accepted root cause.
7. Reference tracker items with `Addresses Bxx in #827` or `Addresses Exx in #828`.
8. Never use `Fixes #827` or `Fixes #828`; that would close the entire tracker.
9. Refresh and repair existing open PRs before opening broad new work.
10. Keep tracker index/status current when PRs merge or findings move.

## Safety rules

- Use isolated worktrees; never test against the live Mnemosyne database.
- Use `TZ=UTC` and unset leaked `MNEMOSYNE*` / `NVIDIA_EMBEDDING*` variables for tests.
- Do not expose memory content, credentials, hostnames, IP addresses, absolute private paths, or live row identifiers/counts.
- Do not ping CodeRabbit manually.
- Use `bayhnf` only; `bellfireg` is suspended.
- GitHub issue/PR comments must be English.
- Check `git status` before commit; do not include unrelated `docs/superpowers/`.

## Local git state at initial handoff

The publication drafts and updated two-issue plan were modified/created locally but were not yet committed at handoff time. Before committing:

```bash
git status --short
git diff --check
```

Stage only:

```text
docs/upstream/github-publication/two-issue-submission-plan-2026-08-20.md
docs/upstream/github-publication/drafts/
docs/upstream/github-publication/GLM-HANDOFF-2026-08-22.md
```

Do not stage `docs/superpowers/`.

Publication artifacts were subsequently committed as `fc0dd1f` (`docs: record upstream tracking issue publication`).
