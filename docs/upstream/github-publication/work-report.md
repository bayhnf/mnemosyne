# Work Report — Publication Package Assembly

**Date:** 2026-08-13.
**Scope:** assemble a maintainer-safe publication package under
`docs/upstream/github-publication/` and refresh the baseline/status portions
of the existing contribution report.

This report records what was written, what was corrected versus prior
internal reports, and the open concerns. It is itself part of the package.

---

## Files produced / refreshed

All paths relative to the repo root. Only files under `docs/upstream/` were
touched; no source, tests, git state, or other untracked docs were modified.

**New files (this package):**

- `docs/upstream/github-publication/README.md` — package index.
- `docs/upstream/github-publication/redundancy-matrix.md` — upstream issue overlap.
- `docs/upstream/github-publication/verified-bugs.md` — conservative bug/fix list.
- `docs/upstream/github-publication/feature-roadmap.md` — PR slicing plan.
- `docs/upstream/github-publication/privacy-and-evidence.md` — security/evidence posture.
- `docs/upstream/github-publication/work-report.md` — this file.

**Refreshed (baseline/status portions only):**

- `docs/upstream/mnemosyne-primary-memory-contribution-report.md` — header
  baseline, §2 baseline/status, diff-stat methodology reference, rollback
  reference, and footer updated from the prior reference point (`f6ba5c8`)
  to the current official baseline (`85a4b88`). The body of the report was
  left intact; `f6ba5c8` remains only in the corrected baseline statement
  and the historical lineage list where it belongs.

---

## Baseline correction

The prior internal report cited upstream `main` `f6ba5c8` (#686) as the
baseline. The current official baseline is `85a4b88` (#702 merge).
`f6ba5c8` is an ancestor of `85a4b88` (17 commits between them, verified
via `git merge-base --is-ancestor`). The merge base between this branch and
upstream remains `6363a4b` (#683). All baseline/status references in the
package now state `85a4b88` as the current official baseline and `6363a4b`
as the merge base, with `f6ba5c8` kept only as the prior reference point.

---

## Corrections to overstated / stale claims

Each correction below was verified against current source before being
written into the package. The conservative statement is in
[verified-bugs.md](verified-bugs.md); the rationale is here.

1. **Long-document context-window cause: unproven.** A prior audit attributed
   a silent embed skip to the embedding model context window. The cause is not
   established: a later re-test of the same embedding service embedded
   documents of comparable size successfully. The incident is consistent
   with a transient backend condition. The package does **not** cite a
   context-window cause.

2. **Orphan rows are expected maintenance, not a bug.** A small number of
   orphan embedding rows are normal debris from deletes. `reclaim_orphans()`
   and the hygiene/doctor surface are the intended maintenance path. The
   package frames them as maintenance, not as a bug fix.

3. **sqlite_vec `.so` suffix: not a code defect.** The `.so` suffix issue was
   a CLI-only operational workaround in manual scripts. Production code loads
   `sqlite_vec` via `sqlite_vec.load(conn)`. Verified at
   `mnemosyne/core/beam.py` (~L200, ~L478). Not carried forward as a bug.

4. **Empty LLM env config: environment, not code.** An empty
   `MNEMOSYNE_LLM_ENABLED` was host-specific environment misconfiguration,
   already tracked upstream via the existing `local_llm`/`MnemosyneConfig`
   path. This branch does not introduce or worsen it.

5. **"Error count since restart: 0": unreliable.** Any error-count figure is
   a single-host operational snapshot. The package excludes all such figures
   and does not cite error counts as correctness evidence.

6. **SHMR fallback is lexical, not "no-NumPy".** `shmr.py` still imports
   `numpy` (guarded; `np = None` on `ImportError`). The fix made the
   *offline lexical fallback path* (`_lexical_vector`, `_cosine_similarity`)
   stdlib-only so it is reachable without NumPy. The dense paths (`_embed`,
   `_compute_harmony_score`, `recall_beliefs`) still call `np.*` and still
   require NumPy. The package describes this as a lexical-fallback fix with
   an explicit scope correction. Verified at `mnemosyne/core/shmr.py` (~L27
   guarded import; numpy still referenced at ~L124, ~L298, ~L324, ~L825).

---

## Explicit exclusion: campaign harness

`scripts/linuxprocessing_campaign.py` and
`tests/test_linuxprocessing_campaign.py` are excluded from every slice in
[feature-roadmap.md](feature-roadmap.md). Rationale:

- Named for a specific deployment; carries operator/trial vocabulary in its
  module docstring, argparse prog, and ack flags.
- Has no product value upstream; it is an on-host evidence harness.
- Two independent internal audits concurred it is trial-only and must not
  enter any PR branch.

The package's [feature-roadmap.md](feature-roadmap.md) states this exclusion
explicitly in a dedicated section.

---

## Privacy / sanitization posture

The package contains no private hostnames, usernames, IPs, absolute private
paths, credentials, live memory content, live counts, agent names, or
internal Task/G vocabulary in maintainer-facing bodies. Operational details
from the source audit (specific row counts, memory IDs, SHA matches, error
counts, host names, embedding service URLs, model paths) were deliberately
excluded.

## No commit / push

No `git add`, `git commit`, `git push`, or remote access was performed. All
files are written to the working tree under `docs/upstream/` only.

---

## Open concerns for the maintainer

1. **S7b (snapshot) is blocked.** The post-replace writer-lock
   re-acquisition hardening was designed but is not implemented in this
   branch. S7b must not open until that fix lands and competing-writer probes
   pass.
2. **S5 hygiene fixes.** Two small docstring fixes (a private path in
   `test_dream_lifecycle.py` and internal-plan-file references in
   `dream.py:20`) must land before the S5 PR opens. They are noted in
   [feature-roadmap.md](feature-roadmap.md) §S5.
3. **CLA timing.** No prior merged contribution is on record for the
   contributing account; confirm CLA-assistant sign before the first PR.
4. **Upstream may move.** Each slice should rebase onto the then-current
   `upstream/main` and re-run its focused suite at PR-open time.
5. **Historical, not current.** All test/evidence references are labeled
   historical. No claim of current upstream CI pass is made anywhere in this
   package.
