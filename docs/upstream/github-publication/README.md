# Mnemosyne Primary-Memory — Upstream Publication Package

**Maintainer-facing index for the `codex/mnemosyne-primary-memory` branch.**
**Official baseline at time of writing:** `mnemosyne-oss/mnemosyne` `main` @ `85a4b88` ("Merge pull request #702 from dplush/fix/690-bank-aware-export").
**Branch HEAD at time of writing:** `770602a` ("fix(test): resolve cross-test module pollution in hermes_memory_provider tests").
**Merge base with the branch:** `6363a4b` ("fix(cli): normalize MSYS backup destinations (#683)").

This package is a maintainer-safe re-statement of a large additive feature
branch. It is derived from local source and history only. It contains **no
hostnames, usernames, IP addresses, credentials, absolute private paths, live
memory content, live counts, or internal operational vocabulary**. Where a
claim is environment-specific it is explicitly labeled as historical or
proposed, never as current upstream CI status.

---

## What this branch is

A coherent **primary-memory subsystem** layered on the existing BEAM engine,
additive and backward-compatible by construction:

- **Inhale** — durable, receipt-backed, idempotent ingest
  (`mnemosyne/core/inhale.py`, new tables `ingest_receipts`,
  `ingest_conflicts`).
- **Exhale** — hard-bounded, read-only recall with strict isolation and a
  deterministic fallback (`mnemosyne/core/recall_bounded.py`, no new tables).
- **Dream** — reviewer/verifier-gated canonical-mutation lifecycle with
  deterministic manifests and dual-actor receipts (`mnemosyne/core/dream.py`,
  new tables `dream_runs`, `dream_actions`, `dream_receipts`).
- Supporting infrastructure: page-level DR snapshot/restore
  (`mnemosyne/dr/snapshot.py`), content-free diagnostics across the public
  surface, a Codex lifecycle-hook integration
  (`integrations/codex-mnemosyne/`), and a large hardening/test campaign.

The legacy `remember()` / `recall()` / `sleep()` contract is untouched. Every
new table is created with `CREATE TABLE IF NOT EXISTS` on open, so existing
databases acquire the schema without a migration step.

---

## Documents in this package

| File | Purpose |
|---|---|
| [README.md](README.md) | This index. Read first. |
| [redundancy-matrix.md](redundancy-matrix.md) | Overlap of this branch with open upstream issues/PRs, with a precise "addresses / adjacent / does not address" verdict per issue. |
| [verified-bugs.md](verified-bugs.md) | Bugs the branch fixes, stated conservatively with the exact failure class. Each fix is scoped to what the code actually prevents. |
| [feature-roadmap.md](feature-roadmap.md) | Dependency-ordered PR slicing plan. The on-host trial harness is explicitly **excluded** from every slice. |
| [privacy-and-evidence.md](privacy-and-evidence.md) | Content-free surface contract, security posture, and a conservative statement of test/evidence status (historical only). |
| [work-report.md](work-report.md) | How this package was assembled, what was corrected vs. the prior report, and open concerns. |
| [../mnemosyne-primary-memory-contribution-report.md](../mnemosyne-primary-memory-contribution-report.md) | The full engineering report. Baseline/status sections refreshed to `85a4b88`; body otherwise unchanged. |

---

## Explicitly excluded from upstream

The on-host evidence harness (`scripts/linuxprocessing_campaign.py` and its
test `tests/test_linuxprocessing_campaign.py`) is **trial/operator tooling**.
It is named for a specific deployment, carries operator-trial vocabulary,
and has no product value upstream. It is excluded from every slice in
[feature-roadmap.md](feature-roadmap.md). See the work report for the
rationale.

---

## How to read this package

1. Start here for scope and baseline.
2. [redundancy-matrix.md](redundancy-matrix.md) to see whether the work
   duplicates anything already open upstream.
3. [verified-bugs.md](verified-bugs.md) for the conservative bug/fix list.
4. [feature-roadmap.md](feature-roadmap.md) for the review plan.
5. [privacy-and-evidence.md](privacy-and-evidence.md) for the security and
   evidence posture.

All claims distinguish **verified-from-source** facts from **proposed** work.
Where prior internal reports overstated a claim, this package states the
correction explicitly in [work-report.md](work-report.md).
