# Two-Issue Upstream Submission Plan

**Prepared:** 2026-08-20
**Upstream baseline:** `7ec6f4c`
**Policy:** audit and reproduce first; publish only verified findings. Do not touch the live Mnemosyne instance.

## Issue 1 — Bugs and verified failure classes

This issue will consolidate verified bugs and regression risks that should be reviewed before implementation is split into focused PRs.

### Candidate findings

- CLI operational failures leaking exception detail, paths, or tracebacks; focused reproduction exists in `tests/test_cli_error_boundary.py`.
- Backup/restore integrity and rollback edge cases; focused recovery suite exists in `tests/test_recovery_paths.py`.
- Hygiene transaction isolation and per-candidate rollback behavior; focused suite exists in `tests/test_hygiene.py`.
- SHMR local-LLM dispatch passing unsupported arguments; covered by `tests/test_shmr_call_llm_local_path.py`.
- Beam health predicates miscounting successful summaries containing `fail`; covered by the beam health regression tests.
- Single-item embedding results being skipped without an operator-visible warning; covered by the embedding warning regression tests.
- Migration dry-run safety and schema fingerprint stability; covered by `tests/test_migration_dry_run_fingerprint.py`.
- Other upstream open bugs should be included only after clean reproduction on the current baseline: #813, #806, #783, #773, #769, #753, #735, #727, #718, #707, #700, #688, #687, #682, #656, #640, #635, #602, #578, and related items.

### Evidence gate

Every item must have: current-baseline reproduction, minimal command/fixture, observed result, expected result, scope, and a regression test or explicit reason a test is not feasible. Host paths, credentials, memory content, and live operational counts stay out.

## Issue 2 — Improvements and enhancements

This issue will consolidate proposed improvements and architecture-level enhancements, then let maintainers choose the smallest landing slices.

### Candidate enhancements

- Receipt-backed ingest/admission policy (Inhale).
- Bounded, read-only recall with strict identity isolation and deterministic fallback (Exhale).
- Proposal-only SHMR candidate generation with a stdlib lexical fallback.
- Reviewer/verifier-gated Dream lifecycle with provenance and rollback.
- Native SDK and CLI/MCP parity.
- Isolated read-only snapshot/restore after writer-lock hardening.
- MCP/doctor parity, packaging CI, and batch containment.
- Hermes `sync_turn` receipts and provider parity.
- Codex lifecycle-hook integration.
- Existing upstream enhancement areas to evaluate for overlap: #790, #789, #784, #766, #761, #732, #724, #715, #712, #695, #661, #651, #598, #586, #543, #514, #450, #449, #446, #403, #372, #370, #327, and #326.

### Evidence gate

Enhancements need a concrete motivation, current behavior, proposed API/behavior, compatibility and rollback plan, focused proof-of-concept or tests where feasible, and explicit non-goals. No implementation PR is opened until maintainer feedback selects the smallest useful slice.

## Planned maintainer workflow

1. Publish one English bug issue and one English improvements issue after the candidate list is audited.
2. Tag the code owners/maintainers once; do not request CodeRabbit review.
3. Wait for maintainer feedback from `dplush` or the other code owner.
4. Convert the agreed scope into the smallest dependency-ordered PRs.
5. Rebase each PR onto current upstream `main`, rerun its focused suite with clean environment, and publish only verified changes.

## Current trial result

The isolated primary-memory worktree test slice passed **378 tests with 1 skip** using `TZ=UTC` and without leaked `MNEMOSYNE*` / `NVIDIA_EMBEDDING*` variables. This is local evidence only, not upstream CI proof.

## Not included

- Raw databases, WAL files, memory content, credentials, private paths, host details.
- The operator-specific `linuxprocessing_campaign` harness.
- Unreproduced or stale claims presented as confirmed bugs.
- Any claim that current upstream CI is passing.

## Next action

Complete the reproduction matrix for the remaining candidate upstream issues, then create the two GitHub issues with links to this plan and the exact evidence. No PR split is published before maintainer direction.

## Status

- Audit scope: complete.
- Initial isolated trial: complete (378 passed, 1 skipped).
- Remaining reproduction matrix: pending.
- Issue publication: pending maintainer-safe evidence review.
