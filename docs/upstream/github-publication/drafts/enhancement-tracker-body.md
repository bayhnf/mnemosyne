# Deployment-informed memory lifecycle and integration improvements

## Context and evidence standard

These proposals came from a Mnemosyne setup that we personally operate and have extended for a real Hermes integration. We exercised provider migration, write/recall policy, consolidation, recovery, identity scope, diagnostics, and agent lifecycle flows beyond isolated happy paths.

Private deployment observations are discovery and design evidence, not claims about every upstream deployment. Data below is sanitized. Proposed changes are local-first, opt-in unless stated otherwise, and include compatibility/rollback notes. Existing official issues are linked instead of duplicated.

This is a tracking/index issue, not one implementation request. We are willing to develop accepted items as small, dependency-ordered PRs. PRs should use `Addresses Exx in this tracker`; they must not close the tracker.

## Status vocabulary

- `PROPOSED_NEW` — no semantic equivalent found in 315 issues / 500 PRs.
- `EXISTING/PARTIAL` — official work covers part or all of the capability.
- `READY_FOR_GUIDANCE` — design and evidence exist; maintainer direction needed before code.
- `BLOCKED` — prerequisite contract or hardening is missing.
- `IMPLEMENTED_LOCALLY` — exercised in our private setup, not yet an upstream claim.

## Improvements index

| ID | Improvement | Status | Existing overlap | Detail |
|---|---|---|---|---|
| E01 | Instrumented provider migration: shadow capture, recall diff, gates, canary, soak, rollback | `PROPOSED_NEW` / `IMPLEMENTED_LOCALLY` | No framework-level equivalent found | [proposal](https://github.com/mnemosyne-oss/mnemosyne/issues/828#issuecomment-5378586315) |
| E02 | Safe recall metadata projection for policy-aware consumers | `PROPOSED_NEW` / `READY_FOR_GUIDANCE` | No read-side equivalent found; PR #644 is write-side | [proposal](https://github.com/mnemosyne-oss/mnemosyne/issues/828#issuecomment-5378586315) |
| E03 | Multi-agent reader-side visibility matrix | `EXISTING/PARTIAL` | #327, #653, #761 | [details](https://github.com/mnemosyne-oss/mnemosyne/issues/828#issuecomment-5378586528) |
| E04 | Durable receipt-backed ingest (Inhale) | `READY_FOR_GUIDANCE` | Adjacent #727, #766, #789 | [details](https://github.com/mnemosyne-oss/mnemosyne/issues/828#issuecomment-5378586528) |
| E05 | Hard-bounded read-only recall (Exhale) | `EXISTING/PARTIAL` | #514, #449 | [details](https://github.com/mnemosyne-oss/mnemosyne/issues/828#issuecomment-5378586528) |
| E06 | Reviewer/verifier-gated canonical mutation lifecycle (Dream) | `EXISTING/PARTIAL` | #410, #434, #449; PR #817 foundation | [details](https://github.com/mnemosyne-oss/mnemosyne/issues/828#issuecomment-5378586528) |
| E07 | Isolated page-level snapshot/restore | `EXISTING/PARTIAL` / `BLOCKED` | #640, PR #815 | [details](https://github.com/mnemosyne-oss/mnemosyne/issues/828#issuecomment-5378586528) |
| E08 | Content-free diagnostics and tool-surface parity | `EXISTING/PARTIAL` | #372, #728; merged #814/#758 | [details](https://github.com/mnemosyne-oss/mnemosyne/issues/828#issuecomment-5378586528) |
| E09 | Durable Hermes `sync_turn` receipts and provider parity | `EXISTING/PARTIAL` | #328, #655, #766 | [details](https://github.com/mnemosyne-oss/mnemosyne/issues/828#issuecomment-5378586528) |
| E10 | Local Codex lifecycle integration | `READY_FOR_GUIDANCE` | No focused Mnemosyne issue; adjacent agent integrations | [details](https://github.com/mnemosyne-oss/mnemosyne/issues/828#issuecomment-5378586528) |
| E11 | Trust-aware capture and prompt-injection quarantine | `EXISTING/PARTIAL` | #109, #410, #821; write-approval/security work | [details](https://github.com/mnemosyne-oss/mnemosyne/issues/828#issuecomment-5378586528) |

## Measured deployment evidence

Sanitized pre-cutover evaluation recorded zero failed mandatory gates. Evaluated gates covered recall, negative controls, stale/current behavior, scope leakage, latency, backup/restore, provider health, and rollback. The candidate canary separately passed import, write, recall, integrity, vector growth, runtime, and clean-console checks. Long-duration soak/canary gates were explicitly recorded as scheduled—not falsely reported complete.

## Design rules

- No cloud dependency or telemetry requirement.
- No raw memory content in reports.
- Preserve provenance, scope, and rollback.
- New schemas are additive and idempotent.
- Read-only capabilities must not mutate recall counters or memory state.
- Mutation requires clear ownership, validation, and failure semantics.
- One focused PR per accepted capability or root boundary.

## Willing to contribute

- [x] We can submit focused PRs after maintainer guidance.
- [x] We will preserve existing defaults unless a separately approved change says otherwise.
- [x] We will provide sanitized fixtures, benchmarks where relevant, and rollback evidence.

Related current work: #815, #816, #817 and roadmap #514.
