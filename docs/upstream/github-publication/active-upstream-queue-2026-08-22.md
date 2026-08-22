# Active Upstream Bug and Improvement Queue

**Refreshed:** 2026-08-22
**Baseline:** upstream `main` @ `cbbfc2a` (findings B01/B15/B16 were reproduced on `4c6b280`; subsequent #823–#825 do not touch their paths)
**Evidence policy:** reproduce in isolated fixtures first; publish only verified findings and tested solutions.

## Merged fix

### CLI operational failure containment — PR #814

Status: **approved and merged**.

Implemented solution:

- Static command-specific failure codes and exit code `1` for operational failures.
- Exact stderr contract with empty stdout on failures.
- Redaction of paths, backend details, canaries, and tracebacks.
- Side-effect-free help/version aliases.
- Command-specific handling for doctor, backup, restore, verify, hygiene, migration, diagnose, store setup, and sync secret-file failures.
- UTF-8-only hygiene candidate JSON decoding, including invalid UTF-8, directory paths, and unexpected decoder failure containment.

Final evidence: full upstream CI/CLA green; approved by `dplush`; merged as `4c6b280`.

## Published trackers

- Bugs and reliability: [#827](https://github.com/mnemosyne-oss/mnemosyne/issues/827)
- Improvements and integrations: [#828](https://github.com/mnemosyne-oss/mnemosyne/issues/828)

New verified findings indexed in #827:

- **B01:** concurrent fresh-DB schema initialization race (`20/20` deterministic reproduction).
- **B15:** batch mutation errors expose raw exception text and traceback details.
- **B16:** SHMR imports NumPy unconditionally despite the supported no-NumPy base-install contract.

New proposals indexed in #828:

- **E01:** instrumented provider-migration governance.
- **E02:** safe recall metadata projection for policy-aware consumers.

## Open fixes already published

| Finding | PR | Solution approach | State |
|---|---:|---|---|
| Beam health predicate miscounts | #719 | Apply zero-item guard only to error-like summaries; boundary regressions | Open, conflicting |
| Silent `None`/wrong-count embedding result | #720 | Persist text fail-soft and emit operator-visible warning; persistence regression | Open, mergeable |
| SHMR local LLM dispatch uses unsupported arguments | #721 | Call prompt-only local helper with supported contract and no-network fallback tests | Open, conflicting |
| Migration dry-run safety | #774 | Report-only schema planning plus byte/fingerprint invariance test | Open, conflicting |
| Backup/restore hardening | #815 | Atomic staged restore, checksums, locking, fsync, integrity and rollback matrix | Open, mergeable; awaiting review |
| Hygiene transaction isolation | #816 | Per-candidate savepoints, rollback isolation, idempotent archive restore | Open, conflicting |
| Dream/model-refresh configuration | #817 | Centralized `dream_active`, auto-apply gating, constrained remote fallback | Open, mergeable; awaiting review |

## Verified bugs with solutions, not yet published as our focused PR

| Issue | Reproduction result | Minimal solution | Upstream state |
|---:|---|---|---|
| #806 | CJK acknowledgements under-detected; CJK prose false-flagged; Chinese-labelled secrets missed | Unicode-aware deterministic markers, punctuation counting, and secret labels | Closed upstream; secret portion landed in #810; remaining hygiene behavior must be rechecked before any PR |
| #727 | Multi-store JSON import commits partial state before later failure | Stage into a temporary DB and atomically publish, or use one caller-owned transaction across stores | Open |
| #735 | API embeddings can report available but return `None` silently | Emit content-free degradation diagnostic or typed result; persist retry/degraded receipt | Open |
| #753 | Episodic vector dimension mismatch propagates `OperationalError` from recall | Catch mismatch at vector voice, log safe reindex guidance, return `[]` so other voices continue | Open |
| #688 | SHMR loads/downloads local GGUF before configured remote path | Skip local loader when remote base URL is configured; assert download is never called | Open |
| #700 | Polyphonic results omit score fields consumed by Hermes prefetch | Normalize per-signal score fields before provider filtering | Open |
| #687 | Per-session auto-sleep can select `sleep_all_sessions()` | Scope eligibility and worker to triggering session; reserve all-session sleep for explicit maintenance | Open |

## Improvements and enhancements with prepared approaches

| Slice | Improvement | Prepared approach | Gate |
|---|---|---|---|
| S2 | Durable Inhale | Atomic memory + receipt write, enrichment outside transaction, retry ownership/admission policy | Rebase after S1 foundations |
| S3 | Bounded Exhale | Read-only post-hydration gate with identity isolation, top-k/token caps, deterministic fallback | Depends on S2 metadata contract |
| S4 | SHMR proposal mode | Proposal-only candidates and stdlib lexical fallback; no direct mutation | Depends on S3 scope contract |
| S5 | Dream lifecycle | Reviewer/verifier-separated receipts, manifest binding, CAS apply/undo | Depends on S4; sanitize two docstrings first |
| S6 | SDK/CLI parity | Native API plus CLI/MCP verbs and deterministic failure exits | Depends on S5 |
| S7b | Isolated snapshots | Page-level backup/restore with 0600/0700 permissions and checksum | Blocked on post-replace writer-lock reacquisition |
| S7c | Tool/packaging parity | Align MCP/doctor schemas, package CI, batch error containment | Depends on S6 |
| S8 | Hermes turn receipts | Atomic `sync_turn` receipts in both provider mirrors | Depends on S2 |
| S9 | Codex lifecycle hooks | Local SessionStart/UserPromptSubmit/Stop/SessionEnd integration with content-free spool | Depends on S2 + S6; desktop checkpoint required |

## Pending reproduction inventory

Bugs still requiring current-baseline isolated recreation and a tested solution/status:

`#813 #783 #707 #682 #656 #640 #635 #602 #578 #573 #560 #559 #552 #548 #537 #523 #506 #487 #434`

Enhancements still requiring overlap/feasibility review:

`#790 #789 #784 #766 #761 #732 #724 #715 #712 #695 #661 #651 #598 #586 #543 #514 #450 #449 #446 #403 #372 #370 #327 #326`

Each must end as `verified`, `not reproduced`, `not feasible`, or `needs evidence`. Add evidence to existing issues where applicable; do not duplicate them.

## Safety exclusions

Never publish raw databases, WAL/SHM files, memory content, credentials, private paths, host details, live row counts, or `linuxprocessing_campaign` operator tooling.
