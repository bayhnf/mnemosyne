# Upstream Publication Status

**Refreshed:** 2026-08-20
**Upstream baseline:** `mnemosyne-oss/mnemosyne` `main` @ `7ec6f4c` (`docs: add native Windows Hermes recovery guide (#812)`)
**Account:** `bayhnf`

This file is the current submission map. It supersedes the older snapshot-specific status notes in the original package documents.

## Published / open upstream

| Slice | PR | Branch | Scope | Status |
|---|---:|---|---|---|
| S1a | [#814](https://github.com/mnemosyne-oss/mnemosyne/pull/814) | `contrib/s1-cli-errors-focused` | CLI error boundaries and focused regression tests | Open |
| S1b | [#815](https://github.com/mnemosyne-oss/mnemosyne/pull/815) | `contrib/s1-recovery` | Backup/restore hardening and recovery tests | Open |
| S1c | [#816](https://github.com/mnemosyne-oss/mnemosyne/pull/816) | `contrib/s1-hygiene` | Hygiene transaction behavior and tests | Open; 6 doctor CLI tests currently fail |
| S1d | [#817](https://github.com/mnemosyne-oss/mnemosyne/pull/817) | `contrib/s1-model-refresh` | Dream/model-refresh configuration and tests | Open |
| S7a | [#774](https://github.com/mnemosyne-oss/mnemosyne/pull/774) | `contrib/migrate-dry-run` | Report-only migration dry run | Open |
| S1 follow-up | [#719](https://github.com/mnemosyne-oss/mnemosyne/pull/719) | `contrib/fix-health-precedence` | Beam consolidation health predicates | Open |
| S1 follow-up | [#720](https://github.com/mnemosyne-oss/mnemosyne/pull/720) | `contrib/fix-remember-embed-warning` | Warning for skipped single-item embeddings | Open |
| S1 follow-up | [#721](https://github.com/mnemosyne-oss/mnemosyne/pull/721) | `contrib/fix-shmr-local-llm` | Restore SHMR local-LLM path | Open |

## Superseded / closed

| PR | Branch | Reason |
|---:|---|---|
| [#788](https://github.com/mnemosyne-oss/mnemosyne/pull/788) | `contrib/s1-cli-error-containment` | Bundled scope was split into focused PRs #814–#817 after maintainer feedback. |

## Not yet published

The following roadmap slices have no upstream PR yet:

- **S2** — Inhale ingest receipts and admission policy
- **S3** — Bounded recall / Exhale
- **S4** — SHMR proposal-only harmony with lexical fallback
- **S5** — Dream lifecycle
- **S6** — Native SDK, CLI parity, and Dream failure exits
- **S7b** — Isolated read-only snapshot/restore; blocked on post-replace writer-lock hardening
- **S7c** — MCP/doctor parity, packaging CI, and batch containment
- **S8** — Hermes `sync_turn` receipts
- **S9** — Codex lifecycle-hook plugin; manual desktop checkpoint remains outstanding

## Refresh rules

- Rebase each open PR onto the then-current `upstream/main` before submitting or updating it.
- Run focused tests with `TZ=UTC` and without leaked `MNEMOSYNE*` / `NVIDIA_EMBEDDING*` environment variables.
- Keep operator-specific trial harnesses, databases, memory content, credentials, host details, and private paths out of upstream commits.
- Treat this file as a planning/status index; verify GitHub live status again before the next submission.

## Verification snapshot

- Upstream `main` verified at `7ec6f4c`.
- Open PRs verified through GitHub CLI: #719, #720, #721, #774, #814, #815, #816, #817.
- #788 verified closed and superseded by #814–#817.
- No merge claim is made here; all listed PRs were open at refresh time.
