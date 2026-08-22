# Upstream Publication Status

**Refreshed:** 2026-08-22
**Upstream baseline:** `mnemosyne-oss/mnemosyne` `main` @ `cbbfc2a` (through merged PRs #823–#825)
**Account:** `bayhnf`

This is the current submission map. Live GitHub state was checked before writing it.

## Merged

| Slice | PR | Final head | Scope | Result |
|---|---:|---|---|---|
| S1a | [#814](https://github.com/mnemosyne-oss/mnemosyne/pull/814) | `699fb97` | Static, sanitized CLI failure boundaries | Approved by `dplush`; merged as upstream `4c6b280`; full CI and CLA green |

## Open upstream PRs

| Priority | PR | Branch | Scope | Live status | Maintainer status |
|---:|---:|---|---|---|---|
| 1 | [#815](https://github.com/mnemosyne-oss/mnemosyne/pull/815) | `contrib/s1-recovery` | Backup/restore hardening | Mergeable | Awaiting maintainer review |
| 2 | [#816](https://github.com/mnemosyne-oss/mnemosyne/pull/816) | `contrib/s1-hygiene` | Hygiene transaction behavior | Conflicting | Awaiting maintainer review; rebase required |
| 3 | [#817](https://github.com/mnemosyne-oss/mnemosyne/pull/817) | `contrib/s1-model-refresh` | Dream/model-refresh configuration | Mergeable | Awaiting maintainer review |
| 4 | [#719](https://github.com/mnemosyne-oss/mnemosyne/pull/719) | `contrib/fix-health-precedence` | Beam consolidation health predicates | Conflicting | Earlier feedback addressed; current-head review still pending |
| 5 | [#720](https://github.com/mnemosyne-oss/mnemosyne/pull/720) | `contrib/fix-remember-embed-warning` | Warning for skipped single-item embeddings | Mergeable | Earlier feedback addressed; current-head review still pending |
| 6 | [#721](https://github.com/mnemosyne-oss/mnemosyne/pull/721) | `contrib/fix-shmr-local-llm` | Restore SHMR local-LLM path | Conflicting | No maintainer review yet |
| 7 | [#774](https://github.com/mnemosyne-oss/mnemosyne/pull/774) | `contrib/migrate-dry-run` | Report-only migration dry run | Conflicting | No maintainer review yet |
| 8 | [#829](https://github.com/mnemosyne-oss/mnemosyne/pull/829) | `contrib/recall-metadata-projection` | Opt-in allowlisted recall metadata projection | Mergeable; CI running | New focused enhancement PR |

## Superseded

| PR | Reason |
|---:|---|
| [#788](https://github.com/mnemosyne-oss/mnemosyne/pull/788) | Bundled scope split into focused PRs #814–#817 after maintainer feedback. |

## Published tracking issues

| Issue | Purpose | State |
|---:|---|---|
| [#827](https://github.com/mnemosyne-oss/mnemosyne/issues/827) | Deployment-informed correctness and reliability findings | Open; awaiting maintainer feedback |
| [#828](https://github.com/mnemosyne-oss/mnemosyne/issues/828) | Deployment-informed lifecycle and integration improvements | Open; awaiting maintainer feedback |

## Not yet published

- **S2** — Inhale ingest receipts and admission policy
- **S3** — Bounded recall / Exhale
- **S4** — Proposal-only SHMR harmony and lexical fallback
- **S5** — Dream lifecycle
- **S6** — Native SDK, CLI parity, and Dream failure exits
- **S7b** — Isolated snapshot/restore; blocked on post-replace writer-lock hardening
- **S7c** — MCP/doctor parity, packaging CI, and batch containment
- **S8** — Hermes `sync_turn` receipts
- **S9** — Codex lifecycle-hook plugin; manual desktop checkpoint remains

## Next action order

1. Rebase and repair #816, then run its focused hygiene/doctor suite.
2. Re-audit #815 and #817 against current `main`, address review-bot findings only when still valid, and request maintainer review.
3. Rebase conflicting #719, #721, and #774; refresh focused evidence.
4. Refresh #720 against current `main`; do not modify it unless current tests or review require it.
5. Review #829 as the first focused enhancement PR; keep #827/#828 as indexes.

## Rules

- Use isolated worktrees; never validate against the live Mnemosyne instance.
- Run tests with `TZ=UTC` and without leaked `MNEMOSYNE*` / `NVIDIA_EMBEDDING*` variables.
- Keep GitHub posts, PR descriptions, and maintainer replies in English.
- Exclude databases, memory content, credentials, host details, private paths, and operator trial tooling.
