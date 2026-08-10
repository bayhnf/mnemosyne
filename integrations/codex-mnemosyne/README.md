# Mnemosyne Codex Plugin

Upstream-ready lifecycle-hook integration that makes [Mnemosyne](https://github.com/mnemosyne-oss/mnemosyne) the sole persistent memory provider for Codex.

## What it does

Hooks Codex's native lifecycle events to Mnemosyne's durable native ingest and bounded recall:

| Event | Behavior |
|---|---|
| `SessionStart` (startup, resume, clear, compact) | Bounded recall of identity/preferences (≤6 items / ≤800 tokens), injected as `additionalContext` |
| `UserPromptSubmit` | Durably ingests the prompt with a stable event ID, then bounded recall (≤8 items / ≤1200 tokens) |
| `Stop` | Durably ingests the acknowledged assistant message. Does **not** parse transcripts |
| `SessionEnd` | Flushes the transport-only spool (under 3 seconds) |

## Design constraints

- **No transcript parsing.** Uses only structured fields from hook payloads.
- **Stable event IDs.** Deterministic from `session_id + turn_id + role`, so replays are idempotent.
- **Bounded recall.** Hard caps on items and tokens; never unbounded.
- **Fail-open for Codex.** Memory failures never block the session. Failures surface as visible, non-sensitive `systemMessage` warnings.
- **0600 transport-only spool.** Failed deliveries are persisted in a mode-0600 SQLite file that is never searchable or recallable. Rows are deleted only after native receipt acknowledgement.
- **stdlib JSON only.** No third-party dependencies in the hook path.
- **Mnemosyne is the sole memory provider.** Built-in Codex memory stays disabled. This plugin adds no MCP tools.

## Configuration

Environment variables (all optional):

| Variable | Default | Purpose |
|---|---|---|
| `MNEMOSYNE_DATA_DIR` | `~/.hermes/mnemosyne/data` | Mnemosyne data directory |
| `MNEMOSYNE_CODEX_ACTOR_ID` | `codex-actor` | Actor identity for ingest |
| `MNEMOSYNE_CODEX_ACTOR_TYPE` | `human` | Actor type |
| `MNEMOSYNE_CODEX_PROJECT_ID` | SHA-256(cwd)[:12] | Project/channel identity |
| `MNEMOSYNE_CODEX_SPOOL_PATH` | `<data_dir>/codex-spool.db` | Transport spool path |

## Testing

```bash
python3 -m pytest integrations/codex-mnemosyne/tests/ -v
```

26 tests cover: manifest/hook contract, stable event IDs, bounded context, post-compact re-hydration, visible failures, 0600 spool creation, spool non-searchability, ack-based deletion, hook timeout, no transcript parsing, and a full end-to-end session lifecycle smoke test.
