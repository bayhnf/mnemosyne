# Mnemosyne Codex Plugin

Upstream-ready lifecycle-hook integration that makes
[Mnemosyne](https://github.com/mnemosyne-oss/mnemosyne) the sole persistent
memory provider for Codex.

## What it does

Hooks Codex's native lifecycle events to Mnemosyne's durable native ingest and
bounded recall:

| Event | Behavior |
|---|---|
| `SessionStart` (startup, resume, clear, compact) | Bounded recall of identity/preferences (≤6 items / ≤800 tokens), injected as `additionalContext`. Performs no ingest. |
| `UserPromptSubmit` | Durably ingests the prompt with a stable event ID, then bounded recall (≤8 items / ≤1200 tokens) |
| `Stop` | Durably ingests the acknowledged assistant message. Does **not** parse transcripts |
| `SessionEnd` | Flushes the transport-only spool under a hard 2-second deadline (well within Codex's 3-second ceiling) |

## Prerequisites

The hooks import the `mnemosyne` Python package directly (no MCP, no
subprocess). Install it before enabling the plugin:

```bash
pip install mnemosyne
```

or, from a source checkout:

```bash
pip install -e /path/to/mnemosyne
```

If the package is not importable at hook runtime, every hook emits a safe,
actionable `systemMessage` warning naming `mnemosyne` and the install step,
then exits 0 (fail-open). No traceback, no silent failure.

## Persistent cross-session memory

Memory is **persistent across Codex sessions** and **isolated per actor +
project**. The hook derives one deterministic opaque *memory scope* from
`actor_id + project_id`:

```
memory_scope = "mem-" + SHA-256(actor_id | project_id)[:16]
```

This scope is the single key used for **both** native ingest
(`IngestEvent.session_id`) and bounded recall (`Mnemosyne.session_id`), so:

- Same actor + project, session-A → session-B: **recalls** (persistent memory).
- Different actor OR different project: **never recalls** (isolation).

The scope never widens to global/shared and never leaks raw actor, project, or
path values. The ephemeral Codex `session_id` is kept only in event metadata
for non-recall provenance.

## Stable host IDs

`UserPromptSubmit` and `Stop` use `payload.turn_id` (the host turn id) when
nonempty, with a deterministic content fallback only if `turn_id` is absent.
This makes hook replays idempotent: the same logical turn always produces the
same `event_id`, so native ingest deduplicates and the spool never accumulates
duplicates.

## Configuration

Environment variables (all optional). In an installed plugin, Codex sets
`PLUGIN_DATA` and `PLUGIN_ROOT` automatically.

| Variable | Default | Purpose |
|---|---|---|
| `PLUGIN_DATA` | (set by Codex) | Writable plugin state dir (spool + default data) |
| `MNEMOSYNE_DATA_DIR` | `<PLUGIN_DATA>` | Mnemosyne database directory |
| `MNEMOSYNE_CODEX_ACTOR_ID` | `codex-actor` | Actor identity for memory scope |
| `MNEMOSYNE_CODEX_ACTOR_TYPE` | `human` | Actor type |
| `MNEMOSYNE_CODEX_PROJECT_ID` | SHA-256(cwd)[:12] | Project/channel identity |
| `MNEMOSYNE_CODEX_SPOOL_PATH` | `<PLUGIN_DATA>/codex-spool.db` | Transport spool path |

## Design constraints

- **No transcript parsing.** Uses only structured fields from hook payloads.
- **Stable event IDs.** Deterministic from `scope + turn_id + role`; replays
  are idempotent.
- **Bounded recall.** Hard caps on items and tokens; never unbounded.
- **Fail-open for Codex.** Memory failures never block the session. Each
  failure condition has a distinct, honest, content-free `systemMessage`.
  Exit 0 is preserved for all normal operational failures.
- **Honest failure reporting.** Never claims an event was queued unless its
  durable spool write succeeded. Never surfaces raw exception messages,
  content, ids, hashes, scope, or transcript paths. `SessionStart` never
  claims an event was queued.
- **0600 transport-only spool.** Failed deliveries are persisted in a
  mode-0600 SQLite file that is never searchable or recallable. Rows are
  deleted only after native receipt acknowledgement.
- **Bounded spool.** Finite row capacity (32), idempotent event IDs, and a
  terminal retry state (8 attempts) so an outage cannot grow without bound
  or retry indefinitely. Corrupt rows are retained, never silently deleted.
  All deletion is ack-only.
- **Bounded SessionEnd.** Returns before the 3-second Codex ceiling even if a
  native ingest attempt is slow or hung, using a SIGALRM-based deadline. Does
  not rely solely on Codex forcibly killing the hook. Unacknowledged events
  are always retained.
- **stdlib JSON only.** No third-party dependencies in the hook path.
- **Mnemosyne is the sole memory provider.** Built-in Codex memory stays
  disabled. This plugin adds no MCP tools.

## Testing

```bash
python3 -m pytest integrations/codex-mnemosyne/tests/ -v
```

50 tests cover: manifest/hook contract, persistent cross-session recall,
actor/project isolation, stable host turn IDs, non-object payload fail-open,
honest content-free messages, bounded spool (0600, idempotent, finite
capacity, terminal state, corrupt-row retention, ack-only deletion), slow
SessionEnd, installed-plugin PLUGIN_DATA default, absent-package warning,
post-compact re-hydration, and a full end-to-end session lifecycle.
