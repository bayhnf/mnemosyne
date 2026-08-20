# Feature Roadmap — Dependency-Ordered PR Slicing

**Baseline:** `85a4b88`. **Merge base:** `6363a4b`. **Branch HEAD:** `770602a`.

This is the review plan. Each slice is a candidate PR. Dependency order is
strict for S1–S6; S7a–S9 follow with their own prerequisites. Every slice
should rebase onto the then-current `upstream/main` at PR-open time, re-run
its focused suite, and bump the MINOR per Simple Versioning with a CHANGELOG
`[Unreleased]` entry.

---

## Explicit exclusion: on-host trial harness

`scripts/linuxprocessing_campaign.py` and `tests/test_linuxprocessing_campaign.py`
are **operator/trial tooling** and are excluded from every slice below. They
are named for a specific deployment, carry trial vocabulary, and have no
product value upstream. No slice in this roadmap includes them.

---

## Slice plan

| Slice | Theme | Depends on | Status |
|---|---|---|---|
| **S1** | DR/hygiene/beam/config maintenance + fail-closed restore + CLI error containment | — | Code-ready |
| **S2** | Native Inhale ingest receipts + retry ownership + admission policy | S1 | Code-ready |
| **S3** | Bounded recall (Exhale) + metadata/observability reconciliation | S2 | Code-ready |
| **S4** | Proposal-only SHMR candidate generator + stdlib lexical fallback | S3 | Code-ready |
| **S5** | Dream lifecycle (plan/apply/undo, provenance, CAS) | S4 | Code-ready after 2 hygiene fixes |
| **S6** | Native SDK + CLI parity + Dream failure exit signals | S5 | Code-ready |
| **S7a** | Migration report-only dry run | — | Code-ready; can open alongside S1 |
| **S7b** | Isolated read-only snapshot/restore | S1 + **pending lock hardening** | **Blocked** on post-replace lock fix |
| **S7c** | MCP/doctor tool parity + packaging CI + batch error containment | S6 | Code-ready |
| **S8** | Hermes `sync_turn` native turn receipts (both provider mirrors) | S2 | Code-ready |
| **S9** | Codex lifecycle-hook plugin | S2 + S6 | Code-ready; manual desktop checkpoint outstanding |

### Dependency edges

- S2 depends on S1 (Inhale's `beam.py` receipt block sits on S1's transaction fixes).
- S3 depends on S2 (recall metadata contract).
- S4 depends on S3 (SHMR proposal references recall scope).
- S5 depends on S4 (`dream_plan` consumes SHMR proposals).
- S6 depends on S5 (CLI/MCP surface exposes Dream verbs).
- S7a is file-disjoint from S1–S6 and can open early, in parallel with S1.
- S7b depends on S1's `recovery.py` helpers **and** on a post-replace
  writer-lock re-acquisition fix that is **not yet implemented** in this
  branch. S7b must not open until that fix lands.
- S7c depends on S6 (MCP surface).
- S8 depends on S2 (`remember_turns_atomic` added by the Inhale wave).
- S9 depends on S2 (admission classifier) and S6 (public surface).

### Recommended landing order

S1 → S2 → S3 → S4 → S5 → S6, then S7a (can open alongside S1) → S7b (after
the lock fix) → S7c, then S8 / S9. Rebase later slices onto merged upstream
`main`.

---

## Slice detail

### S1 — DR/hygiene/beam/config + CLI error containment
DR fail-closed restore, hygiene audit hardening, beam transaction/degrade
fixes, `dream_active` config seam, and CLI content-free error boundaries.
Lowest risk; stabilizes CI for everything below.

### S2 — Inhale (durable receipt-backed ingest)
New `inhale.py` + `ingest_receipts`/`ingest_conflicts` schema + CLI/MCP
surface + admission policy + content-free enrichment logs. Addresses the
transaction-nesting class (#489) and the silent-embedding-orphan class
(#491) for the native path.

### S3 — Exhale (bounded read-only recall)
New `recall_bounded.py` + `RecallPolicy`/`RecallEnvelope` + CLI/MCP surface +
polyphonic fallback content-free logs. Read-only; no data change.

### S4 — SHMR `propose_harmony` + stdlib lexical fallback
Read-only candidate generator for Dream, plus the stdlib `_lexical_vector` /
`_cosine_similarity` rewrite so the offline fallback is reachable without
NumPy (dense paths still require NumPy — see [verified-bugs.md](verified-bugs.md) §6).

### S5 — Dream lifecycle
New `dream.py` + `dream_runs`/`dream_actions`/`dream_receipts` schema +
CLI/MCP surface. Requires two hygiene fixes before opening: (a) commit the
`test_dream_lifecycle.py` docstring fix that removes a private path; (b)
rewrite the `dream.py:20` docstring that references internal plan files.

### S6 — Native SDK + CLI parity + Dream failure exits
CLI/MCP verb surface, Dream failure-exit signals, `recall_bounded` bridge.

### S7a — Migration report-only dry run
Idempotent `CREATE TABLE IF NOT EXISTS` for `memory_events` / `sync_meta` +
`migrate --dry-run`. Proven by `test_migration_dry_run_fingerprint.py`
(schema fingerprint + byte-size identical). No dependency on S1–S6.

### S7b — Isolated read-only snapshot/restore
New `dr/snapshot.py` + tests. **Blocked** on the post-replace writer-lock
re-acquisition fix (designed, not yet implemented). Do not open until that
fix lands and the competing-writer probes pass.

### S7c — MCP/doctor tool parity + packaging CI + batch containment
`mcp_tools.py`, `doctor.py`, `tool_schemas.py`, CI lanes, `setup.py`,
`uv.lock`, and the `batch_tool.py` error-containment fix.

### S8 — Hermes `sync_turn` receipts
Both provider mirrors (`hermes_memory_provider/` and
`integrations/hermes/src/mnemosyne_hermes/`) + content-free tool-JSON
containment + real-beam sync-turn receipt tests.

### S9 — Codex lifecycle-hook plugin
`integrations/codex-mnemosyne/` (SessionStart/UserPromptSubmit/Stop/SessionEnd
hooks). Self-contained plugin with its own tests. A manual desktop
checkpoint (plugin install/activation + four-hook verification) cannot be
automated in CI and remains outstanding.

---

## Versioning

Each user-facing slice bumps the MINOR per Simple Versioning (single source
`mnemosyne/__init__.py`) and adds a CHANGELOG `[Unreleased]` entry. The
branch does not bump `__version__` past `3.16.0`; each slice carries its own
bump. Breaking-change call-outs (none currently expected) get a MAJOR bump.

## CLA

Confirm the contributing GitHub account has signed the project CLA
(`CLA.md`, adapted from the Apache Software Foundation Individual CLA) via
CLA-assistant before the first PR. No prior merged contribution is on record
for this account.
