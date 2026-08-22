## Existing and partially covered improvement directions

These items connect our deployment evidence to official work. They are not presented as entirely new requests.

<details>
<summary><strong>Memory lifecycle — E04 Inhale, E05 Exhale, E06 Dream</strong></summary>

- **E04 — Receipt-backed ingest:** atomically persist memory + event receipt, run enrichment afterward, and transition `pending → ready/degraded/retryable/terminal`. This makes failed vector/fact enrichment observable and idempotently retryable. Adjacent: #727, #766, #789. Compatibility: additive tables; legacy `remember()` unchanged; no remote queue.
- **E05 — Bounded read-only recall:** hydrate retrieval voices, then apply active/scope gates, proposal exclusion, dedup, rank, `top_k`, per-item cap, and total token budget without mutating recall state. This complements roadmap #514 and validation proposal #449; it should be discussed there or as a maintainer-requested child issue.
- **E06 — Gated canonical mutation:** deterministic manifest, separate reviewer/verifier actors, source revalidation, one owned transaction, and before-image undo. Adjacent: #410, #434, #449; PR #817 is only configuration/fallback foundation. Disabled by default and additive.

**Impact:** observable ingest degradation, bounded prompt context, and reviewable/undoable canonical truth instead of silent auto-promotion.
</details>

<details>
<summary><strong>Operations and integrations — E07 snapshots, E08 diagnostics, E09 turn receipts, E10 Codex</strong></summary>

- **E07 — Isolated snapshots:** online page copy, checksum, restrictive permissions, staged integrity-checked replacement, fsync, rollback. Existing: #640 / PR #815. Publication remains blocked until post-replace writer-lock reacquisition is proven.
- **E08 — Content-free diagnostics/tool parity:** stable unavailable/degraded/healthy states and `tools/list == callable handlers`. Existing: #372/#728; merged #814/#758. Add deployment evidence to those threads, not a duplicate issue.
- **E09 — Durable Hermes turn receipts/provider parity:** stable event IDs and content-free saved/filtered/degraded/retryable outcomes for atomic user+assistant capture. Adjacent: #328/#655/#766.
- **E10 — Local Codex lifecycle integration:** SessionStart recall-only, UserPromptSubmit scoped recall, Stop/SessionEnd durable ingest, and a 0600 non-searchable acknowledged failure spool. Optional and local; depends on accepted ingest/recall contracts.

**Impact:** reliable recovery and agent lifecycle capture without cloud storage or changing defaults.
</details>

<details>
<summary><strong>Identity and trust policy — E03 and E11</strong></summary>

- **E03 — Multi-agent reader visibility:** our setup verified exact user/session/project visibility and fail-closed unknown readers. Existing overlap: #327/#653/#761. Contribute the visibility matrix as evidence; do not duplicate. Never guess project context for automatic prefetch.
- **E11 — Trust-aware capture/quarantine:** classify candidate writes, quarantine secret/injection-shaped content, and distinguish proposed/verified/superseded lifecycle before promotion. Existing overlap: #109/#410/#821 and write-approval/security work. Reuse admission/veracity/lifecycle primitives rather than add a second policy engine.

**Impact:** safer multi-agent isolation and auto-capture; retrieved untrusted text is not silently promoted to authoritative instruction or canonical truth.
</details>

### Suggested order

Resolve #815–#817 first, then seek focused guidance: E01/E02 design decisions → E04 ingest → E05 recall → E07 snapshots/E08 diagnostics → E06 Dream → E09 receipts → E10 Codex. Existing issues receive evidence directly.