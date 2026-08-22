## E04 — Durable receipt-backed ingest (Inhale)

**Status:** `READY_FOR_GUIDANCE`; adjacent #727, #766, #789.

### Problem

A write can commit text while enrichment fails later, leaving degradation implicit. Retries also need ownership and idempotency across workers.

### Proposed behavior

`remember_event()` / `remember_turn()` create one durable receipt keyed by caller event ID. The raw memory, receipt, and sync event commit atomically. Enrichment runs outside the write transaction and transitions the receipt:

```text
pending → ready | degraded | failed_retryable | failed_terminal
```

Retries claim work through bounded leases, preserve the original memory ID, and append conflicts rather than silently overwriting evidence.

### Data/example

```text
write committed + embedding endpoint unavailable
legacy outcome: row exists, vector may silently remain absent
receipt outcome: state=failed_retryable, attempts=N, safe error_code, retry available
```

### Impact

- Observable and recoverable enrichment failures.
- Idempotent agent/hook retries.
- Better auditability across Hermes, MCP, CLI, and lifecycle hooks.
- Clear transaction ownership prevents nested-BEGIN failures.

### Compatibility / non-goals

Additive tables only; legacy `remember()` remains unchanged; no remote queue; no raw exception/content in receipts. The proposal should align with streaming-event design #766 and calibrated admission #789 rather than duplicate them.

---

## E05 — Hard-bounded read-only recall (Exhale)

**Status:** `EXISTING/PARTIAL`; overlaps roadmap #514 and proposal #449.

### Problem

Agent-facing recall needs hard result/token limits and identity/lifecycle gates independent of which retrieval voice produced candidates. Raising global `top_k` is not a safe solution.

### Proposed behavior

A read-only post-hydration envelope:

```text
hydrate voices → active/scope predicate → proposal exclusion → dedup →
rank → top_k → per-item cap → total token budget
```

It must never mutate recall counters, and fallback must be deterministic.

### Impact

- Predictable prompt size and latency.
- Stronger session/project isolation.
- Prevents pending mutation proposals from entering ordinary context.
- Gives evidence-pack work in #514 a bounded policy seam.

### Compatibility

Opt-in API; legacy recall unchanged; source IDs and provenance preserved. This should be discussed inside #514 or as a focused child issue if maintainers request it.

---

## E06 — Reviewer/verifier-gated canonical mutation (Dream)

**Status:** `EXISTING/PARTIAL`; adjacent #410, #434, #449 and PR #817.

### Problem

Automatically derived canonical mutations can become durable truth without independent review, while deletion loses provenance.

### Proposed lifecycle

```text
planning → awaiting_approval → ready → applying → applied
                                      ↘ rejected / failed
applied → undoing → undone
```

A deterministic semantic manifest binds reviewer and verifier receipts from different actors. Apply revalidates source hashes and owns one transaction. Undo uses this run's before-images.

### Impact

- Prevents self-approved canonical mutation.
- Preserves audit, provenance, and rollback.
- Addresses canonical overuse without making extraction silently authoritative.

### Compatibility / non-goals

Disabled by default; additive tables; no replacement of legacy sleep/recall; no auto-apply without explicit policy. PR #817 is only configuration/fallback foundation, not the complete lifecycle. Maintainer guidance should determine whether this belongs under #449/#434 or a new focused design issue.

We are willing to contribute E04–E06 in dependency order after existing S1 PRs are resolved.