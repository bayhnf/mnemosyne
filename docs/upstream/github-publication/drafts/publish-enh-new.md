## New proposals for maintainer guidance

<details>
<summary><strong>E01 — Instrumented provider-migration governance</strong></summary>

**Status:** `PROPOSED_NEW`, `IMPLEMENTED_LOCALLY`; no framework-level equivalent found in 315 issues / 500 PRs.

**Problem:** a replacement provider can start successfully while silently regressing recall precision, stale/current selection, scope isolation, durability, or rollback. Basic health checks do not prove migration safety.

**Deployment-informed evidence:** our private Hermes integration used a local workflow:

```text
source manifest → online backup → shadow capture → recall diff →
positive/negative controls → scope gates → go/no-go report →
canary write/recall/integrity → scheduled soak → cutover/rollback
```

Sanitized evaluated gates had zero mandatory failures. They covered recall, negative controls, stale/current behavior, scope leakage, latency, backup/restore, provider health, and rollback. A separate candidate canary passed import, write, recall, integrity, vector growth, runtime, and clean-console checks. Long-duration soak/canary gates remained explicitly scheduled—not reported as complete.

**Proposal:** an optional local toolkit with content-free manifests, bounded non-serving capture, recall-diff metrics, deterministic gate reports, canary/rollback helpers, and explicit operator acknowledgement for cutover or incomplete soak.

**Impact:** detects silent regressions before cutover; produces reproducible evidence; reusable across Hermes/MCP/CLI; no hosted telemetry.

**Compatibility/non-goals:** opt-in; no default provider change; no memory text in reports; no automatic cutover; not a second memory engine; no performance claim from incomplete soak.

**Possible focused PRs:** report schema/evaluator; shadow capture/diff; canary/rollback workflow; documentation.
</details>

<details>
<summary><strong>E02 — Safe recall metadata projection for policy-aware consumers</strong></summary>

**Status:** `PROPOSED_NEW`, `READY_FOR_GUIDANCE`; no read-side equivalent found. PR #644 is write-side scope mirroring.

**Fact:** on upstream `4c6b280`, custom `metadata_json` is persisted but absent from recall results. Selected scope/authorship/lifecycle fields are returned, but a policy-aware consumer must perform per-item DB lookups and table guessing.

```text
stored metadata: {project_id: p1, custom: marker}
recall result: metadata_json absent
consumer cost: up to N additional lookups for N results
```

**Impact:** N+1 SQLite reads, inconsistent hydration across working/episodic tiers, and consumer-specific provenance/scope drift.

**Design options:** explicit `include_metadata=False`; allowlisted projection; bulk hydration by `(tier,id)`; or a documented hydration seam.

**Preferred direction:** allowlisted projection or bulk hydration, not unconditional raw metadata, because metadata can carry private or large values.

**Acceptance:** no default payload/privacy expansion; one query per result page; consistent tier behavior; documented size/field cap; backward-compatible default.
</details>

We are willing to prepare focused design/implementation PRs only after maintainer guidance on E01/E02.