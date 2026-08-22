## E01 — Instrumented provider migration governance

**Status:** `PROPOSED_NEW`, `IMPLEMENTED_LOCALLY`  
**Duplicate audit:** no framework-level equivalent found.

### Problem

A provider can start successfully while silently regressing recall precision, stale/current selection, identity isolation, durability, or rollback. Basic health checks do not prove migration safety.

### Deployment-informed evidence

Our real Hermes integration used an opt-in local workflow:

```text
source manifest → online backup → shadow capture → recall diff →
positive/negative controls → scope gate → go/no-go report →
canary write/recall/integrity → scheduled soak → cutover/rollback
```

Sanitized results:

- mandatory evaluated gates: pass;
- failed mandatory gates: zero;
- recall, negative-control, stale/current, scope-leak, latency, backup/restore, persona/lease, and rollback gates were represented;
- canary import, write, recall, integrity, embedding growth, runtime, and console checks passed;
- long-duration soak/canary gates remained explicitly scheduled rather than being claimed complete.

### Proposed improvement

An optional local migration-evaluation toolkit:

1. content-free source/candidate manifests;
2. bounded non-serving shadow capture ledger;
3. recall-diff records using IDs/metrics, not memory text;
4. deterministic threshold evaluator and go/no-go schema;
5. canary and rollback drill helpers;
6. explicit operator acknowledgement for cutover and incomplete soak gates.

### Impact

- Detects silent quality and scope regressions before cutover.
- Makes provider upgrades reproducible and auditable.
- Gives maintainers comparable evidence instead of anecdotal “works for me.”
- Reusable across Hermes, MCP, CLI, and other integrations.

### Compatibility / non-goals

Opt-in; no default provider change; no hosted telemetry; no automatic cutover; no second memory engine; no performance claim from incomplete soak.

### Suggested PR decomposition

- [ ] Report schema + evaluator
- [ ] Shadow capture/diff command
- [ ] Canary + rollback workflow
- [ ] Documentation and sanitized example

---

## E02 — Safe recall metadata projection

**Status:** `PROPOSED_NEW`, `READY_FOR_GUIDANCE`.

### Fact and example

On upstream `4c6b280`, a memory stored with custom `metadata_json` is recalled without that metadata. The result exposes selected fields (`scope`, authorship, veracity, lifecycle, scores), while the custom metadata remains only in SQLite. Our policy-aware provider therefore performs per-result table lookups to recover identity/project metadata.

```text
stored metadata: {project_id: p1, custom: marker}
recall result: metadata_json absent
consumer cost: up to N additional lookups for N results
```

### Impact

- N+1 SQLite reads for policy-aware consumers.
- Working/episodic table guessing to hydrate metadata.
- Greater risk of consumer-specific scope/provenance drift.

### Proposed options

A maintainer decision is needed because returning arbitrary metadata has privacy and payload costs:

1. add an explicit `include_metadata=False` recall option;
2. expose an allowlisted metadata projection only;
3. add a bulk hydration API keyed by `(tier, id)`;
4. keep recall unchanged but document the supported hydration seam.

Preferred starting point: allowlisted projection or bulk hydration, not unconditional raw `metadata_json`.

### Acceptance

No content/path/secret expansion by default; one query for a result page; consistent working/episodic behavior; documented size cap; backward-compatible default.

---

## E03 — Multi-agent reader-side visibility policy

**Status:** `EXISTING/PARTIAL`: #327, #653, #761.

Our deployment validated user/session/project visibility with exact project matching and fail-closed unknown readers. The useful contribution is additional sanitized evidence for #327/#653, not a duplicate feature issue.

Direction: map Hermes identity once, persist explicit scope metadata, apply the same reader policy on recall, and avoid guessing project context for automatic prefetch. Multi-token auth (#761) can supply stable actor identity, but authorization/visibility remains a separate policy layer.

Please review E01/E02 before implementation; we are willing to split accepted work into focused PRs.