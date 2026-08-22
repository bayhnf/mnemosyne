## E07 — Isolated page-level snapshot and restore

**Status:** `EXISTING/PARTIAL`, `BLOCKED`; overlaps #640 and PR #815.

### Problem and proposed behavior

Virtual-table databases need page-level backup, checksum, restrictive permissions, staged restore, integrity checks, fsync, atomic replacement, and rollback. Our canary proved online-backup write/recall/integrity behavior, but snapshot publication remains blocked until post-replace writer-lock reacquisition is proven against a competing writer.

### Impact

Reliable disaster recovery for live WAL-backed stores without treating SQL dumps as physical backups.

### Compatibility

New API; existing backup remains available; artifacts 0600/directories 0700; no destructive migration. Continue in #815/#640 rather than opening a duplicate.

---

## E08 — Content-free diagnostics and tool-surface parity

**Status:** `EXISTING/PARTIAL`; #372, #728; merged #814 and #758.

### Problem

A provider may degrade while health looks green, and schemas advertised without handlers create tools that always fail. Error output can also leak content/path details.

### Direction

One content-free diagnostic vocabulary across CLI, MCP, provider, ingest, Dream, sync, vector coverage, and recovery. `tools/list` must equal callable handlers. Health should distinguish unavailable, degraded, and healthy without returning memory content.

### Impact

Automation can act on stable states; operators see vector/scope/retry failures; no sensitive data in logs. Add focused evidence to #372 rather than creating a duplicate health issue.

---

## E09 — Durable Hermes turn receipts and provider parity

**Status:** `EXISTING/PARTIAL`; #328, #655, #766.

### Problem

Post-turn capture crosses provider mirrors and can partially succeed before a later step fails. Without a receipt, callers cannot distinguish saved, filtered, degraded, or retryable outcomes. Two provider copies also drift.

### Proposed behavior

Store user+assistant turn events atomically with stable event IDs and content-free receipts. Retries are idempotent. Root and packaged provider share one contract/test matrix.

### Impact

Reliable lifecycle capture, no duplicate turns after retry, observable filtering, and less provider drift. Coordinate with transaction-aware events #766 and provider consolidation #655.

---

## E10 — Local Codex lifecycle integration

**Status:** `READY_FOR_GUIDANCE`.

### Problem

Coding agents lose project decisions across sessions. MCP tools alone require explicit calls and do not align capture/recall with lifecycle events.

### Proposed behavior

A local plugin mapping:

- SessionStart: bounded recall only;
- UserPromptSubmit: scoped bounded recall;
- Stop: durable turn ingest;
- SessionEnd: durable tail ingest;
- failed delivery: 0600 transport-only spool with acknowledgement-based removal.

### Impact

Consistent project memory without transcript scraping, cloud storage, or changes to Mnemosyne defaults. Scope/project identity remains explicit.

### Compatibility / non-goals

Self-contained optional integration; uninstall restores built-in behavior; no automatic disabling of another provider; no searchable spool; depends on accepted ingest and bounded-recall contracts.

---

## Suggested sequencing

Resolve open PRs #815–#817 first, then seek focused maintainer guidance in this order: E01/E02 design decision, E04 ingest, E05 bounded recall, E07 snapshot, E08 diagnostics, E06 Dream, E09 provider receipts, E10 Codex integration.

---

## E11 — Trust-aware capture and prompt-injection quarantine

**Status:** `EXISTING/PARTIAL`; overlaps closed proposals #109/#410 and runtime write-filter issue #821.

Our setup classifies candidate writes, quarantines secret/prompt-injection-shaped content, and separates proposed/verified/superseded lifecycle states before durable promotion. The impact is safer auto-capture: retrieved untrusted text is not silently treated as authoritative instruction or canonical truth.

The upstream direction should reuse existing admission, veracity, write-approval, and lifecycle primitives rather than add another policy engine. Any focused proposal must define trust provenance at write time, recall behavior for proposed/quarantined rows, explicit promotion/rejection, and content-free diagnostics. Since this area already has substantial prior discussion, contribute evidence to existing issues or seek maintainer guidance before opening a child issue.