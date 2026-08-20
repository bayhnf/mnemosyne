# Verified Bugs and Fixes

**Baseline:** `85a4b88`. **Merge base:** `6363a4b`. **Branch HEAD:** `770602a`.

Each entry below states a bug class the branch's code prevents, scoped to what
the code actually prevents — not more. Where a prior internal report
overstated a claim, the correction is stated explicitly. "Native path" means
the new Inhale/Exhale/Dream APIs; the legacy `remember()`/`recall()`/`sleep()`
contract is untouched unless stated.

---

## 1. Transaction nesting on the native write/apply path

**Failure class:** `sqlite3.OperationalError: cannot start a transaction
within a transaction` when a caller opens or defers a transaction around a
write. Tracked upstream as #489 for the legacy `task_progress` path.

**What the branch prevents (native path only):** Inhale raises
`_InhaleTransactionError` and Dream rejects caller-open / `_defer_commit`-
flagged contexts *before any mutation*, so the native write and Dream-apply
paths cannot hit this class.

**What it does not prevent:** the legacy `remember()` / `recall()` /
`sleep()` paths that triggered #489 are unchanged. Operators using the legacy
contract are unaffected and still subject to the upstream bug.

**Evidence:** `mnemosyne/core/inhale.py` transaction-context guard;
`mnemosyne/core/dream.py` ownership reject.

---

## 2. Silent embedding orphan on enrichment failure (native path)

**Failure class:** a memory row is committed, but its embedding is never
written (or is later deleted by a concurrent trim), leaving the row
silently unembeddable. Related to upstream #491.

**What the branch prevents (native path only):** Inhale writes the raw memory
row, the `ingest_receipts` row, and the sync event in one transaction, then
runs enrichment *outside* that transaction. On enrichment failure the receipt
transitions to `failed_retryable` and `retry_pending_ingest()` can re-run it
idempotently. The orphan is now observable and recoverable rather than silent.

**What it does not prevent:** the legacy `remember()` trim ordering and
the legacy enrichment path are untouched.

**Evidence:** `mnemosyne/core/inhale.py` receipt state machine
(`pending → ready | degraded | failed_retryable | failed_terminal`).

---

## 3. Ungated canonical mutation (model-refresh auto-apply)

**Failure class:** the model-refresh path applies canonical mutations
automatically without a review/verify gate.

**What the branch adds (new capability, not a legacy fix):** Dream provides a
gated lifecycle (`planning → awaiting_approval → ready → applying → applied`)
where transition to `ready` requires a PASS reviewer receipt followed by a
PASS verifier receipt with a *different* `actor_id` (no self-approval).
`dream_active` defaults to `False`.

**What it does not change:** the existing model-refresh auto-apply path is
unchanged. Dream is an additive alternative, not a replacement.

**Evidence:** `mnemosyne/core/dream.py` `RECEIPT_ROLES`, manifest-hash
binding, `dream_active` default.

---

## 4. Content leakage via public error surfaces

**Failure class:** ingest receipts, Dream projections, snapshot errors, sync
diagnostics, and MCP error handlers that expose raw exception text, file
paths, SQL fragments, or memory content.

**What the branch prevents:** every new public status/error surface returns a
structured `error_code` / state field, never raw exception text.
`SnapshotError` messages never contain file paths; `_assert_mode` raises with
a content-free message (using an explicit `raise`, not `assert`, so `python
-O` cannot strip it). MCP error handlers project to a closed vocabulary.

**Evidence:** `SnapshotError`, `_assert_mode`, `redact_memory_artifact`,
`mcp_tools.py` error projection; `doctor.py` content-free findings.

---

## 5. Batch tool error payload leakage

**Failure class:** `batch_tool.py` error fields exposed `str(exc)` /
`{type(exc).__name__: exc}`, leaking arbitrary exception detail.

**What the branch prevents:** error fields are replaced with stable codes
(`batch_validation_failed`, `batch_failed`); `logger.exception` →
`logger.error` (no traceback); an `_ALLOWED_BATCH_ACTIONS` allowlist prevents
an untrusted `action` string from round-tripping into the error payload.

**Evidence:** `mnemosyne/batch_tool.py`.

---

## 6. SHMR offline fallback unreachable on base installs without NumPy

**Failure class:** `shmr.py` imported `numpy` unconditionally at module top.
On a base install without NumPy, the import failed and the offline/lexical
fallback path (`_lexical_vector` / `_cosine_similarity`) was unreachable —
even though that path was the documented degradation for the embeddings-off
case.

**What the branch prevents:** `numpy` is now an optional import (guarded;
`np = None` on `ImportError`). The lexical fallback path
(`_lexical_vector`, `_cosine_similarity`) is rewritten in pure stdlib
(`math.sqrt`, list comprehensions) so it works without NumPy installed.
`from __future__ import annotations` keeps the `np.ndarray` type hints
unevaluated when `np is None`.

**Important scope correction (prior report was imprecise):** this is a
**lexical-fallback fix, not a "no-NumPy" fix**. The `numpy` import remains
in the module; the dense-only paths (`_embed`, `_compute_harmony_score`,
`recall_beliefs`) still call `np.*` and still require NumPy. Those paths are
only entered when an embedding backend is configured. No packaging change,
no new dependency. The fix makes the offline lexical path reachable on a
base install; it does not make the dense path NumPy-free.

**Evidence:** `mnemosyne/core/shmr.py` (guarded import at ~L27; stdlib
`_cosine_similarity` / `_lexical_vector`); the `numpy` import line is still
present in the module.

---

## 7. Unbounded recall (no hard result/token caps, no strict isolation)

**Failure class:** legacy recall has no hard top-k / token budget / strict
identity-isolation gate, so a recall call can return more than intended or
cross identity scopes.

**What the branch adds (new capability):** Exhale is a read-only,
post-hydration gate with hard `top_k` / `max_tokens` / `max_item_tokens`
caps, identity allowlists, strict-int validation (rejects `bool` and
`float`), and a deterministic fallback (lexical FTS5-only when
`require_fallback=True`).

**What it does not change:** legacy `recall()` / `recall_enhanced()` /
`_recall_polyphonic()` are untouched. Exhale never calls the side-effecting
legacy paths; it never bumps `recall_count` / `last_recalled`.

**Evidence:** `mnemosyne/core/recall_bounded.py` `RecallPolicy`,
`RecallEnvelope`, `_PROPOSAL_SOURCES` exclusion.

---

## 8. Unreviewed snapshot restore (inode-lock race on replace)

**Known limitation, not yet fixed:** the snapshot restore path reuses the
staged atomic-replace (writer lock → staged rebuild → fsync → atomic replace
→ post-replace integrity check with rollback). A post-replace writer-lock
re-acquisition hardening step was designed but **is not yet implemented** in
this branch. Slices carrying the snapshot module must not be opened until
that hardening lands. See [feature-roadmap.md](feature-roadmap.md) §S7b.

---

## Corrections to claims in prior internal reports

The following prior claims are **not** carried forward as verified, because
the evidence does not support them at the strength stated:

- **"Long document context window causes silent embed skip."** The cause is
  **unproven**. A single operational instance of an unembedded long row was
  observed and fixed during a local audit; a later re-test of the same
  embedding service embedded documents of comparable size successfully. The
  incident is consistent with a transient backend condition, not with a
  proven context-window limitation. Do not cite a context-window cause.
- **"Orphan embeddings indicate a bug."** A small number of orphan embedding
  rows (embeddings pointing at deleted memory IDs) are **expected
  maintenance debris** from normal deletes; `reclaim_orphans()` and the
  hygiene/doctor surface are the intended maintenance path, not a bug fix.
- **"sqlite_vec loadable_path `.so` suffix is a code bug."** The `.so`
  suffix issue was a **CLI-only operational workaround** observed in manual
  scripts; production code loads `sqlite_vec` via `sqlite_vec.load(conn)`,
  not via a `.so` path. It is not a defect in this branch's code.
- **"Empty LLM env config is a code/config defect in this branch."** An empty
  `MNEMOSYNE_LLM_ENABLED` was an **environment misconfiguration** observed on
  one host; it is already tracked upstream via the existing
  `local_llm`/`MnemosyneConfig` path and is not introduced or worsened by
  this branch.
- **"Zero errors since restart."** Any "error count since restart" figure is
  an **operational snapshot from one host at one time** and is not a reliable
  upstream claim. Do not cite error counts as evidence of correctness.
