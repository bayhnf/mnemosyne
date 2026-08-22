## B15 — Batch mutation failures leak backend exception details

**Status:** `VERIFIED_NEW`  
**Upstream baseline:** `4c6b280`  
**Duplicate audit:** no semantic duplicate found across 315 issues and 500 PRs.

### Discovery source

Our customized setup applies a content-free error policy at agent/tool boundaries. Auditing sibling batch paths found that `mnemosyne_batch` still returns arbitrary backend exception text and logs a full traceback.

### Independent reproduction

A sanitized fixture replaced one batch operation with:

```python
raise RuntimeError("PRIVATE_CANARY_PATH_/tmp/private")
```

Observed:

```text
payload.error = "RuntimeError: PRIVATE_CANARY_PATH_/tmp/private"
log_has_canary = true
log_has_traceback = true
```

The behavior comes from `apply_beam_batch()` returning `f"{type(exc).__name__}: {exc}"` and calling `logger.exception(...)`. Validation payloads also echo raw validation messages and the supplied action.

### Impact

- Tool/MCP/agent callers receive arbitrary backend details.
- Logs retain traceback frames and potentially private content or paths.
- Batch behavior violates the static/content-free contract now used by CLI and other public surfaces.
- An untrusted action string can be reflected by validation output unless constrained before projection.

### Root cause

The shared batch implementation treats exception formatting as its public error contract instead of projecting failures to a closed vocabulary.

### Suggested solution

- Return `batch_validation_failed` for validation failures and `batch_failed` for execution failures.
- Keep `failed_index` and an action only after the action passes the allowlist.
- Replace traceback logging with content-free structured/error logging.
- Preserve transaction rollback and audit-event-after-commit semantics.
- Do not include exception class/message in serialized payloads.

### Acceptance tests

- [ ] Inject a private canary in validation and execution exceptions.
- [ ] Assert payloads contain only the static code, safe index, and allowlisted action.
- [ ] Assert logs contain no canary, path, exception text, or traceback.
- [ ] Assert all earlier operations roll back on failure.
- [ ] Assert audit events publish only after a successful commit.
- [ ] Assert unknown actions are never reflected verbatim.

We are willing to submit this as a focused shared-boundary PR.

---

## B16 — SHMR import breaks the supported no-NumPy base install

**Status:** `VERIFIED_NEW`  
**Upstream baseline:** `4c6b280`  
**Duplicate audit:** no SHMR-specific semantic duplicate found. PR #31 established the base-install contract but did not cover `shmr.py`.

### Independent reproduction

A subprocess blocked `numpy` imports and imported `mnemosyne.core.shmr` with embeddings disabled:

```text
ModuleNotFoundError: blocked numpy for base-install probe
```

Current `shmr.py` imports `numpy as np` unconditionally at module import. Therefore callers cannot reach any degradation/fallback behavior on a documented base install without optional dense dependencies.

### Impact

- Importing SHMR crashes on the supported minimal package installation.
- Optional functionality can break callers merely by importing the module.
- Embeddings-off/local lexical operation cannot degrade gracefully.

### Scope correction

This is not a claim that every dense SHMR path can work without NumPy. Dense clustering/vector paths may keep NumPy as a runtime requirement. The bug is unconditional module import and unreachable non-dense/degraded behavior.

### Suggested solution

- Guard the NumPy import and keep annotations unevaluated.
- Implement scalar/list cosine support only where the non-dense path needs it, or fail with one explicit capability error when a dense-only API is invoked.
- Do not silently return fabricated vectors.

### Acceptance tests

- [ ] `import mnemosyne.core.shmr` succeeds without NumPy.
- [ ] Non-dense/degraded operation remains reachable.
- [ ] Dense-only operations fail with a clear capability error.
- [ ] Normal NumPy-backed behavior is unchanged.
- [ ] A fresh base-install subprocess reproduces the supported packaging path.

We are willing to submit this as one focused optional-dependency PR.

---

## B17 — Transaction ownership and nested mutation failures

**Status:** `EXISTING/FIXED` for the reported persona/task-progress path (#489 / merged PR #501); native Inhale/Dream designs guard the same failure class.

The failure occurs when one subsystem leaves or owns a transaction and another operation attempts `BEGIN IMMEDIATE` on the same connection. The result is deterministic `cannot start a transaction within a transaction`, or partial rollback if ownership is ambiguous.

The accepted upstream shape is the right solution: every mutating subsystem explicitly owns its transaction, commits/rolls back before returning, and leaves shared connections out of transaction. New native paths should reject caller-open/deferred contexts before mutation rather than rolling back transactions they do not own. This item requires no duplicate issue; it documents a cross-cutting invariant for future focused PRs.
