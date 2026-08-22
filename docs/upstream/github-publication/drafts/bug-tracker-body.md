# Deployment-informed correctness and reliability findings

## Context and evidence standard

These findings came from a Mnemosyne setup that we personally operate and have extended for a real Hermes integration. Running the system across real write, recall, consolidation, recovery, scope, provider, and lifecycle paths exposed the items below.

Private deployment observations are discovery signals, not upstream proof. New bug claims are independently reproduced against current upstream code with sanitized fixtures. Existing findings link to their official issue/PR instead of duplicating them. No private memory content, credentials, hostnames, IPs, absolute private paths, or live identifiers are included.

This is a tracking/index issue, not a request for one large patch. We are willing to implement accepted items as small focused PRs. A PR should use `Addresses Bxx in this tracker`; it must not close the tracker.

## Status vocabulary

- `VERIFIED_NEW` — reproduced on current upstream and no semantic duplicate found.
- `EXISTING` — already tracked upstream; this tracker adds deployment-informed evidence/context only.
- `FIXED/MERGED` — accepted upstream.
- `OPEN_PR` — a focused fix already exists.
- `NEEDS_REPRODUCTION` — discovery signal only, not a confirmed upstream bug.

## Findings index

| ID | Finding | Status | Existing issue / PR | Detail |
|---|---|---|---|---|
| B01 | Concurrent fresh-DB initialization races check-before-ALTER schema guards | `VERIFIED_NEW` | No semantic duplicate found | [evidence](https://github.com/mnemosyne-oss/mnemosyne/issues/827#issuecomment-5378533682) |
| B02 | CLI operational failures leaked backend details or wrong success/error contracts | `FIXED/MERGED` | PR #814 | [details](https://github.com/mnemosyne-oss/mnemosyne/issues/827#issuecomment-5378534117) |
| B03 | Backup/restore and JSON import can leave unsafe or partial recovery state | `EXISTING` / `OPEN_PR` | #640, #727, PR #815 | [details](https://github.com/mnemosyne-oss/mnemosyne/issues/827#issuecomment-5378534117) |
| B04 | Hygiene candidate failures need per-item transaction isolation | `OPEN_PR` | PR #816 | [details](https://github.com/mnemosyne-oss/mnemosyne/issues/827#issuecomment-5378534117) |
| B05 | Beam health error predicate miscounts successful summaries | `EXISTING` / `OPEN_PR` | #717 / PR #719 | [details](https://github.com/mnemosyne-oss/mnemosyne/issues/827#issuecomment-5378534117) |
| B06 | Single-item embedding failures can silently skip vector storage | `EXISTING` / `OPEN_PR` | #718, #735 / PRs #720, #797 | [details](https://github.com/mnemosyne-oss/mnemosyne/issues/827#issuecomment-5378534117) |
| B07 | Query-side vector dimension mismatch crashes recall | `EXISTING` / `OPEN_PR` | #753 / PR #754 | [details](https://github.com/mnemosyne-oss/mnemosyne/issues/827#issuecomment-5378534117) |
| B08 | SHMR local-LLM dispatch used an incompatible call contract | `EXISTING` / `OPEN_PR` | #716 / PR #721 | [details](https://github.com/mnemosyne-oss/mnemosyne/issues/827#issuecomment-5378534117) |
| B09 | Polyphonic recall omits signals required by automatic prefetch | `EXISTING` / `OPEN_PR` | #700 / PR #701 | [details](https://github.com/mnemosyne-oss/mnemosyne/issues/827#issuecomment-5378534117) |
| B10 | Remote LLM configuration can still trigger local model loading/download | `EXISTING` | #688 | [details](https://github.com/mnemosyne-oss/mnemosyne/issues/827#issuecomment-5378534117) |
| B11 | Automatic consolidation can cross session boundaries or race active access | `EXISTING` / partly fixed | #342, #498, #687; PRs #349, #520, #772 | [details](https://github.com/mnemosyne-oss/mnemosyne/issues/827#issuecomment-5378534117) |
| B12 | Derived summaries can present write time as content age | `EXISTING` | #564 | [details](https://github.com/mnemosyne-oss/mnemosyne/issues/827#issuecomment-5378534117) |
| B13 | Identity/scope mapping can make valid memories invisible to the intended reader | `EXISTING` / partly fixed | #327, #601, #653; PR #604 | [details](https://github.com/mnemosyne-oss/mnemosyne/issues/827#issuecomment-5378534117) |
| B14 | CJK hygiene handling missed secret labels and misclassified content | `EXISTING` / partly fixed | #806 / PR #810 | [details](https://github.com/mnemosyne-oss/mnemosyne/issues/827#issuecomment-5378534117) |
| B15 | Batch mutation failures leak raw exception text and traceback details | `VERIFIED_NEW` | No semantic duplicate found | [evidence](https://github.com/mnemosyne-oss/mnemosyne/issues/827#issuecomment-5378533682) |
| B16 | SHMR import violates the supported no-NumPy base-install contract | `VERIFIED_NEW` | PR #31 fixed core imports, not SHMR | [evidence](https://github.com/mnemosyne-oss/mnemosyne/issues/827#issuecomment-5378533682) |
| B17 | Shared/caller-open transactions can break nested memory mutations | `EXISTING/FIXED` | #489 / merged PR #501; native Inhale/Dream guard the same class | [details](https://github.com/mnemosyne-oss/mnemosyne/issues/827#issuecomment-5378534117) |

## Implementation policy

- One root cause per PR.
- Rebase on current `main`; run a focused regression plus adjacent sibling-boundary tests.
- Preserve local-first behavior and content-free diagnostics.
- Add evidence to an existing issue when one exists; do not create duplicate child issues.
- Keep this tracker open until every item is merged, rejected, or transferred to its official issue.

## Willing to contribute

- [x] We are willing to submit focused PRs.
- [x] We will follow maintainer ordering and scope guidance.
- [x] We will keep production evidence sanitized and independently reproducible.

Related implementation history: #719, #720, #721, #774, #814, #815, #816, #817.
