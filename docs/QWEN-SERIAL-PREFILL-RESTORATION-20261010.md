# Restore Qwen serial prefill boundaries — October 10, 2026

The tail-grouping experiment is removed, rather than retained as a serving
option. The runtime files `MLXReplayPrefill.swift` and `MLXModelService.swift`
are restored exactly to AFMKit commit
`136d8e5586a472bcfbdf406a33d942014da36706`, used by the preserved best Orbital
binary. Existing prefix caching, reasoning preservation, tool parsing, and
other model paths are unchanged. No environment override is required.

## Evidence that triggered the rollback

A fresh three-project qualification of the default-on candidate failed to
reproduce the best Orbital and dashboard outcomes. AFM Orbital reached 60,000
output tokens without a final reply, passed 4/11 checks, and failed startup
because its generated HTML omitted `id="planet-count"` while its JavaScript
accessed that ID. Python workflow passed 19/19; failure was not universal.

The controlled follow-up replayed the **same 18 captured requests** against
the preserved binary and the default-on candidate, sequentially, with the same
checkpoint and startup parameters. Earlier requests were capped at one output
token to reconstruct prompt-cache boundaries; request 18 retained a 16,384
token limit. All subsequent inputs remained the original captured histories.
Returned tool commands were never executed or used to rewrite later requests.
This is a diagnostic, not a full coding-task qualification.

| Request 18 result | Preserved best | Tail-grouping candidate |
|---|---:|---:|
| Input tokens | 22,620 | 22,620 |
| Reused prompt tokens | 22,327 | 22,327 |
| Output tokens | 3,979 | 3,564 |
| Response seconds | 65.732 | 59.037 |
| Generated HTML includes required ID | yes | no |

The grouped candidate reproduced the failing live response's 3,564-token
output and missing ID. The preserved path emitted
`<span id="planet-count" data-testid="planet-count">` instead. The source
delta under investigation is tail grouping; this result demonstrates its
effect on this decision boundary, **not** the cause of every historical
whole-task difference. A shorter response is not a win if it breaks the task.

Binary identities:

- Preserved best: `0e8edd40a46341bab51da5ad7725a45e43ba622a7c4ef2eac1d6951646441156`.
- Default-on candidate: `81b1afb62c6e98e09a15e4ade84cea6768ca70efddad0801a412e4421e09c38e`.
- Consumer source: `6839b0566dbd166f11ad4203db9568165ac8ff49`.
- Candidate provider source: `c5809286372512e783171019b3d9a1a87e33c877`.

Checkpoint for both arms:
`/Volumes/edata2/models/qualification/Qwen3.8-Flash-Next-ddalcu-vision-overlay-20261001`.
Config SHA256: `fe0b5952857299b31d75bedf1d8897faea14b4116a04c73a73152361e7788f59`.
MTP off; prefix cache and thinking on; temperature 0, top-p 1, top-k 0, seed 123;
prefill step 8192. No tuning environment overrides.

## Regression guard and remaining gates

The small CPU/FP32 test verifies the original `[17, 30, 1]` forward geometry
for a 48-token fixture, the real near-end checkpoint, and continuation-state
agreement with an independent same-geometry control. It does not assert
production BF16 coding quality from a toy model.

Restoring source and this HTML decision boundary does **not** establish that
the historical 511.390-second / 11-of-11 live Orbital outcome is recovered.
Fresh live coding and independent acceptance still need to reproduce the
required quality and task-time envelope before promotion. Keep the preserved
best binary immutable, qualify one change at a time, and retain failures.

Local evidence root:
`/Volumes/edata/dev/CODEX/codex-local-coding-eval-20261007`.
See `RestoreHTMLBest20261010`, `RestoreHTMLDefault20261010`, and
`threeProjectDefaultTail20261010`; their manifests include binary identities,
commands, requests, raw responses and usage.
