# Qwen Next: September 18 reference refresh

Continuation of PR #123 after remote checkpoint
`checkpoint/qwen-next-prefix-parity-20260918`. The installed version, runtime
defaults and weights are unchanged. New experiments remain opt-in; **overall
performance and quality parity is not established**.

## Identity and measurement contract

Reference: latest published [v26.9.4](https://github.com/ddalcu/mlx-serve/releases/tag/v26.9.4),
rechecked September 18, source `6991ba3ad3d356891d43211dec941cc6f9ff6dd6`.
Use the released ARM64 executable, not a modified source build. Its archive
matched the release asset digest before extraction. The moving 26.9.5-dev
branch is not substituted into this baseline.

| Component | SHA-256 or Git commit |
|---|---|
| Reference executable | `520b5f733850c219fda59c7a6876ccf78bacc769bd38c54d06f58acd70c88694` |
| Reference archive | `5a8b16317e7e4d5f87f528384a5d28b70bacbdcfce291be2aa6513a87e2a1925` |
| Unchanged AFM control executable | `21ee9bf82b49a361a240abd2fe09a3a416cf5cf9e794e95183a67805983387be` |
| Provider checkpoint | `3dec6c34d3f806bf4e52b882ad4c7d2f2b14e1d2` |
| Consumer checkpoint | `510072d24096649aaa81891093e466dc23a47ac6` |

M3 Ultra 512 GiB, exact unchanged checkpoint and mapped sidecar:

```text
/Volumes/edata2/models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit
```

Configuration/template/weight-index hashes are checked, not every weight shard.
Full commands, requests, raw responses, executable hashes and adaptive reference
calibration files are retained. GPU workloads run serially, requiring 160 GiB
available before load and a 100 GiB runtime floor. The original C15 harness
starts its periodic guard after readiness; the anchor A/B adds an outer guard
from process launch, covering model loading too. Named-process guards do not
detect every possible external GPU user.

Reference PLD/external drafting, decode-attention quantization, KV quantization,
disk prefix cache and tokenization cache are off; top-k is zero. Reference MTP
uses explicit maximum depth 3 and its adaptive planner, not necessarily three
drafts every round. AFM retains documented M25 C1, A11 AR-prefix and M27
MTP-prefix opt-ins, with no timing profiler. These are **not unset-environment
measurements**.

## First four Context points

Temperature 0.6, top-p 1, seed 42, thinking off, C1 and prefix cache off.
493/864/2112/4150 prompt tokens; 128 output tokens, one excluded warmup and
three measured trials per point. Each pair is median **prefill proxy / decode
tok/s**. Prefill is TTFT-derived, not isolated GPU prefill.

| Context | Reference AR | AFM AR | Reference MTP | AFM MTP | MTP decode difference |
|---|---:|---:|---:|---:|---:|
| 0.5K | 915.07 / 69.06 | 896.09 / 69.44 | 903.15 / 105.70 | 956.20 / 88.16 | -16.59% |
| 1K | 1074.67 / 68.50 | 1048.63 / 68.66 | 1066.70 / 91.52 | 1111.28 / 85.26 | -6.84% |
| 2K | 1226.25 / 62.73 | 1197.40 / 62.44 | 1224.13 / 81.29 | 1275.07 / 80.13 | -1.43% |
| 4K | 1317.00 / 62.79 | 1297.29 / 61.50 | 1290.56 / 77.85 | 1270.93 / 84.58 | +8.64% |

All 24 measured AFM responses match their prior same-mode checkpoint texts.
Reference MTP produces 1/2/3/3 distinct measured texts across these points,
despite the requested seed. Inputs match, not necessarily generated workloads.
The latest reference invalidates the older within-10%-everywhere statement:
**short-context MTP misses the gate**. Historical AFM depth-4 ~105.8 tok/s is
preserved but cannot be substituted into this depth-3 curve.

The refreshed performance ledger preserves 220 eligible run/context cells.
AR decode is within 0.9% of recorded peaks. Current MTP is 0.8–4.3% below the
best same-explicit-launch medians; no cause is established by historical timing
alone. Cross-configuration peaks and exclusions remain recorded separately.

## C15 full cache matrix

Fifteen fixed-answer task families: 15 greedy plus 30 sampled cases per mode.
Each 15-prompt window repeats immediately, then moves to the next window.
512-token cap, sampled temperature 0.6/top-p 1, frozen per-case seeds, thinking
off, no forced grammar. All **720 measured responses** completed and passed
artifact audit. That is runtime/audit success, not 720 correct model answers.
Fifteen overlapping HTTP requests do not prove one 15-row GPU batch.

Sampled aggregate output tok/s includes admission, prefill and decode:

| Prefix | MTP | Reference first / repeat | AFM first / repeat | Difference first / repeat | Reference correct first / repeat | AFM correct first / repeat |
|---|---|---:|---:|---:|---:|---:|
| off | off | 54.82 / 55.66 | 47.08 / 47.52 | -14.10% / -14.61% | 25/30 / 21/30 | 24/30 / 24/30 |
| off | on | 54.02 / 53.41 | 53.55 / 53.49 | -0.86% / +0.15% | 23/30 / 22/30 | 21/30 / 21/30 |
| on | off | 132.87 / 150.77 | 120.15 / 156.32 | -9.57% / +3.68% | 22/30 / 23/30 | 24/30 / 24/30 |
| on | on | 126.53 / 143.66 | 136.69 / 177.22 | +8.03% / +23.36% | 24/30 / 24/30 | 23/30 / 23/30 |

Correct means matching the fixed answer, identity, schema and evidence; not an
open-ended judge. AFM AR has one wrong-shape sampled response per phase. The
reference also has one wrong-shape AR cached-repeat response. Neither engine's
failure excuses the other, and attribution remains unresolved.

Greedy results are separate:

| Prefix | MTP | Reference first / repeat tok/s | AFM first / repeat tok/s | Reference correct | AFM correct |
|---|---|---:|---:|---:|---:|
| off | off | 58.43 / 58.52 | 53.71 / 54.16 | 13/15 / 13/15 | 12/15 / 12/15 |
| off | on | 58.24 / 58.54 | 57.04 / 56.89 | 13/15 / 12/15 | 13/15 / 13/15 |
| on | off | 49.33 / 156.23 | 69.05 / 160.78 | 12/15 / 12/15 | 12/15 / 12/15 |
| on | on | 53.26 / 153.49 | 70.92 / 156.94 | 12/15 / 12/15 | 13/15 / 13/15 |

AR first-greedy dropped from the prior 83.97 tok/s despite identical output
texts/tokens. Restored tokens dropped 9722→5834: four formerly cached requests
now miss. Admission/reuse history is a concrete difference, not proof of a
slower arithmetic kernel. Keep this loss visible.

## Quality isolation

The `review-1-07` failure returns the dependency order as an array instead of
`answer.order`, and AFM additionally misorders two tasks. A refined diagnostic
uses separate cold/donor and sampled/greedy server lifetimes. Donor
`review-0-12` demonstrably restores 969 tokens of the target's 998-token prompt.
Cold boundary-enabled sampling is correct; donated-prefix sampling is wrong;
greedy is correct in both. No simultaneous client batch is required.
Cache-disabled sampling also produces the wrong answer: restoration is not
necessary for failure. Numerical prefill geometry remains a hypothesis, not a
diagnosed corruption bug. An earlier donor probe missed and is not counted as
successful restoration coverage.

Three new 4-bit/BF16 resident-PLE tests compare identical-range live/restored
full logits and teacher-forced continuation, same-prompt N-31/N-1 endpoints,
and complete MTP target/head replay under strict/batched, greedy/sampled modes.
Integer hash parameters retain their dtype; fixture PLE width 128 supports
quantization group 32. Reliable-wrapper Release validation before the anchor
experiment: **88 passed,
one optional skipped, zero failed**. Tiny resident fixtures do not certify
full-checkpoint mapped-sidecar behavior or differently tiled forwards.

### Frozen 50-sample C1 refresh

The separate five-greedy/50-sampled agentic suite uses the original requests,
seeds and scorer, with prefix cache off. All 220 measured responses completed;
all 110 AFM texts match the previous same-mode AFM texts exactly. The following
strict score checks structure plus file/request identity, **not correctness of
the proposed code fix**; it is not interchangeable with the C15 semantic score.

| Engine | MTP | Greedy strict | Sampled strict | Sampled output tok/s including prefill | Strict tasks/s |
|---|---|---:|---:|---:|---:|
| Reference v26.9.4 | off | 5/5 | 41/50 | 27.24 | 0.1372 |
| AFM frozen control | off | 5/5 | 38/50 | 26.79 | 0.1223 |
| Reference v26.9.4 | on | 5/5 | 39/50 | 30.66 | 0.1483 |
| AFM frozen control | on | 5/5 | 37/50 | 29.87 | 0.1296 |

This refresh does not close the quality gap. The reference's old 40/50 AR and
41/50 MTP results remain historical, not replacement denominators or claims
that its newer version improves every quality metric. One fixed seed suite
does not establish statistical significance or identify a runtime bug.

## Third-window diagnostic and bounded anchor experiment

Adding a third identical window gives greedy repeat→repeat2 **161.38→183.58
tok/s**, with all 15 endpoints available and unchanged answers. It does not fix
the first-repeat penalty. Subsequent new sampled windows restore only 12645
tokens versus 29070 in the two-window run, slowing first-use to **69.61 tok/s**
from 136.69. Sampled repeat/repeat2 reach 157.46/165.91, with 22/30 and 23/30
correct; one second-repeat answer changes. All 135 responses completed/audited.

Source-backed cause: the broad 969-token donor has not been used by a different
request before its own endpoint promotion, so `sharedReuse` remains false and
promotion deletes it. This is not merely LRU eviction. A frequently reused
973-token donor covers only one request-ID family. CPU rendering identifies a
970-token common prefix; all 45 rendered lengths match API usage, but full
runtime token IDs were not logged. Peak inventory is 16 entries/2.17 GiB, making
entry count the binding limit in this run.

M28 is a default-off experiment: `AFM_QWEN_MTP_REPLAY_ANCHOR=1` added to M27.
Before promotion, select one proper earlier boundary covering the most distinct
other stored source prompts; tie-break by longer boundary. Preserve that key
during promotion, and under pressure prefer other earlier snapshots for
eviction before endpoints. No useful candidate preserves ordinary policy.
Existing entry/byte limits always win, including eviction of the protected
anchor if necessary. No additional snapshot or numerical split is created.
Ranking runs only on insertion, not token decoding or lookups.

This generic cache policy is wired only to Qwen's explicit experiment. One
anchor cannot serve every disjoint prompt family, and coverage count may
overvalue a shallow common prefix. Selecting another existing frontier can
change rounding/answers. Acceptance requires identical three-window A/B,
changed-answer audit, byte/ownership tests and a performance-tradeoff review.
Implementation alone is not evidence of improvement.

### Same-binary M28 A/B, then reversed order

Candidate binary SHA-256:
`b44fe62bce51ece0bea03353a13055fbb1c0af207f603b92171eee647c741976`.
Targeted Release XCTest suite (`ExactPromptReplayCacheTests` and
`QwenNextMTPPipelineTests`): **95 passed, one optional skipped, zero failed**
(96 total), including seven new retention/ownership/budget tests. Reliable-wrapper Release
build succeeded in 76.37 seconds. One binary serves every arm, changing only
`AFM_QWEN_MTP_REPLAY_ANCHOR=1`. Requests, seeds, weights, limits and scorer are
unchanged. Pair A runs off→on; B reverses on→off. Each arm has 135 measured
responses; all 540 pass runtime, structure and identity checks.

Aggregate output tok/s, including prefill and admission:

| Workload | A off → on | B off → on |
|---|---:|---:|
| Greedy first | 62.14 → 85.35 | 74.98 → 84.07 |
| Greedy repeat | 162.21 → 168.99 | 153.39 → 165.65 |
| Greedy repeat2 | 162.50 → 185.10 | 160.15 → 186.90 |
| Sampled first | 126.11 → 136.28 | 139.22 → 142.70 |
| Sampled repeat | 184.92 → 180.10 | 179.44 → 183.83 |
| Sampled repeat2 | 180.33 → 176.85 | 179.67 → 181.33 |

Greedy repeat2 restores all 15 full endpoints with M28 versus 8/7 without it,
improving throughput 13.9%/16.7%. Pair A's greedy responses are identical;
pair B changes one greedy case across all three phases without changing its semantic score.
First-greedy gains also include different admission/cache-hit histories, so
the whole timing difference is not attributable to the retention policy.

Every sampled cell remains **23/30** and every greedy cell **13/15**. This is
not no-regression evidence: pair B improves `review-2-07` but regresses
`review-1-07` in every phase. Pair A has no semantic flips and 134/135 identical
texts; its sole changed answer is wrong in both arms. Pair B has 118/135
identical texts. Sampled repeat timing differences change sign on reversal;
there is no established sampled-repeat gain or penalty. Complete endpoint hits
are 30/30 in both arms, so these differences cannot simply be attributed to
cache-ranking CPU overhead.

**Decision: retain only as an opt-in experiment, not a recommended default.**
The greedy repeat result is worth preserving; per-case quality differences and
admission variation remain acceptance gates. This does not close the C1
short-context MTP gap or uncached AR gap. Independent paired audits verify
both load/runtime guards, exact one-flag argv difference, executable/checkpoint
metadata, payloads, scores, wall-time denominators and changed-response lists.
The first driver has an exceptional runner-timeout cleanup weakness; it did
not trigger. Reversed runs use the reviewed hardened guard with immediate
owned-server signalling and nested final cleanup/reporting.

M28 retained lifecycle qualification passed **565/565 assertions across 79
requests**, including one warmup: mixed sampled/greedy MTP and AR fallback,
logprobs/penalties/stops, >4096-token replay, early disconnect, recovery, and
C15 per-request identity isolation. The log records 182 shared verification
batches and a clean shutdown. This does not test actual model A→B→A switching,
an extended soak, or open-ended answer quality.

A separate 45-prompt working-set screen exceeds the 16-entry cache cap before
repeating the whole set (135 measured requests). All runtime/structure/identity
checks pass and shutdown is clean. Repeats restore the 969-token common donor,
not all full endpoints: sampled first/repeat/repeat2 are 138.29/140.92/139.30
aggregate tok/s, with 23/24/23 correct of 30. Greedy scores change 13→12→12 of
15. Peak RSS is 69.10 GiB (not total Metal allocation accounting). This shows
bounded operational reuse, not quality equivalence, unlimited replay capacity,
or an off/on performance comparison. It is not an extended soak.

## Evidence and remaining gates

Append-only external root:

```text
/Volumes/edata2/afm-benchmarks/qwen-next-mtp-parity-20260909/reference-refresh-20260918
```

See reference identity, Context folders, C15 folders/audits, prefix diagnostics,
`quantized-replay-b.log`, `afm-three-window-a`, `performance-ledger`,
`anchor-tests-a.log`, `anchor-{off,on}-{a,b}`, paired audits and
`anchor-lifecycle-a`, and `anchor-eviction-a`. Completed runner copies and raw failed attempts are
preserved as well as the successful measurements.
The sealed `prefix-boundary-20260917` evidence remains unchanged.
The separate 50-sampled/five-greedy C1 refresh is recorded in the
`reference-agentic-a` and `afm-agentic-a` folders and paired audit files.
Independent review cleared the code and evidence for an experimental
checkpoint, not a default promotion or parity claim. Operational gates still
include real model switching and an extended soak; the per-case quality and
short-context/uncached performance gaps remain open before a merge recommendation.
Do not accept a default or quality/performance trade-off without discussion.

The external evidence manifest verifies **2,753 files**, excluding bytecode.
`SHA256SUMS.txt` SHA-256:
`eaa77d4d0e5dc602b73ef13de7b14d055cafcd7a4a831caca59be0c2390fde84`.
Captured source/report copies are the point-in-time versions before this seal
line was appended; runtime source is unchanged after validation.
