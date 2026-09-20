# Qwen Next: speculative cost and uncached scheduling isolation

September 19 continuation of [PR #123](https://github.com/scouzi1966/AFMKit/pull/123).
This follows the [September 18 baseline refresh](QWEN_NEXT_REFERENCE_REFRESH.md).
No cache policy, sampler, installed binary, or runtime default is changed.
**Overall performance and quality parity remains open.**

## Frozen identities

Same M3 Ultra 512 GiB and unchanged checkpoint throughout:

```text
/Volumes/edata2/models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit
```

| Component | Identity |
|---|---|
| AFMKit starting commit | `7b27b56bd607206db30007ce3afe4ca246865b24` |
| Consumer commit | `510072d24096649aaa81891093e466dc23a47ac6` |
| Frozen AFM executable SHA-256 | `b44fe62bce51ece0bea03353a13055fbb1c0af207f603b92171eee647c741976` |
| Diagnostic AFM executable SHA-256 | `e98c228b66bf7a4a8d77ce0e8f64a7bf3c620381f9262a70b0e59a4090caaa68` |
| Reference | Released v26.9.4, source `6991ba3ad3d356891d43211dec941cc6f9ff6dd6` |
| Reference executable SHA-256 | `520b5f733850c219fda59c7a6876ccf78bacc769bd38c54d06f58acd70c88694` |

The 94 adjacent resources of the diagnostic bundle match the frozen bundle.
Checkpoint configuration, template and weight-index hashes are checked; this
does not mean every weight shard was rehashed. Every inference arm has its own
server lifetime, command, raw responses, exit status and outer resource guard.
GPU work is serial, with 160 GiB available before load and a 100 GiB runtime
floor. Named-process guards do not detect every possible external GPU user.
Existing experimental launch settings are explicit: these are not claims about
unset-environment release defaults.

## What changed in code

`BatchExecutionProfile` adds bounded, per-active-row host accounting to the
existing independent-cache scheduler, enabled only by existing `AFM_PERF=1`.
No new tuning variable, GPU evaluation, synchronization, arithmetic, or
scheduling decision is introduced. When disabled, no clocks or sample storage
are used; optional checks remain.

| Label | Meaning |
|---|---|
| `ar-prefill-service` | Only `prefillOne`, excluding nested independent decode ticks; not speculative or dense-batch prefill |
| `independent-prepare-submit` | Group preparation, model forwards, sampling and submission; can include existing GPU waits, especially MTP verification/repair |
| `independent-readout` | Token materialization and dispatch; may wait on previously submitted work |
| `independent-retire` | Completed-slot retirement |
| `independent-maintenance` | Existing cache clearing/evaluation maintenance |
| `independent-total-inclusive` | Whole tick including cancellation cleanup; do not add this to its component spans |

Counters are emitted once at scheduler shutdown. They include warmups.
`serviced_rows=1` describes prefilling one request. Whole-tick `active_rows`
counts requests before cancellation cleanup; component rows count surviving
requests afterward. Neither is emitted tokens or proof of a GPU batch, and
their per-row buckets need not align when cancellation removes requests.
The independent total, not the sum of its components, is subtracted from a
nested prefill span. All accounting is actor-owned.

Independent review found no remaining blocker after clarifying the labels and
fixing inclusive/nested accounting. Profiler-only reliable-wrapper Release
validation, before adding the two head-projection tests:
**116 tests passed, one optional test skipped, zero failed** (117 executed),
including three new profiling tests. The consumer Release build passed.

## MTP: proposal yield is distinct from round execution cost

C1, prefix off, thinking off, target temperature 0.6/top-p 1/seed 42,
493/864/2112/4150 prompt tokens and 128 output tokens. Each diagnostic arm has
one warmup plus three measured trials at each point. All arms below hold
**three draft positions per round**, unlike the ordinary reference's adaptive
planner. Profiled values are not entered into the clean performance ledger.
The AFM MTP arm uses frozen binary `b44fe62…`; C15 profiled and rebuilt-clean
arms below use diagnostic binary `e98c228…`.

Each cell is median **decode tok/s / reported verification rounds**:

| Context | AFM full-head greedy proposals | Reference sampled proposals | Reference greedy proposals | Reference greedy, predraft off | Reference greedy, coarse head off |
|---|---:|---:|---:|---:|---:|
| 0.5K | 89.25 / 48 | 106.21 / 41 | 101.61 / 43 | 100.38 / 43 | 98.98 / 43 |
| 1K | 84.04 / 49 | 92.56 / 46 | 90.75 / 47 | 89.77 / 47 | 88.87 / 47 |
| 2K | 82.89 / 46 | 82.52 / 47 | 83.93 / 47 | 85.64 / 47 | 83.61 / 47 |
| 4K | 81.52 / 44 | 89.27 / 44 | 85.84 / 46 | 86.84 / 46 | 88.94 / 44 |

“Greedy proposals” does **not** mean greedy target generation. The target
remains sampled at temperature 0.6. At 0.5K, reported draft acceptance is
55.6% for AFM, 70.7% for reference sampled proposals and 65.9% for reference
greedy proposals. Counter totals may include accepted speculative work at the
output cutoff; they are not necessarily counts of emitted draft tokens.

The reference [request-specific proposal policy](https://github.com/ddalcu/mlx-serve/blob/6991ba3ad3d356891d43211dec941cc6f9ff6dd6/src/generate.zig#L7891)
selects sampled proposals at this temperature. It retains the actual proposal
distribution for probability-ratio acceptance and residual rejection sampling.
AFM proposes argmax tokens and accepts when the target sample matches. Both
approaches can preserve the intended target distribution when correctly
implemented; neither establishes numerical or task-quality equivalence.

The reference also has a [coarse-head shortlist path](https://github.com/ddalcu/mlx-serve/blob/6991ba3ad3d356891d43211dec941cc6f9ff6dd6/src/generate.zig#L5586).
Forcing greedy proposals does not disable its 3-bit vocabulary head and exact
top-32 rescoring. Hence the separate full-head control. The shortlist can
change which token is proposed, not merely its execution time.

Reference-only diagnostic overrides, each in a fresh process:

| Arm | Overrides on the frozen reference launch |
|---|---|
| Fixed sampled | `MLX_SERVE_MTP_FORCE_DEPTH=3`, `MLX_SERVE_MTP_TRACE=1` |
| Fixed greedy | Above plus `MLX_SERVE_MTP_DRAFT_GREEDY=1` |
| No predraft | Fixed greedy plus `MLX_SERVE_MTP_PREDRAFT=0` |
| Full draft head | Fixed greedy plus `MLX_SERVE_MTP_DRAFT_RERANK=0` |

Predraft-off reproduces all 12 measured greedy-control texts. Its timing change
is small and mixed (-1.2%, -1.1%, +2.0%, +1.2% by context), not a compelling
reason to add cancellation/EOS/state-ownership complexity. AFM already
asynchronously submits completed draft chains; it is not wholly synchronous.

Disabling the coarse head preserves only 3/12 measured texts; disabling sampled
proposals preserves 0/12. Cross-engine AFM texts also differ. These controls
therefore narrow hypotheses but do **not** isolate identical-trace kernel
performance. Disabling reference optimizations is not achieving parity with
the reference's ordinary best configuration.

The previous merged history/first-draft and generic coarse-head experiments
remain rejected as recorded in [the earlier investigation](QWEN_NEXT_MTP_PARITY_PROGRESS.md).
Do not repeat those implementations unchanged on the strength of a source-code
similarity. Host phase labels also differ by runtime: AFM's verify-build can
include waits that the reference reports under evaluation. They cannot be
compared as pure CPU or isolated Metal kernel timings.

## C15 uncached AR: where the time goes

The unchanged full fixed-answer suite uses 15 greedy plus 30 sampled agentic
cases, then exact repeats: 90 measured requests per arm, rolling client
concurrency 15, prefix off, MTP off and a 512-token cap. It is not the separate
15-prompt replay-window cache experiment. Requests, seeds and scorer remain
frozen; generated lengths can differ across engines.

The instrumented AFM run records 92 prefill services (90 measured requests plus
two warmups), and 536 independent decode ticks:

| Host span | Seconds |
|---|---:|
| Exclusive AR prefill service | 83.496 |
| Whole independent decode ticks | 36.757 |
| — preparation/submission component | 34.425 |
| — readout component | 1.912 |
| — retirement component | 0.285 |
| — maintenance component | 0.129 |

Prefill is 69.4% of the two non-overlapping measured service spans. This is
not 69.4% of all process lifetime or pure GPU time. Preparation/submission is
93.7% of the decode-tick span; output dispatch/cleanup is not the dominant cost.
The logs independently confirm actual mixed-position decode calls up to 15
rows. There are 218 ticks with 15 active requests; smaller cohorts also occur
during admission and retirement. These totals alone do not establish queue
fairness or per-request pause durations.

Both runtimes prefill individual requests in this path. The reference also
[ticks decode between admitted prefills](https://github.com/ddalcu/mlx-serve/blob/6991ba3ad3d356891d43211dec941cc6f9ff6dd6/src/scheduler.zig#L4484).
AFM uses budgeted admission and its measured preset disables chunk interleave.
It is wrong to describe the gap simply as “reference has batched prefill, AFM
does not.” Interleaving can trade aggregate throughput against stream latency
and numerical prefill geometry; it is not automatically a win.

### Clean throughput and independent quality checks

No profiling variables in these clean runs. Rates include admission, prefill
and decode, not just a post-admission decode window:

| Workload | Previous AFM binary | Rebuilt AFM, counters off | Reference v26.9.4 | AFM correct | Reference correct |
|---|---:|---:|---:|---:|---:|
| Greedy first | 52.24 | 54.05 | 58.79 | 12/15 | 13/15 |
| Greedy repeat | 54.28 | 54.31 | 58.78 | 12/15 | 13/15 |
| Sampled first | 47.60 | 47.32 | 55.35 | 24/30 | 25/30 |
| Sampled repeat | 47.64 | 47.58 | 55.04 | 24/30 | 23/30 |

The previous, rebuilt and profiled AFM arms reproduce all 90 texts exactly,
also matching the refreshed AFM baseline. Each has 90/90 runtime successes,
72/90 fixed-answer semantic passes, 5,872 output tokens, zero cache hits and no
length finishes. Rebuilt-clean total wall time is 118.054 s versus previous
118.473 s (-0.35%). This single ordered comparison supports no observable
material counters-off regression, not a speedup claim or statistical bound.
Profiled wall time is 120.136 s (+1.76% versus rebuilt clean), largely in the
first greedy window; do not treat that as a stable profiling-overhead bound.

The independent audit checks 450 responses including the previously refreshed
AFM arm, reconstructing streams, rescoring, checking payload/cap/identity and
guard contracts. Reference totals are 74/90 semantic passes, 6,164 tokens and
109.275 s; its 56.41 aggregate tok/s versus AFM's 49.74 is partly affected by
different answer lengths. Two reference sampled answers change from correct
to incorrect on repeat despite identical payloads/seeds and unchanged lengths:
`review-1-07` and `review-1-11`. Across engines, 60/90 texts match, while 26/90
token counts differ. These are not identical generated workloads.

Model-quality failures remain real and separate from transport success. This
small suite does not establish broad quality parity or determine whether each
wrong answer is stochastic, numerical or an implementation defect.

### Rejected one-token diagnostic comparison

An admission-only `max_tokens=1` screen is retained but is **not** a semantic
quality test or a valid throughput baseline. AFM honors the cap in all 90
requests; the reference returns two tokens in 44/90. The arm therefore cannot
serve as an equal-output prefill subtraction. Immediate slot retirement also
changes scheduling even if both caps were honored. No model regression is
inferred from this diagnostic's expected semantic failures.

## Final-prefill vocabulary projection: head-only experiment

Ordinary AR's final remaining prompt chunk calls the normal model forward,
which projects every hidden row to the 248,320-token vocabulary; the scheduler
then consumes only its last row. Laziness can skip entire unused projections
of earlier chunks, but does not push this final slice through quantized matmul.
At 1,000 BF16 rows the full output is about 474 MiB versus about 485 KiB for one
row, before allocator effects.

A test-only Release microbenchmark now compares projecting-then-slicing against
slicing-then-projecting using the same materialized synthetic BF16 hidden inputs
and original checkpoint head weights. Two fresh processes each ran two warmup
pairs and six measured, alternating-order pairs at every width. Timings include
fresh graph construction, evaluation and stream completion; preparation and CPU
logit comparisons are outside the timer. Both runs passed **5/5 tests** (three
profiler tests, a tiny CPU projection oracle and the opt-in checkpoint test).
Their test-executable SHA-256 matches:
`dfb9b4125ebb8011714ac15a14c8766d7132db4fa6d83497e816f23206c1c31b`.

Median component latency in milliseconds, first run / fresh-process repeat:

| Prompt rows | Project all then slice | Slice then project |
|---|---:|---:|
| 493 | 28.408 / 28.497 | 1.284 / 1.258 |
| 864 | 47.453 / 47.514 | 1.296 / 1.281 |
| 998 | 55.948 / 56.064 | 1.253 / 1.300 |
| 2,112 | 114.733 / 115.189 | 1.333 / 1.335 |
| 4,150 | 225.435 / 226.003 | 1.319 / 1.293 |

At 4,150 rows, first-run maximum MLX active allocation is 2,757,754,880 bytes
for project-all versus 697,204,736 bytes for last-row-only. These are
process-local allocator peaks including the resident head and hidden input,
not full-model RAM, RSS or mapped sidecar pages. Allocator cache is retained
within each width and cleared before that width's warmups.
At 4,150 rows both measured arms retain the same active-plus-cache allocation,
2,801,772,544 bytes. The lower active peak is not a measured total-residency
reduction in this cache-retaining experiment.

All 80 warmup/measured pairs across both runs produce finite logits and agree
on argmax for the synthetic final row. They are **not numerically identical**:
maximum absolute logit difference is 0.015625, with mean absolute differences
below 0.001. This can affect sampling or a close greedy decision in real
generation; synthetic argmax agreement is not model-quality qualification.

The removed work is about 27–225 ms per final remainder in this isolated test,
not a comparable multiplication of whole-model prefill or decode throughput.
For example, 90 requests each with a 998-row remainder would save roughly
4.9 seconds if the component saving translated unchanged. That illustration
is **not** a measured end-to-end result or a promise to close the C15 gap.

Scope matters: `MLXReplayPrefill.prepareWithSnapshot` already leaves a singleton
final prompt token, so recurrent prefix-replay and chunk-interleave routes
generally do not have this redundant full-remainder head. The candidate targets
ordinary `BatchScheduler.prefillOne`'s `prepare(.tokens)` remainder path, not a
new prefix-cache policy. No production routing, sampling or default was changed.
The reference's multirow-tail forward also projects before slicing, but that
alone misrepresents its measured uncached route. With prefix cache disabled,
reference revision `6991ba3` sets checkpoint stride/backoff to zero and leaves
only the final token for its logits forward (`src/scheduler.zig:4081,6122`;
`src/generate.zig:1055,2469–2485`). AFM's ordinary remaining-token route can
instead forward a large final remainder. This is a source-confirmed execution
difference; its whole-model timing and quality contribution remains unmeasured.
Preserving AFM's existing trunk geometry while narrowing only the head avoids
conflating this experiment with a new singleton trunk/chunk policy.

The original checkpoint's vocabulary head is **8-bit affine, group size 64**,
despite the global model's 4-bit label. Its packed weight shape is
`[248320, 640]` (uint32), with `[248320, 40]` BF16 scales/biases and input
width 2560. The experiment preserves that head; it must not accidentally
requantize it to four bits or benchmark a different checkpoint.

The next controlled implementation should preserve the ordinary trunk/chunk
geometry and use the existing `LanguageModel.prepare` / `PrepareResult.logits`
contract to return only the necessary final logits. Do not replace ordinary
forward with a differently fused stream-state trunk merely to reuse an API.
Qualification must compare full responses, first-token/logit differences, cache
state, prefix replay, cancellation, MTP exclusion and clean end-to-end performance
before proposing a production default. Keep any experimental routing opt-in.

## Evidence

All new raw evidence is external and untracked:

```text
/Volumes/edata2/afm-benchmarks/qwen-next-mtp-parity-20260909/execution-costs-20260919
```

Key folders: `mtp-profile-a`, `reference-profile-a`,
`reference-fixed-sampled-a`, `reference-fixed-greedy-a`,
`reference-no-predraft-a`, `reference-full-head-a`, `ar-full-profile-b`,
`ar-clean-new-a`, `ar-clean-old-b`, `reference-full-clean-a`,
`head-projection-poc-b`, `head-projection-poc-c`.
`mtp-costs-analysis.json` retains cycle/yield and text-identity comparisons.
The independently audited response data is separate from timing interpretation.
Final audit: `independent-audit-execution-20260919-v2-reference-complete.json`,
SHA-256 `857bd8f99df2cbd9cb78a5efe0b26ca97b11a33b6c3bc96ff5f3fc30b6de5203`.

`ar-full-profile-a` is a retained **prelaunch harness failure**, not an engine
failure: the ordinary timing validator rejected the wrapper's profiler flag on
its second validation call. `run-uncached-costs-v2.py` makes diagnostic injection
idempotent and labels diagnostic artifacts; normal baseline validation remains
unchanged. No failed or intermediate result was silently overwritten.

`head-projection-poc-a` retained a compile failure in the test's ambiguous
`Stream` annotation (Foundation versus MLX). Qualifying it as `MLX.Stream`
fixed compilation. Attempt B compiled in 91.09 seconds and tested in 4.465;
the fresh-process repeat C compiled incrementally in 6.15 seconds and tested
in 4.467. Each attempt retains its own source delta, command, exit and log.
