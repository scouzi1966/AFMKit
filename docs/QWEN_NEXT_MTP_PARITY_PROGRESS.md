# Qwen Next MTP parity: controlled implementation work

This records intermediate work for AFMKit PR #123, following research PR #122.
**The MTP performance goal is not achieved.** These are targeted experiments,
not release qualification, and default MTP is not yet consistently faster than AR.

## Measurement contract

### Follow-up: non-greedy Qwen Next MTP

Qwen Next's text-generation binding now accepts positive temperature with
top-p filtering and a request-local seed. This also covers text-only requests
in a Qwen Next vision-capable container. The ordinary user default remains
temperature 0.6 / top-p 1.0; `--mtp` no longer forces those eligible requests
to the AR scheduler merely because their temperature is positive. No tuning
environment variable is required to enable sampled MTP. The experimental
batched verifier and dispatch optimizations retain their existing opt-in
settings; this change does not promote them or change greedy defaults.

Draft tokens remain deterministic. The verifier samples all its positions
on the GPU with the **user's** target sampler. Accept matching proposals until
the first mismatch, emit that sampled correction, and discard the remaining
speculative rows. For a deterministic proposal `d`, the accepted mass is
`p[d]` and the correction mass for every `t != d` is `p[t]`. This is exact
one-hot-proposal speculative sampling with respect to the chosen verifier's
probabilities, not a promise of identical seeded text to AR or equivalence
between different floating-point verification policies. David Dalcu's
MIT-licensed mlx-serve implements the same one-hot proposal option; the
acceptance principle comes from Leviathan et al., arXiv:2211.17192.

Only packed target/draft token IDs cross the host boundary. No vocabulary-size
probability arrays are copied to the CPU, no draft probability tensors are
needed for one-hot proposals, and cache commit/rollback still follows the
accepted prefix. The sampler and RNG are created per request, not stored on
the shared generator. Greedy requests retain their existing fused argmax.

The common top-p sampler now gathers within each vocabulary row rather than
cross-producting batch dimensions with `take`. This enables independent
multi-position target sampling. Tests compare the previously valid 1D and
singleton-2D paths against a legacy seeded oracle, and check multi-row and
3D layouts, disjoint row supports, temperature scaling, and observed filtered
probabilities against an independent CPU calculation.

Validation at this checkpoint:

- Release consumer build: passed, 127.12 s. Inference binary SHA-256:
  `096f4c036b15782e094f724c79765628416e3679ef38b534b7eebdbab6c32acb`.
- 46 focused Release tests passed, including existing greedy rollback/dispatch
  tests, exact one-hot acceptance/correction mass, sampled seed repeatability,
  cancellation, EOS, separate requests, and unsupported-contract admission.
- 13 live API requests passed under experimental depth-4 MTP with prefix
  caching and concurrent admission enabled. Tests covered omitted versus
  explicit 0.6/1.0 defaults, temperature 0.6/top-p 0.95, temperature 1/top-p 0.8,
  greedy control, seeded streaming/non-streaming equality, concurrent seed
  isolation, and ordinary fallback for stops/logprobs. Eleven requested MTP
  executions plus one startup warmup are present in the debug trace.
- Uninstrumented sampled context comparisons are recorded separately from the
  frozen greedy baseline. Greedy parity is **not** sampled-MTP parity.

Scope limits: tools, schemas, logprobs, stops, media inputs, top-k, min-p, and
repetition/presence penalties retain ordinary generation when not supported
by this binding. Qwen MTP still owns request-local cold caches in a serial
execution lane; passing concurrent/prefix-cache coexistence tests does not
mean speculative continuous batching or prefix-reuse acceleration. Other
model generators remain greedy-only here. Follow-up issue #124 tracks sampled
MTP for the other families with separate quality and performance qualification.

#### Sampled controls at code checkpoint `aae2c2d4`

All seven uninstrumented arms completed, sequentially on the same M3 Ultra,
using the exact `/Volumes/edata2/models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit`
checkpoint. Temperature is 0.6, seed 42, thinking off and prefix reuse off.
Each context has one excluded warmup and three measured 128-token responses;
actual prompt lengths are 493 / 864 / 2,112 / 4,150 tokens. The reference is
the frozen v26.9.2 binary, not an assertion about subsequent reference releases.
Sampled outputs can differ between engines and policies, so these are
same-input workload comparisons, not identical generated-token workloads.

Median client decode throughput, tok/s, with **top-p 1.0**:

| Context | AFM AR | AFM default MTP | AFM experimental MTP | Reference MTP | Experimental vs reference |
|---|---:|---:|---:|---:|---:|
| 0.5K | 66.68 | 74.25 | 89.01 | 87.28 | +2.0% |
| 1K | 67.15 | 72.06 | 85.01 | 88.51 | -4.0% |
| 2K | 60.56 | 59.50 | 87.54 | 88.83 | -1.5% |
| 4K | 59.57 | 59.32 | 82.98 | 80.37 | +3.2% |

Median client decode throughput, tok/s, with **top-p 0.95**:

| Context | AFM AR | AFM experimental MTP | Reference MTP | Experimental vs reference |
|---|---:|---:|---:|---:|
| 0.5K | 65.37 | 80.60 | 90.08 | -10.5% |
| 1K | 65.43 | 80.24 | 90.20 | -11.0% |
| 2K | 59.52 | 77.05 | 87.65 | -12.1% |
| 4K | 58.14 | 82.83 | 85.33 | -2.9% |

The experimental candidate is 23–45% faster than same-binary AR across these
sampled cases, but **the top-p 0.95 curve misses the 10% gate at three contexts**.
Untuned default MTP is still not consistently faster than AR. Do not infer
overall sampled parity or promote defaults from the top-p 1.0 result alone.
The top-p change also changes generated text and speculative acceptance;
the difference between these curves is not a measurement of sorting cost alone.

The candidate explicitly uses depth 4, batched verification, attention chunk 2,
fused HC/router, native HC chain, verification dispatch stride 8 and draft
dispatch stride 1. The untuned arm uses ordinary `--mtp`, strict verification
and depth 1; neither `AFM_DEBUG` nor `AFM_PERF` is set in any timed arm. Launch
manifests preserve the complete commands and actual inference-binary hashes.

TTFT-derived prompt throughput is retained separately below. It includes
first-token work and API overhead: **it is not pure device-prefill throughput**.

| Context | AFM candidate, p=1 | Reference, p=1 | AFM candidate, p=0.95 | Reference, p=0.95 |
|---|---:|---:|---:|---:|
| 0.5K | 970.90 | 848.55 | 968.60 | 853.08 |
| 1K | 1128.83 | 1038.63 | 1125.53 | 1019.46 |
| 2K | 1288.95 | 1182.19 | 1286.91 | 1182.47 |
| 4K | 1318.56 | 1257.72 | 1311.14 | 1254.56 |

The separate sampled known-answer control passes **24/24 for the candidate
and 24/24 for AR** at temperature 0.6/top-p 0.95. These are eight established
arithmetic, extraction, logic, Unicode and lookup cases repeated three times
with the same seed, not 24 independent prompts or a broad quality benchmark.
Inspected context responses are coherent, with expected truncation at the
128-token timing limit. This is not comprehensive AI-judge qualification.

A same-binary greedy regression also preserves **12/12 prior experimental
candidate responses** from `draft-ladder1-repeat-verify8-depth4-afm-mtp-1`.
Its medians are 90.87 / 82.17 / 89.56 / 74.72 tok/s, with the same explicit
candidate settings. This supports absence of a greedy regression in these
four contexts, not a universal equivalence or performance guarantee.

Evidence is appended under `sampled-t06-*`, `sampled-quality-*`, and
`sampled-qualification-v1-*`, with the greedy control under
`sampled-greedy-regression-v1-*`, in the existing external artifact root.
Original greedy baselines remain unchanged; reports are untracked.

### Follow-up: bounded draft-dispatch overlap

An additional off-by-default scheduling experiment dispatches early draft
tokens while Swift constructs the rest of the fixed-depth head chain:
`AFM_QWEN_MTP_DRAFT_ASYNC_LADDER=<stride>`. It neither adds draft steps nor
changes the target verifier, weights, sampling, precision or committed head
history. The head disables PLE; only the backbone verifier needs to flush
deferred PLE leaves. Source comments distinguish this within-chain experiment
from the reference's chunk-boundary dispatch.

The Release build passes (`build-draft-ladder.log`, 97.21 s); all 23 focused
tests pass (`test-draft-ladder.log`). The added test compares disabled dispatch
with strides 1/2/4 under strict and batched policies, depths 1/3/4, repeated
requests, first-token and later cancellation, EOS and exact output prefixes.

The first uninstrumented stride-1/depth-4/verification-stride-8 arm measures
91.93 / 82.15 / 89.12 / 75.15 tok/s, preserving all 12 previous candidate
texts. **Do not attribute that whole improvement to draft dispatch:** the
same rebuilt binary with draft dispatch disabled measures
89.14 / 81.04 / 88.88 / 74.15. A repeated enabled arm measures
90.63 / 81.06 / 89.66 / 75.50 and preserves all twelve texts. Thus the
repeated enabled path is within about 3% of the frozen reference, but the
increment over its same-binary disabled control is much smaller than the
between-build gain. No claim of a 4–5% dispatch-only win is justified.

The 23-test suite also passes with the native HC chain enabled
(`test-draft-ladder-native-chain.log`). The enabled draft-dispatch candidate
passes the 24/24 known-answer rerun. Full API cache/concurrency coexistence
qualification is separate and must not be confused with continuous MTP
batching: Qwen MTP currently uses a serial lane with request-owned cold caches;
ineligible requests can use the ordinary AR scheduler.

Code is checkpointed at `9f3155b2`. The selected existing API contract sections
6/7/2/3/8/15 also pass with the enabled candidate and
`--enable-prefix-caching --concurrent 2`: **44 functional assertions and 12
repeated preflight assertions**, no failures or skips. These cover safe
shared-prefix fallback, replay consistency, divergent branch isolation,
concurrent requests, stop handling, logprobs, errors, and batch API dispatch.
They validate coexistence, not MTP prefix-reuse acceleration or continuous
speculative batching. No broad quality-equivalence claim follows from them.

### Fresh reference cross-check (frozen baseline retained)

After the AFM runs, the same frozen v26.9.2 reference binary was launched again
with the same checkpoint, prompts, sampling and cache-disabled arguments.
Its three-trial medians were **95.34 / 84.36 / 85.25 / 80.87 tok/s**. Results
are appended under `dispatch-checkpoint-confirmation-reference-mtp-1`; the
original frozen result files are unchanged. The reference's adaptive state
and before/after round-cost tables are captured. At 4K, its first trial also
produced different text from the second and third; this is not an exact-output
fixed-work comparison across all reference trials.

Compared with this fresh reference repeat, AFM's repeated enabled candidate
is **-4.9% / -3.9% / +5.2% / -6.6%**. Against the stronger reference median
at each context from either reference run, it is **-4.9% / -3.9% / -2.5% /
-6.6%**. Thus the experimental 10% gate is met against both reference curves;
equality, default parity, and broad quality equivalence are still not claimed.

### Follow-up: verifier dispatch overlap reaches the experimental 10% gate

Code checkpoint `2bb0c0c6` preserves the lazy-range change and bounded,
off-by-default verification-dispatch experiment. With a dispatch every eight
layers, **two separate uninstrumented runs** meet the first-four-context 10%
minimum. This is not equality with the reference, default-setting parity, or
release qualification.

| Context | Frozen reference | Candidate run 1 | Candidate run 2 |
|---|---:|---:|---:|
| 0.5K | 90.94 | 88.11 | 87.59 |
| 1K | 82.69 | 79.20 | 79.12 |
| 2K | 91.99 | 85.24 | 85.75 |
| 4K | 75.93 | 71.56 | 71.36 |

Each cell is a median of three 128-token requests, with one excluded warmup
per context. Same checkpoint, temperature zero, no thinking or prefix reuse.
Neither run sets `AFM_DEBUG` or `AFM_PERF`. Both explicitly use depth 4,
batched policy, attention chunk 2, fused HC/router, native HC chain, and
`AFM_QWEN_VERIFY_ASYNC_LADDER=8`. The previously reported stride-16 run only
narrowly passed, and its uninstrumented repeat missed 2K; it is not the basis
for this claim. Launch manifests and binary SHA-256 values are retained.

All 12 texts from each new arm match the corresponding pre-ladder depth-4
candidate. Those texts are **not** identical to AR: batched arithmetic remains
an explicitly approximate experiment. Basic coherence and repeatability are
insufficient to promote that policy. Known-answer controls, live cache and
concurrency checks, and further performance work remain necessary.

The final binary's untuned `--mtp` control measures 72.89 / 71.04 / 61.89 /
60.60 tok/s and matches all twelve pre-range, corrected-rollback default texts.
The range construction improvement therefore also benefits default MTP,
without enabling the experimental policy. It does not achieve default parity.

The untuned AR control is 68.90 / 67.24 / 62.39 / 61.27 tok/s; all twelve
texts match the earlier AR control. The bounded known-answer suite passes
24/24 in each of candidate MTP, strict-default MTP, and AR. Candidate and AR
texts also agree for all 24 of these short responses. The eight distinct
cases cover arithmetic, sorting, extraction, logic, code evaluation, Unicode,
current-versus-stale facts, and coordinates, repeated in alternating order.
These checks are not a broad model-quality evaluation and are not included
in the timed context benchmark.

### Follow-up: sampled host hotspot and lazy integer ranges

A separate, non-comparable 512-token diagnostic at 4K was sampled with macOS
`sample`. On the generation thread, the QSA mask path repeatedly entered
`MLXArray.__allocating_init` → Swift generic `Sequence` enumeration when
constructing context-sized integer position arrays. This was CPU range
construction, not Metal compilation or evidence that integer arithmetic was
slow on the GPU. The reference builds these arrays with `mlx_arange`.

Replace those QSA block/position ranges with lazy, explicitly int32 `MLX.arange`
operations. No checkpoint, mask value, attention arithmetic, precision, shared
cache, or growing-context specialization is introduced. Tests compare masks
against a host oracle through 32K, including incomplete causal tails,
sentinels, multiple rows/requests, and position-history extension.

| Diagnostic candidate | 0.5K | 1K | 2K | 4K |
|---|---:|---:|---:|---:|
| Fused HC + fused router, depth 3 | 86.36 | 82.66 | 72.69 | 62.18 |
| Same + lazy QSA ranges, depth 3 | 85.21 | 81.94 | 75.83 | 69.76 |
| Same + lazy QSA ranges, depth 4 | 83.61 | 76.90 | 80.45 | 67.41 |
| Also native HC chain, depth 4 | 84.50 | 76.17 | 81.19 | 68.02 |

These use the same 128-token measurement contract and explicit batched-policy,
chunk-2, HC/router and diagnostic flags. They are not default performance.
The range-only depth-3 comparison preserved **12/12 response texts**; 4K improved
12.2%, while short-context differences are small. At this stage no single arm
met all four reference MTP points. Fixed depth 6 was slower
(71.14 / 65.91 / 61.89 / 53.99), so blindly increasing depth is not a remedy.

The fused router entry point is bounded to independent verification rows;
the existing public decode entry point still declines multirow inputs. Its
outputs match independent decode-router rows exactly in the tested geometry.
It remains opt-in via `AFM_QWEN_VERIFY_FUSED_ROUTER=1`.

Before the range change, a fresh no-tuning MTP control measured
70.85 / 67.92 / 52.20 / 50.61. Nine of twelve responses differed from the
earlier control following the recurrent rollback correction; do not claim
default-mode non-regression from the experimental improvements. AR measured
67.09 / 65.64 / 60.17 / 57.61 and preserved all twelve earlier AR responses.
Fresh default controls remain necessary on the final candidate.

Two further experiments remain disabled by default:

- `AFM_QWEN_VERIFY_FUSED_MASK=1` expands sorted selected blocks with one
  bounded binary-search mask kernel. Exact masks agree with the composed path
  through 32K, including production selection capacities. It did not establish
  an additional throughput win (depth 4: 85.28 / 76.49 / 80.82 / 68.11).
- `AFM_QWEN_VERIFY_ASYNC_LADDER=<stride>` submits bounded batched-verification
  prefixes while Swift builds the rest. This is distinct from the AR ladder.
  Before **every** submission it flushes pending mapped PLE leaves; an unfilled
  leaf must never reach the GPU. Full-model tests compare dispatch strides
  1/4/8 with no early dispatch, including EOS n-gram history, separate request
  caches and every partial-acceptance rollback boundary. The subsequent
  uninstrumented performance measurements are recorded above.

The current focused suite has 22 passing tests (`test-verify-ladder.log`). The
first fused-mask attempt failed Metal compilation because its scalar input was
indexed as a pointer; the corrected one-element-vector input passed the rerun
before any model benchmark. Failed diagnostic logs are retained, not hidden.

### Follow-up: exact rollback gates and bounded verifier experiments

An FP32 state-snapshot experiment exposed a real partial-rollback discrepancy.
The forward verifier consumed gates produced by fused prework, but rollback
recomputed them through a different fused-gating path with different rounding.
Full-window state agreed while 29 intermediate-state assertions failed. The
fix carries the **actual forward gate and beta arrays** through the functional
compiled boundary and reuses their accepted prefix during replay. No recurrent
precision was reduced. This corrects replay; old incorrect token trajectories
are not a correctness oracle.

The snapshot experiment stores the intermediate FP32 states in the recurrent
kernel, then gathers just the accepted state on rollback. It remains off:
depth 3 adds about 324 MiB of temporary recurrent history on this checkpoint
without an observed throughput gain. Snapshot and corrected replay output text
matched across the sampled contexts, and tests require exact FP32 intermediate
state equality at every tested acceptance boundary.

| Diagnostic candidate | 0.5K | 1K | 2K | 4K |
|---|---:|---:|---:|---:|
| Corrected replay, depth 3 | 80.26 | 78.64 | 66.00 | 57.19 |
| FP32 snapshots, depth 3 | 79.98 | 76.68 | 64.41 | 55.91 |
| Corrected replay, depth 4 | 72.44 | 77.16 | 67.72 | 56.90 |
| Radix QSA selector, depth 3 | 80.95 | 78.58 | 66.34 | 57.21 |
| Fused verification HC reads, depth 3 | 82.88 | 82.04 | 72.49 | 61.72 |

All rows explicitly select batched verification and attention chunk 2, with
`AFM_DEBUG=1 AFM_PERF=1`; these are **not default-setting measurements**. Each
additional experiment is opt-in. All use the same original checkpoint. The
radix selector did not establish an end-to-end speedup. It retains current
biased score arithmetic and replaces only selection, using the MIT-licensed
`QSA_SELECT_KERNEL_SOURCE` from David Dalcu's mlx-serve at
`1ec580a8b7f5f051daef892310660bb62b2ece6c`; copyright and license are in source.
Tests cover causal bounds, short rows, deterministic ties, signed zeros, NaNs,
growing score-bank lengths, and incomplete causal tails.

Fused HC reads changed the shorter-context responses, while the 2K and 4K
sampled response texts remained unchanged. Those longer-context gains therefore
are not solely an easier generated text. The summaries inspected were coherent,
but this is not full quality qualification. Even this candidate remains about
21% / 19% below the reference at 2K / 4K: parity is still not achieved.

Experimental controls added in this follow-up (unset = off):
`AFM_QWEN_MTP_STATE_SNAPSHOTS`, `AFM_QWEN_VERIFY_QSA_RADIX`,
`AFM_QWEN_VERIFY_FUSED_HC`, and `AFM_QWEN_VERIFY_DEFER_HC`.
They do not change default depth, strict policy, ordinary AR, or other model
architectures. Cache/concurrency qualification and uninstrumented reruns are
still required before adopting any experiment as a default.

The deferred-HC follow-up preserves the original single-rounded fused
injection when a verification layer consumes its predecessor's pending write.
The old pending path instead rounded the product before addition. An explicit
kernel specialization selects the matching rounding for this experiment;
ordinary AR retains its prior behavior. Eighteen targeted tests pass, both with
the ordinary Metal path and the optional native HC chain. They cover production
HC geometry (Q4/Q8, widths 2/4/7/8), deferred writes versus materialized writes,
PLE boundaries, partial recurrent rollback, and separate requests/models.

### Common setup

- M3 Ultra, same `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit` checkpoint and mapped n-gram sidecar.
- Frozen context prompts: 493, 864, 2,112 and 4,150 input tokens; 128 generated tokens.
- Temperature zero, thinking off, prefix reuse disabled; one GPU workload at a time.
- Three measured trials per candidate/context, after excluded warmups.
- Client decode rate is `(output tokens - 1) / (last text time - first text time)`.
- Prompt throughput in the saved client reports is a **TTFT-derived proxy**, not isolated GPU prefill throughput.
- Frozen reference v26.9.2 uses adaptive MTP; AFM's default is depth 1. Product-default comparisons are not matched-depth comparisons.

| Context | Frozen reference MTP | AFM `80c9fd15` MTP | Deferred PLE only | Deferred PLE + attention chunk 2 |
|---|---:|---:|---:|---:|
| 0.5K | 90.94 | 68.05 | 67.65 | 72.22 |
| 1K | 82.69 | 67.42 | 71.60 | 71.94 |
| 2K | 91.99 | 55.48 | 58.40 | 58.75 |
| 4K | 75.93 | 50.31 | 52.18 | 52.80 |

Values are median decode tokens/s. The PLE-only arm has no tuning variables;
the last column explicitly sets the existing `AFM_QWEN_VERIFY_ATTENTION_CHUNK=2`
control. Its result must not be described as a default-setting measurement.
Both arms match `80c9fd15`'s saved response text in **12/12** measured requests.
Some trials have visible timing variation, especially at the shortest context;
small differences are not independently established wins.

## Deferred mapped n-gram lookup

Previously, CPU construction of the verification graph stopped at PLE until
the GPU finished generating the speculative token IDs. The reference schedules
this host read later. AFM now implements that scheduling within one forward:

```text
GPU:  draft chain -------------------> verification evaluation
CPU:       build verification graph
           using a private PLE leaf
                                  resolve draft IDs
                                  hash/gather rows
                                  fill leaf + advance history
                                  return graph to evaluator
```

The leaf is a realized, privately owned BF16 array. It is filled once, before
any downstream graph is evaluated. The fill queue is local to the forward;
there is no process-global mutable placeholder or cross-request token buffer.
Unfilled abandoned work releases its closures. Flush completes n-gram history
updates before cache rollback can run. Ordinary AR and resident-table execution
retain their previous paths. Synchronizing profilers keep eager lookup.

Tests cover a graph built before its leaf is filled, compiled consumers,
independent requests, abandoned work, mapped-table EOS history and every partial
acceptance/rollback boundary in a small model. These do not replace live API
concurrency, radix-cache reuse and model-switch qualification.

## Masked attention grouping

The reference's `splitMaskedSdpa256` bounds each group by `query rows * GQA <= 32`
to retain the vector-attention kernel. AFM's diagnostic chunk-2 path now preserves
the actual per-row QSA masks instead of forcing all masked verification to
single rows. Tests compare BF16 attention at 24 query heads, 2 KV heads and head
dimension 256 against independent singleton verification, including an odd tail.
The depth-1 live comparison retained all saved token sequences.

## Experiments not adopted

- **Merged draft-head history / last-row-only tail:** little speed benefit and a
  changed token trajectory for the shortest prompt. Removed; ordinary head
  progression remains intact.
- **3-bit auxiliary vocabulary readout + original-head top-32 rescoring:** the
  target head was unchanged, but the generic selector did not justify its added
  memory and latency. Removed. A specialized top-k kernel would need a new A/B.
- **General batched verification:** explicitly measured as a diagnostic, not
  promoted to the default. It did not close the gap.
- **Unsorted multirow affine expert kernels:** earlier experiment did not
  materially improve the default path. Its patch remains in local evidence.

The expert-scheduling experiment follows a concrete source difference:
the reference sorts every multi-token token/expert assignment set; the general
Swift path previously sorted only at 64 assignments, while strict verification
processed singleton rows. Sorting must preserve routing, inverse permutation,
numerical quality and request-state boundaries; its performance is not assumed.

The sorted path is currently wired only to the explicitly selected `.batched`
verification experiment. It does not alter strict-default MTP or other models'
sorting thresholds. Follow-up experiments compile the functional expert tail,
GDN verifier, attention projection/output, and fixed-width indexer projection.
Request cache updates and growing QSA selection graphs remain outside those
closures. GDN convolution/recurrent inputs and rollback intermediates are
explicit arguments/results; recurrence stays FP32. No checkpoint is rewritten.

### Batched verification experiments

These rows explicitly select `.batched`, depth 3, and phase diagnostics. They
are **not strict-default results**. Attention grouping is explicitly enabled
only for the rows marked chunk 2.

| Incremental candidate | 0.5K | 1K | 2K | 4K |
|---|---:|---:|---:|---:|
| Sorted experts | 63.22 | 62.37 | 54.12 | 44.89 |
| Compiled expert tail with row-independent HC | 70.59 | 77.59 | 57.58 | 49.62 |
| Also compiled GDN | 67.54 | 76.78 | 58.98 | 49.68 |
| Also attention chunk 2 | 75.64 | 72.54 | 61.13 | 54.64 |
| Also compiled attention projections | 78.43 | 75.32 | 64.12 | 54.07 |
| Also compiled indexer projections | 74.85 | 74.63 | 64.98 | 54.38 |

Compiling GDN or indexer projections alone did not establish a speedup. The compiled expert-tail
candidate also changes HC arithmetic to the existing row-independent kernels;
its gain cannot be attributed exclusively to compilation. Its saved texts differ
from the prior sorted-only arm in 12/12 requests. Adding compiled GDN preserved
12/12 texts; changing attention grouping preserved 6/12. All inspected context
summaries are coherent, but those observations do **not** establish model-quality
equivalence or exact AR-token parity. Batched-mode arithmetic remains an
experiment, not a default recommendation.

Depth 2 with compiled attention projections measured 77.25 / 78.25 / 62.12 /
53.16 tok/s. No single measured configuration meets the full reference curve.
Repeated requests are deterministic within each of these candidate arms.

The follow-up default `--mtp` control, with no tuning variables or depth
override, measured 67.63 / 68.66 / 58.19 / 52.22 tok/s and reproduced 12/12
saved `80c9fd15`/PLE-only response texts. Do not present the experimental
74–78 tok/s short-context figures as the default behavior.

The same binary's untuned AR control measured 69.24 / 68.69 / 61.26 / 59.03
tok/s, versus the earlier 68.81 / 68.22 / 60.18 / 59.88. All 12 AR response
texts matched the earlier AR control. Neither experimental MTP nor default MTP
is consistently faster than AR at the longer contexts yet.

Focused tests compare compiled and uncompiled functional bodies across verifier
widths, positions and model instances, test FP32 recurrent state and every
rollback prefix, and check production-geometry causal/QSA attention grouping.
The first projection-test build was invalidated when its fixture was corrected
during compilation; the clean rerun passed. This is not release qualification:
live radix-cache reuse, concurrency, cancellation and model-switch qualification
remain required before promoting an experimental policy.

## Provenance and evidence

Scheduling and attention-grouping ideas derive from David Dalcu's MIT-licensed
[`mlx-serve`](https://github.com/ddalcu/mlx-serve) at
`1ec580a8b7f5f051daef892310660bb62b2ece6c`, principally `src/generate.zig`
and `src/transformer.zig`. Original weights are not modified by these changes.

Raw requests, SSE responses, texts, executable hashes, launch settings, build
logs and discarded experiment patches are retained in
`/Volumes/edata2/afm-benchmarks/qwen-next-mtp-parity-20260909`.
The frozen reference is in the adjacent `qwen-next-rebaseline-20260909` folder.
Bulky artifacts remain untracked. The installed Homebrew nightly is unchanged.
