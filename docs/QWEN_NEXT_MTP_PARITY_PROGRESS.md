# Qwen Next MTP parity: controlled implementation work

This records intermediate work for AFMKit PR #123, following research PR #122.
**The MTP performance goal is not achieved.** These are targeted experiments,
not release qualification, and default MTP is not yet consistently faster than AR.

## Measurement contract

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
