# Qwen Next mixed-position verification experiment

Work on AFMKit PR [#123](https://github.com/scouzi1966/AFMKit/pull/123),
September 12, 2026. **Research only, off by default.** This does not establish
reference parity or recommend a production preset.

## What is being isolated

The existing shared MTP verifier requires equal target positions. On the
15-request coding-review workload, its shared groups average about 2.61 rows
despite a four-row window. Requests at different positions cannot join those
groups even when they use the same model, MTP head and speculative depth.

The new adapter shares only compatible fixed-size recurrent/PLE state and
backbone operations. Attention histories remain separate, at their real
positions. It does not pad KV arrays, reuse another request's QSA index, or
pretend that different positions are equal.

```text
Request A: position a, private KV/QSA ─┐
Request B: position b, private KV/QSA ─┼─ shared bounded target backbone
Request C: position c, private KV/QSA ─┤  [requests, verification tokens]
Request D: position d, private KV/QSA ─┘
                                      │
                  attention layers: separate authoritative row updates
                                      │
                  split results and complete rollback metadata
                                      │
                  independent sampler / acceptance / trim / repair
```

Attention cache wrappers clone the existing lazy array values before the
forward; adoption happens only after the shared forward has been constructed.
Recurrent cache splitting reuses the existing rollback implementation,
including PLE histories. Group validation finishes before attention clones
are made. The existing PLE flush-before-submission discipline is unchanged.

Eligible groups have 2–4 requests, one model/head/depth, the explicit batched
verification policy and at most 16 request/token rows. Strict verification,
ordinary AR and unsupported state retain their existing fallback. This is
not a generic continuous-batching implementation for other architectures.

## Controls

Use the complete existing [shared-verifier recipe](QWEN_NEXT_SHARED_VERIFIER_OPT_IN.md)
with ladder 4 and compiled shared tail off, then add this **process-start**
setting only to the benchmark process:

| Setting | Unset behavior | Experimental selection |
|---|---|---|
| `AFM_QWEN_MTP_INDEPENDENT_ATTENTION` | Off | `1` permits mixed-position groups with private attention histories |

Only the independent-attention option is retained. The unsuccessful projection
setting and all three extra projection implementations were removed after
the screens below. Do not add `AFM_QWEN_MTP_ATTENTION_PROJECTIONS` to a launch
command: it is no longer wired to runtime code. The retained option does not
change model quantization, MTP depth, sampler defaults, maximum request
concurrency, replay-cache budget or the exact AFMKit dependency pin.

The removed shared projections used bounded native matrix operations. A follow-up
`compiled` mode placed those operations in model-owned pure compiled regions;
it passed positions as array inputs and captured no mutable request cache.
Each row still performed its own causal attention, index update and KV growth.
Different matrix reduction geometry
can change logits and generated wording; this is not strict non-MTP token
equivalence. No default quality/performance tradeoff is approved by this work.

## Initial independent-attention comparison

M3 Ultra, 512 GiB; exact checkpoint:
`/Volumes/edata2/models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit`.
The frozen workload uses concurrency 15, prefix cache on, MTP depth 3,
temperature 0, top-p 1, seed 42 and a 512-token output budget. One excluded
client warmup precedes 15 distinct requests and their 15 exact-prompt repeats.
The first round is not wholly cache-cold: a common prompt/warmup may be reused.

Both arms in each pair use binary SHA-256
`0908e30e0a079b3ddd253eb30023622c033917822c611ae034e60d818a205999`.
Projection sharing had not yet been added to this binary.

| Order and measure | Equal-position control | Independent attention | Change |
|---|---:|---:|---:|
| Candidate-first: first aggregate tok/s | 65.09 | 63.88 | −1.85% |
| Candidate-first: repeat aggregate tok/s | 99.39 | 102.20 | +2.83% |
| Candidate-first: repeat valid tasks/s | 0.5624 | 0.5570 | −0.97% |
| Control-first: first aggregate tok/s | 63.94 | 67.02 | +4.80% |
| Control-first: repeat aggregate tok/s | 98.67 | 106.77 | +8.21% |
| Control-first: repeat valid tasks/s | 0.5797 | 0.5862 | +1.11% |

Mean shared-group size increased from 2.61–2.62 to 3.44–3.45 rows. All 120
measured requests completed, with no observed compiler/test/other-AFM collision
in the process-name guard. Servers exited 0. Peak RSS was 69.53–69.79 GiB;
these short runs are not a leak soak or a minimum-memory qualification.

Only 1/30 and 2/30 paired responses respectively were identical in text and
token count. Structural checks were 28/30 versus 29/30 in the first pair and
29/30 versus 29/30 in the second. A missing `fix` field moved from first-round
AGENT-13 in the second control to repeated AGENT-13 in its candidate. The
candidate stopped normally after 108 tokens: this was not truncation. Similar
omissions occur in controls, but the cause of the change is not isolated.

"Valid tasks" here means JSON/required-field/request-identity checks, **not**
semantic correctness. In particular, a structurally passing proposed stream
fix can still drop a final token. These checks cannot justify a quality-parity
claim, and higher token counts alone are not evidence of faster useful work.

## Projection-sharing screen

The three-way same-binary screen used
`a29e85232f818109558267359d489f255bbdf651bb2f91dc6d7719a5406f7023`.
Independent attention remained enabled for every arm; all other workload
settings above were unchanged.

| Attention projections | First aggregate tok/s | Repeat aggregate tok/s | Structural checks | Peak RSS, GiB |
|---|---:|---:|---:|---:|
| Separate | 68.35 | 103.98 | 29/30 | 69.784 |
| Shared output only | 66.63 | 102.43 | 30/30 | 69.785 |
| Shared Q/K/V, index, normalization and output | 64.67 | 103.40 | 30/30 | 69.784 |

All 90 measured requests completed and servers exited 0, without observed
process-name guard collisions. Compared with separate projections, output-only
sharing reduced repeated valid-task throughput by 4.62%; all-projection sharing
reduced it by 0.63%. Only 4/30 and 6/30 paired responses respectively were
identical in text and token count. The control omitted `fix` on first-round
AGENT-13; neither projection-sharing arm introduced a structural failure.

**Neither native projection-sharing variant demonstrates a performance win.**
Reading all 30 all-projection diagnoses/fixes found relevant, readable answers,
but also an ambiguous early-break stream fix that can discard final content.
This manual check is not an independent AI judge or semantic pass total.

The source-level reason to test a compiled follow-up is that the shared adapter
bypasses the existing per-row compiled projection regions. This is a hypothesis
about lost graph optimization, not a GPU-profiled explanation of the timings.

### Compiled follow-up and disposition

A fresh same-binary screen ran compiled sharing first, native full sharing
second, separate projections last. Binary SHA-256:
`d3c2d41f6422e66f828a04a9fa3b44ca1d1b6fa08235af53c893af1b40601da9`.
The consumer Release build passed in 96.39 seconds; its focused suite passed
85 tests with two optional skips and no failures, including compiled projection
and rollback oracles.

| Attention projections | First aggregate tok/s | Repeat aggregate tok/s | Structural checks |
|---|---:|---:|---:|
| Compiled full sharing | 67.82 | 104.63 | 29/30 |
| Native full sharing | 67.30 | 104.52 | 29/30 |
| Separate | 68.58 | 105.93 | 29/30 |

All 90 measured requests completed without an observed guard collision and
servers exited 0. Relative to separate projections, compiled sharing reduced
repeat token throughput by 1.23% and repeat valid-task throughput by 5.67%.
Four of 30 paired responses were identical. Again the missing-field failure
moved between first and repeated AGENT-13; equal pass totals do not mean the
same cases passed.

Neither native nor compiled projection sharing earned retention. Their code,
new selector and selector-specific tests were removed, with reproducible
source patches and all failed-to-improve measurements preserved externally.
The mixed-position adapter keeps ordinary per-request attention projections.
No other existing projection control was removed.

## Retained-source confirmation

After removal, the rebuilt retained source was measured control-first with the
same checkpoint, frozen requests, prefix caching, C15, depth 3 and 512-token
budget. Both arms use the final `e0faa66e…` binary identified below.

| Measure | Equal-position control | Independent attention | Change |
|---|---:|---:|---:|
| First distinct aggregate tok/s | 65.90 | 68.48 | +3.93% |
| Repeated aggregate tok/s | 100.31 | 108.43 | +8.09% |
| First valid tasks/s | 0.3750 | 0.3869 | +3.18% |
| Repeated valid tasks/s | 0.5670 | 0.6043 | +6.58% |
| Repeat median TTFT, seconds | 0.1126 | 0.0932 | −17.20% |
| Repeat median request latency, seconds | 23.05 | 21.36 | −7.35% |
| Mean rows/shared group | 2.623 | 3.466 | More shared rows |
| Peak process RSS, GiB | 69.77 | 69.62 | No observed increase |
| First/repeat structural passes | 15/15; 14/15 | 14/15; 14/15 | One new first-round omission |

All 60 measured requests completed with exit-0 servers and empty process-name
collision lists. Only 1/30 paired responses matched in text and token count.
The additional candidate failure was first-round AGENT-13, missing `fix` and
stopping normally after 108 tokens. It is the same type of omission seen in
earlier controls, but its cause is not isolated; it is not silently waived.
Repeat output counts were 2,477 versus 2,512 tokens, so the 8.09% token-rate gain
is not an identical-generated-work timing comparison.

Across the three mixed-position pairs, repeat token gains were 2.83%, 8.21%
and 8.09%, while repeat structurally valid-task gains were −0.97%, +1.11%
and +6.58%. This supports further controlled development, not a large or
universal useful-throughput claim. No semantic quality parity, reference
parity, longer-context memory qualification or production promotion follows.

## Validation and evidence

The initial adapter executed 86 focused Release tests: 84 passed, two optional
probes skipped, no failures. With full projection sharing enabled, the expanded
suite executed 87 tests: 85 passed, two optional probes skipped, no failures.
After removing the unsuccessful projection variants, the retained source again
passed 84 tests with two optional skips and no failures (59.75 seconds of tests,
124.61-second test build). The final consumer Release build passed in 94.97
seconds, reporting `v0.9.20`, with SHA-256
`e0faa66ecf21c42e2f0ff79afbbab5f15fd57c7b95d5b411fc3969e1230d8b26`.
Coverage includes mixed positions crossing sparse-history boundaries, all
draft acceptance frontiers, independent subsequent decode, mapped PLE leaves,
sampling, cancellation and strict fallback. Attention projection/trim oracles
cover FP32 and Q4/BF16 at B2/T4, B3/T4, B4/T4, B2/T7 and B2/T8. These bounded
synthetic-weight tests do not replace actual-checkpoint quality qualification.

The retained binary also passed **120/120 live API lifecycle assertions** at
concurrency six: early disconnect/cancellation, successful subsequent requests,
repeated prefixes, token limits, stops, sampled temperature 0.6/top-p 0.95,
logprobs and ordinary-decoding fallback alongside MTP. The server exited 0;
shutdown confirmed independent attention and 12 actual shared verification
groups covering 29 rows. This separate safety workload is not a throughput
measurement or a full production qualification.

The complete experiment inventory has **12 clean timing arms / 360 completed
measured requests**, excluding warmups and lifecycle requests. Source review
was local; no new independent reviewer or external AI judge was run in this
iteration. All changes and performance proposals remain on the same PR.

Raw evidence stays untracked under
`/Volumes/edata2/afm-benchmarks/qwen-next-mtp-parity-20260909`:

- `mixed-attention-20260912-pair{1,2}-*`: matched initial comparisons and raw SSE.
- `mixed-attention-20260912-initial-source.patch`: exact initial adapter diff.
- `mixed-projections-20260912-source.patch`: projection prototype diff.
- `mixed-*-20260912-tests*.log`: tests, including the retained initial compile error.
- `mixed-projections-20260912-build.log`: successful consumer Release rebuild,
  98.68 seconds, binary `a29e85232f818109558267359d489f255bbdf651bb2f91dc6d7719a5406f7023`.
- `mixed-projections-compiled-20260912-*`: compiled follow-up source, tests and
  same-binary screen, including unsuccessful results.
- `mixed-position-final-20260912-*`: retained-source tests, build, matched
  comparison, lifecycle data and source diff against `06c3d10a`.

No profiling environment variables are enabled in timing arms. Host-lap
diagnostics from earlier experiments are not GPU kernel timings. No release,
installation, consumer dependency bump or default promotion is part of this
experiment.

## Next isolated targets

- Measure a bounded shared vocabulary projection: the shared target backbone
  still calls `targetTokens` separately for each request. Any shared projection
  must preserve request-local sampler order and bound temporary logits; it must
  not replace independent sampling with a shared random stream.
- Evaluate larger useful groups only with the fast-path geometry intact.
  Current shared HC and submission eligibility cap the product of requests and
  verification tokens at 16, with at most four requests. Merely increasing the
  scheduler window would silently fall back from those optimizations. Extend
  or tile those bounded regions, then test mixed positions, rollback, memory,
  fairness and useful aggregate throughput before changing that limit.

These are follow-up experiments, not implementations or promised speedups.
