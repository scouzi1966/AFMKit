# Qwen Next: committed-anchor head repair

This is an original AFM implementation experiment on the MTP parity branch,
not a claim of research priority or a new trained model architecture. It does
not copy the reference's head-history repair implementation. The inspected
reference (`mlx-serve`, `src/generate.zig`, revision
`1ec580a8b7f5f051daef892310660bb62b2ece6c`) restores the round origin and
re-appends all committed pairs from true backbone streams.

## Observation

The draft head consumes `(hidden stream, token)` pairs. At the start of each
round, both the primary token and its preceding backbone stream are already
verified. Consequently, the first appended head-cache row is not speculative.
Only later rows consume predicted streams. Accepting a later token does **not**
prove that its predicted stream was correct.

For draft depth four and two accepted proposals:

```text
Before repair:  [old committed history] [anchor] [predicted] [predicted] [predicted]
                                      valid     discard     discard     discard

Full repair:   [old committed history] + recompute(anchor, true pair 1, true pair 2)
Anchor repair: [old committed history] [anchor] + recompute(true pair 1, true pair 2)
```

At zero acceptance, anchor repair makes no additional head call. At full
acceptance it still repairs every suffix row; it never assumes predicted
streams are valid just because all draft tokens matched.

## Contract

- Requires `AFM_QWEN_MTP_RETAIN_ANCHOR=1` and the explicitly experimental
  batched verification policy. Strict/default generation ignores the option.
- The current head has one trimmable Qwen attention cache. Retention is
  declined if the cache layout or appended-row count differs from expectation.
- Retain exactly one row, trim the remaining draft rows, and append accepted
  token/true-stream pairs at their original absolute positions.
- The final head offset is always `round origin + accepted proposals + 1`.
  Target-cache commit/rollback, EOS, cancellation and token acceptance rules
  are unchanged. Every request owns its caches and random sampler.
- No additional host/GPU synchronization, model-weight copy, shared history,
  worker thread, or growing cache is introduced.

The mathematical committed history is unchanged. Floating-point results need
not be bitwise identical to the old **batched** repair: its matrix widths differ.
That can change future draft proposals and acceptance, so latency wins cannot
be inferred simply from the number of rows removed. The target continues to
verify proposals, but broad quality equivalence is not claimed from that alone.

## Qualification

1. Enumerate every acceptance prefix at depths 1, 3, 4 and 7; check trimming,
   repair range, empty repair and final offsets.
2. Compare retained-anchor cache state with an independent true-stream,
   singleton repair. Exercise QSA block boundaries and deliberately wrong
   predicted streams so accidentally retaining a suffix cannot pass.
3. Check repeated seeded requests, interleaving, cancellation, EOS and the
   strict-policy opt-out; retain the existing pipeline and admission suites.
4. Build Release through the reliable wrapper, then run same-binary enabled
   and disabled comparisons on the exact ddalcu checkpoint. Known answers and
   API qualification are separate from uninstrumented timing.
5. Record both performance and any text/acceptance changes. Do not promote a
   default or describe a performance/quality tradeoff as resolved without data.

The expected saving is bounded: one head-history row per completed cycle,
occasionally an entire repair call. This does not eliminate backbone
verification, PLE lookup, or all rejected-suffix work. It is a first experiment
in reducing redundant work at the acceptance boundary, not a promised parity
breakthrough. Adapting the idea to recurrent or multi-layer heads would require
their own valid-prefix state restoration and independent qualification.

Baseline code checkpoint: `8d08b6c68b835f2de7c119ada4d52f3b1a05ec55`.
Artifacts: `/Volumes/edata2/afm-benchmarks/qwen-next-mtp-parity-20260909`,
using the `anchor-*` and `test-anchor-*` prefixes.

## First completed experiment

The Release build passed in 91.75 s; binary SHA-256 is
`73da1c14abdbdb5a5d4a605bf2bcc4d8089855294a055974936e1c468366c1c1`.
62 focused tests passed and one optional timing test was skipped. The live
enabled path passed 24/24 known-answer checks and all 13 API checks. Debug
traces confirm actual anchor retention, including concurrent seed isolation.

Both arms use this same binary, checkpoint and depth-4 QMM candidate settings
recorded in `QWEN_NEXT_MTP_PARITY_PROGRESS.md`; only anchor retention changes.
Median decode tok/s over three 128-token trials per context, after an excluded
warmup (same M3 Ultra, thinking/prefix reuse off, seed 42):

| Sampling | Context | Disabled | Enabled | Change | Same response text? |
|---|---|---:|---:|---:|---|
| Greedy | 0.5K | 101.55 | 101.53 | -0.0% | Yes |
| Greedy | 1K | 94.62 | 95.05 | +0.5% | Yes |
| Greedy | 2K | 91.70 | 89.31 | -2.6% | Yes |
| Greedy | 4K | 71.22 | 71.63 | +0.6% | Yes |
| Temp 0.6, top-p 0.95 | 0.5K | 87.37 | 87.74 | +0.4% | Yes |
| Temp 0.6, top-p 0.95 | 1K | 85.24 | 84.37 | -1.0% | No |
| Temp 0.6, top-p 0.95 | 2K | 73.98 | 84.62 | +14.4% | No |
| Temp 0.6, top-p 0.95 | 4K | 91.94 | 92.06 | +0.1% | Yes |
| Temp 0.6, top-p 1 | 0.5K | 101.69 | 101.48 | -0.2% | No |
| Temp 0.6, top-p 1 | 1K | 76.41 | 81.16 | +6.2% | No |
| Temp 0.6, top-p 1 | 2K | 82.72 | 83.56 | +1.0% | Yes |
| Temp 0.6, top-p 1 | 4K | 86.39 | 85.63 | -0.9% | Yes |

This is not a broad speedup or a parity claim. The largest improvements occur
where generated text changes, and therefore include changes to speculative
acceptance/workload. The experiment remains disabled by default. It is not
justified to promote it from these bounded quality checks or claim that all
the improvement comes from eliminating the redundant head row.
