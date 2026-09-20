# Qwen Next composed-verifier experiments — 2026-09-13

Two controlled follow-ups to the [verification-width experiment](QWEN_NEXT_VERIFICATION_WIDTH_EXPERIMENTS.md)
used the existing Release runtime, without changing kernels or defaults:

1. Window 8 with vocabulary sharing off/on: small repeat-token gains, with
   additional structural omissions in one pair. **Not a recommended preset.**
2. Window 8 with shared submission every four/two layers, vocabulary off:
   repeat-token gains reproduced in both orders, but first-phase token rate
   decreased slightly. **Separate experimental profile M15/W8-L2**, not a
   replacement for W8 or a universal performance improvement.

The [central opt-in matrix](QWEN_NEXT_OPT_IN_MATRIX.md) contains activation,
rollback, defaults and untested combinations. This remains part of
[PR #123](https://github.com/scouzi1966/AFMKit/pull/123), not a new release.

## Method and identity

- M3 Ultra, 512 GiB RAM, 80 GPU cores; exact checkpoint:
  `/Volumes/edata2/models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit`.
- Provider runtime `a1255dc6`, documentation checkpoint `cc290541`; consumer
  `9acccfc9`. No runtime code changed during these experiments.
- Binary:
  `/Volumes/edata2/dev/CODEX/maclocal-api-qwen-next-mtp-parity/.build/release/afm`.
  SHA-256: `777a3a8708dc8a650e1fe9773caa2aa6fa42030be4f47e8aac486cb47469ed3d`.
- Fifteen simultaneous agentic-style coding reviews, prefix caching on,
  MTP depth 3, batched verification, independent attention, window 8.
  Compiled shared tail and PLE decoded-row cache off. Temperature 0, top-p 1,
  seed 42, maximum 512 output tokens, thinking off; identical frozen prompts.
- After one excluded warmup, fifteen distinct requests followed by exact
  repeats. “First” is not a completely cold cache: each arm reports 1,144
  cached tokens in that phase and 17,127 in repeats. Answers are regenerated.
- Each experiment ran control→candidate and then candidate→control. All
  eight timing arms completed 30/30 requests: **240/240 runtime completions**.
  No concurrent GPU owner or build was observed by the timing guards.
- Aggregate tokens/s includes queueing and prefill. Structural pass requires
  JSON, correct request/file identity and diagnosis/fix/test fields; it is
  **not a semantic judge score**. There was no Codex-judge run in this screen.

## Window 8 plus vocabulary sharing

Only `AFM_QWEN_MTP_SHARED_VOCAB` changes, from `0` to `1`; shared ladder stays 4.

| Pair / execution order | First tok/s, off → on | Repeat tok/s, off → on | Repeat change | Structural passes, off → on |
|---|---:|---:|---:|---:|
| A: off, on | 72.85 → 71.42 | 115.90 → 120.06 | +3.59% | 30/30 → 28/30 |
| B: on, off | 71.79 → 71.44 | 116.56 → 117.68 | +0.95% | 29/30 → 29/30 |

First-phase token throughput changed −1.96% / −0.49%. Repeat structurally
valid tasks/s changed +3.82% / +1.33%; first-phase valid tasks/s changed
−5.10% / +0.23%. Across both pairs, controls passed **59/60** structural
checks; vocabulary-on passed **57/60**. Exact response-text/token-count matches
were only 6/30 and 4/30.

The additional failures in pair A are AGENT-13 missing `fix` in both phases.
Both responses ended normally at 108/107 output tokens, well below 512; these
are not truncations. The reverse pair's control also omitted this field once.
That variation does not prove the extra failures harmless or model-only.

Vocabulary sharing remains eligible only for 2–4-request groups / at most
16 token rows. A window of eight does **not** provide eight-way shared
vocabulary projection: larger groups use independent projection. Startup
confirmed the mode and shutdown confirmed real shared-verifier groups, but
there is no vocabulary-kernel invocation counter. Do not infer that every
group used the vocabulary optimization.

Disposition: the previously untested combination is now measured, with mixed
quality and small performance evidence. Keep vocabulary **off** in the W8
and W8-L2 recipes. The flag remains available as an experiment.

## Window 8, submission every two versus four layers

Only `AFM_QWEN_VERIFY_SHARED_ASYNC_LADDER` changes, from `4` to `2`; vocabulary
stays off. This submits the eligible shared graph more frequently. The
hypothesis was less GPU waiting for host graph construction, balanced against
extra submission overhead; these timings alone do not establish its cause.

| Pair / execution order | First tok/s, 4 → 2 | Repeat tok/s, 4 → 2 | Repeat change | Structural passes, 4 → 2 |
|---|---:|---:|---:|---:|
| A: 4, 2 | 73.44 → 72.37 | 113.12 → 119.21 | +5.39% | 30/30 → 30/30 |
| B: 2, 4 | 72.98 → 72.18 | 114.76 → 117.46 | +2.35% | 29/30 → 30/30 |

First-phase token rate changed **−1.45% / −1.10%**. Repeat structurally valid
tasks/s changed +3.37% / +7.39%; first-phase valid tasks/s changed +0.55% /
−1.51%. Controls passed **59/60** structural checks and ladder 2 passed
**60/60**, with no newly failing structural case. Exact paired text/count
matches were only 3/30 and 6/30.

The second pair is an important caution: repeat wall time barely changed,
22.691 → 22.638 seconds (−0.23%), while output tokens increased 2,604 → 2,659.
Most of its token-rate gain is longer output, and its +7.39% valid-tasks gain
includes one recovered structural pass. The first pair reduced repeat wall
time 22.614 → 21.877 seconds (−3.26%), with all fifteen structures passing in
both arms. Do not describe both pairs as an equivalent acceleration of work.

Repeat median TTFT remained about 0.08–0.11 seconds. Median full-response
latency changed 20.445 → 20.064 seconds in pair A, but 20.054 → 20.535 seconds
in pair B. This is not an across-the-board latency win.

Disposition: retain a separately documented opt-in delta **W8-L2**, with the
first-phase and output-length tradeoffs visible. Keep the existing W8 recipe
at ladder 4. No default change or parity claim follows from two pairs.

## Memory, correctness and validation

Peak process RSS across all eight arms was **69.54–69.84 GiB**, with no material
memory change. This is not a complete Metal-memory measurement or a minimum
hardware requirement. Mean shared-verifier group width was **5.15–5.31**;
the larger-group execution was exercised, not inferred solely from a flag.

Manual inspection found coherent but sometimes unsafe or incomplete advice
in both arms: some streaming fixes suggest dropping a chunk whenever it has
`finish_reason`, which can discard legitimate final content. Passing the JSON
checks does not establish correct code-review reasoning. All raw responses
and changed-text comparisons are preserved for broader quality qualification.

Separate C6 lifecycle runs covered W8-L2/vocabulary-off and W8/vocabulary-on.
Each passed **120/120 assertions**: cancellation 36/36, subsequent requests
42/42, repeated prefixes 42/42. They mix greedy and sampled requests
(temperature 0.6/top-p 0.95), logprobs fallback, stop handling and token limits.
Both servers exited 0. Shared groups/rows were 12/29 and 10/22 respectively;
these small lifecycle groups do not establish eight-way coverage. Lifecycle
uses the preflight ownership check, not the timing wrapper's 1-Hz guard, and
is not a throughput measurement.

Test-only source changes add stride-2 coverage to mapped PLE flushes,
mixed-position verification, quantized B5–B8 same-geometry exact-value
oracles, rollback/cache isolation and dispatch/fallback guards. A new test
checks expanded groups with vocabulary enabled, including its larger-group
fallback, mixed sampling and cancellation. Independent Float32 numerical
oracles retain their existing tolerances; no tolerance was relaxed.

Focused Release tests used `Scripts/swiftpm-reliable.sh`, covering MTP
pipelines, batched projections, admission and exact-prompt replay:

| Configuration | Passed | Skipped | Failed | XCTest time |
|---|---:|---:|---:|---:|
| Tuned verification / shared ladder 2 | 89 | 2 | 0 | 77.41 s |
| Inherited tuning controls unset, same test build | 89 | 2 | 0 | 80.57 s |

The two skips are explicit optional production-shape/latency probes, not
failed tests. The incremental Release test build took 41.38 seconds. These
are focused tests, not a newly rerun full release suite. The application
binary was not rebuilt because its runtime sources did not change.

## Evidence and reproducibility

External, untracked root:
`/Volumes/edata2/afm-benchmarks/qwen-next-mtp-parity-20260909`.

- `combined-w8-{control,vocab}-20260913-{a,b}-*`: vocabulary timing arms.
- `w8-ladder{4,2}-20260913-{a,b}-*`: submission interval timing arms.
- `combined-w8-vocab-comparison-20260913-{a,b}.json` and
  `w8-ladder-comparison-20260913-{a,b}.json`: matched parameters/prompts,
  throughput, latency, memory, structural totals and changed texts.
- `w8-ladder2-lifecycle-20260913-*` and
  `combined-w8-vocab-lifecycle-20260913-*`: separate lifecycle evidence.
- `run_combined_verifier_screen.py`, `compare_combined_verifier.py`:
  new wrappers around unchanged frozen harnesses.
- `composed-verifier-20260913{,-default}-tests.log`: focused test output.
- `COMPOSED-VERIFIER-20260913-README.md` and
  `COMPOSED-VERIFIER-20260913-SHA256SUMS.txt`: new curated inventory.

The authoritative effective command is each arm's `launch.json` argv; the new
`*-shared-vocab.json` records the final window, vocabulary, ladder and actual
AFM hash. Older nested wrapper metadata can describe an intermediate window
4 / ladder 4 even after the outer override. In particular, `launch.json`'s
legacy `binary_sha256` hashes `/usr/bin/env`; **use the actual AFM hash in the
new metadata instead**. Identical duplicate legacy assignments in lifecycle
argv do not change their values. The three new experimental assignments are
unique and are checked in every comparison. Frozen metadata is not rewritten.

There is no new sampled-performance, longer-context, prefix-off, AR, other
model or full six-mode result in this screen. No installed binary, consumer
dependency, main branch, release, runtime source or default was changed.
