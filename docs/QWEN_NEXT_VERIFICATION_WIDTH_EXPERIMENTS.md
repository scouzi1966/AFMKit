# Qwen Next vocabulary and verification-width experiments

September 13, 2026. Continuation of AFMKit [PR #123](https://github.com/scouzi1966/AFMKit/pull/123).
Experimental paired-development code, not a default or release recommendation.

Activation recipes, effective defaults and the distinction between separately
measured versus untested combined options are maintained in the
[central opt-in matrix](QWEN_NEXT_OPT_IN_MATRIX.md).

## Method and scope

Use the exact `/Volumes/edata2/models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit`
checkpoint on the M3 Ultra 512-GiB machine. The frozen workload has 15 concurrent
coding-review requests, prefix caching enabled, MTP depth 3, temperature 0,
top-p 1, seed 42 and a 512-token output budget. An excluded warmup precedes
15 distinct requests and 15 exact-prompt repeats. Prompts are about 1.14K tokens.
The first phase is not entirely cold: it can reuse the warmup's shared prefix.
Repeated answers are generated again; they are not cached response strings.

Both arms enable the prior private-attention adapter and shared submission
ladder 4, with the compiled shared tail disabled. Timing runs have no profiling.
Only one build, test or inference owner runs at once. The frozen harness records
per-request SSE, usage, timings, memory samples, shutdown status and a 1-Hz
competing-process guard. Matched pairs use the same actual AFM binary hash.

JSON field and request-identity checks are **not semantic quality judging**.
Report valid tasks/s as well as generated tokens/s. Different answers and
different token counts mean these are matched-workload, not equal-output,
comparisons. Neither experiment establishes parity with another engine.

## Shared vocabulary projection

`AFM_QWEN_MTP_SHARED_VOCAB=1` shares the final projection from hidden values to
vocabulary scores in eligible batched MTP groups. It is off when unset and
requires the opt-in shared scheduler. It has no effect on ordinary AR or other
model families. The projection is limited to 2–4 requests and 16 total token
rows; unsupported shapes use the existing independent projections. All-greedy
groups produce argmax IDs; mixed/sampled groups retain their own samplers,
seeds, invocation order and decisions. Cache adoption/rollback is unchanged.

With 248,320 vocabulary entries, 16 BF16 rows represent about 7.6 MiB of logical
logits, excluding other intermediates, temporary copies and allocator caching.
This is a projection bound, not a process-memory guarantee.

Release binary SHA-256 for both opposite-order pairs:
`9fcec24ca1dd7d5a2c13a96a0b3dc8118fa4a3e86f13c0fef8fa2d0928b1edbb`.

| Pair / metric | Independent vocabulary | Shared vocabulary | Change |
|---|---:|---:|---:|
| A first-phase aggregate tok/s | 67.96 | 69.39 | +2.10% |
| A repeat aggregate tok/s | 106.88 | 108.72 | +1.72% |
| A repeat valid tasks/s | 0.6200 | 0.5993 | −3.34% |
| A first / repeat structural passes | 14/15; 15/15 | 14/15; 14/15 | One additional omission |
| A peak process RSS GiB | 69.838 | 69.714 | −0.124 GiB |
| B first-phase aggregate tok/s | 67.84 | 68.18 | +0.51% |
| B repeat aggregate tok/s | 107.76 | 110.68 | +2.71% |
| B repeat valid tasks/s | 0.6081 | 0.6238 | +2.59% |
| B first / repeat structural passes | 14/15; 14/15 | 14/15; 14/15 | Same counts |
| B peak process RSS GiB | 69.644 | 69.627 | −0.017 GiB |

All 120 requests completed. Pair A had 12/30 exact text-and-token-count matches;
pair B had 14/30. The additional A failure was `warm-repeat/AGENT-13`, a normal
108-token completion omitting the required `fix` field. The control in pair B
also omitted that field, but this does not prove the candidate harmless or
establish the cause. Useful-task throughput is mixed, so the modest token-rate
signal does not justify changing defaults. Vocabulary sharing is disabled in
the separate larger-group experiment.

Focused tests cover pure FP32 and Q4/BF16 projections, greedy and sampled
sessions, mixed greedy/sampled groups, cancellation after shared submission,
and strict/unsupported-shape fallback. The initial focused suite passed 85
tests with two optional probes skipped; the additional mixed-sampling rerun
passed separately.

## Expanded request groups

Merely increasing the scheduler clamp would abandon the existing fused HC
and shared submission path. The prototype instead extends their bounded
geometry together. The existing `AFM_QWEN_MTP_SUBMISSION_WINDOW=8` becomes an
explicit larger-group request **only with shared verification and independent
attention enabled**. The default owner window and the existing four-row recipe
remain unchanged. Equal-position mode remains capped at four.

New groups support 5–8 requests, at most three drafts / four verification tokens
each, at most 32 request/token rows. Existing 2–4-request width limits are
unchanged. Deeper speculative sessions retain the previous smaller grouping.
The public preparation helper and cache builder default to at most four rows;
the owner must explicitly supply the larger bound. HC's larger kernel bound
is passed only for the qualified batched geometry; it does not widen ordinary
AR or prefill execution. PLE submission barriers remain in place.

```text
request-owned drafts and real-position attention histories
                       │
              bounded group of 5–8
                       │
       shared HC / GDN / PLE / expert verification
       (HC and submission ladder remain eligible)
                       │
       independent vocabulary projection and sampling
                       │
       independent acceptance, rollback and next token
```

### Numerical oracle finding

The first new Q4/BF16 test incorrectly assumed that an independent-row
verification forward was an exact arithmetic oracle for a shared forward.
The diagnostic also failed at the existing four-request width, with hidden
differences around 0.02–0.03, and remained different with custom QMM disabled.
Eight-row diagnostics had outliers up to 1.14 in these randomly initialized
tiny-model tests. These differences must not be described as cache corruption,
nor dismissed as harmless numerical noise: their semantic effect is unqualified.
The experiment already uses the explicitly non-singleton-equivalent batched
policy; it does not redefine strict verification.

The original independent Float32 row oracle retains its 0.002 tolerance.
The new quantized isolation oracle instead compares identical batched
arithmetic with submission scheduling off/on, requiring **exact values** for
hidden state, stream state, committed cache arrays and subsequent logits.
It reverses commit order and checks untouched rows after each commit, across
five to eight requests and all four acceptance frontiers. Separate production-
geometry HC tests require exact equality to independent HC rows through 32
token rows. This separates scheduling/state correctness from arithmetic quality
without widening tolerances. Failed diagnostics and their source patch are
retained with the raw evidence.

### Same-checkpoint results

Both opposite-order pairs used Release binary SHA-256
`777a3a8708dc8a650e1fe9773caa2aa6fa42030be4f47e8aac486cb47469ed3d`.
Consumer Release rebuild: 99.15 seconds. Vocabulary sharing stayed off.

| Pair / metric | Window 4 | Window 8 | Change |
|---|---:|---:|---:|
| A first-phase aggregate tok/s | 68.48 | 72.06 | +5.24% |
| A repeat aggregate tok/s | 108.45 | 112.81 | +4.02% |
| A first-phase valid tasks/s | 0.3816 | 0.4175 | +9.40% |
| A repeat valid tasks/s | 0.5970 | 0.6419 | +7.52% |
| A first / repeat structural passes | 14/15; 14/15 | 15/15; 15/15 | Two omissions absent |
| A mean shared group size | 3.450 | 5.271 | +52.8% |
| A peak process RSS GiB | 69.716 | 69.818 | +0.102 GiB |
| B first-phase aggregate tok/s | 68.81 | 73.21 | +6.40% |
| B repeat aggregate tok/s | 106.38 | 115.67 | +8.73% |
| B first-phase valid tasks/s | 0.3974 | 0.4182 | +5.23% |
| B repeat valid tasks/s | 0.6111 | 0.6503 | +6.41% |
| B first / repeat structural passes | 15/15; 15/15 | 15/15; 15/15 | Same counts |
| B mean shared group size | 3.499 | 5.310 | +51.8% |
| B peak process RSS GiB | 69.851 | 69.767 | −0.084 GiB |

All 120 timing requests completed with clean shutdown and no sampled workload
collision. The candidate had 60/60 structural passes, versus 58/60 in controls.
Only 2/30 and 3/30 paired responses matched both text and token count.
Repeat generated-token totals increased from 2,543 to 2,636 in A and 2,611 to
2,668 in B. Repeat wall times changed from 23.449 to 23.367 seconds in A and
24.544 to 23.065 seconds in B. The aggregate gains therefore cannot all be
interpreted as a speedup on identical output. Median repeat latency improved
21.667→21.282 seconds and 22.993→20.623 seconds; median TTFT stayed near 0.1 s.

Retain the bounded window extension as an **opt-in experiment**, without
changing the four-row recipe or default owner policy. Both raw and useful-task
rates improved here, but longer contexts, heterogeneous arrival/length
distributions, tail fairness, other models and semantic quality remain
unqualified. No eight-way sampled-performance claim is made from greedy timing.

The focused Release suite passed 88 tests, with two optional probes skipped.
The final controls-disabled rerun also passed 88 tests with the same two skips.
The expanded mapped-PLE test passed separately after adding the quantized
state-isolation coverage. Production-geometry HC equality also passed with
native chaining disabled, covering the direct Metal path as well as the native
chain used in the suite. These are focused tests, not a full release suite.
Live lifecycle measurements are recorded separately from timed throughput.

Two C6 API lifecycle runs passed **120/120 assertions each**: window 8 with
vocabulary sharing off, then window 4 with vocabulary sharing on. They cover
greedy/sampled handling, logprobs fallback, stops, streaming, cancellation,
subsequent requests and replay. The window-8 run recorded 12 shared groups /
29 rows; the vocabulary run recorded 9 groups / 21 rows. These counters do not
claim that every lifecycle group contained eight requests. Expanded execution
coverage comes from the C15 timing groups and explicit eight-session tests.
The lifecycle harness has the preflight owner check but does not run the timing
wrapper's 1-Hz guard; no throughput claim is taken from these runs.

## Evidence location

External, untracked root:
`/Volumes/edata2/afm-benchmarks/qwen-next-mtp-parity-20260909`.

- `shared-vocab-{control,on}-20260913-{a,b}-*`: raw timing arms and identity records.
- `shared-vocab-comparison-20260913-{a,b}.json`: matched comparisons.
- `shared-vocab-20260913-prototype.patch`: the exact vocabulary-only source diff.
- `expanded-verification-20260913-independent-oracle.patch`: larger-group source
  and the failed independent-row numerical oracle, before test separation.
- `expanded-verification-20260913-oracle-*.log`: preserved diagnostics, including QMM off.
- `expanded-verification-20260913-tests*.log`: focused tests, including initial failures.
- `expanded-verification-20260913-mapped-ple.log` and
  `expanded-verification-20260913-hc-no-chain.log`: additional focused checks.
- `verify-window{4,8}-20260913-{a,b}-*`: raw timing arms, binary identities and launch records.
- `verify-window-comparison-20260913-{a,b}.json`: larger-group comparisons.
- `verify-window8-20260913-lifecycle-*` and `shared-vocab-20260913-lifecycle-*`:
  the two separate API lifecycle runs.
- `run_shared_vocab_screen.py`, `compare_shared_vocab.py`,
  `run_verification_window_screen.py`, `compare_verification_windows.py`:
  new wrappers; older frozen runners and prior hash manifests are unchanged.
- `VERIFICATION-WIDTH-20260913-README.md` and
  `VERIFICATION-WIDTH-20260913-SHA256SUMS.txt`: consolidated inventory and hashes.

No installed binary, consumer dependency, main branch, release or global
default is changed by this workstream.
