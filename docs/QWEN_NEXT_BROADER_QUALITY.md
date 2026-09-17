# Qwen Next: independent semantic and combined-mode gate

September 16, 2026. Follow-up to the default-off GDN Q/K normalization experiment
on PR #123. This report preserves the earlier five-task quality evidence rather
than replacing it with a different test set. No inference code or defaults
changed during this gate.

## Why another screen

The original sampled screen checks unique-key JSON and file identity, not
whether the proposed repair is correct. Its 38/50 to 43/50 improvement is a
useful local signal, not proof of general answer quality. The fused candidate
preserves that prototype exactly and has no demonstrated material warm speed
advantage. Promotion requires more than repeating those five tasks.

`Scripts/qwen-next-broader-quality.py` adds 15 independent, fixed-answer service
review scenarios. These cover cache identity, LRU, cancellation isolation,
memory admission, causal masks, validation, split stop markers, dependency
ordering, retries, context budgets, prefix matching, quantized storage, lease
expiry, duplicate keys and an authorization repair with an untrusted comment.
Answers, request IDs and evidence-record IDs are checked; valid JSON alone
does not pass. No generated code is executed and no external AI judge is used.
This is a narrow semantic-constraint screen, not open-ended semantic evaluation.

The workload contains all 15 records in a common packet, then a per-request
question. Actual prompt counts are approximately 1K, not the previous 4K
file-selection workload. Exact payloads and prompt-token counts match between
AFM and the reference. A cancellation question is **not** a runtime cancellation
test. Similarly, cache questions do not themselves prove actual cache reuse.

## Controls and provenance

- Exact unchanged ddalcu checkpoint, template and weight-index hashes from the
  previous fixed baseline; mapped n-gram sidecar.
- Default-off executable SHA-256:
  `e7970fd1f444e13f23fc8f589b7dad29723216ea803697d7b02d71c4f85d7f91`.
- Private fused executable SHA-256:
  `f59804bc41db85f57366aa29d5c5a27d2a10a12d71bd820a7f0ad307d2fe8a3c`.
  Both use provider `2c9c8241`; the private build enables only the internal
  normalization diagnostic. Source activation was never committed.
- Preserved reference executable SHA-256:
  `f3ce20fba143e908110d44d1b8304bf305962a4600d23c9c112934112514bbe4`.
  Its frozen launch disables speculation other than MTP, attention quantization,
  prefix/tokenization caches and uses top-k 0. It was run live on these new tasks.
- Fifteen greedy controls plus 30 samples (two fixed distinct seed sets),
  temperature 0/0.6, top-p 1, 512-token cap, thinking off. One unrelated tiny
  warmup per process is excluded. No grammar constraints or forced parser.
- AFM retains the saved M25 opt-ins, MTP depth 3/batched verification. These
  are **not** unset-environment measurements. Same seeds do not promise equal
  RNG consumption between ordinary, speculative or concurrent execution.
- Single GPU owner, no builds during inference, 160 GiB pre-load and 100 GiB
  runtime available-memory floors, one-second process RSS sampling. RSS does
  not account for all Metal allocations; this is not a leak soak.

## C1, prefix off: completed

All six configurations complete 45/45 requests, with 45/45 structural and
request-identity checks. All outputs terminate normally, without token-cap
finishes or unexpected reasoning.

| Semantic passes | Default | Fused candidate | Reference |
|---|---:|---:|---:|
| MTP off, greedy | 13/15 | 12/15 | 12/15 |
| MTP off, sampled | 23/30 | 23/30 | 22/30 |
| MTP on, greedy | 12/15 | 12/15 | 12/15 |
| MTP on, sampled | 22/30 | 23/30 | 23/30 |

The candidate loses the greedy LRU answer in both modes. With MTP off there
are no sampled pass/fail changes. With MTP on it gains a greedy prefix-match
answer, and gains two sampled dependency-order answers while losing one LRU
answer. Thus equal totals do not mean identical failures. These small counts
do not establish statistical quality equivalence or superiority.

The reference reproduces the greedy memory-budget and quantized-storage
mistakes; those observed failures are not AFM-only. This does not attribute
every failure to model behavior or exonerate all runtime arithmetic. Other
failures differ by engine, mode and seed.

Sampled timing, including all cases whether right or wrong:

| Measure | Default AR | Fused AR | Reference AR | Default MTP | Fused MTP | Reference MTP |
|---|---:|---:|---:|---:|---:|---:|
| Median decode tok/s | 68.80 | 68.45 | 69.72 | 128.88 | 127.35 | 130.17 |
| Output / phase-wall tok/s | 34.33 | 34.16 | 34.59 | 47.69 | 47.08 | 45.47 |
| Correct tasks/s | 0.3932 | 0.4014 | 0.3916 | 0.5217 | 0.5522 | 0.5352 |

These are short, structured answers around 1K context, not new Context-curve
peaks. MTP accelerates this workload but does not guarantee better answers.
Phase-wall rates include prefill and harness bookkeeping. The initial default
greedy arm includes a 14.52-second first full-packet request after a tiny
warmup; do not present its difference from later greedy arms as a speedup.
The sampled phases are warmed, but this remains one ordered comparison,
not a counterbalanced repeated performance bound.

**Decision:** keep the fused normalization default-off. The local five-task
gain has not generalized convincingly, and there is an explicit greedy
regression. Retain the implementation and evidence as an experiment; do not
roll back earlier qualified optimizations or overwrite their peak ledger.

## C15, prefix off: completed

Both binaries complete first and exact-repeat workloads with MTP off/on:
360/360 runtime and request-identity checks pass. Structure passes 356/360;
the four exceptions are the same sampled dependency-order case returning an
array where an answer object was requested, in both AR binaries and phases.
All MTP responses pass structure. Semantic correctness remains separate.

| Sampled measure | Default AR | Fused AR | Default MTP | Fused MTP |
|---|---:|---:|---:|---:|
| First semantic passes | 24/30 | 23/30 | 22/30 | 21/30 |
| Repeat semantic passes | 23/30 | 23/30 | 23/30 | 21/30 |
| First aggregate output tok/s | 35.51 | 35.19 | 53.62 | 52.60 |
| Repeat aggregate output tok/s | 35.17 | 34.91 | 53.57 | 52.62 |
| Repeat correct tasks/s | 0.4353 | 0.4428 | 0.6145 | 0.5626 |

Greedy first/repeat scores are default AR 13/15 and 13/15; fused AR 12/15
and 12/15; default MTP 13/15 and 13/15; fused MTP 13/15 and 12/15. In the
sampled MTP repeat the candidate loses the LRU and split-stop answers with no
compensating gains. The candidate does not improve combined-mode quality.

Every phase reaches 15 overlapping client requests. This is not proof that
all 15 occupied one GPU batch: the server reports compatible subgroups and
native request-cache fallback for nonuniform offsets. No tokens were restored
from prefix cache, as required in this cache-off screen. All four servers exit
normally. Peak process RSS is 69.22 GiB; minimum available memory is 359.74 GiB.

## C1, prefix on: coverage correction and replay result

The first attempt, `broader-cache-c1-a`, completed the default AR arm and part
of the candidate but restored **zero tokens**, even on exact repeats. It was
intentionally stopped, with its partial data preserved. This is not a model
crash and does not qualify the cache path. Logs show `exact-replay-bypass`:
the hybrid recurrent state cannot simply be rewound from an end-of-prompt
snapshot. The existing experimental serial replay-boundary option was absent
from M25; prefix enablement alone did not provide a usable boundary here.

A separately labelled run, `broader-serial-replay-a`, adds only the existing
`AFM_PREFIX_REPLAY_BOUNDARIES=1` option to AR/C1 with prefix caching enabled.
No inference code, default, or memory budget was changed. Both binaries
complete 90/90 requests; runtime, structure and identity all pass. Greedy
scores are 12/15 and sampled scores 23/30 in each phase and binary, although
individual greedy failures differ between binaries.

| Sampled measure | Default AR | Fused AR |
|---|---:|---:|
| Repeat semantic passes | 23/30 | 23/30 |
| Repeat restored prompt tokens | 30,132 | 30,132 |
| Repeat output / phase-wall tok/s | 66.03 | 65.76 |
| Repeat correct tasks/s | 0.7316 | 0.7371 |
| Repeat median TTFT, seconds | 0.0364 | 0.0369 |
| Repeat median decode tok/s | 68.95 | 68.82 |

This is an actual benefit from **existing replay caching**, not a speedup from
the new normalization. First requests also share a 768-token boundary; exact
repeats restore almost the full prompt. The default greedy score drops from
13/15 without replay to 12/15 with replay: splitting prefill changes the
numerical path, so unchanged quality cannot be assumed. The sampled pass/fail
set is unchanged between the two replay binaries. Peak RSS is 67.52 GiB and
minimum available memory 363.06 GiB. Both servers exit normally.

## C15, prefix on: capacity result and bounded-window follow-up

In `broader-cache-c15-a`, both AR arms complete 90 requests, with real prefix
hits: sampled repeats reach 75.13/74.78 aggregate tok/s (default/fused), with
23/30 semantic passes in each. The fused arm has one structural exception
per sampled phase (dependency-order array instead of answer object); default
AR has none. The concurrent AR scheduler already captures replay boundaries;
the serial-only option above is not added to this profile.

The MTP candidate then fails the explicit cache-coverage requirement, **not**
runtime correctness: first greedy, first sampled and repeat greedy all finish,
but repeat greedy restores zero tokens. `ExactPromptReplayCache` retains at
most 16 entries; this workload cycles through 45 before repeating. Server logs
confirm 16 retained entries at about 2.33 GB, below the unchanged 4 GiB byte
budget. The working set exceeds the entry cap and evicts the earlier prompts.
The harness stops before repeat sampled and the default MTP arm. All three
servers exit 0; this is an incomplete paired MTP cache run, not a green gate.
The explicit partial audit checks only the two completed AR arms.

The separately labelled `broader-cache-window15-c15-a` retains all 45 questions,
payloads and seeds, but repeats each 15-prompt window immediately. No entry or
memory limits change. Sampled aggregate rates use the summed active durations
of two 15-request windows, excluding the intervening other-phase window; do not
treat these as the original rolling 30-request phase. This tests bounded reuse,
not a solution to the 45-prompt retention limit. The paired run is complete:
360/360 runtime and request-identity checks pass, and 358/360 structural checks
pass. Both structural failures are the candidate AR dependency-order case,
first and repeat. No output hits its token cap.

| Windowed sampled measure | Default AR | Fused AR | Default MTP | Fused MTP |
|---|---:|---:|---:|---:|
| First semantic passes | 23/30 | 23/30 | 21/30 | 21/30 |
| Repeat semantic passes | 23/30 | 23/30 | 21/30 | 21/30 |
| First aggregate output tok/s | 58.66 | 55.96 | 53.66 | 52.24 |
| Repeat aggregate output tok/s | 82.71 | 80.39 | 184.18 | 178.11 |
| Repeat correct tasks/s | 0.9806 | 1.0087 | 1.9110 | 1.9045 |
| Repeat median TTFT, seconds | 0.2229 | 0.2231 | 0.0902 | 0.0909 |
| Repeat restored prompt tokens | 30,132 | 30,132 | 30,162 | 30,162 |

All 45 repeated prompts hit in every arm, with 15 overlapping clients per
window. All first/repeat texts match except the candidate MTP greedy LRU
answer, which changes from correct to incorrect (13/15 to 12/15 greedy passes).
Default MTP stays 13/15. The candidate's sampled total hides two improvements
and two regressions versus default; it is not stronger quality overall.

The existing MTP replay mechanism provides a large bounded-repeat benefit;
the experimental normalization does not. The different output lengths also
matter: default/candidate MTP emit 2,024/1,964 sampled tokens, respectively.
Correct tasks/s is nearly equal, so the token-rate difference alone is not
evidence of a 3.3% implementation slowdown. These are short approximately 1K
semantic tasks, not a replacement for the historical 128-output Context curve.
Peak RSS is 69.22 GiB, minimum available memory 356.48 GiB; guards remain clear
and all servers exit normally. There is no matched concurrent reference run
in this new screen, so no concurrent parity claim follows.

## C1 MTP with the prefix flag

The current serial Qwen MTP branch returns zero restored tokens; its replay
cache is owned by the concurrent scheduler (`maxConcurrent > 1`). The serial
GLM replay code does not apply to Qwen. The separate flag-on first/repeat screen
`broader-mtp-cache-flag-c1-a` completes 180/180 runtime, structure and identity
checks. Each binary reproduces all 45 cache-off answers exactly; first and
repeat also agree. Greedy scores are 12/15 each, sampled default/fused 22/30
and 23/30, in both phases. All requests restore zero tokens, as expected from
source inspection. Repeat sampled phase-wall rates are 47.98/47.29 tok/s.
This qualifies flag-on runtime behavior, **not** serial Qwen MTP reuse.

## Scope of combined evidence

The same M25 controls are retained to isolate the normalization change. In
particular, this is **not** the separately documented fastest AR request-banked
recipe, which enables additional mixed-position/GDN/banking controls. Do not
interpret this matrix as an AR maximum-throughput search. The candidate variant
only applies to B1 prefill of at least 128 tokens; short cached suffixes and
batched prefill can take the unchanged arithmetic path.

Runtime cancellation, longer-context semantic tasks, broader open-ended quality,
and full release regressions remain separate gates.

### Decision after combined screens

Retain the historical decode, shared verification and bounded replay work and
its evidence. Do **not** promote fused GDN normalization: it has no demonstrated
material performance win, does not consistently improve the independent tasks,
and introduces specific regressions. Further production-facing work on this
candidate is not justified by the current benefit relative to requalification
cost. Keep its default-off diagnostic code and frozen binaries for attribution,
not as a recommended user setting. Issue #125 remains open; matching a tensor
more closely does not by itself establish better end-to-end model quality.

No main branch, installed binary, public default or memory limit changed. The
next release-facing work is qualification of retained controls, not a new
automatic policy or another unproven normalization change. Runtime cancellation
and serial Qwen MTP replay limitations must remain explicit in that work.

Across the six completed screens: 1,260/1,260 AFM requests pass runtime and
identity, and 1,254/1,260 pass structure; the 90/90 reference requests pass all
three. Do not combine semantic scores across repeated/dependent workloads into
a general quality percentage. Incomplete attempts and the completed AR portion
of the capacity-miss run are preserved separately, not silently counted as
completed paired configurations. These are controlled model screens, not the
full release regression suite.

## Evidence

External, append-only root:

```text
/Volumes/edata2/afm-benchmarks/qwen-next-mtp-parity-20260909/consolidation-20260915
```

`BROADER-QUALITY-PROTOCOL.md` was recorded before inference. `broader-c1-a`,
`broader-reference-c1-a`, `broader-c15-a`, and `broader-serial-replay-a` preserve
launches, full fixtures, requests, raw SSE, scores, timing and guards. The
matching `AUDIT-BROADER-*.json` files recheck scores and hashes. The interrupted
`broader-cache-c1-a` is explicitly not a completed paired run. The same applies
to `broader-cache-c15-a`; its partial audit covers completed AR arms only.
`broader-cache-window15-c15-a` has a complete paired audit including window
ordering, cache-hit coverage and aggregate denominators. The final serial MTP
flag screen is audited in `AUDIT-BROADER-MTP-CACHE-FLAG-C1.json`.
Preserved runners v1/v2/v3/v4 permit audits after the harness evolves. Fifteen CPU
tests cover scorer failures, fixture identity, exact launch transformations,
aggregate denominators, window ordering and actual client-overlap accounting.
All 36 Qwen harness/ledger CPU tests pass when the required external ledger-test
directory is supplied. An initial broader discovery invocation omitted that
directory and had eight setup errors; those were not model or runtime failures.
