# Shared speculative work: drafting, adaptive depth, persistent state

Workstream: [AFMKit PR #123](https://github.com/scouzi1966/AFMKit/pull/123).
Date: 2026-09-13. These are three separate opt-in experiments. No production
defaults change, and no improvement is assumed from implementing a feature.
See [the central activation matrix](QWEN_NEXT_OPT_IN_MATRIX.md) for other
controls, prerequisites and historical results.

## Execution boundaries

```text
Serialized scheduler owner
  |
  +-- compatible ready requests (same model/head/proposal depth)
  |     |
  |     +-- shared draft transforms --> private per-request head attention
  |     |                            --> greedy draft proposals
  |     +-- shared target verification --> private attention, sampler and RNG
  |     +-- resolve each decision --> commit accepted target state per request
  |     +-- same repair width? --> shared head repair / singleton fallback
  |     +-- emit one token per request; honor its cancellation/EOS/limit
  |
  +-- request-local adaptive policy --> next compatible proposal depth
  +-- bounded fixed-state bank --> reuse unchanged rows / replace changed rows
  +-- completion/cancellation --> evict banks containing departed requests
```

The graph owner is still serialized. This is shared GPU work across requests,
not multiple threads mutating MLX caches. Target sampling remains request-owned.
The arithmetic schedule may change floating-point rounding and emitted wording;
the batched policy has never promised exact ordinary-decoding token equality.

## 1. Shared draft and repair head

`AFM_QWEN_MTP_SHARED_HEAD=1` requires the existing shared-verifier scheduler
and batched-policy Qwen sessions. Compatible groups contain 2–8 requests.
The MTP head shares normalization, projections, HyperConnections and MoE
transforms. Head attention retains separate QSA histories and explicit real
RoPE positions. It is not padded full attention over unrelated requests.

Draft proposals remain greedy, including for sampled target requests. Target
sampling and acceptance are unchanged. Repair batches group equal accepted
lengths and support at most four token positions. Retained-anchor experiments,
incompatible caches and singleton work use the existing independent path.

Zero-acceptance decisions need an explicit resolve-before-repair phase:
otherwise the first ordinary `nextToken()` repairs them immediately and they
never become eligible for shared repair. Resolving a decision does not draw
another sample or emit a token. A repaired-primary marker prevents generating
the next cycle before consuming the current primary.

Shutdown counters report **actual shared draft rows and repair rows**, not
merely the requested switch. Both must be positive in an execution screen.

## 2. Request-local adaptive proposal depth

`AFM_QWEN_MTP_ADAPTIVE_DEPTH=1` enables a reusable controller for batched-policy
sessions owned by this Qwen scheduler. It selects depths 1 through the requested
`--mtp-depth` (bounded at 8), with two initial observations per depth, periodic
probes, a smoothed useful-token/cost estimate and 5% selection hysteresis.

This first experiment does **not** switch MTP off automatically, select depth
zero, or exceed the requested maximum. Strict-policy sessions ignore it. Its
only action is selecting the amount of proposed work; it does not modify
temperature, top-p, RNG, target probabilities or the acceptance rule.

The measured interval covers draft construction through target verification
and commit, including host waits. Deferred head repair is **not included**.
This is not an isolated GPU-cost estimate. Requests selecting different depths
split compatibility groups, which can reduce aggregate sharing. Therefore
per-request acceptance improvements alone are insufficient: judge the policy
by aggregate throughput, task completion time, group width and output quality.
The shutdown depth-cycle histogram proves which depths were actually used.

## 3. Bounded persistent fixed-state bank

`AFM_QWEN_MTP_PERSISTENT_STATE_MIB=1024` enables a 1024-MiB payload budget.
Unset/zero disables it; the accepted range is clamped to 0–2048 MiB. It also
requires independent-attention shared verification. At most four banks are
retained. Budget is **retained array payload**, not peak process/Metal memory.

The generic bank holds fixed-size recurrent/PLE state arrays, ordered by stable
request UUIDs and guarded by state revisions. After a full accepted chain, that
row can retain the final batched state. Rejected or advanced rows are refreshed
from authoritative request state with functional scatter updates. The previous
bank and rollback snapshots must not be mutated. All-changed, reordered or
incompatible groups fall back to ordinary merging. Host n-gram mirrors are
still copied from authoritative per-request state.

64-bit integer token-history columns use row slice/concatenation instead:
MLX's GPU scatter does not support int64/uint64 payloads. These columns are
small; large floating-point recurrent arrays keep the selective-scatter path.
The first real state-cache arm exposed this unsupported operation after 24/30
completed requests. Its failed repeat phase is retained and is **not a valid
throughput result**. The fix preserves wide integer values without narrowing.

This is **not** the PLE decoded-row cache, full attention KV storage, a prefix
cache, response caching, or a globally shared recurrent state. It cannot reuse
a different request's state. Completion/cancellation prunes affected banks and
shutdown clears the owner. It reports reused rows, refreshed rows and retained
bytes. Zero reuse means the optimization was inactive on that workload, even
if the configuration was accepted.

## Reusable components and scope

The generic controller and bank live in the AFMKit-owned MLXLMCommon source
snapshot; the Qwen adapter lives in MLXLLM and the service integration in
AFMKitMLX's scheduler. These are tracked provider sources, not consumer SwiftPM
checkout edits. No model runtime is added to maclocal-api.

Other model architectures can reuse the policy and bounded-bank mechanism,
but must supply correct state revisions, acceptance/rollback ownership and
their own compatibility adapter. No other model is automatically opted in or
claimed faster by this change.

## Qualification and evidence

Final focused Release suite: **100 executed, 98 passed, 2 skipped, zero failures**.
Includes shared draft/repair execution, mixed greedy/sampled requests, different
positions, cancellation, generic policy bounds, budget/eviction and exact
partial-row refresh. A forced full-acceptance tiny model proves actual persistent
bank reuse with adaptive depth and shared heads together. Additional tests prove
exact nonzero-state equivalence against a rebuilt batch after partial rejection,
64-bit signed/unsigned history preservation and fail-closed shape/dtype checks.
It is not a production
model quality or performance benchmark.

Two earlier attempts are retained: a build-time existential identity error was
fixed; then an execution-coverage assertion detected the missing zero-acceptance
shared repair. The corrected suite passed without weakening the assertion.
The initial paired AFM Release build completed in 137.35 seconds. The final
suite took 85.47 seconds after its incremental test build. The two skipped tests
are explicitly gated production-shape latency/512-expert probes, not failed
correctness cases. Test logs A–E also retain intermediate build failures.

Live experiments use the unchanged checkpoint:

```text
/Volumes/edata2/models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit
```

Workload: M3 Ultra 512 GiB; C15, prefix on, maximum depth 3, W8-L2 verifier,
512-token cap, temperature 0, top-p 1, seed 42; fifteen synthetic agentic review
prompts then identical repeats. One excluded warmup. No profiling or competing
build/test/inference owner. Structural JSON/request-identity checks are not a
semantic judge. Samples include aggregate emitted tokens/s, useful tasks/s,
output lengths, latency, cached tokens and process RSS.

Raw evidence is external and untracked under:

```text
/Volumes/edata2/afm-benchmarks/qwen-next-mtp-parity-20260909/
```

Run labels begin `shared-work-20260913-`. Every arm records the actual executable
SHA-256 and runtime-source hashes; the development version alone is not identity.
Earlier frozen wrappers and benchmark artifacts remain unchanged. Intermediate
wrapper metadata can describe its original ladder/window: effective assignments
in `launch.json` and the outer `*-shared-work.json` are authoritative.

Initial pre-fix observations (binary `e4d914ee…`, not the final qualification):

| Option | First / repeat aggregate tok/s | Runtime / structural | Observation |
|---|---:|---|---|
| All three off | 41.51 / 103.34 | 30/30 / 30/30 | First control anomalously slow; do not use it alone to claim gains |
| Shared head | 70.56 / 114.70 | 30/30 / 30/30 | 1,876 shared draft rows; 1,432 repair rows; only 5/30 exact paired texts/counts |
| Adaptive depth | 66.31 / 105.11 | 30/30 / 30/30 | Depths 1/2/3 exercised; mean group 3.28 versus control 5.26 |
| Persistent state 1024 MiB | First 71.33; repeat invalid | 24/30 / 24/30 | Fatal unsupported int64 GPU scatter; remaining six transports failed |

Ten shared-head responses were manually spot-checked across all five task types
in both phases. They were coherent and addressed the supplied defects. This is
not a full semantic evaluation or evidence of exact numerical parity.

## Corrected-binary controlled results

Runtime commit `4d2cf327`, paired consumer `9acccfc`, actual executable SHA-256:
`a78772f6c111715d68d9f428926d6ea035e1e9c16d0ccd9db5ada54a99d6c664`.
Rebuilding the integer fix took 24.43 seconds. All rows in this section use
that same executable and unchanged runtime-source hashes.

| Arm suffix | Options | First / repeat aggregate tok/s | Structural checks | Peak RSS GiB |
|---|---|---:|---:|---:|
| `control-b` | All three off | 71.44 / 115.45 | 29/30 | 69.823 |
| `head-b` | Shared head only | 76.24 / 130.25 | 29/30 | 69.821 |
| `control-c` | All three off, after head-b | 73.71 / 118.47 | 30/30 | 69.813 |
| `head-c` | Shared head only, second candidate | 76.44 / 125.62 | 30/30 | 69.889 |
| `state-fixed` | Persistent state 1024 MiB only | 71.34 / 114.91 | 29/30 | 69.721 |
| `state-2048` | Persistent state 2048 MiB only | 72.95 / 112.98 | 29/30 | 69.840 |
| `all-fixed` | All three, state 1024 MiB | 69.42 / 111.83 | 30/30 | 69.889 |
| `adaptive-b` | Adaptive depth only | 67.88 / 107.30 | 30/30 | 70.029 |

Each row completed **30/30 runtime requests** and exited cleanly. Table rates
are complete phase wall-time rates, not a single stream's decode measurement.
These are screening samples, not confidence intervals or a fresh external-engine
parity benchmark. The anomalously slow first pre-fix control is not used below.
Across these eight corrected-binary arms: **240/240 runtime requests**,
**236/240 structural checks**, with four omitted `fix` fields. The earlier
failed binary's six transport failures remain separate, not erased by reruns.

### Shared head: a repeatable narrow gain

Matched b/b and c/c pairs improve first-phase token rates by **6.73% / 3.70%**
and repeat token rates by **12.81% / 6.03%**. Repeat structurally valid tasks/s
also improves **11.06% / 6.87%**, so the gain is not merely longer answers.
Repeat output counts are 2600→2641 and 2672→2651; phase durations are
22.52→20.28 and 22.55→21.10 seconds. RSS shows no material difference here.

The intervening reverse control also leaves head-b ahead by 9.94% repeat
tokens/s. This reuses the same head-b observation; it is **not** a third
independent candidate measurement. The initial pre-fix head-a run was slower
at 114.70 tok/s and remains in the record; its cause was not isolated.

The b/c candidate arms execute 1811/1876 shared draft rows and 1353/1416
shared repair rows. Structural totals are **59/60 for both control and head**.
The b pair has the same `AGENT-13` omission of `fix`; both c arms pass all
structural checks. Only **6/30 and 2/30** paired texts/token counts are exact.
This supports retaining M16 as a measured opt-in, not silently changing defaults
or claiming semantic equivalence. Broader sampled/long-context quality remains.

### Persistent state: activation is not enough

At 1024 MiB, only **3 reused / 0 refreshed** rows are observed; at 2048 MiB,
**195 reused / 5 refreshed**. Both corrected arms complete without the integer
scatter failure and report zero retained bank bytes at shutdown. Yet repeat
rates versus control-b are **−0.47% / −2.15%**, with structurally valid task
rates **−1.22% / −5.51%**. Larger cache capacity is not a demonstrated speed win.

Both state arms omit `fix` for `warm-repeat/AGENT-13`; control-b has that same
kind of omission in its first phase instead. These are completed, coherent
responses with a missing requested field, not transport failures. The screen
does not establish whether arithmetic/scheduling changes caused an omission.
Do not hide the failure by treating every runtime success as quality success.

### Adaptive depth: request-local efficiency loses batch efficiency

The corrected adaptive-only arm exercises depths 1/2/3 for **793/766/797**
cycles. Average shared width is **3.27** versus control-c's **5.20**.
First/repeat aggregate rates decrease **7.91% / 9.43%**, while structurally
valid task rates decrease **4.03% / 6.96%**. All 30 structural checks pass,
but only two paired outputs/counts match exactly. This is not a recommended
speed preset. The earlier pre-fix adaptive arm is similarly unconvincing;
its small advantage versus the anomalous first control must not be promoted.

### Combined path: functional, not recommended for speed

All three switches together produce **60 reused / 4 refreshed** state rows,
1959 shared draft rows and 1608 repair rows, with depths 1/2/3 all exercised.
Average shared-group width falls to **3.24** versus control-b's **5.27**.
Repeat throughput is **3.14% lower** and structurally valid task rate **2.57%
lower** than control-b. Individual optimizations do not add automatically.

Five combined-repeat responses were also spot-checked. They are coherent, but
the streaming-task answer suggests an imprecise final-chunk fix that could drop
content if copied literally. JSON validity is not proof of a correct code fix.
This reinforces the need for broader semantic qualification before promotion.

## What the next iteration should change

1. Keep M16's shared head as the measured candidate and extend its qualification
   to sampled agentic workloads and longer histories. Do not combine slower
   switches by default or imply the overall parity goal is achieved.
2. Make adaptive selection aware of **cohort cost and loss of sharing**, include
   deferred head repair in its cost accounting, and evaluate stable depth epochs
   for compatible groups. Request-local token/cost learning alone fragmented the
   batch enough to lose throughput in this screen.
3. Investigate stable-slot state banks that survive cohort membership changes,
   with explicit row retirement and rollback-frontier selection. Simply raising
   the whole-group LRU budget increased reuse without a material speed benefit.
   This is a further layout/ownership experiment, not an implemented claim.

Broader semantic qualification, sampled throughput, prefix-off, other client
counts, long-context soak and other model architectures are not qualified by
these measurements. All three settings remain off by default. Lifecycle
qualification is separate from timed agentic output checks.

## API lifecycle qualification

Two separate C6 runs use the corrected executable and checkpoint:

| Suffix | Settings | Cancel / after-cancel / repeat assertions | Total |
|---|---|---|---|
| `head-lifecycle` | M16, shared head only | 36/36, 42/42, 42/42 | **120/120** |
| `all-lifecycle` | M19, all three / 1024 MiB | 36/36, 42/42, 42/42 | **120/120** |

Both servers exit cleanly. The 36 total lifecycle requests include two expected
early client disconnects. Coverage includes greedy and sampled MTP requests,
ordinary fallback requests, finite visible logprobs, token caps, stop-marker
non-leakage, completion and full prompt replay after cancellation. These are
contract checks, not semantic judging of intentionally short answers.

Head-only executes 29 shared draft / 21 repair rows. Combined executes 41/32,
uses depths 1/2/3 for 13/49/24 cycles, reuses 26 state rows and retains zero
bank bytes at shutdown. Wide-integer **partial** row refresh is additionally
exercised in the corrected C15 combined and 2048-MiB runs, plus the GPU unit test.

The frozen lifecycle wrapper repeats some identical owner assignments; there
are no conflicting effective values. The timing arms have unique assignments.
Lifecycle runs are not used as performance measurements and do not have the
timing wrapper's per-second collision sampler.

Final evidence is frozen in `SHARED-WORK-20260913-SHA256SUMS.txt` under the
external evidence root: **630 verified entries**, manifest SHA-256
`0e9a101a527a517841e31b199c924abdfcbe367710a29a18a4e2d109f55c1002`.
The preceding 433-entry composed-verifier and 452-entry width manifests also
still verify unchanged. The snapshots in `shared-work-20260913-binaries` preserve
executable bytes, not complete installable packages. No release assets, main
branches or local Homebrew installation were changed by this experiment.
