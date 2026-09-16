# Qwen Next opt-in activation and experiment matrix

Last source audit: **2026-09-14**, candidate AFMKit runtime `49a97c7a`, paired consumer
`9acccfc`. Workstream: [PR #123](https://github.com/scouzi1966/AFMKit/pull/123).
This is the central index of settings and combinations for the Qwen Next
optimization project. Update it with every new experiment, removal or result.
Historical reports remain the evidence; this index does not rewrite them.

Inventory audit: **46 named Qwen controls, 45 wired and one removed**. This
covers every Qwen control named in the preceding `QWEN_NEXT_*` reports and
shared-batch workstream, plus the existing deferred-token-resolution control.
The generic replay/profiling/SDPA controls and CLI settings are listed separately.
This is a workstream inventory, not a count of every legacy AFM environment
variable. All linked local files/headings and the shell recipe's syntax were
checked when this index was introduced.

**No defaults are promoted by this document.** These are branch experiments,
not installed-release recommendations. Measured combinations, available but
untested combinations, and rejected implementations are different statuses.
Unlisted combinations are **not tested**, not implicitly approved.

Prefill quality checkpoint: the candidate honors bounded MTP prefill and fixes the
five initial greedy target probes. Sampled file-selection remains **17/25**
versus old MTP **15/25**, ordinary **18/25** and frozen reference MTP **24/25**.
The initial slow candidate run is retained; a full timing repeat gives
**28.83 versus 28.60 tok/s**, with about 0.19 s longer candidate median TTFT.
No new runtime environment controls were added and no preset is promoted.
See [the full investigation and test-only capture controls](QWEN_NEXT_MTP_PREFILL_QUALITY.md).

Latest follow-up: [actual filename decisions and sampler analysis](QWEN_NEXT_MTP_FILENAME_QUALITY.md)
reproduce five API prefixes and pass 81920 frozen-logit draws. The seed-73
wrong-file cluster shares an identical random perturbation; reference MTP
same-seed replay also varies. The distinct-seed 512-token-budget comparison
is complete: 220/220 runtime requests, final unique-key JSON + identity scores
37/50 AFM MTP, 38/50 AFM AR, 41/50 reference MTP, 40/50 reference AR. AFM MTP
is within 0.64% in C1 output tok/s, but 14.0% behind in valid tasks/s. The
quality gate remains open. No new runtime controls or default changes.

September 15: [peak/degradation ledger](QWEN_NEXT_PERFORMANCE_LEDGER.md) retains
the context highs. The [paired explicit-prefill screen](QWEN_NEXT_PREFILL_TRADEOFFS.md)
completed 220/220 runtime requests: 8192 is **not** a general recommendation.
The CLI remains available; no automatic selection or default change is adopted.

September 16: [consolidation and calibrated component diagnosis](QWEN_NEXT_CONSOLIDATION.md)
preserves the baseline and reproduces all 134 measured responses after merging
provider main. An actual source-server capture isolates a grouped-normalization
rounding difference and a GDN Q/K preparation difference. Test-only ablations
are complete; a private GDN-only API prototype improves sampled strict passes
from 38/50 to 43/50, with seven improvements and two regressions. No new launch
preset or CLI/API/environment option is enabled. Numerical closeness and this
small quality screen alone are not adoption criteria. Combined concurrent-cache
qualification and broader reference quality parity remain pending.

### Test-only normalization matrix

These arms belong to `QwenNextNormalizationAblationTests`, not AFM launch
recipes. All use the same checkpoint, saved token IDs, MTP off, and fresh
request-owned caches. The test inputs are `AFM_QWEN_PREFILL_QUALITY_MODEL`,
`AFM_QWEN_PREFILL_QUALITY_TOKENS` and a fresh `AFM_QWEN_PREFILL_QUALITY_OUT`.
There are **no new runtime tuning environment variables**. Both internal
diagnostic properties default off and are restored after the test.

| Arm | HC reference rounding | GDN reference Q/K | Prefill geometry | Qualification |
|---|---|---|---|---|
| current | Off | Off | Current 4096 chunks | Frozen logits and greedy answers reproduced |
| reference-norm | On | Off | Current 4096 chunks | 5/5 greedy; mixed probability changes, not adopted |
| reference-geometry | Off | Off | N−1 then singleton | Fixed-prefix diagnostic only |
| reference-norm-and-geometry | On | Off | N−1 then singleton | Fixed-prefix diagnostic only, mixed effects |
| reference-gdn-qk | Off | On | Current 4096 chunks | 5/5 greedy; four probability gains, one unchanged; private API screen below |
| reference-hc-and-gdn-qk | On | On | Current 4096 chunks | 5/5 greedy; mixed probability effects, not adopted |

GDN reference Q/K preparation is limited to unbatched prefill of at least
128 tokens. The initial diagnostic deliberately repeated convolution to
isolate arithmetic; its binary and evidence are preserved. The current
default-off implementation fuses the same arithmetic into prework for
128-dimensional heads, retaining actual values/gates and FP32 recurrence.
Exact full-model and A/B/A API replays pass, with no material warm throughput
change. Wider quality and combined qualification remain required before
considering adoption. GDN-reference arms
with N−1 geometry, MTP, prefix reuse or concurrency are not tested by this
screen; unlisted combinations are not qualified.

### Private API normalization screen (no user activation)

| Profile | Delta / settings | Measured outcome | Status |
|---|---|---|---|
| N0 control | Frozen consolidated binary; saved M25 settings, AR/C1/prefix off, thinking off, prefill 4096 | 55/55 runtime; 5/5 greedy; 38/50 sampled strict; 26.3007 wall tok/s; 60.3155 median decode tok/s | All frozen texts reproduced |
| N1 GDN Q/K | Same launch/payloads/checkpoint; private build enabling only internal GDN normalization diagnostic | 55/55 runtime; 5/5 greedy; 43/50 sampled strict; 26.4605 wall tok/s; 60.2497 median decode tok/s | Diagnostic only, not promoted; +7/-2 strict cases |
| N1 + MTP/prefix/C15 | Not run in this screen | No combined throughput/quality claim | Untested |
| N2 fused implementation | Same internal activation; reference Q/K directly in prework, without duplicate convolution | Six focused tests pass; 90 arrays and 20 greedy answers exact; A/B/A API answers identical, 43/50 sampled strict each; warm decode 61.8237 vs 61.6710 tok/s, wall 26.9248 vs 26.9797 | Retained default-off; no material throughput gain; [follow-up](QWEN_NEXT_CONSOLIDATION.md#fused-gdn-implementation-follow-up) |

The temporary source activation was reverted after each private build and
never committed; the frozen enabled binaries were used for API testing. Both
source diagnostic defaults are off. The mutable development binary has also been
rebuilt default-off and passed a one-case frozen-answer API restore smoke.
The installed nightly is unchanged. Frozen candidate/control hashes, exact
protocol, failure details and limitations are in the
[controlled GDN API screen](QWEN_NEXT_CONSOLIDATION.md#controlled-gdn-api-screen)
and [fused follow-up](QWEN_NEXT_CONSOLIDATION.md#fused-gdn-implementation-follow-up).
The saved 26 M25 overrides are retained: these are **not** unset-environment
results. No inference-setting inventory entry is added for an internal test
property. Memory guards passed. N0/N1 did not measure peak process memory;
N2 includes one-second process RSS sampling, not full Metal-memory accounting.
No prefix hits, repeated-phase gain or aggregate-concurrency gain is claimed.

## How to read the matrix

- **AR**: ordinary decoding, without `--mtp`.
- **MTP**: speculation enabled with `--mtp`; concurrent recipes below use
  `--mtp-depth 3` and explicitly opt into the `batched` verification policy.
  This can change greedy wording; it does not promise exact AR token equality.
- **C15** means fifteen overlapping client requests and server capacity 15.
  Setting `--concurrent 15` does not generate fifteen requests by itself.
- **Prefix on/off** means presence/absence of `--enable-prefix-caching`.
- **First/repeat** are workload phases, not cache modes. After an excluded
  warmup, run fifteen agentic review prompts, then the exact prompts again.
  First requests can already reuse a common prefix. Repeats generate new
  answers; there is no cached-answer-string shortcut. Prefix-off repeats can
  still benefit from warm weights, kernels and filesystem pages.
- Rates below are **aggregate emitted output tokens / phase wall time**,
  including queue/prefill time, unless explicitly marked single-client decode.
  Also record structurally valid tasks/s, latency, output lengths and memory.
  More tokens/s with longer answers does not necessarily complete tasks faster.
- **Historical measurement** identifies the binary in the linked report. It
  is not a claim that every old combination was rerun on `a1255dc6`.

## Identity and launch recipes

Exact qualification checkpoint, unchanged between paired arms:

```text
/Volumes/edata2/models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit
```

Paired-development binary, not the installed Homebrew executable:

```text
/Volumes/edata2/dev/CODEX/maclocal-api-qwen-next-mtp-parity/.build/release/afm
```

Historical runtime SHA-256 (`a1255dc6`, width and composed-verifier experiments):
`777a3a8708dc8a650e1fe9773caa2aa6fa42030be4f47e8aac486cb47469ed3d`.
Corrected shared-work experiment runtime SHA-256 (`4d2cf327`):
`a78772f6c111715d68d9f428926d6ea035e1e9c16d0ccd9db5ada54a99d6c664`.
The initial shared-work binary `e4d914ee…` exposed an int64 state-update crash;
its failed arm remains separate evidence, not a passing qualification.
The cohort-work binary (`bfcb6b5b`) has SHA-256
`0c63e09ab41c6a11b251f54a3a1fa9fd9659b48411e84aa15b4ebb566954c360`.
The long-prompt replay-limit binary (`739cdb6f`) has SHA-256
`4193a708f44f5b59cab393e3e4452d2e469e5e67e7a62faf82ffa607cdc5c93a`.
The bounded-prefill quality candidate (`49a97c7a`) has SHA-256
`10086d6504aa7dbd6e48ed5c8678cc0b1bdfd3a260a64836667de1a141b0b0c6`.
The binary reports development `v0.9.20`; the version string does not establish
source identity. The path is mutable: hash it again after any rebuild.
Earlier reports' hashes identify earlier binaries, not what currently occupies
this path. A documentation-only commit does not change the runtime hash.

These paths and results belong to the M3 Ultra 512-GiB qualification machine.
Roughly 68–70 GiB peak process RSS is **not** a minimum-memory guarantee or a
complete Metal-memory measurement. The existing harness requires at least
160 GiB available before loading. This checkpoint/preset is not M4 Pro qualified.

Use a shell without inherited tuning overrides. Environment assignments apply
to that child process only and require a server restart to change; they are
not API kwargs. Do not put them in shell startup files. The frozen harness
scrubs inherited tuning environments; the manual commands only fix the listed
settings. In particular leave profiling, custom SDPA partition overrides and
unlisted Qwen tuning flags unset during timing. Run only one GPU owner and no
concurrent build. Check port 9998 is free and verify the actual startup URL;
do not interrupt an unrelated server on 9999.

### Recipe A: AR request-banked attention, C15, prefix on

The complete command is maintained in
[the AR launch guide](QWEN_NEXT_BANKED_ATTENTION_OPT_IN.md#2-launch-the-measured-opt-in-preset).
It enables compatible/continuous/mixed-position groups, batch GDN prework and
compilation, shared attention projections, native-arithmetic request banks,
yield interval 8, admission budget 1024, serial replay boundaries and prefill
step 8192. **MTP, prefill interleaving and the PLE row cache are off.**
Use that complete command, not the banked-attention flag by itself.

### Recipe M: MTP shared verifier, C15, prefix on

The complete command is maintained in
[the MTP launch guide](QWEN_NEXT_SHARED_VERIFIER_OPT_IN.md#launch-a-controlled-experiment).
It enables the MTP scheduler, shared verifier, submission window 4, shared
submission ladder 4, replay budget 4096 MiB and depth 3. It leaves the compiled
shared tail, independent-attention follow-up, shared vocabulary and row cache
off. It does not explicitly override `--prefill-step-size`.

### Recipe W8: latest wider MTP group candidate

This is recipe M plus independent attention and window 8; vocabulary sharing
remains off. The explicit zeroes below also prevent accidentally inheriting
those particular experiments. They do not sanitize every possible variable.

```bash
/usr/bin/env \
  AFM_QWEN_VERIFY_SHARED_ASYNC_LADDER=4 \
  AFM_QWEN_VERIFY_SHARED_COMPILED_TAIL=0 \
  AFM_QWEN_VERIFY_DEFER_HC=0 \
  AFM_QWEN_MTP_SCHEDULER=1 \
  AFM_QWEN_MTP_REPLAY_MIB=4096 \
  AFM_QWEN_MTP_REPLAY_MAX_TOKENS=4096 \
  AFM_QWEN_MTP_SUBMISSION_WINDOW=8 \
  AFM_QWEN_MTP_SHARED_VERIFY=1 \
  AFM_QWEN_MTP_INDEPENDENT_ATTENTION=1 \
  AFM_QWEN_MTP_SHARED_VOCAB=0 \
  AFM_QWEN_MTP_SHARED_HEAD=0 \
  AFM_QWEN_MTP_ADAPTIVE_DEPTH=0 \
  AFM_QWEN_MTP_PERSISTENT_STATE_MIB=0 \
  AFM_QWEN_MTP_COHORT_DEPTH=0 \
  AFM_QWEN_MTP_STATE_REMAP=0 \
  AFM_QWEN_MTP_RETAIN_ANCHOR=0 \
  AFM_QWEN_PLE_ROW_CACHE_MIB=0 \
  AFM_QWEN_PLE_VECTOR_UNPACK=0 \
  AFM_QWEN_BATCH_COMPATIBLE_GROUPS=1 \
  AFM_QWEN_BATCH_CONTINUOUS_GROUPS=1 \
  AFM_QWEN_BATCH_YIELD_INTERVAL=8 \
  AFM_QWEN_VERIFY_QMM=1 \
  AFM_QWEN_PLE_NATIVE_READS=0 \
  AFM_QWEN_MTP_VERIFICATION_POLICY=batched \
  AFM_QWEN_VERIFY_ATTENTION_CHUNK=2 \
  AFM_QWEN_VERIFY_FUSED_HC=1 \
  AFM_QWEN_HC_NATIVE_CHAIN=1 \
  AFM_QWEN_VERIFY_FUSED_ROUTER=1 \
  AFM_QWEN_VERIFY_ASYNC_LADDER=8 \
  AFM_QWEN_MTP_DRAFT_ASYNC_LADDER=1 \
  /Volumes/edata2/dev/CODEX/maclocal-api-qwen-next-mtp-parity/.build/release/afm mlx \
  -m /Volumes/edata2/models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit \
  --port 9998 \
  --no-think \
  --mtp \
  --mtp-depth 3 \
  --concurrent 15 \
  --enable-prefix-caching
```

For the **same-binary W4 control**, replace the window value with
`AFM_QWEN_MTP_SUBMISSION_WINDOW=4`; keep independent attention enabled and
vocabulary disabled. For the **V4 vocabulary candidate**, start with W4 and
replace `AFM_QWEN_MTP_SHARED_VOCAB=0` with `AFM_QWEN_MTP_SHARED_VOCAB=1`.
W8+vocabulary was subsequently measured as **M11**, with mixed quality and
small gains; it is not a recommended preset.

### Recipe W8-L2: smaller shared submission interval

Start with the complete W8 command above and **replace**, rather than duplicate,
`AFM_QWEN_VERIFY_SHARED_ASYNC_LADDER=4` with
`AFM_QWEN_VERIFY_SHARED_ASYNC_LADDER=2`. Keep window 8, vocabulary `0` and all
other arguments unchanged. Restore ladder `4` to return to W8.

This is **M15**, a separately measured opt-in: repeat aggregate tok/s improved
2.35–5.39%, while first-phase token rate decreased 1.10–1.45%. Outputs differed;
one pair's repeat completion time barely changed. See
[the complete tradeoff and quality evidence](QWEN_NEXT_COMPOSED_VERIFIER_EXPERIMENTS.md).
W8-L2 plus vocabulary, prefix off, or sampled-performance combinations are
**not measured** by this result. The existing W8 recipe remains ladder 4.

For manual WebUI use append `-w`. Timing runs did not use it. To turn prefix
caching off, omit `--enable-prefix-caching`; leave other settings unchanged
for a controlled comparison. The replay cache then cannot activate. To leave
the entire experiment, stop only the owned server and launch without these
environment settings from a clean shell.

### Recipes M16–M20: three independently controlled shared-work experiments

Start from the **complete W8-L2 recipe** above (ladder 2, vocabulary 0).
Replace only the assignments in the selected row; do not append duplicate
assignments. These options are not API kwargs or installed-release defaults.

| ID | Shared head | Adaptive depth | Persistent state MiB | Meaning |
|---|---:|---:|---:|---|
| Control | 0 | 0 | 0 | Fixed depth 3, independent head, ordinary state merges |
| M16 | 1 | 0 | 0 | `AFM_QWEN_MTP_SHARED_HEAD=1` |
| M17 | 0 | 1 | 0 | `AFM_QWEN_MTP_ADAPTIVE_DEPTH=1`; selects depths 1–3 |
| M18 | 0 | 0 | 1024 | `AFM_QWEN_MTP_PERSISTENT_STATE_MIB=1024` |
| M19 | 1 | 1 | 1024 | All three; separate combination, not summed individual gains |
| M20 | 0 | 0 | 2048 | Larger state-bank budget only; separate screen |

Head + adaptive without state, head + 1024-MiB state without adaptive, and adaptive +
state without head are **not measured**. Neither is the all-three combination
at 2048 MiB. They must not inherit M16's gain or M19's qualification by addition.

Implementation and qualification are documented in
[shared speculative work](QWEN_NEXT_SHARED_SPECULATIVE_WORK.md). Other depths,
sampled throughput and prefix-off combinations need their own evidence.
Actual group width and state-reuse counters matter;
enabled flags alone do not prove an active optimization.

### Recipes M21–M24: cohort epochs and membership remapping

Start from **M16**, the complete W8-L2 recipe with shared head `1`, old adaptive
depth `0`, vocabulary `0`, and replace the three assignments below. These are
not stand-alone commands. `COHORT_DEPTH` takes precedence over `ADAPTIVE_DEPTH`
if both are set, but measured recipes explicitly keep the latter off.

| ID | `AFM_QWEN_MTP_COHORT_DEPTH` | `AFM_QWEN_MTP_PERSISTENT_STATE_MIB` | `AFM_QWEN_MTP_STATE_REMAP` |
|---|---:|---:|---:|
| M16 control | 0 | 0 | 0 |
| M21 | 1 | 0 | 0 |
| M22 | 0 | 2048 | 0 |
| M23 | 0 | 2048 | 1 |
| M24 | 1 | 2048 | 1 |

M21 selects depths 1–3 in shared epochs, not independently per request. M23/24
can reuse certified state rows across changed membership while keeping attention
private. They still perform bank reconstruction, not zero-copy in-place storage.
All remain opt-in. Other budgets, old adaptive + remapping, cohort + old bank
without remapping, and prefix-off/sampled-throughput variants of the state-bank
and remapping combinations remain unmeasured. M21's separate long sampled
screen is recorded below; it did not generalize its short greedy speed gain.
See [implementation and results](QWEN_NEXT_COHORT_WORK_EXPERIMENTS.md).
New evidence is frozen in `COHORT-WORK-20260913-SHA256SUMS.txt` under the same
external root: **442 verified entries**, SHA-256
`3e4309702d7804e88b727fe0dabb724fa08ab60dc6de9b6bf02f9e60393d1c3c`.

### Request controls are separate

The W4/W8/V4 and W8-L2 screens use these fields with the frozen agentic prompts:

```json
{
  "temperature": 0,
  "top_p": 1,
  "seed": 42,
  "max_tokens": 512,
  "stream": true,
  "stream_options": {"include_usage": true},
  "chat_template_kwargs": {"enable_thinking": false}
}
```

This is a request-field fragment, not a complete request: `model` and
`messages` are also required. Most earlier aggregate screens used a 192-token
cap. The 512-token rows must not be compared with 192-token rows as if the
workload were unchanged. Sampled decoding is supported and separately covered
by lifecycle tests; the greedy aggregate results do not qualify its throughput.

## Combination matrix: AR, cache and admission

“A minus/change” means edit the named assignments in the complete recipe A,
not add a second conflicting assignment. The recipes describe how to activate
the current paths; use each historical run's saved `launch.json` and hash for
exact reproduction of its old environment. AR aggregate profiles use C15 and
prefix on unless a row specifies otherwise; cache-only profiles may specify
their own mode. For a one-variable A/B, leave every other setting unchanged.

| ID | Combination / activation | Evidence and disposition |
|---|---|---|
| A0 | AR baseline: omit `--mtp` and all experimental environment settings; select C1/C15 and prefix on/off explicitly | Separate baseline, not the tuned results below. Latest complete six-mode no-tuning matrix is not certified here. |
| A1 | Compatible groups: `AFM_QWEN_BATCH_COMPATIBLE_GROUPS=1`; other new AR switches off | Historical subgroup screen; groups only compatible states, not arbitrary-position or continuous admission. [History](SHARED_BATCH_EXECUTION_WORKSTREAM.md#compatible-uniform-decode-groups-opt-in-prototype). |
| A2 | A1 + `AFM_QWEN_BATCH_CONTINUOUS_GROUPS=1` | Historical burst improvement but one-prefill staggered implementation cost 7–8% aggregate in exchange for late-arrival latency. Not a default recommendation. [History](SHARED_BATCH_EXECUTION_WORKSTREAM.md#continuous-group-admission-separate-opt-in-experiment). |
| A3 | A2 + admission budget `AFM_QWEN_BATCH_PREFILL_TOKEN_BUDGET=1` versus `1024` | Historical staggered repeat gain about 12%, repeated in reverse order; arrival patterns matter. [History](SHARED_BATCH_EXECUTION_WORKSTREAM.md#budgeted-incoming-cohorts). |
| A4 | Continuous owner + `AFM_QWEN_BATCH_YIELD_INTERVAL=64`, `8` or `1` | All screened. Interval 1 reduced TTFT but cost 6–10% aggregate; interval 8 is the recorded tuned preset, not a new default. [Tradeoff](SHARED_BATCH_EXECUTION_WORKSTREAM.md#admission-cpu-turns-latencythroughput-tradeoff). |
| A5 | Mixed-position only: A with banking, attention projections, both GDN controls set to `0` | Historical repeat 150.06 versus 120.04 tok/s control, prefix on. Structural totals unchanged; not exact output equivalence. [Ablations](SHARED_BATCH_EXECUTION_WORKSTREAM.md#same-binary-aggregate-confirmation). |
| A6 | Batch GDN ablation: A with banking, attention projections and mixed positions `0`; compare prework `1` with batch compilation `0`/`1` | Historical repeat 123.53/124.04 versus 120.04 control. Both switches are distinct. [Ablations](SHARED_BATCH_EXECUTION_WORKSTREAM.md#same-binary-aggregate-confirmation). |
| A7 | Mixed positions + fused/compiled GDN: A with banking and attention projections `0` | Historical repeat gains 27.6–29.2% prefix on, 15.0% prefix off. Row cache/interleaving off. [Ablations](SHARED_BATCH_EXECUTION_WORKSTREAM.md#same-binary-aggregate-confirmation). |
| A8 | Shared attention projections: A with only banking `0` | Historical incremental +2.7–3.3% prefix repeat; no-prefix structural count fell at fixed 192-token cap. Not broad quality equivalence. [Evidence](SHARED_BATCH_EXECUTION_WORKSTREAM.md#shared-attention-projection-experiment-2026-09-12). |
| A9 | Full A: native-arithmetic request-banked attention | Historical +7.7–8.3% incremental prefix repeat (175.09–175.86 tok/s), +5.1% no-prefix (87.09). Paired short-case outputs identical. Separate 512-token result 169.84 tok/s, 30/30 structural. [Evidence](QWEN_NEXT_BANKED_ATTENTION_OPT_IN.md). |
| A10 | Prefill interleave: A7 + `AFM_QWEN_BATCH_PREFILL_INTERLEAVE=1`, CLI `--prefill-step-size 1024`, prefix off, staggered 8+7 arrivals | Tested on a separate long-prompt workload: stream pauses improved 5.1x, repeat aggregate −0.9%, late TTFT +6.2%, structural 18/30 → 15/30. Remains off. A9+interleave is not this measured pair. [Evidence](SHARED_BATCH_EXECUTION_WORKSTREAM.md#revised-chunkinterleave-confirmation). |
| C1 | Serial AR + prefix on + `AFM_PREFIX_REPLAY_BOUNDARIES=1` | Historical repeat 46.80 → 65.06 tok/s, restored tokens verified; narrow Qwen text guard. Not MTP replay. [Evidence](SHARED_BATCH_EXECUTION_WORKSTREAM.md#first-implementation-serial-replay-boundaries). |
| C2 | PLE decoded-row cache: `AFM_QWEN_PLE_ROW_CACHE_MIB=0` versus `4`, AR/MTP, C15, prefix on | Historical +0.4–2.2%; all 60 paired texts/token counts identical. Small signal only. **Off in A and M/W8.** [Evidence](SHARED_BATCH_EXECUTION_WORKSTREAM.md#shared-immutable-lookup-cache-opt-in-prototype). |
| C3 | CPU vector unpack: `AFM_QWEN_PLE_VECTOR_UNPACK=1`, suitable q4 mapped-table geometry | Isolated unpack faster, full decode −1.2% to +1.6%; no material decode win. Native small-row worker path bypasses this loop. [Evidence](QWEN_NEXT_PLE_VECTOR_UNPACK_EXPERIMENT.md). |
| C4 | Sidecar residency: CLI `--qwen-ngram-residency mapped` versus `prewarm` | Current CLI supports these two values. Prewarm populates reclaimable filesystem pages; it is **not** a pinned/locked memory mode. No new combined A/M/W8 qualification. |

## Combination matrix: speculative decoding

All M-recipe variants below keep depth 3 unless explicitly stated otherwise.
“M minus/change” is an activation recipe, not proof that all earlier experiments
used the later ladder-4 settings. Older exact flags remain in their manifests.

| ID | Combination / activation | Evidence and disposition |
|---|---|---|
| M0 | Stock `--mtp`, no experimental environment settings | Strict policy, distinct from batched-policy results. Do not quote tuned throughput as stock MTP performance. [Earlier baseline](QWEN_NEXT_MTP_PARITY_PROGRESS.md). |
| M1 | M recipe with `--concurrent 1`, omit prefix flag, select `--mtp-depth 1`, `3` or `4`; concurrent-owner flags do not activate on the serial lane | Greedy, temperature 0.6/top-p 0.95 and temperature 0.6/top-p 1 have separate first-four-context evidence using their recorded singleton tuning flags. Not concurrent aggregate results. [Parity history](QWEN_NEXT_MTP_PARITY_PROGRESS.md), [depth 3](QWEN_NEXT_MTP_DEPTH3_QUALIFICATION.md). |
| M2 | Scheduler-owned independent MTP: `AFM_QWEN_MTP_SCHEDULER=1`, shared verify `0`, submission window `1` | Streaming per-request sessions, not joint verification. [Integration](SHARED_BATCH_EXECUTION_WORKSTREAM.md#scheduler-owned-streaming-sessions). |
| M3 | M2 + prefix on + `AFM_QWEN_MTP_REPLAY_MIB=4096`; compare replay `0` | Complete target/head/history snapshots, not response caching. Historical C15 repeat gain about 49%; C1 serial lane did not use this cache. [Replay](SHARED_BATCH_EXECUTION_WORKSTREAM.md#complete-exact-prompt-mtp-replay-opt-in). |
| M4 | M3 + independent submission window `1`, `2`, `4`, shared verify still `0` | Historical independent/draft-first screens showed about 0–1% repeat difference; no material gain. [Evidence](SHARED_BATCH_EXECUTION_WORKSTREAM.md#bounded-independent-mtp-graph-submission-experiment). |
| M5 | Shared equal-position verification: recipe M with shared ladder `0` | Requires scheduler owner and eligible batched policy. Control for shared graph scheduling, not the independent-session control. [Evidence](SHARED_BATCH_EXECUTION_WORKSTREAM.md#shared-verifier-graph-scheduling-2026-09-12). |
| M6 | Shared ladder sweep: M with ladder `0`, `4`, `8`, `16`; compiled tail `0` | All screened historically. Ladder 4: 512-token repeat 90.08 → 98.31 tok/s (+9.14%); 30/30 paired texts/counts identical in that pair. [Confirmation](SHARED_BATCH_EXECUTION_WORKSTREAM.md#four-layer-scheduling-confirmation). |
| M7 | Shared compiled tail: M with compiled tail `1`, ladder `0` (tail only) or `8` (combined) | Historical 192-token gain weakened at 512; combined repeat +1.48% with one additional structural omission. Not recommended. Remains separately opt-in. [Evidence](SHARED_BATCH_EXECUTION_WORKSTREAM.md#shared-verifier-graph-scheduling-2026-09-12). |
| M8 / W4 | M + `AFM_QWEN_MTP_INDEPENDENT_ATTENTION=1`, window `4`, vocabulary `0` | Private real-offset attention plus shared backbone. Measured repeat gains varied about 2.8–8.2%; useful-task gain not consistently material. [Evidence](QWEN_NEXT_MIXED_POSITION_VERIFICATION.md). |
| M9 / W8 | Full W8 command above: M8 with window `8` | Latest opposite-order pairs: +5.24/+6.40% first aggregate, +4.02/+8.73% repeat; repeat valid tasks/s +7.52/+6.41%. 60/60 candidate structural checks, but answers differ. Still opt-in. [Evidence](QWEN_NEXT_VERIFICATION_WIDTH_EXPERIMENTS.md). |
| M10 / V4 | M8 with `AFM_QWEN_MTP_SHARED_VOCAB=1`, window stays `4` | Latest repeat tok/s +1.72/+2.71%; valid tasks/s −3.34/+2.59%. Mixed quality/useful-work evidence; not a default candidate. [Evidence](QWEN_NEXT_VERIFICATION_WIDTH_EXPERIMENTS.md). |
| M11 | W8 + vocabulary `1`, ladder stays `4` | **Measured**, not recommended: repeat tok/s +0.95–3.59%; candidate 57/60 versus control 59/60 structural passes. Only eligible subgroups of at most four requests share vocabulary; not eight-way shared vocabulary. [Evidence](QWEN_NEXT_COMPOSED_VERIFIER_EXPERIMENTS.md#window-8-plus-vocabulary-sharing). |
| M12 | MTP + `AFM_QWEN_MTP_RETAIN_ANCHOR=1` and batched policy | Earlier depth-4 screen: mixed sampled results, changed answers in several cases, no broad gain. Off in current recipes. Current W8/anchor combination untested. [Evidence](QWEN_NEXT_COMMITTED_ANCHOR_EXPERIMENT.md). |
| M13 | Shared verifier + non-greedy requests (for example temperature 0.6/top-p 0.95) | Per-request sampling/lifecycle coverage exists, including mixed greedy/sampled groups. Long sampled M16/M21/M25 screens and a separate matched-control six-mode matrix now exist; their structural failures remain a qualification gate. They are not interchangeable with the short greedy results. [Long-mode qualification](QWEN_NEXT_LONG_MODE_QUALIFICATION.md). |
| M14 | W8 + replay off, prefix off, other client counts, longer contexts, or PLE row cache | Controls are available where their guards allow. **Latest window-8 throughput/quality matrix not run for these combinations.** Do not extrapolate C15/prefix-on results. |
| M15 / W8-L2 | W8 with `AFM_QWEN_VERIFY_SHARED_ASYNC_LADDER=2`, vocabulary stays `0` | **Measured**, separate opt-in: repeat 117.46–119.21 tok/s, +2.35–5.39% over matched ladder-4 arms; first-phase token rate −1.10–1.45%. Structural 60/60 versus 59/60, but changed outputs and mixed latency. [Evidence](QWEN_NEXT_COMPOSED_VERIFIER_EXPERIMENTS.md#window-8-submission-every-two-versus-four-layers). |
| M16 | W8-L2 + shared head only | **Measured candidate**: corrected repeat 125.62–130.25 tok/s, +6.03–12.81%; structurally valid tasks/s +6.87–11.06%. Candidate/control both 59/60 structural across two pairs, but outputs differ. Still opt-in. [Evidence](QWEN_NEXT_SHARED_SPECULATIVE_WORK.md). |
| M17 | W8-L2 + adaptive depth only | **Measured, not recommended for speed**: 107.30 repeat tok/s, −9.43% versus matched control; smaller shared groups. 30/30 runtime and structural. [Evidence](QWEN_NEXT_SHARED_SPECULATIVE_WORK.md). |
| M18 | W8-L2 + 1024-MiB persistent state only | Corrected live screen: 30/30 requests, 29/30 structural; 114.91 repeat tok/s versus control 115.45, only three reused rows. No demonstrated cache speedup. [Evidence](QWEN_NEXT_SHARED_SPECULATIVE_WORK.md). |
| M19 | W8-L2 + all three | Live screen: 30/30 requests and structural checks; 111.83 repeat tok/s, −3.14% versus control. Reuse 60 / refresh 4; functional, not a speed recommendation. [Evidence](QWEN_NEXT_SHARED_SPECULATIVE_WORK.md). |
| M20 | W8-L2 + 2048-MiB persistent state only | **Measured, no speed win**: 112.98 repeat tok/s, −2.15%; 195 reused / 5 refreshed rows. 30/30 runtime, 29/30 structural. [Evidence](QWEN_NEXT_SHARED_SPECULATIVE_WORK.md). |
| M21 | M16 + cohort depth | **Small repeated gain**: repeat 133.46–133.84 tok/s, +2.45/+2.98% over matched M16 controls; valid tasks/s +0.25/+3.79%. Shared widths 5.31/5.42, candidate and controls each 59/60 structural. Still opt-in. [Evidence](QWEN_NEXT_COHORT_WORK_EXPERIMENTS.md). |
| M22 | M16 + old persistent bank, 2048 MiB | **No speed recommendation**: repeat 124.75 tok/s, −4.51% over control-a. 174 reused rows; 30/30 structural. |
| M23 | M22 + membership remapping | **Functional, no material gain over no bank**: repeat 130.75 tok/s; +4.81% versus old bank, only +0.08% versus M16. 238 reused rows, 27 membership hits, 30/30 structural. |
| M24 | M23 + cohort depth | Fully instrumented repeat 133.98 tok/s, +3.38% versus M16 but only +0.39% versus M21. 30/30 structural, 458 reused rows. Extra memory not yet justified by a repeatable additive gain. One earlier run excluded for missing shutdown counters. |
| M25 | M16 + `AFM_QWEN_MTP_REPLAY_MAX_TOKENS=8192` | **Measured**, sampled 4.43K prompts: repeated aggregate 96.51/103.66 tok/s versus 32.62/32.54 at the default 4096 limit (top-p 1.0/0.95). Reversed top-p 1.0 confirmation: 99.57 versus 33.52. All 15 repeats hit complete state; default had zero hits. Across three pairs: 180/180 runtime, candidate 42/90 versus control 36/90 structural, so not quality-qualified. Long-context cancellation/replay: 120/120 assertions. Default, byte and entry budgets unchanged. This removes repeated prefill, not a 3× uncached decode gain. [Evidence](QWEN_NEXT_LONG_CONTEXT_REPLAY.md). |
| Q1 | `49a97c7a`, same frozen M25 controls, C1/prefix off, existing `--prefill-step-size` now honored by MTP (4096 architecture policy) | **Candidate, not promoted.** Initial greedy probes 5/5 match AR; full greedy 5/5; sampled 17/25 vs old 15/25. First timing run 21.97 tok/s retained; repeat candidate/old 28.83/28.60, with +0.19 s candidate median TTFT. No new environment flag. C15/replay performance not requalified. [Evidence](QWEN_NEXT_MTP_PREFILL_QUALITY.md). |
| Q2 | Same Q1 runtime/preset; test-only filename geometry, actual sampling-cycle capture, frozen full-vocabulary RNG replay | **Diagnostic, not promoted.** Five API prefixes reproduce exactly; 45 target paths captured; 81920 sampling-law draws pass. Four-arm screen: 220/220 runtime; final unique-key structure + identity 37/50 AFM MTP, 38/50 AFM AR, 41/50 reference MTP, 40/50 reference AR. AFM MTP 29.57 versus reference 29.76 output tok/s including prefill; valid tasks/s 0.1283 versus 0.1491. Repeated-seed cluster explained, residual quality gate open. No new runtime control. [Evidence](QWEN_NEXT_MTP_FILENAME_QUALITY.md). |
| Q3 | Same Q1 binary, historical depth-3/depth-4 launch controls; depth 4 adds CLI `--prefill-step-size 8192` | **Context-specific recovery, not quality-qualified.** At 4150 prompt tokens, depth-4 decode 72.31→88.63 tok/s, acceptance 46.7→60.8%; restores prior response text. 54 measured +18 warmups, eight separate diagnostics. Keep the peak, do not extrapolate to other workloads. [Ledger](QWEN_NEXT_PERFORMANCE_LEDGER.md). |
| Q4 | Same Q1/M25 C1, prefix off, depth 3; CLI `--prefill-step-size 4096` versus `8192`, MTP off/on | **Reject blanket 8192 recommendation.** 220/220 runtime + four warmups; both 4096 controls reproduce 55/55 prior answers. AR strict 38→37/50 and output 25.00→24.09 tok/s; MTP strict 37→34/50 and output 28.59→28.47. MTP valid tasks/s −9.18%. All rates include prefill; no semantic-judge or C15 qualification. No new environment variable or default. [Tradeoff gate](QWEN_NEXT_PREFILL_TRADEOFFS.md). |

**Long sampled follow-up on M16/M21:** at 4,427–4,433 prompt tokens,
temperature 0.6 and top-p 1.0/0.95, all 120 requests completed but only 51/120
passed JSON/file-identity checks. Cohort repeat token rate changed +2.53% at
top-p 1.0 and −3.63% at 0.95. This does not generalize the earlier short/greedy
gain. Every measured prompt exceeded the original 4096-token replay limit,
so repeats had **zero** cached tokens despite prefix caching being enabled.
Do not label these as cache-hit rates or a regression against the short fixture.
See [the distinct workload and quality limitations](QWEN_NEXT_LONG_CONTEXT_REPLAY.md).

## Six-mode coverage ledger

The user-requested cross product is MTP off/on × these three scenarios. It
remains an explicit checklist, not “cache tested” inferred from one warm run.

**New, separate sampled matrix:** runtime `739cdb6f` completed all six modes
with the M25 matched-control recipe, 4.43K prompts, temperature 0.6/top-p 1.0
and cap 512: **180/180 runtime, 134/180 structural** (AR 90/90, MTP 44/90).
One previously frozen M25 arm is reused, not counted as a new rerun. See the
[complete six-mode table](QWEN_NEXT_LONG_MODE_QUALIFICATION.md#six-mode-matched-control-matrix).
This holds controls fixed to isolate MTP; it does **not** replace the pending
best-A9 versus best-MTP short-fixture qualification below. C1 cache counters
were zero despite the prefix flag, and the C15 first phases had different
amounts of actual reuse. Those distinctions are explicit in the new report.

| Scenario | Activation delta | Historical integrated checkpoint `75d88a5a`: first / repeat tok/s | Latest A9 or W8 requalification |
|---|---|---:|---|
| AR, prefix only | A recipe, `--concurrent 1`, prefix on | 61.98 / 67.11 | Not rerun as latest full matrix |
| AR, prefix + C15 | A recipe unchanged | 109.55 / 128.24 | A9 historical paired screen: 175.09–175.86 repeat at 192; not a fresh current-binary six-mode run |
| AR, C15 only | A recipe, omit prefix flag | 71.47 / 71.83 | A9 historical paired screen: 87.09 repeat at 192 |
| MTP, prefix only | W8 recipe, `--concurrent 1` | 58.86 / 58.66 | C1 does not exercise the concurrent shared owner; latest W8 matrix not run |
| MTP, prefix + C15 | W8 recipe unchanged; W8-L2 is a separate delta | 59.93 / 87.13 | W8 width screen: 72.06/112.81 and 73.21/115.67. Later W8-L2: 72.37/119.21 and 72.18/117.46. All at **512**, not matched to the old 192-token result. |
| MTP, C15 only | W8 recipe, omit prefix flag | 58.69 / 58.24 | Latest W8 matrix not run |

The historical integrated run used its own common tuned flags, not current
recipe A or W8: see [the full identity and qualification](SHARED_BATCH_EXECUTION_WORKSTREAM.md#integrated-six-mode-checkpoint).
It completed 180/180 requests with 146/180 structural passes. Keep those totals
separate. The newer two-arm window experiment completed 120/120, with candidate
60/60 versus control 58/60 structural passes; exact paired text/count matches
were only 2/30 and 3/30. That is not proof of semantic parity. The subsequent
composed-verifier screen completed another 240/240 requests; M11 and M15's
structural and performance results remain separate in their linked report.
The shared-work follow-up completes **240/240 runtime requests and 236/240
structural checks** on its corrected binary, plus **240/240 lifecycle assertions**
in separate C6 head-only and all-three runs. It does not replace the missing
full six-mode requalification. The pre-fix state-cache crash remains separately
documented; it is not counted as a successful performance arm.

## Setting inventory: effective defaults and prerequisites

“Unset” describes source behavior at the audited commit, **not** values chosen
by a benchmark preset. Boolean experimental gates below require literal `1`;
`0`/unset disable them unless otherwise noted. Limits describe the setting,
not a process-memory or latency guarantee. These are wired controls, not
necessarily stable public CLI/API contracts.

### Scheduler and AR batching

| Environment setting | Unset default | Activation / effective range / prerequisite |
|---|---|---|
| `AFM_QWEN_BATCH_COMPATIBLE_GROUPS` | Off | `1`; concrete text `Qwen4ExpModel` |
| `AFM_QWEN_BATCH_CONTINUOUS_GROUPS` | Off | `1` + compatible groups |
| `AFM_QWEN_BATCH_MIXED_POSITIONS` | Off | `1` + continuous groups; **no Qwen MTP session owner** |
| `AFM_QWEN_BATCH_PREFILL_TOKEN_BUDGET` | 1024 | Integer clamped 1–8192; soft uncached-token admission budget, not GPU prefill chunk size |
| `AFM_QWEN_BATCH_YIELD_INTERVAL` | 64 | Integer clamped 1–64; cooperative admission turns in the selected continuous owner |
| `AFM_QWEN_BATCH_PREFILL_INTERLEAVE` | Off | `1`; qualified continuous text Qwen AR owner; MTP owner excluded |
| `AFM_QWEN_BATCH_GDN_PREWORK` | Off | `1`; supported batched GDN geometry |
| `AFM_QWEN_COMPILE_BATCH_GDN_DECODE` | Off | `1` + batch GDN prework; compiled batch path, not the singleton compile switch |
| `AFM_QWEN_BATCH_ATTENTION_PROJECTIONS` | Off | `1`; shared attention operations in the request-owned AR adapter |
| `AFM_QWEN_BATCH_BANKED_ATTENTION` | Off | `1` + shared attention projections; eligible BF16/head-256 request banks |

### Cache and sidecar

| Environment setting | Unset default | Activation / effective range / prerequisite |
|---|---|---|
| `AFM_PREFIX_REPLAY_BOUNDARIES` | Off | `1`; serial AR + prefix caching + supported text Qwen exact-boundary cache; not VLM/KV-quantized state |
| `AFM_QWEN_PLE_ROW_CACHE_MIB` | 0 / off | Integer clamped 0–64 **per immutable table**; 4 MiB screened |
| `AFM_QWEN_PLE_ROW_CACHE_STATS` | Off | `1`; diagnostic counters every 1024 gathers and normal destruction; leave off for timing |
| `AFM_QWEN_PLE_NATIVE_READS` | **On unless `0`** | Recipe value `0` selects the mapped alternative to native worker reads; not “stock default” |
| `AFM_QWEN_PLE_VECTOR_UNPACK` | Off | `1`; eligible q4 Swift mapped unpack; native worker unpack is separate |
| `AFM_QWEN_RESOLVE_MAPPED_NGRAM_TOKEN_AT_PLE` | **On unless `0`** | Existing deferred token-resolution behavior; not a newly required opt-in |

The PLE row cache stores immutable **unpacked embedding rows**, not KV state,
not speculative acceptance history and not completed answers. Prefix caching
stores request state for reuse. `AFM_QWEN_MTP_REPLAY_MIB` below budgets complete
MTP replay state. Filesystem page warming is a fourth, separate mechanism.

The CLI residency values are exactly `mapped` (default) and `prewarm`, not
`resident` or `locked`. Prewarming waits for the mapped table's page-cache warm
to finish; those pages can still be reclaimed. Do not imply it pins the model.

### MTP ownership, width and projection

| Environment setting | Unset default | Activation / effective range / prerequisite |
|---|---|---|
| `AFM_QWEN_MTP_VERIFICATION_POLICY` | Strict singleton-equivalent | `batched` opts into the faster arithmetic schedule (`fast`/`approximate` aliases); unknown/unset selects strict |
| `AFM_QWEN_MTP_SCHEDULER` | Off | `1` + supported concrete Qwen model + available MTP generator + concurrent scheduler lane |
| `AFM_QWEN_MTP_REPLAY_MIB` | 0 / off | Integer clamped 0–4096; requires owner **and** prefix caching |
| `AFM_QWEN_MTP_REPLAY_MAX_TOKENS` | 4096 | Prompt eligibility only; integer clamped 0–8192, invalid/unset 4096. Requires replay cache enabled; 0 retains no nonempty prompts. Does not increase byte or entry limits. |
| `AFM_QWEN_MTP_SHARED_VERIFY` | Off | `1` + owner; actual sharing requires eligible batched-policy sessions |
| `AFM_QWEN_MTP_INDEPENDENT_ATTENTION` | Off | `1` + shared verifier; request-owned real-offset attention in shared backbone groups |
| `AFM_QWEN_MTP_SUBMISSION_WINDOW` | Requested 1 | Owner: clamp 1–4 independently, **2–4** if shared, **2–8** if shared + independent attention. Thus shared-on/unset effectively means 2, not 1. No owner means 1. |
| `AFM_QWEN_MTP_SHARED_VOCAB` | Off | `1` + shared owner; eligible 2–4 requests, at most 16 request/token rows; otherwise independent projection |
| `AFM_QWEN_MTP_SHARED_HEAD` | Off | `1` + shared owner and batched policy; 2–8 compatible heads, private attention; repair width at most 4; retained anchor uses fallback |
| `AFM_QWEN_MTP_ADAPTIVE_DEPTH` | Off | `1` + owner and batched policy; request-local 1…min(requested depth, 8), not automatic MTP-off |
| `AFM_QWEN_MTP_COHORT_DEPTH` | Off | `1` + shared verifier and batched policy; common depth by active-width band, 32 measured owner steps per epoch; takes precedence over request-local adaptive depth |
| `AFM_QWEN_MTP_STATE_REMAP` | Off | `1` + positive persistent-state budget; reuse certified UUID/revision rows across group ordering/membership changes; attention remains private |
| `AFM_QWEN_MTP_PERSISTENT_STATE_MIB` | 0 / off | Integer clamped 0–2048; independent-attention shared verifier; up to four revision-guarded fixed-state banks; not attention/prefix/answer caching |
| `AFM_QWEN_MTP_RETAIN_ANCHOR` | Off | `1` + batched policy and qualified trimmable head state; strict mode ignores it |

Window 8 does **not** mean MTP depth 8, eight copies of the model, or server
capacity 8. Expanded groups support 5–8 requests with at most 3 drafts / 4
verification tokens each, and at most 32 request/token rows. Deeper sessions
retain smaller grouping. Equal-position mode stays capped at four. Actual
group width depends on compatible ready work; look at counters, not just flags.

### Graph/kernel controls pinned by the recipes

| Environment setting | Unset default | Recipe / experiment meaning |
|---|---|---|
| `AFM_QWEN_VERIFY_QMM` | Off | `1` + batched verification; small-row quantized verifier projection |
| `AFM_QWEN_VERIFY_ATTENTION_CHUNK` | 1 | Integer clamped 1–2; recipe 2; **not** prompt prefill chunk size |
| `AFM_QWEN_VERIFY_FUSED_HC` | Off | `1`; eligible fused verification HyperConnections |
| `AFM_QWEN_HC_NATIVE_CHAIN` | Off | `1`; native-arithmetic HC chain. Source checks `== "1"`, despite an older comment suggesting promotion. |
| `AFM_QWEN_VERIFY_FUSED_ROUTER` | Off | `1`; eligible fused verification routing |
| `AFM_QWEN_VERIFY_ASYNC_LADDER` | 0 / off | Nonnegative layer stride; singleton verifier recipe 8 |
| `AFM_QWEN_MTP_DRAFT_ASYNC_LADDER` | 0 / off | Positive draft dispatch stride; recipe 1 |
| `AFM_QWEN_VERIFY_SHARED_ASYNC_LADDER` | 0 / off | Nonnegative layer stride for eligible shared verification; 2/4/8/16 screened; M/W8 recipe 4, separate W8-L2 experiment 2 |
| `AFM_QWEN_VERIFY_SHARED_COMPILED_TAIL` | Off | `1`; supported BF16/model-owned compile geometry and ordinary non-deferred shared tail; recipe 0 |
| `AFM_QWEN_VERIFY_DEFER_HC` | Off | `1`; separate verifier HC deferral experiment; recipe off, not additive to compiled ordinary tail |
| `AFM_QWEN_FUSED_QK_NORM_ROPE` | **On unless `0`** | Keep on for the banked AR preset; disabling bypasses its fused prerequisites |
| `AFM_QWEN_QSA_FUSED_DECODE_ATTENTION` | Off | `1` selects a different fused QSA decode path, which bypasses request banking |
| `MLX_SDPA_BLOCKS` | No override | A positive override bypasses the custom banked dense specialization; leave unset in A |

Earlier retained diagnostic experiments are also tracked rather than silently
enabled: `AFM_QWEN_MTP_STATE_SNAPSHOTS=1`,
`AFM_QWEN_VERIFY_QSA_RADIX=1` and `AFM_QWEN_VERIFY_FUSED_MASK=1` are off when
unset. Their earlier results are in
[the MTP investigation history](QWEN_NEXT_MTP_PARITY_PROGRESS.md).
The snapshot constructor also accepts an explicit model-owned setting, which
takes precedence over its environment fallback. None of these variables is
part of recipe A/M/W8; no new combined qualification is implied.

`AFM_PERF=1`, `AFM_DEBUG=1` and `AFM_QWEN_PROFILE_HOST=verify` are diagnostic
activation examples, not speed presets. Leave them unset for timed comparisons.
Host laps include existing waits and are not individual GPU kernel durations.

## Compatibility / combinations that must not be conflated

| Combination | Effective behavior / qualification |
|---|---|
| MTP owner + AR mixed positions / banked-attention flags | The AR request-owned adapter is excluded by the MTP owner. Flags do not turn the separate MTP verifier into banked AR attention. |
| MTP owner + prefill interleave | MTP interleaving remains excluded. Setting the flag does not make it work. |
| Shared flags + strict policy / unsupported request | Eligible-path fallback, not proof of shared execution. Preserve strict behavior; inspect actual shared-group counters. |
| Banked attention + fused QSA decode | Fused QSA path bypasses request banking. Not two additive gains. |
| Banked dense attention + positive SDPA block override | Native partition-override fallback, not the measured custom bank family. |
| Shared verification + no independent attention + requested window 8 | Clamped to four; does not activate the wider experiment. |
| W8 + shared vocabulary | Measured at ladder 4 with mixed quality; vocabulary only shares eligible smaller subgroups, not the >4-request groups. W8-L2 + vocabulary is still untested. |
| PLE row cache + prefix replay | Different data/lifetimes; conceptually compatible, but latest W8/A9 combinations have not been jointly requalified. Do not add the individual gains. |
| Prefix off + positive MTP replay budget | Replay cache not created; prefix-off repeated prompts are not replay hits. |
| C1 + concurrent shared settings | No concurrent group to accelerate; serial C1 replay and scheduler-owned replay are distinct. |
| Prefill step 8192 + admission budget 1024 | GPU work per prompt pass versus soft incoming uncached-token budget. A large head prompt can exceed the latter estimate. Neither is verifier chunk 2. |
| VLM, quantized KV, another model/checkpoint | Not qualified by these concrete text-Qwen experiments. Use existing fallbacks; do not copy throughput claims. |
| Sampled requests + latest greedy performance numbers | Functional support is not a throughput match; acceptance and output lengths change with prompt and sampling. |

## Removed / rejected variants

| Variant | Current disposition |
|---|---|
| `AFM_QWEN_MTP_ATTENTION_PROJECTIONS` | Removed, **not wired**. Do not confuse it with retained `AFM_QWEN_BATCH_ATTENTION_PROJECTIONS` for AR. [Investigation](QWEN_NEXT_MIXED_POSITION_VERIFICATION.md). |
| Initial one-pass request-banked attention | Replaced by the native-arithmetic implementation; same surviving flag does not recreate the rejected kernel. |
| Grid-z verifier request banks / packed-row verifier projection | Rejected patches retained as evidence, not supported activation recipes. [History](SHARED_BATCH_EXECUTION_WORKSTREAM.md#rejected-grid-z-verifier-request-banks-2026-09-12). |
| Independent-row QMM experiment | Rejected; do not confuse it with retained singleton `AFM_QWEN_VERIFY_QMM`. [History](SHARED_BATCH_EXECUTION_WORKSTREAM.md#rejected-independent-row-qmm-screen). |
| Locked n-gram residency | Not an accepted CLI value. Supported choices remain mapped/prewarm. |

## Evidence and maintenance contract

Source of effective defaults/guards:

- [Scheduler owner, budgets and lane selection](../Packages/AFMKitMLX/Sources/AFMKitMLX/Models/BatchScheduler.swift).
- [Verification policy](../Packages/AFMKitMLX/Sources/AFMKitMLX/AFMMLXMTPRuntimePolicy.swift).
- [Model graph and shared-verifier geometry](../vendor/MLX/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4Exp.swift).
- [Mapped sidecar / row cache / residency](../vendor/MLX/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4ExpMappedNGramTable.swift).
- [Attention bank selection](../vendor/MLX/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4ExpRequestAttentionBatch.swift).

External, untracked evidence root:

```text
/Volumes/edata2/afm-benchmarks/qwen-next-mtp-parity-20260909
```

The frozen `VERIFICATION-WIDTH-20260913-SHA256SUMS.txt` inventories 452
files. The follow-up `COMPOSED-VERIFIER-20260913-SHA256SUMS.txt` separately
inventories the new timing arms, lifecycle runs, comparisons and test logs.
Older manifests and helper scripts are immutable evidence. Do not
edit a frozen runner or relabel an old result as a new combination.

`SHARED-WORK-20260913-SHA256SUMS.txt` freezes **630 additional/current-workstream
files**, including timing/lifecycle records, comparisons, build/test logs,
the failed attempt and executable-byte checkpoints. Its SHA-256 is
`0e9a101a527a517841e31b199c924abdfcbe367710a29a18a4e2d109f55c1002`.
All 630 entries verified, as did the unchanged preceding 433-file and 452-file
manifests. Some frozen shared workload inputs are intentionally also inventoried;
630 is a manifest-entry count, not a count of new tests.

Each new experiment must update this matrix **in the same workstream commit**:

1. Assign a profile ID; record the full CLI/environment and the exact one-variable
   delta from its control. State missing/inherited overrides explicitly.
2. Record provider/consumer revisions, actual AFM binary SHA-256, checkpoint
   identity/path, machine, prompt fixture, output budget, sampling, concurrency,
   prefix/replay/row-cache configuration, and warmup policy.
3. Save actual path-activation counters, peak RSS, cached tokens, first/repeat
   aggregate tokens/s, valid tasks/s, TTFT/stream gaps, and completion counts.
4. Keep runtime assertions, structural checks and semantic judging separate.
   Record changed texts, truncations, omissions and excluded/contended attempts.
5. Mark **measured**, **untested**, **fallback/inactive**, or **rejected**. A
   source-compatible combination is not automatically a measured one. Numerical
   or quality tradeoffs require user discussion before changing defaults.
6. Link the evidence report and frozen manifest; retain the prior control.
   Continue on PR #123, rather than opening one PR per matrix row.

Updating this matrix does not change runtime defaults or imply background
benchmarks are running. Each linked report states which tests actually ran.
