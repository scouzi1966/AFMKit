# Qwen Next opt-in activation and experiment matrix

Last source audit: **2026-09-13**, AFMKit runtime `a1255dc6`, paired consumer
`9acccfc`. Workstream: [PR #123](https://github.com/scouzi1966/AFMKit/pull/123).
This is the central index of settings and combinations for the Qwen Next
optimization project. Update it with every new experiment, removal or result.
Historical reports remain the evidence; this index does not rewrite them.

Inventory audit: **40 named Qwen controls, 39 wired and one removed**. This
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

Latest measured runtime SHA-256 (`a1255dc6`, width and composed-verifier experiments):
`777a3a8708dc8a650e1fe9773caa2aa6fa42030be4f47e8aac486cb47469ed3d`.
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
  AFM_QWEN_MTP_SUBMISSION_WINDOW=8 \
  AFM_QWEN_MTP_SHARED_VERIFY=1 \
  AFM_QWEN_MTP_INDEPENDENT_ATTENTION=1 \
  AFM_QWEN_MTP_SHARED_VOCAB=0 \
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
| M13 | Shared verifier + non-greedy requests (for example temperature 0.6/top-p 0.95) | Per-request sampling/lifecycle coverage exists, including mixed greedy/sampled groups. Latest full greedy matrix is **not** a sampled-performance matrix. |
| M14 | W8 + replay off, prefix off, other client counts, longer contexts, or PLE row cache | Controls are available where their guards allow. **Latest window-8 throughput/quality matrix not run for these combinations.** Do not extrapolate C15/prefix-on results. |
| M15 / W8-L2 | W8 with `AFM_QWEN_VERIFY_SHARED_ASYNC_LADDER=2`, vocabulary stays `0` | **Measured**, separate opt-in: repeat 117.46–119.21 tok/s, +2.35–5.39% over matched ladder-4 arms; first-phase token rate −1.10–1.45%. Structural 60/60 versus 59/60, but changed outputs and mixed latency. [Evidence](QWEN_NEXT_COMPOSED_VERIFIER_EXPERIMENTS.md#window-8-submission-every-two-versus-four-layers). |

## Six-mode coverage ledger

The user-requested cross product is MTP off/on × these three scenarios. It
remains an explicit checklist, not “cache tested” inferred from one warm run.

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
| `AFM_QWEN_MTP_SHARED_VERIFY` | Off | `1` + owner; actual sharing requires eligible batched-policy sessions |
| `AFM_QWEN_MTP_INDEPENDENT_ATTENTION` | Off | `1` + shared verifier; request-owned real-offset attention in shared backbone groups |
| `AFM_QWEN_MTP_SUBMISSION_WINDOW` | Requested 1 | Owner: clamp 1–4 independently, **2–4** if shared, **2–8** if shared + independent attention. Thus shared-on/unset effectively means 2, not 1. No owner means 1. |
| `AFM_QWEN_MTP_SHARED_VOCAB` | Off | `1` + shared owner; eligible 2–4 requests, at most 16 request/token rows; otherwise independent projection |
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
