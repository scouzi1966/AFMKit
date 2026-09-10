# Shared batch execution workstream

## Objective

Maximize sustainable useful aggregate throughput with bounded memory and
qualified correctness. There is no 2x ceiling. Preserve the measured
single-request fast path; discuss meaningful latency, memory or quality
tradeoffs before promoting new defaults.

## Execution order

1. Freeze same-checkpoint agentic concurrency/reference curves.
2. Share complete, exact-boundary replay handling between execution paths.
3. Introduce scheduler-owned speculative sessions with bounded draft/verify/
   commit operations, not one blocking whole-response generator.
4. Batch compatible per-slot operations, including variable-position state.
5. Admit new slots continuously and interleave bounded prefill work.
6. Tune speculation and CPU/GPU overlap using measured useful work per round.
7. Qualify adapters for other architectures while retaining existing fast paths.

## First implementation: serial replay boundaries

`MLXReplayPrefill` provides bounded checkpoint planning and independent snapshot
storage. BatchScheduler delegates its existing planner and snapshot helper to
this implementation. An optional TokenIterator prepared-prefill hook allows
the serial service to use the same boundary scheme and seed logit processors
with the complete prompt, including the reused prefix.

`AFM_PREFIX_REPLAY_BOUNDARIES=1` enables the serial AR experiment only when
prefix caching is active, the model is `Qwen4ExpModel`, input is text-only, KV quantization is off and the
cache requires exact-boundary restoration. Snapshot before the last prompt
token, then compute that final token for next-token logits. Do not save a
post-generation recurrent cache under an earlier prompt boundary. Models with
additional `LMOutput.State` do not have that state serialized by this helper;
such outputs are deliberately not inserted into radix snapshots.

This is not yet the complete shared state contract, MTP replay integration,
continuous MTP admission, or multi-request verification. The experiment stays
off by default pending model-specific validation and cold-prefill cost checks.
Existing unsafe replay overrides are not needed and are not enabled.
The VL wrapper is excluded even for text requests because its `prepare` method
creates additional position-delta state; it needs its own qualified adapter.
Checkpoint spacing does not increase the configured prefill chunk size. Each
forward pass still obeys that limit, including on long-context cache misses.

### Initial controlled result (2026-09-10)

Same release binary `cc0c73d68cf7fed1c3b724efa0f49daec40aa388a3112cce0e816ac8f34b1f1f`,
same ddalcu four-bit checkpoint, MTP off, prefix cache on, one client. Fifteen
synthetic agentic review requests, then their exact repeats; 192 output-token
limit. Only `AFM_PREFIX_REPLAY_BOUNDARIES` differs between arms. Other Qwen
experimental flags are fixed in both arms; this is not a no-flags release result.

| Measurement | Replay off | Replay on |
|---|---:|---:|
| First round aggregate output tok/s | 46.93 | 59.98 |
| Repeat round aggregate output tok/s | 46.80 | 65.06 |
| First round median TTFT, seconds | 1.044 | 0.266 |
| Repeat round median TTFT, seconds | 1.038 | 0.040 |
| Reused tokens, first / repeat | 0 / 0 | 15,479 / 17,112 |
| Peak process RSS, GiB | 67.61 | 67.49 |
| Completed responses / JSON checks | 30 / 24 | 30 / 24 |

The first round follows one excluded warmup, so it is not a completely cold
cache. Every repeat matches its corresponding first-round text within its own
arm. None of the 15 responses is byte-identical across the two arms. Different
prefill execution shapes can change greedy wording; this observation alone
does not establish that rounding is the only cause. All six JSON failures in
each arm terminate at the token limit, on the same task type. These structural
checks are not a semantic quality judge. A later build adds a long-prefill
chunk-size guard and the Qwen-only activation guard; the hash above identifies
the initial measured build, not that later build.

Raw requests, responses, launch flags, memory samples and summaries remain
untracked in `/Volumes/edata2/afm-benchmarks/qwen-next-mtp-parity-20260909/`,
under `shared-replay-{off,on}-v1-afm-mtp-0/`.

Repeat with the chunk-size guard (binary
`d0625e41d3382d7a340460aa859b01e6c67417b3bffa4ce5d46e9789a036de2c`):

| Measurement | Replay off | Replay on |
|---|---:|---:|
| First / repeat aggregate tok/s | 47.97 / 47.84 | 61.79 / 66.88 |
| First / repeat median TTFT, seconds | 1.030 / 1.029 | 0.262 / 0.038 |
| Peak process RSS, GiB | 67.60 | 67.49 |

Cached token totals, completion counts, truncations and structural check totals
were unchanged. Every response matched the corresponding first-run text in
its own arm. All 51 focused regression tests passed; one opt-in production
geometry latency probe was skipped. These cover replay snapshot immutability,
small chunk sizes, single-token prompts, planning bounds, MTP pipeline and
cache-selection policy, not complete long-context release qualification.
After this repeat, a final guard-only change excludes the VL wrapper because
its prepare-created continuation state is not represented by this adapter.
The measurements above refer to the identified pre-guard binary, not a new
performance run of the final artifact.

Final-artifact smoke (SHA-256
`794eee0d6f391e59304213556a889c81f0a4b98839e03ee73c2dbd4d20f1b381`)
passed streaming and non-streaming cold/repeat pairs with logprobs and a
nonzero presence penalty. Each response was `READY`; repeats restored 24/25
and 22/23 prompt tokens respectively. Raw records are in
`shared-replay-api-v3-afm-mtp-0/`. This four-request smoke is not a replacement
for comprehensive API or quality qualification.

## Shared immutable lookup cache (proposed experiment)

The user's "engram cache" suggestion is useful as a separate immutable-data
optimization. PLE currently maps the sidecar and may warm filesystem pages,
but does not retain decoded rows between gathers.

Proposed stages: deduplicate row IDs within a scheduled batch; coalesce misses;
test a bounded cache of unpacked rows scoped to the immutable table instance.
Use table/model identity plus row ID, not row ID alone in a global dictionary.
Avoid disk IO or GPU synchronization under a shared cache lock. Record hit
rate, unpacked bytes avoided, lookup overhead, eviction and added resident
memory. Keep request-specific n-gram history outside the shared cache. Other
architectures can reuse the immutable-row cache mechanism only where they have
a suitable lookup table; this is not a generic replacement for KV/recurrent
state or a claim that all models have PLE.

### Text-derived reuse estimate

An offline CPU-only study retokenized the 15 saved replay-enabled outputs
(2,503 tokens). It simulated a cold LRU of decoded row groups, eight
head-specific rows per bigram/trigram, 160 BF16 values per row. The first two
output tokens were excluded because their history depends on the prompt.

| Payload budget | Serial request ordering | Synthetic 15-slot interleaving |
|---|---:|---:|
| 1 MiB | 30.2% estimated hits | 53.8% estimated hits |
| 4 MiB | 56.5% | 57.8% |
| 16 MiB | 57.9% | 57.9% |

Synthetic per-step deduplication alone found 32.5% repeated lookup groups.
This warrants an opt-in implementation experiment, not a speedup claim.
The study uses reconstructed tokens and artificial interleaving, not actual
generated-token or scheduler traces. It omits prompt seeding, EOS transitions,
MTP draft/repair traffic, hash collisions, metadata memory, copying and locking
cost. Real workloads can differ substantially. A 58% row-cache hit rate would
**not** imply a 58% end-to-end throughput gain, especially given the earlier
PLE unpack microbenchmark's limited full-decode benefit.

The initial implementation should use a table-instance-scoped cache and a
fixed byte budget, preserve exact BF16 row bytes, and keep disk reads and
dequantization outside metadata locks. Row history remains per request. Do
not serialize requests behind a global cache lock or change the default
residency policy. Qualify warm/cold storage, diverse and shared-prefix agent
traffic, MTP off/on, concurrency 1/15, and cached-prefix bypasses separately.
Retain it only if measured end-to-end benefit justifies its added memory and
lookup overhead. Raw study: `ple-reuse-text-estimate.json` beside
`study_ple_reuse.py` in the artifact root.

## Matched concurrency reference (2026-09-10)

Frozen reference v26.9.2, same ddalcu checkpoint and synthetic agentic requests.
MTP uses the reference's adaptive policy; other speculation is disabled. Both
engines use greedy temperature 0 / top-p 1 requests with a 192-token cap.
Prefix capacity is 64 entries when enabled; the reference also has an internal
hybrid-cache byte budget, so entry counts alone do not equate residency.

| MTP | Prefix | Clients | First / repeat aggregate tok/s | Peak RSS GiB |
|---|---|---:|---:|---:|
| Off | On | 1 | 45.91 / 49.94 | 66.76 |
| Off | On | 15 | 83.09 / 152.71 | 67.93 |
| Off | Off | 15 | 85.63 / 84.53 | 67.96 |
| On | On | 1 | 56.06 / 57.54 | 69.15 |
| On | On | 15 | 55.77 / 60.29 | 69.72 |
| On | Off | 15 | 56.81 / 56.44 | 69.01 |

AR data is from `agentic-reference-six-20260910-*`; MTP data is from the
uncontended `agentic-reference-clean-20260910-*` repeat. The initial MTP runs
overlapped compilation and are exploratory only. All 180 responses completed;
JSON structure checks were not universally successful and are not a semantic
quality certification. Throughput counts actual emitted tokens, includes
queue/prefill time and is not normalized for different response lengths.
These results do not establish AFM concurrent parity: shared replay above
qualifies only the one-client AR experiment. True batch execution and bounded
speculative sessions remain the main implementation work.

## Evidence policy

Record end-to-end aggregate output throughput separately from decode-only
throughput, and report actual cached tokens, request queue time and memory.
Concurrency enabled does not prove batched execution. MTP enabled does not
prove every API feature is speculative-eligible. Truncated unconstrained JSON
is not automatically an engine defect. No benchmark run overlapping a build
qualifies as an uncontended reference baseline.
