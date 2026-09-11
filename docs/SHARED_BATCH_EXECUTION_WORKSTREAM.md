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

## Delivery status

These are branch experiments, not production-default or release qualifications.

| Plan item | Implemented / measured | Still required |
|---|---|---|
| Same-checkpoint baseline matrix | Six AFM/reference configurations and saved raw responses; integrated six-mode checkpoint repeated | Wider concurrency curve and workload coverage |
| Shared exact-prefix replay | Serial AR and scheduler boundary helpers; opt-in API checks | Broader quality, long-context and model-switch qualification |
| Qwen MTP replay | Opt-in complete exact-prompt target/head/history snapshots; focused and lifecycle tests pass | Working-set/memory qualification, partial-prefix continuation and serial-lane reuse |
| Scheduler-owned Qwen MTP sessions | Opt-in streaming scheduler integration; mixed MTP/AR lifecycle and six-mode aggregate screen pass | Staged multi-request verification and wider qualification |
| Genuine GPU batches | Persistent equal-offset subgroups with row-removal tests | Arbitrary-position batches and multi-request verification |
| Continuous admission | Opt-in independent/group ownership; burst and staggered measurements | Avoid fragmentation across arbitrary arrival/position patterns |
| Prefill/decode interleaving | Soft uncached-token admission budget across whole prompts | Suspend/resume individual prefills at token-chunk boundaries |
| Adaptive speculation | Existing fixed-depth Qwen path retained | Workload-aware depth and useful-token cost policy |
| CPU/GPU overlap | Existing single-request overlap retained | Cross-slot scheduling after bounded sessions exist |
| Optional immutable PLE row cache | Bounded cache, exact-bit tests, limited measured benefit | Repeats before any default proposal; not the main throughput lever |
| Batch kernels and graph overhead | Existing fused kernels retained; independent-row QMM screened and rejected | Profile remaining batch hot paths |
| Memory budgets and reclamation | Group ownership/filtering and row-cache budget tested; RSS recorded | Long-context concurrency soak and request-memory admission budgets |
| Qualification and cross-model reuse | Qwen-only guards and focused tests; lifecycle API smoke and six-mode matrix | Semantic/long-context qualification, then model-specific adapters |

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

## Shared immutable lookup cache (opt-in prototype)

The user's "engram cache" suggestion is useful as a separate immutable-data
optimization. PLE currently maps the sidecar and may warm filesystem pages,
but does not retain decoded rows between gathers.

The prototype deduplicates row IDs within a gather, coalesces misses, and
uses a bounded cache of unpacked rows scoped to the immutable table instance.
Use table/model identity plus row ID, not row ID alone in a global dictionary.
Avoid disk IO or GPU synchronization under a shared cache lock. Record hit
rate, unpacked bytes avoided, lookup overhead, eviction and added resident
memory. Keep request-specific n-gram history outside the shared cache. Other
architectures can reuse the immutable-row cache mechanism only where they have
a suitable lookup table; this is not a generic replacement for KV/recurrent
state or a claim that all models have PLE.

`ImmutableRowCache` uses fixed four-way sets with age-based replacement. Its
budget includes row payload and fixed per-slot tag/age arrays, but excludes
constant object headers and temporary per-gather buffers. Lookup and insertion
locks cover at most 64 rows at a time. Decoding misses occurs outside the lock;
two independent callers may decode the same simultaneous cold miss to avoid
waiting behind file IO. Exact bits are copied into request-owned output.
No borrowed cache pointers reach the GPU. The row cache dies with its table.

`AFM_QWEN_PLE_ROW_CACHE_MIB=4` enables a 4 MiB experiment; unset/0 disables it.
Values are bounded at 64 MiB per table. `AFM_QWEN_PLE_ROW_CACHE_STATS=1` emits
cumulative diagnostic counters every 1,024 gathers and at destruction when
the table is released normally. Leave diagnostic logging unset for timing.
The cache-off transport selection and arithmetic remain unchanged.

Eight new tests cover isolation, caller mutation, miss coalescing, eviction,
failed decoders, budget bounds, concurrent users, non-blocking slow misses and
mapped-table gather parity. The focused Release run passed 61 tests with two
optional microbenchmarks skipped. End-to-end performance qualification is
separate; hit rate does not establish a speedup.

Priority remains genuine GPU batching and scheduler-owned speculative work.
This suggestion gets a bounded A/B screen, not an open-ended tuning campaign
or a new performance target. The earlier 3.4x unpack microbenchmark without a
material full-decode win is reason to be cautious about expected impact.

### Bounded same-binary screen (2026-09-10)

Release SHA-256 `e7b3c67961604d897c6d048c51b8f381b1fec99ab304dc1da78443e27fb26f7c`,
same ddalcu checkpoint, prefix caching enabled, 15 clients, 192-token cap.
AR runs cache off/on; MTP runs on/off, with no concurrent compilation.
Only the row-cache budget differs within each pair; the other experiment
flags are fixed. This is not a no-flags release qualification.

| MTP | Row budget | First / repeat aggregate tok/s | Peak RSS GiB |
|---|---:|---:|---:|
| Off | 0 MiB | 61.11 / 65.01 | 67.72 |
| Off | 4 MiB | 61.53 / 66.45 | 67.68 |
| On | 0 MiB | 57.91 / 58.45 | 69.41 |
| On | 4 MiB | 58.50 / 58.66 | 69.42 |

All 60 matched off/on response pairs have identical text and output token
counts. Each arm completed 30 requests with 24 JSON/identity passes and six
token-limit truncations. These are structural checks, not a semantic judge.
The observed 0.4–2.2% gains are small and need repeats to establish significance;
the prototype remains off by default. Do not delay the scheduler work for
additional row-cache tuning. Records: `ple-row-cache-screen-*` in the artifact
root. Debug cache counters were disabled during timing.

## Compatible uniform decode groups (opt-in prototype)

`AFM_QWEN_BATCH_COMPATIBLE_GROUPS=1` allows text-only Qwen4ExpModel requests in
an independent mixed-offset cohort to form persistent compatible subgroups.
Compatibility uses every layer's concrete type, actual offset, metadata,
tensor shape and dtype, not just prompt length. Unknown state, multimodal
inputs and other models are excluded. The existing all-uniform fast path is
unchanged. This does not add continuous admission or scheduler-owned Qwen MTP.

`UniformDecodeGroup` retains one merged cache for its members across steps.
Slots release obsolete decode caches, while immutable radix snapshots remain
separate. Completed/cancelled rows are filtered by stable request identity;
one-row survivors keep their current state. Periodic graph materialization
includes group caches, and shutdown releases them. Sampling, grammar state,
stops, logprobs and token dispatch remain per request in original slot order.
Admission diagnostics report group widths; completion diagnostics count actual
group forwards and slot steps.

### Initial subgroup API screen (2026-09-10)

Same release binary in all four arms, SHA-256
`8dbc950c4ce76d103daed3f38a9a57a506b65800a0eb88a706b94adcd412f0ed`.
Same checkpoint, prompts, greedy settings, 15 clients and 192-token cap as the
matrix above. Row cache disabled. Prefix-enabled arms ran grouping off/on;
prefix-disabled arms ran on/off. No compilation overlapped measurement.

| Prefix | Grouping | First / repeat aggregate tok/s | Peak RSS GiB |
|---|---|---:|---:|
| On | Off | 61.30 / 67.08 | 67.70 |
| On | On | 103.69 / 116.98 | 67.61 |
| Off | Off | 48.12 / 48.21 | 67.75 |
| Off | On | 68.87 / 68.00 | 67.63 |

This screen improves aggregate throughput about 69–74% with prefix reuse and
41–43% without it. The previous matched reference reached 152.71 tok/s on the
prefix-enabled repeat: aggregate parity is not achieved. These are complete
phase output tokens divided by wall time, not a sum of individual decode rates.

All 120 requests completed. Prefix arms retain 24/30 JSON/identity passes and
the same 15,479/17,112 cached tokens. Without prefix, grouping changes 24/30 to
26/30 structural passes. That is not a broad semantic quality certification.
Thirteen of fifteen first-round prefix responses change wording between arms;
representative outputs are coherent, but changed arithmetic is not proof that
every difference is harmless. Both features remain opt-in. Group telemetry
confirms real merged forwards: prefix first-round 699 forwards service 2,344
slot steps, with group widths 3/5/3/3 and remaining independent work.

The focused Release suite passes 53 tests, including actual Qwen forward-state
comparisons while removing the middle and first rows, snapshot immutability,
geometry/offset/type gating, stale-cache avoidance, and existing MTP regressions.
The grouping-disabled control retains all 30 preceding PLE-cache-off control
responses. Raw records: `compatible-group-screen-*` in the artifact root.

The `compatible-group-safety-v2` API smoke passes 255/255 checks across 45
requests: three early cancellations, mixed 8/24/64-token limits, sampled
temperature 0.6/top-p 0.95, request-local seeds and presence penalty, alternating
logprobs or stop sequences, followed by two further request rounds. Group
telemetry confirms merged execution in every round; repeated requests reused
17,022 tokens. The first `compatible-group-safety` run passed the serial
fallback combination (stops plus logprobs), but does not qualify grouped decode.
This coverage distinction is intentional and its raw records are retained.
Long-context/soak tests and broader semantic qualification remain outstanding.

Reverse-order prefix/C15 repeat on the same binary confirms the effect:
grouping off first/repeat **62.03 / 66.08 tok/s**, grouping on
**104.31 / 117.34 tok/s**. Each arm again completed 30 requests with 24
structural passes and unchanged cached-token counts. Records:
`compatible-group-repeat-g{0,1}-afm-mtp-0/`.

Sparse-attention follow-up: the expanded agentic prompt contains roughly
4,096 input tokens. At the original 192-token cap, grouping off measured
52.07 / 58.74 aggregate tok/s (first/repeat); grouping on measured
84.19 / 102.68. Peak RSS was 67.55 / 67.56 GiB. Every request completed and
every response repeated identically within its own arm. However, grouping
changes 30/30 structural passes to 26/30: two responses per round reach the
192-token cap. This fixed-cap difference remains part of the qualification
record; it is not erased by a larger-cap rerun.

Separate 384-token-cap diagnostic: both paths pass 30/30 structural checks.
Grouping on measures 82.54 / 101.56 tok/s; grouping off 52.32 / 59.33.
The previously truncated tenant-cache response completes at 204 tokens.
The stream-task response changes wording in the rerun and completes cleanly;
not every new response is an exact continuation of the truncated text.
These checks establish completed JSON, not broad semantic or greedy equivalence.
Records: `compatible-group-sparse-*` and `compatible-group-sparse-quality-*`.
The early generic `launch.json` metadata retained the context harness's
128-token default; the actual agentic request code used 192, as shown by its
saved completions. Future runs explicitly record the effective request cap and
sampling parameters; existing evidence is not rewritten.

## Continuous group admission (separate opt-in experiment)

`AFM_QWEN_BATCH_CONTINUOUS_GROUPS=1` additionally requires compatible grouping
to be enabled. It bypasses the fixed-cohort admission barrier only for the
qualified text Qwen adapter. Every request, including the initial singleton,
keeps independent/group-owned caches, so later arrivals cannot accidentally
enter the legacy padded batch representation. Existing groups are never
padded or remapped to a new request's offset.

The first version admits at most one available-slot prefill before returning
to decode. The follow-up below adds a soft uncached-token budget across arriving
prompts; full token-chunk prefill/decode interleaving remains outstanding.
New arrivals with incompatible positions remain independent; arbitrary-offset
multi-request verification is also not implemented by this switch. Both
experiments stay off by default pending throughput/latency and quality review.

### Initial continuous-admission screen (2026-09-10)

Release SHA-256 `219006b93856e75a0d3bd20ff2ab86e96fc9e6e46b840c6dcf935af27baaa65a`.
Same checkpoint, flags, C15, prefix enabled, MTP off and 192-token cap.
The burst comparison measures 101.39 / 114.10 aggregate tok/s with continuous
admission off, and 107.83 / 127.48 on. Both arms complete 30 requests with
24 structural passes, identical reused-token totals and approximately 67.57
GiB peak RSS. The extra request joins while 14 slots are active, before their
first grouped forward; this is not yet proof of a late-arrival throughput win.

The separate genuinely staggered workload starts eight requests, then seven
more after an initial request emits three nonempty chunks:

| Admission | First / repeat aggregate tok/s | Late median TTFT, seconds | Peak RSS GiB |
|---|---:|---:|---:|
| Fixed cohort | 78.87 / 87.36 | 14.20 / 12.25 | 67.58 |
| Continuous, one prefill per turn | 72.41 / 81.45 | 6.56 / 6.09 | 67.68 |

Both paths complete 30 requests with 28 structural passes and equal cached
token totals. Continuous admission reduces late TTFT but costs 7–8% aggregate
throughput, consistent with its one-at-a-time admissions fragmenting new GPU
groups. It is **not** a default recommendation. Next experiment: bounded
multi-request admission for cheap cache hits while preserving decode fairness.

Timestamp audit counts only initial requests already streaming when the late
group was submitted: fixed-cohort late requests overlap 0 such requests;
continuous late requests overlap active initial work in all 7 cases each round.
The raw summary's looser `late_streamed_before_initial_drained` field counts
an eighth initial request that was itself queued and is not a reliable overlap
test. Raw per-request timestamps and scheduler admission logs are retained.

The continuous-enabled lifecycle smoke passes 255/255 checks, with actual
grouping and admission telemetry. Together with 55 focused Release tests,
these cover the implemented guards and lifecycle, not arbitrary-position
batching, bounded speculative sessions, long-context quality or soak testing.
Records: `continuous-group-screen-*`, `continuous-group-staggered-*`, and
`continuous-group-safety-*` in the artifact root.

### Budgeted incoming cohorts

With continuous groups enabled, `AFM_QWEN_BATCH_PREFILL_TOKEN_BUDGET` controls
a soft per-turn admission budget (default 1024; bounded to 1–8192). Setting 1
reproduces the previous one-prompt admission. This has no effect when the
continuous experiment is disabled. The queue stays FIFO; a large first prompt
is admitted alone so it cannot starve. Exact-boundary radix lookup estimates
uncached token work outside the queue lock; ordinary prefill still validates
and restores state. A failed restore or oversized head can exceed this soft
estimate. This is not a hard latency bound or chunked-prompt suspension.

Same-binary staggered screen, SHA-256
`050436205b7df63ef53d88290871557d308f44c2499d09337889d164268797ce`:

| Token budget | First / repeat aggregate tok/s | Late median TTFT, seconds | Peak RSS GiB |
|---|---:|---:|---:|
| 1 | 71.05 / 80.10 | 6.93 / 5.76 | 67.63 |
| 1024 | 73.89 / 89.61 | 6.40 / 2.90 | 67.61 |

Both arms complete 30 requests with 28 structural passes and equal reused-token
totals. Logs confirm multi-request admission under the larger budget. The repeat
round begins emitting with only three initial requests visible versus eight in
the control, so arrival/cohort timing varies: the 12% repeat gain is promising,
not a stable throughput claim. Every late request overlaps already-streaming
initial work in both arms. The new lifecycle smoke passes 255/255 checks and
57 focused Release tests pass. Records: `budgeted-group-staggered-*` and
`budgeted-group-safety-*`. Default production behavior is unchanged.

Reverse-order repeat after removing the projection experiment below, final
artifact `fce3015516659fcafe0d9086e69df8c0517a21687ca11f4c4e18c49430973cb1`:

| Token budget | First / repeat aggregate tok/s | Late median TTFT, seconds | Peak RSS GiB |
|---|---:|---:|---:|
| 1 | 72.95 / 77.90 | 6.42 / 5.64 | 67.60 |
| 1024 | 77.36 / 86.94 | 6.23 / 5.30 | 67.57 |

All eight initial requests were already streaming at late submission in each
round this time. All seven late requests overlap that initial work. Completion,
structural and cached-token totals remain unchanged. This repeat supports a
roughly 12% repeat-throughput gain from budgeted admission; the much larger
initial TTFT improvement does not reproduce. It does not establish arbitrary
arrival-pattern or long-context performance. Records: `budgeted-group-repeat-*`.

## Resumable Qwen Next MTP session

`Qwen4ExpMTPGenerator.makeSession` creates request-owned target/head caches,
RNG, current target stream/position, accepted-token cursor and verification
snapshot. `nextToken` returns the already-known primary immediately, then
returns a buffered accepted token or performs at most one configured-depth
verification cycle. Head repair remains deferred until accepted tokens have
been consumed, so EOS, output caps and early cancellation do not force an
unnecessary repair. The existing whole-response `generate` method drives this
same session; there is no second production algorithm to maintain.

The session is deliberately not Sendable: use a serialized model/GPU executor.
It adds no global synchronization, cache publication or independent model
ownership. Prefill is still a whole-prompt operation. This is a prerequisite
for scheduler integration, **not** completed speculative batching or prefix
replay. Complete target/head prompt snapshots and staged batched verification
remain outstanding.

The focused Release suite passed 62 tests, with one optional microbenchmark
skipped. Added coverage checks uneven interleaving of two requests under
greedy/sampled strict/batched policies, independent RNG, EOS/length without an
extra cycle, cancellation/deallocation and mapped PLE history interleaving.

Frozen API control: same ddalcu checkpoint, C1, MTP depth 3, explicit batched
experiment settings, temperature 0/top-p 1, 192-token cap, 15 agentic prompts
and their repeats. Prefix caching is enabled but Qwen MTP does not reuse it.

| Implementation | First / repeat aggregate tok/s | Peak RSS GiB |
|---|---:|---:|
| Prior generator | 59.06 / 59.15 | 69.41 |
| Session-backed generator | 59.00 / 58.78 | 69.38 |

All 30 response texts and completion-token counts match exactly across builds;
both complete 30 requests with 24 structural passes and zero cached tokens.
The observed throughput difference is below 0.7%, not a throughput gain.
Baseline artifact: `fce3015516659fcafe0d9086e69df8c0517a21687ca11f4c4e18c49430973cb1`;
candidate: `d30a0ee94deedd3cf0826b45a6888da55aac8d86e01ef8f0503e11860602a595`.
Records: `session-refactor-{before,after}-*` in the artifact root.

### Scheduler-owned streaming sessions

`AFM_QWEN_MTP_SCHEDULER=1` enables the new lane only for the direct text
`Qwen4ExpModel`, with a loaded, matching-model MTP generator and concurrency
greater than one. All other models and the unset default retain their paths.
Transfer the existing model-identity binding into scheduler construction;
neither the generator nor mutable sessions are newly declared Sendable.

The scheduler advances independent speculative sessions through the same
token dispatcher as AR. A common private session adapter also retains the
existing GLM implementation. Qwen AR fallback admissions remain independent,
so later speculative requests cannot coerce a live dense cache. Output caps,
EOS, cancellation, stops, streaming and accounting stay with the dispatcher.
Ineligible contracts retain AR/serial fallback. Prefill remains whole-prompt,
and one session's verification is not yet merged with another's GPU forward.

Qwen speculative sessions deliberately do not read or write AR radix snapshots.
Those snapshots lack the head/target/rolling-history boundary. Cache-hit counts
remain zero for Qwen MTP rather than claiming false reuse. AR requests can
still restore and save their independently owned radix entries in this mode.

Validation: 127 focused Release tests pass, including GLM architecture and
existing admission tests. Consumer Release build passes (68.71 seconds).
The mixed live API test passes 120/120 assertions across 18 requests, plus an
excluded successful warmup: greedy/sampled MTP, logprob/presence/stop AR
fallbacks, early cancellation, caps, subsequent/repeat requests and absence
of false MTP cache hits. Telemetry verifies ten user MTP admissions; AFM's own
prewarm is excluded. The first harness counted that prewarm and incorrectly
flagged coverage after all assertions passed; the corrected rerun also passes.
Artifacts: `qwen-scheduler-lifecycle-v{1,2}-*`.

Same-binary C15 comparison completed (prefix enabled but no MTP replay):

| Ownership | First / repeat aggregate tok/s | Peak RSS GiB |
|---|---:|---:|
| Prior serial MTP lane | 57.65 / 57.70 | 69.44 |
| Scheduler-owned MTP sessions | 57.89 / 58.35 | 69.75 |

All 30 response texts and output-token counts match exactly. Both complete all
requests with 24 structural passes; cached tokens remain zero. Telemetry shows
14 sessions admitted initially and the last joining before the cohort drains.
The roughly 1% change does not establish a throughput improvement. This is a
state-ownership prerequisite, not simultaneous multi-request verification.
Records: `qwen-scheduler-{control,owned}-v1-*`.

### Complete exact-prompt MTP replay (opt-in)

The snapshot is defined inside the self-contained model layer. It contains
target and MTP-head caches, last target hidden/HC stream, sparse-attention
index/position/pooled banks, recurrent/PLE array slots and CPU n-gram history.
Optional array positions are preserved explicitly; compacting absent slots
would corrupt their meaning on restoration. Buffers are materialized into
independent snapshot storage and copied back into fresh request-owned caches.
Rollback/verification-window metadata is cleared at this pre-generation boundary.

A generator-scoped UUID rejects another generator or model instance even if
prompt IDs match. Exact token identity is required. Samplers, sampled primary
tokens and RNG state are never cached: each request resamples its first token
from the retained hidden state with its own settings. Capturing/taking a prompt
snapshot does not create a second generation algorithm.

`ExactPromptReplayCache<Value>` supplies model-agnostic bounded exact-key LRU
storage. Access remains on the owning executor, without a new shared lock.
Its budget covers estimated retained value bytes plus key storage, not allocator
headers, transient copy buffers or active requests' private states. Oversized
entries are rejected without evicting existing valid entries. Shutdown clears
the cache; generator identity prevents stale cross-model reuse.

`AFM_QWEN_MTP_REPLAY_MIB` is **0/unset by default**, capped at 4096 MiB. The
prototype additionally limits storage to 16 entries and 4096 prompt tokens.
It only activates with prefix caching and the opt-in Qwen streaming MTP
scheduler. This does not accelerate the serial MTP lane, share incomplete
radix prefixes, or qualify replay beyond 4K. Other model families are unchanged.

131 focused Release tests pass, including mapped PLE and sparse-boundary
replay, greedy/sampled policies, independent interleaved copies, one-token
prompts, changed seeds, post-decode immutability, scope rejection and eviction.
The 512 MiB live mixed MTP/AR test passes 120/120 checks, including exact full
prompt cache-hit accounting after cancellation and on repeats.

Initial 512 MiB C15 screen: repeat throughput **58.41 versus 58.71 tok/s** with
replay disabled; all 30 texts/token counts match, and both retain 24 structural
passes. The small budget gets one initial warmup reuse but zero repeat-phase
hits as the working set cycles. This is not a replay speedup. Records:
`qwen-replay-{on,off}-v1-*`, binary
`1b04b0065a8fd3bb89ab62f8c68f72051035e81064d4ae4532e78bd3c478d98e`.
An explicit larger-budget screen will quantify actual retained bytes and
throughput before any default proposal.

The larger-budget same-binary screen completed with a 4096 MiB cap:

| Replay capacity | First / repeat aggregate tok/s | Repeat reused tokens | Peak process RSS GiB |
|---|---:|---:|---:|
| Disabled | 58.43 / 58.50 | 0 | 69.70 |
| 4096 MiB | 59.72 / 87.23 | 17,127 | 69.63 |

Repeat aggregate output throughput improves **49.1%**. All 30 texts and
completion-token counts match exactly; both complete 30 responses with 24
structural passes. The remaining six outputs truncate at the unchanged cap,
not newly introduced replay failures. Representative diagnoses/fixes are
coherent, but no broader semantic quality certification is implied.

The cache accounts for **2,364,401,124 bytes (2.20 GiB)** across 16 entries,
including startup prewarm and the 15 test prompts. This is estimated retained
tensor/key storage, not a claim of zero added memory because process RSS stayed
flat. RSS alone does not account for the full Metal/unified-memory allocation
picture. A 512 MiB budget cannot retain this working set; using the larger cap
is explicit, not a new default.

Warm per-request prefill logs drop from roughly 0.96 s to 0.002 s and median
client TTFT is 0.088 s. Lazy restoration work can move into the first decode
cycle; the end-to-end aggregate measurement includes it and is the performance
gate. This exact-repeat benefit does not establish shared partial-prefix
acceleration for divergent agent histories or multi-request verification.

Artifact binary `32aa87b91d08ba56470f35b7bd199af092f9069f158146537e7333fa32fdf4bf`;
records `qwen-replay-budget-{off,on}-*`. Reverse-order confirmation reproduced
**86.94 versus 58.24 tok/s** on repeats (+49.3%); first-phase rates were
59.88 versus 58.27. All 30 texts/token counts again match across arms, with
the same completion, structural-check and cache-hit totals. Records:
`qwen-replay-repeat-{on,off}-*`.

### Integrated six-mode checkpoint

The final screen used source checkpoint `75d88a5a` and the same binary hash
above, exact ddalcu checkpoint, 15 first requests plus 15 exact repeats per
configuration, temperature 0 / top-p 1 and a 192-token output cap. First rounds
follow an excluded warmup and are not completely cold. No competing build or
GPU run was active.

| MTP | Prefix cache | Clients | First aggregate tok/s | Repeat aggregate tok/s | JSON structure / completed | Peak process RSS GiB |
|---|---|---:|---:|---:|---:|---:|
| Off | On | 1 | 61.98 | 67.11 | 24 / 30 | 67.48 |
| Off | On | 15 | 109.55 | 128.24 | 24 / 30 | 67.64 |
| Off | Off | 15 | 71.47 | 71.83 | 26 / 30 | 67.69 |
| On | On | 1 | 58.86 | 58.66 | 24 / 30 | 69.40 |
| On | On | 15 | 59.93 | 87.13 | 24 / 30 | 69.54 |
| On | Off | 15 | 58.69 | 58.24 | 24 / 30 | 69.73 |

All **180/180 requests completed**; **146/180** passed the narrow JSON
structure/identity checks. These are separate totals, not 100% behavioral
qualification. Saved raw responses remain available for semantic review.
Replay-disabled configurations report zero reused tokens. C15 MTP replay
reports 1,144 / 17,127 cached tokens in the first / repeat phases; AR replay
reports 15,479 / 17,112. Serial C1 MTP still reports zero: its generator path
does not yet use the new scheduler-owned replay cache.

Every row has the same explicit experimental flags, including compatible and
continuous groups, yield interval 8, serial replay boundaries, the Qwen MTP
scheduler, a 4096 MiB MTP replay cap, and the previously documented batched
verification/fusion flags. Their model/path guards determine which apply.
MTP-on rows use fixed depth 3. The immutable PLE row cache is disabled. These
are **not production-default/no-flags measurements**. Full argument lists and
the binary hash are saved beside each run.

The new MTP repeat result reproduces the ~49% replay improvement, but remains
below AR grouped throughput because target verification is not yet shared
across requests. AR repeat throughput is still about 16% below the frozen
152.71 tok/s reference row; no overall parity claim is justified. The MTP
comparison also differs in replay memory policy and fixed versus adaptive
draft depth. Raw records: `shared-batch-checkpoint-mtp*-prefix*-c*-afm-mtp-*/`.

Build note: adding a provider source file exposed a stale consumer native build
manifest. `Scripts/swiftpm-reliable.sh build -c release --product afm
--disable-build-manifest-caching` replanned the source inventory while retaining
compiled objects; the successful build took 70.61 seconds. Its source list was
checked for the new cache file. No source checkout or compiled tree was deleted.

## Bounded independent MTP graph submission experiment

The scheduler previously called `nextToken()` for each speculative slot in
turn. A slot that needs a new verification cycle submits its graph and waits
for target/draft IDs before the next slot can build verification work. This
leaves an explicit per-request synchronization boundary even though model
weights are shared.

`Qwen4ExpMTPSession.prepareNextToken()` splits graph construction/submission
from the host decision and commit. It is idempotent while work is pending;
accepted-token buffers and deferred head repairs preserve their existing
behavior. Every session retains its own cache, rollback snapshot, PLE history
and request-local sampler. Deferred PLE leaves are filled before submission.
Cancellation discards the result; already-submitted GPU work cannot be undone.
The session is not Sendable and is only driven by the serialized model owner.

`AFM_QWEN_MTP_SUBMISSION_WINDOW=2` opts into preparing bounded windows of
independent requests before consuming their decisions. Values are clamped to
1–4, with 1/unset retaining the existing submission order. This requires the
separate Qwen MTP scheduler switch; AR and other architectures are unchanged.
The cap limits additional in-flight graphs, not total request memory. This is
**not joint multi-request GPU verification** and does not share cache rows or
change verification arithmetic. Focused tests and same-binary throughput,
memory and output comparisons are required before retaining the experiment.

## Rejected independent-row QMM screen

A bounded adapter reused the existing MTP q4 projection shader for 2–7
independent one-token batch rows. This changed linear row layout only, not
attention positions or cache representation. Numerical/row-isolation tests
passed (62 tests passed, one optional probe skipped across the focused suite),
as did 255/255 API lifecycle checks.

Same binary `fb8d3d5ba4595ab537df6341835d40fc12f34bf2798ce71959109698092f9a61`,
continuous groups and budget 1024 in both arms, same prefix/C15 workload:

| Projection | First / repeat aggregate tok/s | Peak RSS GiB |
|---|---:|---:|
| Ordinary | 112.06 / 128.66 | 67.57 |
| Adapted small-row QMM | 109.70 / 128.74 | 67.68 |

Both arms completed 30 requests with 24 structural passes and identical cached
token totals. Only 2/30 paired responses retained identical text. A 0.06%
repeat difference, slower first round and changed wording do not justify an
extra runtime switch. The adapter and its flag were removed; the existing MTP
kernel is unchanged. The rejected patch and raw evidence remain untracked as
`rejected-independent-qmm.patch`, `independent-qmm-screen-*`, and
`independent-qmm-safety-*` in the artifact root. This negative result does not
rule out other batch kernels or broader graph/scheduling improvements.

## Appendix: text-derived PLE reuse estimate

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
These results do not establish AFM concurrent parity. The subgroup and admission
screens above improve AR throughput, but arbitrary-position batch execution and
bounded speculative sessions remain major implementation work.

## Evidence policy

### Admission CPU turns: latency/throughput tradeoff

Input preparation shares the scheduler actor. Its existing 64-step cooperative
yield interval can delay tokenization of new requests while GPU work is active.
A yield every step passed 255 lifecycle assertions, but it fragmented incoming
compatible groups. On the same-checkpoint staggered 8+7 workload:

| Interval | First / repeat aggregate tok/s | Late median TTFT, first / repeat |
|---|---:|---:|
| 64 steps, control | 78.56 / 84.19 | 6.07 / 5.30 s |
| Every step | 74.13 / 75.85 | 1.15 / 0.24 s |

Both completed 30 requests with 28 structural passes and identical cached-token
totals. The first arm had eight initial streams active at late arrival in both
phases; the every-step repeat had seven. Arrival timing affects grouping, so
these are not identical GPU schedules. Earlier units, including independent
sparse-attention MTP sessions, passed 64 tests with one optional probe skipped.

Do not promote the 6–10% aggregate regression in exchange for latency without
user agreement. `AFM_QWEN_BATCH_YIELD_INTERVAL` permits a bounded 1–64-step
experiment only within opt-in continuous Qwen groups. Its unset default and
every other architecture retain 64. Raw records: `fair-admission-before-*`,
`fair-admission-after-*`, and `fair-admission-safety-*`. An intermediate interval
is being screened; no improved-throughput claim is made for it yet.

The same-binary intermediate screen subsequently measured interval 8 versus
64: first/repeat aggregate **80.04/80.35** versus **74.03/81.99** tok/s; late
TTFT **1.54/0.64** versus **6.45/5.74** seconds. All eight initial streams were
active at late submission in every phase. Both arms retain 30 completions,
28 structural passes and equal cached-token totals. There is no consistent
aggregate improvement; default remains 64. Records: `fair-admission-interval*`.
The interval experiment also applies when the explicit Qwen MTP scheduler
switch selects independent continuous admission.

Record end-to-end aggregate output throughput separately from decode-only
throughput, and report actual cached tokens, request queue time and memory.
Concurrency enabled does not prove batched execution. MTP enabled does not
prove every API feature is speculative-eligible. Truncated unconstrained JSON
is not automatically an engine defect. No benchmark run overlapping a build
qualifies as an uncontended reference baseline.
