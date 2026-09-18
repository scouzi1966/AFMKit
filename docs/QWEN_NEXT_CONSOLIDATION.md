# Qwen Next: consolidation and qualification order

September 15. Continue on PR #123; do not create a new optimization PR or
promote experimental defaults. This page is the entry point for the current
work, not a replacement for historical evidence.

September 18: [latest-reference refresh and cache diagnosis](QWEN_NEXT_REFERENCE_REFRESH.md)
compares the preserved AFM checkpoint with released reference v26.9.4. Cached
C15 MTP wins do not close the short-context MTP, uncached AR or quality gates.
The new one-anchor cache experiment is default-off; no promotion is implied.

September 17 follow-up: [retained lifecycle and matched presets](QWEN_NEXT_RETAINED_QUALIFICATION.md)
passed 565 lifecycle assertions and 540 runtime/structure/identity checks
(lifecycle total corrected September 18 from the saved phase sums).
Cached C15 throughput beats the preserved reference in this workload; first-use
prefix handling and broad quality parity remain open. Single-client requests
also benefit from the existing scheduler replay when explicitly selected.
Actual model switching is not tested by the consumer's acknowledgment routes.

Current state (September 16): consolidation and fixed-baseline replay are
complete; all 134 measured responses reproduce. Quality diagnosis has isolated
HC and GDN Q/K normalization differences. The GDN-only prototype improves the
fixed sampled API screen from 38/50 to 43/50 strict passes with essentially
unchanged throughput. The fused implementation now preserves the prototype's
full-model tensors and all API answers exactly, with no material throughput
change against a warm repeat. It is **not** a production default: two original
passes regress, broader quality parity is unproven, and full release/lifecycle
qualification remains open.

The subsequent [independent semantic gate](QWEN_NEXT_BROADER_QUALITY.md) is now
testing 15 new task families against both AFM binaries and the live preserved
reference. C1 sampled counts are close, but the candidate introduces a greedy
LRU regression; the earlier five-task improvement has not generalized
convincingly. Completed C15 and bounded-prefix-repeat screens also do not
justify promotion. It remains default-off: further production-facing work on
this normalization candidate is not justified by the observed benefit.

The default-normalization build, under M25 opt-ins, reaches 184.18 aggregate
tok/s for C15 sampled exact repeats with MTP, versus 178.11 for the candidate,
both 21/30 semantic passes. This is
a new short-task, 15-prompt-working-set screen, **not** an improved Context
curve or concurrent reference-parity claim. A 45-prompt reuse-distance run
exceeded the existing 16-entry MTP cache cap and restored zero tokens; both
the capacity miss and bounded-hit results are preserved. Existing optimizations
and historical performance peaks remain intact. Runtime cancellation remains
a separate gate; source defaults and the installed nightly are unchanged.

## Decisions retained

| Area | Decision | Evidence |
|---|---|---|
| Context decode improvements | Preserve the exact working configurations and historical peaks | [Performance ledger](QWEN_NEXT_PERFORMANCE_LEDGER.md) |
| Shared/batched verification, cache/replay experiments | Retain opt-ins; C15 bounded reuse measured, full lifecycle/release qualification still required | [Opt-in matrix](QWEN_NEXT_OPT_IN_MATRIX.md) |
| Bounded MTP initialization | Keep the correctness/memory repair; track trajectory-dependent speed differences | [Quality investigation](QWEN_NEXT_MTP_PREFILL_QUALITY.md) |
| Prefill 8192 as a general default | Rejected: no agentic throughput benefit in the paired screen, fewer strict passes | [Prefill tradeoffs](QWEN_NEXT_PREFILL_TRADEOFFS.md) |
| Precision/normalization changes | Keep diagnostic only; the original local gain did not generalize convincingly and no material speed benefit justifies promotion | [Independent and combined results](QWEN_NEXT_BROADER_QUALITY.md) |
| New automatic tuning | Not authorized; keep explicit CLI choice and guidance | [Prefill tradeoffs](QWEN_NEXT_PREFILL_TRADEOFFS.md) |

## Ordered work

1. Commit the existing diagnostics, scripts, tests and findings. Preserve the
   previously tested executable and adjacent resources outside the repository.
2. Integrate the ten missing main commits into this feature branch, preserving
   Qwen changes. Main adds GLM/DeepSeek bounds, DwarfStar snapshot updates, and
   release-test gating. Its documented DSpARK parity skip is not a test pass.
3. Build through the consumer's reliable wrapper and run focused regressions.
   Recheck incoming memory/sparse-attention tests; report any failures explicitly.
4. Reproduce a fixed baseline before changing inference code: exact ddalcu
   checkpoint, saved prompts, temperature 0.6, top-p 1, fixed seeds, thinking off,
   C1/prefix off, MTP off and depth 3. Keep the existing documented M25 settings;
   do not describe these as unset-environment tests. Context and agentic metrics
   remain separate.
5. Resume the MTP-off numerical investigation, checking an unmodified reference
   diagnostic build against the frozen executable before trusting layer traces.
   Locate meaningful divergence before selecting a production change. A closer
   intermediate tensor alone is not proof of better output quality.
6. Once numerical/quality gates justify it, requalify retained MTP, prefix/radix
   cache, C15 aggregate throughput and cancellation together. Do not claim older
   isolated results certify the newly combined build.

The first-four-context curve uses 493/864/2112/4150 prompt tokens, 128 output
tokens, one excluded warmup and three trials per cell. Agentic replay uses the
saved five task families, five greedy controls and 50 sampled requests per mode,
with a 512-token cap. Report runtime, identity and strict structure separately;
the latter is not semantic judging or statistical proof of quality parity.

Record prefill proxy, decode-only rate, output tokens per wall second and
successful tasks/sec separately. Never substitute a new lower result for a
historical peak. Ask about performance/quality tradeoffs before adopting them.

Local evidence root:
`/Volumes/edata2/afm-benchmarks/qwen-next-mtp-parity-20260909/consolidation-20260915`.

## Consolidation checkpoint, September 15–16

Steps 1–3 are complete. Diagnostics and findings were checkpointed at
`f0c9782a`; the ten main commits merged cleanly at `c356e5be`. The merge did
not change Qwen, batching or replay runtime files. The fixed-control replay
harness is committed at `0ec5db44`; all 21 CPU harness tests pass.

The pre-merge executable and its adjacent resource bundles are preserved in
`pre-merge-binary`, with a verified `PRE-MERGE-SHA256SUMS.txt`. Its executable
SHA-256 is
`10086d6504aa7dbd6e48ed5c8678cc0b1bdfd3a260a64836667de1a141b0b0c6`.
The rebuilt executable SHA-256 is
`3d449739c0535de9426403d074140e86317902f15fff0c5c1c8e52b4cc282074`.
No installed version or main branch was changed. Unrelated uncommitted work
in the user's main consumer checkout was left untouched.

### Build and focused regression results

The first incremental link failed on the newly added DwarfStar image C
translation unit: SwiftPM's generated release plan still contained the old
source list. Regenerating that plan, without deleting compiled objects,
compiled the missing source and passed in 21.51 seconds. The paired consumer
now invalidates the generated configuration plan when its local provider
fingerprint changes (`e8b8e7a`). A second build exercised that safeguard and
passed in 4.16 seconds with the same executable hash. This is a build-plan
repair, not a change to Qwen inference or the reference libraries.

Focused Release XCTest: **209 passed, 10 skipped, one failed test case** out
of 220. Another 17 Swift Testing streaming tests passed. All selected Qwen
and GLM tests passed, including GLM score-allocation bounds. Optional
checkpoint diagnostics that require explicit fixture paths were skipped,
not counted as checkpoint validation.

The failing case is
`DeepseekV4DSparkPrefillTests.testFullAndChunkedPrefillPreserveLogitsAndNextProposalAcrossBoundaries`
(59 failed assertions). Main already documents this exact case as its RC5
DSpARK exception. This run deliberately did not enable the release script's
skip flag and reproduced the failure. It is not a green release gate; no
new DeepSeek fix or full cross-model qualification is claimed here.

### Fixed context replay

Same M25 controls, unchanged checkpoint and prompts. Each value is the median
of three measured trials; four warmups per mode are excluded. Pairs are
**prefill proxy / decode tokens per second**, not agentic wall-time throughput.

| Context | Before, MTP off | Consolidated, MTP off | Before, MTP depth 3 | Consolidated, MTP depth 3 |
|---|---:|---:|---:|---:|
| 0.5K | 893.08 / 68.72 | 883.70 / 66.70 | 959.63 / 89.00 | 956.14 / 89.47 |
| 1K | 1045.58 / 68.28 | 1036.53 / 66.69 | 1119.43 / 86.45 | 1117.71 / 85.97 |
| 2K | 1190.89 / 62.00 | 1186.54 / 62.01 | 1276.50 / 81.85 | 1271.38 / 82.12 |
| 4K | 1293.68 / 60.77 | 1291.81 / 60.54 | 1261.43 / 82.92 | 1258.23 / 84.29 |

All 24 measured responses match the frozen pre-merge text exactly. These are
historical-versus-current timings, not a counterbalanced same-session causal
experiment: the roughly 2.9% short-context non-MTP decrease remains recorded,
not attributed to a code regression without a paired replay. MTP decode is
within -0.6% to +1.7% of the same preset's previous medians. Historical
depth-4 peaks remain in the performance ledger; none is replaced by this run.

### Fixed agentic replay

Both modes completed all 55 measured cases plus one excluded warmup. All
110 measured texts and payloads reproduce the frozen pre-merge controls.

| Mode | Runtime | Greedy strict | Sampled identity | Sampled strict | Sampled wall-time output tok/s | Strict tasks/s | Sampled median TTFT | Sampled median decode tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MTP off | 55/55 | 5/5 | 40/50 | 38/50 | 25.01 | 0.11416 | 3.536 s | 53.90 |
| MTP depth 3 | 55/55 | 5/5 | 39/50 | 37/50 | 27.79 | 0.12054 | 3.588 s | 68.25 |

Against the immediately preceding paired 4096-prefill screen, wall-time output
rate is +0.03% without MTP and -2.82% with MTP. Against September 14's earlier
independent screen it is -5.90% and -6.03%, respectively. Those earlier rates
remain recorded: unchanged answers do not establish unchanged performance,
and historical timing differences alone do not isolate a cause.

Across context and agentic replay, **134 measured responses are identical**
to their saved same-mode controls. Ten warmups are excluded. All four servers
exited normally; resource guards recorded no competing named build/inference
process and at least 364.45 GiB available memory. This is not a leak soak or
a guarantee that every possible external GPU workload was detected.

`audit-replay.py` independently rechecks payloads, texts, token counts, strict
scores, executable/runner/checkpoint metadata hashes and lifecycle results.
`AUDIT-REPLAY.json` and `REPLAY-SHA256SUMS.txt` preserve its findings. The
rebuilt executable and adjacent resources are frozen in `consolidated-binary`
with `CONSOLIDATED-SHA256SUMS.txt`; its version smoke check reports `v0.9.20`.

Step 4 is complete. This demonstrates consolidation equivalence on these
controls, **not reference quality parity**. Numerical diagnosis and the
combined C15/prefix/cancellation gate remain open. No experimental default
is promoted.

## Calibrated component diagnosis, September 16

Step 5 now has an actual reference-server capture, rather than only a replay
of its equations. The unmodified ReleaseFast source build reproduces all five
saved greedy answers and five 32-token top-20 logprob traces from the frozen
v26.9.2 executable. A second build adds a one-shot diagnostic capture and
reproduces the same controls. It reuses the frozen release's backend libraries;
this is not a new MLX backend build or an AFM runtime dependency.

The capture contains the actual first 4,431 prompt tokens and layer-zero
intermediates. Every token ID matches the frozen task. An earlier capture
selected the loader's eight-token warmup and is excluded. The portable patch
is [the reference capture patch](../Scripts/diagnostics/qwen-next-reference-capture-v26.9.2.patch),
credited to the MIT-licensed ddalcu/mlx-serve source. Its local research commit
is `748acfb6`; no upstream reference PR was opened.

[QwenNextReferenceCaptureTests](../Packages/AFMKitMLX/Tests/AFMKitMLXTests/QwenNextReferenceCaptureTests.swift)
compares components with the same checkpoint and real captured inputs. It
first verifies exact embeddings and the initial HyperConnection stream.
Changing the *diagnostic equation* to round normalized values to BF16 before
the learned multiply makes both layer-zero HyperConnection reads and injection
gates bit-for-bit identical to the captured reference. No production equation
has changed.

| Isolated comparison | Maximum absolute difference | RMS difference | Fraction of elements different |
|---|---:|---:|---:|
| AFM normalization, attention HC read | 0.125 | 0.002327 | 21.57% |
| Reference normalization, attention HC read | 0 | 0 | 0% |
| AFM normalization, MLP HC read | 0.046875 | 0.001544 | 43.55% |
| Reference normalization, MLP HC read | 0 | 0 | 0% |
| AFM GDN, identical reference input | 0.0107422 | 0.00042247 | 63.75% |
| AFM MoE, identical reference input | 0.000488281 | 0.000000504 | 0.00158% |

This isolates normalization rounding and a separate recurrent-path difference.
It does **not** establish that the reference rounding is more mathematically
accurate, that GDN is buggy, or that changing either improves answer quality.
The next gate is a whole-model controlled experiment, with the frozen baseline
retained and performance measured separately. No automatic precision/default
selection is authorized by this result.

The diagnostic XCTest passed in 2.839 seconds after a 249.66-second build.
Its guard recorded no competing named workload and at least 385.29 GiB
available memory. The build/test wrapper unnecessarily invalidated this test
scratch after a consumer build; this is a build-efficiency issue, not a model
failure. Two earlier harness attempts stopped before test execution and are
excluded (process-ownership detection and a missing prebuild executable).

Reproduction requires explicit `AFM_QWEN_PREFILL_QUALITY_MODEL`,
`AFM_QWEN_REFERENCE_CAPTURE` and a fresh `AFM_QWEN_PREFILL_QUALITY_OUT` directory;
otherwise the diagnostic skips. These are test inputs, not inference tuning
options. In the external evidence root, see `reference-source-control-a`,
`reference-capture-width-control-a`, `reference-components-c/summary.json`,
`components-c-test.log` and `components-c-exit.json`. Diagnostic timings are
not throughput benchmarks. Reference quality parity and combined concurrent
cache qualification remain open.

### Whole-model normalization experiment

`QwenNextNormalizationAblationTests` completed 20 fixed-prefix arms (five task
families, current/reference rounding crossed with current/reference chunk
geometry) and ten independently generated greedy answers. The control exactly
reproduces all five saved full-vocabulary decision-logit vectors. Its five
greedy answers also match the saved API texts after trimming outer whitespace.
Both current and reference-rounding arms pass **5/5** greedy structure/identity
checks and stop normally. Four candidate answers change wording.

Correct-token probabilities below use full-vocabulary softmax at temperature
0.6 on the identical forced prefix. These are not sampled task pass rates.

| Filename decision | Current | Reference rounding only | Reference rounding + reference chunk geometry |
|---|---:|---:|---:|
| cache.swift | 55.17% | 77.67% | 60.24% |
| retry.swift | 83.72% | 86.52% | 73.66% |
| queue.swift | 81.11% | 81.12% | 65.12% |
| stream.swift | 90.81% | 92.41% | 93.74% |
| limits.swift | 94.83% | 93.71% | 95.76% |

The isolated rounding change improves three margins, leaves one essentially
unchanged, and worsens one. Matching the first HC layer therefore does **not**
establish full-model parity or justify a default change. Reference chunking
also has mixed effects; it is not silently folded into the candidate. The
remaining GDN difference warrants component-level isolation before a broad
sampled/performance retest.

This experiment only sets `referenceGroupedPrefillRoundingForTesting` on
the test's own norm modules. There is no CLI/API/environment activation. It
defaults off, is restored after the test, and is limited to unbatched grouped
prefill of at least 128 rows. Singleton decode, small verification blocks and
batched inputs retain their existing normalization path. All installed and
frozen API executables remain unchanged. This internal test seam is **not a
production precision policy**.

The test passed in 152.324 seconds after a 135.53-second incremental build;
minimum available memory was 365.93 GiB. Per-arm diagnostic timings include
synchronization and are not a server-throughput comparison. The next sampled
or performance claim requires the same API harness, not these XCTest times.
Evidence: `normalization-a`, `normalization-a-test.log`,
`normalization-a-command.json`, `normalization-a-exit.json` and the independent
`AUDIT-NORMALIZATION-A.json`. Historical decode and prefill peaks are unchanged
in the ledger.

### GDN difference isolated further

The corrected component replay is bit-identical to AFM's actual first-layer
GDN output. Its composed convolution/Q/K values also exactly match AFM's
fused preprocessing on this input. This validates the control before the
ablation is interpreted.

Keeping projections, values, gates, FP32 recurrent state, recurrence kernel
and output projection unchanged, replacing **only Q/K normalization** with
the reference equation reduces output RMS error versus the real reference
capture from **0.000422468 to 0.00000999913 (97.63%)**. The fraction of
different output elements falls from 63.75% to 0.33%. The reference computes
an FP32 RMS reduction, rounds to model dtype and applies Q/K scaling; AFM's
existing path uses model-dtype L2 operations. Epsilon placement differs too.
This identifies an arithmetic contract difference, not evidence that the
reference equation is necessarily more mathematically accurate.

Replacing the gate equation gives no further output change in this probe.
The remaining small difference is not yet isolated. The next bounded test is
the same whole-model fixed-prefix/greedy screen with Q/K normalization alone
and together with HC rounding, preserving both original controls. Only after
that should a candidate enter the larger sampled/API performance screen.

The initial GDN replay (`reference-components-d`) incorrectly used the
unfused fallback dispatch as its AFM control; its exactness assertion failed.
Those candidate results are excluded. The corrected replay
(`reference-components-e`) uses the actual fused prework and packed recurrence;
both selected tests pass (3.021 seconds total, 47.69-second build). The
test-only HC setting's scope test also verifies that singleton, small-block
and batched inputs are unchanged. Minimum available memory: 383.32 GiB.
None of these diagnostic RMS reductions is a speedup or quality pass rate.

### Whole-model GDN follow-up

The six-arm extension (`normalization-b`) completed 30 fixed-prefix arms and
20 independent greedy answers. All four greedy configurations pass 5/5
structure/identity checks and terminate normally. The previous experiment's
20 scalar-result rows and all its greedy texts/token IDs reproduce exactly;
the unchanged control again reproduces the saved full logits and API answers.

| Filename decision | Current | GDN Q/K only | HC + GDN Q/K |
|---|---:|---:|---:|
| cache.swift | 55.17% | 60.25% | 60.25% |
| retry.swift | 83.72% | 90.45% | 88.73% |
| queue.swift | 81.11% | 86.69% | 84.10% |
| stream.swift | 90.81% | 94.86% | 95.78% |
| limits.swift | 94.83% | 94.83% | 93.68% |

These are full-vocabulary probabilities at temperature 0.6 on fixed prefixes,
not sampled quality scores. GDN-only improves four probes and essentially
preserves the fifth; combining HC rounding has mixed effects. Advance only
GDN-only to the fixed sampled API screen. Do not promote either precision
change, alter chunk defaults or claim parity from these five related tasks.

The GDN diagnostic property is model-local, internal, default-off, and limited
to B=1 prefill of at least 128 tokens. It preserves actual fused values/gates,
cache updates and FP32 recurrence but deliberately duplicates convolution to
isolate Q/K arithmetic. This is not the proposed fast implementation. A useful
sampled result would still need fusion and performance requalification.

Test: 258.665 seconds after a 136.71-second build; guard passed with at least
365.26 GiB available memory. Evidence: `normalization-b`, its command/test/exit
records, and `AUDIT-NORMALIZATION-B.json`. No installed executable was changed.

### Controlled GDN API screen

The GDN-only prototype has now completed the same frozen 55-request API screen
as the control. Exact ddalcu checkpoint; saved M25 controls; MTP off; C1;
prefix off; thinking off; unchanged 4096 prefill; five greedy controls plus
50 sampled cases at temperature 0.6/top-p 1 with saved distinct seeds and a
512-token cap. One warmup per binary is excluded. This is a sequential
control-then-candidate screen, not a counterbalanced performance study.

| Measure | Frozen control | GDN Q/K prototype |
|---|---:|---:|
| Runtime completion | 55/55 | 55/55 |
| Greedy structure + identity | 5/5 | 5/5 |
| Sampled identity | 40/50 | 45/50 |
| Sampled unique-key JSON + identity | 38/50 | 43/50 |
| Sampled output tokens / total request wall seconds | 26.3007 | 26.4605 |
| Sampled strict tasks/s | 0.120051 | 0.135131 |
| Sampled median TTFT | 3.5366 s | 3.5427 s |
| Sampled median decode | 60.3155 tok/s | 60.2497 tok/s |

Seven prior failures pass and two prior passes fail, net **+5/50**. The two
regressions both identify `stream.swift` correctly: case 13 omits the required
`fix` field, while case 33 reaches the 512-token cap with incomplete JSON.
They remain failures; the token limit and scoring rules were not relaxed.
Strict passes by family (ten samples each) change from 6/9/6/8/9 to
7/10/9/8/9. All 55 control texts reproduce the frozen control; 52 of 55
candidate texts differ. Each candidate greedy answer matches its independent
direct-model test, verifying that the experimental arithmetic was active.

Within this screen, raw wall-time token rate changes **+0.61%**, median decode
**-0.11%**, and strict tasks/s **+12.56%**. The last improvement comes from
more passing tasks, not a 12.56% faster decoder. These five related task
families do not establish statistical or semantic quality parity. The frozen
reference AR results, rescored on identical payloads, are 42/50 identity and
40/50 strict. That historical reference was not rerun in this timing pair;
43/50 versus 40/50 is not a claim of general superiority.

Both servers exited normally, with no detected named competing workload and
at least 371.63 GiB available memory. This guard is not peak process/Metal
memory measurement or a leak soak. Prefix reuse, MTP and concurrency were
not exercised. Context peaks and the earlier agentic timing records remain
unchanged in their ledgers.

The private candidate is AFMKit `f8a7c073` plus one recorded temporary
default-activation line, consumer `e8b8e7a`; executable SHA-256
`4f16295fe193e39bf7b5a71d31d98d399709de53f9e72ae99f9d71b1ec1524aa`.
It is frozen outside the repository with the control's unchanged adjacent
resources. The source activation was restored immediately after building and
was **never committed or pushed**. Both internal diagnostic defaults remain
off, with no new CLI/API/environment option. The installed nightly is unchanged.

After the comparison, the mutable development executable was rebuilt from
the clean default-off source: SHA-256
`afb80df2ba0182436c18b41f985e127863d60517913aa989436bc81373c90737`.
A first-case API restore smoke passes and reproduces the frozen control text;
this one-case smoke is not a second full qualification. The frozen control
and private candidate remain available independently of that mutable path.

Evidence: `NORMALIZATION-API-SCREEN.md`, `AUDIT-NORMALIZATION-API.json` (including
hashes and all changed pass/fail cases), `normalization-api-control-a`,
`normalization-api-candidate-a`, and `restored-default-smoke-a` in the external
evidence root. The audit rechecks payloads, scores, binary/runner/checkpoint
identity, greedy activation controls and normal server exits.

**Next gate at this checkpoint (completed below):** preserve this quality signal while moving the Q/K arithmetic
into fused preprocessing, eliminating the diagnostic duplicate convolution.
Require component-equivalence tests and a same-checkpoint quality/performance
repeat before broader sampled and MTP/cache/concurrency qualification. Keep
HC-only rounding and automatic chunk/precision selection out of that change.
No default promotion or claim that the full parity goal is complete.

### Fused GDN implementation follow-up

The internal/default-off GDN experiment now prepares reference Q/K directly
inside `Qwen4ExpGatedDeltaPrework`, instead of recomputing convolution and
running two additional RMSNorm graphs. A compile-time variant retains the
prototype's FP32 reduction, model-dtype RMS output rounding, and separate
model-dtype Q/K scales. The existing decode/verifier variant keeps its
low-precision L2 arithmetic. Source comments credit ddalcu/mlx-serve and MLX's
RMSNorm implementation; no external runtime dependency is introduced.

The reference variant fails closed outside B=1, at least 128 tokens, and
128-dimensional heads. No CLI/API/environment switch or default activation
is added. Existing singleton, verification and batched calls keep the old
variant. The original prototype binary and outputs remain frozen separately.

Six focused Release tests pass (29.875 seconds after a 261.04-second rebuild).
Twenty-four BF16/FP16 synthetic combinations cover zero/small/ordinary/large
inputs, nonempty convolution history, and widths 128/129/257. Fused Q/K match
the composed prototype exactly. On the real 4,431-token reference input,
queries, keys, values, gates, beta and next convolution history are also
bit-identical to their expected counterparts. Existing batched-row isolation
and compile-eligibility tests pass. Minimum available memory: 378.60 GiB.

Evidence: `reference-components-f`, `components-f-command.json`,
`components-f-test.log` and `components-f-exit.json`. These are component
correctness checks, not a throughput or complete model-quality measurement.
Whole-model replay and paired API timing are separate gates.

Whole-model replay is now complete: `normalization-c` reproduces all 30 prior
configurations, all 20 greedy token sequences/texts, and **90 full-vocabulary
arrays exactly** (initial logits, decision logits and probabilities). The
unchanged control still matches the older frozen logits. All 20 greedy answers
pass strict structure/identity and terminate normally. The CPU audit is
`AUDIT-NORMALIZATION-C.json`; its comparison target is the preserved unfused
`normalization-b` output, not a newly regenerated oracle. Test duration:
290.283 seconds; incremental build: 5.73 seconds; minimum available memory:
362.17 GiB. These diagnostic times are not a throughput comparison. The
next gate is paired API timing with exact saved-prototype answer checks.

That API gate is complete as an A/B/A screen: unfused prototype, fused
implementation, then the identical unfused prototype again. Each arm completes
55/55 runtime requests, 5/5 greedy strict and 43/50 sampled strict (45/50
identity). All 55 responses and completion-token counts in each arm exactly
match the original saved prototype, including its seven sampled failures.
Three warmups are excluded. These are repeated controls, not 165 independent
quality tasks.

| Sampled measure | Unfused before | Fused | Unfused after |
|---|---:|---:|---:|
| Output / request wall tok/s | 24.6049 | 26.9248 | 26.9797 |
| Strict tasks/s | 0.125654 | 0.137502 | 0.137783 |
| Median TTFT | 3.4923 s | 3.5031 s | 3.4924 s |
| Median decode tok/s | 52.4445 | 61.8237 | 61.6710 |
| Peak process RSS, whole arm | 69.1528 GiB | 69.1797 GiB | 69.2210 GiB |

Against the warm after-control, fusion changes median decode **+0.25%**, wall
token rate/strict tasks per second **-0.20%**, and TTFT **+0.31%**: no material
throughput improvement or regression is established. The apparent large gain
versus the first control does not survive the warm repeat. The same unfused
binary recovers from 52.44 to 61.67 tok/s; its timing variation has not been
causally isolated. Run order or filesystem warming are possibilities, not
proven explanations. A single triplet is not a statistical performance bound.
The prior 60.25 tok/s prototype observation and historical context/MTP peaks
remain preserved, not overwritten.

The useful result is **removing redundant computation while preserving the
observed quality benefit**, not a new decoder-speed breakthrough. Further
normalization-only micro-tuning offers no demonstrated material return to
justify more numerical risk/retesting. Retain this default-off implementation
and move to broader sampled/semantic and MTP/cache/concurrency qualification.

All three servers exited 0; available memory remained at least 365.13 GiB.
No named competing build/inference process was detected. The one-second RSS
samples are not full Metal-memory accounting, a leak soak, or evidence of
memory savings. No MTP, prefix reuse or concurrent serving was run here.

Fused code checkpoint: `2c9c8241`. Private enabled binary SHA-256:
`f59804bc41db85f57366aa29d5c5a27d2a10a12d71bd820a7f0ad307d2fe8a3c`.
The temporary one-line activation was restored and never committed. Evidence:
`FUSED-NORMALIZATION-API-SCREEN.md`, `AUDIT-FUSED-API.json`,
`normalization-unfused-repeat-a`, `normalization-fused-a`, and
`normalization-unfused-repeat-b`. The independent audit checks scores,
payloads, texts, token counts, hashes and normal exits. No default is promoted.

After the screen, the mutable consumer executable was rebuilt from the clean,
default-off `2c9c8241` source in 98.12 seconds. Restored SHA-256:
`e7970fd1f444e13f23fc8f589b7dad29723216ea803697d7b02d71c4f85d7f91`.
`restored-fused-default-smoke-a` passes its first greedy API case and exactly
matches the frozen default-control response, with an excluded warmup and
normal server exit. This is a restoration smoke, not broad qualification.
The separate private binaries and all previous results remain preserved;
the installed nightly and production defaults were not changed.
