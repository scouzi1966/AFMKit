# Qwen Next: consolidation and qualification order

September 15. Continue on PR #123; do not create a new optimization PR or
promote experimental defaults. This page is the entry point for the current
work, not a replacement for historical evidence.

Current state (September 16): consolidation and fixed-baseline replay are
complete; all 134 measured responses reproduce. Quality diagnosis has isolated
HC and GDN Q/K normalization differences. GDN-only normalization is advancing
to a sampled API screen, **not** to a production default. Reference quality
parity and combined MTP/C15/prefix/cancellation qualification remain open.

## Decisions retained

| Area | Decision | Evidence |
|---|---|---|
| Context decode improvements | Preserve the exact working configurations and historical peaks | [Performance ledger](QWEN_NEXT_PERFORMANCE_LEDGER.md) |
| Shared/batched verification, cache/replay experiments | Retain opt-ins; combined qualification remains required | [Opt-in matrix](QWEN_NEXT_OPT_IN_MATRIX.md) |
| Bounded MTP initialization | Keep the correctness/memory repair; track trajectory-dependent speed differences | [Quality investigation](QWEN_NEXT_MTP_PREFILL_QUALITY.md) |
| Prefill 8192 as a general default | Rejected: no agentic throughput benefit in the paired screen, fewer strict passes | [Prefill tradeoffs](QWEN_NEXT_PREFILL_TRADEOFFS.md) |
| Precision/normalization changes | Diagnostic only; no proven whole-model quality fix | [Quality investigation](QWEN_NEXT_MTP_PREFILL_QUALITY.md) |
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
