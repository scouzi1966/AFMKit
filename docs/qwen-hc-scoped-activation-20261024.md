# Qwen HC activation arithmetic investigation

## Current acceptance state

No candidate in this document establishes coding parity or qualifies a release.
The preserved serving baseline is provider `eb9acc43`, binary `270adc06…`.
Its five coding runs average 804.59 seconds versus 504.81 for the reference,
with 44/55 versus 52/55 acceptance checks. Keep those full-task results separate
from component equality, fixed-input responses, and raw decode throughput.

The output-gate candidate `7a0065bd` is preserved as binary `48d07045…`.
It improves raw winning-token overlap on the native-prefix diagnostic from
52/57 to 56/57, but increases total output and HTTP time on the five-request
screen. It has **not** replaced the baseline. The next controlled experiment
combines the known short-HC correction with this output-gate correction;
the earlier short-HC-only candidate remains rejected. Main and installed
binaries are unchanged.

The sections below retain the history of each intervention, including failures.

This branch is an experiment, not a replacement for the preserved performance
builds. It combines the already isolated request-budget safeguard (`2235ab88`)
with the earlier HC normalization correction (`6598268f`, replayed as
`886d21a7`). The first activation probe was test-only. After it passed, the
candidate was wired into Qwen HC SiLU, sigmoid injection/mixing, and its BF16
sigmoid table. Shared MLX arithmetic and non-BF16 paths remain unchanged.

**Promotion rejected after the five-request comparison.** All 34 focused tests
passed, including the exhaustive activation oracle, but full-model behavior
worsened. Primitive agreement must not be presented as a completed reasoning fix.

| Fixed request | Reference tokens / seconds | Preserved HC tokens / seconds | Scoped candidate tokens / seconds |
|---|---:|---:|---:|
| 1 | 72 / 6.61 | 58 / 6.12 | 72 / 8.95 |
| 2 | 67 / 1.88 | 79 / 2.14 | 67 / 1.97 |
| 3 | 84 / 2.08 | 181 / 3.70 | 1,038 / 17.34 |
| 4 | 123 / 2.46 | 74 / 1.72 | 1,706 / 27.68 |
| 5 | 1,824 / 29.13 | 4,030 / 65.72 | 8,291 / 133.46 |

All five normalized request hashes match. These are fixed HTTP requests with
unexecuted tool calls, not five complete coding projects. Same ddalcu vision
overlay checkpoint; MTP off, thinking and prefix cache on. Durations include
request startup/JIT overhead. Candidate request 5 repeats its plan before the
budget transition and a tool call. The candidate is `d626cbe5`, consumer
`6839b056`, binary SHA-256
`874a682cd1d34dba1e3b559a4894914abd073f383dd4ec93f7def8d5d565260a`.
Preserved HC control `6598268f` reproduced its historical output counts exactly.

The conditional 512-token speed comparison did not run because requests 3 and
4 already exceeded its cap. No controlled performance acceptance is claimed.
The saved native capture explicitly calls `compileQwen4Hc`; its compiled HC
path was verified before further investigation. It remains a forward diagnostic,
not a replacement for HTTP and complete coding-task qualification.

Full outputs and comparison are preserved outside Git in
`QwenHCScopedFive20261024/COMPARISON.md` under the artifact root below.

## Evidence

The saved layer-0 reference capture contains 2,773,440 BF16 HC activation
inputs (`[1, 8667, 320]`). The normalized/down-projection inputs already agree.
There are 6,767 differing activation outputs. Every one has input `-6.84375`
(BF16 bits `0xc0db`): the reference activation is `-0.00726318359375` and the
previous AFM activation is `-0.00732421875`.

The reference's source explicitly identifies this same BF16 sigmoid boundary
in its `SWIGLU_SOURCE` discussion. Source: `ddalcu/mlx-serve`, MIT license,
commit `02bee553f48cd3bc7d82aba0f8073820bd924738`, `src/transformer.zig`.
Copying an expression is insufficient if compiler math modes or intermediate
rounding differ. The native `hcSiluCallback` uses MLX sigmoid and multiply.

A CPU calculation against the captured tensor confirmed the rounding
requirements: round the exponential, denominator addition, reciprocal, sigmoid
result, and multiplication to BF16 at their native boundaries. That calculation
matches all 2,773,440 captured outputs. Omitting the exponential, addition, and
reciprocal intermediate roundings produces 738,801 mismatches even when the
sigmoid output is rounded. Omitting only reciprocal rounding produces 1,479.
This establishes arithmetic on this input set, not a GPU performance result.

## Qualification

`QwenHCScopedActivationTests` probes a model-scoped precise-exponential kernel.
It preserves the typed intermediate operations rather than replacing them with
an FP32 sigmoid. Gates and activations are checked against an exhaustive finite
BF16 table generated through the frozen reference's actual MLX-C library, and
against the captured full HC activation. The known boundary is a standalone
regression fixture. Fixture-dependent tests are explicitly opt-in diagnostics;
they do not create runtime environment requirements.

The initial probe and normalization tests passed 9/9: zero mismatches on the
65,280 finite BF16 inputs and the 2,773,440 captured HC outputs. The integrated
helper must repeat those tests, cover irregular extents and legacy dtypes, and
pass the existing compiled-HC and injection tests. It uses exact grid dispatch
without an additional scalar buffer or shape-specific shader specialization.

Then recapture the first downstream mismatch, replay the five original HTTP
requests, and compare full coding outcomes. A shorter generated answer alone
does not establish quality. Neither this test nor the earlier global sigmoid
experiment qualifies changes to other models.

Artifacts and scripts:
`/Volumes/edata/dev/CODEX/codex-local-coding-eval-20261007/`:

- `analyze-hc-bfloat-rounding-20261024.py` and its JSON result;
- `capture-hc-activation-oracle-20261024.py` for exact reference-library tables;
- original captures in `QwenNativeR5Layer0CaptureRetry20261012` and
  `QwenWarmR5HCNormReference20261012` (unchanged).

The original captured payload hashes are recorded in the CPU analysis JSON.
Library, metallib, and derived fixture hashes belong in the generated oracle
manifest. GPU diagnostics must run after the paired coding tasks, never during
their timing measurements.

## Downstream isolation after the failed candidate

`QwenHCCapturedChainTests` compared seven successive same-input operations over
the captured five prefill spans (6,630, 595, 530, 172 and 740 rows). All 35
comparisons were bit-exact: 139,504,032 compared BF16 values, zero differences.
This includes HC down/up, activation, fused/composed mixing and a/b projections.
It qualifies these operators on the saved inputs, not the whole forward graph.
Artifacts: `QwenHCChain20261024` and `qwen-hc-chain-tests-20261024.log`.

The next diagnostic executes the frozen reference's exact `GDN_PREWORK_SOURCE`
with its native MLX-C library. Both engines receive the same QKV, convolution
history, a/b and checkpoint parameters. These stimuli derive from the captured
layer-zero HC output; they are not claimed as a full native forward capture.
At widths 1, 3, 8, 17, 30, 128, 172 and 256, the AFM production-default
prework has query/key differences. Values, convolution history, forget gates
and beta match on this fixture. For example, at width 30, Q differs in
22,562/61,440 values and K in 23,090/61,440. At width 256, the differences are
194,736/524,288 and 196,476/524,288 respectively.

The existing test-only reference normalization removes every measured
difference at widths 128, 172 and 256; it is not wired into normal serving.
The next intervention must qualify this arithmetic for short tails/decode as
well as wide prefill, then repeat the identical HTTP requests. This evidence
does not yet attribute all reasoning loops to that normalization difference.
Artifacts: `QwenGDNPreworkOracle20261024`,
`capture-gdn-prework-oracle-20261024.py` and
`qwen-gdn-native-prework-tests-20261024.log`.

The subsequent GDN intervention selects reference RMS normalization for BF16
128-dimensional Q/K heads, including singleton decode. It uses the qualified
sigmoid table at widths >=17 and retains the native narrow JIT sigmoid below
that boundary. Other dtypes/head geometries retain the existing path. No
environment variable or user option enables this candidate. It remains on
the isolated experimental branch pending API and task qualification.

Focused tests passed 20/20. All 48 native-kernel component comparisons match
exactly (6,602,400 BF16 values), across widths 1, 3, 8, 17, 30, 128, 172 and
256. A separate synthetic boundary test exercises the known -6.84375 sigmoid
case at prefill widths 17, 30 and 128. The existing FP16/legacy normalization,
compiled HC state/continuation and request-budget tests also pass. This is
neither a new performance baseline nor proof that coding loops are resolved.
Log: `qwen-gdn-native-arithmetic-tests-retry-20261024.log`; component results:
`QwenGDNPreworkOracle20261024/afm-native-arithmetic-results.json`.


## Integrated GDN candidate: not accepted

Candidate `a3de3d1675420ed95347024ad3ca379f934471dc`, consumer
`6839b0566dbd166f11ad4203db9568165ac8ff49`, binary SHA-256
`6dfb6764598be8364f367bc533ddebfaac63c04bec1882aa9909fc45397193d4`
completed all five saved requests with calls, but did not recover the preserved
control or reference behavior. Results (output tokens / HTTP seconds):

| Request | Reference | Preserved HC | HC + GDN candidate |
|---|---:|---:|---:|
| 1 | 72 / 6.61 | 58 / 6.12 | 59 / 6.40 |
| 2 | 67 / 1.88 | 79 / 2.14 | 70 / 2.04 |
| 3 | 84 / 2.08 | 181 / 3.70 | 626 / 10.80 |
| 4 | 123 / 2.46 | 74 / 1.72 | 335 / 6.02 |
| 5 | 1,824 / 29.13 | 4,030 / 65.72 | 6,256 / 101.10 |

The final response repeated sections of planned code. Do not promote this
candidate based on its primitive test results. All normalized request hashes
agree; tools were not executed. Artifacts: `QwenGDNNativeFive20261024`.

A separate final-request cap of 512 tokens, preserving the first four request
limits at 16,384, took 9.095 seconds for preserved HC, 9.136 seconds for this
candidate, and 8.825 seconds for the reference. AFM's native generation rates
were 62.70 and 62.61 tok/s respectively; the reference profile did not expose
native generation timing. Both engines reused 7,927/8,698 input tokens according
to their server logs. This single-observation latency diagnostic does not qualify
quality or statistical performance. Report: `QwenGDN512-comparison-20261024.md`.
The initial reference launch failed before starting inference because the harness
was pointed at an AFM-only archive. Its log is preserved; the successful retry
uses the established reference archive and identical saved input messages.

Inspection after this failed candidate found incomplete call-site integration:
`makeCompiledDecode` still called prework with the legacy default. The ordinary
prefill/verifier call site selected native normalization, but production compiled
single-token decode did not. The helper oracle therefore did not establish that
the serving decoder actually used the candidate arithmetic. The next test must
compare the real compiled one-token block with the ordinary block, including
persistent state, before another whole-model acceptance run.


The new `QwenGDNCompiledArithmeticTests` first reproduced eight assertion
failures: output and recurrent state differed at each of four continuation steps
for BF16/128-head geometry. FP16/128 and BF16/64 controls passed. After wiring
the same normalization policy into `makeCompiledDecode`, all comparisons pass
exactly. Existing tests for batch row removal/reordering, compile eligibility,
and compiled verification rollback also pass (3/3). Tests use the actual compiled
model block, not only the prework helper. This fixes the experimental call-site
inconsistency; full-model behavioral acceptance remains pending.

Evidence: `qwen-gdn-compiled-red-retry-20261024.log`,
`qwen-gdn-compiled-green-20261024.log`, and
`qwen-gdn-compiled-regression-20261024.log`. The first test compilation had an
async/autoclosure error in fixture construction, corrected before the red run;
its separate `qwen-gdn-compiled-red-20261024.log` is retained.

## Compiled normalization correction: serving results

Provider `eb9acc431cc67bb315c6ee49675a231b9b402e04`, consumer `6839b056`,
preserved binary SHA-256
`270adc062464ae3612bad67b24d96e1542c0027e47bafdd930218bae7d02e3c1`.
The release build passed. The five identical saved requests returned valid tool
calls with output counts 65, 62, 710, 136, and 1,928, in 6.518, 1.904, 12.324,
2.782, and 32.010 seconds. Total HTTP time is 55.54 seconds against 42.16 for
the frozen reference: still 31.7% slower over those five requests. These calls
were not executed and do not establish coding-task quality.

A separate capped 512-token final request took 9.162 seconds, versus preserved
HC 9.095 and reference 8.825. AFM native decode was 62.46 tok/s, versus preserved
HC 62.70. Reference native decode timing is unavailable in this profile. These
single observations do not establish statistical performance equivalence.

Fresh Orbital runs use the pinned Codex 0.162.1 client, identical first request,
the same ddalcu vision-overlay checkpoint, MTP off, thinking/prefix cache on,
temperature 0, top-p 1, top-k 0, seed 123, and 180k-token/60-minute task limits.
Each starts a fresh server and project. One inference process runs at a time.
Five fresh runs per engine are complete. This is a failed parity qualification;
all outcomes are retained:

| Engine/run | Seconds | Output tokens | Requests | Checks | Native decode tok/s |
|---|---:|---:|---:|---:|---:|
| AFM 1 | 363.04 | 20,224 | 20 | 10/11 | 62.20 |
| Reference 1 | 390.64 | 22,851 | 24 | 11/11 | not logged |
| AFM 2 | 1,216.66 | 66,794 | 81 | 3/11 | 60.87 |
| Reference 2 | 426.21 | 25,702 | 17 | 11/11 | not logged |
| AFM 3 | 413.19 | 22,849 | 27 | 11/11 | 61.36 |
| Reference 3 | 544.99 | 31,370 | 30 | 10/11 | not logged |
| AFM 4 | 1,585.20 | 86,551 | 88 | 10/11 | 59.68 |
| Reference 4 | 512.30 | 29,522 | 37 | 10/11 | not logged |
| AFM 5 | 444.88 | 25,371 | 21 | 10/11 | 61.93 |
| Reference 5 | 649.91 | 37,987 | 41 | 10/11 | not logged |

| Distribution | AFM | Reference |
|---|---:|---:|
| Mean seconds | 804.59 | 504.81 |
| Median seconds | 444.88 | 512.30 |
| Sample standard deviation seconds | 560.52 | 102.39 |
| Mean output tokens | 44,357.8 | 29,486.4 |
| Checks passed | 44/55 | 52/55 |
| Runs passing all checks | 1/5 | 2/5 |
| HTTP/API errors | 0 | 0 |

The faster AFM median does not establish parity. Its mean elapsed time is
59.4% worse, mean output count 50.4% higher, and the two long repair trajectories
are retained as failures of the acceptance goal. Native decode stays between
59.68 and 62.20 tok/s, but reference native timing is unavailable here, so these
runs cannot support a raw-decode comparison. Runs 4 and 5 also violate the
case-sensitive dropdown contract; run 4 repeatedly rebuilds and rewrites code.
This is evidence of behavioral variance, not proof that any particular floating
point discrepancy caused the task failures.

AFM run 1 violates the explicit dropdown value/label contract. Run 2's generated
JavaScript requires `id="planet-count"`, but its HTML only supplies
`data-testid="planet-count"`; startup fails and dependent interaction checks
fail. It also emits one malformed tool call (`command` instead of `cmd`). Run 3
passes all checks. No run has an HTTP/API failure or forced reasoning transition.
These observations identify generated-code defects but do not resolve whether
an inference discrepancy contributes to their frequency. Do not attribute them
solely to model quality. A successful first or third run does not cancel run 2.

The first response is identical across all three AFM runs. The first differing
model-visible input between runs 1 and 2 is the shell's generated chunk ID in
request 2. Later requests and trajectories diverge. This is evidence about
repeatability, not proof that the identifier caused the eventual app failure.
No receipts, task requirements, or evaluation rules were changed to improve scores.

Artifacts: `QwenGDNConsistentFive20261024/COMPARISON.md` and
`QwenGDNConsistentRepeatability20261024/REPORT.md` under the artifact root.
The latter records the completed batch and includes every failure,
per-run launches, binary identities, raw interactions, and distribution statistics.

## Next bounded diagnosis, not a qualified fix

The reference's `hc_prefill.zig` permits BF16 HC prefill for B=1...2 and S>16.
AFM's `normalizeGroupedPrefill`, `normalizeGroupedPrefillAfterInjection`, and
`mixGroupedPrefill` currently require at least 128 rows. Short fragments in the
17...127 range therefore use different normalization/mixing paths. Existing HC
capture parity covered widths 6,630, 595, 530, 172 and 740; it did not cover this
boundary. The singleton fused path is separate and must not be widened casually.

The native fixture capture completed: `capture-hc-tail-oracle-20261024.py` and
`QwenHCTailReferenceTests.swift` are preserved in the artifact root. The oracle executes the
pinned reference's exact HC normalization Metal source using its actual MLX-C
library at widths 17, 23, 30, 31, 64, 127 and 128. Inputs reuse saved normalized
HC activations as shared stimuli and actual checkpoint gamma; they are not a
capture of the model's actual pre-normalization activation. The test will compare
the production HC norm against these outputs; it was launched after the timed coding batch.
No runtime policy has changed for this diagnostic. Any subsequent intervention
must preserve non-HC, non-BF16, singleton, MTP and batching contracts and repeat
the same-request latency and behavior checks before promotion.


## Short HC prefill: reproduced; correction rejected after serving check

`qwen-hc-tail-red-20261024.log` reproduced six numerical failures on the
unchanged runtime: 690,565 of 2,990,080 BF16 values differed at widths 17, 23,
30, 31, 64 and 127. The 128-token control matched exactly. This is a numerical
boundary discrepancy; it does not prove causation of the Orbital quality failures.

The correction reuses the existing native-reference normalization kernel for
ordinary single-sequence BF16 HC prefill above 16 tokens. HC owners explicitly
pass the prefill context; generic/PLE normalization, speculative verification,
batch rows, FP16, singleton decode and the existing wide-prefill envelope retain
their paths. No new environment option or CLI switch is introduced. Pending HC
writes and mixing retain their existing implementations; only normalization is
changed in this bounded experiment.

`qwen-hc-tail-green-fixture-20261024.log`: 11 tests pass, including exact agreement
on all 4,300,800 native-oracle BF16 values, actual HC read/final-mixer and pending
call sites, negative scope controls, dynamic-row mixing and compiled GDN
arithmetic. The first synthetic call-site fixture used unit RMS and could not
distinguish the two paths; its explicit sensitivity assertion failed in
`qwen-hc-tail-green-20261024.log`. It was corrected to use non-unit RMS, without
changing runtime code or relaxing the assertions. Both logs are retained.

Correction commit `c6c432c9f210e9d8b298396916cc827a87b791f9` was built and
preserved as binary SHA-256
`75c66faaf24ff2ad4fef325857e01464c16c8b7578121559a63f0853a7dc734b`.
The same five saved requests returned valid, parseable tool calls, but generated
8,301 tokens in 140.51 seconds versus the preceding binary's 2,901 tokens in
55.54 seconds and the reference's 2,170 in 42.16 seconds. Individual candidate
counts are 55, 79, 1,275, 871 and 6,021. Tools were not executed; these are
fixed-request behavior/latency results, not whole-task quality scores.

Three fresh-server fixed-length comparisons followed. The final request hash
is identical across all nine runs; it generates exactly 512 tokens each time.
Earlier four requests retain their original 16,384 output limit.

| Arm | Median HTTP seconds | Min–max seconds |
|---|---:|---:|
| Preceding AFM `eb9acc43` | 9.145 | 9.079–9.161 |
| Short-tail candidate `c6c432c9` | 9.099 | 9.071–9.116 |
| Reference `02bee553` | 8.797 | 8.742–8.803 |

HTTP time includes prefill and transport; reference native decode timing is
unavailable in this profile. AFM cache reuse remains 7,927/8,698 input tokens
in the final fixed-length request. The small median difference is insufficient
to justify the observed increase in generated work.

**Decision: reject the short-tail experiment for recovery-baseline promotion.**
Restore the provider runtime and tests exactly to `eb9acc43`, retaining its
compiled-decode consistency correction and request reasoning budget. The short-tail
candidate, diagnostic tests, red/green logs and binary are retained in commit
`c6c432c9` and external artifacts; they are not part of the active runtime.
The arithmetic mismatch is still known and has not been declared resolved.
The preceding binary still fails full coding parity; this rollback does not
make it release-qualified or recover the historical best task result.

Artifacts: `QwenHCTailQualification20261024/REPORT.md`, `analysis.json`,
`status.json`, every command and response, plus frozen
`QwenHCTailCandidateBinary20261024/PROVENANCE.json`. The five-versus-five task
results above belong to `eb9acc43`, not the rejected candidate. Main and the
installed release were not modified.

## Recurrence source/compiler control after restoring the baseline

`diagnose-gdn-recurrence-source-20261024.py` executes AFM's packed recurrence
source and the pinned reference's stock/blocked recurrence source in the same
reference MLX-C runtime. All inputs are shared checkpoint-derived prework;
both arms deliberately use FP32 state. Widths 17, 30, 64, 128 and 256 each
have zero and nonzero initial-state controls. This isolates reduction order
from state-storage precision; it is not the reference's complete model policy.

`QwenGDNRecurrenceSourceTests` then runs the ten saved fixtures through AFM's
actual `gatedDeltaKernel` dispatch. All output and final-state values agree
exactly with the independently executed AFM-source control. At widths 17 and
30, they also agree exactly with the reference recurrence. At 64–256, the
blocked reference reduction differs in 9–97 output elements per fixture
(393,216–1,572,864 values), with maximum output difference 0.00048828125.
Final FP32-state differences have maximum absolute magnitude 0.000001669.

`qwen-gdn-recurrence-production-20261024.log`: two tests passed, including the
existing compiled-decode continuation test. Build: 301.44 seconds; tests:
0.184 seconds. Fixtures and source hashes live in
`QwenGDNRecurrenceSource20261024/analysis.json` under the artifact root.

This does not establish whole-model or coding parity. It rules out a large
recurrence formula/dispatch discrepancy on these bounded same-precision
fixtures. The active runtime remains the restored `eb9acc43` implementation;
no recurrence, precision, CLI or default policy was changed for this control.

## Current common-prefix scores: tail scheduling and state precision

`QwenNativePrefixGeometryTests` reuses the validated native cold-R2 capture
from October 12, on the same checkpoint and exact 7,255 input token IDs.
The native capture's on/off and historical semantic controls are verified,
and the diagnostic binds 75 files by SHA-256, including source, native binary,
libraries, model descriptors and all 57 raw score tensors. These are preserved
native captures, not a newly timed reference run. Only native-emitted tokens
are fed back; the final terminal decision is compared but never fed.

| Current AFM diagnostic arm | Raw winning-token sets overlap reference | Disjoint decision indices (zero-based) |
|---|---:|---|
| Production split: final 30 tokens then singleton | 52/57 | 7, 29, 30, 39, 43 |
| Test-only joined final 31 tokens | 54/57 | 7, 29, 43 |
| Joined final 31, BF16 persistent-state values | 54/57 | 7, 29, 30 |

All arms use the same production startup54 and identical cold-prefix cache
contents before the tail intervention. The third arm rounds only GDN state
after the cold prefix, final tail and each subsequent singleton; buffers
remain FP32 to keep AFM dispatch unchanged. It is a numerical counterfactual,
not a complete port of the reference's storage/kernel policies. It does not
change convolution, PLE or attention caches, and does not change serving code.

At decision 7 the reference prefers token 13, while all AFM arms prefer token
310. AFM's margin over token 13 is 0.5, 0.375 and 0.25 respectively. The reference
selected token is a raw winner in every one of the 57 native rows, so an
unobserved native sampling mask is not needed to explain these comparisons.
Raw score agreement is not coding quality and neither intervention resolves
the first difference. No scheduling or precision default is promoted.

The first two arms were run again when adding the state control. All 57 raw
score tensors per arm reproduce exactly (dtype, shape and payload); archive
byte ordering differs because tensor dictionary serialization order is not
stable. This distinction is checked explicitly rather than mistaking archive
SHA differences for numerical nondeterminism.

Logs: `qwen-current-native-prefix-geometry-retry-20261024.log` (one diagnostic,
16.487 seconds), `qwen-current-native-prefix-state-20261024.log` (one diagnostic,
23.018 seconds; incremental build 92.73 seconds). The initial compile failed
on a missing explicit `self` in the new test closure; that original log is
retained. Reports and all score vectors are in
`QwenCurrentNativePrefixGeometry20261024` and `QwenCurrentNativePrefixState20261024`.
Fixture preparation: `prepare-current-native-prefix-fixture-20261024.py`.

## GDN output gate: new source discrepancy and candidate correction

The reference's `GDN_NORMGATE_SOURCE` switches to its MLX sigmoid lookup at
S >= 17 and fuses the output RMS normalization/gate for prefill. AFM's output
gate retained inline fast exp through width 32 and a composed MLX fallback
above it. The already qualified precise HC activation had not been applied to
this separate output-gate call site.

`capture-gdn-output-gate-oracle-20261024.py` executes the exact native kernel
in the frozen native MLX-C library. Inputs combine saved shared recurrence
stimuli with checkpoint-derived Z projections and norm weights; this isolates
the output gate, and is not a full-layer capture. An additional synthetic
case covers every finite BF16 gate value. On the unchanged runtime the native
comparison fails six cases: 14/104,448 and 24/184,320 output values differ at
widths 17/30, and 55, 91 and 173 values differ at widths 64/128/256. Widths 1
and 8 match. The synthetic finite sweep has exactly one differing value.

The candidate reuses the qualified precise sigmoid table and extends the
existing fused gate only to single-sequence BF16 sigmoid-gated prefills with
128-wide heads. Narrow decode retains its inline arithmetic; non-BF16,
batched and other head widths retain fallback, and wide swish gates remain
excluded. There is no state precision change or new runtime option. The table
is immutable and evaluated once. Fusion removes separate prefill norm/gate
temporaries; actual serving performance still requires measurement.

`qwen-gdn-output-gate-red-20261024.log` retains the six baseline failures.
`qwen-gdn-output-gate-green-retry-20261024.log`: five tests pass, with exact
agreement on all 3,201,024 output-gate values, a small always-running numerical
regression test, scope controls, compiled-decode continuation and recurrence
source controls. A test-only scalar argument compile error in the initial
green build is retained in its original log. No assertion was relaxed.

This candidate is not yet a demonstrated coding-quality or serving-speed
improvement. The preserved `270adc06…` binary remains the serving baseline
until the candidate's score, fixed-request and timing gates are evaluated.

## Output-gate candidate: serving screen completed, not promoted

Provider `7a0065bd8eff471c34a28987e9df733348203d21` and consumer
`6839b0566dbd166f11ad4203db9568165ac8ff49` build successfully in 246.63 seconds.
Frozen binary SHA-256:
`48d07045eb3688c0368af9635a8c4ed1dde2bf221b7c9ddc63d1ebcd5c7d0e78`.

On the preserved native-prefix score fixture, production overlap improves to
56/57, but decision 7 remains different: AFM prefers token 310 over the native
token 13 by 0.875 (previous margin 0.5). The joined-tail arm is 54/57 and the
joined/BF16-state counterfactual is 56/57. A winning-token-set overlap can also
be a tie; these are not sampled response or quality claims.

Fresh serving screen, same checkpoint and five normalized HTTP request hashes:

| Arm | Output tokens | Summed HTTP seconds |
|---|---:|---:|
| Preserved control `eb9acc43` | 2,901 | 54.713 |
| Output-gate candidate `7a0065bd` | 3,069 | 57.883 |
| Reference `02bee553` | 2,170 | 41.114 |

Candidate counts are 72, 67, 428, 2,008 and 494; control counts reproduce
65, 62, 710, 136 and 1,928. Reference reproduces 72, 67, 84, 123 and 1,824.
All responses contain parseable tool arguments. Tools were not executed.
The shorter candidate request 5 asks for another inspection instead of the
control's file write, so it is not evidence of increased task productivity.

The predeclared 512-token timing guard fails: the candidate finishes naturally
at 494 tokens while the control reaches 512. The failed driver log is retained;
do not compare those unlike lengths as a fixed-length speed improvement.
Remaining repeated timing runs were not run. Future timing qualification must
use a cap every arm reaches. The native candidate generation rate on this
494-token request is approximately 62.5 tok/s; this is not reference raw-decode
parity because reference native generation timing is absent from this profile.

Artifacts: `QwenOutputGateQualification20261024/REPORT.md`, `analysis.json`,
`status.json`; `QwenOutputGateCandidatePrefixScores20261024`; every free-length
and attempted fixed-length request, response, launch command, and server log.
This candidate remains unpromoted. The combined HC/output-gate intervention
must first pass native component and common-prefix diagnostics, then be judged
by actual coding outcomes rather than an isolated shorter answer.

## Combined HC/output-gate correction and state/chunk factorial

Reapply only the bounded short-HC runtime/test correction from `c6c432c9`
on top of `7a0065bd`. This is a new interaction experiment, not a reversal of
the short-HC-only acceptance decision. No serving state precision, prefill
chunk policy, MTP behavior, CLI option or environment default is changed.

`qwen-output-gate-hc-interaction-tests-20261024.log`: 14 tests pass. The HC
and output-gate fixtures remain bit-exact, and the ordinary/compiled GDN
continuation control passes. Build: 308.97 seconds; tests: 23.20 seconds.
The production common-prefix score comparison is 56/57; joined final31 is
56/57; joined final31 with BF16 state values reaches 57/57, with unique winners.

The follow-up completes the chunk/state factorial. Before interpreting its
state intervention, a manually reconstructed split30+1 path must reproduce
**every byte of all 57 production score tensors**. That control passes.

| Diagnostic chunk policy | Persistent GDN values | Raw winner overlap | Disjoint decisions |
|---|---|---:|---|
| Actual production split30+1 | FP32 | 56/57 | 7 |
| Reconstructed split30+1 | FP32 | 56/57 | 7 |
| Reconstructed split30+1 | BF16-rounded | 56/57 | 43 |
| Joined final31 | FP32 | 56/57 | 7 |
| Joined final31 | BF16-rounded | 57/57 | None |

The split/BF16 arm also has a tie at decision 39; the joined/BF16 arm has no
ties. Rounding alone removes decision 7's mismatch but introduces a later one.
This identifies an interaction on this fixture; it does not establish that
the complete native runtime has been replicated or that coding quality is fixed.

The checkpoint declares `text_config.mamba_ssm_dtype=float32`, and AFM uses
FP32 state. Reference `src/model.zig:ModelConfig.ssmStateDtype` chooses BF16
for non-Hadamard packs. The test deliberately retains FP32 buffers/dispatch
while rounding state values between forwards. Its startup54 is unchanged.
This counterfactual avoids conflating arithmetic with a different kernel path,
but is not the reference's full storage/runtime policy. Do not silently change
serving precision based on this result.

`qwen-output-gate-hc-state-factorial-tests-20261024.log`: one diagnostic passes,
94.81-second incremental build, 36.47-second execution. All five arms, input
hashes, raw scores and source bindings are retained in
`QwenOutputGateHCStateFactorialScores20261024`. Next: API qualification of the
combined arithmetic correction with unchanged serving precision/chunking.

## Combined-correction serving qualification: not accepted

The `12a44c24` provider was built with consumer `6839b056`. Frozen binary
`QwenOutputGateHCCandidateBinary20261024/afm` has SHA-256
`a597e2135d9f4ff067a0fb078961839972797f123817a053c113c9014fe569a6`.
Neither main nor the installed AFM was changed. The checkpoint remains
`/Volumes/edata2/models/qualification/Qwen3.8-Flash-Next-ddalcu-vision-overlay-20261001`.

The five saved API histories generated 58, 56, 523, 233 and 179 tokens
(1,049 total; 25.559 seconds summed HTTP time). These were independent saved
histories, not an executed coding task: the fifth response inspects files
instead of implementing the reference response's work. Its shorter length
was not accepted as a productivity win.

The subsequent fresh Orbital pair used pinned Codex 0.162.1, greedy sampling,
MTP off, thinking on, prefix cache enabled, and the established
180k-output-token/60-minute/240-request limits. Tools and their results were
preserved; each engine started in a fresh task directory.

| Engine | Elapsed seconds | Output tokens | Requests | Project checks | Native decode tok/s | Prefill seconds |
|---|---:|---:|---:|---:|---:|---:|
| AFM candidate | 1,110.15 | 61,721 | 58 | 8/11 | 61.10 | 57.37 |
| Frozen reference | 544.12 | 33,140 | 24 | 11/11 | Not logged | Not logged |

AFM took 2.0403 times as long and generated 1.8624 times as many tokens.
This is one exploratory pair, not a new five-run distribution, and does not
prove which arithmetic change caused the outcome. It is insufficient for
promotion. The earlier five-run baseline and every preserved binary remain
available; no result has been replaced by this sample.

The AFM failures were an insufficient rendered scene, failed raycast selection,
and a browser exception from a negative animation delta. Both engines had zero
API errors. AFM generation consumed 1,010.17 seconds, about 91% of task time;
1,519,702 of 1,564,942 reported prompt tokens were cached. Its trace contains
repetitive late self-checking, including one 5,282-token inspection response.
The cache worked on this task; the dominant expense was extra generation.
There were zero detected forced reasoning transitions for AFM and one for the
reference. Do not equate the reference's HTTP token rate with native decode.

An opening-request audit found identical model-visible input and identical
semantic opening responses across the five `eb9acc43` baseline processes.
The combined candidate changes the opening reasoning wording but retains the
same first tool call. Later histories contain changing shell timings/IDs and
different model actions. This does not support blaming the five-run variance
on random opening answers or claiming identical later prompts.

Evidence under the existing external evaluation root:

- `QwenOutputGateHCOrbitalPair20261024/REPORT.md`, `analysis.json`,
  `report-sha256.json`, launch pins and all request/response artifacts.
- `OrbitalOutputGateHCAFM20261024` and
  `OrbitalOutputGateHCReference20261024`: full projects and independent checks.
- `QwenOutputGateHCFive20261024`: fixed-history API diagnostic.

## Warm-cache state/chunk factorial: controls qualified, no default change

`QwenWarmStatePolicyTests` adds a test-only diagnostic using five captured native
AFM prompt-ID histories. It restores the actual radix boundaries
0, 6630, 7225, 7755 and 7927, generates between restores, and retains each
response's prompt-backoff snapshot. It does not execute emitted tools or feed
newly generated text into the subsequent frozen history.

The production control reproduces all five current API output counts.
External comparison also verifies exact reasoning text and parsed tool names
and arguments for all five. The reconstructed split path reproduces every
generated control token ID. This is stronger than matching response length
alone. It is not a complete SSE lifecycle comparison.

| Test-only policy | R1 | R2 | R3 | R4 | R5 |
|---|---:|---:|---:|---:|---:|
| Production split, FP32 state | 58 | 56 | 523 | 233 | 179 |
| Reconstructed split, FP32 state | 58 | 56 | 523 | 233 | 179 |
| Split, BF16-rounded state values | 63 | 79 | 128 | 485 | 2048, capped |
| Joined tail, FP32 state | 63 | 51 | 458 | 96 | 2048, capped |
| Joined tail, BF16-rounded state values | 61 | 72 | 274 | 80 | 1354 |
| Saved reference API responses | 72 | 67 | 84 | 123 | 1824 |

Counts describe different generated responses, not equivalent accomplished
work. The alternative policies do not reproduce the reference responses.
The earlier 57/57 cold-prefix winner agreement therefore does not generalize
to these warm histories. Neither precision nor chunking is approved as a
serving-default change from this experiment.

Limitations: the BF16 intervention rounds stored values while retaining FP32
buffers and dispatch; it is not the full native BF16-storage implementation.
The captured startup prompt is known, but its historical four generated IDs
were not recorded; the test generates four fresh startup tokens. Alternate
responses are bounded at 2,048 tokens and API reasoning-budget transitions are
not simulated. The actual API control's responses are all below that bound.

The Release diagnostic passes one test with both controls enforced:
308.25-second build, 208.06-second test. Evidence is
`QwenWarmStatePolicyScores20261024`, its `api-control-semantics.json`, fixture
with 89 hash bindings, and `qwen-warm-state-policy-tests-20261024.log`.
No serving runtime or default was changed by this diagnostic.

The actual-storage follow-up initializes the recurrent buffers as BF16 before
the first request and retains that dtype through each restored prefix and
decode. This changes AFM's prefill dispatch: the current packed kernel admits
FP32 state only. Both original controls pass again. The BF16-storage arm emits
61, 72, 274, 80 and 1,354 tokens, with **every generated ID identical** to the
joined-tail/BF16-value-rounding arm above. Changing the buffer dtype does not
close the remaining difference on these histories. This is not justification
for a serving precision change or a performance claim.

`QwenWarmBF16StorageScores20261024` retains the outputs; the test passes in
94.24 seconds after a 96.21-second incremental build. Its fixture explicitly
selects three arms, requires the production/reconstructed controls first, and
rejects unknown or duplicate arm names. The earlier five-arm fixture/results
remain unchanged.

## Opening-request native scores: prompt identity proved, parity not proved

The frozen reference, capture-disabled reference, and capture-enabled reference
produce identical opening answers and usage in fresh processes. The captured
6,661 prompt token IDs hash to
`92f782746a9817f9e41ba2f850d1a457e7905880ff183aeffb352625eb99b575`,
identical to AFM's captured prompt. This rules out prompt formatting as the
opening-request difference, without making a claim about later warm histories.

`QwenNativePrefixGeometryTests` now accepts an explicit `opening-r1-64` fixture
in addition to the unchanged `cold-r2-57` case. Each arm receives the reference's
same emitted prefixes; AFM-generated tokens do not feed the next decision.

| Diagnostic arm | Overlapping top-score choices |
|---|---:|
| Production split, FP32 state | 58/64 |
| Reconstructed split, FP32 state | 58/64, byte-identical scores to production |
| Joined final 31, FP32 state | 59/64 |
| Joined final 31, BF16-rounded state | 59/64 |
| Reconstructed split, BF16-rounded state | 61/64 |

No arm reproduces every native decision. State precision and tail geometry
alone are not established fixes. The release test passes its control assertions
in 34.82 seconds after a 96.14-second build; a passing diagnostic is **not** a
passing parity gate. Runtime and serving defaults remain unchanged.

Evidence: `QwenOpeningNativeDecisionsRetry20261024`,
`QwenOpeningPrefixFixture20261024.json`, `QwenOpeningPrefixScores20261024`,
and `qwen-opening-prefix-tests-20261024.log` under the external evaluation root.
The initial capture attempt is retained separately: its copied capture binary
could not find a relative libllama dependency, so no score comparison used it.
