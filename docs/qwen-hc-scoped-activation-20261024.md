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
screen. It has **not** replaced the baseline. The combined short-HC/output-gate
candidate (`12a44c24`, binary `a597e213…`) also failed its exploratory task pair:
1,110.15 seconds and 8/11 checks versus the reference's 544.12 seconds and
11/11 checks. It remains rejected for promotion. Current work is a bounded
same-prefix arithmetic diagnosis; no new coding-parity result is claimed.
Main and installed binaries are unchanged.

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

## Blocked recurrence and native stride diagnostics: not promoted

The pinned reference's exact blocked GDN kernel was ported with its MIT and
underlying oMLX Apache-2.0 attribution retained. It is selected only by an
internal per-model test switch, never by serving configuration. Native oracle
outputs and final states match byte-for-byte at widths 64, 128 and 256 with
zero and nonzero state. Short widths decline the kernel. The existing packed
route still exactly matches its own source-algorithm control.

Interleaved Release medians, one layer and identical FP32 state:

| Width | Current packed ms | Ported blocked ms |
|---|---:|---:|
| 64 | 0.3659 | 0.3956 |
| 256 | 0.5568 | 0.6773 |
| 2048 | 1.4996 | 2.0565 |

These include binding/dispatch/evaluation, not just GPU time. Width 2048 repeats
checkpoint-derived input rows. The port is not a speed win on this experiment.
The primitive tests passed 2/2 after a 180.62-second build.

With the test switch disabled, all 320 saved opening score arrays are exactly
unchanged. The compiled-decode continuation test also passes. With it enabled,
opening winner overlaps are 58/64 (production split), 58/64 (joined), 59/64
(joined BF16-rounded), 58/64 (reconstructed split), and 57/64 (split BF16-rounded).
This is not an improvement. No runtime policy is promoted.

The actual native warm-R5 capture passes frozen/capture-off/capture-on controls:
all five saved replies and usage match, as do the 8,698 AFM/native prompt token
IDs and the 7,927-token restored prefix. Native source plus its `S=475` log show
an absolute 8,192 boundary: R5 prefill is 265 + 475 + 31, rather than 740 + 31.
This missing boundary limits the earlier synthetic forward comparison.

A new test-only stride arm reproduces those widths. It retains both mandatory
production controls. The controls still emit 58,56,523,233,179 tokens. Joined
BF16 storage emits 61,72,274,80,1354; matching the native stride instead gives
61,72,274,80,2048 (last response capped). Adding the ported blocked recurrence
gives 58,81,74,563,2048 (capped). None matches native 72,67,84,123,1824.
R5 initial-score RMS error is 0.9411 for production, 1.0528 for joined BF16,
1.1066 for native stride/BF16, and 0.9570 with blocked recurrence too. The latter
three select a different first word from the reference. Counts and score error
are diagnostic, not task quality; no generated tools were executed.

The warm-stride test passes its controls in 201.74 seconds after a 102.87-second
build. Its outputs, all first-score tensors, exact fixture hashes, and native
capture controls are retained under `QwenWarmNativeStrideScores20261024` and
`QwenWarm5NativeDecisions20261024`. Kernel timing and disabled-switch comparison
are `qwen-blocked-kernel-timing-20261024.json` and
`qwen-blocked-disabled-control-20261024.json`.

### Long-prefill projection policy: no observed numerical change

Native `prefillDqGemm` uses transient BF16 dequantization plus dense GEMM at
2,048+ rows. A test-only `QuantizedLinear` subclass applies this policy using
the exact packed checkpoint arrays. Its disabled control preserves all 320
saved opening score arrays. The enabled policy records engagement at 6,630
rows, but **also preserves all 320 score arrays byte-for-byte**. The five
winner-overlap counts remain 58,59,59,58,61 out of 64. This does not explain
the divergence. The two saved safetensors containers can differ in header
ordering; the comparison checks individual tensor payloads, not file hashes.

The enabled test passes in 35.13 seconds. Peak active allocation is
73,582,902,754 bytes; this is a diagnostic allocation measurement, not a
production memory benchmark. No speed claim or default change follows.
Evidence: `QwenOpeningDQControlScores20261024`,
`QwenOpeningDQScores20261024`, and
`qwen-dq-payload-comparison-20261024.json` under the external evidence root.

The next diagnostic retains ordinary-forward HC/attention/MLP intermediates
without re-forwarding or flushing a pending HC write. Native capture runs in
its own checkout; both engines must preserve their uncaptured score controls
before the retained values can support a numerical-divergence claim.

### Passive ordinary-forward capture locates the first difference

The native diagnostic binary is built in a separate checkout from pinned
reference source. Frozen/capture-off/capture-on replies and usage match.
All 64 captured raw score tensors match the prior qualified native capture
byte-for-byte. AFM's passive capture also preserves all 320 prior score arrays.
Its test passes in 34.88 seconds after a 186.86-second Release build. Neither
capture changes the pending-HC schedule or recomputes an observed stage.

On the ordinary first 6,630-token chunk, the full first-layer attention input
matches exactly: 16,972,800/16,972,800 values. The first observed difference is
the GDN output: 23,534 values differ (0.1387%), with RMS 0.00000738 and relative
RMS 0.00007634. The next HC/MLP input differs in 105,707 values; this is after
the differing GDN output, so it is not an independent HC bug diagnosis.
At layer seven, final-row MLP output relative RMS is 0.02188. This demonstrates
accumulated numerical divergence, not that the first small difference alone
causes the full coding-quality gap.

Evidence: `QwenOpeningPassiveNative20261024`,
`QwenOpeningPassiveScores20261024`, `qwen-passive-afm-control-20261024.json`,
and `qwen-passive-forward-comparison-20261024.json`. Native binary SHA-256:
`3afa9724564391ceeb633930a81955992c39698bb90a9b3ac2bc270843e7b064`.
The source manifest/patch and added diagnostic source files are retained beside
those reports. The next isolated test feeds this identical first-layer input
into the current and source-matched GDN implementations, with both state
precisions, and requires the ordinary control to reproduce the captured AFM
output before comparing counterfactuals.

The isolated first-GDN test now passes: ordinary FP32 state reproduces the
captured AFM output exactly. Current recurrence with either FP32 or BF16
initial zero state differs from native in the same 23,534 output values.
The blocked recurrence with either state type reproduces **all 16,972,800
native output values exactly**. Thus arithmetic order in the recurrence,
rather than initial state precision, accounts for this first cold-chunk
difference. This does not establish the cause of every later divergence.

Eight alternating measurements of the complete isolated first GDN layer:

| Recurrence / initial state | Median ms | Differing native output values |
|---|---:|---:|
| Current / FP32 | 38.7662 | 23,534 |
| Current / BF16 | 45.1738 | 23,534 |
| Blocked / FP32 | 40.2229 | 0 |
| Blocked / BF16 | 40.2724 | 0 |

This includes projections, prework, recurrence, output gate/projection, dispatch
and evaluation. It is a first-layer microbenchmark, not a full-model speed claim.
The blocked/FP32 increase is 3.76% here. Evidence is
`QwenNativeGDNForward20261024`; test duration 4.27 seconds. No serving switch
has changed. The follow-up retains later stages with the blocked diagnostic
enabled to locate the next divergence rather than assume this solves quality.

### PLE transition: lookup agrees, normalization differs

With blocked recurrence, first-layer attention output and subsequent MLP input
match native across the entire chunk. MLP output differs in 175/16,972,800
values, but its last 16 rows match exactly. The next layer is the model's PLE
layer (configured as one-based layer 2). Its attention input still differs;
this is not adequately explained by the already-corrected first GDN stage.

An expanded passive capture records the last 16 PLE rows. Native frozen/off/on
controls again preserve all replies, usage and 64 raw score tensors. Adding
the AFM capture preserves every earlier stage and score payload. The incoming
PLE stream is exact (163,840 values), as are the lookup embedding and value
projection (40,960 each). Key/query normalization are the first differing PLE
stages. Switching only the three PLE norms to reference-style staged BF16
rounding makes all 163,840 key and 163,840 query values exact. Gated values and
the convolution input/output still differ. Lookup identity is therefore not
the source of this observed PLE discrepancy.

These captures are `QwenOpeningPLENative20261024`,
`QwenOpeningPLEControlScores20261024`, and `QwenOpeningPLENormScores20261024`.
The comparison JSON files are `qwen-passive-ple-control-comparison-20261024.json`
and `qwen-passive-ple-norm-comparison-20261024.json`. Native capture binary:
`8d2c4c4eb63183f4c8c909eb1f3c9c82c4e588a1a1137747515f147e38f445e9`.
The earlier native capture binary `3afa9724…` is preserved separately under
`QwenPassiveNativeStageBinary20261024`.

The PLE-normalization diagnostic improves opening winner overlap to
59/64,60/64,62/64,59/64,59/64 across the five arms. It does not make the later
layers or full response equivalent. The next controlled arithmetic comparison
is gate scaling: AFM divides by a model-dtype rounded square root, while the
pinned native path multiplies by a model-dtype rounded reciprocal.

The gate-scale unit test now reproduces both captures exactly: the existing
division reproduces all 163,840 AFM gated values, while reciprocal
multiplication reproduces all 163,840 native values on identical inputs.
The disabled diagnostic preserves every prior stage and all 320 score arrays.
The Release build took 186.17 seconds; the gate and prefix controls passed
in 35.38 seconds combined.

With blocked recurrence, staged PLE normalization and reciprocal gate scaling
enabled together, every captured PLE stage is byte-identical to native,
including convolution input/output. The next layer's final-row attention
input also matches. Nevertheless, the full opening-prefix winner overlaps
are **54/64, 57/64, 57/64, 54/64, 57/64**, worse than the prior experiment.
This is a negative whole-model result, not a quality fix. These switches
remain test-only and default false; no serving default changed.

The first-layer MLP still differs in 175 of 16,972,800 output values on
identical input. That is the next localized comparison. Also, the staged PLE
normalization diagnostic currently covers widths of at least 128 only;
short-tail and singleton normalization have not been matched by that switch.
Neither local stage equality nor winner overlap qualifies coding quality.
The last full paired coding result remains 1,110.15 seconds and 8/11 checks
for AFM versus 544.12 seconds and 11/11 for the reference.

Evidence: `QwenOpeningPLEScaleControlScores20261024`,
`QwenOpeningPLEScaleScores20261024`,
`qwen-ple-scale-disabled-control-20261024.json`, and
`qwen-passive-ple-scale-comparison-20261024.json` under the preserved evaluation
root. The enabled prefix diagnostic passed in 35.22 seconds; it verifies
capture integrity, not a positive model-quality outcome.

### First MoE: routing agrees; shared expert differs

A qualified passive native capture now retains the first MoE intermediates.
Frozen/off/on replies and usage agree, all 64 raw score arrays agree with the
original capture, and every previously retained intermediate is unchanged.
The retained MoE output equals the ordinary first-layer MLP output exactly.
Native diagnostic binary SHA-256:
`b38326f35cdfde40d5f3dbbb62edd41afc9d0f47c158f7f9751ae931e4c673c0`.

The isolated AFM test reproduces both the ordinary AFM output and its stage
trace on the identical native input. Results:

| Stage | Differing values / total |
|---|---:|
| Router logits | 0 / 3,394,560 |
| Expert indices | 0 / 66,300 |
| Expert weights | 0 / 66,300 |
| Routed expert output | 0 / 16,972,800 |
| Shared gating logits | 0 / 6,630 |
| Shared expert body | 649 / 16,972,800 |
| Gated shared output | 608 / 16,972,800 |
| Combined MoE output | 175 / 16,972,800 |

The combined differences occupy 28 prompt positions. Replacing the routed
expert's combined compiled activation with the separate SiLU helper changes
none of these results. Replacing the shared SiLU helper with explicit
`gate * MLX.sigmoid(gate) * up` also leaves the same 649/608/175 differences.
Thus activation compilation alone does not explain this discrepancy. The
next capture separates the shared gate/up projections, activation and down
projection. No serving behavior has been changed for these diagnostics.

Evidence: `QwenOpeningMoENative20261024`, `QwenNativeMoEForward20261024`,
`QwenNativeSharedMoEForward20261024`, and
`qwen-moe-native-stage-control-20261024.json`. The first test passed in 3.16
seconds after a 184.33-second Release build; the expanded shared-expert test
passed in 3.16 seconds after a 98.17-second incremental test build. Separate
source snapshots preserve both test versions and their fixture hashes.

The expanded shared-expert capture confirms both 4,243,200-value input
projections are identical. Exactly 28 activation values differ; every one
has input gate value `-6.84375`, the sigmoid edge documented by the reference.
All prior native intermediate payloads and 64 raw score tensors remain exact
under the expanded capture. Source/binary manifests are preserved with it
(`QwenOpeningSharedMoENative20261024`, binary SHA-256
`976234c554b755a210112cbc8ec6807b26b1c16ae26c929d44b26fa80fc9d500`).

Using the already qualified `Qwen4ExpHCActivation.silu` for this shared expert
removes all 649 body differences and all 175 combined MoE differences:
**16,972,800/16,972,800 values match** for each output. The explicit-MLX-op
control did not help because its sigmoid still has the differing rounding;
the issue is not merely the presence of an explicit compile call.

Ten alternating complete shared-body timings (gate/up/down projections,
activation, dispatch and evaluation) give medians of 3.150104 ms current and
3.179250 ms precise, +0.93%. This is one layer at prompt width 6,630, not a
decode or full-model speed claim. No statistically established slowdown or
end-to-end improvement is claimed from these short measurements.
Evidence: `QwenNativePreciseSharedMoEForward20261024`,
`qwen-shared-moe-projection-comparison-20261024.json`.

The follow-up enables the precise shared activation only through an internal
test property, independently from an all-width PLE normalization diagnostic.
Both default false. Full-prefix comparisons first require disabled controls
to preserve the previous score arrays; component equality is not a release
qualification or a claim that the coding gap has been resolved.

The disabled full-prefix control preserves all previous intermediate and score
payloads. Enabling precise shared activation improves winner overlap in the
production-split diagnostic from 54/64 to 60/64. The other four arms give
59/64, 59/64, 60/64 and 57/64. Adding all-width PLE normalization gives
60/64, 60/64, 59/64, 60/64 and 58/64 across the five arms. These are fixed
reference-prefix decisions, not fresh coding outcomes or response-quality
scores. Both enabled runs complete in about 35.18 seconds.

With precise shared activation, all four full first-layer captures match
native exactly. The recorded last-row attention/MLP inputs and outputs in
layers 1 and 2 also match. Layer 3's last-row attention input matches, but its
attention output differs (relative RMS 0.00355). Earlier positions in layers
1--3 have not yet been captured in full, so this is the next observed boundary,
not proof that layer 3 is the sole remaining source. Evidence:
`QwenOpeningSharedControlScores20261024`,
`QwenOpeningSharedPreciseScores20261024`,
`QwenOpeningSharedAllWidthsScores20261024` and their comparison JSON files.


### Scalar text RoPE control and negative gate result

The attention helper already limits fused QK normalization/RoPE to sequence
width <=32. An initial proposed additional long-prefill exclusion was therefore
redundant; its disabled control reproduced prior payloads, and it was removed
without running or claiming an enabled result.

The actual long-text distinction is manual BF16 cosine/sine arithmetic in AFM
versus scalar MLXFast RoPE in the native text path. A default-false test switch
selects scalar RoPE only for B=1, S>32, with no explicit multimodal position IDs.
Its disabled control preserves all previous intermediate and 320 score arrays.
The enabled run matches all four recorded layer-3 last-row stages exactly.
However, five-arm winner overlaps become 58/58/58/58/57 out of 64, versus
60/60/59/60/58 before this intervention. Local agreement did not translate to
better final token agreement. The next observed mismatch is layer-4 attention
output, 893/2560 values exact, relative RMS 0.00459395; its recorded input row
is exact. Earlier rows are not yet fully compared.

A separate default-false precise attention-gate test at S>32 changes neither
these overlaps nor the recorded stage comparisons. Native's precise core gate
begins above 16 rows; this arm does not yet match the 17--32 tail policy and
must not be described as complete gate parity. The disabled control and both
enabled tests pass their capture-integrity assertions in about 35 seconds.
No serving defaults, installed binary, performance or coding-quality claim
has changed. Next evidence should capture earlier full rows, because a
recurrent layer depends on preceding positions, not only the last-row input.

Evidence: `QwenOpeningQKControlScores20261024`,
`QwenOpeningScalarControlScores20261024`, `QwenOpeningScalarRoPEScores20261024`,
`QwenOpeningScalarGateScores20261024`, their fixture/source manifests and
`qwen-passive-scalar-{rope,gate}-comparison-20261024.json` under the evaluation
root. Both scalar interventions remain diagnostic-only.


### Full-row capture localizes the next boundary to QSA attention

The expanded native capture saves every row of the first eight layers after
ordinary evaluation. Frozen/off/on semantic and usage controls pass, all 64
raw score payloads retain their original hashes, and all 51 older native
intermediates are unchanged after projecting the enlarged captures to their
previous rows. AFM's expanded capture likewise preserves every prior
intermediate and all 320 raw score payloads. The AFM test passes in 35.48 s.

| Captured stage, zero-based layer | Different values | Affected prompt positions |
|---|---:|---:|
| Layers 0--2, all four stages | 0 / 203,673,600 | 0 |
| Layer 3 attention input | 0 / 16,972,800 | 0 |
| Layer 3 attention output | 1,704,054 / 16,972,800 | 1,327 |
| Layer 4 attention input | 1,613,193 / 16,972,800 | 1,327 |
| Layer 4 attention output | 6,978,154 / 16,972,800 | 4,501 |

The first differing output is at position 2,129, above the 2,048-token QSA
selection budget. The layer-4 last-row discrepancy is therefore downstream
of earlier differing attention rows; its matching final input row did not
clear the incoming sequence. This is a concrete reason to capture all rows,
not an inference that recurrent-state memory is corrupt.

Source inspection identifies a second scalar-RoPE boundary: native QSA query
vectors use `mlx_fast_rope` for plain text at all widths, while AFM prepares
them with manual model-dtype cos/sin arithmetic. Both rotate 64 channels.
Native pooled keys intentionally retain the cos/sin path. The next isolated
test changes only plain-text, single-batch QSA query rotation at widths >32;
it does not change pooled keys, selection, masks, or explicit vision positions.
This proposed intervention remains disabled in serving and needs its own
control and result. No coding-quality or throughput recovery is claimed.

Evidence: `QwenOpeningFullRowsNative20261024`,
`QwenOpeningFullRowsScores20261024`, `qwen-full-rows-native-control-20261024.json`,
`qwen-full-rows-comparison-20261024.json`. Reference capture files are about
1.26 GB, kept externally and untracked. Earlier binaries and captures remain
preserved separately.


The scalar QSA query test's disabled control reproduces all expanded captures
and 320 scores. Enabled, layer-3 attention discrepancies fall from 1,704,054
values across 1,327 positions to 27,575 values across 20 positions (first
position 4,317). Layers 0--2 and the full layer-3 attention input remain exact.
The five fixed-prefix winner overlaps are 58/57/60/58/60, so no overall token
or coding-quality parity is claimed. The remaining 20 positions require
inspection of QSA's prepared queries, pooled keys, and selected mask. Evidence:
`QwenOpeningIndexerControlScores20261024`, `QwenOpeningIndexerScalarScores20261024`,
`qwen-indexer-scalar-full-comparison-20261024.json`. Build 186.11 seconds;
capture tests about 35 seconds each. The query intervention remains disabled.


### Passive QSA tensors identify a pooled-key spectrum mismatch

The next frozen/off/on native capture again preserves replies, usage and all
64 raw score arrays. All 51 prior native intermediate payloads are exact.
AFM's capture test passes in 35.44 seconds and preserves all prior intermediate
and 320 score payloads. This qualified capture gives:

| QSA stage | Different values / total |
|---|---:|
| Projected queries and keys | 0 / 4,243,200 |
| Rotated queries, scalar diagnostic | 0 / 3,394,560 |
| Pooled rotated keys | 214 / 212,096 |
| Actual boolean attention mask | 160 / 43,956,900 |

The 160 mask differences occur at exactly the 20 positions with differing
layer-3 attention outputs. Each swaps one four-token block for another. The
first position, 4,317, selects token block 1,612--1,615 only in AFM, versus
2,552--2,555 only in the reference. This localizes the remaining difference to
the selection inputs or calculation, rather than unexplained attention output.

Native `ropeInvFreq` constructs its spectrum as `exp(j * (-2*log(theta)/D))`;
AFM uses `1 / pow(theta, 2*j/D)`. Both describe the same mathematical spectrum
but can round differently in FP32 before BF16 cosine/sine values are formed.
A separate captured-input test will compare that formula with the actual
native key bank, first requiring the existing formula to reproduce AFM's
captured bank exactly. No serving switch is enabled by this new test.

Evidence: `QwenOpeningQSANative20261024`, `QwenOpeningQSACaptureScores20261024`,
`qwen-qsa-native-control-20261024.json`,
`qwen-qsa-passive-comparison-20261024.json`. Earlier captures and binaries are
preserved. The capture-only reference build had an import-name compile error;
the corrected retry succeeded before any GPU capture ran.


The captured-input pooled-key test passes both assertions: ordinary AFM
pool/norm/rotate reproduces its captured 214 differences, and the exponential
reference spectrum matches **all 212,096 native values exactly**. Twenty
alternating complete pool/norm/rotate construction+evaluation samples yield
medians 0.775125 ms current and 0.5394375 ms reference formula. These are one
captured key bank, not a model throughput result; the candidate constructs its
inverse spectrum inside each timed call while the ordinary model owns its
existing inverse spectrum. No broad speed or statistical quality claim follows.

Evidence: `QwenNativeQSAPooled20261024`, its hash-bound fixture and test-source
snapshot; Release test passes in 3.55 seconds. A default-false provider test
property now permits the formula for implicit text pooled-key positions.
Explicit multimodal positions retain their previous path. The next test checks
the entire ordinary prefix and saved QSA mask with a disabled control first.


The pooled-spectrum disabled control preserves every earlier intermediate
and score array. With the new text spectrum enabled, **all four captured
stages at every position across layers 0--7 match native exactly**:
543,129,600 layer-intermediate values, plus the saved PLE and QSA tensors.
The entire 43,956,900-value attention mask matches. This removes the 20 block
swaps and their downstream recurrent propagation in the captured layers.

The five fixed-prefix winner overlaps are 61/57/58/61/60 of 64. Production-split
improves from 58 to 61; mean raw-score RMS error across those 64 decisions falls
from 0.609524 to 0.552031. It remains nonzero: uncaptured layers and the tail /
decode policies are not yet proven equal. There is no new whole-task quality
or model-speed result and no serving-default promotion. The enabled prefix
test passes in 35.37 seconds after a disabled-control build of 185.25 seconds.

Evidence: `QwenOpeningPooledControlScores20261024`,
`QwenOpeningPooledSpectrumScores20261024`,
`qwen-pooled-spectrum-full-comparison-20261024.json`. The next diagnostic
extends the same passive comparison to the remaining layers; previous captures
remain immutable and hash-bound.

### Full 48-layer localization, 2026-10-24 22:30 EDT

The passive comparison now covers four ordinary-forward stages at all 6,630
positions in all 48 layers (3,258,777,600 layer-intermediate values). Native
frozen / capture-off / capture-on replies and usage match, and all 64 raw score
arrays remain identical to the earlier reference capture. AFM's five sets of
64 score arrays remain identical to its eight-layer capture. Prior captured
intermediates remain identical as subsets of the expanded files. The original
strict file-control helper correctly rejected the expanded passive key set;
a separate score-only comparison and subset payload controls establish the
intended invariant instead. No result was accepted from that failed assertion.

Layers 0--41 match fully. Layer 42's attention input/output and MLP input also
match. Its MLP output first differs by 10 of 16,972,800 BF16 values, at prompt
positions 5403, 5815 and 6078. Differences subsequently propagate through
layers 43--47. This is the first divergent operation in the captured cold
chunk; it does not establish the cause of every subsequent token decision or
the coding time gap. The next captured-input experiment isolates its shared
expert output gate, requiring ordinary AFM computation to reproduce its saved
output before interpreting the changed gate.

Release test build 97.95 s; prefix test 37.288 s, zero failures. Five winner
agreements remain 61/57/58/61/60 of 64. No new coding or throughput measurement,
no serving defaults changed. Evidence under the existing external root:
`QwenOpeningAllLayersNative20261024`, `QwenOpeningAllLayersScores20261024`,
`qwen-all-layers-full-comparison-20261024.json`,
`QwenOpeningAllLayersScores20261024-control.json`. Native instrumented binary
preserved in `QwenPassiveNativeAllLayersBinary20261024` with SHA manifest.

### Routed expert activation isolation, 2026-10-24 22:50 EDT

The shared-output-gate hypothesis was rejected. Its isolated ordinary forward
reproduced all 10 saved differences; changing only that gate left the same
10 differences. The proposed zero-difference assertion failed and the source,
fixture and results are preserved (`QwenNativeMoETailGate20261024`). Both gate
outputs were identical at every position; none of the three affected positions
had the earlier activation-boundary input.

A new native passive capture selected the 43rd occurrence of each ordinary
MoE stage, requiring exactly 48 occurrences. Frozen/off/on controls and all
64 raw scores remained unchanged; all prior layer, PLE and QSA captures remain
unchanged. Layer 42 comparison then showed exact router logits, indices,
weights and shared expert. Routed output differed in 13 values; adding the
identical shared output left 10. Evidence: `QwenOpeningLayer42Native20261024`,
`qwen-layer42-moe-stage-comparison-20261024.json`.

Isolating the existing sorted projections and activation reproduced those
13 differences. Separate SiLU also retained 13; the reference-matching precise
SiLU removed all 13. Exactly three activation values differed, each with input
-6.84375 (up values 4.1875, 4.96875, -2.609375). The existing precise helper
already has exhaustive finite-BF16 oracle coverage; its routed use was absent.
This is an activation-rounding discrepancy, not expert routing or the shared
output gate. The reference's fused prefill reduction is a separate possible
memory-traffic optimization; this experiment does not establish a reduction bug.

A default-false model diagnostic now changes only ordinary routed prefill
activation for widths >32, retaining ordinary sorting, projections and weighted
sum. A package-only override in SwitchGLU is nil for existing callers. The
captured-input regression asserts exact native output after this intervention.
Ten alternating complete MoE samples per arm give median 41.1207085 ms current
and 41.255229 ms candidate (+0.33%). This one-layer measurement is not a whole
model throughput qualification. Disabled control preserves every captured
intermediate and all 320 score arrays. Build 199.10 s; both tests pass (46.58 s).

With the routed diagnostic enabled, all four captured stages in all 48 layers
match native exactly: **3,258,777,600 values**, plus the retained PLE/QSA values.
The enabled prefix test passes in 39.114 s. Five 64-prefix winner overlaps are
61/58/58/61/62. Thus cold-chunk intermediate equality is established within
this capture scope, while tail/decode scores and coding parity remain unproven.
No serving default changed. Evidence: `QwenNativeMoERoutedAssert20261024`,
`QwenOpeningRoutedControlScores20261024`,
`QwenOpeningRoutedPreciseScores20261024`,
`qwen-routed-precise-full-comparison-20261024.json`. The next diagnostics target
plain-text QSA query rotation at short widths and the attention gate's native
>16-row threshold; state precision and final-tail geometry remain separately
tracked counterfactuals, not automatic production policy changes.

### Short-query and attention-gate controls, 2026-10-24 23:10 EDT

Extending scalar text-query rotation to short widths did not improve the
64-decision diagnostic. Adding the reference's >16-row attention-gate
threshold also did not improve the winner counts. These are internal,
default-false experiments, not selected serving policies.

| Arm | Prior diagnostic | Short scalar queries | Queries + tail gate |
| --- | ---: | ---: | ---: |
| Production 30+1 tail | 61/64 | 60/64 | 60/64 |
| Joined 31 tail | 58/64 | 58/64 | 58/64 |
| Joined 31, BF16 state values | 58/64 | 59/64 | 59/64 |
| Reconstructed 30+1 | 61/64 | 60/64 | 60/64 |
| Reconstructed 30+1, BF16 state values | 62/64 | 60/64 | 60/64 |

Production raw-score RMS changed from 0.552837 to 0.557651 with scalar
queries. Joined-31/BF16 RMS changed from 0.503345 to 0.526038. None of this
establishes a coding-quality benefit. Removing passive capture preserved all
320 score arrays (`QwenOpeningTailControlScores20261024-control.json`).
The three fixtures/output sets are `QwenOpeningTailControl`,
`QwenOpeningTailQueries`, and `QwenOpeningTailQueriesGate` with the existing
`Fixture20261024.json` and `Scores20261024` suffixes.

The next capture targets the actual final 31-token prompt chunk at offset
6630. The native frozen/off/on controls reproduced the saved response and
all 64 raw score arrays. Its binary and source are preserved under
`QwenPassiveNativeTailBinary20261024` and the `qwen-passive-tail-native-*`
artifacts; binary SHA-256:
`adeccc077c776d5d7d245637340130f3f1e8ec129175dcafe94e271b82ae7edb`.

The first AFM tail-capture attempt failed its four-stage assertion: passive
hooks covered deferred long prefill but not ordinary short prefill. During
failure cleanup XCTest also trapped in MLX's thread-local stream lookup.
This failed run is retained in `QwenOpeningTailPassiveScores20261024` and
`qwen-tail-passive-prefix-tests-20261024.log`; it is not qualified evidence.
The fix retains the ordinary short-tail arrays without evaluating or
re-forwarding them. The retry must preserve all prior scores before its
intermediates are interpreted.

The short-tail retry passed (release build 186.33 s, test 34.971 s). All 320
score-array payloads match the pre-capture control. At joined-31/BF16-state
geometry, all four stages of layers 0–2 and all retained PLE stages match
the native capture. Layer 3's attention input also matches; its attention
output is the first mismatch: 40,107/79,360 values across all 31 positions,
RMS 0.000383942, maximum absolute difference 0.0078125. Later layers already
receive different inputs, so their differences are not independent causes.
Evidence: `QwenOpeningTailPassiveRetryScores20261024`,
`QwenOpeningTailPassiveRetryScores20261024-control.json`, and
`qwen-tail-passive-comparison-20261024.json`.

The next passive capture includes the layer-3 sparse index tensors, rotated
Q/K, current and full cached V/K, attention heads, and output gate. The
expanded native capture remains identical under frozen/off/on response and
64-score controls (`QwenOpeningTailAttentionNative20261024`). No cache reset,
second forward, graph flush, or sampling change is introduced by these hooks.

Source comparison identifies a specific hypothesis: AFM's fused short Q/K
rotation narrows angles and individual products to BF16, while the pinned
native `QK_NORM_ROPE_256_SOURCE` keeps rotation in FP32 and narrows the result.
An internal default-false kernel template argument allows that hypothesis to
be tested without changing the norm, projections, launch geometry or serving
policy. It accepts FP32 angles only. This section is not a claim that the
hypothesis is confirmed or that coding performance has recovered.

One build-only attempt (`build -c release --build-tests`) failed because
release modules were not testable. The wrapper's native retry did not repair
that invocation. Subsequent work uses the established `test -c release`
wrapper path. The failed log is `qwen-tail-attention-swift-build-20261024.log`.

### Short Q/K rotation result, 2026-10-24 23:40 EDT

The expanded capture localizes the tail discrepancy to the rotated first 64
Q/K dimensions. Unrotated dimensions, V, sparse-index tensors and the output
gate already match. The FP32-rotation diagnostic removes every captured
attention input/output difference, including the full cached keys and values.
Across 211 retained tensors (24,181,403 values), only layer 47's MLP output
has differences: 46 values at nonfinal positions 6, 13 and 21, RMS 7.44e-6.
The final prompt token's complete score array matches native exactly under
joined-31/BF16-state geometry. This is a localized rounding correction, not
whole-task quality or throughput qualification.

| Fixed-prefix arm | Control winners | FP32 rotation winners | Control mean score RMS | FP32 rotation mean RMS |
| --- | ---: | ---: | ---: | ---: |
| Production 30+1 | 60/64 | 56/64 | 0.557651 | 0.603458 |
| Joined 31, F32 state | 58/64 | 58/64 | 0.613297 | 0.597363 |
| Joined 31, BF16 state values | 59/64 | 58/64 | 0.526038 | 0.472641 |
| Reconstructed 30+1 | 60/64 | 56/64 | 0.557651 | 0.603458 |
| Reconstructed 30+1, BF16 state values | 60/64 | 62/64 | 0.508311 | 0.458862 |

The ordinary production geometry does not improve in this probe. All new
switches remain default false. Even in the matched-geometry arm, the first
decode step diverges (score RMS 0.357259), so the next isolation target is
that step's ordinary intermediate tensors. No new whole-coding benchmark or
serving throughput measurement has been claimed.

Disabled capture controls preserve all 320 score arrays. Release test passes
in 35.238 s; the earlier build-mode mistake caused a full 307.66 s rebuild.
The enabled diagnostic passes in 35.171 s using the same test executable.
Evidence: `QwenOpeningTailAttentionScores20261024`,
`QwenOpeningTailRotationScores20261024`,
`qwen-tail-attention-comparison-20261024.json`, and
`qwen-tail-rotation-comparison-20261024.json` under the existing evidence root.
The expanded native binary is preserved in
`QwenPassiveNativeTailAttentionBinary20261024`, SHA-256
`8521929e1694973cd5a400dc8f854fe62b634db0a008ca29c2cde46fedfc300a`.

### First decode boundary capture, 2026-10-24 23:52 EDT

The native first-decode capture selects sequence length 1 at offset 6661,
retains existing layer-boundary arrays, and saves after the ordinary token
has been evaluated. Frozen/off/on controls preserve response/usage and all
64 raw score arrays. Native binary SHA-256:
`bab21de9d28becd26a66f53e4d2fa31925605ecd537b5d697fc60e7b6746d412`.

AFM captures only arrays already outside the compiled tail: attention input,
attention output, and MLP output. It does not add compiled outputs, disable
compilation, or re-forward layers. The capture is enabled only after prompt
completion, avoiding startup and singleton prompt-tail calls. Release build
141.21 s; test passes in 35.055 s. All 320 score payloads remain identical to
`QwenOpeningTailRotationScores20261024`.

On the matched joined-31/BF16-state arm, first-decode layer 0 attention input
and output both match native exactly (2,560 values each). Its MLP output is
the first retained mismatch: 2,028/2,560 values, RMS 0.0004024972, maximum
0.00390625. Later layers already receive different inputs. This localizes
the next investigation to the post-attention HC read and expert computation;
it does not yet distinguish the two, nor establish a coding/performance win.

Evidence: `QwenOpeningFirstDecodeNative20261024`,
`QwenOpeningFirstDecodeScores20261024`, its `-control.json`, and
`qwen-first-decode-comparison-20261024.json`. All remain under the evidence
root above. Main and installed binaries remain unchanged.
