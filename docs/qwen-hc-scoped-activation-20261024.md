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

As of 2026-10-25 02:57 EDT, the combined diagnostic matches all **64/64
opening and 57/57 second-request full vocabulary score arrays bit for bit**.
All 405 retained intermediate arrays at second-request decision 16 also match.
This requires the joined-tail / BF16-state-value arm and the complete diagnostic
recipe. It is not the production-default or API cache path. The last missing
second-request difference was PLE's convolution activation at the BF16 input
-6.84375. The older 64-decision guard still passes after that correction.
There is no new coding or throughput acceptance. The eight-lane expert
reduction's earlier component median was 2.74% slower and is not promoted.


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


### First decode HC gate isolation, 2026-10-25 00:12 EDT

A focused layer-0 test supplies the exact native MLP input to AFM's ordinary
and compiled MoE. Both outputs match native exactly, as do all captured
routing and shared-expert stages. Reconstructing the HC read from token 9764
and the exact native attention output reproduces AFM's original mismatch.
This separates the expert calculation from its input.

Expanded native capture now retains the four HC injection gates. Frozen,
capture-off and capture-on responses and all 64 raw score arrays still match.
Native binary SHA-256:
`81b42611b52be3e01e19eb64a81033a4df5a31934188238258f09495a9798871`.

| First-decode layer-0 comparison | Different / values | RMS |
| --- | ---: | ---: |
| Attention input, reconstructed vs native | 0 / 2,560 | 0 |
| HC injection gates | 2 / 4 | 0.00049208116 |
| Next MLP input | 1,682 / 2,560 | 0.002306329 |
| Reconstructed MLP output vs captured AFM | 0 / 2,560 | 0 |
| MLP input with only native gates substituted | 0 / 2,560 | 0 |
| MLP output with only native gates substituted | 0 / 2,560 | 0 |

The native kernel computes sigmoid in FP32, rounds it to BF16, and then
multiplies by two. AFM's fused gate instead performs low-precision exp,
addition and division. The gate substitution confirms the local source of
divergence; it does not establish that changing it repairs all later tokens
or coding quality. An instance-level default-false diagnostic changes only
this formula, preserving kernel launches and caching the policy in the
prepared-chain key. It is set before compiled owners are traced.

The earlier gamma-rounding hypothesis was ruled out: checkpoint layer-0
normalization scales are identical under the tested expressions. Two failed
harness attempts are retained: direct Module parameter assignment trapped;
the attempted MLXArray update did not compile. Neither changed model files.
Both mutations were removed before the successful gate isolation test
(2.455 s, no failures). No serving defaults have changed.

Evidence under the existing root: `QwenOpeningFirstDecodeMoENative20261024`,
`QwenNativeDecodeMoE20261024`, `QwenOpeningFirstDecodeHCNative20261025`,
`QwenNativeDecodeHCGates20261025`, and corresponding source/build/test logs.
The formula change still requires enabled/control and full-prefix checks.


The formula implementation passes the focused test (release build 188.48 s;
test 2.685 s): all four gates, all MLP inputs and all MLP outputs match native
exactly. Switching back to the old formula in the same process reproduces the
old gates exactly. Disabled full-model control preserves all 320 prior score
arrays (test 34.954 s).

| Fixed-prefix arm | Control winners / 64 | Corrected gate winners / 64 | Control mean RMS | Corrected mean RMS |
| --- | ---: | ---: | ---: | ---: |
| Production 30+1 | 56 | 59 | 0.603458 | 0.558301 |
| Joined 31, F32 state | 58 | 58 | 0.597363 | 0.609586 |
| Joined 31, BF16 state values | 58 | 59 | 0.472641 | 0.504906 |
| Reconstructed 30+1 | 56 | 59 | 0.603458 | 0.558301 |
| Reconstructed 30+1, BF16 state values | 62 | 59 | 0.458862 | 0.539180 |

The enabled full-model result is mixed, not a promotion gate. Layer 0's three
captured boundaries now match exactly. The next first-decode mismatch is
layer 1's attention input: 1,259/2,560 values, RMS 0.0013312501. Layer 1 is the
PLE boundary; its incoming stream, lookup and normalization must be separated
before attribution. In the matched joined-31/BF16 arm, initial scores remain
exact but first-decode RMS only moves from 0.357259 to 0.339074. No end-to-end
speed, quality or parity improvement is claimed. Default remains false.

Evidence: `QwenNativeDecodeHCCorrected20261025`,
`QwenOpeningHCGateControlScores20261025`,
`QwenOpeningHCGateCorrectedScores20261025`,
`qwen-first-decode-hc-gate-comparison-20261025.json`. Native expanded capture
also preserves all 203 previous intermediate tensors exactly.

The same focused assertions also pass with the separate native-chain diagnostic
enabled (`QwenNativeDecodeHCChain20261025`). The formula policy is part of
the chain cache key; the false/true/false sequence preserves the legacy result.


### Materialized stream write isolation, 2026-10-25 00:26 EDT

Expanded captures retain the existing deferred-tail residual and injection
outputs and the existing PLE intermediates. No compiled outputs, flushes or
extra forwards are added. Native frozen/off/on controls preserve all 64 score
arrays; AFM preserves all 320 gate-candidate score arrays. All previously
captured intermediates also remain identical. Build-only Swift qualification
completed in 146.00 s (fixture deliberately absent; test skipped), followed
by the actual fixture-driven release test.

Layer 0 attention read/output, both sets of four injection gates, the MLP
output and the 10,240-value MLP residual all match native exactly. The next
mismatch is the materialized write just before layer 1 PLE: 1,396/10,240
values, RMS 3.9189748e-5. PLE lookup embedding, normalized keys and projected
values match exactly; its query inherits the stream difference. This rules
out lookup content as the first difference at this boundary.

On those exact native operands, a CPU BF16 round-to-nearest-even calculation
reproduces native's PLE input exactly when it rounds the product before the
residual addition. The single-round expression differs in the same 1,396
positions. This agrees with native `hcWriteCallback` (multiply then add) and
AFM's `injectKernel` (FP32 product plus residual, narrow once). The pending
write path already rounds the product; the materialized write is distinct.

A second instance-level default-false diagnostic tests product rounding in
the existing injection kernel, with no new dispatch or intermediate buffer.
It still needs focused and full-prefix validation. It does not change serving
precision policy or claim recovered coding performance.

Evidence: `QwenOpeningFirstDecodePLENative20261025`,
`QwenOpeningFirstDecodePLEScores20261025`,
`qwen-first-decode-ple-comparison-20261025.json`. Preserved native binary:
`QwenPassiveNativeFirstDecodePLEBinary20261025/mlx-serve`, SHA-256
`19c603622d0640d2d16fd78e6ebe3bc051a64840f8a626b86b210db24bd06930`.


The materialized-write implementation passes the focused release test
(build 187.74 s; test 2.588 s). Its corrected output matches all 10,240 native
values; the default and restored-default outputs reproduce the captured AFM
stream. Disabled full-prefix control preserves all 320 previous score arrays.

| Fixed-prefix arm | Gate-only winners / 64 | Gate + write winners / 64 | Gate-only mean RMS | Gate + write mean RMS |
| --- | ---: | ---: | ---: | ---: |
| Production 30+1 | 59 | 60 | 0.558301 | 0.580782 |
| Joined 31, F32 state | 58 | 59 | 0.609586 | 0.599941 |
| Joined 31, BF16 state values | 59 | 60 | 0.504906 | 0.463317 |
| Reconstructed 30+1 | 59 | 60 | 0.558301 | 0.580782 |
| Reconstructed 30+1, BF16 state values | 59 | 58 | 0.539180 | 0.525417 |

The matched first-decode capture now has exact PLE intermediates and exact
layer-2 boundaries. Layer 1's MLP output differs in only one value (maximum
6.103515625e-5); that difference does not survive to layer 2's captured
boundaries. Layer 3's attention input matches, but its output differs in
1,931/2,560 values (RMS 0.0004821501). This is the next independent boundary
to isolate. First-decode whole-score RMS falls from 0.339074 to 0.284543 in
the joined-31/BF16-state arm. Aggregate results remain mixed; neither change
is promoted and there is no new coding-quality or throughput result.

Evidence: `QwenNativeDecodeHCWrite20261025`,
`QwenOpeningHCWriteControlScores20261025`,
`QwenOpeningHCWriteCorrectedScores20261025`,
`qwen-first-decode-hc-write-comparison-20261025.json`, and
`qwen-hc-write-rounding-scalar-cases-20261025.json`.


### First decode attention capture, 2026-10-25 00:40 EDT

A model-free BF16 residual-write regression test passes at 1, 2 and 16 rows,
including default/true/false/true kernel-cache transitions (0.207 s). The
release build completed in 146.20 s. The selected scalar expectations come
from the independent native capture, not a second copy of the tested formula.

The first-decode attention capture retains the existing projection results,
cache views and attention heads without adding outputs to a compiled closure.
Native frozen/off/on controls preserve response and all 64 score arrays; AFM
preserves all 320 previous arrays (35.060 s). Previous captured intermediates
remain unchanged in both engines. Native capture binary SHA-256:
`a3aa6fa5bc6bca26ae2a3b86bbed61823c0faac1b2acf6f59c4a836229c04965`.

| Layer 3 stage | Different / values |
| --- | ---: |
| Query | 0 / 6,144 |
| Current key | 0 / 512 |
| Current value | 0 / 512 |
| Full cached keys | 0 / 3,410,944 |
| Full cached values | 0 / 3,410,944 |
| Output gate | 0 / 6,144 |
| QSA queries | 0 / 512 |
| QSA pooled keys | 0 / 213,120 |
| Attention heads | 1,234 / 6,144 |

Head RMS is 0.0005583751; maximum difference is 0.015625. This rules out Q/K/V
projection or cached operand differences at this boundary. It does not yet
separate selection from attention reduction. A model-free captured-tensor
test compares native-mask SDPA, AFM direct sparse attention and the actual
AFM selector, with alternating component timings. These are not whole-model
throughput measurements.

Evidence: `QwenOpeningFirstDecodeAttentionNative20261025`,
`QwenOpeningFirstDecodeAttentionScores20261025`,
`qwen-first-decode-attention-comparison-20261025.json`, and
`QwenHCWriteRoundingTests.swift`.


### First decode attention isolation and fallback, 2026-10-25

The captured-tensor release test passes (build 99.46 s; test 0.087 s).
AFM selects exactly the same 512 blocks and 6,662-position mask as native.
Stock masked SDPA reproduces all 6,144 native head values; AFM's direct sparse
kernel differs in 1,234 values and exactly reproduces the captured AFM heads.
This identifies attention arithmetic, rather than selection or cache contents,
as the cause of the layer-3 difference. Thirty alternating component samples
(after eight warmups) give medians 0.352500 ms for stock masked SDPA and
0.366812 ms for direct sparse attention. These include component evaluation;
they are not server throughput or whole-model latency measurements.

A fresh full-prefix diagnostic run uses the existing
`AFM_QWEN_QSA_DECODE_ATTENTION=0` fallback, retaining all prior diagnostic
policies. No serving default or runtime implementation was changed.

| Fixed-prefix arm | Direct winners / 64 | Stock winners / 64 | Direct mean RMS | Stock mean RMS |
| --- | ---: | ---: | ---: | ---: |
| Production 30+1 | 60 | 59 | 0.580782 | 0.573638 |
| Joined 31, F32 state | 59 | 61 | 0.599941 | 0.564778 |
| Joined 31, BF16 state values | 60 | 58 | 0.463317 | 0.491325 |
| Reconstructed 30+1 | 60 | 59 | 0.580782 | 0.573638 |
| Reconstructed 30+1, BF16 state values | 58 | 59 | 0.525417 | 0.500456 |

Layer 3 now matches exactly, but first-decode whole-score RMS slightly worsens
from 0.284543 to 0.288002. The mixed 64-step result prevents promotion. Layer 1
and layer 4 MLP outputs each differ by one BF16 value (maximum
6.103515625e-5); layer 1's difference disappears by layer 2, while layer 4's
survives into layer 5. Later differences cannot yet be attributed independently.
The next isolation uses native layer-4 MLP input and native MoE stage captures.
There is no new coding-quality, API latency, or full-model throughput claim.

Evidence: `QwenNativeDecodeAttention20261025`,
`QwenOpeningStockAttentionScores20261025`,
`qwen-stock-attention-comparison-20261025.json`, and
`qwen-first-decode-attention-isolation-tests-20261025.log`.


### First decode layer-4 expert isolation, 2026-10-25 00:55 EDT

The generalized layer-selectable MoE test builds in 104.53 s and passes in
2.463 s. Frozen/off/on reference controls preserve all 64 score arrays and the
saved response. Reference capture binary SHA-256:
`e3acbe553072857417e780a16c1ad1a6b5ce56b5f889c271943f7159f6532e9f`.
On exactly the native layer-4 input, both ordinary and compiled AFM reproduce
the captured AFM output. Router logits, selected experts, route scores, shared
expert body and shared gate/output all match native exactly. The routed output
alone differs by one value (RMS 1.2063132e-6, maximum 6.103515625e-5).
This excludes the surrounding HC input and compiled shared-expert tail here.

Source inspection identifies a candidate: native `gatherQmvDownReduceSource`
uses eight lanes per output row and four rows per SIMD group, with XOR shuffle
reduction; AFM's group-64 path uses 32 lanes per row and `simd_sum`. The gate/up
formulas and sequential BF16 expert weighting have the same written order.
The lane-layout hypothesis is not yet validated; no default is changed.

Evidence: `QwenOpeningFirstDecodeLayer4MoENative20261025`,
`QwenNativeDecodeLayer4MoE20261025`, and
`qwen-first-decode-layer4-moe-tests-20261025.log`.


### Eight-lane routed reduction experiment, 2026-10-25

A default-false instance policy selects a group-64, singleton-only port of
native `gatherQmvDownReduceSource` (ddalcu/mlx-serve, MIT, revision 02bee553).
The kernel uses eight lanes per output row, hoists packed words, and reduces
with XOR shuffles before the unchanged BF16 route weighting. Group-32,
independent-row batching, and the existing native-chain plans remain unchanged;
the diagnostic uses separate ordinary dispatches when selected. No new user
or environment option was added, and no serving default changed.

On the captured native layer-4 input, both ordinary and compiled experimental
outputs match all 2,560 native values. Restoring the default exactly reproduces
the former one-value difference. Build: 198.38 s; focused test: 2.654 s.
Thirty alternating whole-MoE component samples after warmup show median
0.422479 ms default versus 0.434063 ms experimental (+2.74%); means are
0.417082 and 0.422690 ms. This short measurement does not establish either a
model throughput gain or a stable slowdown. It is not grounds for promotion.
The disabled full-prefix control preserves all 320 prior score arrays exactly
(test 35.450 s). The enabled full-prefix comparison is recorded separately.

Evidence: `QwenNativeDecodeLayer4Lanes20261025`,
`QwenOpeningEightLanesControlScores20261025`, and
`QwenOpeningEightLanesControlScores20261025-control.json`.


Enabled full-prefix experiment passes in 35.457 s:

| Fixed-prefix arm | Control winners / 64 | Eight-lane winners / 64 | Control mean RMS | Eight-lane mean RMS |
| --- | ---: | ---: | ---: | ---: |
| Production 30+1 | 59 | 59 | 0.573638 | 0.616627 |
| Joined 31, F32 state | 61 | 59 | 0.564778 | 0.623552 |
| Joined 31, BF16 state values | 58 | 63 | 0.491325 | 0.137201 |
| Reconstructed 30+1 | 59 | 59 | 0.573638 | 0.616627 |
| Reconstructed 30+1, BF16 state values | 59 | 60 | 0.500456 | 0.489578 |

The reference-matched geometry/state arm improves mean RMS by 72.1%, with
one differing raw winner at decision 56. First-decode score RMS falls from
0.288002 to 0.008032. Of 305 captured intermediate arrays, 303 are exact;
all captured layer boundaries through layer 46 match. Remaining differences:
layer 47 attention output 7/2,560 values (RMS 9.984095e-6), and layer 47 MLP
residual stream 2/10,240 (RMS 1.364788e-5). This motivates capturing layer 47's
attention operands; it does not establish coding parity. Production geometry
has no winner improvement and worse score RMS, so this is not a deployable fix.
The default regression selection passes 3 tests, with 1 optional checkpoint
test skipped and zero failures (0.543 s).

Evidence: `QwenOpeningEightLanesCorrectedScores20261025`,
`qwen-eight-lanes-comparison-20261025.json`, and
`qwen-eight-lanes-default-regression-tests-20261025.log`.


### Last-layer decode attention gate, 2026-10-25 01:12 EDT

The layer-47 native capture preserves frozen/off/on responses and all 64 score
arrays. AFM's added layer-47 capture preserves all 320 prior arrays (35.606 s).
Native capture binary SHA-256:
`ac69b37bce14b16429078499c23a7266789fd4f4fbc063d15d624f639217d8e6`.
All captured Q/K/V, full cache contents, QSA selection, heads and gate operands
match exactly. The isolated output projection reproduces all seven captured
AFM output differences on identical native operands. Using the existing precise
BF16 sigmoid removes all seven differences and matches the native output.
There is one differing sigmoid input: BF16 -6.84375, flattened gate index 1996.
Native `stridedSigmoidGateMul` uses its ordinary MLX sigmoid lookup table;
AFM's compiled output tail evaluates sigmoid inside the compiled expression.

The first focused attempt failed due to a harness broadcast mistake: native
records the gate as [1,1,24,256], whereas AFM's tail accepts [1,1,6144]. The
retry reshapes the same tensor, with no arithmetic change, and passes in
2.499 s. Both failed and successful artifacts are retained. A default-false
model-instance decode-gate policy now tests the existing precise sigmoid in
compiled, deferred and ordinary singleton output paths. Promotion is pending
full-prefix, task and performance qualification.

Evidence: `QwenOpeningFirstDecodeLayer47AttentionNative20261025`,
`QwenOpeningLayer47AttentionScores20261025`,
`QwenNativeDecodeLayer47AttentionRetry20261025`, and
`qwen-first-decode-layer47-attention-retry-tests-20261025.log`.


Combined decode-gate prefix test: disabled control preserves all 320 previous
score arrays (build 186.79 s; test 35.600 s). Enabled run passes in 35.493 s.
All 305 captured intermediate arrays now match native at the first decode,
including both previously differing layer-47 arrays. Whole-score error remains
RMS 0.007985 at this decision; the final mixer/output projection is not covered
by these 305 arrays and is the next capture boundary.

| Fixed-prefix arm | Control winners / 64 | Precise decode gate winners / 64 | Control mean RMS | Candidate mean RMS |
| --- | ---: | ---: | ---: | ---: |
| Production 30+1 | 59 | 59 | 0.616627 | 0.616519 |
| Joined 31, F32 state | 59 | 59 | 0.623552 | 0.623632 |
| Joined 31, BF16 state values | 63 | 63 | 0.137201 | 0.136938 |
| Reconstructed 30+1 | 59 | 59 | 0.616627 | 0.616519 |
| Reconstructed 30+1, BF16 state values | 60 | 60 | 0.489578 | 0.489500 |

The gate intervention resolves a proven local discrepancy but does not change
the 64-step winner counts. No release, coding-quality, or throughput acceptance
is claimed. Evidence: `QwenOpeningDecodeGateCorrectedScores20261025`,
`QwenOpeningDecodeGateControlScores20261025-control.json`, and
`qwen-decode-gate-comparison-20261025.json`.


### Final mixer isolation, 2026-10-25 01:25 EDT

Passive final-hidden retention preserves AFM's 320 score arrays (build 58.86 s;
test 35.730 s). Native frozen/off/on controls also preserve all 64 score arrays.
Native capture binary SHA-256:
`d0804ebdcf3086532602e5da2456d0a3ef97b9273382d7107388e46f94b94c2e`.
A short-lived native instrumentation syntax error was corrected before capture;
its failed build log is retained alongside the successful retry.

The default final mixer differs from native in 1,257/2,560 hidden values even
though every preceding captured layer boundary matches. Selecting the existing
`AFM_QWEN_FUSED_FINAL_MIXER=1` diagnostic reproduces all 2,560 native hidden
values and the full first-decode vocabulary scores exactly. This is an existing
runtime experiment, not a newly enabled default. The reference uses its fused
HC read here, while AFM's final fused mixer defaults off.

| Fixed-prefix arm | Default final mixer winners / 64 | Fused final mixer winners / 64 | Default mean RMS | Fused mean RMS |
| --- | ---: | ---: | ---: | ---: |
| Production 30+1 | 59 | 59 | 0.616519 | 0.616343 |
| Joined 31, F32 state | 59 | 58 | 0.623632 | 0.623297 |
| Joined 31, BF16 state values | 63 | 63 | 0.136938 | 0.135386 |
| Reconstructed 30+1 | 59 | 59 | 0.616519 | 0.616343 |
| Reconstructed 30+1, BF16 state values | 60 | 60 | 0.489500 | 0.489292 |

In the reference-matched arm, score arrays are now bit-exact at decisions
0–5 and 7–10 (10/64). Decision 6 RMS is 0.000374; decision 11 is 0.001884;
decision 12 jumps to 0.265082. The sole winning-token discrepancy remains at
56. No whole-task, API-performance or release qualification is implied.
Next capture targets the onset of later drift rather than rerunning a coding
project before its numerical cause is understood. Evidence:
`QwenOpeningFirstDecodeFinalHiddenNative20261025`,
`QwenOpeningFinalHiddenScores20261025`,
`QwenOpeningFusedFinalMixerScores20261025`, and
`qwen-fused-final-mixer-comparison-20261025.json`.


### Later decode divergence, 2026-10-25 01:44 EDT

Added fixture-selected passive decode indices. Captures at decisions 6, 11 and
12 preserve all 320 prior AFM score arrays (build 104.15 s; test 35.617 s).
Reference frozen/off/on controls preserve all 64 raw arrays and the original
response/usage. Native multi-step binary SHA-256:
`9d05f5bd313391ecd498e8765200de6ec362d7ff476376cbfd585b1896c9b0f2`.
A first launch from the archive directory failed before loading the model
because its original relative dylib paths require the source build layout.
The retry uses the original executable location and verifies its hash against
the preserved copy; it does not alter the binary or library identity.

| Decision | Exact captured arrays | First observed divergence |
| --- | ---: | --- |
| 6 | 304 / 306 | Layer 47 MLP output: 2 / 2,560 values; final hidden: 1 |
| 11 | 304 / 306 | Layer 47 MLP output: 3 / 2,560 values; final hidden: 7 |
| 12 | 213 / 306 | Layer 33 MLP output: 1,880 / 2,560; subsequent layers amplify it |

The compiled tail does not expose its internal MLP read. Exact earlier
captured boundaries therefore do **not** establish that this input is exact.
An isolated layer-33 test on the native MLP input produces the native output
exactly with the eight-lane experiment; its full-forward capture differs.
This directs the next test at the HC preparation boundary, rather than
attributing the large drift to the expert multiplication prematurely.

At decisions 6 and 11, isolated layer-47 forwards reproduce the captured AFM
output, including the small mismatch. A second passive native build captures
MoE stages at layers 47, 47, 33 for these decisions. SHA-256:
`99898d3653ac1c3f3b44e157d196d8fc0ddfd959bd282fca94fe334abe69549b`.
All 64 control scores remain unchanged. Router logits, selected experts, route
scores and shared expert output match on native inputs. The legacy routed
output differs in five values at each layer-47 input; the eight-lane path
still leaves the final output mismatch there. Layer 33 differs in one legacy
routed value, removed by the eight-lane path.

A build-only XCTest attempt using `build --build-tests` failed because the
Xcode package build did not enable testability for required modules. Its logs
are retained. The ordinary reliable `test` wrapper is used for the corrected
HC boundary test; no production binary was changed by these experiments.

Evidence under the experiment root: `QwenOpeningLaterDecodeScores20261025`,
`QwenOpeningLaterDecodeNativeRetry20261025`, `QwenOpeningLaterMoENative20261025`,
`QwenNativeDecodeStep{06,11,12}MoE20261025`,
`qwen-later-decode-comparison-20261025.json`, and
`qwen-later-moe-stages-comparison-20261025.json`.


### HC normalization cause and combined decode result, 2026-10-25 01:55 EDT

The boundary test reconstructs attention/HC from native preceding-layer
operands. Attention read and the 10,240-element written stream are exact;
the layer-33 MLP read differs in 32/2,560 values. Feeding that reconstructed
read into the isolated MoE reproduces all 1,880 captured output differences.
Changing only `precise::rsqrt` to the reference's ordinary `metal::rsqrt` in
the decode HC norm removes all 32 read differences and all 1,880 output
differences. Restoring the original policy reproduces the control exactly.
This is a numerical-path difference, not evidence of cache-state corruption
or proof that one normalization is mathematically more accurate.

The model-instance diagnostic defaults false and is included in the native
chain plan key. It changes neither grouped prefill normalization nor other
models. Source attribution remains in the two Metal kernels. Focused test:
build 189.09 s; completed in 2.706 s. The earlier failed build-only attempt
caused a 311.28 s dependency/test rebuild before the boundary test; normal
wrapper `test` invocations are retained going forward.

Disabled full-prefix control preserves all 320 previous score arrays.
A malformed scratch-path in the first enabled launch was rejected by the
wrapper before running any test. The corrected launch uses a small external
runner with fixed paths, a fresh-output guard, an inference-process guard,
recorded command/revision/diff hash, and only fixture-declared diagnostic
options. No build or result was substituted after that rejection.

| Fixed-prefix arm | Control winners / 64 | Native norm winners / 64 | Control mean RMS | Candidate mean RMS |
| --- | ---: | ---: | ---: | ---: |
| Production 30+1 | 59 | 59 | 0.616343 | 0.626392 |
| Joined 31, F32 state | 58 | 61 | 0.623297 | 0.614931 |
| Joined 31, BF16 state values | 63 | 64 | 0.135386 | 0.008942 |
| Reconstructed 30+1 | 59 | 59 | 0.616343 | 0.626392 |
| Reconstructed 30+1, BF16 state values | 60 | 60 | 0.489292 | 0.489292 |

The matched arm is now exact at 27/64 full vocabulary score arrays, including
decision 12 and the former winner mismatch at 56. At decision 12 all 306
captured arrays match. Decisions 6/11 retain their small last-layer routed
expert mismatch (304/306 exact arrays each). The 93.4% RMS reduction and all
64 matching winners are diagnostic results only; production prompt geometry
is still not qualified and no whole-task/performance acceptance is claimed.

Evidence: `QwenNativeDecodeStep12HCBoundary20261025`,
`QwenNativeDecodeStep12HCNorm20261025`,
`QwenOpeningDecodeNorm{Control,Corrected}Scores20261025`,
`qwen-decode-norm-scores-comparison-20261025.json`, and
`qwen-decode-norm-intermediates-comparison-20261025.json`.


### Routed decode activation cause and opening equality, 2026-10-25 02:15 EDT

The remaining layer-47 expert differences at decisions 6/11 come from the
BF16 sigmoid lookup table. AFM built the fused expert's table with generic
MLX sigmoid; the reference library rounds an edge value differently. The
already qualified, all-finite-BF16 helper now supplies an optional table to
the same fused expert kernel. It is materialized before any compiled trace.
This adds no per-token kernel dispatch; the table contents differ. The
model-instance diagnostic defaults false, and the default table stays intact.

The isolated step-6/11 expert mismatches drop from 2/3 output values to zero
in ordinary and compiled forwards. Step 12 remains exact. Tests restore the
original policy and require the control to reproduce. The disabled combined
control preserves all 320 earlier score arrays. With the complete recipe,
the opening request's joined-31 / BF16-state-value arm is bit-exact across
all 64 complete vocabulary score arrays. At decisions 6, 11 and 12 each of
306 captured intermediate arrays is also exact. Other geometry/state arms
are not exact; a partial recipe can worsen their result.

This proves arithmetic agreement for this fixed-input sequence, not a faster
server or a completed coding task. The production geometry, radix snapshots,
batching, MTP and end-to-end quality still need qualification before promotion.

Regression coverage: 8 selected tests, 7 passed and 1 optional real-checkpoint
fixture skipped. All 65,280 finite BF16 activation values agree with the saved
native oracle. Irregular shapes, restored policies, HC write rounding and
existing MoE group-size paths pass.

Second-request holdout: `cold-r2-57`, 7,255 prompt tokens and 57 reference
steps, retains a mismatch from its first prefill result (matched-arm RMS
0.024707716 initially, mean 0.259641590; 56/57 winner sets overlap). The first
holdout fixture omitted the passive layer-count field. Repeating with all
48 layers selected yields identical records in all five arms, ruling out
that diagnostic setting as the cause. The PLE correction only affects its
actual PLE-bearing layer. The holdout is not presented as parity.

Evidence: `QwenOpeningDecodeActivation{Control,Corrected}Scores20261025`,
`qwen-decode-activation-{scores,intermediates}-comparison-20261025.json`,
`QwenNativeDecodeStep{06,11,12}Activation20261025`,
`qwen-decode-activation-regression-tests-20261025.log`,
`QwenColdR2DecodeAllLayersScores20261025`, and
`qwen-cold-r2-decode-all-layers-comparison-20261025.json`.


### Independent cold-prefix localization, 2026-10-25 02:25 EDT

The prefix harness now derives capture geometry from the qualified prompt,
accepts the second-request fixture for passive capture, and applies the PLE
arithmetic policy independently of the number of layers being inspected.
Optional fixture expectations make exact score counts and winner overlaps
explicit regression assertions. A capture-disabled opening run passes the
64/64 exact-score and winner assertions and preserves all 320 score payloads
from the prior opening run (build 102.39s; test 35.355 s).

The second-request first-chunk capture uses 7,224 tokens followed by the same
31-token native tail. AFM passive capture preserves all 285 score payloads.
Its 200 captured arrays match native exactly: the four normal model boundaries
per layer plus eight PLE arrays. Layer 0 retains all rows; subsequent layers
retain their final row, and PLE retains its final 16 rows. This bounds the
localization; it does not prove that every unretained row or recurrent state
matches. The final 31-token span and carried state remain under investigation.

The first native capture exporter rejected a flattened HC injection tensor
with `InvalidPassiveShape`. Raw score and semantic checks still passed; the
export failure was recorded and that run was not treated as complete evidence.
The corrected exporter saves only the four sequence-shaped prefill boundaries.
Retry frozen/off/on semantic controls and all 57 historical raw score arrays
match. Native capture binary SHA-256:
`ebe016ca1e995d72d54bb786bf679a4ddbf6182d0d912916999e5565abcbdcb8`.

The focused step-12 HC normalization test also passes with the compiled native
HC chain selected (2.772 s), including the exact native read/output checks.
No serving policy changed. No fresh coding or throughput claim is made.

Evidence: `QwenOpeningExactScoreGuard20261025`,
`QwenOpeningExactScoreGuardFixture20261025.json`,
`QwenColdR2PrefillCaptureScores20261025`,
`QwenColdR2PrefillNativeRetry20261025`,
`qwen-cold-r2-prefill-comparison-20261025.json`, and
`QwenNativeDecodeStep12NativeChain20261025`.


### Short-tail activation fix and next boundary, 2026-10-25 02:43 EDT

Full-row tail capture finds the first second-request difference at layer 45:
its MLP input is exact, but 2/79,360 MLP output values differ. Later layers
amplify those two differences. The routed activation diagnostic covered long
prefill chunks and singleton decode, but omitted widths 2 through 32.
Extending the existing false-by-default policy to those short prefill widths
removes both differences. The isolated control reproduces the original
capture, router/indices/scores/shared output remain identical, and restoring
the policy restores the old output. This is a demonstrated causal intervention.

Twenty alternating complete expert-layer samples give medians 1.3292505 ms
(control) and 1.322146 ms (candidate): effectively unchanged, not evidence of
a whole-model speedup. Focused Release test passed in 2.540 s after a 140.18 s
build. The corrected full tail matches all 211 captured arrays; the first 16
complete vocabulary score arrays are now exact. The opening request's
qualified 64/64 score assertion still passes after the correction.

The second-request 57-exact target check fails at 16 exact arrays, with a
later winner mismatch at decision 30. XCTest then exits with signal 11 after
that assertion failure; its cause is not yet established. The run is retained
as a failure. A subsequent non-asserting capture run completes normally and
preserves every one of the 171 score arrays already written by the three
completed arms. It does not convert the failed target into a pass.

The next passive capture retains all 48 experts and all attention layers at
decision 16. Native frozen/off/on controls and all 57 historical score arrays
remain exact. The first difference is PLE's final output (1/10,240 values);
its stream, n-gram embedding, key/query/value, gated value and convolution
input are exact. Layer 0 also matches throughout. This localizes the next
experiment to the PLE convolution/activation boundary, not n-gram lookup.
Native step-16 capture binary SHA-256:
`e8495b7db7e88bb8c7d177bb611847173c4e330099edfaee952b729b2a8e93c4`.

Evidence: `QwenColdR2TailActivation20261025`,
`QwenColdR2TailCorrectedScores20261025`,
`QwenOpeningTailCorrectedGuard20261025`,
`QwenColdR2Step16CaptureScores20261025`, `QwenColdR2Step16Native20261025`,
`qwen-cold-r2-tail-corrected-comparison-20261025.json`, and
`qwen-cold-r2-step16-comparison-20261025.json`.


### PLE convolution activation and cold-prefix equality, 2026-10-25 02:57 EDT

Retaining the existing raw convolution result does not change any of the 285
score payloads. The isolated test reproduces the saved AFM output with the
ordinary activation, including its single different value. Replacing only the
activation with the already qualified precise helper removes that difference.
The offending input is -6.84375, the same BF16 activation boundary found in HC,
attention and routed experts. No n-gram data or convolution weights changed.

The full cold second request now has 57/57 bit-exact complete vocabulary score
arrays in the joined-final31-bf16-state arm, including the terminal decision.
All 405 step-16 intermediate arrays shared by the two captures are exact;
the extra AFM-only raw convolution is excluded from that comparison. The other
four arms remain useful negative controls: none has exact complete score
arrays. This is evidence for the complete recipe, not a partial fix promotion.
The opening regression guard remains 64/64 exact. The PLE diagnostic defaults
to false and has no CLI/environment selection. Serving defaults are unchanged.

Focused activation test: 0.180 s; second-request diagnostic: 37.220 s; opening
guard: 35.423 s. These include diagnostic work and are not throughput benchmarks.
All model runs used the same ddalcu four-bit vision-overlay checkpoint and were
serialized on the GPU. No whole coding project was rerun for this checkpoint.

The earlier post-assertion XCTest crash has more specific evidence: main-thread
Metal teardown overlaps a still-running async test task entering the next arm.
The expectation failure ended the XCTest case before that task unwound. This
is a diagnostic harness lifecycle problem; it is not evidence of a serving
crash. A throwing failure path and an intentional failure run are next.

Evidence: `QwenPLEConvActivation20261025`,
`QwenColdR2PLEConvCaptureScores20261025`,
`QwenColdR2PLEConvCorrectedScores20261025`,
`QwenOpeningPLEConvGuard20261025`,
`qwen-cold-r2-step16-ple-corrected-comparison-20261025.json`, and
`qwen-prefix-assertion-lifecycle-20261025.json`.


### Async failure guard and interior cache capture, 2026-10-25 03:03 EDT

The diagnostic now saves the completed arm before checking expected counts,
then throws a Swift error when the qualification target is missed. An
intentional impossible expectation (production-split exact count -1) produces
exit 1, records precisely one 57-decision arm, and never enters the next arm.
There is no signal 11. This verifies the failure path; it is deliberately not
a passing model result. The subsequent wrapper correctly refuses --skip-build
after that failed run, so the cache test uses an ordinary wrapper build.

The existing model interior-capture adapter processes the full final 31 tokens
and independently captures all 48 layers at N-1 = 7,254. Its complete 57 score
arrays remain bit-exact against the frozen native reference. Thus retaining
the final snapshot does not require splitting the final projection into 30+1.
The opening adapter guard also passes 64/64 (42.593 s); the cold holdout
passes 57/57 (44.537 s). These test the model adapter. The application prefill orchestrator,
restored growing conversations, throughput, and complete coding tasks remain
to be qualified. Do not infer serving acceptance from this result.

Evidence: `QwenPrefixIntentionalFailure20261025`,
`qwen-prefix-intentional-failure-tests-20261025.log`,
`QwenColdR2InteriorCaptureRetry20261025`, and their fixtures/launch manifests.


### Real prefill preparer and cache guards, 2026-10-25 03:15 EDT

`MLXReplayPrefill.prepareWithSnapshot` now has a default-false internal
`captureFinalTokenInline` qualification parameter. It calls the existing
interior-capture adapter only when the complete joined tail fits the caller's
chunk bound and no earlier inline anchor is pending. Unsupported input returns
nil before mutation and takes the unchanged split path. A capture error never
retries a possibly mutated cache. The N-1 snapshot/callback contract remains
unchanged, while the live cache includes the final token. Cancellation after
the joined forward prevents final checkpoint publication.

Using this actual preparer, both frozen-reference comparisons still match all
vocabulary scores exactly: opening 64/64 and second request 57/57. Their
complete diagnostic times are 49.228 s and 51.492 s, not throughput figures.
The second-request build took 140.49 s. No serving caller enables the new path.
Recurrent BF16-valued persistence is still supplied by the diagnostic callback;
a model-owned arithmetic policy remains necessary before API qualification.

The full retained-request/cache suite passes 22/22 in 19.670 s (105.26 s build).
The four new tests cover equality with an unsplit model forward, snapshot
independence after donor decode, restoration, FP32 fallback, a 30-token chunk
limit, cancellation before final checkpoint publication, and a mutating
capture failure without fallback. Existing batching, growing-turn/radix,
vision-wrapper and interior-capture tests also pass. There is no fresh coding
or serving-speed result, and no release/default change.

Evidence: `QwenColdR2PreparedCapture20261025`,
`QwenOpeningPreparedCapture20261025`,
`qwen-prepared-cache-regression-tests-20261025.log` and associated fixtures.


### Model-owned recipe without tuning environment, 2026-10-25 03:28 EDT

The complete serial-AR arithmetic recipe is now consolidated inside the Qwen
model. It owns BF16-valued recurrent persistence and requests a joined final
prefill through the existing capture interface. The VLM wrapper delegates that
preference to its text trunk. The prefill orchestrator retains bounds,
cancellation and fallback. No serving caller enables the recipe at this stage.

The first eligibility guard was incorrect: it assumed the four-bit MoE also
had a four-bit output head. The actual saved `language_model.lm_head.weight`
is U32 [248320, 640] for 2,560 input features, with BF16 scales [248320, 40]:
an eight-bit output head. The failed 4.124 s attempt is retained. Eligibility
now checks loaded four-bit/group-64 experts, the existing eight-bit head,
affine four/eight-bit supporting modules and BF16 floating parameters.
It does not infer precision from the repository name or default config bits.
Fully eight-bit experts are excluded; no eight-bit serving path is changed.

With the corrected guard, the model-owned arm passes 57/57 second-request and
64/64 opening full-score equality. The fixtures supply no manual arithmetic
switches and the launch manifests contain no AFM_QWEN tuning environment.
Recurrent persistence comes from the model, with no harness rounding callback.
Times: 9.776 s and 9.453 s for one diagnostic arm each, including loading and
score capture. These are not comparable with earlier multi-arm elapsed times
and are not serving/decode performance measurements. API/coding qualification,
MTP, retained concurrent batching and vision remain outstanding.

Evidence: `QwenColdR2ModelOwned20261025` (failed eligibility attempt),
`QwenColdR2ModelOwnedMixedPrecision20261025`,
`QwenOpeningModelOwnedMixedPrecision20261025` and their launch manifests.


### Serial HTTP candidate wiring, 2026-10-25 03:34 EDT

The isolated branch now configures the qualified model-owned arithmetic once
at model load for serial, MTP-off serving. Text and VLM wrappers delegate to
the same language model. The loaded-weight guard excludes fully eight-bit
experts and other architectures. No CLI or tuning environment enables the
candidate. MTP/concurrent startup retains the prior behavior; runtime mode
changes, vision, MTP and retained batching are not qualified by this experiment.
Main, installed binaries and the preserved best binaries are untouched.

Release compilation succeeds. The next gate is three fresh-process fixed-input
HTTP trials: frozen reference, this candidate, and the preserved October 9 best
AFM binary. All use the same five captured requests, checkpoint, thinking-on /
MTP-off / prefix-enabled setup and 16,384 output-token limit. Returned tool
calls are saved but never executed. These trials cannot establish coding-task
quality or replace the subsequent fresh-directory coding comparison.


### Fresh HTTP and preserved-binary control, 2026-10-25 03:41 EDT

The fresh fixed-input HTTP trial is not an acceptance result. Parsed tool calls
match the frozen reference on requests 1, 3 and 4. Request 2 is a different
read-only inspection command. Request 5 is materially different: reference
writes `src/orbits.js`, whereas the candidate inspects `OrbitControls.js`.
The generated calls were never executed. Fewer tokens are not proof of better
coding quality or equal progress.

| Request | Reference tokens / seconds | Candidate tokens / seconds | Preserved October 9 tokens / seconds |
|---|---:|---:|---:|
| 1 | 72 / 6.326 | 72 / 6.737 | 73 / 6.305 |
| 2 | 67 / 1.783 | 76 / 2.248 | 79 / 2.136 |
| 3 | 84 / 2.009 | 84 / 2.392 | 171 / 3.544 |
| 4 | 123 / 2.351 | 123 / 2.808 | 82 / 1.881 |
| 5 | 1,824 / 28.663 | 479 / 9.766 | 14,963 / 242.266 |

All 15 requests complete with HTTP 200. Both AFM binaries report identical
cached-token counts on requests 2–5: 6,630 / 7,225 / 7,755 / 7,927. The reference
reports zero in the API usage object; this does not prove it did no reuse.

The candidate's generation logs report 54.7–56.8 tok/s, while the preserved
binary reports 62.0–62.7 tok/s. This is an unacceptable speed concern; no default
promotion is justified. Different outputs prevent treating the elapsed totals
as equal-work coding throughput. Reference raw decode timing is not present
in this run. Matched tool calls on requests 3 and 4 have candidate HTTP times
about 19% longer than reference.

Candidate AFMKit: `41bb389ce81964ecd187b7d4a906731b6e9e82d4`;
consumer: `6839b0566dbd166f11ad4203db9568165ac8ff49`;
binary SHA-256: `ef343650a5412784dbdef23ab8877dad1a400d05c395ad5bd6e4f874907e61fc`.
Preserved AFM binary: `0e8edd40a46341bab51da5ad7725a45e43ba622a7c4ef2eac1d6951646441156`.
Reference: `46d017b5f6e49890e654dbefd2b750eda44b627a20f4ec7a01015a4d0d8ea1e0`.
Every case uses the same ddalcu four-bit vision-overlay checkpoint, thinking on,
MTP off, prefix caching on, and no tuning environment. Full launch records and
raw responses are retained under `QwenModelOwnedHTTPFiveRetry20261025` and
`QwenPreservedBestHTTPFive20261025`. The consolidated comparison is
`QWEN-MODEL-OWNED-HTTP-FIVE-COMPARISON-20261025.md` beside its machine-readable JSON.


### Recurrent-state store experiment, 2026-10-25

The model-owned serial recipe persists BF16-valued state in FP32 buffers.
Moving that rounding into the existing GDN kernel's final store avoids the
separate FP32-to-BF16-to-FP32 graph. The current step's output still uses the
unrounded accumulator; only state carried to the next step is rounded.
The specialization requires one token, scalar gates, no mask, BF16 inputs and
FP32 state. Other requested geometries use the original kernel and casts;
callers that do not request this policy retain the original path.

Release tests preserve output and state bytes across 16 dependent decode steps
and widths 2/17 fallback. Alternating timings for 36 independent state updates
show median 1.2285205 ms with casts versus 0.950291 ms with the fused store
(22.65% less component time). This is **not** a whole-model speed claim.
An earlier grouped post-update rounding experiment was rejected: 1.364438 ms
versus 0.7684375 ms for its original cast control. Rejected source and results
remain outside the repository under `qwen-rejected-grouped-state-rounding-*`.

Full-model teacher-prefix checks after the fused-store change preserve all
57/57 cold-second-request and 64/64 opening-request score tensors exactly.
The warm second request matches 16/64 score tensors and 63/64 winning tokens.
Its first score divergence is decision 16; its winner first differs at 37.
Artifacts: `QwenColdR2FusedPersistence20261025`,
`QwenOpeningFusedPersistence20261025`, `QwenWarmR2FusedPersistence20261025`,
`qwen-fused-state-persistence-timing-retry-20261025.json`.

### Warm request divergence: masked QSA selection, 2026-10-25 04:21 EDT

Passive capture isolates the first differing operation to layer 15 (zero-based)
at decision 16 of the 7,256-token second request. PLE and layers 0–14 match
exactly. Layer 15's query, gate, pooled indexer keys and complete attention KV
cache also match. Its attention output differs in 506/6,144 BF16 values.
All 64 native and AFM raw score payloads are unchanged by passive capture.
Frozen/off/on native controls produce exactly the same response semantics.

The masked decode selector subtracts `block_index * 1e-7` before ranking.
This does more than break ties: captured block 994 scores 114.6330795288086,
while block 1727 scores 114.63313293457031. The bias reverses their ordering.
The native reference's exact radix selector ranks the raw scores and uses the
lower index only for exact ties. AFM's gather selector already follows that
rule, but its masked decode selector did not.

A GPU component reconstruction establishes causality:

| Component | Differences from captured reference |
|---|---:|
| Exact raw-score block selection | 0/512 block IDs |
| Exact selection expanded to mask | 0/7,272 mask values |
| Existing biased masked selector | 8/7,272 mask values (two four-token blocks) |
| Stock attention with exact mask | 0/6,144 output values |
| Stock attention with biased mask | 506/6,144 output values |
| Biased-mask reconstruction versus captured AFM | 0/6,144 output values |

Thus this mismatch is a selection error, not an immediate cache-restore error
or evidence of unavoidable stochastic sampling. It does not establish that
all remaining divergences have the same cause. A regression test now covers
unequal nearby scores, exact ties, complete/incomplete tails and capacity
padding. The implementation fix and full-model rerun follow this red test.

Artifacts: `QwenWarmR2Step16NativeRetry20261025`,
`QwenWarmR2Step16Scores20261025`, `QwenWarmR2Step16Attention20261025`,
`qwen-warm-r2-step16-comparison-20261025.json`, and
`qwen-warm-r2-passive-capture-controls-20261025.json` under the external evidence
root recorded above. The first frozen-copy attempt could not resolve its
relative dylibs; the retry preserves the source library layout and verifies
the loaded MLX library hashes. That startup failure did not execute inference.


### Mask correction validated, 2026-10-25 04:29 EDT

The regression test failed before the fix on all three unequal-score/tail
cases and passed its exact-tie controls. Removing the numeric index bias from
the existing rank kernel fixes those failures without adding a kernel or
changing its dispatch geometry. Five selected release correctness tests pass;
the optional timing test was skipped in this invocation.

The captured layer-15 mask and attention output now match the reference
exactly (0/7,272 and 0/6,144 differences). Full-model checks with explicit exact
score assertions pass **185/185**: opening 64/64, cold second request 57/57,
and cached second request 64/64. The cached case improves from 16/64 exact
score arrays and 63/64 winning tokens to 64/64 for both. No sampled tools are
executed in these checks. This is not a coding-task or throughput acceptance.

Artifacts: `QwenWarmR2Step16AttentionFixed20261025`,
`QwenWarmR2RawMask20261025`, `QwenOpeningRawMask20261025`,
`QwenColdR2RawMask20261025`, and the paired
`qwen-masked-score-regression-{red,green}-20261025.log` logs.
The complete source checkpoint is next rebuilt through the consumer wrapper
for fixed-input HTTP correctness and speed measurements before any further
algorithm changes. Main and the installed binary remain untouched.

A separate remaining difference is known past 8,192 KV tokens: the reference
uses gathered ordinary SDPA, while this serial candidate still forces masked
SDPA. It requires its own controlled comparison; the warm-request fix does
not resolve or qualify that route.


### Rebuilt API checkpoint fc23ebfe, 2026-10-25 04:39 EDT

The consumer release build succeeds in 302.09 seconds. The frozen candidate is
`QwenRawMaskHTTPCandidateBinary20261025/afm`, SHA-256
`6f22e8de397334f3da58c8a92c8bc5a2cd3a1465a74b5ad72b2f9c2cd4333ca1`.
It comes from AFMKit `fc23ebfecb482d4fdc0e266c44f605a2a13dc188` and unchanged
consumer `6839b0566dbd166f11ad4203db9568165ac8ff49`; provenance records source
binary modification time, copy time, resources and build-log hash.

| Request | Reference tokens / seconds | AFM tokens / seconds | AFM raw decode tok/s | Reasoning and parsed calls equal |
|---|---:|---:|---:|---|
| 1 | 72 / 6.335 | 72 / 6.724 | 58.6 | yes |
| 2 | 67 / 1.836 | 67 / 2.088 | 57.8 | yes |
| 3 | 84 / 2.020 | 84 / 2.353 | 57.2 | yes |
| 4 | 123 / 2.374 | 123 / 2.730 | 57.2 | yes |
| 5 | 1,824 / 28.801 | 2,612 / 48.421 | 55.1 | no |

The first four now match complete reasoning text and parsed tool arguments,
not merely token counts. The fifth now writes `src/orbits.js` on both engines,
but the generated programs and reasoning differ; no tools were executed.
The candidate remains below the preserved binary's 62.5–62.7 tok/s on the first
four. The fixed-input workload does not establish end-to-end coding parity.

AFM's cached-token counts remain 6,630 / 7,225 / 7,755 / 7,927. Its STATS
prefill rates divide by the full prompt despite processing only the suffix;
those cached rates must not be quoted as actual suffix prefill throughput.
The Responses usage object also reports zero reasoning tokens despite a
nonempty reasoning item. The text and total token counts are preserved, but
this accounting discrepancy remains unqualified and is not dismissed as model
behavior. Reference raw decode timing is absent from this capture.

Full HTTP artifacts: `QwenRawMaskHTTPFive20261025`; normalized comparison:
`qwen-raw-mask-http-five-comparison-20261025.json`. The warm fifth-request raw
score capture (`QwenWarmR5Native20261025`) matches frozen/off/on semantic
controls. A five-request cache reconstruction (`QwenWarmR5RawMask20261025`)
already differs at decision zero: 0/64 exact score arrays, 60/64 winning tokens.
The first winner is AFM 9764 versus reference 7601. Therefore decoding-route
alignment alone cannot resolve this case; the next capture targets both the
740-token suffix prefill and the final 31-token forward before any sampling.


### 2026-10-25: absolute 8K prefill boundary investigation

The qualified native warm fifth request restores 7,927 tokens and forwards
265 tokens to absolute position 8,192, then 475 to 8,667, then the final 31.
The native `generate.zig:nextChunkEnd` aligns recurrent checkpoint strides in
absolute token coordinates. AFM's existing suffix loop instead forwards 740
then 31. Its QSA prefill gather crossover also uses `>= 8192`, while the
reference uses `> 8192` on this M3 Ultra configuration.

Two fixture-only counterfactuals were measured independently. These are raw
score comparisons on fixed native prefixes, not completed coding tasks:

| Warm request 5 diagnostic | First score-vector RMS error | First-token winner equal | Winner overlap, 64 decisions |
| --- | ---: | --- | ---: |
| Current `fc23ebfe` model profile | 0.791700 | no | 60/64 |
| Absolute prefill alignment only | 0.793733 | yes | 61/64 |
| Alignment plus strict gather crossover | 0.619836 | no | 59/64 |

None restored raw score equality; neither is promoted as a serving fix. The
alignment exists in the optional model diagnostic, and the crossover probe
is an internal test property defaulting to false, not a new user setting.

Passive comparison of the aligned strict case matched all 332 captured stages
for the complete 265-token segment exactly. In the next 475-token segment,
layer zero's four captured boundaries match exactly; the first observed
layer-one attention-read differences affect 666 values on row 12 and one on
row 15. Full PLE intermediate captures are required because the initial PLE
capture retained only the last 16 rows and excluded these differing rows.

Two diagnostic failures are retained and explicitly disqualified:
- Native capture initially requested a 740-token forward that does not occur.
  The ordinary five responses and 64 decision-score capture still qualified;
  only its passive-stage capture failed.
- AFM's first matching width-31 stage could belong to an earlier conversation
  request. Capture selection now additionally gates on the requested history.
  The old passive file is invalid for comparison; its measured raw decision
  scores are unchanged. Fresh outputs use `AlignedStrictFull31Retry`.

External evidence root: `/Volumes/edata/dev/CODEX/codex-local-coding-eval-20261007`.
See `qwen-warm-r5-aligned-prefill-comparison-20261025.json`, the separate
`QwenWarmR5Aligned*` folders, and frozen `QwenWarmR5Prefill*NativeBinary*`
provenance manifests. Diagnostic compilation and inference timing is not a
serving-performance measurement. No baseline binary or historical result was
replaced.


### 2026-10-25: PLE gate mismatch isolated and corrected

Full-row passive captures show identical PLE stream, embedding, key, query,
and value projections. The first difference is the sigmoid gate at segment
row 12, HC group 1: input `-6.84375`. Generic AFM MLX sigmoid produces
`0.001068115234375`; the pinned native library produces
`0.00106048583984375`, altering all 2,560 gated values in that group. This is
the same BF16 operation-rounding boundary already qualified for HC.

`qwen4ExpPLEGateSigmoid` reuses `Qwen4ExpHCActivation.sigmoid` only in the
qualified serial arithmetic profile; other profiles retain generic sigmoid.
No shared MLX operation, checkpoint, user option, or main/release binary changes.
The regression test failed before the fix and passed afterward. The captured
475-row operation reconstructs both old AFM and native outputs exactly across
4,864,000 values each. Focused tests: 9 passed, 3 optional diagnostics skipped,
0 failed (`qwen-ple-gate-activation-{red,green}-20261025.log`).

Full-model checks after this fix:
- Opening: 64/64 exact score vectors and winners.
- Cold second request: 57/57 exact vectors and winners.
- Warm second request: 64/64 exact vectors and winners.
- Aligned strict-boundary warm fifth request: first score vector now exactly
  equal (previous RMS 0.619836); subsequent decode differs from decision 1,
  with 61/64 overlapping winner sets overall. This is **not** parity.
- Existing unaligned serving geometry still differs: 0/64 exact vectors,
  60/64 winner overlaps. The activation fix alone does not align that geometry.

The next source-confirmed distinction is native selected-row SDPA decode above
8,192 keys (`qsaDecodeGatherAttn`) versus this diagnostic profile's forced
masked decode. It needs independent qualification; custom sparse AFM attention
is not assumed numerically identical to native gathered stock SDPA.

The corrected warm final-31 capture now has matching layer-zero boundaries and
PLE intermediates, removing the spurious early-history capture discrepancy.
All 64 native score payloads and all five ordinary responses remain unchanged
with either passive capture version. External reports:
`qwen-warm-r5-full-prefill-comparison-20261025.json`,
`qwen-warm-r5-native-passive-controls-20261025.json`,
`qwen-ple-gate-native-fixture-20261025.json`, and `Qwen*PLEFixed20261025`.
No new serving throughput or completed coding-task score is claimed here.


### 2026-10-25: gathered decode and runtime boundary integration

`Qwen4ExpSelectedRowsDecodeAttention` mirrors the pinned reference's
`qsaDecodeGatherAttn`: ascending selected dense KV rows, incomplete trailing
block, additive BF16 validity mask, then stock MLX SDPA. It does not use the
custom direct-index attention reduction. The qualified serial model profile
uses this path above 8,192 keys and the existing masked path at/below 8,192.
The selection still uses the exact raw-score radix selector. Cache append and
ownership remain with the existing attention caller; the helper only reads.

Two component tests cover all four possible tail lengths, sentinel masking,
no-sparsity decline and unsupported dtype. The fixed-prefix warm fifth request
then matches all 64 raw score vectors, including the previously differing decode
steps. This confirms the decode execution-path distinction independently of
sampling or generated tool execution.

The existing interior-prefill protocol now exposes an optional absolute
alignment stride (default nil). The qualified Qwen serial profile supplies
8,192, and its VLM wrapper delegates this preference for eligible text use.
The request-owned scheduler retains its memory limit, snapshots, cancellation
checks and joined final-token segment. Alignment changes only non-final forward
boundaries; it does not retain extra cache snapshots or alter admission/batching.
The qualified QSA prefill crossover is strict `> 8192`.

Validation without any test-side boundary override or tuning environment:

| Fixed native prefix case | Exact score vectors | Overlapping native winner sets |
| --- | ---: | ---: |
| Opening | 64/64 | 64/64 |
| Cold second request | 57/57 | 57/57 |
| Warm second request | 64/64 | 64/64 |
| Warm fifth request crossing 8K | 64/64 | 64/64 |
| Total | **249/249** | **249/249** |

The seven scheduler-boundary, VLM backoff and selected-row tests pass. Boundary
tests cover restored coordinates, caller chunk caps, unchanged nil policy,
joined final span across 8K, and integer-overflow limits. External artifacts:
`Qwen*RuntimeAligned20261025`, `qwen-runtime-prefill-alignment-tests-20261025.log`,
and `qwen-selected-rows-decode-retry-tests-20261025.log`.

A first compile rejected an incorrect Swift range initializer; it was changed
to MLX arange before the passing build. No failed build was used for inference.
This is a numerical qualification checkpoint, not full coding or speed parity.
Next gate: clean consumer rebuild, frozen binary/resources/source provenance,
and fresh identical five-request HTTP comparison against frozen reference and
preserved AFM. Main, releases and local installed binaries remain untouched.


### 2026-10-25: rebuilt HTTP and fresh coding qualification

The clean release consumer rebuild completed in 136.77 seconds. Frozen artifact:
`QwenMatchedRuntimeHTTPBinary20261025/afm`, SHA-256
`bcee3078cbfc62ccfefcf1448d293c5b7005ce5c3c9dc8f7a783fef591871e6f`.
Provider `aabb2eaf0bc5bd75e4cc33f12dc9644d88bfdd45`; consumer
`6839b0566dbd166f11ad4203db9568165ac8ff49`. Resource hashes and full build
provenance are retained beside the executable. This is a diagnostic release-mode
artifact, not a published release. No installed binary or main branch changed.

All arms used the same ddalcu four-bit qualification checkpoint, MTP off,
thinking on, prefix caching on, temperature 0, top-p 1, top-k 0, and seed 123.
There were no tuning environment overrides. Engine processes ran sequentially.
The pinned reference executable and preserved AFM executable remained unchanged.

Five fixed HTTP inputs, with generated tools never executed:

| Request | Candidate seconds / tokens | Reference seconds / tokens | Preserved AFM seconds / tokens | Candidate raw decode tok/s | Preserved raw decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 6.473 / 72 | 6.302 / 72 | 6.324 / 73 | 58.8 | 62.6 |
| 2 | 2.010 / 67 | 1.801 / 67 | 2.139 / 79 | 58.1 | 62.5 |
| 3 | 2.277 / 84 | 2.053 / 84 | 3.574 / 171 | 57.6 | 62.5 |
| 4 | 2.691 / 123 | 2.413 / 123 | 1.879 / 82 | 57.3 | 62.3 |
| 5 | 27.463 / 1,603 | 28.816 / 1,824 | 242.228 / 14,963 | 61.6 | 62.0 |

The first four candidate responses match the reference's reasoning and parsed
calls exactly. The fifth shares 2,111 reasoning characters before differing;
it is not output parity. Compared with `fc23ebfe`, fifth-request candidate
elapsed fell from 48.421s to 27.463s and raw decode rose from 55.1 to 61.6 tok/s.
The short-request raw decode deficit against preserved AFM remains visible.
Reference raw decode timing is unavailable in this frozen serving profile.
API elapsed is not raw decode, and AFM's cached prefill-rate display includes
reused tokens; neither is substituted for an unmeasured throughput value.

Fresh independent Orbital tasks, with the pinned Codex 0.162.1 client, frozen
launcher/task/harness, new empty project path each time, and unchanged
180k-output-token / 60-minute / 240-request task limits:

| Arm | Task seconds | Output tokens | Requests | Original checks | Native decode tok/s | Client exit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Candidate aabb2eaf | 815.743 | 44,274 | 50 | 9/11 | 60.149 | 0 |
| Frozen reference | 808.060 | 49,041 | 48 | 4/11 | not logged | 0 |
| Preserved 136d8e55 | 297.219 | 17,326 | 5 | 2/11 | 62.277 | 1 |

Candidate elapsed is 0.95% greater than this reference run, but **no arm passed
all checks** and a single triple does not establish parity or variance bounds.
Candidate native times: 736.066s generation, 47.846s prefill; 1,098,441 of
1,134,710 prompt tokens reused (96.8%). No HTTP/API errors occurred in any arm.

Failure evidence is preserved without repairing generated projects:
- Candidate: diagnostics return `boreal` where the test expects `Boreal`;
  the mobile screenshot also hit the unchanged 2.5s evaluation timeout,
  preventing the final mobile-overflow assertion. The scene rendered and the
  other nine checks passed. Screenshot failure needs a separate untouched-output
  recheck; it is not silently relabeled as a pass.
- Reference: `new OrbitControls(renderer.domElement, renderer.domElement)`
  supplies a canvas instead of the camera. The caught initialization error is
  `Cannot read properties of undefined (reading 'clone')`, visible in the saved
  screenshot; no scene or diagnostics initialize. This is a generated-code
  defect, not an inference-server crash or an evaluation-only failure.
- Preserved: request five consumed all 16,384 output tokens as reasoning, with
  no forced reasoning transition. It returned `status=incomplete` /
  `max_output_tokens`; Codex failed its turn. Its shorter task duration is an
  early failure and must never count as a faster completed project.
- Candidate and reference each had one forced reasoning transition. The
  candidate also emitted two undeclared tools and encountered a shell-policy
  rejection; the reference had no client tool-router errors.

The first model-visible request is identical across today's candidate and
reference (normalized SHA-256
`20096f2fb9496f0f57888c96645f1d3a8c07b9082a7e3d75f6f5fa61ece77c5c`), and both
return the same first response. Later inputs differ in tool-result chunk IDs
before their trajectories diverge. Against the previous October 24 run, the
opening prompt differs in Codex's injected current-date field. These are
observed input differences, not proof that they alone explain timing variance.
A follow-up comparison sends today's candidate inputs unchanged to both engines.

Evidence under the existing external root:
`QwenMatchedRuntimeHTTPFive20261025`, `QwenMatchedPreservedHTTPFive20261025`,
`qwen-matched-runtime-http-five-comparison-20261025.json`,
`QwenMatchedRuntimeOrbitalTriple20261025/{REPORT.md,analysis.json,pins.json}`,
and the three `OrbitalMatchedRuntime{Candidate,Reference,Preserved}20261025`
folders. Source, prompts, interactions, generated code, screenshots, evaluation
failures and binary identity remain available for audit.


#### Same inputs from the fresh candidate task

`QwenMatchedRuntimeActualInputFive20261025/comparison.json` records both engines
receiving the candidate task's exact first five request histories. All output
counts match: 58, 65, 71, 209 and 8,277 tokens. Reasoning and parsed calls match
for requests 1, 2, 3 and 5. Request 4 differs only in a tool argument, `head -40`
versus `head -60`. The entire 8,277-token fifth response, including the forced
reasoning transition and following tool call, matches. The reference therefore
also reproduces this long planning episode when given the same input.

| Request | AFM seconds | Reference seconds | AFM elapsed overhead |
| --- | ---: | ---: | ---: |
| 1 | 6.196 | 6.098 | +1.6% |
| 2 | 2.016 | 1.758 | +14.6% |
| 3 | 2.027 | 1.761 | +15.1% |
| 4 | 4.267 | 3.696 | +15.5% |
| 5 | 136.639 | 126.349 | +8.1% |

These are API elapsed values, not native reference decode throughput. This
isolates prompt sensitivity from some engine differences; it does not prove
all subsequent task histories equivalent or turn one coding pair into a
variance baseline. The known fifth-request difference on the older fixed
history also remains documented above.

A fresh browser-only recheck of the untouched candidate output reproduced both
original failures under the same 2.5s screenshot timeout (9/11 again).
`QwenMatchedRuntimeBrowserRecheck20261025` retains separate evidence; the
original evaluation was not overwritten. The next bounded performance
experiment replaces only the qualified serial profile's dense decode mask
construction with the already-qualified exact radix block selection expanded
to the same mask. It must preserve score vectors and serving output, and
improve measured serving latency, before retention. Other quantizations and
MTP/batched profiles remain outside that experiment.


### Serial radix-mask experiment, October 25

The qualified serial arithmetic profile now tries the existing exact radix
selector followed by block-to-boolean-mask expansion before its old quadratic
rank-mask fallback. The attention operation, scores, tie ordering and numerical
profile are unchanged. The new helper only accepts one row whose visible length
is the full key span, with the exact complete-block count. Padded capacities,
batched requests and unqualified profiles retain their existing behavior. The
existing mask-fusion disable switch still applies; no new setting is introduced.

Release qualification: 11 tests passed, 3 optional diagnostics skipped, no
failures. New host-oracle cases cover 513/1024/1818/2048 blocks, near ties and
exact ties, every remainder modulo four, and rejected geometries. All 249 fixed
native score vectors and winner sets remain exact (64 opening, 57 cold second,
64 cached second, 64 cached fifth crossing 8K).

Evidence: `qwen-radix-mask-qualification-20261025.log` and
`Qwen{Opening,ColdR2,WarmR2,WarmR5}RadixMaskVerified20261025` under the existing
external artifact root. An initial run correctly stopped before inference
because the reused manifest pinned the previous candidate source hash. Its
failure log is retained; separate experiment manifests update only that source
identity. Native scores, histories, model hashes and expected counts are unchanged.

This is an isolated experimental checkpoint. Serving output and latency still
need comparison before the optimization can be retained or promoted.


#### Radix-mask serving result and remaining serial overlap gap

Frozen experimental source `37758bf2a9088490f94fd90e77e2ef9f9b4a56b8`;
binary SHA-256 `9d2d55c066dfcef10377586a7ebbc60f0baee44a1156d0ce15b85cddfc6d7e96`.
All five actual-task fixed-input AFM responses are unchanged from `aabb2eaf`.
The latest long response is 8,277 tokens and identical to the reference: AFM
137.434s, reference 124.178s (+10.7% elapsed); no long-response gain is claimed.

Three balanced fresh-process cycles for the first four requests confirm:

| Request | Previous AFM median s | Radix AFM median s | Reference median s | AFM time change | AFM/reference elapsed | AFM decode before/after tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6.185 | 6.146 | 6.094 | -0.6% | +0.9% | 57.6 / 61.3 |
| 2 | 2.014 | 1.944 | 1.762 | -3.5% | +10.4% | 57.0 / 61.5 |
| 3 | 2.029 | 1.932 | 1.775 | -4.8% | +8.8% | 56.9 / 61.4 |
| 4 | 4.312 | 3.998 | 3.687 | -7.3% | +8.5% | 56.3 / 61.4 |

Every response is unchanged against its own engine baseline. Request 4 retains
the already documented cross-engine argument difference. Reference raw decode
is not logged; API elapsed must not be presented as that metric. Evidence:
`QwenRadixMaskActualInputFive20261025` and
`QwenRadixMaskShortRepeated20261025/{REPORT.md,comparison.json}`. This supports
retaining the mask change on the experimental branch, not a release or a claim
of full coding-task parity.

A separate bounded host profile attributes roughly 5.3ms/token to graph
construction and 8.6–9.2ms/token inside submission. Submission includes runtime
backpressure and cannot be called CPU compute or pure GPU time. The follow-up
schedule sweep preserves output but disabling AFM's eight-layer submission
schedule makes it slower; four layers is too small a gain to justify a default
change on this single diagnostic pass. The first sweep launcher accepted but
did not apply its override; `QwenRadixMaskLadderDiagnostic20261025/VALIDITY.json`
marks those runs invalid as a sweep. `QwenRadixMaskLadderVerified20261025`
records the corrected, checked child-environment overrides and results.

The reference source identifies a separate overlap mechanism:
`src/generate.zig::lazyForward` builds the next graph on private PLE leaves,
then calls `flushDeferredPle` after graph construction. AFM's serial path still
loads PLE eagerly; its existing private-leaf barrier is currently limited to
multi-token verification. The next isolated experiment reuses that barrier for
the qualified serial mapped-table path, fills all leaves before dispatch, and
checks exact score/output equivalence before any performance conclusion.


#### Reject serial deferred PLE; preserve the verified mask improvement

Experimental commit `d7630e42ac680afda8d089a1ef7133fc912d8ec7` and frozen
binary SHA-256 `6a2f4f26466e959e23222302443413b1e4515949b6c468c1278ffb50c8956b4a`
passed 16 unit tests (3 optional skips) and all 249 exact-score comparisons with
its active policy asserted. All five serving responses also remained unchanged.
However, it reduced short decode to roughly 56–57 tok/s and increased the
identical 8,277-token response to 147.071s versus reference 126.151s (+16.6%).
The preceding radix-mask binary took 137.434s on that same fixed input.

The serial deferral change is therefore **reverted**, including its policy and
diagnostic additions. Source/test files are verified identical to `a98ae995`
(the retained `37758bf2` mask implementation plus documentation). No installed
binary, main branch, 8-bit path or release was changed. The rejected source,
binary and raw results remain preserved for analysis; they must not be selected
as the current performance candidate. Evidence:
`QwenSerialDeferredPLEActualInputFive20261025/comparison.json` and
`QwenSerialDeferredPLEHTTPBinary20261025/PROVENANCE.json`.

The retained candidate is `QwenRadixMaskHTTPBinary20261025/afm`, source
`37758bf2a9088490f94fd90e77e2ef9f9b4a56b8`, SHA-256
`9d2d55c066dfcef10377586a7ebbc60f0baee44a1156d0ce15b85cddfc6d7e96`.
The reference's deferred ordering alone does not reproduce its speed in this
Swift path. Further attribution is required before any replacement schedule.

#### Fresh coding variance batch: midpoint, October 25

The retained `37758bf2` executable above is frozen for five fresh Orbital tasks
per engine, with alternating pair order. No source, launcher, sampler, budget,
checkpoint or execution setting changes are made during the batch. Each task
starts in an empty project directory; generated programs, every request and
response, client interactions, and independent browser checks are preserved.
The experiment remains MTP-off, thinking-on and prefix-cache-on, with the same
180k-output-token / 60-minute / 240-request limits as the preceding comparison.

The following is **incomplete midpoint evidence**, not a parity or release claim:

| Engine/run | Task seconds | Output tokens | Checks | Native decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| AFM 1 | 1160.924 | 65848 | 10/11 | 60.251 |
| Reference 1 | 529.095 | 31770 | 8/11 | not logged |
| Reference 2 | 915.901 | 56010 | 10/11 | not logged |
| AFM 2 | 424.390 | 23571 | 11/11 | 60.645 |
| AFM 3 | 845.485 | 48189 | 10/11 | 60.358 |

AFM's elapsed variation here is chiefly variation in generated work rather than
a comparable change in native decode rate. AFM 1's final 15 responses account
for 33,753 tokens and 576.94 API seconds, including repeated reviews, repeated
tests and a few edits. All five completed tasks used an identical first
model-visible request. Later tool results contain changing shell timings and
chunk IDs, and generated actions diverge; fixed sampling settings do not make
the full live task inputs identical. This does not exclude an engine defect.

AFM 1, AFM 3 and reference 2 fail the unchanged diagnostic assertion requiring
`Boreal` rather than `boreal`. Reference 1 records a negative-animation-delta
exception and a blank 3D viewport; its rendered-scene, raycast and exception
checks fail. AFM 2 passes every check. Reference 2 has one malformed client tool
call; AFM 3 has one unsupported `write_file` call. No API error is recorded in
these five tasks. Generated code is not repaired and scores are not adjusted.

The mask optimization only changes the qualified path through 8K context;
the gathered attention path above 8K is unchanged. The short-request speedup
must not be presented as a corresponding long coding-task gain.

Evidence: `QwenRadixMaskOrbitalFiveEach20261025/{REPORT.md,analysis.json,status.json}`
under the existing external artifact root. The report updates after each task
and checks executable/client/task hashes and 113 checkpoint-file identities
(metadata, not a new full weight digest). `QUALITY-TRIAGE.json` records the
read-only screenshot/source inspection. `REVIEW-LOOP-DIAGNOSTIC-INPUTS.json`
and `AFM3-EARLY-REASONING-CASE.json` preserve candidate inputs for subsequent
identical-input reference controls. Those controls have not run at this midpoint;
no additional GPU workload is admitted during the live batch.
