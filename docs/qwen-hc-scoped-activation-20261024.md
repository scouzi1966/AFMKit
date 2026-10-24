# Qwen HC activation arithmetic investigation

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


## Short HC prefill: reproduced and bounded correction

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

Serving latency and behavioral qualification of this correction are pending.
The frozen five-versus-five results above belong to the preceding binary, not
this correction. Nothing has been promoted to main or a release.
