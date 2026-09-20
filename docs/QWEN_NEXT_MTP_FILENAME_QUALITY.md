# Qwen Next: the actual filename decision — 2026-09-14

Continuation of [the prefill investigation](QWEN_NEXT_MTP_PREFILL_QUALITY.md),
on PR #123 and issue #125. **Not a default promotion or quality certificate.**
This investigation separates model arithmetic, seeded sampling, and the
small repeated-seed quality screen. It does not weaken the wrong-file check.

## Why the previous probe was insufficient

The previous 32-token teacher-forced probe started after the filename decision
on a correct continuation. It did not explain the observed wrong filename.
The new fixtures reconstruct the original chat prompt (4427–4433 tokens),
then decode the actual 19-token common JSON prefix ending at `"file": "`.
Those 19 tokens are **not** folded into prompt prefill.

Prepared token IDs come from the frozen tokenizer responses. We checked that
removing the retokenized common prefix recovers each saved API prompt count,
and that prefix plus continuation retokenizes identically to the saved text.
The exact checkpoint remains:

```text
/Volumes/edata2/models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit
```

Temperature 0.6, top-p 1, top-k disabled, thinking off, C1, prefix off,
4096-token prefill chunks, MTP depth 3, and the recorded M25 experimental
preset. This is not an unset-environment default-performance result.

## Arithmetic at the actual decision

Each task is independently prefetched for ordinary decoding and strict/batched
width-4 verification. Four alignments place the decision in every verifier
row. All preceding tokens are fixed. Future padding uses the saved answer;
these controlled alignments are not assumed to be actual acceptance cycles.

| Requested file | Ordinary correct-token probability | Batched range over four alignments | Actual sampled MTP cycle |
|---|---:|---:|---:|
| cache.swift | 55.17% | 49.97–60.24% | 49.97% |
| retry.swift | 83.72% | 80.92–90.52% | 90.52% |
| queue.swift | 81.11% | 81.11–81.11% | 81.11% |
| stream.swift | 90.81% | 90.81–94.86% | 90.81% |
| limits.swift | 94.83% | 94.83–95.76% | 95.76% |

These probabilities concern the **first filename token**, not whole-answer
correctness. `services` is the dominant incorrect alternative. Arithmetic
changes the distribution, but does not consistently push it toward errors.
The strict policy also differs from ordinary decoding (maximum total
variation about 0.0995 here); its name must not be read as a full-model
bitwise-equivalence guarantee. Its public documentation now states that limit.

Production categorical temperature scaling promotes these BF16 logits to
FP32. An explicit FP32-first probability calculation agrees to negligible
error; a missing temperature promotion is not the explanation in these cases.

## Reproduce the real MTP path, not just an approximation

An internal test-only sampler seam selects a recording delegate once at
session initialization. No observer or additional branch is placed in the
decode loop, and public generation continues to select its existing sampler.

The test loads the real embedded draft head and executes normal speculative
cycles. It asserts all 28 saved API tokens match both the observed session
and an unobserved session with the same seed. All five cases pass. At the
filename, every case uses sampling call 5, row 2 (zero-based), with prior call
output positions `[0, 1, 5, 9, 13, 17]`. The captured filename probabilities
agree exactly with the corresponding teacher-forced alignment above.

```text
identical prompt -> bounded prefill -> 19 generated prefix tokens
                                          |
                +-------------------------+-------------------------+
                |                                                   |
        ordinary target logits                           MTP target logits
                |                                                   |
          55.17% cache                               49.97% cache (example)
                                                                    |
                                                     request-local random draw
                                                                    |
                                                     sampled filename token
```

Synchronous capture is diagnostic overhead, **not** a performance measurement.
Observer equality is checked rather than assumed.

## The repeated-seed cluster is now explained

MTP's multi-row categorical sampler adds a Gumbel random perturbation to each
temperature-scaled token logit and selects the highest result. All four
seed-73 fixtures reach the filename with the same PRNG key, array shape and
row. Consequently the random perturbation of `services` is **identical**:
6.359271. These are correlated choices, not independent trials.
This is **not shared mutable RNG state between requests**: separate request
RNGs initialized with the same explicit seed deliberately reproduce the same
key sequence. Changing that behavior to hide a benchmark failure would break
seeded replay; no such production change is proposed.

| Task, seed 73 | Correct-token advantage before randomness | Correct-token perturbation | services perturbation | Outcome |
|---|---:|---:|---:|---|
| retry.swift | 2.2917 | 0.2503 | 6.3593 | services |
| queue.swift | 1.4583 | 0.0313 | 6.3593 | services |
| stream.swift | 2.2917 | 4.9381 | 6.3593 | stream |
| limits.swift | 3.1250 | -0.0436 | 6.3593 | services |

The positive logit advantage is `(correct_logit - services_logit) / 0.6`.
The large reused perturbation explains even the limits case, whose correct
token had 95.76% probability. It does not establish a biased sampler.

The frozen-logit test then:

1. Replays **every recorded sample** without model weights and checks exact IDs.
2. Reconstructs Gumbel-max using the captured call's derived key and checks IDs.
3. Draws 8192 samples through each of the single-row inverse-CDF and multi-row
   Gumbel branches for each of five full 248320-token vocabularies: **81920 draws**.
4. Compares the correct/services frequencies to probabilities independently
   calculated in CPU Double arithmetic, using predeclared six-sigma tolerances.

All checks pass. Maximum observed frequency error is below 0.7 percentage
points. This rules out a large sampling-law error in these captured decisions,
not every possible RNG issue or arbitrarily small statistical deviations.

## Reference repeatability controls

The initial distinct-seed, 40-output-token screen completed AFM MTP at 39/50
filename checks, with 5/5 greedy controls. The reference short-run control then
failed to reproduce the historical 512-token answer: seed 73 selected
`services/module_0.swift` instead of `cache.swift`. The harness stopped safely.
That incomplete run is retained, not combined with a full-budget reference
score. Budget, adaptive execution and reference RNG state must be distinguished.

An eight-request control alternated 512/40/512/40 budgets for seed 73, then
seed 42, in one fresh reference process. Seed 73 chose the wrong filename
on all four requests; seed 42 chose correctly on all four. The two full-budget
responses per seed were not byte-identical. Thus the mismatch with the
historical answer cannot be attributed simply to the 40-token cap. This is a
repeatability limitation, not proof of an incorrect marginal sampling law.

The inspected [reference source](https://github.com/ddalcu/mlx-serve/blob/1ec580a8b7f5f051daef892310660bb62b2ece6c/src/generate.zig)
(`1ec580a8`, `mtpBatchedAcceptGraph`) uses a
request-seeded host acceptance PRNG but passes a null/global MLX key for its
batched residual/bonus categorical draw. Its startup seeds global MLX from
time. This source observation cautions against treating seed equality as a
cross-engine replay contract; it is not a claim that the reference's marginal
sampling distribution is incorrect, nor proof that this research checkout
exactly matches every instruction in the frozen release binary.

## Completed four-way comparison

The follow-up retained the original **512-token request budget** and saved full
answers, with 50 distinct seeds derived before running from task/trial labels.
All four arms used identical request payloads. Five greedy controls per arm
are separate; all pass both identity and unique-key structure checks.
**220/220 measured runtime requests completed**, plus eight warmup/control
requests. Every server exited cleanly; resource guards recorded no collision
or low-memory stop. The final arm ran after the first three, not concurrently.

| Mode | File + request ID / 50 | Exact unique-key JSON + identity / 50 | Output tok/s, including prefill | Valid tasks/s | Median TTFT, s | Peak process RSS, GiB |
|---|---:|---:|---:|---:|---:|---:|
| AFM, MTP on | 39 | 37 | 29.57 | 0.1283 | 3.568 | 71.61 |
| Reference, MTP on | 43 | 41 | 29.76 | 0.1491 | 3.479 | 68.30 |
| AFM, MTP off | 40 | 38 | 26.58 | 0.1213 | 3.518 | 69.13 |
| Reference, MTP off | 42 | 40 | 27.07 | 0.1294 | 3.459 | 66.93 |

Valid tasks/s here means unique-key structural successes divided by total
sampled request wall time, including unsuccessful requests. It is **not** an
AI semantic-judge score. Output tok/s is total emitted output tokens divided
by total request wall time: neither decode-only nor concurrent aggregate speed.
Process RSS is not full unified-memory/GPU allocation accounting. MTP depth is
capped at three; the reference retains its adaptive controller.

The initial structure audit used ordinary JSON loading, which silently keeps
the last duplicate key. Manual answer review caught repeated `diagnosis` in
AFM MTP case 38 and repeated `diagnosis`/`fix` in AFM AR case 53. Rejecting
duplicate members corrects the initially reported structure counts **38 → 37**
and **39 → 38**, respectively. Reference counts remain 41 and 40. Both audit
versions are retained; the final checker includes duplicate/nested-duplicate
and fenced-JSON regression assertions. No generated response was repaired.

Other identity-passing structure failures: AFM MTP omitted `fix`; AFM AR
misspelled `diagnosis`; each reference arm produced one extra-field answer
and one fenced answer. The prompt requires exactly five keys and raw JSON.
This was **not a grammar-constrained decoding request**.

On this screen AFM MTP is 0.64% behind reference in output tok/s, but 14.0%
behind in structurally valid tasks/s. MTP improves AFM's raw rate by 11.3%
and valid-task rate by 5.7% over AFM AR. None of these C1 numbers qualifies
C15, prefix reuse, broader prompts, or a default change.

The large historical 17/25 versus 24/25 gap does not recur at that magnitude:
the distinct-seed identity difference is 39/50 versus 43/50, while AFM itself
changes by one pass between AR and MTP. All four exploratory 95% Wilson
identity intervals overlap (AFM MTP 64.8–87.2%; reference MTP 73.8–93.0%).
**Overlap is not proof of equivalence/noninferiority**, and five related task
families limit generalization. We do not erase the old failures or declare
quality parity from this result.

## Semantic spot review and interpretation

A fixed subset (trials 0, 6 and 9 × five tasks × four modes = 60 answers) was
read directly. This is post-hoc, non-blinded Codex inspection, not a separately
invoked judge or a semantic score for the full 200 sampled answers. The external
`SEMANTIC-REVIEW.md` records the rubric and specific failures.

- Replies are coherent and usually identify the supplied cause even when
  they choose the wrong file. Some fabricate a relationship between the named
  file and a distractor module: a real task failure, not harmless formatting.
- Some apparently passing streaming replies discard final-chunk content when
  `finish_reason` appears; that can lose valid output. The selected AFM AR
  answers exhibit this problem, and one reference MTP answer is ambiguous.
  One AFM MTP answer omits the fix entirely. Filename checks miss these issues.
- Some cache sketches lack collision-safe composite encoding or describe
  required model/template isolation as optional. Queue tests sometimes fail
  to assert recovered capacity. Without source code, plausible advice is not
  an executable repair certificate.

The evidence points to **three distinct effects**, not one universal MTP bug:

1. A real prefill-policy mismatch was repaired by `49a97c7a` before this run.
2. Repeated explicit seeds correlated the original wrong-file choices; actual
   trajectories and full-vocabulary sampler tests now explain that mechanism.
3. Target arithmetic still changes probabilities with verifier geometry, and
   sampled instruction-following/semantic failures remain in both AR and MTP.
   Their residual cross-engine quality impact is not fully isolated.

No biased sampler or rejection-state corruption has been established by the
current probes. This does not justify removing the quality gate: broader
semantic qualification, remaining verifier arithmetic isolation, and refreshed
C15/prefix/cancellation measurements are still required. A slower numerical
policy or changed sampling/default policy needs an explicit performance and
quality tradeoff decision; none is introduced here.

## Reproducibility and performance scope

External evidence root (untracked):

```text
/Volumes/edata2/afm-benchmarks/qwen-next-mtp-parity-20260909/quality-decision-20260914
```

- `decision-a`: 45 real-checkpoint paths, all four verifier alignments.
- `trajectory-a`: five actual MTP trajectories and unobserved controls.
- `sampling-a`: full-vocabulary seed replay and probability-law checks.
- `independent-screen-a`: completed AFM short screen and failed reference
  repeatability guard; not a completed cross-engine comparison.
- `reference-budget-a`: alternating-budget repeatability controls.
- `independent-full-a`: original-budget, distinct-seed full-answer screen.
- `reference-ar-full-a`: final ordinary-reference arm of the same comparison.
- `AUDIT-v2-unique-keys.json`: final four-arm audit; earlier audits preserved.
- `SEMANTIC-REVIEW.md`: disclosed 60-answer inspection, including failed fixes.
- Guarded launchers, fixtures, release-build logs and exit/resource records.

All 475 evidence files passed SHA-256 verification. The frozen manifest
`SHA256SUMS.txt` has SHA-256
`18c782898ac56d27e0360f3c731194aadd5910fbe01eb5a6e9e747d485c9a9c3`.
Earlier frozen evidence is unchanged. The README distinguishes the benchmarked
API runtime from the newly compiled diagnostic tests and records provenance
limits for the initial untracked test fixture.

The API comparisons use the unchanged candidate Release executable
`10086d6504aa7dbd6e48ed5c8678cc0b1bdfd3a260a64836667de1a141b0b0c6`
and frozen reference
`f3ce20fba143e908110d44d1b8304bf305962a4600d23c9c112934112514bbe4`.
One GPU workload at a time; no installed binary, default or weight changes.
All Swift test builds use the consumer's reliable wrapper. The three new
diagnostic tests are opt-in through the existing test-only path variables;
they add no runtime environment controls.

The initializer seam does not change any production sampling/kernel decision.
The 70-test MTP pipeline run passes (69 passes, one opt-in skip); diagnostic
capture tests pass separately. This is not a new concurrency, prefix-cache,
long-context or release qualification.
