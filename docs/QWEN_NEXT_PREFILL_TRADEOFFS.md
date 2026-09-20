# Qwen Next: prefill optimization and the benefit/risk gate

## User contract

Continue optimizing both performance and correctness. Warn the user **before
adopting** a proposal when its anticipated gain becomes too small to justify
its risks and re-qualification effort. There is no arbitrary maximum speedup
target, and no license to trade away quality for tokens/sec.

Keep user control through `--prefill-step-size`. Do not introduce automatic
per-request chunk selection or silently override an explicit choice. A stable
default and an optional override are separate decisions. Neither changes here.

## What the reference actually does

The frozen reference's saved launch does **not** set `--prefill-chunk`. Its
source advertises an 8192 ceiling, with memory/architecture caps; the saved
4432-token request records `width=4096` at admission. Therefore the comparison
must not be described as simply "reference 8192 versus AFM 4096."

David Dalcu's `ddalcu/mlx-serve`, source checkout `1ec580a8` (v26.9.2), implements
short-tail merging in `src/generate.zig::nextChunkEnd`. A trailing remainder
under 512 tokens can join the preceding chunk, subject to checkpoint
boundaries. Its prefix-disabled path leaves the last token for the final
forward. The saved log records a 4431-row QSA prefill after admitting width
4096. AFM's ordinary/MTP 4096 policy instead splits this prompt at 4096.

```text
4432-token prompt, prefix off

Reference: nominal 4096 -> short-tail merge -> [4431] [1]
AFM 4096:  explicit bound preserved        -> [4096] [336]
AFM 8192:  explicit larger bound           -> [4432]
```

The three layouts are not numerically interchangeable. Previous teacher-forced
diagnostics found that copying the reference's layout alone did not resolve
the filename probability discrepancy. An 8192 result must not be relabeled as
a test of the reference's exact layout.

Reference evidence is frozen under
`quality-decision-20260914/reference-ar-full-a/reference-ar-reference-mtp-0/`
in the existing September 9 parity evidence root. This is a local source and
saved-runtime comparison, not a fresh qualification of a newer reference.

## Completed bounded experiment

`Scripts/qwen-next-prefill-quality.py` replays the saved five agentic fixtures,
50 distinct sampled seeds and five greedy controls with the same Release
executable and checkpoint. Four arms:

| Order | Prefill CLI value | MTP | Purpose |
|---|---:|---|---|
| 1 | 4096 | off | Reproduce every saved ordinary answer exactly |
| 2 | 8192 | off | Isolate chunk-policy effects without speculation |
| 3 | 8192 | on, depth 3 | Test larger chunks under the quality-study preset |
| 4 | 4096 | on, depth 3 | Reproduce every saved MTP answer exactly |

One excluded warmup per arm. C1, prefix off, temperature 0.6 for sampled cases,
top-p 1, thinking off, 512 output cap. The complete M25 experimental environment
is retained: this is **not** an unset-environment qualification. The depth-4
128-token peak experiment is a different workload and is not directly compared
to these end-to-end agentic rates.

The runner records request payloads, timestamped streams, token counts, text,
launches and hashes. It rejects a changed binary or changed payload. A 4096
control that no longer reproduces its frozen answer stops the experiment.
The lifecycle helper owns server cleanup; memory/process guards refuse
competing inference and stop on low available memory.

Score request/file identity and strict unique-key JSON separately. Full
diagnosis/fix/test semantics are **not** certified by that check. Report output
tokens per request wall second (including prefill), valid tasks/sec, TTFT and
decode separately. Peak RSS is not isolated GPU allocation or a leak soak.
Repeated seeds on five existing task families do not add independent evidence
of general quality or establish statistical noninferiority.

Evidence destination:

```text
/Volumes/edata2/afm-benchmarks/qwen-next-mtp-parity-20260909/prefill-quality-20260915/paired-a
```

### Results and decision

All **220/220 measured requests and four excluded warmups** completed. Both
4096 controls reproduce **55/55 saved answers byte-for-byte**, including the
existing failures. All four owned servers exited zero and resource guards
passed. Ten new CPU-only scoring/launch/audit tests and the eight prior
performance-ledger tests pass. No Release rebuild was necessary or performed.

| Chunk | MTP | Greedy strict | Sampled identity | Sampled strict | Output tok/s incl. prefill | Strict tasks/s | Median TTFT (s) | Median decode tok/s | Post-load peak RSS GiB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | off | 5/5 | 40/50 | 38/50 | 25.00 | 0.1141 | 3.548 | 53.86 | 69.11 |
| 8192 | off | 4/5 | 37/50 | 37/50 | 24.09 | 0.1128 | 3.712 | 55.74 | 69.05 |
| 4096 | on, depth 3 | 5/5 | 39/50 | 37/50 | 28.59 | 0.1240 | 3.592 | 72.87 | 71.60 |
| 8192 | on, depth 3 | 5/5 | 34/50 | 34/50 | 28.47 | 0.1127 | 3.422 | 67.99 | 71.66 |

Changing to 8192 gains three/loses four sampled strict passes without MTP,
and gains two/loses five with MTP. Only 2/50 and 3/50 sampled texts remain
unchanged respectively. The non-MTP greedy cache case now selects the wrong
file. These are paired observations, not proof of a population-wide deficit.

MTP's 8192 output rate is **0.45% lower**, and valid tasks/sec is **9.18% lower**.
Without MTP, output rate is **3.64% lower**. There is no demonstrated memory
benefit; observed post-readiness RSS differs by only about 0.06 GiB. These
samples do not qualify load-time memory or long-context concurrency.

**Decision: reject 8192 as a blanket default/performance recommendation.**
Keep the explicit CLI option and the 4096 default. Do not infer that reference
tail merging has been benchmarked here, or that 8192 is bad on every workload.
The separate depth-4 128-token context result (72.31 → 88.63 tok/s) remains in
the peak ledger with its exact scope. It is not generalized to agentic quality.
No costly cross-model or C15 qualification of a new 8192 default is justified
by this result. This rejects one candidate, not the overall optimization effort.

Timing degradation is retained too: the unchanged 4096 binary and answers
measured **25.00 vs 26.58** prior ordinary output tok/s (-5.92%), and **28.59 vs
29.57** prior MTP (-3.30%). Their cause is unisolated; do not attribute them to
the code, silently discard them, or replace the historical peaks. These are
agentic wall-time rates, not the 128-token prefill/decode context curves.

The read-only auditor produces `audit-a/README.md`, `audit-a/audit.json` and
`audit-a/INPUT-SHA256SUMS.txt` beside `paired-a`. It independently re-scores
responses, verifies payloads/control replay, checks guards and records paired
gains/losses. The hash manifest covers every input file. No semantic AI judge
was used in this increment, and quality parity remains unresolved.

## Prioritization and warning criteria

| Candidate | Benefit evidence | Risk / re-test scope | Decision before results |
|---|---|---|---|
| Explicit larger prefill | Up to 22.57% decode recovery on one depth-4 4K fixture; 2.82% prefill-proxy improvement | Changes sampled trajectory, draft acceptance and memory; requires quality and then cache/concurrency tests | Worth the controlled screen; no default promotion |
| Reference-like short-tail merge | Avoids a small extra forward; no measured AFM win from an implementation yet | Could exceed the explicit chunk bound and change cache preparation; ordinary/MTP/replay must agree | Study, but do not copy into production on assumption |
| Recurrent precision or normalization rounding changes | Arithmetic differences identified; no broad quality or speed win established | Model-wide numerical changes; verification, rejection, prefix restore, long context and other paths need qualification | Do not adopt without a causal quality result and tradeoff discussion |
| Small isolated kernel/scheduling tweaks | Must beat repeat-to-repeat variation on matched prompts | Depends on whether output/state changes and which shared paths are touched | Warn if expected gain is only noise-sized while risk or validation is broad |

A repeatable low-risk improvement can still be worthwhile even when small.
A one-off ~1% result does not justify a numerical policy change or broad cache
rewrite. Consider combined throughput, successful tasks/sec, memory headroom,
tail latency and maintenance cost—not just the best token-rate sample.

Do not silently abandon a promising path, promote a workaround, or spend a
large validation cycle on a weak hypothesis. Record the evidence and consult
the user when the benefit/risk balance becomes doubtful.
