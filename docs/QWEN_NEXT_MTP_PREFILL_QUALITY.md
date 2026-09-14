# Qwen Next MTP prefill and quality investigation — 2026-09-14

Work remains on [PR #123](https://github.com/scouzi1966/AFMKit/pull/123), with
the quality gate open in [issue #125](https://github.com/scouzi1966/AFMKit/issues/125).
**The initial-token discrepancy is repaired in a branch candidate; sampled
MTP is not quality-qualified.** No main merge, default promotion, installed
binary change, release, or cross-model promotion is included.

## What PR #121 contributed

Merged its main checkpoint `6f0bf041` into this workstream as `a561c9eb`,
preserving the earlier branch at
`checkpoint/qwen-quality-before-pr121-20260914` (`1f77590a`). This is not a
merge of every subsequent main change.

PR #121's Apertus prefill/RoPE corrections are architecture-specific. Its
shared quantized-KV fixes do not directly repair this Qwen experiment, which
uses Qwen's own unquantized caches. The useful lesson is to compare actual
prefill arithmetic and prepared token IDs, rather than attribute every
different answer to either model quality or the speculative accept/reject loop.

The Qwen discrepancy remained reproducible after that merge, before the
Qwen-specific repair. An Apertus-focused test is included in the regression
run, but this is not a new full Apertus checkpoint qualification.

## Isolated discrepancy and repair

The frozen diagnostic has **4451 identical prompt token IDs**, ending just
before the requested filename. With temperature zero and one output token:

| Target prefill | Top token | Relevant logits |
|---|---|---|
| Whole 4451-token forward, previous MTP setup | `services` | services 21.500; cache 21.375 |
| 4096 + 355, ordinary prefix-disabled path | `cache` | cache 21.500; services 21.125 |
| 4096 + 354 + 1 diagnostic | `cache` | Same winning token; not the ordinary default path |

No draft or verification cycle has occurred at this point. Applying an 8192
prefill override to ordinary decoding also reproduced `services`. Merely
loading the MTP head, then taking the ordinary fallback, did not.

The layer trace agrees exactly with the actual whole/split forward logits.
Their embedding rows match; the first decoder-layer last-row maximum delta
is 0.0009765625, growing to 0.4375 after 48 layers and 1.75 after final mixing.
This identifies accumulated prefill-geometry sensitivity, not one proven bad
kernel. The final-one-token trace uses a different non-deferred diagnostic
path and has a 0.22265625 logit discrepancy from the actual forward; it must
not be used as an exact production-layer oracle. Whole versus last-row output
projection alone has maximum error 0.0625 here and does not change the winner.

Candidate runtime `49a97c7a` passes the request's `prefillStepSize` through both
the serial MTP lane and the scheduler. Target and head prefill are bounded:

```text
same prompt IDs -> target chunks -> final hidden row -> first target sample
                        |
                        +-> stream[p] + token[p+1] -> head cache
                                                        |
                          later draft -> verifier -> accept / rollback
```

The head's shifted pairs remain continuous across chunks and replay prefixes.
Each chunk materializes cache state and retained last rows to bound the lazy
graph. Exactly one request-local initial sample is taken. Replay snapshots
retain their prefill policy; mismatched policies are rejected rather than
silently reusing a differently prepared state. The direct model-level API
keeps its nil/whole-prompt compatibility behavior; AFM supplies its policy.

No new runtime environment variable is introduced. The existing
`--prefill-step-size` is now honored by this MTP setup; the Qwen architecture
policy in this branch is 4096. This does not restore an older 8192 performance
recommendation or promote a new experimental preset.

## Same-checkpoint API screen

M3 Ultra 512 GiB; exact checkpoint, unchanged weights and mapped sidecar:

```text
/Volumes/edata2/models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit
```

Consumer `9acccfc`; new Release binary SHA-256:
`10086d6504aa7dbd6e48ed5c8678cc0b1bdfd3a260a64836667de1a141b0b0c6`.
Old binary SHA-256:
`4193a708f44f5b59cab393e3e4452d2e469e5e67e7a62faf82ffa607cdc5c93a`.
The old executable was preserved before rebuilding. A mutable binary path or
the development version string alone does not establish provenance.

The same five agentic tasks, five sampled seeds (17, 42, 73, 123, 271), plus
five greedy controls were retained. C1, prefix off, thinking off, top-p 1,
top-k 0, 512 output cap, sampled temperature 0.6, MTP depth 3. The complete
M25 experimental preset is frozen in each launch: **not unset/default tuning**.

| Mode/run | Runtime | Greedy structural | Sampled structural | Sampled output tok/s | Sampled median TTFT |
|---|---:|---:|---:|---:|---:|
| Old ordinary | 30/30 | 5/5 | 18/25 | 23.81 | See original raw data |
| Candidate ordinary | 30/30 | 5/5 | 18/25 | 24.64 | 3.533 s |
| Old MTP, original screen | 30/30 | 5/5 | 15/25 | 27.67 | See original raw data |
| Candidate MTP, first screen | 30/30 | 5/5 | 17/25 | 21.97 | 3.582 s |
| Candidate MTP, timing repeat | 30/30 | 5/5 | 17/25 | 28.83 | 3.584 s |
| Old MTP, matched timing repeat | 30/30 | 5/5 | 15/25 | 28.60 | 3.390 s |
| Frozen reference MTP, historical screen | 30/30 | 5/5 | 24/25 | 29.48 | See original raw data |

Rates are completion tokens / summed request wall time, **including prefill**,
not decode-only or concurrent aggregate throughput. The reference was not
rerun in this increment. Related tasks and repeated seeds are not independent
real-world quality samples. The two additional candidate passes both belong
to seed 42; this is not broad statistical proof of improvement.

All 30 ordinary responses are byte-identical before/after. All 30 responses
for each MTP binary reproduce exactly in its timing repeat, although the old
and candidate MTP responses differ. Every actual request payload matches its
original counterpart. The five initial greedy probes now match the ordinary
control, including `cache` instead of `services`.

The initial large slowdown is retained, not discarded: it did not reproduce
in either the short diagnostic or the full no-debug timing repeat. Its cause
is unresolved. The full repeat is essentially equal in output rate (+0.8%),
but candidate median TTFT is about **0.19 s longer**. Neither the slow first
run nor the faster repeat alone establishes a general performance regression
or win. New MTP peak process RSS is 70.54–70.57 GiB versus 70.34 GiB in the old
repeat; not total Metal memory, a minimum-memory claim, or a leak soak.

The post-hoc debug diagnostic retained original tasks 0, 7 and 8. Draft
acceptance changed 53.4→56.0%, 64.4→51.4%, and 50.8→58.9% respectively. There
is no general acceptance collapse in those three cases. They are diagnostic
requests, not additional independent quality or primary speed measurements.

Structural checks require JSON, exact request ID/filename, and diagnosis,
fix and test fields. They are not semantic AI judging. Wrong filenames remain
failures; no criterion or seed was selected after the fact to hide them.

## Fixed-token verifier and rollback diagnostics

After identical 4096+355 prefill, a 32-token ordinary-greedy continuation is
held fixed across the ordinary and verifier paths. Eight width-4 all-accepted
blocks remove head proposals, random sampling and divergent continuations
from the comparison. Both strict and batched paths preserve **32/32 top-token
choices** with the frozen M25 preset, but their logits are not byte-identical.
Maximum probability total variation at temperature 0.6 is about 0.0557. This
does not mean a 5.57-point task-quality loss; it measures one-position
probability redistribution, and the largest raw logit differences can be on
very unlikely tokens.

M25 explicitly sets attention chunk 2, including in the strict diagnostic.
Consequently this is not a test of every strict/default control left unset.
Do not infer universal strict-policy equivalence from the tiny-model tests,
or universal failure from one overridden real-checkpoint configuration.

A second capture changes only `AFM_QWEN_VERIFY_ATTENTION_CHUNK=1`. Strict
logits and their maximum probability variation are unchanged; attention
grouping alone therefore does not explain that difference. Batched maximum
variation becomes 0.0446, still with 32/32 matching top choices. This setting
is a diagnostic, not a newly selected default or a full quality-score rerun.

For each policy, rejection frontiers accepting 0, 1, 2 or 3 drafts are
independently prefilled. Four subsequent fixed ordinary-token predictions
check the committed state. No production diagnostic branch or synchronization
was added to the decode hot path; the capture lives in an opt-in XCTest.

The corrected M25 capture passes every offset comparison, and **16/16
post-rollback top choices per policy** match the ordinary oracle. Maximum
post-rollback probability variation is below 0.00008 on these positions. The
chunk-1 control also passes. This does not establish exact state arithmetic
or certify arbitrary rejected tokens, contexts, shared batches or the head.

The first rollback capture included eight invalid harness assertions equating
every cache's offset to a token count. Recurrent `ArraysCache` does not have
that contract. The corrected assertion compares every cache offset to its
independently computed ordinary-path counterpart. The original failed log and
artifact remain preserved; changing the harness is not a runtime rollback fix.

## Validation and remaining gate

- Release consumer build passed (140.26 s).
- Focused Release regression suite: **82 passed, 2 optional skips, 0 failures**
  across 84 tests, including bounded head/target pairing, seeded first-token
  sampling, max-token limits, replay and policy mismatch, and Apertus prefill.
- The checkpoint prefill and fixed-token captures ran separately. Compiling
  an opt-in test and observing it skip without paths is not a checkpoint pass.
- New full API screens: **120/120 runtime requests** across four arms; these
  include timing repeats, not 120 independent quality samples. Ten initial
  greedy probes and six debug diagnostic requests are separate; warmups are
  excluded. Owned servers exited normally. Memory/process guards passed but
  are not a proof that every possible system GPU workload was absent.

Still required before promotion: broader independent sampled/semantic checks,
isolation of any remaining verifier arithmetic effect, same-checkpoint
performance confirmation, and renewed C15/prefix replay/cancellation tests
after the initialization change. No concurrency/replay speed result from the
old binary is automatically carried forward. Consult on any material
performance tradeoff instead of silently selecting a slower default.

## Evidence and reproduction

External, untracked evidence root:

```text
/Volumes/edata2/afm-benchmarks/qwen-next-mtp-parity-20260909/quality-repair-20260914
```

This contains the pre-rebuild executable, layer/logit safetensors, complete
request/SSE/usage records, launch settings and binary hashes, guard records,
all timing runs (including the slow one), build/test logs and diagnostic
launchers. `report_quality_repair_20260914.py` in the parent directory audits
payload identity, counts and guards without modifying raw records. The earlier
`QUALITY-20260914-SHA256SUMS.txt` and its referenced data remain unchanged.

For the test-only captures, explicitly provide an existing checkpoint path,
the frozen token-ID JSON and a **new** external output directory through
`AFM_QWEN_PREFILL_QUALITY_MODEL`, `AFM_QWEN_PREFILL_QUALITY_TOKENS` and
`AFM_QWEN_PREFILL_QUALITY_OUT`. These are test inputs, not production tuning
controls. Use the reliable SwiftPM wrapper to build Release tests, then the
guarded external launcher for the selected capture. Do not overlap model
inference with compilation or another GPU owner.
