# Qwen Next cohort-aware speculative work

Follow-up on [shared speculative work](QWEN_NEXT_SHARED_SPECULATIVE_WORK.md),
on the same PR #123. Both new controls default off. No installed binary,
release, main branch or other model's execution policy is changed.

## Hypotheses and boundaries

The previous per-request adaptive controller reduced mean verifier group width
from roughly 5.2 to 3.3 and lost throughput. The whole-group state cache reused
more rows with a larger budget but did not improve throughput. These experiments
test whether shared depth epochs and membership-independent row identity help.

```text
serialized Qwen owner
  width-band depth policy ──> idle requests adopt one depth
                              in-flight cycles finish unchanged
  private request states ──> compatible shared head/target groups
             ↑                        ↓
     authoritative rollback     immutable fixed-state banks
             └── UUID + revision + full-acceptance certificate
```

### Cohort depth: `AFM_QWEN_MTP_COHORT_DEPTH=1`

Requires the Qwen MTP scheduler, shared verifier and batched arithmetic policy.
Overrides the older request-local adaptive control when both are requested.
Ready requests use the same selected depth in active-width bands 1, 2–4, 5–8,
9–16 and greater than 16. Depth stays within 1…min(requested depth, 8).
No MTP-off fallback is silently introduced.

Each depth is measured over 32 owner decode steps after nine settling steps.
The score is emitted tokens divided by the measured owner-step durations.
These intervals include shared draft, target verification, token materialization,
deferred head repair submissions and existing GPU waits. They exclude admission,
HTTP dispatch and periods outside the measured block. Lazy GPU execution can
cross step/epoch boundaries: this is amortized owner cost, **not exact GPU time**.
No extra eval/synchronize is inserted for timing. Periodic probes revisit depths,
and a five-percent hysteresis avoids small winner changes.

A session only accepts a new depth with no staged draft, decision or pending
repair. Its target sampler, acceptance rule and request-local RNG are unchanged.
Changing proposal length can nevertheless change sampled sequences and batched
floating-point wording; this is not an exact AR-output equivalence promise.

### Membership reuse: `AFM_QWEN_MTP_STATE_REMAP=1`

Requires independent-attention shared verification plus a positive
`AFM_QWEN_MTP_PERSISTENT_STATE_MIB` (still clamped to 2048 MiB).
It retains the existing four-bank payload limit, but permits a new ordered group
to reuse rows from a prior compatible bank by UUID, revision and a full-acceptance
certificate. Rejected, advanced or newly admitted rows come from authoritative
request state. Real-position attention histories are never stored in these banks.

Same-position rows use the existing functional update path. Reordered/reshaped
groups use supported slice/concatenation operations; 64-bit integer history never
uses unsupported GPU scatter. No quantization, narrowing or arithmetic change is
introduced. An inactive row's identity is retired immediately. Its unused physical
payload may remain until surviving rows leave or the bounded bank is evicted;
all retained bytes are counted. Nothing is retained after all rows depart.

This is **immutable-bank remapping**, not a zero-copy persistent-slot kernel or
an in-place bank allocator. Remapping can still copy the complete recurrent
payload. A hit is therefore not a speedup; counters and end-to-end task rates must
support any recommendation. The bank tracks membership hits, moved rows and peak
retained payload separately from total process RSS.

## Controlled screen

Use the complete W8-L2/M16 recipe in the
[central opt-in matrix](QWEN_NEXT_OPT_IN_MATRIX.md). Keep shared head on and the
old request-local adaptive control off. Change only the listed new controls and
state budget between arms. Same exact ddalcu checkpoint, Release binary, C15,
prefix on, temperature 0, top-p 1, MTP depth 3, 512-token cap, fifteen agentic
JSON tasks plus identical repeats. Aggregate output tokens/s includes the whole
phase; also report structurally valid tasks/s, output lengths, latency and RSS.
Structural checks are not semantic judging.

Evidence root:
`/Volumes/edata2/afm-benchmarks/qwen-next-mtp-parity-20260909`.
New files use the `cohort-work-20260913-` prefix. Earlier manifests and workloads
remain immutable. New wrappers add only the two controls around frozen inputs.

## Results (2026-09-13)

Runtime commit **`bfcb6b5b`**, consumer **`9acccfc`**, binary SHA-256
`0c63e09ab41c6a11b251f54a3a1fa9fd9659b48411e84aa15b4ebb566954c360`.
The paired binary reports development `v0.9.20`, not an installed nightly.
Build time was 131.64 s. The retained byte-copy
`cohort-work-20260913-afm-bfcb6b5b` is a checkpoint, not a complete install package.

All arms below use the same binary and checkpoint. Prefix-on first requests can
already share a prefix; repeats are regenerated responses, not answer caching.

| Evidence suffix | Configuration | First aggregate tok/s | Repeat aggregate tok/s | Structural passes |
|---|---|---:|---:|---:|
| control-a | M16 shared head | 77.25 | 130.65 | 29/30 |
| depth-a | M21 cohort depth | 79.13 | 133.84 | 29/30 |
| bank-a | M22 old bank, 2048 MiB | 77.44 | 124.75 | 30/30 |
| remap-a | M23 remapped bank, 2048 MiB | 77.54 | 130.75 | 30/30 |
| depth-b | M21, before control-b | 78.30 | 133.46 | 30/30 |
| control-b | M16, after depth-b | 78.29 | 129.60 | 30/30 |
| combined-b | M24 cohort + remapped bank | 79.10 | 133.98 | 30/30 |

**210/210 runtime requests completed; 208/210 structural passes** in the seven
fully instrumented screens. Peak process RSS ranged about 69.75–70.05 GiB;
this is not a total Metal-allocation measurement or a minimum-memory guarantee.
All seven process-isolation records show no competing compiler/XCTest/AFM owner.

### Cohort depth: small repeated gain, fragmentation avoided

Matched a/a and reverse-order b/b pairs give first token-rate changes
**+2.42% / +0.02%**, repeat changes **+2.45% / +2.98%**. Average shared group
width rises from **5.14 / 5.03** to **5.31 / 5.42**, rather than the previous
request-local policy's roughly 3.3. Actual policy telemetry records 16/15
completed epochs and 9/8 depth changes. This confirms the changed mechanism.

Repeat structurally valid tasks/s changes **+0.25% / +3.79%**. The first pair
swaps a missing `fix` field for AGENT-13 between first and repeat phases; the
second pair is 30/30 on both sides. Only 4/30 and 3/30 exact texts/token counts
match. These are small workload-specific gains, not a semantic-quality claim
or the overall throughput target being achieved.

Repeat output totals were 2576→2457 and 2698→2677; phase times were
19.717→18.357 s and 20.818→20.058 s. First-phase time in the a pair increased
33.022→33.694 s despite a higher token rate. Do not conceal that tradeoff.

### Remapping: more reuse, no material gain over no bank

Old bank: 174 reused / 4 refreshed rows. Remapping: **238 reused / 79 refreshed**,
including **27 changed-membership hits and 48 moved rows**. Both retain at most
2,081,562,912 bytes of array payload (about 1.94 GiB) in the observed run, and
zero at final shutdown. Its budget remains a payload bound, not a graph bound.

Remapping improves repeat throughput **4.81%** and valid tasks/s **5.81%** over
the old-bank arm, but versus shared-head **without any bank**, token rate is
only **+0.08%** and valid tasks/s **−1.94%**. This repairs the old cache layout's
performance penalty; it does not establish a cache-driven throughput win.

### Combined: functional; no additive speed claim

The fully instrumented combined-b run records 458 reused / 265 refreshed rows,
96 membership hits, 190 moved rows, and average shared width 5.57.
Compared with control-b, repeat tokens/s is **+3.38%** and valid tasks/s **+7.90%**.
Compared with depth-b, tokens/s is only **+0.39%**, valid tasks/s **+3.96%**, and
outputs differ. One combined confirmation is not enough to prefer the extra
bank memory. Both remain separate opt-ins; shared-head remains the simpler
established candidate and cohort depth is the new modest-gain candidate.

## Qualification and excluded evidence

- Focused Release XCTest: **105 executed, 103 passed, two opt-in performance
  probes skipped, zero failures**, 87.09 s execution after 144.72 s test build.
  Includes mixed sampled/greedy depth transitions, cancellation, nonzero
  recurrent-state rollback, remapping, 64-bit history, budget and retirement.
- C6 lifecycle: `depth-lifecycle` and `combined-lifecycle` each pass **120/120**
  API assertions (36 cancel, 42 after-cancel, 42 replay). Actual depth changes
  occur in both. Combined reuses 22 rows, with two membership hits and three
  moved rows. These are separate short-answer contracts, not timed agentic QA.
- Four response spot-checks were coherent and addressed their stated tasks.
  Neither this inspection nor JSON structure constitutes broad semantic judging.

`combined-a` is **excluded from performance qualification**: all 30 requests
completed (29 structural), exit status was zero, but shutdown emitted no owner
counters and the unchanged coverage assertion failed. Its observed 79.08/132.71
tok/s is retained only as incomplete evidence. No assertion was relaxed and no
artifact was overwritten. A fresh `combined-b` completed all counter checks.

Inspection of consumer `9acccfc` found `handleShutdown` launching
`AFMServer.shutdown()` and immediately clearing the CLI run-loop flag, while
the server launches an unawaited Task for asynchronous unload. That sequence is
consistent with exiting before scheduler diagnostics; it is **not a proven
isolation of the one failed run's cause**. It needs a separate consumer shutdown
regression test/fix, tracked as
[maclocal-api #304](https://github.com/scouzi1966/maclocal-api/issues/304).
The failed run is not classified as an OOM or model failure.
No consumer/runtime change was mixed into this frozen comparison.

Evidence manifest: `COHORT-WORK-20260913-SHA256SUMS.txt`, **442 verified entries**,
SHA-256 `3e4309702d7804e88b727fe0dabb724fa08ab60dc6de9b6bf02f9e60393d1c3c`.
The preceding shared-work manifest's 630 entries still verify unchanged.

## Next decision

Do not promote these defaults. Cohort depth merits longer and sampled-workload
qualification. Bank reuse now survives membership changes, but a genuinely
copy-avoiding stable-slot kernel would be a different experiment. Before pursuing
it, measure whether recurrent merge/copy time is large enough to justify the
complexity; current hit counts alone do not establish that. Other models,
prefix-off, larger concurrency and long-context memory behavior are unqualified.
