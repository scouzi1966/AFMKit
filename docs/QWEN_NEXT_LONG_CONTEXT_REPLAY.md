# Qwen Next sampled long-context work and replay limits

Follow-up on the cohort experiments, on [PR #123](https://github.com/scouzi1966/AFMKit/pull/123).
These are branch experiments, not a release qualification or a default-policy change.

## Controlled workload

The checkpoint remains:

```text
/Volumes/edata2/models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit
```

The M3 Ultra 512-GiB machine runs one Release server on port 9998, with no
concurrent build/test/inference process admitted by the existing process-name
guard. The installed Homebrew executable is not replaced. The workload uses
fifteen synthetic coding-review tasks, then identical repeats, after one excluded
warmup. No tools execute. Each request asks for `request_id`, `file`, `diagnosis`,
`fix` and `test` in JSON. The system prompt contains 104 unrelated module entries
to stress selection of the correct file from a longer context.

Actual prompt lengths are **4,427–4,433 tokens**, not the earlier roughly
1.1K-token workload. Sampling is **temperature 0.6, seed 42**, with **top-p 1.0
and 0.95 tested separately**. Thinking is disabled and the output cap is 512.
The cap is not a claim that every response generates 512 tokens. All settings,
messages, outputs, usage and streamed chunks are saved per request.

Concurrency is 15 and prefix caching is enabled. The M16 control uses shared
head, fixed depth 3, window 8 and shared ladder 2. M21 adds cohort-aware depth
only; persistent-state banks and membership remapping remain off.

Rates are **aggregate output tokens / complete phase wall time**, including
admission and prefill. Also inspect structurally valid tasks/s, latency, token
counts and RSS. These rates must not be compared directly with the earlier
short, greedy, cache-hit workload or with single-request decode throughput.

## Concrete replay eligibility limit

The existing `ExactPromptReplayCache` defaults to a **4,096-token maximum
prompt**, independently of its byte budget. A request above that boundary fails
`canStore`; increasing `AFM_QWEN_MTP_REPLAY_MIB` does not change eligibility.
The Qwen scheduler consequently performs full target and head prefill again.

```text
4,430-token request
  → replay lookup: no stored matching state
  → canStore: false (4,430 > 4,096)
  → full target + head prefill
  → no prompt snapshot retained
  → identical repeat follows the same miss path
```

The long-context baseline measured **zero cached tokens in both phases** despite
prefix caching and a 4-GiB replay budget. Server logs show sequential request
prefills around 3.3–3.6 seconds each. Repeated requests are therefore not evidence
of successful prefix reuse in this test. The one small retained entry in the log
does not establish that the measured long prompts were stored.

This is an explicit storage-policy limit, not proof of a broken radix match or
a PLE sidecar problem. The complete speculative snapshot contains target state,
MTP-head state, recurrent history, and the final hidden/stream boundary. Raising
its eligible prompt length must preserve these ownership and replay contracts,
the entry count and byte-budget bounds, and per-request sampling.

### New independent control

`AFM_QWEN_MTP_REPLAY_MAX_TOKENS=8192` raises eligibility for the existing
complete-state replay cache. Unset/invalid stays **4096**; explicit integers are
clamped to **0–8192**. Zero prevents retaining nonempty prompt snapshots.
It requires the experimental Qwen MTP owner, prefix caching and a positive
`AFM_QWEN_MTP_REPLAY_MIB`. It does not enable any of those by itself.

The 16-entry limit and the configured byte budget are unchanged. A larger token
limit is not a reservation or a guarantee that every snapshot fits. Oversized
values remain rejected, and LRU eviction still enforces the budget. The option
is resolved once when the scheduler is created; no environment lookup, lock,
synchronization, math or sampler change is added to decode. A shutdown counter
prints the effective prompt limit to establish that the binary exercised it.

For a controlled test, use **M16**, change only this limit between **4096** and
**8192**, and keep cohort depth off. Do not combine replay and depth changes and
then attribute all improvement to either one. Extending eligibility is Qwen-owner
wiring; the bounded complete-state cache itself remains generic. Other model
replay adapters are not changed or newly qualified by this experiment.

## Quality boundary

All file-identity/JSON failures remain failures. In manually inspected outputs,
many diagnoses and suggested tests are coherent but the response substitutes
`services/module_N.swift` for the specifically requested file. Some stream fixes
are ambiguous about handling the final content token. JSON structure alone
therefore cannot establish semantic correctness.

The first top-p 1.0 pair completed 60/60 runtime responses. Fixed depth passed
15/30 structural checks, cohort depth 14/30. Cohort repeated throughput rose
from 31.66 to 32.46 tok/s, but repeated structurally valid tasks/s fell by 20.9%.
That is not sufficient evidence to promote cohort depth. Neither this result
nor a future replay speedup establishes parity in quality with ordinary
non-MTP decoding or another engine; that requires a separate matched comparison.

The complete first screen:

| Top-p | Depth policy | First aggregate tok/s | Repeat aggregate tok/s | Structural checks | Peak RSS GiB |
|---|---|---:|---:|---:|---:|
| 1.0 | Fixed 3 (M16) | 26.42 | 31.66 | 15/30 | 70.48 |
| 1.0 | Cohort (M21) | 32.07 | 32.46 | 14/30 | 70.76 |
| 0.95 | Fixed 3 (M16) | 31.83 | 31.56 | 10/30 | 70.35 |
| 0.95 | Cohort (M21) | 31.21 | 30.41 | 12/30 | 70.55 |

All **120/120 runtime requests completed** and the per-arm isolation guards
recorded no competing processes. Structural total is **51/120**, not 120/120.
Top-p 1.0 ran control then candidate; 0.95 ran candidate then control. Each
sampling pair has only one run per policy. The 21.4% first-phase token-rate gain
at top-p 1.0 is not a reproduced general speedup. Repeated throughput changes
are +2.53% at 1.0 and **−3.63% at 0.95**. Changed wording and file selection mean
there is no across-workload performance-and-quality recommendation for M21.

## Replay-limit A/B on the rebuilt binary

Runtime **`739cdb6f`**, unchanged consumer **`9acccfc`**. Release binary SHA-256:
`4193a708f44f5b59cab393e3e4452d2e469e5e67e7a62faf82ffa607cdc5c93a`.
Focused Release validation executed **107 tests: 105 passed, two optional
performance probes skipped, zero failures**. Test build was 83.82 seconds;
execution was 86.92 seconds. Consumer Release build was 77.82 seconds.

Both arms use this exact binary and M16 with fixed depth. The only launch
difference is `AFM_QWEN_MTP_REPLAY_MAX_TOKENS=4096` versus `8192`.

| Top-p | Replay limit | First aggregate tok/s | Repeat aggregate tok/s | Repeat wall seconds | Repeat median TTFT | Structural | Peak RSS GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 4096 | 34.61 | 32.62 | 82.87 | 27.31 s | 12/30 | 70.62 |
| 1.0 | 8192 | 34.96 | 96.51 | 27.64 | 0.279 s | 16/30 | 70.49 |
| 0.95 | 4096 | 31.85 | 32.54 | 73.18 | 27.25 s | 10/30 | 70.30 |
| 0.95 | 8192 | 32.44 | 103.66 | 22.16 | 0.294 s | 11/30 | 70.20 |

The top-p 1.0 confirmation reversed the order (8192 first, then 4096):

| Replay limit | First aggregate tok/s | Repeat aggregate tok/s | Repeat wall seconds | Repeat median TTFT | Structural | Peak RSS GiB |
|---|---:|---:|---:|---:|---:|---:|
| 4096 | 34.83 | 33.52 | 75.78 | 27.20 s | 14/30 | 70.59 |
| 8192 | 37.23 | 99.57 | 26.21 | 0.295 s | 15/30 | 70.80 |

This reproduced a **2.97×** repeated token-rate gain and a **65.4%** wall-time
reduction. Both repeated phases passed 8/15 structural checks; structurally
valid tasks/s improved **2.89×**. Output totals differed by +2.8%. These are
two order-reversed top-p 1.0 pairs, not a statistical confidence interval.

All **180/180 runtime requests completed** in these six arms, with **78/180
structural passes** (candidate 42/90, control 36/90). The expanded
limit produced **15/15 complete prompt hits per repeated phase**, totaling
66,447 cached tokens; the default produced zero. The expanded arm also reused
the warmup's exact prompt once in the first phase (4,432 cached tokens), so
that phase is not entirely cold.

Repeated throughput rose **2.96× at top-p 1.0 and 3.19× at 0.95**. Output totals
changed by −1.3% and −3.5%, while wall time fell 66.6% and 69.7%. Structurally
valid tasks/s improved roughly fourfold, but absolute structural quality remains
low and wording differs. This is a removal of repeated prefill, **not a claim
of threefold uncached decode or reference-engine parity**.

The expanded cache retained **3,901,070,292 accounted bytes** (about 3.63 GiB)
across 16 entries, within the unchanged 4-GiB budget. Peak process RSS did not
increase in these pairs, but that does not make retained snapshots free: RSS is
not Metal active allocation and includes other cached/resident pages. Larger
prompts or more distinct requests can still evict entries. No residency locking,
larger memory budget, altered checkpoint or default change was used.

### Cancellation and mixed-mode replay

A separate long-context lifecycle run passed **120/120 assertions**: 36 during
intentional cancellation, 42 after cancellation, and 42 on the next repeat.
It mixed greedy and sampled MTP with ordinary-decoding fallbacks for logprobs,
penalties and stop handling. The canceled stream was intentional, not a runtime
success claim. After cancellation and on repeat, all three eligible MTP
requests reported complete prompt hits above 4096 tokens. This establishes the
tested ownership/replay path, not all possible cancellation interleavings.

## Still to qualify

- Compare these exact long sampled prompts with ordinary decoding and the
  reference before assigning the file-selection failures to the model or engine.
- Refresh the full six-mode matrix on one binary, keeping short/long, sampled/
  greedy, cold/repeated and C1/C15 results separate.
- Profile the remaining target-verification execution cost independently of
  replay and host-only timings before changing state-bank or kernel architecture.
- Fix consumer issue #304's asynchronous shutdown lifetime separately; these
  included runs exited normally and retained their required shutdown counters.

## Evidence location

Raw evidence is external and untracked under:

```text
/Volumes/edata2/afm-benchmarks/qwen-next-mtp-parity-20260909
```

New sampled arms use `sampled-work-20260913-*`; the frozen preceding cohort and
shared-work manifests are not edited. The sampled baseline uses runtime
`bfcb6b5b`, consumer `9acccfc`, binary SHA-256
`0c63e09ab41c6a11b251f54a3a1fa9fd9659b48411e84aa15b4ebb566954c360`.
The new snapshot is indexed by `SAMPLED-REPLAY-20260913-SHA256SUMS.txt` and
`replay-limit-20260913-qualified-identity.json`, including both binary identities,
the preserved rebuilt binary, request/response data, comparisons, logs and
the documentation at evidence-freeze time.
The manifest verifies **576 files**, SHA-256
`c1e502eb8a6657d1e45579beb6c1d228d76e34e962519882c5463c0deb6e022b`.
This hash stamp follows the frozen documentation snapshot. The two preceding
cohort/shared-work manifests and all their entries were also reverified.

Profiling runs are separate from throughput A/B runs. Existing host-phase laps
include some GPU waits, not exact GPU kernel time; shared draft/repair creation
is not covered by every existing counter. Do not mistake a small host merge
duration for proof that device-side recurrent-state copying is free.

The diagnostic-only fixed-depth/top-p 1.0 run covered 398 shared verifier groups
and 2,016 rows (mean width 5.07). Recorded shared host-phase totals were:

| Phase | Seconds |
|---|---:|
| Target forward construction and its waits | 38.839 |
| Submit | 2.018 |
| Host draft-ID materialization | 1.069 |
| Adopt per-request state | 0.788 |
| Merge request caches | 0.137 |
| Head projection/sampling construction | 0.038 |
| Snapshot bookkeeping | 0.007 |
| Input construction | 0.006 |

Target forward is **90.5% of these recorded intervals**, not 90.5% of all server
or GPU time. Shared draft/repair operations are incompletely represented here;
sampling and copies can execute later when the lazy graph is consumed. The
separate two-second process stack sample landed in `prefillQwenMTP` waiting on
the initial target-token evaluation, not in steady-state decode. Preserve that
phase label. It corroborates the prefill path but does not isolate the slowest
Metal kernel. These observations prioritize the confirmed repeated-prefill cost;
they do not yet justify claiming a zero-copy state-bank redesign is the main win.
