# Qwen Next: earlier complete-state cache boundaries

September 17–18, 2026. Continuation of AFMKit PR #123. These are opt-in
experiments, not a default change, release, or installed-binary update.

## Problem and mechanism

The retained M25 replay cache stores complete target, recurrent/PLE, QSA and
MTP-head state at the **end** of a prompt. It can reuse an exact prompt or
continue it, but cannot rewind recurrence when a new prompt diverges before
that endpoint. In the retained C15 sampled first-use screen, AFM MTP reused
zero tokens while the reference reused 29,106. These are new related prompts,
not universally cache-cold inputs.

Source credit: mlx-serve's `src/generate.zig` (`SSM_SNAPSHOT_BACKOFF`) and
`src/prefix_cache.zig` complete-state restoration. The reference excludes the
last prompt token from chunked prefill, then holds back 30 more: AFM's equivalent
full-prompt backoff is **31**, not 30. The model checkpoint, template and
arithmetic kernels are unchanged by this experiment.

```text
shared prompt body          changing suffix / assistant prefix
-------------------------|------------------------------------|
                    earlier complete state             full endpoint
                    useful for related prompts          useful for repeats
```

`Qwen4ExpMTPSession` can now capture the earlier state before suffix processing
or sampling. All layers and the MTP head are captured at their correctly paired
positions; `lastStream` bridges the next head/token pair. Restoring copies all
mutable state into a request-owned session. UUID, token prefix and prefill
geometry checks remain mandatory. Never substitute AR-only radix state for
complete MTP state or trim recurrence from a later position.

## Explicit controls

Keep the entire existing M25/A9 recipe and change only the named delta.
Full launches and executable hashes in the external evidence are authoritative;
these flags alone on a stock installed binary do not reproduce the experiment.

| Delta | Unset | Scope and cost |
|---|---|---|
| `AFM_QWEN_MTP_REPLAY_BACKOFF=31` | 0: endpoint | MTP owner + prefix + replay budget. One earlier snapshot replaces each endpoint snapshot. Exact repeats recompute the tail. |
| `AFM_QWEN_MTP_REPLAY_BACKOFF_ON_MISS=1` | off | With positive backoff: misses seed an earlier boundary; hits retain the full endpoint. Unshared original-prompt boundaries are replaced during promotion; boundaries that actually served another prompt remain separately budgeted. |
| `AFM_QWEN_PREFIX_REPLAY_BACKOFF=31` | 0: existing grid | Qwen Next scheduler AR lane. Replace coarse interior checkpoints with one earlier boundary, keeping the final prompt-minus-one snapshot. Refinement under qualification. |

Backoff controls accept integers clamped to 0–256; invalid/unset means zero.
Short prompts retain their endpoint. No backoff affects MTP with its replay
cache disabled. AR helper policy is reusable, but only text Qwen Next's
scheduler opts in; other architectures and serial AR keep their existing policy.
Every forward still respects the caller's prefill chunk bound.

MTP retains the existing 4096 MiB, 16-entry, 8192-token experimental limits.
An endpoint and an earlier snapshot, when both admitted, each count as an entry
and consume the same byte budget. No extra unaccounted cache is introduced.
The optional on-miss policy can still evict useful entries; it must be measured.

## First controlled screen

Same M3 Ultra 512 GiB, same ddalcu checkpoint:
`/Volumes/edata2/models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit`.
Release binary A SHA-256:
`f0a33de2b979836eea3319eb6c1fa3e91fd5d3b7e4eef1b35ea4f61d44285f1a`.
The retained reference is the same preserved executable as the
[previous qualification](QWEN_NEXT_RETAINED_QUALIFICATION.md), not its newest release.

All 45 fixed-answer agentic payloads, seeds, thinking-off policy, temperature
0/.6, top-p 1, 512-token cap and C15/window-15 replay order are unchanged.
Rates below are aggregate output tokens / workload wall seconds, **not**
isolated decode or Context prefill rates.

| Sampled, 30 tasks | First-use tok/s | Correct | Exact-repeat tok/s | Correct |
|---|---:|---:|---:|---:|
| Same binary, backoff unset (`control-a`) | 53.61 | 21/30 | 182.60 | 21/30 |
| Backoff 31 (`backoff31-a`) | 117.70 | 21/30 | 143.34 | 22/30 |
| Backoff 31 rerun (`backoff31-b`) | 136.70 | 21/30 | 147.42 | 22/30 |
| Preserved reference, MTP on | 89.87 | 23/30 | 91.85 | 22/30 |
| Preserved reference, MTP off | 116.01 | 22/30 | 129.20 | 23/30 |

All 270 measured responses across the three new arms pass runtime, JSON
structure and identity checks; all arms shut down cleanly and were rescored by
the independent audit script. First-use token reuse changed from zero to
29,126 / 29,114. Peak RSS: control 69.24 GiB; backoff 69.11 GiB. RSS does not
include complete Metal allocation accounting.

The first `backoff31-a` long request took 67.592 s to prefill; its greedy
first-use window was consequently **11.74 tok/s**. This result is preserved,
not silently discarded as warmup. The same-binary, same-policy restart
`backoff31-b` took 0.976 s for its first long prefill and 74.12 tok/s for the
greedy first-use window (control 56.04; reference MTP 62.92). The cause of
that initial delay is not isolated; do not label it definitively JIT or I/O.

This establishes a material first-use improvement, but the prefix-only
snapshot costs roughly 19–22% of exact-repeat throughput versus the current
endpoint control. It is **not a default recommendation**. The sample is too
small to establish general quality parity with the reference. Historical
Context/decode peaks remain separate and are not replaced by these numbers.

## Validation and evidence

The first implementation passed 79 targeted XCTest cases: 78 passed, one
explicit production-shape opt-in skipped, zero failures. New tests cover
short prompts, disabled retention, changed suffixes, both verification policies,
greedy/sampled RNG isolation, cancellation, geometry and generator rejection,
and immutable sparse/recurrent/PLE/head state. The first attempt compiled but
failed to locate XCTest's metallib; the wrapper retry staged the actual bundle
and passed. No inference-code fix was needed for that resource-layout failure.

### Endpoint promotion, binary B

Release SHA-256:
`1c9b9f52b3c965fdcbeba1a978fa107a4099a3b508e13152c8ca82633fcfb06f`.
All comparisons below retain the full M25 or A9 recipe, changing only the
documented cache delta. The AR control was rerun on this same executable.

| Mode / sampled 30 tasks | New related prompts tok/s | Correct | Exact repeats tok/s | Correct | Structure, first/repeat |
|---|---:|---:|---:|---:|---:|
| MTP, earlier-on-miss (`onmiss-a`) | 135.57 | 22/30 | 178.36 | 23/30 | 30/30, 30/30 |
| AR, unchanged A9 (`ar-control-a`) | 87.60 | 23/30 | 159.24 | 23/30 | 30/30, 30/30 |
| AR, earlier boundary (`ar-backoff31-a`) | 119.92 | 24/30 | 158.67 | 24/30 | 29/30, 29/30 |

MTP earlier-on-miss recovers most repeat throughput, but its greedy initial
repeat remains 150.55 versus the endpoint control's 184.85 tok/s. AR improves
sampled first-use throughput by 36.9%, with repeat throughput −0.36%. Its
`review-1-07` response is valid JSON but supplies an array directly as `answer`
instead of the required `{"order": [...]}` object; it also orders two tasks
incorrectly. The same-seed control succeeds on that case. This is an observed
per-case regression even though aggregate correct answers increase. Do not
call it malformed transport, hide it in a net score, or claim broad quality
parity. Changed snapshot/chunk boundaries can change floating-point execution
and sampled wording; the exact cause of this case is not isolated.

Binary B's targeted XCTest run: **83 cases, 82 passed, one explicit production
shape skipped, zero failures**. The harness suite passed **47 tests**. The
consumer wrapper now identifies the actual `Package` name after helper target
declarations, fixing repeatable pre-staging into the wrong `CmlxPackageTests`
bundle. Three CPU tests cover that expression. This is a build/test tooling
fix, not an inference performance change.

### Capacity regression found by lifecycle qualification

`lifecycle-onmiss-a` passed long mixed MTP/AR requests, cancellation recovery,
and identity isolation, but failed **nine expected cache-hit assertions** in
the C15 post-cancellation wave. All of those responses had correct ownership,
valid caps and normal completion; the server exited cleanly. They reported
zero cached tokens, not corrupt or invented cache hits. Qualification stopped
at the failure and the initial run is preserved.

The initial promotion policy stored both a unique 34-token boundary and its
65-token exact endpoint, displacing other owners from the fixed 16-entry LRU.
Adding capacity would mask the admission problem. The refinement records
bounded source-prompt ownership metadata in `ExactPromptReplayCache`:

```text
earlier snapshot → exact repeat of original prompt → replace unshared entry
                 → a different related prompt     → mark shared; keep boundary
                                                     + budget endpoint separately
```

Metadata never feeds inference, is charged to the existing byte budget, and
is absent for default callers. A failed/oversized insertion leaves the prior
snapshot intact. Shared entries can still be evicted under ordinary LRU
pressure; there is no guarantee for arbitrary larger working sets. The new
CPU tests exercise a 15-prompt working set, shared-boundary preservation, and
invalid/oversized promotion without eviction. The failed run is not
retroactively marked passed.

### Ownership fix, binary C

Release SHA-256:
`21ee9bf82b49a361a240abd2fe09a3a416cf5cf9e794e95183a67805983387be`.
Targeted tests: **86 cases, 85 passed, one explicit production-shape skip,
zero failures**. The release consumer build passed. Companion wrapper fix is
committed on `perf/qwen-next-mtp-consumer` at `510072d`; no provider source
was patched in a consumer dependency checkout.

`lifecycle-owned-a` now passes **565/565 assertions across 79 requests**,
with 184 actual shared-verification batches, clean shutdown, and passing
memory/competing-work guards. C15 post-cancellation and repeat waves have
all expected replay hits and no foreign identities. This is not a real
model A→B→A test or a full release qualification.

The identical M27 agentic launch (`owned-a`) measures:

| Tasks | New related prompts tok/s | Correct | Exact repeats tok/s | Correct |
|---|---:|---:|---:|---:|
| Greedy, 15 | 71.88 | 13/15 | 160.42 | 13/15 |
| Sampled, 30 | 137.94 | 23/30 | 182.75 | 22/30 |

All 90 runtime/structure/identity checks pass. Peak RSS is 69.079 GiB. The
sampled repeat rate is within 0.8% of the retained 184.18 tok/s high watermark;
the earlier greedy repeat cost has narrowed, but is not eliminated. Correct
sampled counts equal the preserved reference MTP counts in both phases, with
higher aggregate throughput; this small task set does not establish general
quality equivalence.

Same-binary endpoint control (`owned-control-a`), with both new MTP controls
unset, measured sampled first/repeat **54.37 / 186.11 tok/s** at 22/21 correct
of 30. Thus the final policy improves sampled first-use throughput **153.7%**
while repeat throughput is **1.8% lower**. Greedy first/repeat control is
57.30/187.27 versus candidate 71.88/160.42: the **14.3% greedy-repeat cost**
remains a reason not to promote defaults. Candidate and control each reproduce
44/45 first-use texts on exact repeat; sampled execution is not byte-identical
across every scheduling/cache path. Do not claim an exact-token-equivalent win.

Across eight new agentic arms, independent re-audit confirms 720/720 runtime
and identity checks, 718/720 structure checks. Both structure failures are the
same AR answer-shape failure on first/repeat. Correct-answer scores are
separate from those totals. All memory guards pass (minimum available memory
356.83 GiB); observed process peaks are 68.08–69.24 GiB, not total GPU memory.
MTP source-ownership bookkeeping runs at cache lookup/admission, not per token;
it adds no synchronization lock or capacity increase.

### Uncached Context regression check

Binary C; frozen M25 launch with prefix **off**, C1, MTP depth 3 when enabled,
temperature .6, top-p 1, seed 42, 128 generated tokens. No boundary controls
are set. All **24 measured responses** match their saved same-mode text
byte-for-byte; eight excluded warmups are retained. Both arms exit cleanly.
These are default-path regression controls, not measurements of boundary
caching with prefix enabled. Each table entry is a three-trial median,
**client prefill proxy / decode** in tokens/s, not C15 aggregate throughput.

| Context | MTP off | MTP depth 3 on |
|---|---:|---:|
| 0.5K | 893.48 / 69.63 | 962.96 / 90.88 |
| 1K | 1042.31 / 69.25 | 1121.94 / 86.60 |
| 2K | 1200.45 / 62.99 | 1293.54 / 83.71 |
| 4K | 1305.33 / 61.94 | 1275.51 / 85.29 |

Versus the September 15 same-launch baseline, median decode changes range
from +0.18% to +2.85%; prefill proxy from −0.31% to +1.34%. This is no
observed material regression, not attribution of small gains to the cache
change, which is inactive here. The refreshed `peak-ledger` retains the
105.79 tok/s depth-4 short-context high (current depth 3 is 14.09% lower),
87.54 at 2K and 88.63 at 4K from other historical configurations. Different
depths/prefill policies are not substituted for same-launch controls.

Initial AR warmup TTFTs were 4.81/4.41/11.10/15.47 seconds. They remain in the
evidence and are not included in warmed medians. The earlier 67.592-second
agentic first-prefill outlier also remains open for a separate cold-path
investigation; these cache changes do not claim to resolve all cold starts.

## Decision

Retain M26/M27 and A11 as explicit research opt-ins on PR #123. The identified
first-use C15 throughput deficit is closed in this frozen same-checkpoint
screen, but general model-quality equivalence, actual model switching, and
latest-reference-version parity are **not** certified. M27's greedy-repeat
cost and A11's wrong answer shape prevent a silent default promotion. The
cache policy is reusable, but no other model's default or qualification is
changed. No compiler precision switch, new quantization, cache-budget increase,
release build deployment, or installed nightly replacement was made.

Raw evidence, frozen binaries/resources, launch manifests, patches and audits:

```text
/Volumes/edata2/afm-benchmarks/qwen-next-mtp-parity-20260909/prefix-boundary-20260917
```

No reports or binaries enter Git. No default, main branch or installation changes.
