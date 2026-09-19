# Qwen Next final-prefill head experiment

September 19 continuation of PR #123, based on checkpoint
`586e5af33c9602bbe435bd7ab4f389aa095cd0b9`.
This follows the [isolated head experiment](QWEN_NEXT_EXECUTION_COSTS.md).
**Default off. C15 uncached AR and C1 context tests show bounded prefill gains
with identical responses in the warmed comparisons. C15 AR prefix reuse and C1
MTP route guards also pass. Broader qualification remains open; the head-only
speed ratio is not a full-model claim.**

## Bounded change

`AFM_QWEN_PREFILL_LAST_LOGITS=1` is captured when `Qwen4ExpModel` is created.
Only the exact value `1` enables it; unset, `0` and other values disable it.
This is an experimental provider setting, not a new production CLI default.

The model implements the existing `LanguageModel.prepare` contract. It keeps
the ordinary strict `remaining > window` chunk loop and final trunk/mixer math,
but returns `PrepareResult.logits` after projecting only the final mixed hidden
row. It does not change the prompt, chunk size, KV/recurrent/PLE state, sampler,
or full-row APIs. Final host-token IDs are passed through when required.

| Entry point | Effect of opt-in |
|---|---|
| Ordinary unmasked text `prepare` | Final hidden row only is projected; returns `[1,1,vocab]` logits |
| Valid input with flag off | Existing remaining-token return contract |
| Explicit replay helper / interleave | Bypasses this override; existing boundaries retained |
| Native MTP sessions / verification | Existing full-row hidden/logit contracts retained |
| Qwen VLM wrapper | Existing preparation entry point retained |
| Masked or media input | Existing remaining-token fallback |
| Empty input | No final-row projection; returns empty remaining tokens |
| Nonpositive explicit window, either flag setting | Throws before cache mutation rather than entering an invalid/stalled loop |

Not every prefix-cache-enabled request necessarily uses the explicit replay
helper. Ordinary serial preparation can still reach this override; tests and
claims must name the actual route rather than inferring it from a cache flag.

The provider snapshot is directly tracked in AFMKit and documented in
`vendor/MLX/README.md`; no consumer checkout patch or metallib rebuild is needed.

## Numerical and performance qualification

The prior synthetic benchmark preserved the checkpoint's original 8-bit/group64
head. Moving the slice changes a multirow GEMM into a single-row operation:
all tested synthetic argmax values agreed, but logits differed by up to 0.015625.
Real sampled continuations or near-tie greedy decisions can therefore change.
No automatic promotion is justified by the component result alone.

Six passing CPU/FP32 tests cover environment selection, chunk boundaries, exact
cache bytes and metadata, populated-cache suffixes, forced continuation, tied
heads, full-row APIs, invalid/empty inputs, and masked/media fallback. They do
not establish BF16 checkpoint quality or speed.

The focused Release suite completed 123 tests: 122 passed, one optional
512-expert production-shape probe skipped, zero failures. It includes the six
new tests plus batching, cache selection, exact replay and MTP contracts.
`final-head-tests-d-exit.json` records a clean contention/memory guard; the
tested executable SHA-256 is
`dba7d43b7ae630b5db21d26cc312d3f52e3393d5e67189d1f7714a31b3144bff`.

Controlled measurements use the unchanged checkpoint:

```text
/Volumes/edata2/models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit
```

- C15 uncached AR: frozen 90-request fixed-answer workload, 512-token cap,
  greedy and sampled phases plus full-suite repeat; compare aggregate output
  tok/s, completed/correct tasks per second, response differences and latency.
- C1 context: first four contexts, one warmup and three measured trials each,
  128 generated tokens, temperature 0.6, top-p 1, seed 42; compare prefill proxy,
  TTFT, decode rate and saved text independently.
- Explicit replay and MTP route guards: preserve source contracts and test their
  behavior separately rather than attributing the ordinary-head gain to them.

Every run uses a fresh owned server, the same frozen launch preset and artifact
identity, no profiling flags, and one GPU workload at a time. The sole candidate
setting is the opt-in above. Existing tuning settings remain explicit: these
experiments are not unset-environment production performance claims.

## C15 uncached AR result

The same rebuilt executable (`95c115b95c642f125d32a45e9015eb026d72ac947e02346f17eaaee79806e600`)
was run off, on, then off again, each in a fresh server process. All 94 adjacent
resources match the frozen control bundle. The final off arm reproduces all
90 responses of the saved control; the on arm also reproduces all 90.

| Measured workload | Off (`quality-off-ar-b`) | On (`quality-on-ar-a`) | Change |
|---|---:|---:|---:|
| First greedy aggregate tok/s | 54.22 | 56.20 | +3.67% |
| First sampled aggregate tok/s | 47.47 | 49.33 | +3.93% |
| Repeat greedy aggregate tok/s | 53.84 | 56.59 | +5.10% |
| Repeat sampled aggregate tok/s | 47.41 | 49.73 | +4.90% |
| All four windows, aggregate tok/s | 49.688 | 51.876 | +4.40% |
| Correct tasks/s | 0.60925 | 0.63608 | +4.40% |
| Correct fixed-answer tasks | 72/90 | 72/90 | Unchanged |
| Structural passes | 88/90 | 88/90 | Unchanged |

Both arms generate 5872 tokens. The aggregate is total output divided by the
sum of the four non-overlapping window durations, not the average of per-request
or per-phase speeds. Greedy phases are 12/15 correct each; sampled phases are
24/30 each. Runtime and 512-token-cap checks pass for all 90 measured responses.
The tiny warmup is audited separately and is not a scored task.

Against the preserved reference's 56.408 aggregate tok/s, the candidate is
8.04% slower on this workload (previous matched control about 12% slower).
The reference scores 74/90, not 72/90: **this is not quality parity or overall
MTP/context/concurrency parity**. The result supports keeping an opt-in for
further qualification, not promoting a default. A single on run bracketed by
two off runs is not a confidence interval.

### Retained first-use anomaly, not optimization credit

`quality-off-ar-a` is retained in full. Its first greedy window took 75.101 s
instead of the saved control's 20.168 s; the first measured 1011-token prefill
took 48.414 s. Subsequent prefills returned near one second. All launch,
request, scorer and checkpoint metadata checks matched, but the cold cohort
geometry differed: first mixed-position decode used 5 rows rather than the
saved control's 15, then grew. Admission is arrival/timing dependent.

That first window changed two texts and improved one fixed answer (73/90 total
rather than 72/90); all 45 repeat texts matched the saved control and repeated
window durations differed by less than 0.2%. Existing evidence does not isolate
the first-use delay's cause or prove the changed membership caused those two
answers. JIT latency alone is not an explanation of greedy answer changes.

The naive off-a/on-a comparison shows a large, misleading first-window speedup
and one semantic regression. It is preserved as `quality-pair-a.json`, not used
as head-only optimization credit. `quality-pair-b.json` independently audits
the on run against the subsequent off control, including every retained
SDK-decoded stream, payload, score, cap, guard and phase duration; it finds all
90 texts identical. These audits do not verify HTTP wire bytes/SSE completion
sentinels, independently retokenize responses, or rehash the checkpoint shards.

## C1 first-four-context AR result

Each fresh server runs a warmup and three measured trials per context, with
128 output tokens, temperature 0.6, top-p 1 and seed 42. Prefix caching is off.
Unlike the C15 fixture's explicit 8192 step, this frozen C1 launch uses the
architecture's 4096 policy. These are client-side prefill proxies, not isolated
kernel rates. All 16 response texts (12 measured plus four warmups) are identical
between off/on and match the saved control; there is no new semantic score for
these context continuations.

| Context / actual prompt tokens | Prefill tok/s off → on | Prefill change | Decode tok/s off → on | TTFT seconds off → on |
|---|---:|---:|---:|---:|
| 0.5K / 493 | 887.54 → 937.45 | +5.62% | 67.71 → 66.42 | 0.5563 → 0.5275 |
| 1K / 864 | 1039.58 → 1104.51 | +6.25% | 67.62 → 67.25 | 0.8320 → 0.7839 |
| 2K / 2112 | 1192.00 → 1277.34 | +7.16% | 60.82 → 60.53 | 1.7727 → 1.6550 |
| 4K / 4150 | 1286.22 → 1293.16 | +0.54% | 60.41 → 60.49 | 3.2274 → 3.2108 |

The 4K prompt has only 54 tokens in its final remainder after the 4096-token
chunk, so it does not eliminate a 4150-row vocabulary projection. Decode changes
range from −1.91% to +0.14%; **no decode gain is claimed**, and these small
differences need repetition before attributing a regression or improvement.
The prior peak ledger is unchanged. Evidence: `context-pair-ar-a.json` with
unmodified prompt hashes, caps, raw retained chunks, response text and guards.

## Prefix-cache route guard

C15 AR with prefix caching enabled uses the saved immediate 15-request replay
window rather than the cache-off whole-suite repeat. Both settings complete
90/90 requests within the cap; 90/90 texts are identical, with 72/90 fixed-answer
and 88/90 structural passes. Repeated requests report cache hits in every repeat
window. Each arm reuses 15066 prompt tokens in the greedy repeat and 30132 in the
two sampled repeat windows combined.

This is a compatibility result, **not a cache-speedup claim**. The first-use
cached-token totals differ (7776 versus 8745 in greedy, 29110 versus 29114 in
sampled), making the apparent first-use timing change non-isolated. Repeated
greedy aggregate rates are 163.76/163.78 tok/s; sampled 153.23/155.67 tok/s.
The explicit replay helper bypasses the new override. Peak sampled process RSS
is 68.00/68.07 GiB, not a meaningful memory-saving result. Evidence:
`quality-prefix-pair-a.json`.

## MTP route guard

The C1 first-four-context pair also ran with `--mtp --mtp-depth 3`, otherwise
retaining its frozen launch and requests. All 16 off/on texts match, including
all 12 measured responses; all caps, exits and guards pass. Native MTP prepares
its own hidden/logit state and bypasses the new ordinary-prepare override.

| Context | Prefill tok/s off → on | Decode tok/s off → on |
|---|---:|---:|
| 0.5K | 955.01 → 958.43 | 89.28 → 89.61 |
| 1K | 1119.05 → 1119.17 | 84.61 → 84.76 |
| 2K | 1270.74 → 1273.23 | 79.02 → 78.10 |
| 4K | 1261.94 → 1267.96 | 82.80 → 84.61 |

This is an unchanged-route check, **not an MTP speedup or quality-parity claim**.
The measured prefill differences are below 0.5%; decode varies −1.16% to +2.19%.
Evidence: `context-pair-mtp-a.json`. C15 MTP, serial prefix reuse, cancellation
under full-model load, media inputs and additional architectures were not
requalified by this benchmark set; focused source tests are not substitutes
for those full-model cases.

## Next bounded experiment: deferred prefill completion

Source inspection confirms that uncached independent-cache AR admission still
finishes each `prefillOne` before the next request: lazy sampling is followed by
`didSample`, cache/token evaluation, host token readout and publication.
`didSample` itself can read the GPU token for penalties or grammar; deferring
only the later `.item()` would not remove that boundary.

A potential next experiment is a **two-request, token/memory-bounded pending
cohort**: preserve each existing model/chunk call and lazy sampler graph, then
complete materialization and publication after staging the small cohort. The
independent decode path already stages sampled graphs before cohort evaluation.
This is not implemented here and has no measured performance claim.

Required gates before accepting such a change:

- Retain request-local caches, output/continuation state, model and reservations
  until completion; keep construction serialized under the existing owner.
- Account for mapped PLE host reads and earlier-chunk cache evaluation, which
  can still serialize construction. Do not substitute the existing interleave
  helper: it changes the final-token split and therefore prompt geometry.
- Bound retained graphs/activations explicitly; the current soft admission
  token budget does not bound a cold cohort.
- Preserve processor order and first-token EOS/cap/logprob behavior; handle
  custom processors conservatively because they may read the CPU immediately.
- Recheck cancellation before publication, release reservation/matcher/metrics
  ownership exactly once, and drain submitted work before model teardown.
- Independently measure quality: delayed publication can change later batch
  membership even if each prefill uses identical arithmetic.

This is a scheduling/lifecycle investigation, not another cache-policy change.

## Evidence

External untracked root:

```text
/Volumes/edata2/afm-benchmarks/qwen-next-mtp-parity-20260909/final-prefill-head-20260919
```

Builds use the consumer reliable wrapper in Release mode. Source deltas include
new tests before compilation; separate attempt labels preserve failures and
do not overwrite older results. `run-final-head.py` reuses the frozen context
and quality harnesses; optional prefix qualification retains their immediate
15-case replay window and checks cache-hit coverage. A post-run cap audit is
separate from semantic scoring. No generated response is executed.
