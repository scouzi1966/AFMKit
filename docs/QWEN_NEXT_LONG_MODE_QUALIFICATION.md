# Qwen Next long-context sampled mode qualification

Follow-up to [the replay-limit A/B](QWEN_NEXT_LONG_CONTEXT_REPLAY.md), on
[PR #123](https://github.com/scouzi1966/AFMKit/pull/123). This is a qualification
screen, not a release approval or permission to promote defaults.
The unresolved quality gate is tracked in
[issue #125](https://github.com/scouzi1966/AFMKit/issues/125); implementation
continues on the same PR rather than fragmenting this workstream.

## Fixed inputs and interpretation

- AFMKit runtime `739cdb6f`, consumer `9acccfc`; Release binary SHA-256
  `4193a708f44f5b59cab393e3e4452d2e469e5e67e7a62faf82ffa607cdc5c93a`.
- Exact checkpoint `/Volumes/edata2/models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit`,
  on the M3 Ultra 512-GiB host. No model substitution or weight conversion.
- The frozen long sampled fixture: fifteen coding-review requests, one excluded
  warmup, then first and identical-repeat phases; 4427–4433 prompt tokens.
- Temperature 0.6, top-p 1.0 unless explicitly labeled 0.95, seed 42,
  max_tokens 512, thinking off, streamed requests with usage. All messages,
  response chunks, outputs, usage and errors remain saved.
- C1 sends requests sequentially. C15 sends fifteen overlapping requests.
  Rates are aggregate completion tokens divided by whole-phase wall time,
  including prefill and queueing, not steady-state decode-only speed.
- Structural pass means parseable JSON with the correct request ID, the
  specifically requested filename and the three explanation fields present.
  It is not an AI-judge score or a proof of a correct code fix.
- Seed 42 is reused; identical repeated outputs are not independent quality
  samples. Different speculative sampling schedules need not produce identical
  wording. Multiple seeds and semantic evaluation remain necessary.

## Controlled recipe, not every optimization at once

The matched AFM arms retain the complete M25 environment and change only
`--mtp`, prefix enablement and concurrency. MTP uses fixed depth 3 and the
explicit batched verifier. The separate strict-verifier arm changes only that
policy on the C1 MTP path.

For AR, MTP-only controls are inactive. This is **not** the separately tuned A9
banked-attention recipe: that preset also enables mixed-position adapters, GDN
prework, banked attention, replay boundaries and a prefill override. Keeping
these separate avoids attributing several simultaneous changes to MTP alone.

The prefix flag does not prove a cache hit. Record actual cached-token usage.
In particular M25's complete MTP replay belongs to the concurrent scheduler;
it is not wired into the C1 generation lane. These C1 arms must not inherit
the C15 cache-hit speedup.

## Six-mode matched-control matrix

All rows below use temperature 0.6/top-p 1.0. The MTP+C15+prefix row is the
already-frozen expanded-limit A arm on the same binary and fixture; it is not
rerun and counted twice as new evidence.

| Mode | Prefix flag | Clients | First / repeat aggregate tok/s | Repeat cached tokens | Structural | Peak RSS GiB |
|---|---|---:|---:|---:|---:|---:|
| AR | On | 1 | 26.77 / 27.14 | 0 | 30/30 | 67.94 |
| AR | On | 15 | 74.15 / 103.02 | 66,432 | 30/30 | 67.75 |
| AR | Off | 15 | 29.50 / 29.75 | 0 | 30/30 | 67.81 |
| MTP | On | 1 | 28.71 / 29.19 | 0 | 16/30 | 69.78 |
| MTP | On | 15 | 34.96 / 96.51 | 66,447 | 16/30 | 70.49 |
| MTP | Off | 15 | 35.21 / 35.68 | 0 | 12/30 | 70.52 |

The six rows completed **180/180 runtime requests**, but only **134/180
structural checks**. AR passed 90/90; MTP passed 44/90. The stricter 22/30 C1
diagnostic and top-p 0.95 AR run are outside this six-row total.

At C15 with prefix reuse, MTP repeated token rate is 6.3% below matched AR,
while the number of structurally valid repeated tasks/s is 56.6% lower
(8/27.64 versus 15/22.51). Without prefix reuse, MTP repeated token rate is
20.0% higher but valid tasks/s is 62.7% lower (6/77.66 versus 15/72.48).
These are not qualified speed wins: answer length and correctness both matter.

The large first-phase AR cache total (62,051 tokens) means it already reused
common prefixes between distinct requests. MTP's complete replay does not
provide equivalent divergent-prefix reuse here (one warmup hit, 4,432 tokens).
Labeling both phases "cold" without these counters would hide that difference.

## Matched quality isolation

| Mode | Top-p | Structural checks | First / repeat aggregate tok/s | Peak RSS GiB |
|---|---:|---:|---:|---:|
| AR, C15, prefix on | 1.0 | 30/30 | 74.15 / 103.02 | 67.75 |
| AR, C15, prefix on | 0.95 | 28/30 | 72.91 / 97.40 | 67.99 |
| MTP batched, C1, prefix on | 1.0 | 16/30 | 28.71 / 29.19 | 69.78 |
| MTP strict, C1, prefix on | 1.0 | 22/30 | 25.24 / 25.57 | 69.88 |

All four arms completed 30/30 runtime requests, exited normally and passed the
1-Hz competing-process guard. The ordinary-decoding paths use request-local
temperature/top-p samplers; this is not an accidental greedy comparison.

MTP's wrong-file responses also occur without shared verification or concurrent
requests, and with the strict arithmetic policy. Therefore the new replay
eligibility limit and shared batching are not sufficient explanations. The
matched ordinary-decoding advantage is a **potential MTP quality regression**,
not proof that the model alone is responsible. The strict improvement also
does not establish that verifier rounding is the only cause.

Source inspection identifies two further differences worth isolating:

1. Ordinary decoding processes the final prompt token separately; MTP's initial
   target/head setup forwards the whole prompt and preserves aligned head history.
   Their prefill arithmetic and cache boundaries differ.
2. Ordinary decoding makes one request-local sample per generated token. MTP
   samples a verification block, then commits its accepted prefix. The same seed
   does not imply the same random draws per emitted position. The existing
   small-vocabulary sampling-law tests pass, but do not certify this checkpoint's
   long-context behavior over multiple seeds.

Neither observation is presented as the established root cause. Do not change
defaults or relabel failed file-selection checks as passing model behavior.

## Reference comparison controls

The existing reference binary is tested against the same checkpoint and saved
messages, with explicit **top-k 0**. AFM normalizes omitted top-k to 0, whereas
the reference would inherit **top-k 20** from this checkpoint's
`generation_config.json` without the override. Temperature and top-p remain
explicit request parameters.

The comparison disables prompt-lookup speculation, external drafters, attention
requantization, KV quantization, tokenization cache and the disk prefix tier.
Prefix capacity is 16 entries with a 4-GB byte budget; concurrent capacity 15.
The MTP depth cap is 3, not the reference's unconstrained/adaptive best-speed
configuration. Equal byte/entry limits do not imply identical snapshot contents.

The measured reference identity is in `*-reference-identity.json` and
`launch.json.measured_binary_sha256`; legacy sampled-fixture metadata also
contains the AFM hash, which is **not** the reference executable identity.

Measured reference: **mlx-serve 26.9.2**, MLX 0.32.2, mlx-c `56b2d39fc831`,
NAX off on this host. Binary SHA-256:
`f3ce20fba143e908110d44d1b8304bf305962a4600d23c9c112934112514bbe4`.
Its existing persistent round-cost tables were not cleared; the harness saves
them before and after each arm. Depth 3 is a cap: the reference still chooses
execution/width policies within it. This is not a claim about its maximum
performance with every optional speculation feature enabled.

| Engine / mode, prefix on, C15 | First / repeat aggregate tok/s | Repeat median TTFT | Repeat cached tokens | Structural | Peak RSS GiB |
|---|---:|---:|---:|---:|---:|
| AFM AR matched control | 74.15 / 103.02 | 0.459 s | 66,432 | 30/30 | 67.75 |
| Reference AR | 41.26 / 105.30 | 1.227 s | 66,012 | 14/30 | 68.39 |
| AFM MTP M25 | 34.96 / 96.51 | 0.279 s | 66,447 | 16/30 | 70.49 |
| Reference MTP, depth cap 3 | 30.68 / 35.67 | 28.721 s | 22,020 | 28/30 | 68.37 |

All **60/60 reference runtime requests completed** with the same actual prompt
token counts. The reference log confirms temperature 0.60, top-p 1.00, top-k 0,
thinking false and active MTP for its MTP arm. Both exits were normal and the
resource-isolation guards found no competing process.

AFM AR repeat token rate was **2.2% below** the reference in this pair, with
shorter outputs and better file-identity results. AFM MTP repeat token rate was
**2.71×** the reference, but file-identity checks were **16/30 versus 28/30**.
Even though repeated structurally valid tasks/s was higher for AFM MTP
(0.289 versus 0.213), its much lower per-request pass fraction remains a gate.
Do not call this qualified parity or an across-workload speed victory.

Reference AR also substituted unrelated module filenames. Thus the failure
pattern is not unique to AFM or MTP, and one fixed seed cannot establish a
distribution-level implementation defect. Conversely, that does not excuse
AFM's within-engine AR/MTP difference. The proper next step is controlled
multi-seed and teacher-forced target-logit/prefill comparisons, not changing
the expected filenames, choosing only a favorable seed, or silently selecting
a slower default verifier.

## Evidence and next gate

New mode evidence is external and untracked under:

```text
/Volumes/edata2/afm-benchmarks/qwen-next-mtp-parity-20260909/long-matrix-20260913-*
```

The read-only `report_long_mode_matrix.py` validates the fixture hash, exact
messages and request parameters, runtime completion, normal exit and isolation
for every included arm. It preserves each structural failure verbatim.
The new nine arms completed **270/270 runtime requests**: AFM 210/210 with
168/210 structural passes, reference 60/60 with 42/60 structural passes. The
reused M25 arm is excluded from these new-run totals. These are overlapping
subsets of the comparison tables, not numbers to add to the six-mode total.

`LONG-MODE-20260913-SHA256SUMS.txt` indexes the frozen raw evidence, helper
sources, reference bundle and documentation snapshot. The previous 576-file
replay manifest remains intact and is independently reverified at this freeze.
The new manifest verifies **419 files**, SHA-256
`ca8113b07f6416a1b929032deafe20ea6ecbee057bae1a9896f5599fabe56218`.
This stamp was added after the immutable documentation snapshot was made.

Before promoting sampled MTP: isolate prefill versus verifier/sampling effects,
repeat over independent seeds, inspect semantic quality and compare throughput
and memory at the same time. A faster cache-hit path alone does not satisfy
that gate. Consumer shutdown issue #304 remains a separate open task.
