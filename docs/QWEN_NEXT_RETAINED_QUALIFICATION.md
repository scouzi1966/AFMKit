# Retained Qwen Next qualification — September 17

PR #123. Runtime, defaults and installed binary are unchanged. The private
normalization candidate remains off. These are fixed-answer agentic tasks,
not new Context curves, comprehensive QA or general quality-parity proof.

## Frozen identities and method

- M3 Ultra, 512 GiB; one GPU owner; available-memory guards of 160 GiB before
  loading and 100 GiB during execution. No competing builds/inference observed.
- Checkpoint: `/Volumes/edata2/models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit`.
  Config/template/weight-index hashes match the preserved control.
- AFM SHA-256: `e7970fd1f444e13f23fc8f589b7dad29723216ea803697d7b02d71c4f85d7f91`.
  Consumer `e8b8e7a`, provider runtime `2c9c8241`; later commits add evidence.
- Reference SHA-256: `f3ce20fba143e908110d44d1b8304bf305962a4600d23c9c112934112514bbe4`.
  This is the preserved reference, **not a claim about its newest release**.
- AFM AR: documented A9 banked-attention preset, prefill 8192. AFM MTP: M25,
  depth 3, replay 4096 MiB/16 entries/8192 prompt limit. No private normalization.
- Reference: admission 15, MTP off/on at depth 3, prefix cache 16 entries/4GB;
  disk/tokenization caches, PLD/drafter and additional decode/KV quantization off.
- Identical 45 payloads per arm: 15 greedy and 30 sampled (temperature .6,
  top-p 1, fixed seeds), 512-token cap, thinking off. Each 15-prompt window is
  repeated immediately. Warmup excluded; first/repeat wall times kept separate.
- AFM AR retains its existing 64-entry radix policy. Cache geometry differs;
  this is a matched bounded-workload comparison, not identical cache internals.

## Lifecycle

`Scripts/qwen-next-retained-lifecycle.py` passed **665/665 assertions** across
79 requests including warmup: mixed MTP/AR logprobs, penalties, stop sequences,
sampled MTP, early disconnect/recovery, and prompts above 4096 but below 8192
tokens. Fifteen isolated identity controls precede 15-client cancellation,
follow-up and repeat waves. No foreign identifiers appeared in completed output.
Expected MTP repeats restored full prompts. There were 182 actual shared
verification batches and a clean exit. Peak RSS: 68.70 GiB; minimum available
memory: 359.70 GiB. RSS is not complete Metal allocation accounting.

**Real model A→B→A switching remains unqualified.** The consumer's
`/models/load` and `/models/unload` only acknowledge requests, and its MLX chat
controller serves the active model even when a different model is requested.
Those routes cannot prove a switch. The remaining test must directly exercise
AFMKit's load lifecycle, including rejection while requests are active.

## C15 matched presets

All **360/360** measured responses pass runtime, JSON structure and request
identity. Semantic correctness is separate. All four arms exit cleanly and
have been independently rescored by the audit script.

### Sampled tasks: 30 per cell

| Preset | First-use aggregate tok/s | Correct | Repeat aggregate tok/s | Correct | Repeat correct tasks/s |
|---|---:|---:|---:|---:|---:|
| AFM A9, MTP off | 82.29 | 23/30 | 154.36 | 23/30 | 1.830 |
| Reference, MTP off | 116.01 | 22/30 | 129.20 | 23/30 | 1.490 |
| AFM M25, MTP depth 3 | 53.05 | 21/30 | 179.69 | 21/30 | 1.872 |
| Reference, MTP depth 3 | 89.87 | 23/30 | 91.85 | 22/30 | 1.071 |

### Greedy tasks: 15 per cell

| Preset | First-use aggregate tok/s | Correct | Repeat aggregate tok/s | Correct |
|---|---:|---:|---:|---:|
| AFM A9, MTP off | 26.42 | 12/15 | 164.60 | 12/15 |
| Reference, MTP off | 51.46 | 13/15 | 129.10 | 12/15 |
| AFM M25, MTP depth 3 | 55.92 | 13/15 | 183.71 | 13/15 |
| Reference, MTP depth 3 | 62.92 | 12/15 | 95.81 | 12/15 |

Every window has 15 overlapping client requests. This does **not** prove
15-row GPU MTP verification in the reference; its MTP responses largely
finish sequentially. AFM shared-verifier execution is observed separately.

The best reference aggregate setting here is MTP **off**. Against it, AFM AR
repeats are 19.5% faster at the same sampled pass count. AFM MTP adds tokens/s
but loses two sampled answers versus AFM AR; it is not a quality-neutral
default recommendation. These small samples establish neither broad quality
parity nor statistical significance of a quality difference.

The first-use column is **not universally cold**: earlier tasks can supply a
shared prefix. On sampled first-use requests, AFM MTP restored zero tokens;
reference MTP restored 29,106. AFM AR restored 23,040 versus reference AR
29,070. Missing boundary coverage is a concrete part of the first-use gap,
alongside scheduling/prefill costs. Faster decode alone cannot remove it.

First/repeat text matches: AFM AR 45/45; AFM MTP 44/45; reference AR 43/45;
reference MTP 41/45. No AFM semantic pass/fail flips occurred. Reference AR had
a greedy pass→fail and sampled fail→pass; reference MTP had one sampled
pass→fail. Equal seeds do not imply equal rounding or RNG consumption.

Peak RSS: AFM AR/MTP 68.13/69.28 GiB; reference AR/MTP 67.62/68.68 GiB.

## One client through the existing scheduler

Only the server's explicit `--concurrent` value changes from 1 to 2; client
concurrency stays **one**, verified from request intervals. All M25 settings,
prefix flag, binary, checkpoint, payloads and seeds remain unchanged. This
uses the existing complete-state replay, not a new serial-cache implementation.

| Sampled repeat metric | Serial server (1) | Scheduler server (2) |
|---|---:|---:|
| Output / wall second | 47.95 tok/s | 120.70 tok/s |
| Correct answers | 22/30 | 22/30 |
| Correct tasks / second | 0.525 | 1.320 |
| Median first-token latency | 869 ms | 20 ms |
| Median decode-only rate | 128.66 tok/s | 126.42 tok/s |
| Restored prompt tokens | 0 | 30,162 |
| Peak RSS | 69.12 GiB | 68.98 GiB |

All **90/90 paired answers are byte-identical**; all 180 runtime, structure
and identity checks pass. The 2.52× repeat benefit comes from avoiding prefill,
not faster decode. First-use sampled rates are 48.05 versus 47.09 tok/s (about
2% slower through the scheduler in this run). Do not automatically select it.
The measured opt-in is the full M25 recipe with `--concurrent 2`, not that
single flag on a stock installed binary. Raw launch manifests are authoritative.
The existing 16-entry/byte limits remain; larger reuse distances can still miss.

## Remaining high-value experiment: pre-suffix complete-state boundaries

The reference's `src/generate.zig` deliberately holds back 30 prompt tokens
(`SSM_SNAPSHOT_BACKOFF`) so a reusable recurrent-state snapshot exists before
the changing assistant-template suffix. Its `src/prefix_cache.zig` restores
MTP head/QSA state together with the target boundary. Source credit belongs to
the mlx-serve project; this is an identified mechanism, not a new AFM invention.
The observed log records 967/973-token restores and inherited checkpoints.

AFM's `Qwen4ExpMTPSession` currently publishes a snapshot only after the full
prompt. `ExactPromptReplayCache.find(allowPrefix: true)` can continue an entire
saved prompt, but cannot invent a shorter recurrent/MTP state for two prompts
that diverge before that endpoint. Increasing the entry cap alone does not fix it.

The next controlled prototype should:

1. Add an explicit default-off pre-suffix snapshot policy in the self-contained
   Qwen model/session layer; retain target, recurrent/PLE, QSA and MTP head state
   at the **same** token frontier, before sampling. Never trim recurrent state.
2. Use the existing bounded cache and account every snapshot; do not silently
   double its memory or entry budgets. A prefix-only prototype can fit the
   current budget, but must report the lost exact-endpoint advantage.
3. Test replay isolation, changed suffixes, cancellation and generator identity
   on tiny models before the real checkpoint. Modified chunk boundaries can
   change rounding, so compare cold and warm correctness explicitly.
4. Measure first-use related prompts and exact repeats together, including
   correct tasks/sec, memory and the historical Context ledger. Keep it opt-in
   unless the user accepts any demonstrated quality/performance tradeoff.

This prototype is **not implemented in this qualification commit**. A separate
serial replay implementation is also not yet added; the single-client experiment
establishes an explicit existing path worth preserving first.

## Retention and evidence

Previous M25 sampled repeats: 184.18 tok/s; current: 179.69 (-2.4%). Outputs
also differ slightly (2024 versus 2016 tokens). This same-binary rerun does not
establish a source regression; neither result is erased. The old Context curve
and depth-4 peaks remain separate from these short-task aggregate rates.

External, untracked root:

```text
/Volumes/edata2/afm-benchmarks/qwen-next-mtp-parity-20260909/retained-qualification-20260917
```

Folders: `mtp-lifecycle-a`, `best-ar-a`, `reference-a`, `best-mtp-a`,
`single-client-scheduler-a`, `single-client-serial-a`. They retain launches,
payloads, raw chunks, scores, resource guards and exits. `*-audit.json`
independently checks hashes, scores, payloads, cache coverage and wall-time
denominators. The C15 runner is preserved as `broader-runner-c15.py`; pass it
to the auditor using `--runner` when reauditing those earlier arms.

No raw reports enter the repository. No release or installation changed.

The append-only evidence is sealed by `SHA256SUMS.txt` (778 files), SHA-256
`74a56f83d236073872f4e974e9b774ee4cf3dc6f4215f2024bf29f50704c974d`.
The manifest covers raw reports, exact runner copies, audits, logs and the
point-in-time summary; all entries verified. The qualification checkpoint is
`8b179910`. The 44 CPU harness tests pass; no new Swift XCTest run or runtime
rebuild is claimed for this scripts-and-documentation checkpoint.
