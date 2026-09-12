# Opt in to shared Qwen Next verification graph scheduling

AFMKit PR [#123](https://github.com/scouzi1966/AFMKit/pull/123), September 12, 2026.
**Experimental, off by default. No production-default or release recommendation.**

This is the concurrent **MTP** experiment, not the separate
[ordinary decode attention preset](QWEN_NEXT_BANKED_ATTENTION_OPT_IN.md).
It submits bounded pieces of a shared verification graph while Swift constructs
later layers. Every submission first completes private PLE/n-gram leaf writes.
It does not add a global GPU synchronization, change quantization or speculative
depth, enlarge the request cohort, or replace request-owned cache rollback.

## Build and identify the binary

Use the existing paired-development worktrees on the qualification machine.
These paths are not portable installation instructions. Stop the test server
and make sure no other build or GPU inference is running before rebuilding:

```bash
cd /Volumes/edata2/dev/CODEX/maclocal-api-qwen-next-mtp-parity
MACLOCAL_AFMKIT_PATH=/Volumes/edata2/dev/CODEX/AFMKit-qwen-next-mtp-parity \
  Scripts/swiftpm-reliable.sh build -c release --product afm \
  --disable-build-manifest-caching
shasum -a 256 /Volumes/edata2/dev/CODEX/maclocal-api-qwen-next-mtp-parity/.build/release/afm
```

The measured Release binary reports `v0.9.20` and has SHA-256
`87afdd39fdc5ea22eb4c3d91a20e68d9f8acb02aecf74b7890f5f2d93e713faf`.
A rebuild can have a different hash; record it rather than treating `--version`
as proof of identical code. Nothing here installs AFM or changes an exact
AFMKit dependency pin.

## Launch a controlled experiment

Check that port 9998 is free. Do not stop an unrelated server. Use a shell
without inherited tuning overrides; the command below fixes the recorded
preset, not every possible environment variable. The saved harness additionally
scrubs inherited `AFM_`, `MACAFM_`, `MACLOCAL_`, `MLX_`, `MTPLX_`, `OMLX_` and
`DYLD_` settings. Keep diagnostic profiling off for timed comparisons.

```bash
/usr/bin/env \
  AFM_QWEN_VERIFY_SHARED_ASYNC_LADDER=4 \
  AFM_QWEN_VERIFY_SHARED_COMPILED_TAIL=0 \
  AFM_QWEN_MTP_SCHEDULER=1 \
  AFM_QWEN_MTP_REPLAY_MIB=4096 \
  AFM_QWEN_MTP_SUBMISSION_WINDOW=4 \
  AFM_QWEN_MTP_SHARED_VERIFY=1 \
  AFM_QWEN_BATCH_COMPATIBLE_GROUPS=1 \
  AFM_QWEN_BATCH_CONTINUOUS_GROUPS=1 \
  AFM_QWEN_BATCH_YIELD_INTERVAL=8 \
  AFM_QWEN_VERIFY_QMM=1 \
  AFM_QWEN_PLE_NATIVE_READS=0 \
  AFM_QWEN_MTP_VERIFICATION_POLICY=batched \
  AFM_QWEN_VERIFY_ATTENTION_CHUNK=2 \
  AFM_QWEN_VERIFY_FUSED_HC=1 \
  AFM_QWEN_HC_NATIVE_CHAIN=1 \
  AFM_QWEN_VERIFY_FUSED_ROUTER=1 \
  AFM_QWEN_VERIFY_ASYNC_LADDER=8 \
  AFM_QWEN_MTP_DRAFT_ASYNC_LADDER=1 \
  /Volumes/edata2/dev/CODEX/maclocal-api-qwen-next-mtp-parity/.build/release/afm mlx \
  -m /Volumes/edata2/models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit \
  --port 9998 \
  --no-think \
  --mtp \
  --mtp-depth 3 \
  --concurrent 15 \
  --enable-prefix-caching
```

This is the **full measured experimental preset**, not a minimal set of flags.
The 4-GiB replay setting is an explicit request-state cache budget, not a model
memory limit. Runs use an M3 Ultra with 512 GiB memory and approximately 70 GiB
process RSS; this is not a minimum-memory guarantee or M4 Pro qualification.
The local performance harness requires at least 160 GiB available before load.
Do not substitute another checkpoint, add KV quantization or use `--vlm`.

Send 15 overlapping requests; `--concurrent 15` alone does not create work.
The recorded synthetic coding-review prompts use temperature 0, top-p 1,
seed 42, streaming with usage, and `enable_thinking: false`. Compare equal
output budgets: the 192- and 512-token screens are separate workloads.
Record response completeness/quality and latency as well as aggregate tok/s.
A single chat does not demonstrate a concurrency throughput gain.

## Controls, coverage and rollback

| Setting | Unset behavior | Meaning |
|---|---|---|
| `AFM_QWEN_VERIFY_SHARED_ASYNC_LADDER` | `0` / off | Positive integer: submit every N completed layers in eligible shared verification; measured values 4, 8, 16 |
| `AFM_QWEN_VERIFY_SHARED_COMPILED_TAIL` | Off | `1` compiles the shared post-attention pure block; retained as a separate experiment, **not enabled in the recipe above** |
| `AFM_QWEN_VERIFY_ASYNC_LADDER` | `0` / off | Existing singleton verifier setting; independent of the new shared control |
| `AFM_PERF` | Off | `1` adds host-lap reports, including `[shared-host-phases]`; do not use these runs as timing controls |

The new controls are process-start settings, not API kwargs. Restart after
changing them. They are limited to explicitly batched-policy verification,
2–4 compatible requests, supported verification widths and at most 16 total
request/token rows. The compiled-tail experiment additionally requires BF16,
model-owned compiled execution support and the ordinary non-deferred tail path.
The recorded preset leaves `AFM_QWEN_VERIFY_DEFER_HC` unset.
Strict verification, ordinary AR, unsupported shapes and other model families
keep their existing execution paths.

Look for actual nonzero `Qwen MTP shared verification: batches=... | rows=...`
shutdown counters, not just a scheduler-enabled launch flag. In a separate
diagnostic run, shared-host phase records also count actual groups and rows.
If counters are absent, coverage is unknown; do not report it as zero or assume
the optimization ran. Request samplers, cancellation and cache frontiers remain
independent; output wording need not be identical across concurrent batched
verification schedules. The observed missing `fix` field in one review task is
documented in the findings and must not be dismissed as truncation.

For a same-binary control, stop the owned server with Ctrl-C and relaunch the
complete preset with `AFM_QWEN_VERIFY_SHARED_ASYNC_LADDER=0`; leave the compiled
tail at `0`. To leave the whole experiment, launch without the preset from a
shell with no inherited tuning variables. Do not add these variables to shell
startup files or enable them globally.

See [measurements, exclusions and limitations](SHARED_BATCH_EXECUTION_WORKSTREAM.md#shared-verifier-graph-scheduling-2026-09-12).
These measurements do not establish reference-engine parity or justify
promoting either experiment to a default.
