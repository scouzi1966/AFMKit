# Opt in to Qwen Next request-banked attention

Experimental AFMKit PR [#123](https://github.com/scouzi1966/AFMKit/pull/123),
runtime checkpoint `2f62451e`; instructions checked on 2026-09-12.
**Off by default. For controlled experiments, not a production recommendation.**

This shares an attention GPU dispatch across independent requests without
padding/copying their full KV histories. It does not enable MTP, change weight
quantization, or enable the optional PLE row cache. The measured C15 raw-token
gains were about 2.4–6.6%, but structural task completion regressed. In the
512-token prefix-on diagnostic, control passed 30/30 and candidate 28/30:
one response omitted a required field even when it stopped normally.
See [implementation and qualification](SHARED_BATCH_EXECUTION_WORKSTREAM.md#request-banked-attention-prototype-2026-09-12).

## 1. Use the branch-built Release binary

The following are the existing qualification-machine paths, not portable
installation paths. A normal Homebrew/PyPI `afm` is not evidence that this
unmerged experiment is present. `--version` alone is insufficient: the measured
development binary reports `v0.9.20`.

```bash
shasum -a 256 /Volumes/edata2/dev/CODEX/maclocal-api-qwen-next-mtp-parity/.build/release/afm
```

Measured binary SHA-256:
`70853786986f431012bc94749e0207b8e3ee9df034d11e14250e1e8eff9a245f`.
Use this binary to reproduce the recorded results; a rebuild may have a new
hash and must be recorded as a new artifact.

If rebuilding is necessary, first stop your test server and verify the local
AFMKit worktree contains the PR code. Use the existing paired worktrees:

```bash
cd /Volumes/edata2/dev/CODEX/maclocal-api-qwen-next-mtp-parity
MACLOCAL_AFMKIT_PATH=/Volumes/edata2/dev/CODEX/AFMKit-qwen-next-mtp-parity \
  Scripts/swiftpm-reliable.sh build -c release --product afm \
  --disable-build-manifest-caching
```

This is a local paired build, not a dependency-pin change or installation.
Do not patch resolved SwiftPM checkouts, build while inference is running, or
use raw SwiftPM build/test commands. Explicit replanning avoids the stale
source-list problem encountered when the new Swift file was added.

## 2. Launch the measured opt-in preset

Use the **same** checkpoint, already present at the path below. Measurements
were on an M3 Ultra with 512 GiB memory, with about 68–70 GiB process RSS.
That is not a minimum-memory guarantee; this recipe is not qualified for a
small-memory M4 Pro. The saved benchmark harness requires at least 160 GiB
available before loading. Run only one GPU benchmark/server at a time.

Check that port 9998 is free before starting; AFM can fall back to another port
if it is occupied. Do not stop an unrelated server. Check the startup URL.
Start without inherited tuning overrides: the command below overrides only
the listed variables, whereas the saved benchmark harness scrubs inherited
`AFM_`, `MACAFM_`, `MACLOCAL_`, `MLX_`, `MTPLX_`, `OMLX_`, and `DYLD_` settings.

```bash
/usr/bin/env \
  AFM_QWEN_BATCH_BANKED_ATTENTION=1 \
  AFM_QWEN_BATCH_ATTENTION_PROJECTIONS=1 \
  AFM_QWEN_BATCH_GDN_PREWORK=1 \
  AFM_QWEN_COMPILE_BATCH_GDN_DECODE=1 \
  AFM_QWEN_BATCH_MIXED_POSITIONS=1 \
  AFM_QWEN_BATCH_CONTINUOUS_GROUPS=1 \
  AFM_QWEN_BATCH_COMPATIBLE_GROUPS=1 \
  AFM_QWEN_BATCH_YIELD_INTERVAL=8 \
  AFM_QWEN_BATCH_PREFILL_TOKEN_BUDGET=1024 \
  AFM_QWEN_PLE_ROW_CACHE_MIB=0 \
  AFM_PREFIX_REPLAY_BOUNDARIES=1 \
  AFM_QWEN_BATCH_PREFILL_INTERLEAVE=0 \
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
  --concurrent 15 \
  --enable-prefix-caching \
  --prefill-step-size 8192
```

This is the complete recorded preset, **not a minimal set of required flags**.
The verifier/draft settings are retained from that preset; there is deliberately
**no `--mtp`**. They do not turn speculation on by themselves. For manual WebUI
use, append `-w`; the throughput measurements did not use it. To test without
prefix caching, omit `--enable-prefix-caching` and leave the other settings
unchanged. The replay-boundary setting then has no prefix reuse to accelerate.

All environment values above apply only to that child process; do not add
them to shell startup files, a global service, or production configuration.
They are process-start settings, not request kwargs or live WebUI switches.
Changing them requires stopping and restarting the server.

### Which settings actually enable the path?

| Setting | Unset behavior | Role |
|---|---|---|
| `AFM_QWEN_BATCH_COMPATIBLE_GROUPS=1` | Off | Enable Qwen-compatible decode grouping |
| `AFM_QWEN_BATCH_CONTINUOUS_GROUPS=1` | Off | Enable continuous request ownership/admission on that path |
| `AFM_QWEN_BATCH_MIXED_POSITIONS=1` | Off | Use the request-owned mixed-position AR adapter |
| `AFM_QWEN_BATCH_ATTENTION_PROJECTIONS=1` | Off | Share attention projections and eligible Q/K normalization |
| `AFM_QWEN_BATCH_BANKED_ATTENTION=1` | Off | Select the new banked attention prototype within that adapter |
| `AFM_QWEN_FUSED_QK_NORM_ROPE` | Enabled unless `0` | Must remain enabled for the banked adapter; not an extra opt-in |
| `AFM_QWEN_QSA_FUSED_DECODE_ATTENTION` | Off | Leave unset/`0`; enabling this separate experiment bypasses banking |

The other preset controls fix the comparison's GDN, scheduling, replay and
lookup configuration. Setting only the new banked-attention flag is not enough.
`--concurrent 15` sets a ceiling; it does not manufacture 15 active requests.

## 3. Exercise and verify it

Send overlapping text-generation requests to `http://127.0.0.1:9998/v1`.
For the recorded performance settings, each request explicitly sets:

```json
{
  "temperature": 0,
  "top_p": 1,
  "seed": 42,
  "max_tokens": 192,
  "stream": true,
  "stream_options": {"include_usage": true},
  "chat_template_kwargs": {"enable_thinking": false}
}
```

These are request options, not a complete request: also supply the exact model
path and messages. A single curl or WebUI conversation can check availability,
but cannot demonstrate the aggregate-throughput benefit. The measured workload
used 15 concurrent agentic review tasks and their repeats, not 15 sequential calls.

- Look for `[BatchScheduler] Request-owned mixed-position decode: rows=N`
  with `N > 1`. This confirms the scheduler adapter ran, **not proof by itself
  that every attention call used the new kernel**.
- Eligible attention uses BF16 Q/K/V, 256-wide heads, one decode token per
  request, native Qwen caches and compatible GQA geometry. Text `Qwen4ExpModel`
  is the supported adapter; do not add `--vlm` or `--kv-bits` to this recipe.
- Dispatches contain at most four requests. Larger admitted groups are split
  into banks; singletons and unsupported/masked banks use the existing path.
  Fused-score selection also keeps its existing path. This fallback is intentional.
- Definitive kernel-level confirmation, if tracing separately, is a dispatch
  whose name contains `qwen_request_banked_attention_256_b2`, `b3`, or `b4`.
  Do not compare instrumented throughput to the uninstrumented results.

For exact recorded prompts and environment cleanup, the existing external
harness can own the launch, run and teardown. **Stop the manually launched
server first.** Run these commands sequentially on the qualification machine:

```bash
cd /Volumes/edata2/afm-benchmarks/qwen-next-mtp-parity-20260909
/Volumes/edata/dev/git/CODEX/llm_context_benchmarks/.venv/bin/python \
  run_banked_attention_screen.py --banked 0 --prefix 1 --label manual-banked-control-01
/Volumes/edata/dev/git/CODEX/llm_context_benchmarks/.venv/bin/python \
  run_banked_attention_screen.py --banked 1 --prefix 1 --label manual-banked-candidate-01
```

These scripts and that Python environment are local qualification assets,
not files supplied by an AFM/AFMKit installation. Use new labels for every run:
the harness refuses to overwrite existing result arms. Results are retained
under `<label>-afm-mtp-0/`, including launch settings, prompts/responses, usage,
memory samples, summaries and exit status. It does not run a semantic AI judge.
The `*-banked-controls.json` records the actual AFM binary hash; the frozen
launcher's legacy `launch.json.binary_sha256` hashes `/usr/bin/env` instead.

Compare repeat phases with matching checkpoint, binary, prompt and output
budget. Record aggregate tok/s, completed requests/s, structurally valid
requests/s, response quality, latency and memory. A higher token rate alone
does not satisfy the qualification gate. Do not replace the 192-token baseline
with the separate 512-token diagnostic or dismiss a normally finished omission
as truncation.

## 4. Turn it off

Stop your server with Ctrl-C. To disable **only banked attention** for an A/B
control, rerun the full preset with `AFM_QWEN_BATCH_BANKED_ATTENTION=0`.
Keep all other values and request parameters the same. The companion batching
experiments remain active; this is not a production-default baseline.

To leave the entire preset, stop the server and launch the same binary without
the environment prefix, from a shell without inherited tuning overrides:

```bash
/Volumes/edata2/dev/CODEX/maclocal-api-qwen-next-mtp-parity/.build/release/afm mlx \
  -m /Volumes/edata2/models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit \
  --port 9998 \
  --no-think \
  --concurrent 15 \
  --enable-prefix-caching \
  --prefill-step-size 8192
```

This disables the experimental environment preset while retaining the same
explicit CLI settings; it is not an all-CLI-default run. No model deletion,
cache deletion, recompilation or reinstallation is needed to switch off.
It does not change which `afm` Homebrew/PyPI resolves on your PATH.
