# Qwen Next MTP throughput profile

`AFM_QWEN_MTP_PROFILE=throughput-v1` expands the recovered Qwen Next throughput settings inside AFMKit. It is opt-in. It does not mutate the process environment, install anything, or change the default verifier. Explicit individual `AFM_QWEN_*` values override the profile, including `0` and empty values. Unknown profile names fail before model loading.

Set the profile before starting the process. The kernel owners capture their settings during initialization, so switching profiles in a running process is not supported.

```sh
AFM_QWEN_MTP_PROFILE=throughput-v1 /path/to/afm mlx \
  --model /path/to/Qwen-Next-checkpoint \
  --mtp --mtp-depth 3 --concurrent 2 --prefill-step-size 8192 \
  --port 9999
```

The explicit command arguments matter: the profile selects provider tuning, while the CLI selects MTP depth, scheduler capacity, and prefill chunk size. Capacity two enables the scheduler path; it does not mean a single-request benchmark sends two requests. Add `--no-think` only when the workload requires reasoning disabled. Prefix caching stays enabled unless explicitly disabled.

The profile selects batched verification, fused expert rows and routing, draft shortlist, native hyperconnection operations, scheduler/head sharing, bounded replay, and native checkpoint CPU ngram lookup. Replay has a 4096 MiB budget. Batched reduction arithmetic can change greedy decisions relative to strict singleton-equivalent verification. The profile uses one-pass prompt capture to retain the replay boundary without splitting the cold prompt at that boundary. The profile does not select sampled proposals, adaptive depth, or unrelated crowd experiments.

Use `AFM_QWEN_MTP_PROFILE=off` (or omit it) to disable the named profile. This controls configuration only; it does not revert the underlying recovered provider changes. Individual legacy switches remain available. An override can remove a speed benefit or change behavior; it is no longer the unmodified qualified recipe.

## Qualification boundary

This profile is for the recovered Qwen Next runtime, including the native mlx-community group32 checkpoint and the distinct ddalcu group64 representation. Those checkpoints are not interchangeable benchmark inputs. Compare AFM and a reference engine using the exact same checkpoint whenever claiming engine parity.

The underlying explicit settings recovered short-context, single-request reference-level speed on the shared ddalcu checkpoint. The native configuration passed 283/283 llmprobe conformance checks and 6/8 agentic tasks; two malformed tool-call tasks remain unresolved. Those are measurements of the sealed pre-profile binary with explicit settings. The named-profile binary built successfully, passed 203 provider tests (10 optional skips), and reproduced all 16 saved short-context responses from those explicit settings. Its native-checkpoint context sweep completed two trials at each size from 0.5K through 32K, and the API assertion suite passed 116/116 checks (two skips), including streaming parity, cache idempotency and batch interactions. These checks do not establish long-context reference parity or full model quality. Sustained concurrency and full release qualification remain incomplete. This document does not claim a release-ready default.

## Current regression gate

### October 4 vision qualification repair

The shared ddalcu vision overlay failed color probes while the native community
checkpoint passed. Its 333 vision tensors use `model.visual.*`; the Qwen Next
wrapper previously accepted only `vision_tower.*`, silently discarding the
overlay tower. The sanitizer now accepts native and HF vision namespaces and
normalizes channels-first patch weights through the existing vision sanitizer.
Canonical `vision_tower.*` names take precedence when aliases coexist.

Two preprocessing problems were also corrected: the Qwen processor now reads
nested `size.shortest_edge`/`longest_edge` pixel budgets, and Qwen Next no longer
receives a forced 1024-pixel service resize before its model-owned processor.
Other architectures retain the service resize policy. Legacy top-level pixel
budgets retain precedence. Text-only execution is unchanged by these repairs.

Ten targeted release tests passed. The live shared-overlay color probes now
return red and blue correctly, with 94 prompt tokens, matching the reference
instead of the previous 1,054 tokens. Resizing alone did not fix the wrong
answers; accepting the supplied vision tensors did. Native community color
probes also passed at 94 tokens after the resize correction. These two-color
checks are smoke coverage, not complete vision qualification.

Evidence is under
`/Volumes/edata/afm-release-artifacts/nightly-qualification-20261004/` in
`vision-layout-fixed-overlay`, `vision-sizing-fixed-community`, and
`vision-layout-unit-tests.log`. Full shared-checkpoint qualification follows
separately; no throughput parity or release readiness is implied by this fix.

The complete matched Promptfoo run passed 341/406 cases on the throughput candidate versus 357/406 on the checkpoint-compatible control (`382efd5a`). Request hashes match for all 406 cases. There are 24 candidate-only failures (4 native and 20 forced XML), 8 candidate-only passes, and 41 shared failures. Both builds passed all 76 native protocol cases. Do not describe the remaining differences as baseline failures or model limitations.

The no-think prefix introduced in `49a4e57d` passed unit tests but changed previously passing forced-XML tool decisions. Commit `c05916eb` withdraws that change and restores the established compatibility prompt. The restored prompt passes the missing-required-arguments case in four live trials; the complete forced-XML comparison remains pending. The partial fallback-candidate run was stopped after 346/406 cases, with ten candidate-only failures in forced-XML modes and seven improvements. No completed native case regressed. The earlier paired consumer suite passed 495 tests with two skips, but it is not final-provider coverage. Unit test success and short-context throughput do not qualify this candidate for release.

### Quantized hyper-connection fallback

The recovered fused quantized injection path was isolated as a cause of the four native candidate-only failures. A diagnostic binary restoring only that fallback passes all four cases in two trials both with MTP disabled and with the throughput profile enabled; disabling all hyper-connection fusion restores only two cases. Keep the existing residual-injection kernel enabled. Quantized injection fusion now requires `AFM_QWEN_FUSED_QUANTIZED_HC=1` and is not included in `throughput-v1`. Its numerical-tolerance tests are not evidence of unchanged model decisions. A native-checkpoint speed screen with this fallback measured 83.2 and 71.5 decode tok/s at 0.5K and 2K context, about 20–23% below the earlier named-profile run. This is a diagnostic binary, not final qualification. The corrected default passes the full provider suite (1,171 passed, 51 skipped). Full live-model and final-binary throughput validation remain pending.

### Experimental sparse verification attention

`AFM_QWEN_VERIFY_SPARSE_ATTENTION=1` enables a dense-BF16 split-K QSA
verification kernel only for the batched policy, one request, 2–8 verification
rows, and at least 8,192 cached-plus-current tokens. It is not part of the named
profile defaults. Singleton-equivalent verification and unsupported shapes retain
the existing path. If kernel admission declines after selecting block IDs, the
fallback reconstructs their exact mask and appends KV only once.

The Metal split and merge arithmetic comes from the MIT-licensed mlx-serve
v26.10.1 source at `02bee553f48cd3bc7d82aba0f8073820bd924738`; the full license
notice is included in `Qwen4ExpQSAVerificationSparseAttention.swift`. This uses
the verification-specific split-K grid, not the prefill gather grid.

Initial component validation: nine width/context pairs (widths 2/4/7 at
8K/16K/32K), a poisoned unselected-block/causal-tail test, and unsupported-shape
checks pass. Maximum observed difference from current chunk-2 masked attention
is 0.0009765625; this is not bit-identical arithmetic. At width four, a dependent
12-layer component screen measures roughly 3.37 to 2.16 ms at 16K and 4.61 to
2.20 ms at 32K. These are component timings, not full-model tokens per second.
Full-model throughput and behavioral regression qualification remain required.

### Prompt capture qualification

Snapshot backoff previously split a cold prompt before its final 31 tokens. On the native community checkpoint, a focused HTTP comparison reproduced an incomplete 589-token answer with that split. One-pass capture and disabled backoff both produced the same complete 999-token answer with the same binary and request. One-pass capture retains prefix reuse without changing cold prefill chunk geometry. Explicit `AFM_QWEN_MTP_ONE_PASS_CAPTURE=0` remains available for comparison.

Current-source direct-model checks passed five snapshot tests and a native growing-conversation test covering 64 known-answer responses and eight cancellation/isolation checks. These checks use strict verification and short contexts; they do not establish full serving-profile quality, long-context replay equivalence, or reference speed parity. Replayed and cold greedy output need not be token-identical because the forwarded matrix shapes differ. Broader qualification remains required before release promotion.
