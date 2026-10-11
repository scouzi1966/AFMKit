# Qwen Next thinking replay qualification

## Two distinct reproduced failures

This investigation does not attribute every historical slow coding run to one
cause. Fixed captured histories exposed two different failures:

1. Long tool conversations could repeat planning text until the output cap.
   The 33K example exhausted 16,384 tokens without a complete tool call; another
   44K example generated 7,996 tokens of repetitive reasoning.
2. A short second user turn could exhaust 4,096 tokens with no visible answer
   when the native template preserved previous-turn thinking. The corresponding
   cold request completed, and disabling previous-turn preservation fixed the
   warm request without removing its useful cache hit.

Neither observation alone establishes corrupted persisted cache tensors.

## Long-history numerical fix

`Qwen4ExpAttentionCache.metaState` now preserves implicit/explicit position
provenance across radix snapshots. Tensor equality alone cannot validate all
cache metadata. Dedicated snapshot tests cover this separately.

`Qwen4ExpGatedDeltaPrework` uses the qualified reference-style Q/K normalization
for eligible single-slot prefill sequences of at least 16 tokens. Its arithmetic
matches the reference RMSNorm sequence: F32 square/reduction, mean plus epsilon,
reciprocal square root, BF16 normalization rounding, and separate BF16 scaling.
The recurrent state remains FP32. Decode and verifier arithmetic are unchanged.

An initial 128-token threshold repaired free generation but failed independent
prefix-state assertions for shorter prefill chunks. It was rejected rather than
weakening those assertions. The final 16-token policy retains one-pass prompt
capture and passes the independent prefix-state controls.

Controlled BF16 recurrent-state and disabled packed-kernel experiments did not
provide an acceptable remedy. They are not serving defaults. Canonical full
chunk-boundary replay aligned logits but required substantially more recompute;
it was not adopted as a default performance tradeoff.

## Previous-turn history default

`MLXHistoryTemplatePolicy` detects support from the effective chat template,
not a checkpoint-name allowlist. For templates exposing `preserve_thinking`,
the default is false. Explicit server defaults or request kwargs retain priority.
Templates without the option are unchanged. `enable_thinking` remains separate:
the new answer can still think, and the template retains current-turn reasoning.

The native template is read once per model load. A selected parser/native-tool
or built-in template override takes precedence over that native template when
deciding support. No filesystem lookup is added to each decode step or request.
Loading another model refreshes the stored template rather than inheriting the
previous model's capability.

This is also the previous-turn preservation policy observed in the pinned
reference's `src/chat.zig` extra-context handling. No external optimization
environment setting is needed in AFM. Users can explicitly retain old thinking
with `chat_template_kwargs: {"preserve_thinking": true}`; doing so restores the
old prompt behaviour and is not claimed to repair its reproduced quality issue.

## Evidence as of October 11, 2026

All runtime comparisons below use the same physical checkpoint:
`/Volumes/edata2/models/qualification/Qwen3.8-Flash-Next-ddalcu-vision-overlay-20261001`.
The pinned reference is mlx-serve 26.10.1, not an assertion about its latest release.

Candidate provider source: `cc77cde05e0fc2fdb6701f0dad16e95329318980`.
Consumer source: `6839b0566dbd166f11ad4203db9568165ac8ff49`.
Release binary SHA256:
`e767dab82905352a9f5a7487427a8f308fc9d6d0c30af11e368e5aaee9dc8a10`.

| Fixed history | Output tokens | Cached prompt tokens | API seconds | Result |
|---|---:|---:|---:|---|
| Short user follow-up | 744 | 51 | 11.32 | Coherent visible answer, stop |
| 33K tool conversation | 717 | 33,320 | 12.61 | Complete exec_command |
| 44K tool conversation | 757 | 44,404 | 13.41 | Complete exec_command |

The long-history outputs retain the successful counts from the preceding fixed
candidate. Captured-history tests never execute generated tool calls and never
feed generated output into the next frozen request. They are not full coding
project evaluations.

Four history-policy tests pass, including a real native-template rendering
assertion for dropping old reasoning while retaining current-turn reasoning.
The preceding numerical qualification had 100 passed tests, one optional
production-shape skip, and zero failures. Do not conflate these selections with
a complete fresh release suite.

The subsequent broader selection on the new default passes 134 tests (114
XCTest and 20 Swift Testing cases), with one optional production-shape skip and
no failures. A separate diagnostic confirms both no-tool turns actually used
native MTP: 316 and 1,050 cycles, respectively, with zero backbone fallbacks.
Diagnostic timings are excluded from throughput comparisons.

Native no-tool MTP off/on two-turn probes both produced coherent visible answers.
The MTP follow-up was verbose (2,695 output tokens, 53.73 seconds), so this is
not a performance win. Requests containing tools retain the existing AR fallback;
loading an MTP head alone is not proof that a tool request used speculation.

The new binary's first four Context sizes, three measured trials per context
after separate warmup, changed median prefill by -0.27% to +0.12% and decode by
-0.24% to +0.35% versus the preceding fixed candidate. No material regression
was observed. Against the pinned reference the existing 0.5K prefill gap still
exceeds 10%; do not claim all measurements are within the target.

## Remaining qualification and reporting limits

A fresh coding-agent run on the new default passed 11/11 automated checks in
666 seconds, generating 38,503 tokens across 22 requests. This is slower than
the preceding successful AFM run (542 seconds, 30,705 tokens), not evidence of
improved agentic latency. Its input-token reuse fraction was 92.9%.

The preceding candidate's single matched Orbital pair passed 11/11
automated checks for both engines (AFM 542 seconds, reference 504 seconds).
That pair does not establish statistical superiority.

To isolate the newest default from different fresh-project trajectories, the
expensive ninth request was run against both preserved AFM binaries after the
same eight frozen preceding inputs. Both restored 11,168 of 11,527 prompt tokens,
generated 13,581 output tokens, and returned exactly identical reasoning and
tool arguments. Prior/new times were 219.8/222.1 seconds. This request therefore
does not demonstrate a regression introduced by the new history default.
All 22 project requests have no generated assistant items before their latest
user message; the previous-turn default removes no reasoning from that trace.

The pinned reference on the same frozen history, with its thinking budget
explicitly disabled for this diagnostic, completed in 168.0 seconds and 10,755
output tokens. Its reasoning was also lengthy, but shorter than AFM's. This
remaining difference is not resolved or dismissed as harmless randomness.
A separate cold-versus-warm AFM control is in progress to investigate the effect
of prefix execution. Generated calls in all frozen-history diagnostics are
inspected, never executed.

The Responses boundary still reports `reasoning_tokens: 0` despite returning
actual reasoning items. Total output-token counts and saved reasoning text were
used for this investigation; the separate accounting defect is not fixed here.

Evidence is untracked under:
`/Volumes/edata/dev/CODEX/codex-local-coding-eval-20261007`.
See `thinking-fix-qualification-status-20261011.json` and
`thinking-history-default-performance-20261011.md`, along with the preserved
raw requests, responses, build logs and binary provenance. Nothing in this
document implies a main merge, release, or completed nightly qualification.
