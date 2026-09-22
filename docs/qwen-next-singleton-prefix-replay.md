# Qwen Next singleton-prefix replay qualification

## Failure and diagnosis

A Qwen Next coding-agent task that checks an already-correct file passed from
a fresh cache but failed after an unrelated raw request containing only
`<|im_start|>`. The subsequent 776-token chat request reused one token. This
was reproducible twice per mode, with identical model weights and greedy
sampling. Omitted tool strictness, native tool-history formatting, and raw
completion parsing were corrected separately before this investigation.

The differential regression distinguishes computation from persistence:

```text
cold:              [0 .. 745) -> [745 .. 775) -> [775 .. 776)
split in place:    [0 .. 1) -> [1 .. 745) -> [745 .. 775) -> [775 .. 776)
snapshot/restore:  [0 .. 1) -> save/restore -> same split suffix
```

On the qualified BF16/affine-q4 checkpoint, split and restored logits and final
cache tensors were bitwise identical; advancing the donor and recipient did
not change the saved snapshot. Cold versus split differed by up to 5.359375 in
logits and changed the greedy first token. The first differences arise in the
singleton HyperConnection and quantized projection reductions. Gated-delta
recurrence was exact when supplied identical operands. The result does not
support diagnosing this reproducer as snapshot corruption.

## Bounded policy change

For Qwen Next text and vision model types, a **one-token partial match** is
recomputed with the new prompt rather than restored. Serial streaming,
non-streaming, batch prefill, and batch admission estimates use the same
policy. Exact repeated prompts still use their saved logits, including
one-token prompts. Longer reusable prefixes and other model families keep
their existing behavior. No kernel, decode default, sampler, or user tuning
environment variable changes are involved.

This is a narrow mitigation for a demonstrated execution-width boundary,
not a claim of shape-invariant floating-point arithmetic or general quality
parity. Longer-prefix schedules can also differ numerically and require
independent behavioral qualification. The reference segregates tool-enabled
and tool-free cache entries; its cancelled-prefill commit floor is not a
universal prefix-reuse threshold and was not copied as one.

## Verification

- 18 targeted tests pass, including the exact-checkpoint differential.
- Fresh HTTP controls: cold, one-token priming, two-token priming, and a
  32-token prompt creating a one-token backoff snapshot each pass the complete
  already-done task twice. Two-token reuse remains active.
- Serial and two-request concurrent HTTP controls pass with streaming both
  off and on. A second admission may legitimately reuse the first admission's
  completed prefill; it must not extend the singleton snapshot.
- Exact-repeat responses report 776 cached prompt tokens.
- Independent review found no confirmed production regression. Its requested
  adjacent-boundary and production-wiring checks are the controls above.

The full llmprobe, matched Context benchmark, and comprehensive suite remain
separate gates. Do not promote these targeted checks to a release pass or a
throughput claim. Qualification artifacts are retained outside the repository
under `/Volumes/edata2/afm-benchmarks/afm-agentic-divergence-20260921`.

`QwenNextPrefixBoundaryTests` runs its small CPU fixture by default. The real
checkpoint differential requires `AFM_QWEN_PREFILL_QUALITY_MODEL`,
`AFM_QWEN_PREFILL_QUALITY_TOKENS` (frozen prompt fixture), and
`AFM_QWEN_PREFILL_QUALITY_OUT` (new output directory). These configure the test
only, not runtime optimizations. Run through the consumer's
`Scripts/swiftpm-reliable.sh test --package-path <AFMKit> ...` wrapper so the
canonical Metal resources are staged correctly.
