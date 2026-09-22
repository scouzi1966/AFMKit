# GLM prefix replay and tool-history corrections

The September 21–22 qualification of PR #128 uncovered three separate provider
defects. These are not interchangeable with the Qwen singleton arithmetic
divergence described in `qwen-next-singleton-prefix-replay.md`.

## Composite state was not restored

GLM sparse-attention layers use `CacheList(KVCacheSimple(), KVCacheSimple())`:
one child owns main attention history, the other owns sparse-indexer history.
The key-only indexer has a valid zero-width value tensor.

```text
Saved boundary: [main K, main V, index K, index V] + first-token logits
                           |
Old fresh CacheList setter: split by children's CURRENT state counts [0, 0]
                           |
                     empty children
                           |
Saved logits mask the loss at token 1; token 2 lacks attention history
```

The retained failing regression restored zero tensors rather than four and an
offset of zero rather than three. This is actual state loss, not evidence of a
floating-point scheduling difference.

`MLXPrefixReplayPolicy` now validates the snapshot before accepting its saved
logits. Known composites of simple K/V children require complete pairs, matching
batch/head/sequence dimensions and the recorded source-token boundary. Invalid
snapshots become a cold miss, with no partial mutation or saved-logit reuse.
Valid pairs are installed directly into each child. Plain `ArraysCache` offsets
are restored from the saved boundary; subclasses retain their own contracts.
Serial, serial-streaming and scheduler paths use the same policy.

The existing snapshot ownership/copy behavior is retained. The new validation
examines array metadata at request admission; it does not add tensor copies,
GPU evaluation or per-token decode work.

## Structured output acquired a synthetic reasoning prefix

GLM's template can end in `<think>` even when thinking is reduced. AFM previously
injected that opening delimiter into output even when a JSON grammar constrained
generation. Preserving literal reasoning markers in JSON then exposed the
synthetic prefix as invalid JSON.

`MLXOutputReasoningPolicy` chooses framing per request. Structured JSON and raw
completions do not acquire synthetic chat reasoning delimiters. Ordinary chat
keeps its existing reasoning channel and token accounting. Literal marker-shaped
strings in JSON remain application data, including during stop-sequence handling.

## Native tool history was duplicated

The GLM template renders structured calls using native function/argument tags
and wraps tool results itself. AFM also added generic JSON call markup to
assistant content and wrapped results in a second JSON-like envelope. The model
therefore saw two incompatible versions of each prior call. If it copied the
generic form, the native parser interpreted the whole JSON object as a function
name. This error occurred before cache lookup, including on cold requests with
history.

Default GLM history now preserves assistant prose and tool-result text verbatim,
leaving call and result framing to its native template. Structured arguments
remain dictionaries. Forced-parser and unrelated architecture policies are
unchanged. No permissive output-parser workaround was added.

## Validation scope

- Regression tests cover fresh composite restore, next-token state advancement,
  shared snapshot stability, malformed boundaries, and subclass offset isolation.
- Framing tests cover JSON marker preservation, ordinary reasoning, raw output,
  token accounting and stop handling at every two-chunk split.
- GLM template-fragment tests use the production ownership predicate and cover
  parameterless/parallel calls, assistant prose and unchanged source results.
- Live serial checks preserve complete multi-token visible and reasoning output
  on exact warm replay and validate marker-bearing JSON, streaming and nonstreaming.
- A simultaneous two-request check changed one coherent answer relative to a
  single-row donor. That comparison changes batch geometry. It is retained as a
  diagnostic, not reported as established cache corruption or concurrency parity.

Full post-fix llmprobe, Context and comprehensive qualification must be reported
against the actual tested binary hash. Earlier Qwen scores do not qualify a later
binary automatically. Test evidence is retained outside git under
`/Volumes/edata2/afm-benchmarks/afm-agentic-divergence-20260921`.
