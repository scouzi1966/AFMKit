# Apertus 2509 integration

Reference architecture: [Python mlx-lm Apertus](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/apertus.py).
Model contract: [Swiss AI Apertus 8B Instruct 2509](https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509).

## Capabilities and controls

Apertus 2509 is a text-only decoder model. It has no vision encoder or trained
vision projector; selecting a VLM loader cannot supply those missing weights.
There is no model-native MTP head in this checkpoint.

Deliberation uses AFM's existing `enable_thinking` chat-template argument.
`--no-think` remains the overriding disable control. The native template's
default is preserved when no setting is supplied. Apertus supports a binary
switch, not distinct low/high reasoning budgets; supported nonzero effort
values map to enabled deliberation. Explicit disabling takes precedence.

The existing reasoning extractor consumes `<|inner_prefix|>` and
`<|inner_suffix|>`, sending thought text through the reasoning channel rather
than displaying model-specific markers in the ordinary answer. Raw mode retains
its existing bypass semantics. An enabled model need not emit thoughts on every
request.

## Native tools

The model-owned template accepts flat function definitions. AFM's OpenAI tool
schema envelope is unwrapped at input preparation, independently of output
parser selection. Assistant history retains native structured calls; tool
results must be JSON values, so ordinary text is JSON-quoted rather than wrapped
in generic XML.

Native generated calls have this shape:

```text
<|tools_prefix|>[{"weather":{"city":"Bern"}},{"clock":{}}]<|tools_suffix|>
```

The checkpoint declares the suffix token as EOS. Generation may consume it
before detokenization. Finalization therefore accepts a complete JSON array
without that suffix, but never repairs a truncated array, fabricates arguments,
or executes only the valid portion of a malformed array. Ordinary streaming
buffers until a complete marker or finalization. Completed calls are emitted
before completion metadata; cancellation does not finalize buffered calls.

## Regression coverage and reference comparisons

- Checkpoint activation parameter loading, BF16 xIELU evaluation, and integer
  versus floating-point RoPE metadata.
- Quantized KV-cache prefill and decode through shared cache-aware attention.
- Native array parsing and provider streaming across every chunk boundary.
- EOS-finalized arrays, malformed members, and reasoning marker splits.
- Optional local-checkpoint generation tests via `APERTUS_TEST_MODEL`.
- Optional same-token Python reference comparison via `APERTUS_REFERENCE_JSONL`.

The reference JSONL starts with a record containing the exact local `model`
path and runtime versions. Each following record contains `case`, `input_tokens`,
and `output_tokens`. Greedy comparison excludes declared EOS tokens on both
sides; no prompt rerendering occurs in that comparison. No checkpoint download
is triggered by default tests.

The six-bit checkpoint matched the initial four greedy reference probes. The
four-bit checkpoint showed divergences that still require investigation; neither
reference parity on a few prompts nor an assertion pass rate proves broad model
quality. Comprehensive Codex-judged qualification must report engine defects,
model-behavior failures, and forced-parser experiments separately. Do not relax
assertions to achieve a target failure count.
