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

### Qualification checkpoint (2026-09-09)

The comprehensive nightly run using provider revision `0d325fa8` completed
91 records for each quantization, with Codex judging enabled:

| Checkpoint | Passed assertions | Failed assertions | Skipped |
| --- | ---: | ---: | ---: |
| `mlx-community/Apertus-8B-Instruct-2509-4bit` (`c5e6ae2e52c4149f36cb8e47b7ab1489ef885fed`) | 82 | 9 | 0 |
| `mlx-community/Apertus-8B-Instruct-2509-6bit` (`a2bfd2b25c841b6fc6e920e883a88619be3d6012`) | 88 | 3 | 0 |

These are point-in-time results, not final qualification of subsequent changes.
Codex also identified degeneration and incorrect generated code in some records
whose mechanical assertions passed. The four-bit result does not satisfy a
seven-failure allowance. The forced `qwen3_xml` experiment is included in the
counts and does not establish a failure of the native Apertus parser.

The installed Python reference (`mlx-lm` 0.31.3, MLX 0.32.2), using the exact
four-bit checkpoint and comprehensive tool schemas/prompts with deliberation
disabled, also returned prose rather than native calls in all six tool probes.
This is evidence of cross-runtime behavior, not proof of token-level parity:
the actual generated prose differs, and the HTTP prompt-rendering path still
requires separate comparison.

Using those six tool probes on the six-bit checkpoint, Swift matched every
reference output token after excluding declared EOS tokens. Both emitted the
same incorrect flattened/string-valued arguments for the nested-object probe.
This isolates that mistake from output parsing. It does not guarantee that the
model will call tools reliably with every schema or prompt.

Four-bit **KV-cache** quantization (distinct from weight quantization) produces
poor output in both the Python reference and Swift on the six-bit checkpoint's
machine-learning summary probe. Shared attention masking defects were corrected
and covered by dtype, batched-head, cached-decode, and fully-masked-row tests;
those unit passes do not establish generation quality or throughput parity.

Final-binary reruns with provider code `84942015` retained the same totals:
four-bit 82/91 and six-bit 88/91, both Codex-judged with zero skips. Nine targeted
six-bit API cases passed, including cancellation followed by coherent cached
request recovery and raw tool-parser bypass. A KV8 summary probe was coherent
in both runtimes but not token-identical; this is not broad KV8 qualification.

The four-bit reference reproduced the fenced-JSON and missing-HTML/newline
formatting failures when given AFM's default system instruction (`You are a
helpful assistant`). Disabling reference compilation did not change any of
the six tool-probe token sequences. The four-bit acceptance shortfall remains;
it must not be hidden by changing prompts, defaults, or assertion accounting.

The six-bit KV4 run also exposed expensive consumer response-tail sanitization
after generation. That separate, model-independent latency fix is tracked in
[maclocal-api PR #300](https://github.com/scouzi1966/maclocal-api/pull/300).
It does not change model outputs or resolve KV4 degeneration.
