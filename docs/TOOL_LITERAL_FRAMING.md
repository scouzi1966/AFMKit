# Literal tool delimiters and streaming boundaries

## Confirmed defect

A valid JSON argument or Qwen XML parameter can contain literal text such as
`</tool_call>`, `</think>`, `<|im_end|>`, or `</function>`. The serial processor
and provider fallback previously treated the first closing substring as the
end of the invocation. Complete, valid output could therefore become a missing
tool, a partial argument dictionary, or leaked visible markup. This is an
engine defect, demonstrated without a model or sampling.

The initial `ToolEnvelopeLiteralTests` run reproduced failures in six of seven
tests before the implementation changed. The marker-only DeepSeek DSML control
passed: this finding does not explain all live literal-fidelity failures in all
architectures. A separate required-tool token-limit correction is also part of
the qualification branch; these are distinct defects.

## Boundary ownership

```text
Serial model output
  -> MLXLMCommon ToolCallProcessor / JSON or XML parser
  -> service stop filtering of visible text
  -> provider fallback -> event translation

Batch model output
  -> provider ToolCallStreamingRuntime
  -> tool events (arguments bypass visible-text stop filtering)
  -> visible text -> stop filtering, including recovered EOF text
```

`ToolCallEnvelopeScanner` owns JSON/Qwen lexical boundaries. It tracks JSON
quotes and escapes, or XML parameter headers/bodies, across appended UTF-8
chunks. It does not decode JSON to decide whether a new token closes a call.
Completed-text parsing uses the same structural boundaries, classifying a bare
JSON object or bare XML function before searching its contents for other
envelopes. Adjacent calls remain separate and ordered.

This scanner is not a universal parser: GLM keeps its native `arg_value`
framing, and DSML/ATEM retain their existing format-specific paths. Raw DSML
strings are not interpreted as JSON quote grammar.

## Native parsing versus explicit repair

Native incomplete calls stay incomplete. Existing `afm_adaptive_xml`
compatibility mode can still repair its recognized malformed opener families
and salvage a missing parameter close. Such repairs must validate a candidate
through the compatibility parser; native parsing never retries arbitrary first
closing tags to make an invalid call succeed.

Malformed adaptive streams whose quotes cannot be framed safely are retained
until EOF. Their actual remaining visible text, including a failed repair, is
preserved rather than discarded or augmented with a fabricated closing tag.
Batch EOF text goes through the existing stop policy before flushing. Serial
generation already performs that filtering upstream; adding a second serial
filter would change semantics.

When a final chunk closes additional XML parameters and the envelope together,
their argument deltas are emitted before the streaming JSON object is closed.
The streamed arguments and final collected arguments must agree.

## Regression coverage and performance limits

`ToolEnvelopeLiteralTests` covers direct parsers, the serial processor, shared
streaming/batch runtime, completed and bare fallbacks, and the
serial-to-provider-to-translator composition. Fixtures include every two-chunk
split and character-sized chunks, adjacent calls, quoted/escaped JSON,
Unicode, nested literal function/envelope examples, native incomplete data,
explicit compatibility repairs, DSML raw quotes, and batch EOF stop filtering.
Existing Qwen and GLM suites remain separate regression controls.

The new framing scan is incremental and stores only lexical state and a byte
offset. It does not allocate a decoded copy of each growing buffer or change
GPU kernels, sampling, cache policy, model weights, or request concurrency.
Ordinary text without tool calls keeps its existing fast path. The pre-existing
incremental XML parameter regex still rescans accumulated content; this change
does not claim that the entire parser is linear-time.

Live llmprobe, same-checkpoint Context comparisons, and the comprehensive suite
with Codex-GLM judging are separate qualification gates. Passing these unit
fixtures does not establish model behavior parity, eliminate every literal
fidelity warning, or establish a throughput improvement. Record the binary
hash and preserve both successful and failed artifacts for each candidate.
