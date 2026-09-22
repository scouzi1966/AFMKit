# Preserve tool history through multimodal message adaptation

## Reproduction and cause

The four-model full suite found a Qwen Next follow-up that called `get_weather`
again instead of answering from the supplied result. The captured prompt was
missing the historical assistant call even though AFM's provider constructed
`Chat.Message.toolCalls` correctly. The affected checkpoint selects the VLM
factory and `Qwen3VLProcessor`, including for text-only requests.

```text
OpenAI assistant tool_calls + tool result
  -> MLXModelService.buildUserInput
       structured Chat.Message.toolCalls / toolResponses / name
  -> Qwen3VLMessageGenerator
       BEFORE: role + multimodal content only; metadata lost
       AFTER: DefaultMessageGenerator metadata + multimodal content
  -> checkpoint-owned Jinja template
       exactly one native call followed by its result
  -> tokenization -> existing prefill/cache/decode paths
```

`QwenVisionToolHistoryTests` reproduces the loss without weights or GPU work.
Before repair, three of four test methods failed (12 assertions). Media order
was the passing control. The published template fixture matches both the
qualified Qwen Next and Qwen 3.8 27B checkpoints byte-for-byte; its source and
SHA256 are recorded alongside the fixture.

An adapter-only repair revealed a second interaction: the dense checkpoint's
`qwen3_5` model type was not included in native history ownership. It received
both generic JSON history in content and native XML history from metadata, and
its results were wrapped twice. The production-policy composition test failed
16 assertions for that architecture while the original four adapter tests
passed. Native ownership now covers the verified `qwen4_exp` and `qwen3_5`
paths, only for the existing automatic / `qwen3_xml` parser policy.

## Scope and performance

Only message adaptation and the narrow ownership predicate change. The
generator preserves the common implementation's `name`, `tool_calls`, and
`tool_responses`, replacing only `content` with the existing image/video/text
array. Shared GLM processors also retain metadata; their native template
ownership rule already exists. Forced-parser ownership remains unchanged and
has a separate assistant-rendering control.

This work adds dictionary handling during prompt construction, not per-token
GPU execution. Kernels, weights, quantization, sampling, prefix-cache policy,
batch scheduling, and media ordering are unchanged. Restoring missing history
can legitimately change prompt length and model output. Do not compare those
requests as if the old and new token sequences were identical, or claim zero
end-to-end overhead without measurement.

The cached-versus-uncached behavioral difference is a separate qualification
question. The missing metadata is proven; it does not by itself prove cache
corruption or explain every model-quality failure. Recheck the corrected live
prompts, serial/batch and cache modes, independent review, protocol coverage,
same-checkpoint performance, and the comprehensive judged suite before release.
