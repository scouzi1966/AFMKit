# GLM-5.3-Flash MLX Swift support

AFMKit's vendored `MLXLLM` runtime supports the language decoder published as
[`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash). Both
`glm5_next` wrapper configurations and flat `glm5_next_text` configurations are
registered.

## Implementation provenance

The port was checked on 2026-08-27 against:

- Hugging Face checkpoint/config revision
  `04c4e9e95c5da8862dced7e5056455116f83a7e0`.
- Hugging Face Transformers `glm5_next` at main revision
  `83d024e1bfed0d425d20bcde2b46a56b2333906e`.
- Blaizzy `mlx-vlm`'s MLX reference at revision
  `1f507f19276843fc9a8a230712c995ed9ebef016`.
- `ml-explore/mlx-lm` main revision
  `ff8289c67a4661b232e30466b231b34dbac3428b` and
  `ml-explore/mlx-swift-lm` main revision
  `db767efca373bcc215e2c340e97751c28f570491`; neither official MLX repository
  contained a GLM-5.3 implementation at inspection time.

The Swift implementation includes mHC residual streams, the hybrid Kimi Delta
Attention/NoPE MLA decoder, pooled DSA selection, dense and routed-expert FFNs,
incremental caches, continuous-batch padding masks, checkpoint key sanitation,
and the GLM chat-template tool-call format.

The published template accepts `reasoning_effort` values of `low`, `high`, or
`max` and has no true off switch. AFM therefore maps `--no-think` to the
documented lowest available setting, `reasoning_effort=low`, rather than sending
the unrelated `enable_thinking` flag.

## Checkpoint scope

The official upstream repository publishes FP8 Transformers shards, not a
converted MLX checkpoint. The runtime implementation is intended for an MLX
conversion whose weights retain the published architecture and names. A public
candidate is
[`Vontra/GLM-5.3-Flash-MLX-4bit-MTP`](https://huggingface.co/Vontra/GLM-5.3-Flash-MLX-4bit-MTP).
Focused tests validate configuration decoding, weight sanitation, cache
construction, serial and left-padded batched prefill/decode, and tool-format
selection with a tiny model. Full-weight qualification is recorded separately
from these architecture tests.

This is language-only support. The LLM sanitizer deliberately excludes the
checkpoint's image/video tower and its extra next-token prediction layer. The
corresponding self-contained `MLXVLM` work is tracked by AFMKit
[#43](https://github.com/scouzi1966/AFMKit/issues/43), and GLM-specific MTP
speculative decoding by
[#44](https://github.com/scouzi1966/AFMKit/issues/44). The original FP8
Transformers shards are not claimed as directly loadable on Apple Silicon;
their raw `weight_scale_inv` keys remain visible so the strict loader rejects
them instead of silently interpreting FP8 byte payloads as ordinary integers.
