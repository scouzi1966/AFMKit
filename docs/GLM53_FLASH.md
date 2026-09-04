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
  `805a9e939fa8c1bff8d8ffdf041c051b71a914aa`.
- Blaizzy `mlx-vlm`'s MLX reference at revision
  `3fd38f4c2d01ef6f8b58f7c1ebcc2859937c1b04`.
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

## Decode submission ladder

GLM decode uses a four-layer asynchronous submission ladder. The model still
constructs exactly the same lazy MLX graph and updates the same per-request
caches; the intermediate `asyncEval` calls only let GPU execution begin while
Swift constructs later layers. The ladder is restricted to single-token decode,
so prefill graph structure and throughput are unchanged. Set
`VMLX_GLM53_DECODE_ASYNC_LADDER=0` to recover the unsplit diagnostic path.

The default stride was selected on an M3 Ultra with the same local
`GLM-5.3-Flash-oQ4e` checkpoint, temperature zero, prefix reuse disabled, and
byte-for-byte output checks:

| Decode configuration | 128-token mean | 256-token mean | Output |
| --- | ---: | ---: | --- |
| Unsplit control | 25.24 tok/s | 24.86 tok/s | reference |
| Ladder stride 2 | 28.68 tok/s | — | identical |
| Ladder stride 4 | 28.69 tok/s | 28.52 tok/s | identical |
| Ladder stride 8 | 28.56 tok/s | — | identical |
| Ladder stride 12 | 27.71 tok/s | — | identical |

Stride four improved the longer serial control by 14.7%. A two-request
continuous-batch check improved aggregate throughput from 29.65 to 39.50 tok/s
(33.2%) with identical output for both lanes. Stride four was preferred over
the statistically tied stride two because it introduces half as many explicit
submission points. This is an architecture-specific policy: the same ladder
was neutral on Qwen3.8-27B and DeepSeek-V4-Flash-0731 and was not enabled for
either model.

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

The self-contained `MLXVLM` implementation supports the published image/video
tower and GLM processor contract. It uses the official aspect-preserving resize
and right/bottom zero-padding path, temporal patch grouping, timestamped video
frame expansion, and video-boundary-aware feature placement. Runtime admission
requires all six multimodal token IDs, compatible image and video processor
metadata, and a complete shape/dtype-validated `model.visual.*` or converted
`vision_model.*` tower. A checkpoint missing any of those assets loads through
the language-only path and cannot advertise or accept media.

The published processor's `max_frames=2048` is also a hard qualification and
runtime ceiling. Checkpoints requesting a larger video sampling budget are not
admitted, and direct sampler overrides above 2,048 frames fail before index or
frame extraction allocation.

The LLM sanitizer still deliberately excludes the checkpoint's extra
next-token prediction layer. GLM-specific MTP speculative decoding is tracked by
[#44](https://github.com/scouzi1966/AFMKit/issues/44). The original FP8
Transformers shards are not claimed as directly loadable on Apple Silicon;
their raw `weight_scale_inv` keys remain visible so the strict loader rejects
them instead of silently interpreting FP8 byte payloads as ordinary integers.
