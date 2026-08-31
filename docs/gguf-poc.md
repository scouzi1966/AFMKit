# Native MLX GGUF proof of concept

This branch enables MLX's existing GGUF reader in AFMKit's vendored SwiftPM
build and adds a small C/Swift bridge for tensors and typed metadata.

The POC proves checkpoint ingestion and construction of the existing dense
Qwen3.5 text graph, not architecture-independent inference. GGUF contains
tensors and metadata but no executable computation graph, so each architecture
still needs a validated adapter before generation can run.

## Supported path

- GGUF v3 parsing through MLX and pinned `gguflib`
- Q4_0, Q4_1, and Q8_0 packed tensor loading with no second quantization
- opt-in, faithful FP16 compatibility decoding for Q5_0, Q5_1, Q2_K,
  Q3_K, Q4_K, Q5_K, Q6_K, and Q8_K tensors
- numeric metadata as `MLXArray`
- strings and string arrays
- dimension, overflow, metadata-bound, and tensor-bound validation
- descriptive errors identifying an unsupported tensor name and GGUF type
- dense `qwen35` metadata conversion (the architecture used by Qwen 3.8 GGUF)
- canonical GGUF-to-MLXLLM tensor mapping, including mixed 4-bit/8-bit packed
  sidecars
- reversal of llama.cpp's value-head tiling, SSM decay encoding, and
  convolution layout transforms
- preservation of llama.cpp's already shifted Qwen3.5 RMSNorm weights, which
  are the conventional RMSNorm values expected by MLXLLM
- strict mixed-quant model construction with no missing or unused main-model
  parameters
- explicit exclusion of the appended MTP block from the 64-layer decoder graph
- OpenAI chat-completions routing for a local `.gguf` file
- standard Hugging Face tokenizer assets, resolved from the GGUF directory or
  from an explicit `AFM_GGUF_TOKENIZER_PATH`

The HTTP POC intentionally accepts only dense Qwen3.5/Qwen3.8 text models.
Multimodal input, MTP, architecture-independent GGUF dispatch, IQ/TQ formats,
and the auxiliary Q8_1 dot-product layout are not claimed by this slice.

Legacy and K-quant support is experimental and must be enabled with
`AFM_EXPERIMENTAL_GGUF_QUANTS=1`. Those tensors are decoded faithfully to FP16;
they are not silently requantized into MLX's affine format. This provides broad
checkpoint compatibility at low implementation risk, but increases load time,
resident size, and peak memory in proportion to the tensors that require the
fallback. Q4_0, Q4_1, and Q8_0 continue to use the existing packed MLX path.
Native MLX kernels for the K layouts are the intended performance follow-up.

Run the AFMKit test target through maclocal-api's reliable wrapper. The first
run builds the test bundle; stage MLX's resource beside that bundle before the
opt-in checkpoint run:

```sh
MACLOCAL_ROOT=/path/to/maclocal-api
AFMKIT_ROOT=/path/to/AFMKit-gguf-poc
SCRATCH="$MACLOCAL_ROOT/.build/afmkit-gguf-tests"

"$MACLOCAL_ROOT/Scripts/swiftpm-reliable.sh" test \
  --package-path "$AFMKIT_ROOT" --scratch-path "$SCRATCH" \
  -c release --filter AFMMLXGGUFLoaderTests

cp "$AFMKIT_ROOT/Packages/AFMKitMLX/Sources/AFMKitMLX/Resources/default.metallib" \
  "$SCRATCH/arm64-apple-macosx/release/AFMKitPackageTests.xctest/Contents/MacOS/mlx.metallib"

AFMKIT_GGUF_INTEGRATION_MODEL=/path/to/model.gguf \
"$MACLOCAL_ROOT/Scripts/swiftpm-reliable.sh" test \
  --package-path "$AFMKIT_ROOT" --scratch-path "$SCRATCH" \
  -c release --filter AFMMLXGGUFLoaderTests
```

Run the strict model-construction integration separately because it reads the
entire checkpoint:

```sh
AFM_EXPERIMENTAL_GGUF_QUANTS=1 \
AFMKIT_QWEN35_GGUF_MODEL=/path/to/Qwen3.8-27B-Q4_0.gguf \
"$MACLOCAL_ROOT/Scripts/swiftpm-reliable.sh" test \
  --package-path "$AFMKIT_ROOT" --scratch-path "$SCRATCH" \
  -c release \
  --filter AFMMLXQwen35GGUFAdapterTests.constructsIntegrationModel
```

The verified reference file contains 64 decoder layers plus one MTP block and
constructs successfully as an MLXLLM Qwen3.5 model.

An even slower, explicit graph-evaluation check is available with
`AFMKIT_QWEN35_GGUF_FIRST_TOKEN` and the test filter
`AFMMLXQwen35GGUFAdapterTests.evaluatesIntegrationToken`. It evaluates a single
token without involving tokenizer or HTTP behavior. The Qwen 3.8 27B Q8_0
reference completed this check successfully in 232.8 seconds on a cold run.

## Run through maclocal-api

Build the consumer against this paired AFMKit worktree, then point AFM at the
exact GGUF file. If tokenizer assets are not colocated with that file, identify
a compatible standard Hugging Face tokenizer directory explicitly:

```sh
MACLOCAL_AFMKIT_PATH=/path/to/AFMKit-gguf-poc \
  ./Scripts/swiftpm-reliable.sh build -c release --product afm

AFM_GGUF_TOKENIZER_PATH=/path/to/qwen-tokenizer-assets \
AFM_EXPERIMENTAL_GGUF_QUANTS=1 \
MACAFM_MLX_MODEL_CACHE=/path/to/model-cache \
  .build/arm64-apple-macosx/release/afm mlx \
  -m /path/to/Qwen3.8-27B-Q4_0.gguf --port 9999 --prewarm n
```

The tokenizer directory must contain at least `tokenizer.json` and
`tokenizer_config.json`; `chat_template.jinja` is used when present. The model
identifier exposed by the server is the exact GGUF path supplied on startup.
The tested GGUF also embeds `tokenizer.chat_template`, but this POC does not yet
construct its tokenizer or template from GGUF metadata. It therefore validates
the paired standard-tokenizer path, not fully self-contained GGUF startup.

The Qwen 3.8 27B Q8_0 reference was exercised through the release server with
llmprobe 0.6.0. The quick suite reported Core 6/6 and 100% engine conformance
(models 4/4, chat 29/29), including SSE streaming and tool-call serialization.
The default suite then processed 36,118 tokens in 31m27s and reported Core 9/9,
97.2% conformance on implemented surfaces, 100% model capability, 3/3 agentic
tasks, and 100% engine fidelity. Its two conformance failures were existing
consumer-contract behavior: `n > 1` silently returns one choice, and `stop`
does not accept the OpenAI-compatible bare-string form. Missing Responses,
Anthropic, vision, and frontier endpoints describe the consumer API surface
and are independent of GGUF loading.

In the same release environment, Q8_0 decoded at roughly 4.3–5.2 tokens/s,
versus roughly 14.2 tokens/s for the native MLX 4-bit checkpoint. Cold Q8_0
startup took about 218 seconds and transient memory reached roughly 46 GB before
settling near 12.7 GB. These figures are baseline evidence that the POC is a
compatibility implementation, not yet a performance-equivalent GGUF runtime.

The mixed `Qwen3.8-27B-Q4_0.gguf` reference (which includes Q5_K and Q6_K
tensors despite its filename) loaded successfully with the experimental flag.
On the same release build it reached server-ready state in about 101 seconds,
peaked near 34.7 GB during loading, and settled near 19.9 GB. A deterministic
no-thinking smoke request returned the exact requested text at 2.63 tokens/s.
The lower disk size therefore does not imply lower runtime memory or faster
generation while K tensors use the FP16 compatibility decoder.
