# Native MLX GGUF proof of concept

This branch enables MLX's existing GGUF reader in AFMKit's vendored SwiftPM
build and adds a small C/Swift bridge for tensors and typed metadata.

The POC proves checkpoint ingestion and construction of the existing dense
Qwen3.5 text graph, not architecture-independent inference. GGUF contains
tensors and metadata but no executable computation graph, so each architecture
still needs a validated adapter before generation can run.

## Supported path

- GGUF v3 parsing through MLX and pinned `gguflib`
- Q4_0, Q4_1, and Q8_0 packed tensor loading
- numeric metadata as `MLXArray`
- strings and string arrays
- dimension, overflow, metadata-bound, and tensor-bound validation
- descriptive errors identifying an unsupported tensor name and GGUF type
- dense `qwen35` metadata conversion (the architecture used by Qwen 3.8 GGUF)
- canonical GGUF-to-MLXLLM tensor mapping, including Q8_0 sidecars
- reversal of llama.cpp's value-head tiling, RMSNorm offset, SSM decay encoding,
  and convolution layout transforms
- strict Q8_0 model construction with no missing or unused main-model parameters
- explicit exclusion of the appended MTP block from the 64-layer decoder graph

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
AFMKIT_QWEN35_GGUF_MODEL=/path/to/Qwen3.8-27B-Q8_0.gguf \
"$MACLOCAL_ROOT/Scripts/swiftpm-reliable.sh" test \
  --package-path "$AFMKIT_ROOT" --scratch-path "$SCRATCH" \
  -c release \
  --filter AFMMLXQwen35GGUFAdapterTests.constructsIntegrationModel
```

The verified reference file contains 64 decoder layers plus one MTP block and
constructs successfully as an MLXLLM Qwen3.5 model. The next slice is tokenizer
construction (preferably from colocated standard tokenizer assets first), AFM
model routing, and first-token parity. This ordering keeps tokenizer behavior
and HTTP integration independent from checkpoint-layout correctness.
