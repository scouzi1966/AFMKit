# Native MLX GGUF proof of concept

This branch enables MLX's existing GGUF reader in AFMKit's vendored SwiftPM
build and adds a small C/Swift bridge for tensors and typed metadata.

The POC proves checkpoint ingestion, not architecture-independent inference.
GGUF contains tensors and metadata but no executable computation graph. AFMKit
must still map canonical GGUF tensor names and metadata to a supported model
implementation before generation can run.

## Supported path

- GGUF v3 parsing through MLX and pinned `gguflib`
- Q4_0, Q4_1, and Q8_0 packed tensor loading
- numeric metadata as `MLXArray`
- strings and string arrays
- dimension, overflow, metadata-bound, and tensor-bound validation
- descriptive errors identifying an unsupported tensor name and GGUF type

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

The next slice is a `qwen35` adapter that converts GGUF metadata into
`Qwen3_5MoEConfiguration`, renames canonical tensors into mlx-swift-lm module
paths, and supplies a tokenizer from GGUF metadata or a colocated tokenizer.
