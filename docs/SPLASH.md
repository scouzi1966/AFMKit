# Splash native provider

`AFMKitSplash` is a macOS provider module depending only on `AFMKitCore` and
Swift Tokenizers. It does not depend on MLX, DwarfStar, Vapor, or Python.
`AFMSplashProviderFactory` registers as `splash`; models implement `AFMModel`
and `AFMTextTokenizing`, so AFMEngine and application HTTP adapters use their
existing provider interfaces.

The adapter launches the precompiled C++/Metal engine as a supervised child and
speaks Splash's binary v5 protocol. It does not launch the Python HTTP server.
Swift applies the model package's chat template and tokenizer. Native inference
still requires macOS 26.4+, M3 or newer, and sufficient unified memory. The
platform check fails with a readable error on earlier macOS versions. The
native engine enforces its hardware and memory requirements.

## Immutable runtime

`Sources/AFMKitSplash/Resources/splash-release.json` pins upstream release
**1.0**, commit `c675ed23e6942b5353961246e68b08cd63fb4ee9`, protocol 5, and
SHA-256 `dc752f0aab8419c46fe2803a0e7059c1515e5df48def11b9d517bbbf1fb2dddc`.

`Scripts/stage-splash-runtime.py --destination /path/beside/afm/splash-runtime`
downloads that exact precompiled archive, validates its checksum, extracts it
safely, validates the native binary/metallib hashes, and records the pin beside
`release.json`. It never runs the engine or downloads models. It rejects a
different existing runtime rather than replacing arbitrary directories.

The entire upstream release is retained, including its license and bundled
Python, to support applications that also expose the upstream CLI. The native
provider only executes `engine/splash`; `AFMSplashRuntime` exposes the upstream
Python and launcher paths for an application's CLI adapter. Applications own
packaging and command dispatch; this package owns the provider, release identity,
and staging implementation.

## Provider use

```swift
import AFMKitCore
import AFMKitSplash

let registry = AFMProviderRegistry()
try registry.register(AFMSplashProviderFactory())
let model = try registry.makeModel(
    providerID: "splash", modelID: "my-splash-model",
    configuration: .init(values: [
        "modelPath": .string("/path/to/existing/splash-package"),
        "runtimePath": .string("/path/to/splash-runtime"),
        "maxContext": .integer(32768)
    ]))
// model.load/respond/streamResponse start native inference; do not call these
// while another workload needs exclusive GPU access.
```

Model packages must already contain sibling `target/`, `draft/`, and `tokenizer/`
directories and upstream manifests. There are no implicit model downloads.
Without `runtimePath`, the provider finds `splash-runtime` beside the executable
or under `../libexec/afm/` (local install) or `../libexec/` (Homebrew).

The initial adapter supports text chat, incremental streaming, temperature,
top-p, top-k (0...32; positive temperature requires 1...32 and defaults to 32),
seed, max output tokens, usage, and native speculative decoding. It serializes
requests, bounds startup/request time, detects framing/truncation/sequence errors,
and tears down a failed or cancelled engine before reuse. Cancellation sends the
native Cancel frame and terminates the child; a later request starts a clean
engine. `unload()` terminates the owned process.

Thinking is disabled in the chat template. Tools, reasoning, image inputs,
structured-output constraints, custom stop strings, logprobs, and unsupported
sampling penalties are explicitly rejected. The full upstream CLI remains the
path for those features. Live tokenizer/model parity and GPU inference have not
been qualified by the CPU fixture suite.

## CPU-only validation

```sh
AFM_SWIFTPM_WRAPPER=/path/to/maclocal-api/Scripts/swiftpm-reliable.sh \
  Scripts/test-splash-cpu.sh
```

The script creates an isolated package containing the actual Core/Splash sources
and Tokenizers only. Tests exercise the registry, v5 wire encoding, fragmented
frames, repeated requests, streaming, errors, timeouts, cancellation, and release
staging. An inert Python peer stands in for the engine; it imports only the
standard library, uses fixed token IDs, and never accesses the GPU.

Sources: [release 1.0](https://github.com/incoai/splash/releases/tag/1.0),
[protocol](https://github.com/incoai/splash/blob/1.0/runtime/engine/Protocol.hpp),
[package builder](https://github.com/incoai/splash/blob/1.0/dev/tools/package.py).
