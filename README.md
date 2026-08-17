# AFMKit

AFMKit is the provider SDK being extracted from maclocal-api. The goal is to make the AFM provider contract usable by Swift applications without pulling in the full AFM server, patched MLX runtime, DwarfStar adapter, Vapor stack, or model download machinery.

The private development checkpoint contains dependency-free provider and OpenAI-compatible API surfaces, an Apple provider package, and extracted MLX and DwarfStar provider adapters. Both runtime adapters are usable through the stable AFMKit provider contract while their implementation details remain package-scoped.

## Current Products

| Product | Status | Dependencies |
| --- | --- | --- |
| `AFMKitCore` | Extracted | Swift Foundation only |
| `AFMOpenAICompat` | Extracted | Swift Foundation only |
| `AFMKitApple` | Extracted | Apple FoundationModels, on-device execution, and macOS 27 Private Cloud Compute |
| `AFMKitMLX` | Extracted and Release-verified | Tagged AFM-compatible MLX package stack |
| `AFMKitDwarfStar` | Extracted and Release-verified | Vanilla DwarfStar submodule plus AFM-owned Swift/C adapter |

## Build

```bash
git clone --recurse-submodules git@github.com:scouzi1966/AFMKit.git
cd AFMKit
swift test
Scripts/check-afmkit-core-api.sh
Scripts/check-afmkit-core-api.sh AFMOpenAICompat
Scripts/check-afmkit-core-api.sh AFMKitApple
Scripts/check-afmkit-core-api.sh AFMKitMLX
Scripts/check-afmkit-core-api.sh AFMKitDwarfStar
```

Normal consumers resolve the tagged AFM-compatible MLX stack directly:

```bash
swift test -c release
```

The standalone [MLX quickstart](Examples/AFMKitQuickstart/README.md) shows the downstream provider
registry and structured streaming path without linking the maclocal-api server, CLI, or WebUI:

```bash
Scripts/check-downstream-example.sh
```

When developing the compatibility packages themselves, their persistent local checkouts can replace the tagged dependencies:

```bash
AFMKIT_MLX_SWIFT_PATH=/Volumes/edata/dev/git/CODEX/AFMKit-dependencies/mlx-swift-afm \
AFMKIT_MLX_SWIFT_LM_PATH=/Volumes/edata/dev/git/CODEX/AFMKit-dependencies/mlx-swift-lm-afm \
swift test -c release
```

The normal `AFMKitMLX` API intentionally exposes only `AFMMLXProviderFactory`, `AFMMLXModel`, `AFMMLXKernelEngine`, and `AFMMLXRuntimeConfiguration`. Runtime services, cache policies, converters, scheduling internals, and UI-oriented selection policies are package-scoped so maclocal-api can migrate without making them permanent third-party API.

The normal `AFMKitDwarfStar` API exposes only `AFMDwarfStarProviderFactory`, `AFMDwarfStarModel`, and `AFMDwarfStarRuntimeConfiguration`. AFMKit pins an unmodified `antirez/ds4` submodule and compiles it behind an AFM-owned C bridge. Hub resolution, checkpoint selection, scheduling, prefix-cache policy, DSpark integration, DSML parsing, and runtime coordination remain package-scoped.

`AFMKitApple` keeps the package minimum at macOS 26 and runtime-gates macOS 27 APIs. Its reusable native runtime owns Apple model availability and quota snapshots, entitlement-first PCC validation, reasoning selection, and `LanguageModelSession` reuse. The host app still owns its signed entitlements, provisioning profile, provider UI, and app-specific chat/workflow DTOs.

## Transition Plan

The durable plan is tracked in [docs/TRANSITION_PLAN.md](docs/TRANSITION_PLAN.md). Update that document when scope changes so maclocal-api, Vesta, and AFMKit stay aligned.
