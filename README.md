# AFMKit

AFMKit is the provider SDK being extracted from maclocal-api. The goal is to make the AFM provider contract usable by Swift applications without pulling in the full AFM server, patched MLX runtime, DwarfStar adapter, Vapor stack, or model download machinery.

The private development checkpoint contains dependency-free provider and OpenAI-compatible API surfaces, an Apple provider package, and extracted MLX and DwarfStar provider adapters. Both runtime adapters are usable through the stable AFMKit provider contract. Some advanced MLX engine types are still public during extraction; they are provider-specific APIs, not part of the stable cross-provider boundary.

The canonical architecture description, interface boundaries, macOS 27 feature
map, dependency ledger, runtime diagrams, and decisions are in
[`Architecture/`](Architecture/README.md).

## Current Products

| Product | Status | Dependencies |
| --- | --- | --- |
| `AFMKitCore` | Extracted | Swift Foundation only |
| `AFMOpenAICompat` | Extracted | Swift Foundation only |
| `AFMKitApple` | Extracted | Apple FoundationModels, on-device execution, and macOS 27 Private Cloud Compute |
| `AFMKitMLX` | Extracted and Release-verified | Tagged AFM-compatible MLX package stack |
| `AFMKitFoundationModelsMLX` | Extracted, tested, and API-baselined | macOS 27 `LanguageModel` / `LanguageModelExecutor` bridge backed by AFMKitMLX |
| `AFMKitDwarfStar` | Extracted and Release-verified | Vanilla DwarfStar submodule plus AFM-owned Swift/C adapter |

## Build

```bash
git clone --recurse-submodules git@github.com:scouzi1966/AFMKit.git
cd AFMKit
swift test
Scripts/test-api-gate.sh
Scripts/check-api-baselines.sh
```

The checked-in symbol graphs are qualified only with Xcode 27 Beta 3 build
`27A5218g` and macOS SDK 27.0 build `26A5378i`, as recorded in
`docs/api-baselines/toolchain.env`. The checker fails with selection guidance
when the active toolchain differs. It always asks SwiftPM to build the requested
target; `AFMKIT_API_SKIP_BUILD` is rejected so a stale or unrelated
`.swiftmodule` cannot satisfy the gate.

Normal consumers resolve the tagged AFM-compatible MLX stack directly:

```bash
swift test -c release
```

Release qualification runs the API-gate regressions, the Core, OpenAICompat,
MLX, and FoundationModelsMLX API baselines, Release package tests, and the
downstream quickstart build:

```bash
Scripts/validate-release.sh
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

The normal `AFMKitMLX` entry point is `AFMMLXModel`. It implements the provider-neutral
`AFMModel` contract and the public `AFMMLXOpenAIChatServing` contract used by server and app hosts.
The serving contract covers OpenAI-compatible generation and streaming, request admission, batch
lifecycle, response-format and tool policy, and request profiling. Some concrete engine and
model-service APIs are still public for advanced consumers; applications should isolate them because
the intended stable facade is narrower than the module's complete public symbol graph.
maclocal-api therefore keeps HTTP routing, Prometheus exposure, files, and request orchestration
local while consuming a stable AFMKit model facade instead of importing MLX engine internals.

The normal `AFMKitDwarfStar` API exposes only `AFMDwarfStarProviderFactory`, `AFMDwarfStarModel`, and `AFMDwarfStarRuntimeConfiguration`. AFMKit pins an unmodified `antirez/ds4` submodule and compiles it behind an AFM-owned C bridge. Hub resolution, checkpoint selection, scheduling, prefix-cache policy, DSpark integration, DSML parsing, and runtime coordination remain package-scoped.

`AFMKitApple` keeps the package minimum at macOS 26 and runtime-gates macOS 27 APIs. Its reusable native runtime owns Apple model availability and quota snapshots, entitlement-first PCC validation, reasoning selection, and `LanguageModelSession` reuse. The host app still owns its signed entitlements, provisioning profile, provider UI, and app-specific chat/workflow DTOs.

## Transition Plan

The durable plan is tracked in [docs/TRANSITION_PLAN.md](docs/TRANSITION_PLAN.md). Update that document when scope changes so maclocal-api, Vesta, and AFMKit stay aligned.
