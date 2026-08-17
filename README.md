# AFMKit

AFMKit is the provider SDK being extracted from maclocal-api. The goal is to make the AFM provider contract usable by Swift applications without pulling in the full AFM server, patched MLX runtime, DwarfStar adapter, Vapor stack, or model download machinery.

The private development checkpoint contains dependency-free provider and OpenAI-compatible API surfaces plus an Apple provider package that downstream apps can import without the full AFM server/runtime graph.

## Current Products

| Product | Status | Dependencies |
| --- | --- | --- |
| `AFMKitCore` | Extracted | Swift Foundation only |
| `AFMOpenAICompat` | Extracted | Swift Foundation only |
| `AFMKitApple` | Extracted | Apple FoundationModels, on-device execution, and macOS 27 Private Cloud Compute |
| `AFMKitMLX` | Planned | AFM-compatible MLX package path |
| `AFMKitDwarfStar` | Planned | Vanilla DwarfStar plus AFM-owned adapter |

## Build

```bash
swift test
Scripts/check-afmkit-core-api.sh
Scripts/check-afmkit-core-api.sh AFMOpenAICompat
Scripts/check-afmkit-core-api.sh AFMKitApple
```

`AFMKitApple` keeps the package minimum at macOS 26 and runtime-gates macOS 27 APIs. Its reusable native runtime owns Apple model availability and quota snapshots, entitlement-first PCC validation, reasoning selection, and `LanguageModelSession` reuse. The host app still owns its signed entitlements, provisioning profile, provider UI, and app-specific chat/workflow DTOs.

## Transition Plan

The durable plan is tracked in [docs/TRANSITION_PLAN.md](docs/TRANSITION_PLAN.md). Update that document when scope changes so maclocal-api, Vesta, and AFMKit stay aligned.
