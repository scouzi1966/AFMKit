# AFMKit

AFMKit is the provider SDK being extracted from maclocal-api. The goal is to make the AFM provider contract usable by Swift applications without pulling in the full AFM server, patched MLX runtime, DwarfStar adapter, Vapor stack, or model download machinery.

The first private checkpoint contains dependency-free provider and OpenAI-compatible API surfaces that downstream apps can import without the full AFM server/runtime graph.

## Current Products

| Product | Status | Dependencies |
| --- | --- | --- |
| `AFMKitCore` | Extracted | Swift Foundation only |
| `AFMOpenAICompat` | Extracted | Swift Foundation only |
| `AFMKitApple` | Planned | Apple FoundationModels and macOS 27 provider surfaces |
| `AFMKitMLX` | Planned | AFM-compatible MLX package path |
| `AFMKitDwarfStar` | Planned | Vanilla DwarfStar plus AFM-owned adapter |

## Build

```bash
swift test
Scripts/check-afmkit-core-api.sh
Scripts/check-afmkit-core-api.sh AFMOpenAICompat
```

## Transition Plan

The durable plan is tracked in [docs/TRANSITION_PLAN.md](docs/TRANSITION_PLAN.md). Update that document when scope changes so maclocal-api, Vesta, and AFMKit stay aligned.
