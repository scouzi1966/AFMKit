# AFMKit

AFMKit is the provider SDK being extracted from maclocal-api. The goal is to make the AFM provider contract usable by Swift applications without pulling in the full AFM server, patched MLX runtime, DwarfStar adapter, Vapor stack, or model download machinery.

The first private checkpoint contains `AFMKitCore`: dependency-free provider identity, capability, availability, reasoning-stream, download-progress, and generation-loop contracts.

## Current Products

| Product | Status | Dependencies |
| --- | --- | --- |
| `AFMKitCore` | Extracted | Swift Foundation only |
| `AFMOpenAICompat` | Planned | Swift Foundation only |
| `AFMKitApple` | Planned | Apple FoundationModels and macOS 27 provider surfaces |
| `AFMKitMLX` | Planned | AFM-compatible MLX package path |
| `AFMKitDwarfStar` | Planned | Vanilla DwarfStar plus AFM-owned adapter |

## Build

```bash
swift test
Scripts/check-afmkit-core-api.sh
```

## Transition Plan

The durable plan is tracked in [docs/TRANSITION_PLAN.md](docs/TRANSITION_PLAN.md). Update that document when scope changes so maclocal-api, Vesta, and AFMKit stay aligned.
