# AFMKit Transition Plan

Last updated: 2026-08-17

Source baseline: `maclocal-api` commit `2581e82410f50c2427a6a11c19344738856850e9`.

## Objective

Build AFMKit as a standard Swift provider layer for maclocal-api, Vesta, and third-party macOS apps. The SDK should let developers write providers against stable contracts while maclocal-api continues its release cadence as the full CLI/server distribution.

The long-term shape should align with macOS 27 LanguageModel-style provider concepts: provider identity, availability, model descriptors, streaming events, reasoning, tool calls, multimodal capability descriptors, cancellation, quotas, privacy boundaries, and unload behavior. macOS 27-specific features must be additive; macOS 26-compatible consumers should still build against the core contracts.

## Repository Strategy

AFMKit starts as a private repository while the API is still moving. It can become public after the core contract and first provider packages are stable enough that external adopters are not forced to track maclocal-api internals.

Use tagged SwiftPM packages for normal consumers:

| Package | Purpose | Dependency Strategy |
| --- | --- | --- |
| `AFMKitCore` | Stable provider contracts and stream events | No external dependencies |
| `AFMOpenAICompat` | OpenAI-compatible request/response DTOs and schema helpers | No external dependencies |
| `AFMKitApple` | FoundationModels and macOS 27 provider bridge | Apple frameworks only; no maclocal-api server dependency |
| `AFMKitMLX` | MLX provider adapter | Depends on an AFM-compatible MLX package or an isolated compatibility fork |
| `AFMKitDwarfStar` | DwarfStar provider adapter | Depends on vanilla DwarfStar plus AFM-owned adapter code |
| `maclocal-api` | CLI, OpenAI-compatible HTTP server, WebUI, packaging | Aggregates AFMKit packages and server/runtime services |
| `Vesta` | Desktop app | Uses AFMKit providers directly where possible and maclocal-api only where process/server isolation is wanted |

## Phase 1: Extract `AFMKitCore`

Status: extracted and verified in this repository.

Scope:

- Copy dependency-free core types from maclocal-api.
- Preserve the public symbol graph baseline.
- Preserve unit tests for reasoning parsing, reasoning stream reduction, download progress metadata, and generation-loop policy.
- Keep macOS deployment target at macOS 26.

Exit criteria:

- `swift test` passes.
- `Scripts/check-afmkit-core-api.sh` passes.
- The package has no external dependency graph.

## Phase 2: Extract `AFMOpenAICompat`

Status: extracted and verified in this repository; downstream adoption is intentionally deferred to Phase 5.

Scope:

- Move OpenAI-compatible request/response DTOs, tool call payloads, JSON schema helpers, and error surface into a dependency-free package.
- Keep provider APIs event-based instead of string-only.
- Ensure tool-call parser tests are portable outside maclocal-api.

Initial extraction:

- `OpenAIRequest.swift`
- `OpenAIResponse.swift`
- `OpenAIResponseFormatPolicy.swift`
- Focused tests for response-format policy and reasoning-effort request decoding.

Deferred:

- Tool-call parser runtime tests remain in maclocal-api until parser code is split out of the MLX/server targets.
- Nullable Jinja schema regression tests remain with the runtime package because they require `swift-jinja`.

Exit criteria:

- maclocal-api imports `AFMOpenAICompat` from AFMKit instead of its local copy.
- Vesta can use the DTO package without linking Vapor or model runtimes.

Current checkpoint:

- Public API has a normalized Swift symbol-graph baseline.
- Unit tests pass without server or inference dependencies.
- maclocal-api adoption remains deferred until the Apple/provider packages are extracted, so its release cadence is not tied to a partially split package graph.

## Phase 3: Apple and macOS 27 Provider Surface

Status: extracted and verified in this repository; the common AFM provider adapter is implemented,
and downstream Vesta adoption remains in Phase 5.

Scope:

- Extract FoundationModels provider interfaces into `AFMKitApple`.
- Add macOS 27-only provider registration and availability types without making macOS 27 the package-wide minimum.
- Move reusable Private Cloud Compute capability, quota, and availability checks from Vesta into AFMKit where they are app-agnostic.
- Keep app-specific entitlements, profile IDs, signing, and UI selection logic in Vesta.

Extracted runtime boundary:

- `AFMFoundationNativeProviderProbe` exposes typed on-device and PCC availability, locale, context, and quota snapshots.
- `AFMFoundationNativeExecutionPlanner` validates the signed-app entitlement before quota access and carries PCC reasoning selection.
- `AFMFoundationNativeSessionRuntime` creates and reuses on-device and PCC `LanguageModelSession` instances without importing Vesta types.
- Foundation Models stream processing, telemetry, tool snapshots, structured completion, transcript windows, and generation option policies are reusable package APIs.
- `AFMFoundationProviderFactory` registers stable on-device and PCC model identifiers through the
  same `AFMProviderFactory`/`AFMModel` contract used by MLX and DwarfStar.
- Common requests use a fresh native session and render the complete AFM conversation exactly once;
  lower-level reusable sessions remain available for stateful interactive consumers.
- Native host tools are supplied as executable `FoundationModels.Tool` values and are advertised
  only when a matching implementation exists.
- Public API changes are guarded by the `AFMKitApple` normalized symbol-graph baseline.

Deliberate app boundary:

- The host app detects entitlements from its own signed binary and supplies the result to AFMKit.
- Vesta continues to own provisioning profiles, entitlement files, provider selection UI, route metadata, and chat workflow DTOs.
- Vesta's native request factory can migrate to the AFMKit runtime in Phase 5 without copying those app concerns into the SDK.

## Phase 4: Runtime Adapters

Status: extracted and Release-verified. `AFMKitMLX` passes through both the tagged remote dependency graph and the persistent local compatibility stack. `AFMKitDwarfStar` compiles against a pinned, unmodified DwarfStar submodule.

Scope:

- `AFMKitMLX`: expose model loading, download progress, reasoning, tool calling, streaming, cancellation, prefix-cache/concurrency capability metadata, and multimodal descriptors without requiring consumers to depend on the full AFM server.
- `AFMKitDwarfStar`: expose the same AFMKit provider contract on top of vanilla DwarfStar. Interface-level adapter patches are acceptable; underlying engine patches should remain upstream or documented as limitations.

Known transition blockers:

- maclocal-api currently uses `Scripts/apply-mlx-patches.sh` to patch `vendor/mlx-swift-lm`.
- maclocal-api currently uses `Scripts/apply-mlx-cpp-patches.sh` for MLX C++/Metal support that can be lost on clean rebuild.
- Clean downstream apps selecting `AFMKitMLX` inherit the tagged AFM-compatible MLX graph; apps selecting `AFMKitDwarfStar` inherit the vanilla DwarfStar engine. Neither provider product inherits Vapor.
- A clean showcase app found that focused FoundationModels-style usage is practical, while full MLX downstream use still exposes patched-runtime coupling.

Current `AFMKitMLX` checkpoint:

- The normal public API is limited to `AFMMLXProviderFactory`, `AFMMLXModel`, `AFMMLXKernelEngine`, and `AFMMLXRuntimeConfiguration` (46 normalized public symbols).
- Model loading, download progress, reasoning, tool calls, structured output, streaming, cancellation, prefix-cache configuration, concurrency configuration, MTP, multimodal model selection, and runtime telemetry flow through the AFMKit provider contract.
- Server implementation types remain package-scoped. This lets maclocal-api reuse them during Phase 5 without freezing cache, scheduler, converter, and UI policy details as community API.
- The full package Release test suite passes against the local AFM-compatible dependency stack.
- A clean downstream copy resolves the published compatibility tags and passes the full Release test suite without path overrides.

Current `AFMKitDwarfStar` checkpoint:

- The public API is limited to `AFMDwarfStarProviderFactory`, `AFMDwarfStarModel`, and `AFMDwarfStarRuntimeConfiguration` (25 normalized public symbols).
- The engine is the unmodified `antirez/ds4` submodule pinned at `84cc882352757baf628a1776badf7cc54d584e28`; AFMKit owns the Swift provider, C bridge translation units, Hub/checkpoint resolver, scheduling policy, and runtime integration.
- Reasoning, DSML tool calls, DSpark generalized draft depth, prefix-cache identity, continuous-prefill scheduling, Metal resource validation, and resumable Hub downloads are covered by 40 focused Release tests.
- The full provider implementation remains package-scoped so the public contract does not freeze engine internals or force consumers to understand DwarfStar's C API.
- There are no source patches applied to `vendor/ds4`. Engine-level limitations remain documented or upstreamed; AFMKit changes are restricted to the interface and orchestration layer.

AFM-compatible dependency checkpoint:

| Repository | Local path | Published tag | Pinned revision | Purpose |
| --- | --- | --- | --- | --- |
| `mlx-afm` | `/Volumes/edata/dev/git/CODEX/AFMKit-dependencies/mlx-afm` | `0.31.6-afm.1` | `b9af157b016a470be1ca609531693b822d40f95f` | AFM-compatible MLX C++/Metal source |
| `mlx-c-afm` | `/Volumes/edata/dev/git/CODEX/AFMKit-dependencies/mlx-c-afm` | `0.31.6-afm.1` | `1692252c78e634a90ae09bd77a9f68929982b8a0` | C bridge pinning `mlx-afm` |
| `mlx-swift-afm` | `/Volumes/edata/dev/git/CODEX/AFMKit-dependencies/mlx-swift-afm` | `0.31.6-afm.1` | `6000b7b26b70be2713c74e9ec2adeb89be07b9e5` | Swift MLX bindings and bundled Metal runtime |
| `mlx-swift-lm` compatibility branch | `/Volumes/edata/dev/git/CODEX/AFMKit-dependencies/mlx-swift-lm-afm` | `0.31.6-afm.2` | `90303ec548cb5d0dd6da16b86c82dad03240bd44` | AFM model architectures, parsers, and generation behavior |

## Phase 5: Consumer Migration

Status: in progress. A standalone MLX quickstart now compiles against the public provider registry
and runtime adapter without importing maclocal-api server targets. Tagged-package migration for
maclocal-api and Vesta remains outstanding.

Scope:

- Make maclocal-api consume AFMKit packages from tags.
- Make Vesta consume `AFMKitCore`, `AFMOpenAICompat`, and Apple/PCC packages directly.
- Keep high-throughput server, WebUI, Homebrew, PyPI, release harness, and model qualification in maclocal-api.
- Add a downstream example app that exercises AFMKit without local maclocal-api source checkout assumptions.
- Keep a standalone downstream build gate that imports public AFMKit products only. During private
  pre-tag development it uses a relative package path; the first AFMKit tag replaces that path with
  the repository URL and becomes the clean-clone release gate.

Exit criteria:

- maclocal-api release builds from tagged AFMKit packages.
- Vesta release builds against the same tags.
- A new Swift app can use `AFMKitCore` and at least one provider package from a clean checkout.

## Compatibility Rules

- `AFMKitCore` must stay dependency-free.
- macOS 27 features must be opt-in and runtime-gated.
- macOS 26 builds must not import macOS 27-only symbols outside guarded provider packages.
- Provider streams should emit structured events for text, reasoning, tool calls, structured output, generated media, usage, timing, progress, completion, and errors.
- Runtime-specific capabilities must be explicit. Swapping engines should not silently remove tool calling, reasoning, structured JSON, or multimodal support without advertising that limitation.

## Open Questions

- Whether `AFMKitApple` should be one package with guarded files or two products: macOS 26 FoundationModels compatibility and macOS 27 advanced providers.
- Whether AFMKit should own a small download abstraction or leave all Hugging Face/Xet transport in runtime adapters.
- Whether patched MLX functionality should live in an AFM-compatible fork, upstream MLX packages, or a narrow adapter package with generated patch application.
