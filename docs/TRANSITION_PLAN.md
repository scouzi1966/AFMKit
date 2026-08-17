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

Status: started in this repository.

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

Scope:

- Move OpenAI-compatible request/response DTOs, tool call payloads, JSON schema helpers, and error surface into a dependency-free package.
- Keep provider APIs event-based instead of string-only.
- Ensure tool-call parser tests are portable outside maclocal-api.

Exit criteria:

- maclocal-api imports `AFMOpenAICompat` from AFMKit instead of its local copy.
- Vesta can use the DTO package without linking Vapor or model runtimes.

## Phase 3: Apple and macOS 27 Provider Surface

Scope:

- Extract FoundationModels provider interfaces into `AFMKitApple`.
- Add macOS 27-only provider registration and availability types without making macOS 27 the package-wide minimum.
- Move reusable Private Cloud Compute capability, quota, and availability checks from Vesta into AFMKit where they are app-agnostic.
- Keep app-specific entitlements, profile IDs, signing, and UI selection logic in Vesta.

Known gap:

- Current PCC execution is primarily in Vesta. AFMKit has capability metadata but not the full reusable PCC request/session runtime.

## Phase 4: Runtime Adapters

Scope:

- `AFMKitMLX`: expose model loading, download progress, reasoning, tool calling, streaming, cancellation, prefix-cache/concurrency capability metadata, and multimodal descriptors without requiring consumers to depend on the full AFM server.
- `AFMKitDwarfStar`: expose the same AFMKit provider contract on top of vanilla DwarfStar. Interface-level adapter patches are acceptable; underlying engine patches should remain upstream or documented as limitations.

Known transition blockers:

- maclocal-api currently uses `Scripts/apply-mlx-patches.sh` to patch `vendor/mlx-swift-lm`.
- maclocal-api currently uses `Scripts/apply-mlx-cpp-patches.sh` for MLX C++/Metal support that can be lost on clean rebuild.
- Clean downstream apps cannot depend on the full AFMKit umbrella without inheriting the patched MLX/DwarfStar/Vapor graph.
- A clean showcase app found that focused FoundationModels-style usage is practical, while full MLX downstream use still exposes patched-runtime coupling.

## Phase 5: Consumer Migration

Scope:

- Make maclocal-api consume AFMKit packages from tags.
- Make Vesta consume `AFMKitCore`, `AFMOpenAICompat`, and Apple/PCC packages directly.
- Keep high-throughput server, WebUI, Homebrew, PyPI, release harness, and model qualification in maclocal-api.
- Add a downstream example app that exercises AFMKit without local maclocal-api source checkout assumptions.

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
