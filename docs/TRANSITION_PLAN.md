# AFMKit Transition Plan

Last updated: 2026-08-21

Extraction baseline: `maclocal-api` commit `2581e82410f50c2427a6a11c19344738856850e9`.

Current synchronization baseline: `maclocal-api` consumer-migration commit `a197c3d`, including
the Qwen 3.8 tool-call and automatic MTP-sidecar runtime changes, the server-owned AFMKit serving
facade, removal of the tracked AFMKit shadow targets, and clean Release packaging validation.

## Objective

Build AFMKit as a standard Swift provider layer for maclocal-api, Vesta, and third-party macOS apps. The SDK should let developers write providers against stable contracts while maclocal-api continues its release cadence as the full CLI/server distribution.

The long-term shape should align with macOS 27 LanguageModel-style provider concepts: provider identity, availability, model descriptors, streaming events, reasoning, tool calls, multimodal capability descriptors, cancellation, quotas, privacy boundaries, and unload behavior. macOS 27-specific features must be additive; macOS 26-compatible consumers should still build against the core contracts.

## Repository Strategy

AFMKit starts as a private repository while the API is still moving. It can become public after the core contract and first provider packages are stable enough that external adopters are not forced to track maclocal-api internals.

Fourteen public modules use three tagged SwiftPM package boundaries for normal
consumers. This is package-level isolation: the root package has no global
provider dependency graph.

All manifests use Swift tools 6.1 and declare macOS 26. Xcode 26 / Swift 6.3
exposes the source-compatible Core, OpenAI, DwarfStar, and MLX products. Xcode 27
/ Swift 6.4 additionally exposes the Apple and FoundationModelsMLX products that
import macOS 27 Foundation Models APIs. A two-toolchain CI matrix verifies both
manifest surfaces and clean-builds the root Xcode 26 surface.

| Published package | Public modules | Dependency strategy |
| --- | --- | --- |
| `AFMKit` | Core, OpenAI compatibility, inference, provider-free evaluation contracts, Apple provider, four independently selectable Apple services, and the `AFMKitServices` umbrella | No SwiftPM dependencies; `AFMEvalKit` depends only on OpenAI compatibility, and Apple frameworks are isolated to selected service/provider targets. |
| `AFMKitDwarfStar` | `AFMKitDwarfStar` | Exact AFMKit release, exact public Hub/Xet graph, vanilla DwarfStar plus AFM-owned adapter code. |
| `AFMKitMLX` | `AFMKitMLX`, `AFMKitFoundationModelsMLX` | Exact AFMKit release and exact AFM-compatible MLX/private dependency graph. |
| `maclocal-api` | Host executable modules | Aggregates selected AFMKit packages and owns CLI, HTTP server, WebUI, and packaging. |
| `Vesta` | Host app modules | Selects AFMKit packages directly where possible and maclocal-api only for process/server isolation. |

The development monorepo keeps the provider packages under `Packages/`. Release
automation materializes and tags separate provider repositories. Existing
product/module imports remain source-compatible; migration changes only SwiftPM
package declarations and each `.product(..., package:)` owner.

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

Status: extracted, verified, and consumed by maclocal-api and Vesta.

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
- maclocal-api imports the package from the immutable AFMKit revision used by the completed
  consumer-migration checkpoint.

## Phase 3: Apple and macOS 27 Provider Surface

Status: extracted and verified in this repository. The common AFM provider adapter is implemented,
and Vesta consumes the typed AFMKit provider event surface directly.

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

- AFMKit validates the current host's strict non-ad-hoc code signature and Boolean PCC entitlement,
  and also accepts an injected entitlement check for deterministic tests and specialized hosts.
- Vesta continues to own provisioning profiles, entitlement files, provider selection UI, route metadata, and chat workflow DTOs.
- Vesta's native request factory and AFM27 chat reducer consume AFMKit runtime events while those
  app concerns remain outside the SDK.

Current `AFMKitApple` checkpoint:

- The provider-neutral downstream quickstart builds in Release and exposes MLX, Apple on-device,
  and Apple PCC routes through the same registry/model contract.
- A live on-device quickstart smoke test loaded Apple Intelligence, generated the requested exact
  response, emitted usage, and completed normally.
- An unsigned or ad-hoc-signed PCC quickstart exits cleanly with the missing managed-entitlement
  reason instead of crashing or masking it as a locale failure. Live PCC generation remains a
  signed-host integration test because the entitlement belongs to the consuming app's code
  signature. Deterministic tests cover valid, invalid, ad-hoc, missing, and non-Boolean host states.
- The normalized `AFMKitApple` symbol graph includes the intentional provider, model, managed
  capability, and configuration-key additions with no removed public symbols.
- Release suites cover Core, OpenAI compatibility, Apple, the four independently selectable
  Apple services plus their umbrella, MLX, Inference, FoundationModelsMLX, DwarfStar, and EvalKit. All fourteen public API
  baselines and fresh split-package downstream builds are release gates.

## Phase 4: Runtime Adapters

Status: extracted and Release-verified. `AFMKitMLX` passes through both the tagged remote dependency graph and the persistent local compatibility stack. `AFMKitDwarfStar` compiles against a pinned, unmodified DwarfStar submodule.

Scope:

- `AFMKitInference`: provide the dependency-free high-level load/respond/stream/batch facade over
  `AnyAFMModel`. It depends only on Core and OpenAI-compatible DTOs; provider selection and runtime
  configuration stay in provider packages or a host compatibility shim. Streaming preserves
  append/replace actions, full usage, tool lifecycle, cancellation, and provider errors.

- `AFMKitMLX`: expose model loading, download progress, reasoning, tool calling, streaming, cancellation, prefix-cache/concurrency capability metadata, and multimodal descriptors without requiring consumers to depend on the full AFM server.
- `AFMKitDwarfStar`: expose the same AFMKit provider contract on top of vanilla DwarfStar. Interface-level adapter patches are acceptable; underlying engine patches should remain upstream or documented as limitations.

Resolved transition blockers:

- Legacy maclocal-api releases used `Scripts/apply-mlx-patches.sh` and
  `Scripts/apply-mlx-cpp-patches.sh`. The consumer-migration branch disables those mutation steps
  by default; its clean Release and packaging gates resolve the normal build through immutable
  AFM-compatible packages.
- Clean downstream apps selecting `AFMKitMLX` inherit the tagged AFM-compatible MLX graph; apps selecting `AFMKitDwarfStar` inherit the vanilla DwarfStar engine. Neither provider product inherits Vapor.
- The downstream quickstart and clean maclocal-api checkout both resolve the immutable
  AFM-compatible MLX graph without source mutation or local package-path overrides.

Current `AFMKitMLX` checkpoint:

- `AFMMLXModel` is the stable public runtime facade. In addition to `AFMModel`, it implements
  `AFMMLXOpenAIChatServing`, which exposes OpenAI-compatible generation and streaming, request
  admission, batch lifecycle, serving policy, and request profiling without exposing the concrete
  MLX service implementation.
- Model loading, download progress, reasoning, tool calls, structured output, streaming, cancellation, prefix-cache configuration, concurrency configuration, MTP, multimodal model selection, and runtime telemetry flow through the AFMKit provider contract.
- Engine implementation types remain package-scoped inside `AFMKitMLX`; Swift `package` access does
  not cross into maclocal-api. Provider-owned scheduling, cache, and generation tests therefore move
  with the implementation into AFMKit, while maclocal-api consumes the public serving facade and
  retains HTTP routing, Prometheus metric exposition, files, and cancellable request tasks.
- The full package Release test suite passes against the local AFM-compatible dependency stack.
- A clean downstream copy resolves the published compatibility tags and passes the full Release test suite without path overrides.
- Qwen 3.8 MTP heads resolve from checkpoint architecture and quantization metadata rather than
  repository names. A host can override the sidecar with `mtpModelID`; direct files require a
  sibling quantization configuration.
- MTP generators are bound to the loaded model/container identity, so retries and model switches
  cannot reuse a stale process-global generator. Disabled MTP prefetches a matching head without
  delaying base-model startup; explicit MTP resolves synchronously and fails closed.
- The provider-level MTP policy, resource resolution, local path resolution, and configuration
  propagation are covered by focused Release tests in this repository. HTTP `tool_choice` and
  response-finalization policy remains in maclocal-api because it is a server concern.
- Provider-level runtime telemetry is currently exposed through runtime snapshots and counters
  owned by `AFMKitMLX`. This is intentionally not part of `AFMKitCore`, and it must not grow
  HTTP, Prometheus, vLLM, GuideLLM, Vapor, or WebUI presentation responsibilities. Before a public
  AFMKit 1.0 tag, this surface should either be renamed as provider runtime observations or moved
  behind a small provider-neutral telemetry event contract.
- `AFMKitMLX` owns its committed `default.metallib`, canonical SwiftPM bundle lookup, and
  `Scripts/rebuild-mlx-metallib.sh`. Consumer builds package the immutable provider bundle; only
  AFMKit maintainers rebuild the Metal artifact.

Current `AFMKitDwarfStar` checkpoint:

- The preferred facade is `AFMDwarfStarProviderFactory`, `AFMDwarfStarModel`, and
  `AFMDwarfStarRuntimeConfiguration`; checkpoint, Hub, and projection utilities
  bring the checked-in baseline to 42 normalized public symbols.
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
| `mlx-swift-lm` compatibility branch | `/Volumes/edata/dev/git/CODEX/AFMKit-dependencies/mlx-swift-lm-afm` | `0.31.6-afm.3` | `e0d7fa7` | AFM model architectures, parsers, generation behavior, and quantization-aware Qwen MTP loading |

The compatibility repositories are AFM-owned distribution dependencies. Changes are published as
immutable AFM tags after their own Release build/test gate; AFMKit pins exact tags. They do not
require pull requests against upstream MLX repositories. The consumer migration has proved that
the tagged graph covers the clean maclocal-api Release build and packaging paths.

During the atomic maclocal-api consumer migration, AFMKit supports two build-only path overrides:

- `AFMKIT_MLX_SWIFT_PATH` selects a persistent checkout of AFM's MLX fork. The patched
  `mlx-swift-lm` manifest consumes this same environment variable, ensuring that both dependency
  chains resolve one package path and identity.
- `AFMKIT_MLX_SWIFT_LM_PATH` selects maclocal-api's persistent patched `vendor/mlx-swift-lm`
  checkout. AFMKit derives the SwiftPM package identity from the directory name so product lookup
  remains correct for either the legacy checkout or the tagged compatibility package.

These switches are not runtime configuration and do not change the normal
published `AFMKitMLX` graph. The monorepo provider manifest separately uses
`AFMKIT_PUBLIC_PATH` only to locate the root AFMKit package during development;
release materialization always emits an exact HTTPS AFMKit dependency.

## Phase 5: Consumer Migration

Status: implementation-complete at the pre-tag checkpoint. A provider-neutral quickstart compiles
against the public registry and runtime adapters, and exercises Apple on-device generation without
importing maclocal-api server targets. Vesta consumes AFMKit provider events directly. The
maclocal-api migration branch consumes AFMKit products from an immutable revision, has removed its
tracked shadow copies of `AFMKitCore`, `AFMKitMLX`, and `AFMKitDwarfStar`, and passes clean Release,
direct-install, relocated-tarball, WebUI, and provider-resource packaging validation.

Current maclocal-api consumer-migration checkpoint:

- Branch `codex/afmkit-consumer-migration` imports `AFMKitCore`, `AFMOpenAICompat`, `AFMKitMLX`,
  and `AFMKitDwarfStar` from AFMKit. It keeps OpenAI HTTP routing, Vapor request lifecycle,
  response files, cancellable request tasks, dashboard rendering, and
  Prometheus/vLLM/GuideLLM-compatible exposition in maclocal-api.
- `AFMServer` owns the `AFMChatServing` facade and server transport DTOs. It converts typed
  provider events into OpenAI-compatible HTTP responses instead of reaching into MLX internals or
  parsing provider-specific raw tool-call text in the controller.
- The focused Release migration gate passed for streaming controllers, batch dispatch,
  reasoning propagation, MLX provider facade behavior, and concurrent batch behavior.
- AFMKit now owns MLX and DwarfStar provider resources and maintenance scripts. The consumer branch
  resolves and packages the AFMKit SwiftPM bundles instead of rebuilding or copying local shadow
  targets. Its pre-existing dirty `vendor/mlx-swift-lm` checkout remains a legacy worktree artifact,
  not an input to the normal immutable dependency path.
- The clean migration checkout builds `afm` in Release as `v0.9.17-a6e358c`, packages both
  `AFMKit_AFMKitMLX.bundle` and `AFMKit_AFMKitDwarfStar.bundle`, verifies the MLX Metal library and
  WebUI, and passes a relocated tarball smoke test from persistent external storage.
- The Vesta consumer resolves AFMKit revision `dfeab23`, builds unsigned Release with Xcode 27
  Beta 3, embeds the AFMKit MLX Metal resource bundle, and passes 1,003 Release unit tests with
  coverage enabled (six environment-dependent tests skipped, zero failures).
- maclocal-api's Python distribution gate builds an Apple-Silicon wheel with the explicit
  `py3-none-macosx_26_0_arm64` compatibility tag rather than incorrectly advertising the native
  AFM executable and provider resources as a platform-independent wheel.

Scope:

- Make maclocal-api consume AFMKit packages from tags.
- Make Vesta consume `AFMKitCore`, `AFMOpenAICompat`, and Apple/PCC packages directly.
- Keep high-throughput server, WebUI, Homebrew, PyPI, release harness, and model qualification in maclocal-api.
- Add a downstream example app that exercises AFMKit without local maclocal-api source checkout assumptions.
- Keep a standalone downstream build gate that materializes the same root,
  DwarfStar, and MLX manifests intended for publication, resolves each without a
  lock, compares provider pins to the qualified locks, and builds all fourteen public
  products.

Exit criteria:

- maclocal-api release builds from tagged AFMKit packages.
- Vesta release builds against the same tags.
- A new Swift app can use `AFMKitCore` and at least one provider package from a clean checkout.

Pre-tag exit status:

- The implementation and clean-checkout evidence for all three criteria are complete.
- Replacing immutable revision pins with the first AFMKit release tags, merging the consumer
  branches, and publishing release artifacts remain explicit maintainer-controlled release actions.
- A signed Vesta Release still requires the consuming app's Apple Development identity and managed
  PCC provisioning profile. The unsigned Xcode 27 Beta 3 Release build is the architecture gate;
  signing is deliberately not an AFMKit package concern.

## Compatibility Rules

- `AFMKitCore` must stay dependency-free.
- macOS 27 features must be compiler-gated at manifest evaluation and runtime-gated.
- Xcode 26 must expose and build the macOS 26 source-compatible package surface, including the
  Apple service products, without parsing macOS 27-only targets; Xcode 27 exposes the complete
  fourteen-product surface.
- Provider streams should emit structured events for text, reasoning, tool calls, structured output, generated media, usage, timing, progress, completion, and errors.
- Runtime-specific capabilities must be explicit. Swapping engines should not silently remove tool calling, reasoning, structured JSON, or multimodal support without advertising that limitation.
- Server and transport observability are not core-provider responsibilities. `AFMKitCore` must
  not define HTTP, Prometheus, vLLM, GuideLLM, Vapor, WebUI, or dashboard concepts. Provider
  packages may expose runtime observations; maclocal-api turns those observations into `/metrics`
  and benchmark-compatible server surfaces.

## Issue #192 Observability Boundary

The transition target for vLLM Playground and GuideLLM compatibility is:

- `AFMKitCore`: portable requests, responses, capabilities, generation events, usage, timing,
  cancellation, and provider lifecycle only.
- Runtime packages such as `AFMKitMLX` and `AFMKitDwarfStar`: provider-owned runtime observations
  such as queue depth, token counts, cache hits, prefill/decode timings, and engine-specific
  capability metadata. These must remain structured Swift data and avoid HTTP or Prometheus
  presentation details.
- `AFMOpenAICompat`: portable OpenAI DTOs and policy normalization when the behavior is genuinely
  model/provider-agnostic.
- `maclocal-api` / `AFMServer`: OpenAI HTTP routes, streaming framing, active HTTP connection
  lifecycle, `/metrics`, Prometheus text exposition, vLLM-compatible metric names, GuideLLM
  request semantics, WebUI state, release harnesses, and benchmark adapters.

Fields such as `ignore_eos`, `continuous_usage_stats`, benchmark concurrency knobs, and active
connection gauges are server/transport policy unless they can be reduced to a provider-neutral
generation control. If they remain vLLM-specific, they belong in maclocal-api compatibility layers,
not in `AFMKitCore`.

## Open Questions

- Whether AFMKit should own a small download abstraction or leave all Hugging Face/Xet transport in runtime adapters.
- Whether patched MLX functionality should live in an AFM-compatible fork, upstream MLX packages, or a narrow adapter package with generated patch application.
- Whether provider runtime telemetry should stay as per-provider implementation detail or become a
  small `AFMKitCore` observation protocol after the maclocal-api `/metrics` adapter is stable.
