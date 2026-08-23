# Provider Contracts and Configuration

Provider interchangeability applies to the neutral lifecycle and event contract;
it does not mean every provider discovers, downloads, validates, caches, or
schedules models identically. This document makes those differences explicit.

## Stability tiers

| Tier | Intended consumer | Compatibility treatment |
| --- | --- | --- |
| **Tier 1: neutral contract** | All apps and third-party providers | `AFMKitCore` source API is the most stable boundary and requires API-baseline review. |
| **Tier 2: provider facade** | Apps intentionally choosing Apple, MLX, or DwarfStar | Factory/model/runtime-configuration facades are supported but can evolve with provider capabilities. |
| **Tier 3: advanced provider API** | Server hosts and engine specialists | Public scheduling, cache, converter, model-service, and profiling APIs have narrower compatibility expectations and should be isolated by consumer adapters. |
| **Tier 4: implementation/SPI** | AFMKit maintainers | `package`, `internal`, C/C++/Objective-C bridges, Metal resources, and third-party engine interfaces are not consumer contracts. |
| **macOS 27 bridge** | macOS 27 custom-provider adopters | `AFMKitFoundationModelsMLX` is implemented, tested, and protected by a checked-in public API baseline. Apple beta SDK changes still require explicit compatibility review. |

The current Swift visibility is broader than the intended architecture in parts
of `AFMKitMLX`: `AFMMLXModel.service` and `AFMMLXRuntime.service` expose
`MLXModelService`, and the module exports many engine-specific symbols. This is
known architectural debt, not evidence that all those types belong in Tier 1.

## Provider behavior matrix

| Behavior | Apple | MLX | DwarfStar |
| --- | --- | --- | --- |
| Discovery | Returns on-device and PCC descriptors. | Revalidates local/model registry through `MLXModelService`. | `modelDescriptors()` currently returns an empty list; host supplies a path/model ID. |
| Pre-load availability | Probes live model, locale, entitlement, and PCC quota. | Currently reports available before load; asset/download/load errors can still occur later. | Checks local path existence. |
| Asset download | Apple-managed. | Provider resolver/Hub path. | Local path or provider checkpoint resolver/Hub path. |
| Capability source | Apple native capability snapshot plus registered tools. | Model metadata/catalog plus provider heuristics/configuration. | Fixed engine capability set adjusted by configuration. |
| Privacy | On-device or PCC, explicitly identified. | Device execution; network may be used to obtain assets. | Device execution; network may be used to obtain assets. |
| Session/cache | Apple `LanguageModelSession`; reusable helpers also exist. | MLX scheduler and KV/recurrent/prefix-cache policies. | DwarfStar runtime/cache policy. |
| Token IDs | Not universally exposed. | Optional `AFMTextTokenizing`. | Not part of the preferred facade today. |
| Concurrency | Apple framework policy. | Provider configuration and scheduler. | `maxConcurrent`, subject to engine support. |

An app must not treat `.available` as “all assets loaded and generation is
guaranteed.” It means the provider currently sees no preflight blocker under its
documented semantics. `load()` remains the authoritative preparation step.

## Core capability semantics

`AFMModelCapabilities` includes vocabulary for text, vision, audio input/output,
reasoning, tools, structured output, streaming, embeddings, speculative decoding,
and prefix caching. The executable `AFMModel` contract currently provides general
respond/stream operations; typed embeddings, generated audio/media, progress, and
in-stream error protocols are not all defined as dedicated core operations/events.

Therefore:

- A provider must advertise only capabilities that its current request/event
  mapping can actually execute.
- `audioInput`, `audioOutput`, and `embeddings` are reserved vocabulary until the
  corresponding capability-specific protocol is adopted or a documented content
  mapping exists.
- A thrown stream error is the current error channel; there is no
  `AFMGenerationEvent.error` case.
- Load progress is a `load(progress:)` callback; there is no general generation
  progress event.
- Generated media should use a documented typed content/result extension before
  providers advertise it as a portable capability.

## Typed configuration versus dictionary configuration

The most robust provider-specific entry point is a typed runtime configuration:

- `AFMMLXRuntimeConfiguration`
- `AFMDwarfStarRuntimeConfiguration`
- Apple provider constants and typed reasoning values

`AFMProviderConfiguration` remains the registry-friendly, string-keyed transport.
Its keys form a compatibility surface and require documentation, validation, and
stable precedence.

### Apple configuration

| Key | Type/default | Meaning |
| --- | --- | --- |
| `systemPrompt` | String / empty | Provider session instructions. |
| `reasoningLevel` | Provider-defined enum/string / provider default | Requested Apple reasoning level when supported. |

PCC entitlement is not configuration. It must exist in the signed host process.
Executable Foundation Models tools are passed to the factory initializer rather
than encoded as dictionary values.

### MLX configuration

| Key | Type/default | Meaning |
| --- | --- | --- |
| `kvBits` | Integer / model default | KV-cache quantization. |
| `enablePrefixCaching` | Boolean / `true` | Enable provider prefix reuse where compatible. |
| `mtpEnabled`, `mtpDepth`, `mtpModelID` | Boolean/integer/string | Multi-token prediction settings. |
| `eagle3DrafterPath` | String / none | Optional speculative drafter. |
| `maxConcurrent` | Integer / automatic-or-serial provider policy | Provider concurrency limit. |
| `toolCallParser` | String / auto | Explicit tool parser selection. |
| `enableGrammarConstraints` | Boolean / `false` | Enable grammar-guided generation. |
| `prefillStepSize` | Integer / engine default | Prefill chunk policy. |
| `kvEvictionPolicy` | String / `none` | Provider cache eviction policy. |
| `fixToolArguments` | Boolean / `false` | Provider compatibility repair for tool arguments. |
| `forceVLM` | Boolean / `false` | Force VLM loading path. |
| `cacheProfilePath` | String / none | Provider cache profile asset. |
| `trace` | Boolean / `false` | Enable diagnostic tracing. |
| `gpuCapturePath`, `gpuTraceDuration` | String/integer / none | GPU capture controls. |
| `gpuProfile`, `gpuProfileBandwidth` | Boolean / `false` | GPU profiling controls. |
| `mlxKernels` or `kernelEngine` | String / `native` | Kernel engine selector. `mlxKernels` currently has precedence. |
| `forceDisableThinking` or `noThinking` | Boolean / `false` | Disable model thinking. `forceDisableThinking` currently has precedence. |

The factory descriptor does not yet advertise every accepted alias/key. Typed key
constants and a single generated schema are recommended follow-up work.

### DwarfStar configuration

| Key | Type/default | Meaning |
| --- | --- | --- |
| `modelPath` | String / model ID | Local GGUF/checkpoint path. |
| `contextWindow` | Integer / 32,768 | Maximum context. |
| `prefillChunk` | Integer / `0` | Engine prefill chunk policy. |
| `powerPercent` | Integer / `100` | Engine power/performance setting. |
| `dsparkSupportPath` | String / none | DSpark support asset. |
| `dsparkDraftTokens` | Integer / `5`, clamped 1...16 | Speculative draft count. |
| `dsparkConfidenceThreshold` | Number / `0.7`, clamped 0...1 | Acceptance threshold. |
| `dsparkStrict` | Boolean / `false` | Strict speculative behavior. |
| `enablePrefixCaching` | Boolean / `false` | Provider prefix-cache request. |
| `maxConcurrent` | Integer / `1` | Provider concurrency limit. |

## Apple API layering

`AFMKitApple` has three distinct integration levels:

1. **Preferred neutral provider:** `AFMFoundationProviderFactory` and
   `AFMFoundationModel` through `AFMKitCore`.
2. **Advanced session toolkit:** native capability probes, dynamic profiles,
   reusable session/runtime helpers, and usage projections for applications that
   deliberately use Apple semantics.
3. **Legacy/specialized service:** `FoundationModelService`, which exposes a
   separate stateful service model and should not be the default new-app entry
   point.

Applications should choose one lifecycle model per feature rather than mixing
stateless neutral requests and a separately managed legacy session accidentally.

## Configuration evolution rules

1. Add a typed property first, then map a documented dictionary key.
2. Validate type and range at model construction; do not silently coerce invalid
   values where behavior or privacy changes.
3. Alias precedence is explicit and aliases are deprecated before removal.
4. Provider descriptors enumerate accepted public keys from the same source of
   truth used by decoding.
5. Defaults are recorded in tests and documentation.
6. Configuration affecting Apple executor identity participates in its hashable
   executor configuration.
7. Unknown keys are either rejected in strict mode or surfaced diagnostically;
   they are never assumed to have taken effect.
