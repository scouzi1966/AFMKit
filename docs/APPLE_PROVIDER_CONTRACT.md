# AFMKit Apple Provider Contract

Last updated: 2026-08-21

## Purpose

`AFMKitApple` exposes Apple's macOS 27 Foundation Models implementations through the same
`AFMProviderFactory` and `AFMModel` contracts used by AFMKit's local providers. All manifests use
Swift tools 6.1 and keep the AFMKit deployment floor at macOS 26. The root manifest exposes this
product only when evaluated by Xcode 27 / Swift 6.4 or newer; the provider factory and model adapter
are available only when the host is running macOS 27 or newer. Xcode 26 consumers retain source
compatibility through `AFMKitCore` and `AFMOpenAICompat` without parsing macOS 27-only targets.

## Stable identity

- Provider: `apple.foundation-models`
- On-device model: `apple.system.default`
- Private Cloud Compute model: `apple.private-cloud-compute`

The factory lists both models. Availability is evaluated at runtime so clients can display why a
model cannot currently run instead of hiding it from discovery.

## Host-owned capabilities

AFMKit cannot grant a signed application an entitlement. By default, the provider first validates
the current host with Security.framework using strict, all-architectures code-signing checks. It
rejects invalid and ad-hoc signatures, then requires the signed entitlement
`com.apple.developer.private-cloud-compute` to be the Boolean value `true`. The host may inject a
different entitlement probe for deterministic tests or an unusual embedding environment. PCC is
unavailable until the signed process passes all checks, even when the Apple developer account has
received managed-capability approval.

Apple tools are executable Swift values, while `AFMToolDefinition` contains only a portable name and
JSON schema. The factory therefore accepts host-provided `FoundationModels.Tool` values. A model
created without matching executable tools does not advertise tool calling and rejects requests that
contain tool definitions. A model created with tools validates that every requested tool has a host
implementation before generation.

## Request semantics

`AFMRequest` is a complete, stateless conversation. Each call creates a fresh
`LanguageModelSession`, moves system messages into session instructions, and renders the remaining
conversation into one prompt. Reusing a stateful Apple session here would append a full history to
the session's existing transcript and duplicate prior turns. Interactive applications that want
stateful sessions may continue using `AFMFoundationNativeSessionRuntime` directly.

Configuration values understood by the common provider are:

- `systemPrompt`: additional host instructions prepended to request system messages.
- `reasoningLevel`: `automatic`, `light`, `moderate`, or `deep`; used only by PCC.

The entitlement is deliberately not a configuration value. Treating JSON configuration as proof of
a code-signing entitlement would create a false security boundary.

## Capabilities

| Capability | On device | PCC | Notes |
| --- | --- | --- | --- |
| Text | Yes | Yes | Streaming and single-response APIs map to AFM events. |
| Vision | Yes | Yes | File references and in-memory image data are attached to the prompt. |
| Reasoning | No | Yes | PCC supports light, moderate, and deep reasoning. |
| Tool calling | Conditional | Conditional | Advertised only when executable host tools are supplied. |
| Structured output | Yes | Yes | JSON Schema maps to Apple's constrained `GenerationSchema`. |
| Streaming | Yes | Yes | Cumulative Apple snapshots become append/replace AFM events. |

Unsupported AFM options are rejected rather than silently ignored. This includes grammars,
log-probabilities, seeds, repetition/presence penalties, top-k/min-p, and stop sequences until a
native implementation exists.

## Event mapping

- Response snapshots emit `responseText`.
- PCC transcript reasoning emits `reasoningText`.
- Native tool transcript entries emit `toolCall` events.
- Apple usage maps input, cached-input, output, and reasoning token counts to `AFMUsage`.
- Provider, model, and privacy-route identity are emitted as metadata.
- A terminal `completed` event is always emitted after successful generation.
- Cancellation terminates the native task and surfaces `CancellationError`.

## Consumer migration

Vesta can first replace its provider probing and execution helpers with this common factory while
retaining its signed entitlement, provider-selection UI, and app-specific route metadata. It may keep
the lower-level reusable-session API for its interactive chat path until it adopts stateless AFM
requests end to end.
