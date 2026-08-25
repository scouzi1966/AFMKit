# Runtime Flows

## MLX audio model and synthesis flow

```mermaid
sequenceDiagram
    participant Host as Host UI
    participant Store as AFMMLXAudioModelStore
    participant Shared as AFMMLXModelStore / Hub cache
    participant Runtime as AFMMLXAudioRuntime
    participant Audio as Vendored mlx-audio

    Host->>Store: isDownloaded(modelID)
    alt user approves missing model
        Host->>Store: download(modelID, progress)
        Store->>Shared: downloadTTSModelPackage
        Shared-->>Store: shared snapshot reference
    end
    Host->>Runtime: load(modelID)
    Runtime->>Store: runtimeLocation(modelID)
    Store-->>Runtime: repository ID + exact cache root
    Runtime->>Audio: load from existing snapshot
    Host->>Runtime: synthesize or stream(request)
    Runtime-->>Host: samples, metrics, completion
    Host->>Runtime: cancel or unload
```

The host owns the consent prompt, selected model, and presentation state.
AFMKit owns discovery, download patterns and progress, cache resolution,
import-reference staging, deletion, model loading, generation, cancellation,
telemetry, and cache release. Loading defaults to local-only and must not perform
an unapproved second download.

## Provider-neutral generation

```mermaid
sequenceDiagram
    actor User
    participant App as Host app
    participant Registry as AFMProviderRegistry
    participant Factory as Provider factory
    participant Model as AnyAFMModel
    participant Engine as Provider runtime

    User->>App: Submit request
    App->>Registry: makeModel(providerID, modelID, configuration)
    Registry->>Factory: makeModel(...)
    Factory-->>Registry: AnyAFMModel
    App->>Model: availability()
    Model-->>App: availability + reason
    App->>Model: load(progress)
    Model->>Engine: resolve and initialize
    Engine-->>App: progress + descriptor
    App->>Model: streamResponse(AFMRequest)
    loop Structured stream
        Model-->>App: AFMGenerationEvent
    end
    App->>Model: unload()
```

**Figure 1 — Provider-neutral lifecycle.** The application can render progress,
availability, and typed events without knowing the concrete engine. Model reuse
and unload policy are application lifecycle decisions.

## Structured streaming and tools

```mermaid
sequenceDiagram
    participant App
    participant AFM as AFMModel
    participant LLM as Provider runtime
    participant Tool as Host-owned tool executor

    App->>AFM: streamResponse(messages + tool definitions)
    AFM->>LLM: Provider-native request
    LLM-->>AFM: reasoning/text deltas
    AFM-->>App: reasoningText / responseText
    LLM-->>AFM: tool call name + arguments
    AFM-->>App: toolCall(started/delta/completed)
    App->>Tool: Authorize and execute
    Tool-->>App: Tool result
    App->>AFM: Follow-up AFMRequest with result/history
    AFM-->>App: response events + usage + completed
```

**Figure 2 — Tool calling is a host-controlled loop.** AFMKit describes tools
and emits calls; it does not grant permissions or execute arbitrary application
actions. The host validates arguments, obtains consent, executes, and records the
result.

## Apple on-device and PCC routes

```mermaid
sequenceDiagram
    participant App
    participant Apple as AFMKitApple
    participant Probe as Capability/entitlement probe
    participant Device as SystemLanguageModel
    participant PCC as PrivateCloudComputeLanguageModel

    App->>Apple: select model ID + configuration
    Apple->>Probe: availability, locale, entitlement, quota
    alt apple.system.default
        Probe-->>App: on-device availability
        Apple->>Device: LanguageModelSession request
        Device-->>Apple: response stream
    else apple.private-cloud-compute
        Probe-->>App: PCC availability/privacy/network metadata
        Apple->>PCC: LanguageModelSession request
        PCC-->>Apple: response stream
    end
    Apple-->>App: normalized AFMGenerationEvent stream
```

**Figure 3 — Explicit Apple route selection.** On-device and PCC share an app
contract but have different privacy/network characteristics. AFMKit does not
silently send an on-device request to PCC; the app selects the model and must
possess the managed entitlement for PCC.

## macOS 27 MLX `LanguageModel` bridge

```mermaid
sequenceDiagram
    participant AppleHost as Foundation Models client
    participant Model as MLXLanguageModel
    participant Exec as MLXLanguageModelExecutor
    participant Adapter as Transcript/event adapters
    participant AFM as AFMMLXModel

    AppleHost->>Model: capabilities + executorConfiguration
    AppleHost->>Exec: init(configuration)
    AppleHost->>Exec: prewarm(model, transcript)
    Exec->>AFM: load/prime asynchronously
    AppleHost->>Exec: respond(request, channel)
    Exec->>Adapter: convert transcript, tools, schema, options
    Adapter->>AFM: streamResponse(AFMRequest)
    AFM-->>Adapter: AFMGenerationEvent stream
    Adapter-->>AppleHost: LanguageModelExecutorGenerationChannel updates
```

**Figure 4 — MLX as an Apple Foundation Models provider.** The bridge preserves
the Apple executor lifecycle while reusing AFMKit’s MLX model. Unsupported schema
or tool capabilities fail explicitly instead of being silently ignored.

## Cancellation and failure behavior

1. The app owns the Swift `Task` consuming the stream and cancels it in response
   to UI/service cancellation.
2. Provider adapters stop engine generation and terminate their stream promptly.
3. A thrown error ends the stream; a normal terminal state emits `.completed`.
4. Partial text/reasoning already emitted remains application-visible; the app
   decides whether to persist or annotate it.
5. Provider load failures map to `AFMError` categories while retaining useful
   provider details.
6. Tool calls not completed before cancellation must not be executed implicitly.

## State and cache ownership

| State | Owner | Lifetime |
| --- | --- | --- |
| Conversation history | Host app | User/app policy. |
| Provider registry | Host process | Usually application lifetime. |
| Loaded weights | Provider model/runtime | From load until unload/eviction. |
| KV/recurrent/prefix cache | Provider runtime | Provider policy; observable only through neutral snapshots. |
| Apple `LanguageModelSession` | Apple adapter | Session/request policy. |
| Tool side effects | Host tool implementation | Host transaction and audit policy. |
| Downloaded checkpoints | Provider asset resolver / host-selected cache | Filesystem/cache policy. |
