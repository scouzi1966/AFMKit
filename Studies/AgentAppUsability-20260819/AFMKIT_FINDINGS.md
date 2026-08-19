# AFMKit Findings

## Scope

These findings come from one agent-built macOS product against AFMKit commit `7bd110144a80d4b4c44f2714e398b494116bddd2`. DecisionBrief used the public provider registry and AFMKitMLX with `mlx-community/Qwen3.8-27B-4bit`. Claims below distinguish observed evidence from recommendations; one study does not establish ecosystem-wide frequency.

## Validation result

Phase A produced a signed Release app, 10 passing app-owned tests, a 655 ms observed cancellation transition, and a successful live Qwen generation with 303 input, 431 output, and 0 reasoning tokens. The generated brief was visibly non-empty, used all required sections and source labels, and was grounded in the selected fixture. Production-source audit found no destructive or external-service capabilities.

## Prioritized findings

### P1: Reasoning needs a typed request option

The first live run completed after about 39.6 seconds but exposed no response text under a 768-token cap. Discovering and applying the undocumented MLX `chatTemplateKwargs.enable_thinking` metadata convention required implementation-source inspection. With thinking disabled, the same app produced a useful brief in 20.7 seconds and reported 0 reasoning tokens.

Action: add optional `AFMGenerationOptions.reasoningEnabled`, map it in AFMKitMLX, preserve the metadata fallback, and make the typed value authoritative when both are supplied. This is implemented in Phase B with focused MLX and Apple-adapter tests plus a public API-baseline update.

### P1: Relative package identity must be explicit

The isolated coder's first resolve failed because SwiftPM derived `afmkit-agent-app-study` from the worktree directory while the quickstart requested products from package `AFMKit`. Hard-coding the worktree identity made the app non-portable.

Action: use `.package(name: "AFMKit", path: ...)` in relative-development manifests. The Phase B quickstart manifest now demonstrates this.

### P2: Source-checkout preflight should include submodules

An uninitialized `vendor/ds4` produced an invalid-resource warning on every pass, including apps that selected only core and MLX products. Initializing the baseline-pinned submodule removed the warning.

Action: put `git submodule update --init --recursive` in source-checkout onboarding. Do not infer that package decomposition is required from this warning alone.

### P2: A macOS host integration path is missing

The public quickstart taught a command-line generation loop. The coder consequently stopped at a SwiftPM executable and did not implement a normal app bundle, host lifecycle unload, or independently launchable UI artifact. Supervising work added these app-owned pieces.

Action: add a small macOS-host guide covering bundle construction or Xcode embedding, main-actor UI state, stream event actions, cancellation, retryable failure, and `unload()` during shutdown. A full template is not yet justified by one study.

### P2: Build onboarding is materially expensive

The minimal MLX app planned roughly 2,565 nodes. From a reset package state, resolve took 25 seconds, Release tests took 166 seconds, and the subsequent Release app build took 137 seconds on the study machine. Build logs also carried AFMKitMLX, MLX Metal, and explicit-module warnings.

Action: publish expected first-build cost and cache guidance. Measure other downstream apps and machines before changing package architecture.

## Residual observations

AFMKit's append/replace stream actions, terminal completion event, usage, cancellation, and unload contracts were sufficient for the product after the app handled them explicitly. The initial coder missed some of those semantics, so discoverability and host guidance are the issue supported by this study, not a need to redesign the events.
