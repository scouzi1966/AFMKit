# Agent App Usability Study Trace

Study date: 2026-08-19  
Persistent worktree: `/Volumes/edata/dev/git/CODEX/vesta-mac/Worktrees/afmkit-agent-app-study`  
Branch: `codex/agent-app-usability-study-20260819`  
AFMKit Phase A baseline: `7bd110144a80d4b4c44f2714e398b494116bddd2`

This file records observed events only. Exact prompts and material reports are preserved verbatim. Large build output may be stored as a reproducible artifact and referenced here by path and command rather than pasted in full.

## T-001: Worktree and baseline verification

- Observed at: 2026-08-19T13:30Z (UTC; command completed before the exact timestamp capture in T-003)
- Command:

```bash
pwd && git status --short --branch && git rev-parse HEAD && git branch --show-current && git remote -v && git worktree list --porcelain && printf '\nTOP LEVEL\n' && find . -maxdepth 2 -type f | sort | sed -n '1,160p' && printf '\nCODEX\n' && command -v codex || true && codex --version 2>/dev/null || true
```

- Result: exit `0`. The designated worktree was clean, branch was `codex/agent-app-usability-study-20260819`, and HEAD was exactly `7bd110144a80d4b4c44f2714e398b494116bddd2`. `origin` was `https://github.com/scouzi1966/AFMKit.git`. Codex CLI identity was `codex-cli 0.146.0` at `/Users/syl/.local/bin/codex`.

## T-002: Public onboarding inspection

- Commands:

```bash
sed -n '1,260p' README.md
sed -n '1,240p' Package.swift
rg --files | sort
sed -n '1,200p' .gitignore
sed -n '1,260p' Examples/AFMKitQuickstart/README.md
sed -n '1,220p' Examples/AFMKitQuickstart/Package.swift
sed -n '1,320p' Examples/AFMKitQuickstart/Sources/AFMKitQuickstart/main.swift
```

- Result: exit `0`. The public quickstart documents the registry/provider path and model invocation. No AFMKit source implementation file was used to prepare the product brief or future coder handoff.

## T-003: Toolchain verification

- Observed at: 2026-08-19T13:32:22Z / 2026-08-19T09:32:22-0400
- Commands:

```bash
xcode-select -p
xcodebuild -version
swift --version
```

- Result: exit `0`.

```text
/Applications/Xcode-27.0.0-Beta.3.app/Contents/Developer
Xcode 27.0
Build version 27A5218g
swift-driver version: 1.168.4 Apple Swift version 6.4 (swiftlang-6.4.0.25.4 clang-2100.3.25.1)
Target: arm64-apple-macosx27.0.0
```

## T-004: Protocol artifact creation

- Observed at: after T-003 and before any app or SDK implementation.
- Created:
  - `Studies/AgentAppUsability-20260819/PRODUCT_BRIEF.md`
  - `Studies/AgentAppUsability-20260819/SAFETY.md`
  - `Studies/AgentAppUsability-20260819/DECISIONS.md`
  - `Studies/AgentAppUsability-20260819/FRICTION_LOG.md`
  - `Studies/AgentAppUsability-20260819/TRACE.md`
- Product decision: DecisionBrief, a read-only local decision pre-read app for explicit text/Markdown inputs.

## T-005: Coding sub-agent handoff

### Exact implementation handoff prompt

The text between `BEGIN EXACT PROMPT` and `END EXACT PROMPT` is the complete initial prompt that will be passed on standard input to a fresh, ephemeral `codex exec` process. The process has no inherited conversation context.

BEGIN EXACT PROMPT

You are the coding sub-agent for Phase A of an AFMKit agent-usability study. You have no inherited conversational context. Work only in this existing git worktree:

`/Volumes/edata/dev/git/CODEX/vesta-mac/Worktrees/afmkit-agent-app-study`

The target app directory is:

`/Volumes/edata/dev/git/CODEX/vesta-mac/Worktrees/afmkit-agent-app-study/Studies/AgentAppUsability-20260819/App`

Read only these product/onboarding materials before planning:

1. `Studies/AgentAppUsability-20260819/PRODUCT_BRIEF.md`
2. `Studies/AgentAppUsability-20260819/SAFETY.md`
3. `README.md`
4. `Examples/AFMKitQuickstart/README.md`
5. `Examples/AFMKitQuickstart/Package.swift`
6. `Examples/AFMKitQuickstart/Sources/AFMKitQuickstart/main.swift`

AFMKit is fixed for Phase A at the current baseline commit `7bd110144a80d4b4c44f2714e398b494116bddd2`. Do not edit AFMKit source, tests, root package files, public docs, API baselines, or anything outside the target app directory. Do not inspect AFMKit implementation source. Learn the integration only from the public materials listed above and compiler diagnostics. Do not preemptively work around or fix the SDK.

First, inspect the allowed materials and the target directory, then report a concise implementation and verification plan before making any implementation edit. After the plan, implement the complete DecisionBrief macOS app in the target directory without waiting for approval unless a missing credential/entitlement or irreversible decision truly blocks you.

Requirements:

- Use Swift 6 with the selected Xcode 27 toolchain.
- Build a native SwiftUI macOS app, not a CLI or marketing/demo screen.
- Import and use AFMKit directly through its public `AFMKitCore` and `AFMKitMLX` products and documented registry/provider contract.
- Default and use model ID `mlx-community/Qwen3.8-27B-4bit`.
- Follow `PRODUCT_BRIEF.md` and `SAFETY.md` exactly, including the read-only, no-export, local-only action boundary.
- Use no destructive file or remote capabilities, arbitrary process/shell execution, external-service mutation, account access, telemetry, or app-controlled network API. AFMKit-managed model download/cache writes are allowed.
- Add focused unit tests for source validation/loading, prompt construction, generation state transitions, cancellation, and injected failure without requiring a real model download.
- Add a UI test or deterministic integration smoke test that launches the app and verifies initial state plus Generate gating.
- Make a Release/non-debug build. Do not draw performance conclusions from Debug.
- Keep test seams app-owned and do not expose or depend on AFMKit internals.
- Do not run live 27B model inference during implementation unless it is quick and clearly feasible; the supervising agent owns final live-inference validation.
- Do not commit, push, or modify git configuration.

Acceptance criteria are the ten numbered criteria in `PRODUCT_BRIEF.md`, including clean dependency resolution, Release build, automated tests, normal app launch, cancellable/error states, direct public-contract AFMKit use, and safety auditability. Implement everything within your control and report any criterion that still requires supervising-agent validation.

During work, report material questions, compiler/build/test failures, retries, commands, and workarounds truthfully. In the final report include:

1. implementation summary;
2. exact build/test commands and outcomes;
3. observed AFMKit onboarding/API/documentation friction;
4. acceptance criteria status and remaining validation;
5. every changed or created path, one per line; and
6. your agent/session identity if available.

END EXACT PROMPT

### Process identity and execution

- Invocation:

```bash
awk '/^BEGIN EXACT PROMPT$/{capture=1; next} /^END EXACT PROMPT$/{capture=0} capture' Studies/AgentAppUsability-20260819/TRACE.md | codex exec --ephemeral --json --sandbox danger-full-access -c 'approval_policy="never"' -C /Volumes/edata/dev/git/CODEX/vesta-mac/Worktrees/afmkit-agent-app-study -
```

- Emitted identity: Codex thread/session `01a01a3c-6a15-7081-8ede-8f08e44474ce`, started by `codex-cli 0.146.0`. The sub-agent's final report said that no separate session ID was available; the supervising process captured the emitted `thread.started` identifier above.
- Context isolation: a fresh `--ephemeral` process received the exact prompt on standard input. It was not resumed from another session.
- Material questions/answers: none. The sub-agent did not ask a blocking or material product question.
- Initial target observation: `find` reported `Studies/AgentAppUsability-20260819/App: No such file or directory`.

### Plan returned before implementation

Verbatim:

> The specified target directory does not yet exist, so I’ll create it as the app boundary. The allowed materials establish a Swift 6/macOS 26+ SwiftUI app using `AFMProviderRegistry` + `AFMMLXProviderFactory`, `AFMModelID`, `AFMRequest`, and streamed `AFMGenerationEvent` handling; the app must remain session-local and read-only for user files.
>
> Implementation plan:
>
> 1. Create a local Swift package/app target under `Studies/AgentAppUsability-20260819/App` with only `AFMKitCore`, `AFMKitMLX`, Apple frameworks, and app-owned test seams.
> 2. Implement source validation/loading, stable labels, grounded prompt construction, provider/model lifecycle, streaming, cancellation, failure recovery, and observable SwiftUI state.
> 3. Build a native macOS UI with file importer, objective field, gated Generate/Cancel controls, visible model/status state, streamed brief headings, and accessibility identifiers.
> 4. Add unit tests for validation/loading, prompt grounding, state transitions, cancellation, and injected failure, plus a deterministic launch/gating smoke test.
> 5. Resolve/build/test in Release and run a safety-focused source audit. I will not run live 27B inference.

### Commands, failures, and retries

Commands are listed in emitted order with their observed status:

| Event | Status | Command | Material result |
| --- | --- | --- | --- |
| `item_1` | passed | allowed-material `sed` reads plus target `find` | Read the six permitted files; target did not exist. |
| `item_6` | failed | `swift package resolve && swift test -c release` | Unknown package `AFMKit`; valid local identity was `afmkit-agent-app-study`. |
| `item_9` | failed | `swift package resolve && swift test -c release` | Reached Release compilation; app fallback `??` expression could not coalesce two concrete service types. Existing warning: missing `../../vendor/ds4/metal`. |
| `item_13` | failed | `swift test -c release` | Tests attempted to assign to the view model's inaccessible `sources` setter. |
| `item_16` | failed | `swift test -c release` | Swift 6 region-isolation errors in the asynchronous test polling helper. |
| `item_18` | failed | `swift test -c release` | Build completed but command exited nonzero before a useful test summary in the sub-agent report. |
| `item_20` | failed | `swift test -c release --disable-swift-testing --verbose` | XCTest ran 5 tests with 2 failures: cancellation race expected `cancelled` but got `ready`, and prompt-label assertion failed. |
| `item_23` | passed | `swift test -c release --disable-swift-testing` | 5 tests, 0 failures after app-only fixes. |
| `item_25` | passed | `swift build -c release` | Release executable built. |
| `item_26` | failed | `timeout 5 .build/arm64-apple-macosx/release/DecisionBrief` | `timeout` utility was unavailable. |
| `item_28` | failed | bounded background launch at `.build/arm64-apple-macosx/release/DecisionBrief` | Path did not identify the Xcode 27 SwiftPM build product. |
| `item_29` | passed | bounded background launch at `.build/out/Products/Release/DecisionBrief` | Executable remained alive for 3 seconds and was terminated by the probe. |
| `item_30` | passed | static `rg` safety audit and path/status listing | No prohibited production-source match; test fixture writes/removes were present. |
| `item_33` | passed | `swift test -c release --disable-swift-testing` | 6 tests, 0 failures after adding the named integration-smoke test. |
| `item_34` | passed | app path and git status listing | Reported five created paths. |

The sub-agent made four implementation/test correction passes after the initial creation. It did not edit AFMKit source or any path outside `Studies/AgentAppUsability-20260819/App`.

### Final report returned by the coding sub-agent

Verbatim:

> Implemented DecisionBrief in the target directory.
>
> Summary:
> - Native SwiftUI macOS app with source picker, objective field, gated Generate, streaming output, progress, cancellation, retryable errors, and accessibility identifiers.
> - Direct public AFMKit use via `AFMKitCore`, `AFMKitMLX`, `AFMProviderRegistry`, and `AFMMLXProviderFactory`.
> - Default model: `mlx-community/Qwen3.8-27B-4bit`.
> - Added deterministic app-owned tests for loading, prompts, state transitions, cancellation, failure, and gating.
>
> Verification:
>
> - `swift package resolve` — passed; dependency graph resolved.
> - `swift test -c release --disable-swift-testing` — passed, 6 tests, 0 failures.
> - `swift build -c release` — passed.
> - Release executable launch probe — passed; remained running for 3 seconds.
> - Safety audit — no prohibited capabilities in app sources. Temporary fixture writes/deletes occur only in tests.
> - Live 27B inference — not run; requires supervising-agent validation.
> - Finder `.app` bundle packaging/signing — not performed; Release executable launch was verified.
>
> AFMKit friction:
> - Local package identity was `afmkit-agent-app-study`, not the public quickstart’s `AFMKit`; adjusted only the app manifest.
> - Existing dependency warning: missing `../../vendor/ds4/metal` resource.
> - Existing AFMKit/MLX deprecation warnings remain untouched.
>
> Acceptance status:
> - Criteria 1–4, 7–10 implemented or verified locally.
> - Criterion 5 requires live inference validation.
> - Criterion 6 has deterministic cancellation coverage; real-model cancellation remains for supervisor validation.
>
> Changed/created paths:
>
> `Studies/AgentAppUsability-20260819/App/Package.swift`
>
> `Studies/AgentAppUsability-20260819/App/Package.resolved`
>
> `Studies/AgentAppUsability-20260819/App/Sources/DecisionBrief/main.swift`
>
> `Studies/AgentAppUsability-20260819/App/Sources/DecisionBriefCore/DecisionBriefCore.swift`
>
> `Studies/AgentAppUsability-20260819/App/Tests/DecisionBriefCoreTests/DecisionBriefCoreTests.swift`
>
> Agent/session identity: Codex coding sub-agent; no separate session ID available.

## T-006: Supervising review opened

- The sub-agent process completed with exit `0`.
- Unverified claims retained as gaps: live Qwen generation, real-model cancellation, and a Finder/LaunchServices `.app` bundle.
- Initial independent findings:
  - The workaround that names package identity `afmkit-agent-app-study` is tied to this worktree directory and would fail in a normal checkout named `AFMKit`.
  - AFMKit `.responseText` actions are ignored, so a `.replace` event would be appended instead of replacing prior output.
  - Adding files in separate picker operations can duplicate `Source 1` labels.
  - A cancelled operation can receive a late load-progress callback that changes the visible state back to loading.
  - The produced artifact is a SwiftPM executable, not the required normal `.app` bundle.
- Disposition: correct these issues only in the app/study paths before independent clean validation. No Phase A AFMKit source change is permitted.

## T-007: First independent Release validation and live run

- Initialized the baseline-declared `vendor/ds4` submodule at `84cc882352757baf628a1776badf7cc54d584e28`. The prior invalid-resource warning disappeared.
- `swift test --package-path Studies/AgentAppUsability-20260819/App -c release --disable-swift-testing`: exit `0`, 9 tests, 0 failures after supervising fixes.
- `Studies/AgentAppUsability-20260819/App/Scripts/build-release-app.sh`: exit `0`; produced `.build/DecisionBrief.app` after a Release build reported 130.60 seconds.
- `codesign --verify --deep --strict --verbose=2 .../DecisionBrief.app`: exit `0`; bundle was valid on disk and satisfied its designated requirement.
- Computer Use launch of the exact bundle showed model `mlx-community/Qwen3.8-27B-4bit`, empty source list, objective field, and disabled Generate control.
- Selected committed fixture `Studies/AgentAppUsability-20260819/Evidence/decision-notes.txt` through the app's system open panel and entered objective `Choose a safe sequence and readiness gates for the desktop beta`.
- Real load-cancellation check: the app visibly entered `Loading model...`; a measured repeat observed `Cancelled | ready to retry` 655 ms after the Cancel click.
- First uninterrupted live run: the app visibly entered `Generating...` and then `Completed` about 39.6 seconds after generation start. No brief text or `briefOutput` accessibility element was present. This run does not satisfy live-inference acceptance because no non-whitespace response was observed.
- Public API inspection found no reasoning control in `AFMGenerationOptions`. AFMKitMLX implementation inspection found the existing `AFMRequest.metadata["chatTemplateKwargs"]["enable_thinking"]` convention.
- Decision: apply the existing metadata workaround in the Phase A app, record F-005, reject empty terminal output, and rerun live inference. No SDK source was changed in T-007.

## T-008: Successful live Qwen generation

- Rebuilt and relaunched the exact packaged app after adding the app-owned reasoning metadata workaround and terminal empty-response handling. An older running app instance was first found after rebuilding; it was quit through the normal app menu, absence of a running `DecisionBrief` process was confirmed, and the rebuilt bundle was launched.
- Re-selected `Studies/AgentAppUsability-20260819/Evidence/decision-notes.txt` through `NSOpenPanel` and entered the exact objective `Choose a safe sequence and readiness gates for the desktop beta`.
- Observed actual generation through AFMKit with model `mlx-community/Qwen3.8-27B-4bit`. The UI reached `Completed` 20.7 seconds after Generate.
- Observed terminal usage `stop | 303 input, 431 output, 0 reasoning tokens` and a non-empty `briefOutput` accessibility element.
- The generated brief contained all requested sections: `Situation`, `Supported decisions`, `Open questions`, `Risks and conflicts`, and `Recommended meeting focus`. It used exact `[Source 1 - decision-notes.txt]` labels and grounded its recommendations in the selected notes, including dogfood-before-beta ordering, the rollback-playbook gate, the missing crash-free threshold and cohort choice, and the September 8 versus September 10 conflict.
- Mechanical screenshot evidence: `Studies/AgentAppUsability-20260819/Evidence/live-qwen-completed.png`, 900 by 592 pixels, SHA-256 `3d43826be10f384ad3893e3fc1bb30b9438401bb679a4625a426017ce00e1e27`.
- The app was quit through its normal application menu after the run, and no running `DecisionBrief` application remained. This exercised the app's shutdown/unload path.
- This is the only run claimed as successful live inference. T-007's first run remains recorded as an empty-output failure.

## T-009: Reset-state Phase A validation and safety audit

- A first cleanup attempt using `rm -rf Studies/AgentAppUsability-20260819/App/.build` was rejected by the command-execution safety policy before execution. It changed no files. The retry used SwiftPM's supported reset operation:

```bash
swift package --package-path Studies/AgentAppUsability-20260819/App reset
swift package --package-path Studies/AgentAppUsability-20260819/App resolve
swift test --package-path Studies/AgentAppUsability-20260819/App -c release --disable-swift-testing
Studies/AgentAppUsability-20260819/App/Scripts/build-release-app.sh
codesign --verify --deep --strict Studies/AgentAppUsability-20260819/App/.build/DecisionBrief.app
```

- All commands exited `0`. Measured durations were 25 seconds for clean dependency resolution, 166 seconds for Release compilation plus 10 XCTest cases, and 137 seconds for the Release bundle pass.
- XCTest result: 10 tests, 0 failures. Covered source validation, grounded labels, unique labels, Generate gating, append/replace semantics, injected failure, stream-without-completion failure, empty-response failure, deterministic cancellation, late-progress suppression, and retry usability.
- The bundle was an arm64 Mach-O `APPL` with identifier `com.afmkit.study.decisionbrief`, minimum macOS 26, and a valid ad-hoc signature satisfying its designated requirement.
- Static production-source audit searched for file mutation, process/shell execution, network clients, scripting/automation, telemetry, account persistence, URL opening, and external-service APIs. No prohibited capability matched. The only file access is explicit user-selected security-scoped `Data(contentsOf:options:.mappedIfSafe)` reading with format and size validation. AFMKit model download/cache writes remain the documented exception.
- Tests create and remove only their own temporary fixtures. The app-owned packaging script replaces only its generated `.build/DecisionBrief.app`; it is developer tooling and is not linked into or callable by the product runtime.
- `git diff --check` passed. No AFMKit source, test, API-baseline, or public documentation path had been changed at this Phase A boundary.

## T-010: Phase B first focused test attempt

- Phase A was committed and pushed as `e72f6bc` before any AFMKit source or public documentation edit.
- Implemented the evidence-derived `AFMGenerationOptions.reasoningEnabled` draft, MLX mapping, Apple unsupported-option validation, focused tests, explicit quickstart package identity, and quickstart guidance.
- Command:

```bash
DEVELOPER_DIR=/Applications/Xcode-27.0.0-Beta.3.app/Contents/Developer \
swift test -c release --disable-swift-testing --filter AFMKitMLXProviderTests
```

- Result: failed during test compilation after planning about 2,941 nodes. Four new MLX test fixtures omitted the required `AFMRequest.messages` argument. No test executable ran.
- Retry decision: pass `messages: []` in those fixtures and rerun the same focused Release test before continuing to Apple, quickstart, or API-baseline validation.

## T-011: Phase B focused tests and API baseline

- Corrected the four test fixtures and reran with the Xcode 27 beta 3 developer directory selected.
- `swift test -c release --disable-swift-testing --filter AFMMLXProviderTests`: exit `0`; 41 tests, 0 failures. This included typed false/true mapping, typed-over-metadata precedence, and nil preserving legacy metadata.
- `swift test -c release --disable-swift-testing --filter FoundationProviderTests`: exit `0`; 10 tests, 0 failures. This included explicit rejection of unsupported `reasoningEnabled` by the Apple adapter.
- `swift build -c release --package-path Examples/AFMKitQuickstart`: exit `0`; the independently resolved quickstart built from the study worktree with its explicit relative package name.
- First API-check wrapper invoked `Scripts/check-afmkit-core-api.sh AFMKitCore`, then failed when it attempted to assign the checker exit code to zsh's read-only `status` parameter. The checker itself had completed and produced the expected intentional API mismatch; the wrapper changed no source or baseline.
- Structured symbol-identifier comparison found exactly two additions and one replacement: the new `reasoningEnabled` property, the initializer including that argument, and the prior initializer symbol removed. Relationship count increased from 517 to 518.
- Replaced `docs/api-baselines/AFMKitCore.symbols.json` with the normalized generated graph after that review.
- `Scripts/check-afmkit-core-api.sh AFMKitCore`: exit `0`; `AFMKitCore public API matches its checked-in baseline.`

## T-012: Phase B broad validation

Commands used Xcode 27 beta 3 through `DEVELOPER_DIR`:

```bash
swift test -c release --disable-swift-testing
swift test --package-path Studies/AgentAppUsability-20260819/App -c release --disable-swift-testing
Scripts/check-downstream-example.sh
```

- Full AFMKit Release suite: exit `0`; 275 tests across 6 test executables, 0 failures.
- DecisionBrief Release suite against the Phase B SDK: exit `0`; 10 tests, 0 failures. Recompilation reported about 2,565 planning nodes and took 145.35 seconds before tests.
- Downstream example check: exit `0`; incremental Release build completed and printed `AFMKit downstream example built successfully.`
- Known non-fatal output remained: CoreData XPC messages during tests, AFMKitMLX/MLX compile warnings, and Xcode 27 explicit-precompiled-module path warnings already classified in F-006/F-007.

## T-013: Typed-option app migration

- Committed and pushed the Phase B SDK/docs/findings checkpoint as `9a07381` before changing the app.
- Replaced DecisionBrief's Phase A `metadata["chatTemplateKwargs"]["enable_thinking"] = false` workaround with `AFMGenerationOptions(reasoningEnabled: false)`. No other app behavior or safety boundary changed.
- This migration is intentionally isolated from the SDK commit so the final history preserves both the baseline workaround and the downstream adoption step.
- `swift test --package-path Studies/AgentAppUsability-20260819/App -c release --disable-swift-testing`: exit `0`; 10 tests, 0 failures.
- `Studies/AgentAppUsability-20260819/App/Scripts/build-release-app.sh`: exit `0`; the Release product build reported 127.20 seconds and produced the app bundle.
- `codesign --verify --deep --strict --verbose=2 Studies/AgentAppUsability-20260819/App/.build/DecisionBrief.app`: exit `0`; valid on disk and satisfied its designated requirement.
