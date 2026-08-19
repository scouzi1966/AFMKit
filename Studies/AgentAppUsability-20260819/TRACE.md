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

Pending. The process invocation, emitted session/thread identity, questions/answers, reports, commands, failures, and retries will be appended after the process starts.
