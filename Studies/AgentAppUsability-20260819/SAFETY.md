# DecisionBrief Safety Contract

## Boundary

DecisionBrief is a local, read-only document analysis app. Its only user-data input is a set of plain-text or Markdown files explicitly selected in a system file picker for the current app session. It reads those files to construct a local inference prompt and does not alter them.

AFMKit may download and cache the configured Qwen model and its dependencies. Those model/runtime cache writes are the sole non-app-data write exception allowed by this study.

## Allowed capabilities

- Present a user-mediated system open panel.
- Read the bytes of files the user explicitly selected for this session.
- Decode and validate supported text inputs.
- Run `mlx-community/Qwen3.8-27B-4bit` locally through AFMKit.
- Display streamed model output and transient model/runtime status.
- Persist app-owned settings such as window state or a non-sensitive preference in the app container/UserDefaults.
- Write ordinary Apple/Xcode build products and test artifacts during development.

## Prohibited capabilities

- Delete, truncate, replace, edit, rename, move, or change metadata or permissions of a user file.
- Create an output beside an input or export a generated brief.
- Read a directory recursively, watch a folder, retain security-scoped bookmarks, or reopen prior sources automatically.
- Invoke a shell, process, script, AppleScript, Shortcuts action, privileged helper, XPC action service, or arbitrary executable.
- Make application-controlled network requests, use remote inference, submit telemetry, or send source/model output to a third party. AFMKit-managed model download/cache traffic is the documented exception.
- Sign in to, read from, or mutate an account or external service.
- Use model tool calls or allow generated text to trigger an action.
- Store selected source contents, prompts containing source contents, or generated briefs beyond the running app session.

## Data flow

`Explicit file picker selection -> in-memory validated text -> in-memory AFMKit prompt -> local Qwen generation -> on-screen transient brief`

No application feature branches from generated text into an action. Closing the app discards loaded source text, objective text, and generated output. AFMKit/model caches can remain according to AFMKit's normal cache policy.

## Failure behavior

- Unsupported, unreadable, empty, oversized, or invalid-text files are rejected without mutation.
- A model load/generation failure is displayed locally and leaves sources available for retry.
- Cancel requests stop the owned asynchronous task, suppress later stale events, and return the interface to a usable non-running state.
- The app must not fall back to a remote provider when local generation fails.

## Verification

Acceptance requires tests for validation, failure, and cancellation plus a source audit for file-writing APIs, process execution, outbound networking, account frameworks, telemetry, and external mutations. Any discovered destructive or exfiltrating path blocks acceptance until removed.

