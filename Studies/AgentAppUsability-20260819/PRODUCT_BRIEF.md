# Product Brief: DecisionBrief

## Product idea

DecisionBrief is a focused macOS app that turns user-selected plain-text or Markdown meeting notes into a concise, grounded decision pre-read. A user opens one or more local notes, states the decision or meeting objective, and asks the local Qwen 3.8 model to produce:

1. a short situation summary;
2. decisions already supported by the notes;
3. unresolved questions;
4. risks or conflicting evidence; and
5. source references using the displayed source labels.

It is a working document-analysis tool, not a chat playground, SDK demonstration, or marketing screen.

## Target user

A product or engineering lead preparing for a decision meeting from scattered local `.txt`, `.md`, and `.markdown` notes. They value privacy, need a fast pre-read, and want claims traceable to the notes they selected.

## Primary workflow

1. Launch DecisionBrief and see an empty source list, an objective field, and a disabled Generate action.
2. Use the system open panel to select one or more readable plain-text or Markdown files.
3. Review source names and remove unwanted sources without changing the source files.
4. Enter a decision objective, such as `Choose the rollout sequence for the desktop beta`.
5. Start local analysis. The app visibly reports model download/load progress and then streams the brief.
6. Cancel an in-progress load or generation and return to an actionable state.
7. Read the generated brief and its source-label references. The user can select text using standard macOS text selection, but the app does not write an output file.

## AFMKit and model requirements

- Import and use `AFMKitCore` and `AFMKitMLX` directly through Swift Package Manager.
- Use `AFMProviderRegistry`, `AFMMLXProviderFactory`, and the public provider/model contract rather than MLX implementation internals.
- Default model ID: `mlx-community/Qwen3.8-27B-4bit`.
- Run inference locally through AFMKit. Network access is permitted only for AFMKit/model dependencies and the model download/cache managed by the runtime.
- Use Swift 6 and the selected Xcode 27 toolchain.
- Evaluate runtime performance only from a Release/non-debug build.

## Functional requirements

- Native SwiftUI macOS app in `Studies/AgentAppUsability-20260819/App`.
- Source import through `NSOpenPanel` or SwiftUI `fileImporter`, restricted to explicit user selection and supported text types.
- Read source bytes without modifying the selected file. Reject unreadable, binary/invalid-text, empty, or unreasonably large inputs with an actionable error.
- Assign stable, visible source labels and include those labels in the model prompt.
- Require at least one valid source and a non-empty objective before generation.
- Show states for empty, ready, model downloading/loading, generating, completed, cancelled, and failed.
- Stream response text as AFMKit emits it.
- Provide a working Cancel action during model loading and generation.
- Keep a loaded model available for another brief during the app session when practical; unload it when the owning service is released or explicitly reset.
- Prevent concurrent generation requests.
- Make the default model visible in a low-emphasis settings or status surface; no provider/model picker is required.
- Persist only app-owned preferences needed for the experience. Do not persist source contents or generated briefs across launches.
- Include accessibility labels/identifiers for primary controls and states needed by UI automation.

## Output contract

The generation prompt must tell Qwen 3.8 to use only the provided source text, distinguish supported facts from uncertainty, avoid fabricating missing details, and use the exact visible source labels in references. The rendered brief should request these headings:

- `Situation`
- `Supported decisions`
- `Open questions`
- `Risks and conflicts`
- `Recommended meeting focus`

## Acceptance criteria

1. A clean checkout resolves dependencies and builds the app in Release using the selected Xcode 27 toolchain.
2. Automated unit tests cover source validation/loading, prompt construction and grounding instructions, generation state transitions, cancellation behavior, and an injected model failure without downloading the real model.
3. A UI or integration smoke test launches the app and verifies its initial state and Generate gating.
4. The Release app launches as a normal macOS application.
5. With a small explicit fixture note selected, the app performs an observed AFMKit generation using `mlx-community/Qwen3.8-27B-4bit`; evidence records the prompt/fixture identity, visible load/generation states, non-empty generated output, completion, and any reported usage without claiming facts not observed.
6. Cancellation is exercised against a real or deterministic controlled generation path and leaves the UI usable.
7. An injected error is surfaced with a retry-capable state.
8. Static review finds no delete/overwrite APIs, arbitrary command execution, external-service mutation, account access, telemetry, or third-party data transmission in app code.
9. App code depends only on AFMKit public products/contracts and normal Apple frameworks.
10. Every changed path is reported by the coding sub-agent and independently reviewed.

## Measurable completion criteria

- Release build command exits `0` from a clean app build directory.
- Automated test command exits `0` with zero failures.
- Launch/smoke evidence identifies the built `.app` and records a successful process launch or UI-test launch.
- Live inference evidence contains at least one non-whitespace response event from Qwen 3.8 and a terminal completion event; otherwise the study explicitly records live inference as unverified.
- Cancellation evidence shows an active operation transitions to `cancelled` or `ready` within 5 seconds after Cancel in the tested path.
- Safety audit command/search and manual review produce zero destructive-capability findings, or the app is not accepted.

## Explicit non-goals

- Editing, renaming, moving, deleting, or overwriting selected files.
- Exporting or auto-saving briefs.
- Watching folders, indexing a home directory, reading files without explicit selection, or retaining security-scoped bookmarks.
- Shell command execution, code execution, plugins, tools, function calling, or agentic actions.
- Network search, remote inference, analytics, telemetry, crash upload, cloud sync, account sign-in, or external API access.
- Calendar, email, task, issue-tracker, or document-service integration.
- PDF, rich-text, image, audio, or web-page ingestion in Phase A.
- A general chat interface, conversation history, multi-model selection, or model parameter laboratory.
- Claims that the generated brief is authoritative; it is a preparation aid grounded in user-selected notes.

