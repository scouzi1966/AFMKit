# Decision Log

## D-001: Build DecisionBrief

- Date: 2026-08-19
- Phase: Study setup
- Decision: Build a local decision pre-read app for explicitly selected text and Markdown notes.
- Rationale: The workflow is useful to product and engineering leads, exercises real multi-source prompt construction and streaming, and remains focused enough to test AFMKit integration rather than app breadth.
- Alternatives considered: a generic chat client was rejected as a framework-demo-shaped product; a release-note rewriter was rejected because its value depends on exporting/copying an artifact and offers weaker grounding/citation pressure.

## D-002: Make inputs session-only and read-only

- Date: 2026-08-19
- Phase: Study setup
- Decision: Do not retain source bookmarks, source contents, objectives, or generated briefs across launches, and do not implement export.
- Rationale: This creates a clear, testable non-destructive boundary while preserving the full primary workflow.

## D-003: Limit Phase A input formats

- Date: 2026-08-19
- Phase: Study setup
- Decision: Accept only `.txt`, `.md`, and `.markdown` text in Phase A.
- Rationale: PDF/RTF parsing would add unrelated framework and extraction uncertainty. Plain text is enough to exercise explicit file selection, validation, prompt grounding, and local generation.

## D-004: Use the public registry/provider path

- Date: 2026-08-19
- Phase: Study setup
- Decision: Require direct use of AFMKit's public `AFMProviderRegistry` plus `AFMMLXProviderFactory`, defaulting to `mlx-community/Qwen3.8-27B-4bit`.
- Rationale: This follows the documented downstream boundary and directly tests whether that contract supports a real macOS app.

## D-005: Separate app and SDK evidence checkpoints

- Date: 2026-08-19
- Phase: Study setup
- Decision: Complete, validate, commit, and push the Phase A app against commit `7bd1101` before making any AFMKit SDK or documentation improvement.
- Rationale: This preserves causal evidence and keeps app-only changes separate from Phase B SDK/docs changes.

## D-006: Reject worktree-specific package identity

- Date: 2026-08-19
- Phase: Phase A supervising review
- Decision: Name the relative SwiftPM dependency explicitly as `AFMKit` instead of naming products against `afmkit-agent-app-study`.
- Rationale: The sub-agent's workaround compiles only while the checkout directory has this study-specific name. A downstream sample must remain reproducible in a normal `AFMKit` checkout.

## D-007: Package a real app bundle without changing AFMKit

- Date: 2026-08-19
- Phase: Phase A supervising review
- Decision: Keep the SwiftPM app source layout but add an app-owned Release bundling script and `Info.plist` to produce an ad-hoc-signed `DecisionBrief.app`.
- Rationale: This closes the acceptance gap without altering the Phase A SDK baseline or adding a generated Xcode project to the study.

## D-008: Preserve AFMKit stream semantics at the app boundary

- Date: 2026-08-19
- Phase: Phase A supervising review
- Decision: Carry append-versus-replace semantics through the app-owned model event and reject a stream that ends without AFMKit's terminal completion event.
- Rationale: The initial implementation discarded public event semantics and could display incorrect or falsely completed output.

## D-009: Disable Qwen thinking for the focused pre-read

- Date: 2026-08-19
- Phase: Phase A live validation
- Decision: Use AFMKitMLX's existing `chatTemplateKwargs.enable_thinking = false` request metadata for Phase A, retain a 768-token response cap, and fail visibly if completion contains no response text.
- Rationale: The first observed live run reached completion with no usable response. DecisionBrief needs concise grounded output, not hidden reasoning, and the public core options do not expose a reasoning toggle at baseline.
