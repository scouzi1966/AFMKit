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

