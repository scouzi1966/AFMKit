# Workflow Recommendations

## Human-to-agent handoff

1. State the product workflow and safety boundary before naming framework APIs. The useful constraint in this study was "selected notes to grounded pre-read," not "build an AFMKit demo."
2. Give the coding agent public onboarding paths and acceptance criteria, but withhold implementation internals during the baseline phase. This exposed the package-identity and reasoning-control discovery costs.
3. Require a plan before edits and an exact final changed-path list. Preserve the prompt, process identity, questions, failures, retries, and report in a trace owned by the supervising agent.
4. Separate delegated implementation from independent product validation. The delegated agent's executable and deterministic tests were necessary but did not prove a normal app bundle, real cancellation, or live Qwen output.
5. Freeze and push the baseline app before changing the SDK. This keeps observed friction distinguishable from improvements made in response.

## AFMKit downstream checklist

1. Use an explicit package name for relative dependencies and initialize repository submodules.
2. Build and test in Release when evaluating MLX integration cost or runtime behavior.
3. Treat `.responseText(.replace, ...)` as replacement, require a terminal `.completed`, surface usage/finish reason, and make an empty terminal response retryable.
4. Give every load/generation operation an identity so cancelled tasks cannot update later UI state.
5. Set `reasoningEnabled` deliberately when a thinking model shares a bounded response budget; leave it `nil` only when the provider default is intended.
6. Pair automated tests with a signed-bundle launch, real-model cancellation, and one observed non-empty live generation before claiming end-to-end success.
7. Audit product runtime sources separately from tests and build scripts. Model-cache writes are an AFMKit operational exception, not permission for the app to mutate user inputs.

## Maintainer follow-up

Prioritize the typed reasoning option and explicit relative-package example now. Add source-checkout and macOS-host guidance next. Track warning cleanup and dependency/build-shape work separately, with broader measurement before architectural changes.
