# Trusted Private Qualification Harness

`PrivatePackage.swift` is loaded only from the immutable trusted base selected by
the authoritative `workflow_run` API record after its event, status, conclusion,
repository, workflow path, default base, and trusted SHA have been validated.
Candidate package manifests and dependency locks are included only as inert
inputs. Their direct dependencies and complete lock graph, including both
vendored MLX package manifests, must be byte-for-byte equal to the trusted
default-branch files before compilation begins.
Candidate scripts and plugins are excluded from the artifact and never run.

The harness compiles allowlisted candidate source, vendored dependencies, and
tests with fixed target types and exact dependency constraints. Trusted compiler
wrappers and the macOS sandbox deny candidate compiler processes access to
unrelated repository caches. No candidate-built executable or test bundle runs
on the trusted runner. Cleanup removes the entire isolated build root, including
compiled products, SwiftPM state, homes, and the downloaded artifact, even when
qualification fails.
