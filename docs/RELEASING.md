# Releasing AFMKit

AFMKit publishes one qualified version from one repository and one root SwiftPM
manifest. No provider mirror repositories or coordinated secondary tags exist.

## Published product set

The single `AFMKit` tag exposes all fourteen public products:

- Core: `AFMKitCore`, `AFMOpenAICompat`, `AFMKitInference`, `AFMEvalKit`
- Apple: `AFMKitApple`, `AFMKitEmbeddings`, `AFMKitSpeech`,
  `AFMKitSpeechSynthesis`, `AFMKitVision`, `AFMKitServices`
- MLX: `AFMKitMLX`, `AFMKitFoundationModelsMLX`
- DwarfStar: `AFMKitDwarfStar`, `AFMKitFoundationModelsDwarfStar`

Provider source remains organized under `Packages/`, but those directories are
ordinary targets in the root manifest. They are not separately versioned packages.

## External prerequisites

- Hosted Xcode 26 and Xcode 27 runners verify the two compiler product matrices.
- The qualification runner uses the exact Xcode, SDK, Swift version, and compiler
  digest in `docs/api-baselines/toolchain.env`.
- Every exact dependency in `Package.swift` and `Package.resolved` is anonymously
  readable before a public release. Qualification may use
  `AFMKIT_DEPENDENCY_TOKEN` during the private-development phase, but the
  unauthenticated consumer gate prevents publishing a graph that requires it.
- The root workflow token receives `contents: write` only in the final publish job.

## Pull request qualification

Public CI packages an allowlisted immutable source artifact from the exact PR
head. The trusted `workflow_run` handler verifies provenance, compares the
candidate `Package.swift` and `Package.resolved` byte-for-byte with trusted graph
inputs, prebuilds dependencies, and compiles candidate sources through sandboxed
compiler wrappers. Candidate scripts and test executables never run with private
credentials. Private sources, products, caches, and credentials are destroyed
after qualification.

## Release sequence

1. Run **Request release** on `main` with a strict SwiftPM-compatible tag such as
   `v0.1.0` or `v0.1.0-rc.1`.
2. The request records its run ID, immutable default-branch SHA, and tag.
3. `Scripts/validate-release.sh` validates the root manifest and lock, all
   fourteen API baselines, Release tests, workflow/security regressions, and a
   fresh downstream build of all fourteen products from one staged root tag.
4. The publish job records an immutable publication-intent ref at the qualified
   commit, then creates the AFMKit tag and GitHub release.

Publication is idempotent. An existing intent, tag, or release must resolve to
the exact qualified commit and matching release state. Conflicts fail closed.
The SemVer tag is never created before qualification succeeds.
