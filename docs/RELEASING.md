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
- The two approved repository-relative dependencies under `vendor/MLX` are
  included in the tag and qualification artifact with their licenses and
  provenance. Every other dependency in `Package.swift` and `Package.resolved`
  is an exact, anonymously readable HTTPS pin.
- The root workflow token receives `contents: write` only in the final publish job.

## Pull request qualification

Public CI packages an allowlisted immutable source artifact from the exact PR
head, including the vendored MLX stack. The trusted `workflow_run` handler
verifies provenance, compares all root and nested package manifests plus the
lockfile byte-for-byte with trusted graph inputs, and compiles candidate sources
through sandboxed compiler wrappers. Candidate scripts and test executables do
not run in that job. Products, caches, and the downloaded artifact are destroyed
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
