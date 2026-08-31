# Releasing AFMKit

AFMKit publishes one qualified version from one repository and one root SwiftPM
manifest. No provider mirror repositories or coordinated secondary tags exist.

## Published product set

The single `AFMKit` tag exposes all sixteen public products:

- Core: `AFMKitCore`, `AFMOpenAICompat`, `AFMKitInference`, `AFMEvalKit`
- Apple: `AFMKitApple`, `AFMKitEmbeddings`, `AFMKitSpeech`,
  `AFMKitSpeechSynthesis`, `AFMKitVision`, `AFMKitServices`
- MLX: `AFMKitMLX`, `AFMKitMLXAudio`, `AFMKitMLXImage`,
  `AFMKitFoundationModelsMLX`
- DwarfStar: `AFMKitDwarfStar`, `AFMKitFoundationModelsDwarfStar`

Provider source remains organized under `Packages/`, but those directories are
ordinary targets in the root manifest. They are not separately versioned packages.

## External prerequisites

- The local release machine provides the supported Xcode toolchains needed to
  verify the compiler product matrices. Optional hosted PR CI may provide the
  same checks, but it is not a release prerequisite.
- Local qualification uses the exact Xcode, SDK, Swift version, and compiler
  digest in `docs/api-baselines/toolchain.env`.
- The two approved repository-relative dependencies under `vendor/MLX` are
  included in the tag and qualification artifact with their licenses and
  provenance. Every other dependency in `Package.swift` and `Package.resolved`
  is an exact, anonymously readable HTTPS pin.
- The local operator has authenticated `gh` with permission to create repository
  refs and releases. Authentication comes from the `gh` credential store or
  environment; release commands never accept or print a token argument.

## Optional hosted pull request qualification

Public CI packages an allowlisted immutable source artifact from the exact PR
head, including the vendored MLX stack. The trusted `workflow_run` handler
verifies provenance, compares all root and nested package manifests plus the
lockfile byte-for-byte with trusted graph inputs, and compiles candidate sources
through sandboxed compiler wrappers. Candidate scripts and test executables do
not run in that job. Products, caches, and the downloaded artifact are destroyed
after qualification.

## Canonical local release sequence

Release publication does not depend on GitHub Actions. From a clean checkout at
the exact current GitHub default-branch commit, run:

```bash
node Scripts/release-local.js v0.1.0
```

The command displays the repository, tag, and exact commit and requires typing
`publish v0.1.0`. `--yes` is available only for an explicitly non-interactive
invocation, and `--repo OWNER/REPO` can pin repository discovery.

The command performs these operations in order:

1. Validate strict SwiftPM-compatible SemVer such as `v0.1.0` or
   `v0.1.0-rc.1`; leading zeroes and build metadata are rejected.
2. Require a clean worktree, clean initialized recursive submodules, and a full
   commit SHA. Read GitHub's default branch, publication-intent ref, and final
   tag without making a remote change. A new release must be the exact current
   default-branch commit.
3. `Scripts/validate-release.sh` validates the root manifest and lock, all
   fifteen API baselines, Release tests, workflow/security regressions, and a
   fresh downstream build of all sixteen products from one staged root tag.
4. Recheck the local commit and worktrees. Only then record the immutable
   publication-intent ref, create or recover the annotated AFMKit tag, and create
   or recover the GitHub release.
5. Re-read the peeled tag and release through the GitHub API and `gh release
   view`. The tag must resolve to the qualified commit; the release must be
   non-draft with the correct prerelease and latest state.

Publication is idempotent. An existing intent, tag, or release must resolve to
the exact qualified commit and matching release state. Conflicts fail closed.
The SemVer tag is never created before qualification succeeds. If publication
stops after the intent or tag is created, check out that same clean commit and
run the same command again. The matching immutable intent or tag permits this
recovery even if the default branch has advanced; without that state, stale
commits are rejected. Do not delete or force-update refs to recover.

## Optional GitHub Actions path

The request and release workflows may remain available as an optional remote
operator. They use the same `release-publication.js` state machine and full
validator, but they are not required by the canonical local release path.
