# Releasing AFMKit

AFMKit publishes one qualified version across three Swift packages. Qualification
uses private staging mirrors; no final SwiftPM SemVer tag is exposed until every
package, API baseline, and fresh downstream graph has passed.

## Published package set

| Repository | Products | Release dependency on AFMKit |
| --- | --- | --- |
| `AFMKit` | `AFMKitCore`, `AFMOpenAICompat`, `AFMKitApple` | None; root manifest has zero package dependencies. |
| `AFMKitDwarfStar` | `AFMKitDwarfStar` | Exact same-version `AFMKit` tag. |
| `AFMKitMLX` | `AFMKitMLX`, `AFMKitFoundationModelsMLX` | Exact same-version `AFMKit` tag plus the exact private MLX graph. |

The provider package source remains under `Packages/` in the AFMKit development
repository. `Scripts/materialize-provider-package.py` creates self-contained
release trees and replaces only the trusted local AFMKit dependency declaration.
Product names and Swift imports do not change for existing consumers; their
`Package.swift` files must add the provider repository that owns each product.

## External prerequisites

- `AFMKIT_XCODE_RUNNER` selects the exact Xcode, SDK, Swift version, and compiler
  digest in `docs/api-baselines/toolchain.env`.
- `AFMKIT_DEPENDENCY_TOKEN` has read-only access to `mlx-swift-afm` and
  `mlx-swift-lm`.
- The `AFMKitDwarfStar` and `AFMKitMLX` GitHub repositories exist at the URLs in
  `AFMKIT_DWARFSTAR_PUBLISH_URL` and `AFMKIT_MLX_PUBLISH_URL` (repository
  variables may override the owner-derived defaults).
- `AFMKIT_PROVIDER_PUBLISH_TOKEN` can create tags in those two provider
  repositories. It is not used to resolve candidate code or dependencies.
- The root workflow token has `contents: write` only in the final publish job.

Protected environments and tag rulesets remain useful defense in depth. The
workflow itself fails closed on tag, source SHA, bundle, provenance, release
state, or remote-tag mismatches.

## Pull request qualification

1. Public CI checks out the immutable PR head SHA, runs token-independent gates,
   and uploads an allowlisted source/test/resource artifact named for that run.
   Candidate manifests, scripts, plugins, symlinks, and special files are not in
   the artifact.
2. The default-branch `workflow_run` handler uses only the successful run's
   immutable `head_sha`, run ID, repository, and pull-request base SHA. It never
   re-fetches a mutable PR head.
3. Trusted base code validates and extracts the exact artifact, supplies the
   fixed qualification manifest and lock, and prebuilds private dependencies
   from a trusted seed target.
4. Candidate Swift/C/C++ compilation runs without credentials through trusted
   compiler wrappers. The macOS sandbox denies those compiler processes all
   reads from private checkouts and SwiftPM repository caches; the build also
   proves the Swift and Clang wrappers were used.
5. Trusted API checks run under the same read restrictions. Private source,
   repository caches, and the isolated dependency HOME are then deleted before
   the prebuilt candidate test bundle executes.

Fork pull requests are ineligible for private qualification. A missing private
dependency token reports the qualification as unavailable instead of passing a
partial private check.

## Release sequence

1. Run **Request release** on `main` with a strict SwiftPM-compatible tag such as
   `v1.2.3` or `v1.2.3-rc.1`. Build metadata (`+build`) is rejected because
   SwiftPM does not treat it as a distinct package release.
2. The unprivileged request records the run ID, exact default-branch SHA, and tag.
   The privileged workflow starts through `workflow_run` and verifies that
   immutable provenance.
3. `Scripts/validate-release.sh` validates all three manifests, both provider
   locks, all six API baselines, Release tests, workflow/security regressions,
   the unauthenticated Core consumer, and DwarfStar resources.
4. The downstream gate materializes provider release repositories in private
   temporary storage. It creates local staging tags only, resolves fresh no-lock
   consumers, compares their complete dependency pins with the committed
   provider locks, and builds all six products from the exact release manifests.
5. Qualification uploads the two verified Git bundles plus `publication.json`.
   No root or provider tag has been pushed at this point.
6. The publish job verifies bundle tags, source SHA, release manifest, and
   production AFMKit URL. It creates or recovers the two provider tags first.
   Any existing tag must resolve to the qualified provider commit.
7. The root AFMKit tag and GitHub release are created last. Stable releases use
   `prerelease=false` and `make_latest=true`; prereleases use `prerelease=true`
   and `make_latest=false`. Existing releases must already have the matching
   draft, prerelease, and latest state.

Publication is idempotent. A partial provider-tag publication can be rerun only
when every existing tag resolves to the exact qualified commit. The root tag is
never a staging signal and is never created before final qualification succeeds.
