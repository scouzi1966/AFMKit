# Releasing AFMKit

AFMKit qualifies an untagged default-branch commit before staging a tag, then
qualifies the actual remote tag before creating its GitHub release. Publication
does not run from a pushed tag or directly from a manually selected workflow
branch.

## External prerequisites

- `AFMKIT_XCODE_RUNNER` must select a runner with the exact Xcode, SDK, Swift
  version, and compiler digest recorded in `docs/api-baselines/toolchain.env`.
- `AFMKIT_DEPENDENCY_TOKEN` must have read-only access to the private
  `mlx-swift-afm` and `mlx-swift-lm` repositories.
- Repository write access remains the ultimate trust boundary for changes to
  `main`. The workflow design prevents an arbitrary feature-branch dispatch from
  receiving release credentials or publishing, without depending on GitHub
  environments or rulesets that may be unavailable on the private-repository
  plan.

Protected environments and tag rulesets remain useful defense in depth when the
repository plan supports them, but the workflow does not require them. A tag
created outside this workflow is not release qualification evidence and does not
trigger publication.

## Pull request qualification

1. `CI` runs token-independent checks on the pull request candidate and never
   receives repository secrets.
2. After it succeeds, the default-branch `Private graph qualification` workflow
   starts through `workflow_run`.
3. Trusted base code resolves the unchanged `Package.resolved` with the private
   token scoped through a temporary Git configuration.
4. The token and configuration are removed before the resolved graph is moved to
   the candidate checkout. Candidate API extraction and Debug tests then run
   without credentials.
5. Missing private access skips this second qualification instead of failing the
   pull request. Fork pull requests are deliberately ineligible for private
   source provisioning.

## Release workflow

1. Commit and push the clean candidate to `main`; complete CI and private graph
   qualification.
2. Run **Request release** on `main` with a strict SemVer tag such as
   `v1.2.3-rc.1+build.5`. This workflow has read-only permissions and no secrets.
3. The request uploads its tag, run ID, and exact SHA. The privileged
   **Qualify and publish release** workflow starts through `workflow_run`, which
   GitHub loads from the default branch.
4. The privileged workflow verifies the artifact provenance, rejects malformed
   tags, and requires either that `main` still points to the requested SHA or
   that a previous attempt already created the same tag for that SHA.
5. `Scripts/validate-release.sh` rejects local overrides, unstable root
   branch/revision requirements, and unsafe target flags; validates the root
   remote lock; runs all API baselines and Release tests; and builds a fresh
   six-product consumer from an isolated Git tag.
6. The workflow creates the annotated tag for the exact qualified SHA. It then
   resolves the real GitHub tag into a brand-new lock and builds all six public
   products from that remote graph.
7. Only after remote-tag qualification succeeds does the workflow create the
   GitHub release.

Tag and release operations are idempotent. A rerun accepts a lightweight or
annotated tag only when it resolves to the qualified commit, resumes after a
tag-only partial publication, accepts an existing release for that matching tag,
and recovers from concurrent create responses. Any tag-to-commit mismatch fails
closed.

The dependency token is available only to trusted default-branch steps. Its Git
URL rewrites are limited to the two private dependencies; remote-tag validation
adds a similarly scoped current-repository rewrite using the read-only workflow
token. Every temporary authentication file is removed on success, failure,
interruption, or termination.
