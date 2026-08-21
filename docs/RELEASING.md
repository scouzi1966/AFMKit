# Releasing AFMKit

AFMKit releases are qualified before a tag or GitHub release exists. A pushed
tag is not a qualification trigger.

## External prerequisites

- `AFMKIT_XCODE_RUNNER` must select a runner with the exact Xcode, SDK, Swift
  version, and compiler digest recorded in `docs/api-baselines/toolchain.env`.
- `AFMKIT_DEPENDENCY_TOKEN` must have read-only access to the private
  `mlx-swift-afm` and `mlx-swift-lm` repositories. Full package builds and API
  extraction cannot resolve that private graph without this external access.
- The `release` GitHub environment should require the repository's intended
  release approval.
- A repository tag ruleset must restrict direct creation or update of `v*`
  tags so maintainers cannot bypass the qualified release workflow with a
  manual push. Repository rulesets are external GitHub configuration and cannot
  be enforced by files in this repository alone.

## Workflow

1. Commit and push the candidate. The candidate worktree must be clean.
2. Run CI. Token-independent gates always run; the private-graph API and Debug
   package jobs must also pass.
3. Dispatch **Qualify and publish release** from the candidate commit and enter
   a semantic `v*` tag.
4. The workflow rejects existing or malformed tags, then runs
   `Scripts/validate-release.sh` against the untagged commit.
5. Only after qualification succeeds and any `release` environment approval is
   granted does the publication job create an annotated tag for the exact
   qualified SHA and publish the GitHub release.

`Scripts/validate-release.sh` rejects local MLX path overrides, validates remote
resolved pins, uses `--disable-automatic-resolution`, and verifies that neither
`HEAD`, `Package.resolved`, nor any tracked or untracked worktree state changed.
The private dependency token is scoped through a temporary Git configuration
and is removed on success, failure, interruption, or termination.
