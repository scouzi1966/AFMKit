# ADR 0007: One Repository, Manifest, Version, and Tag

- Status: Accepted
- Date: 2026-08-23
- Supersedes: ADR 0006

## Context

The original product requirement called for AFMKit to be one canonical
repository. ADR 0006 introduced separate MLX and DwarfStar repositories to keep
Core dependency resolution isolated, but that multiplied publication,
credentials, tags, and consumer failure modes. The operational cost outweighed
that resolver-level isolation.

## Decision

Publish all fifteen products from the root `Package.swift` and one AFMKit tag.
Provider source may remain organized under `Packages/`, but those directories do
not contain independently versioned manifests or locks. Consumers declare one
exact AFMKit dependency and select the products they need.

The root lock pins the complete provider graph. Target dependencies still
prevent applications from linking provider or Apple frameworks they do not use,
although SwiftPM resolves the package-level dependency graph once.

## Consequences

- One repository, version, tag, lock, release workflow, and rollback boundary.
- No provider mirror repositories or cross-repository publication token.
- Core-only consumers resolve the root dependency graph but compile and link
  only Core targets.
- Every external dependency must be anonymously readable before AFMKit is made
  public; the unauthenticated consumer gate enforces this.
- The first public release remains blocked while any pinned dependency is private.

## Verification

- `Scripts/check-sdk-product-exposure.sh`
- `Scripts/check-api-baseline-coverage.sh`
- `Scripts/check-unauthenticated-core-consumer.sh`
- `Scripts/check-downstream-example.sh`
- `Scripts/validate-release.sh`
