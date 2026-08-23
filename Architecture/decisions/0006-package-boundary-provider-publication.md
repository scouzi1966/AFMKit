# ADR 0006: Separate Public and Provider Package Boundaries

- Status: Accepted
- Date: 2026-08-21

## Context

SwiftPM resolves dependencies for an entire package manifest, not only for the
product a consumer imports. Keeping Core and runtime adapters as separate targets
inside one manifest still forced a Core-only consumer to resolve the private
AFM-compatible MLX repositories. This contradicted the dependency-free public
Core contract and made authentication a package-global requirement.

Provider releases must also be tested against the graph a fresh consumer will
resolve. Publishing the root SemVer tag before that qualification creates a
public package version that cannot be retracted safely if the graph fails.

## Decision

Publish twelve public modules through three repositories and package manifests:

- `AFMKit`: Core, OpenAI compatibility, provider-neutral inference, Apple,
  four independently selectable Apple services, and their compatibility umbrella;
  no SwiftPM dependencies.
- `AFMKitDwarfStar`: DwarfStar provider; exact same-version AFMKit dependency.
- `AFMKitMLX`: MLX and FoundationModelsMLX providers; exact same-version AFMKit
  dependency and exact private MLX graph.

The AFMKit monorepo remains the coordinated source repository. Qualification
materializes provider release trees, rewrites only their trusted local AFMKit
dependency, resolves fresh no-lock consumers, compares the complete provider
graph with committed locks, and builds every product. Only after this succeeds
may automation publish provider tags and then the root AFMKit tag last.

Product and module names remain unchanged. Consumers migrate package URLs and
the package identity in `.product` declarations, not Swift imports or API calls.

## Consequences

- Core-only and Apple-only consumers have no inference dependency or private
  authentication path.
- Provider repositories and a cross-repository publishing credential are release
  prerequisites.
- All three repositories use one version, and provider manifests pin the exact
  root version.
- Partial publication is recoverable only when existing tags resolve to the
  qualified commits. Build metadata is forbidden in release tags because
  SwiftPM does not give it distinct package-version identity.
- The root tag is a final publication result, never a qualification/staging ref.

## Verification

- `Scripts/check-unauthenticated-core-consumer.sh`
- `Scripts/check-downstream-example.sh`
- `Scripts/test-provider-publication.sh`
- `Scripts/test-release-qualification.sh`
- `Scripts/test-workflow-security.sh`
- `Scripts/validate-release.sh`
