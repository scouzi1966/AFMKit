# ADR 0004: Engine adapters and qualified dependency pins

- **Status:** Accepted
- **Date:** 2026-08-19

## Context

MLX and DwarfStar evolve independently, expose incompatible native types, and
carry performance-sensitive Metal code. Unbounded upgrades or leaked engine
types make AFMKit consumers fragile.

## Decision

Wrap each engine in an AFMKit-owned provider adapter. Pin execution-critical
dependencies explicitly, keep vanilla ds4 unmodified as a submodule, and put
AFM behavior in AFM-owned Swift/C bridges.

For MLX, maclocal-api's checked-in patch catalog is the authoritative AFM delta.
AFMKit consumes immutable tagged materializations of upstream MLX plus that delta
because SwiftPM cannot safely run repository patch scripts while resolving a
remote package. The tagged repositories are distribution artifacts, not a second
development source. Release builds qualify dependency pins with real models and
record both upstream and patch provenance.

## Consequences

- Engine churn is localized.
- Patch, materialization, and pin maintenance is a real ongoing cost.
- Materialization must be reproducible and must never diverge from the recorded
  maclocal-api patch source.
- Upgrades require correctness, performance, concurrency, cache, and downstream
  validation rather than compilation alone.
- Provider-specific advanced APIs may exist but cannot become neutral-core
  requirements.
