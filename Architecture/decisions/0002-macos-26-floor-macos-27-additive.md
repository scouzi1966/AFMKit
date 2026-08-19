# ADR 0002: macOS 26 floor with additive macOS 27 support

- **Status:** Accepted
- **Date:** 2026-08-19

## Context

AFMKit consumers need local providers on macOS 26 while macOS 27 introduces
Foundation Models provider protocols, PCC, reasoning, and related features.
Raising the whole package floor would unnecessarily split applications/builds.

## Decision

Keep the package deployment target at macOS 26. Compile macOS 27 Foundation
Models integrations conditionally and annotate their public APIs with
`@available(macOS 27.0, *)`. Hosts hide or reveal features using OS availability,
provider capability descriptors, and runtime availability.

## Consequences

- One source package can support macOS 26 and 27.
- Consumers must use availability checks correctly.
- Tests need both compatibility and macOS 27 feature paths.
- New macOS 27 APIs cannot leak into the neutral core’s unconditional signatures.
