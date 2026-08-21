# Trusted Private Qualification Harness

`PrivatePackage.swift` is loaded only from the immutable trusted base selected by
the `workflow_run` payload. Candidate package manifests, scripts, plugins, and
dependency locks are deliberately excluded from the public CI artifact and are
never evaluated while private dependency source is present.

The harness compiles allowlisted candidate source and tests with fixed target
types and exact dependency constraints. The workflow removes private checkouts,
bare repositories, the isolated SwiftPM home, and temporary Git authentication
before it launches the already-built XCTest bundle. This permits source/API
qualification without giving executing candidate code access to dependency
source or credentials.
