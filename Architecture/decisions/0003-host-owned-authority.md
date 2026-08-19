# ADR 0003: Host-owned authority and system integration

- **Status:** Accepted
- **Date:** 2026-08-19

## Context

Model tool calls, PCC access, credentials, App Intents, Siri/Spotlight exposure,
and application data can produce user-visible or destructive effects. A reusable
model package lacks the application context and signing identity to authorize
them safely.

## Decision

The host app owns signed entitlements/provisioning, credentials, Keychain policy,
App Intents and app entities, executable tools, user consent, side effects, and
persistence. AFMKit transports definitions/events and probes current-process
capabilities but does not grant authority.

## Consequences

- AFMKit remains reusable and least-privileged.
- Apps must implement a tool authorization boundary.
- PCC cannot be enabled merely by adding a package dependency.
- Common host patterns may be documented or placed in optional helper products,
  but must not weaken the authority boundary.
