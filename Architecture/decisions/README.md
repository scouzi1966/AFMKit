# Architecture Decision Records

ADRs capture decisions that materially constrain AFMKit structure or evolution.
They are immutable after acceptance except for status and supersession links; a
new decision supersedes an old one.

| ADR | Status | Decision |
| --- | --- | --- |
| [0001](0001-provider-neutral-core.md) | Accepted | Keep a provider-neutral dependency-free core. |
| [0002](0002-macos-26-floor-macos-27-additive.md) | Accepted | Keep macOS 26 floor; add macOS 27 features through availability-gated products. |
| [0003](0003-host-owned-authority.md) | Accepted | The host owns entitlements, tools, credentials, intents, and side effects. |
| [0004](0004-engine-adapters-and-pins.md) | Accepted | Isolate engines behind adapters and qualify explicit dependency pins. |
| [0005](0005-transport-outside-afmkit.md) | Accepted | Keep HTTP/server transports outside AFMKit. |
| [0006](0006-package-boundary-provider-publication.md) | Accepted | Publish dependency-free and runtime products through separate package boundaries. |

New ADRs use the next four-digit number and contain context, decision,
consequences, alternatives, and verification implications.
