# Independent Architecture Review — 2026-08-19

## Scope

An independent software-architect agent performed a read-only review of AFMKit
commit `1dc5b2197c16083748c37db88dc580f266e304b5`. It inspected package products,
public symbol baselines, provider implementations, Apple/macOS 27 integration,
dependencies, examples, and transition documents. It did not edit code or rerun
tests.

## Findings incorporated into this architecture set

| Finding | Documentation response | Follow-up disposition |
| --- | --- | --- |
| MLX implementation types are public beyond the claimed facade. | Added explicit stability tiers and identified `MLXModelService` exposure. | Narrow through a future compatibility-reviewed change; do not misstate current access. |
| `AFMKitFoundationModelsMLX` is a shipping product but omitted from product governance and lacks an API baseline. | Added it to all module/product views and marked it experimental. | Add a symbol baseline before declaring its public API stable. |
| Core capabilities include audio/embeddings without dedicated executable contracts. | Documented reserved vocabulary and no-over-advertising rule. | Add capability-specific protocols/events before portable use. |
| Apple, MLX, and DwarfStar discovery/availability semantics differ. | Added a provider behavior matrix and clarified what `.available` means. | Improve provider guarantees and tests without forcing false uniformity. |
| Not all Git dependencies are exact-pinned for downstream consumers. | Dependency ledger distinguishes exact/revision pins from semver ranges. | Decide whether release-critical semver ranges should become exact pins. |
| String-keyed configuration is an undocumented second API. | Added key/type/default tables, alias precedence, and evolution rules. | Generate descriptors/validation from typed key schemas. |
| Apple module has neutral, advanced-session, and legacy/specialized surfaces. | Added explicit Apple API layering and preferred entry point. | Deprecate or isolate overlapping lifecycle surfaces intentionally. |
| Siri/App Intents/Spotlight are not implemented by AFMKit. | Marked them host-owned and described the composition pattern. | No claim of implementation; optional helpers require separate evidence/decision. |

## Additional recommendations retained for future decisions

- Add ADRs for event append/replace semantics, Apple stateless versus reusable
  sessions, metadata namespacing, and the API stability tiers.
- Resolve the excluded duplicate xgrammar source tree under
  `Sources/CXGrammar/xgrammar` to remove supply-chain ambiguity.
- Add provider-specific discovery, availability, download, validation, and
  capability-confidence contract tests.
- Ensure signed PCC generation claims are validated in a signed downstream host;
  package unit tests alone cannot reproduce entitlement/provisioning behavior.

## Review conclusion

The package has a coherent provider-neutral center and real Apple/MLX/DwarfStar
adapters, but the target facade is narrower than the current public symbol graph.
Architecture documentation must therefore describe both the implemented state and
the intended boundary. This review record is evidence of consultation, not a
substitute for code review or release qualification.
