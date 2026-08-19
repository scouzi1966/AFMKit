# External Git Dependencies

AFMKit deliberately keeps `AFMKitCore` dependency-free, but provider products
integrate substantial third-party runtime stacks. This ledger distinguishes
direct dependencies, the pinned DwarfStar submodule, and major transitive groups.
`Package.swift` and `Package.resolved` are authoritative for exact resolution.

## Dependency and derivation graph

```mermaid
flowchart TB
    Core["AFMKitCore"]
    MLX["AFMKitMLX"]
    DS["AFMKitDwarfStar"]
    FMMLX["AFMKitFoundationModelsMLX"]
    UpstreamMLX["ml-explore/mlx-swift\nupstream 0.31.6 lineage"]
    UpstreamLM["ml-explore/mlx-swift-lm\nupstream lineage"]
    RuntimePatches["maclocal-api patch catalog\nmlx-swift-deepseek-v4"]
    LMPatches["maclocal-api/Scripts/patches\nAFM LM source of truth"]
    MLXSwift["scouzi1966/mlx-swift-afm\n0.31.6-afm.1\ntagged materialization"]
    MLXLM["scouzi1966/mlx-swift-lm\n0.31.6-afm.3\ntagged materialization"]
    Transformers["huggingface/swift-transformers\nfrom 1.3.0"]
    HF["huggingface/swift-huggingface\nfrom 0.8.1 + Xet trait"]
    Xet["huggingface/swift-xet\n0.2.3"]
    XGrammar["mlc-ai/xgrammar\nrevision pin"]
    Dwarf["antirez/ds4\ngit submodule pin"]
    Apple["Apple frameworks\nFoundationModels / Metal / Security / IOKit"]

    MLX --> Core
    FMMLX --> MLX
    DS --> Core
    MLX --> MLXSwift
    MLX --> MLXLM
    UpstreamMLX --> MLXSwift
    RuntimePatches --> MLXSwift
    UpstreamLM --> MLXLM
    LMPatches --> MLXLM
    MLX --> Transformers
    MLX --> HF
    MLX --> XGrammar
    DS --> HF
    DS --> Xet
    DS --> Dwarf
    MLX --> Apple
    DS --> Apple
    FMMLX --> Apple
```

**Figure 1 — Runtime dependencies and their derivation.** Solid provider arrows
show what a downstream SwiftPM build resolves. The upstream-and-patch arrows show
where the AFM-compatible MLX artifacts come from; the two `scouzi1966` tags are
distribution materializations, not independent sources of AFM behavior. A
consumer that imports only `AFMKitCore` does not resolve an inference engine.

## MLX source-of-truth model

The normal maclocal-api development workflow remains upstream-compatible source
plus the checked-in patch catalog:

1. MLX LM changes are authored and reviewed under
   `maclocal-api/Scripts/patches/`.
2. MLX runtime/C++ changes are authored under the corresponding
   `maclocal-api/Scripts/patches/mlx-swift-*` directories.
3. maclocal-api applies those patches to its pinned dependency checkout for local
   builds and qualification.
4. The patched trees are materialized and tagged in the `scouzi1966` compatibility
   repositories for AFMKit distribution.
5. AFMKit pins those immutable tags because SwiftPM has no package-manifest phase
   that can safely patch a remote dependency during consumer resolution.

The compatibility repositories therefore must not be edited as a second source
of truth. A tag must identify its upstream base, maclocal-api source commit, patch
set, and qualification evidence. `mlx-swift-lm` already records regenerated
commits from maclocal-api. Reproducible publication of the `mlx-swift-afm` runtime
materialization remains a supply-chain automation gap and must be completed
before AFMKit is made public.

## Direct dependency ledger

| Dependency | Resolution | Used by | Purpose | Architecture impact |
| --- | --- | --- | --- | --- |
| [`scouzi1966/mlx-swift-afm`](https://github.com/scouzi1966/mlx-swift-afm) | Exact `0.31.6-afm.1` | `AFMKitMLX` | Tagged distribution materialization of upstream MLX plus the AFM runtime compatibility delta. | Not an independent source of truth; updates require provenance plus Metal/runtime regression tests. |
| [`scouzi1966/mlx-swift-lm`](https://github.com/scouzi1966/mlx-swift-lm) | Exact `0.31.6-afm.3` | `AFMKitMLX` | Tagged distribution materialization generated from the maclocal-api MLX LM patch catalog. | Do not edit independently; public app code must not import it directly. |
| [`huggingface/swift-transformers`](https://github.com/huggingface/swift-transformers) | `from: 1.3.0` | `AFMKitMLX` | Tokenizers and Hub facilities. | Semver range can advance transitively; lockfile and qualification are required. |
| [`huggingface/swift-huggingface`](https://github.com/huggingface/swift-huggingface) | `from: 0.8.1`, Xet trait | MLX, DwarfStar | Hub repository metadata/download. | Network/cache behavior belongs to provider layer. |
| [`huggingface/swift-xet`](https://github.com/huggingface/swift-xet) | Exact `0.2.3` | DwarfStar; also Hub trait path | High-throughput Hub transport. | Retry/resume and filesystem-space behavior need integration tests. |
| [`mlc-ai/xgrammar`](https://github.com/mlc-ai/xgrammar) | Revision `c1570cdb4f8c867a4dbd07b7ff90581f4a2a432b` | `AFMXGrammar`, MLX | Grammar-constrained generation. | C++ ABI/build risk; revision updates require grammar correctness tests. |
| [`antirez/ds4`](https://github.com/antirez/ds4) | Submodule `84cc882352757baf628a1776badf7cc54d584e28` | DwarfStar | Vanilla DwarfStar Metal source/resources. | Kept unmodified; AFM-specific behavior belongs in AFM-owned bridge/adapter. |

Local development can replace the two tagged MLX materializations through
`AFMKIT_MLX_SWIFT_PATH` and `AFMKIT_MLX_SWIFT_LM_PATH`. Release qualification
must use the tagged dependencies unless the artifact explicitly records local
overrides.

## Apple platform dependencies

| Framework/library | Product | Role |
| --- | --- | --- |
| Foundation Models | Apple provider, MLX bridge | On-device/PCC sessions and macOS 27 provider protocols. |
| Metal | MLX, DwarfStar | Local GPU execution. |
| Security | Apple, MLX | Entitlement/Keychain-related platform integration. |
| IOKit / IOReport | MLX | Hardware/runtime telemetry. |
| SQLite3 | MLX | Provider cache/metadata facilities. |
| Foundation | All | Swift data, concurrency, files, networking support. |

## Transitive dependency groups

The resolved graph currently includes the following major groups through Hub and
network packages:

- SwiftNIO, NIO HTTP/2, SSL, transport services, and async-http-client.
- Swift Crypto, Certificates, ASN.1, and HTTP structured headers/types.
- Swift Collections, Algorithms, Atomics, Numerics, System, Log, and service
  lifecycle/context/tracing packages.
- Jinja and EventSource used by model templates/streaming dependencies.
- yyjson and xgrammar implementation dependencies.

These are not AFMKit public interfaces. A transitive update can still affect
binary size, minimum OS support, TLS/network behavior, or build compatibility,
so `Package.resolved` changes require review.

## Dependency policy

1. **Pin execution engines exactly.** MLX compatibility materializations, Xet,
   xgrammar, and ds4 advance through explicit PRs with model and performance
   qualification.
2. **Keep vanilla upstreams vanilla.** Do not push AFM changes to ds4; adapt it in
   `CDwarfStar`/`AFMKitDwarfStar`.
3. **Keep one source of truth.** AFM MLX deltas live in maclocal-api's patch
   catalog. Tagged compatibility repositories are generated distribution output,
   never an alternate development branch.
4. **No transitive API leakage.** Public neutral signatures use AFMKit or Apple
   standard types, not third-party engine types.
5. **Record provenance.** Release artifacts record commit, dependency pins,
   model/checkpoint identity, compiler, SDK, and Metal resource identity.
6. **Review licenses and notices.** Before public distribution, verify each
   dependency’s current license and attribution requirements from its pinned
   source; `vendor/ds4` is currently recorded as MIT.
7. **Automate supply-chain checks.** CI should detect uncommitted submodules,
   changed pins, unexpected local overrides, missing licenses, package graph
   drift, and a tagged materialization that cannot be reproduced from its
   recorded upstream base and maclocal-api patch commit.

## Removing a compatibility materialization

For each AFM-compatible MLX materialization:

1. Maintain a machine-readable inventory of changes relative to upstream.
2. Classify changes as model architecture, performance, API, Metal compatibility,
   or bug fix.
3. Upstream generally useful fixes where feasible.
4. Keep AFM-only integration in AFMKit adapters or the maclocal-api patch catalog
   rather than editing the materialized repository.
5. Run the same compatibility/model matrix against upstream releases.
6. Replace the materialization with upstream packages only when public API,
   model correctness, performance, and macOS 26/27 package tests meet the
   recorded baseline.
