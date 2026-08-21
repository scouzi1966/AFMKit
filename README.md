# AFMKit

AFMKit is a provider SDK for Swift applications. It supplies provider-neutral
model contracts, OpenAI-compatible DTOs, Apple Foundation Models integration,
and optional MLX and DwarfStar runtime adapters without importing the
maclocal-api server, CLI, WebUI, or transport layer.

The canonical boundaries, interfaces, dependency ledger, runtime diagrams, and
decisions are in [`Architecture/`](Architecture/README.md).

## Package architecture

Six public modules are published across three real Swift package boundaries:

| Package | Public products | Dependency/authentication boundary |
| --- | --- | --- |
| `AFMKit` | `AFMKitCore`, `AFMOpenAICompat`, `AFMKitApple` | Zero SwiftPM dependencies. Core-only consumers never resolve or authenticate to MLX. |
| `AFMKitDwarfStar` | `AFMKitDwarfStar` | Exact AFMKit release plus public, exact-pinned Hub/Xet dependencies and the AFM-owned ds4 adapter/resources. |
| `AFMKitMLX` | `AFMKitMLX`, `AFMKitFoundationModelsMLX` | Exact AFMKit release plus the exact private AFM-compatible MLX graph. |

All three manifests use Swift tools 6.1 and keep a macOS 26 deployment floor.
With Xcode 26 (Swift 6.3), the root package exposes `AFMKitCore` and
`AFMOpenAICompat`, and the MLX package exposes `AFMKitMLX`. Xcode 27 (Swift 6.4)
also exposes `AFMKitApple` and `AFMKitFoundationModelsMLX`; those products import
macOS 27 Foundation Models APIs and remain runtime-gated to macOS 27. CI checks
both product matrices with `Scripts/check-sdk-product-exposure.sh`.

The monorepo keeps provider source under `Packages/` for coordinated development.
Release automation materializes those directories into separate provider
repositories and rewrites their local AFMKit dependency to the exact same release
version. Existing Swift `import` and product names remain compatible; consumers
only migrate their package dependency declarations.

A Core-only consumer needs only the public root package:

```swift
.package(url: "https://github.com/scouzi1966/AFMKit.git", exact: "0.1.0")
```

Consumers selecting a runtime add its independently published package:

```swift
.package(url: "https://github.com/scouzi1966/AFMKitDwarfStar.git", exact: "0.1.0"),
.package(url: "https://github.com/scouzi1966/AFMKitMLX.git", exact: "0.1.0")
```

## Local development

```bash
git clone --recurse-submodules git@github.com:scouzi1966/AFMKit.git
cd AFMKit
swift test
swift test --package-path Packages/AFMKitDwarfStar
swift test --package-path Packages/AFMKitMLX
Scripts/test-api-gate.sh
Scripts/check-api-baselines.sh
```

The provider manifests use the root checkout by default. Local MLX compatibility
checkouts can replace the tagged private dependencies during development:

```bash
AFMKIT_MLX_SWIFT_PATH=/path/to/mlx-swift-afm \
AFMKIT_MLX_SWIFT_LM_PATH=/path/to/mlx-swift-lm-afm \
swift test --package-path Packages/AFMKitMLX -c release
```

Release qualification rejects these overrides. Direct and qualified transitive
provider dependencies are constrained exactly, and a fresh no-lock consumer must
reproduce each committed provider lock before publication.

## Qualification

The checked-in API baselines use Xcode 27 Beta 3 build `27A5218g`, macOS SDK 27.0
build `26A5378i`, and the compiler identity in
`docs/api-baselines/toolchain.env`. Baseline coverage discovers all six modules
across the three manifests. The current DwarfStar baseline contains 42 normalized
public symbols.

Public CI also proves that a fresh Core consumer builds with isolated credentials
and no MLX checkout. Private PR qualification consumes only an immutable artifact
from the exact successful workflow run. Candidate manifests and locks are inert
inputs whose dependency graph must equal the trusted default-branch graph before
the trusted qualification manifest can prebuild private dependencies. Candidate
compiler processes are sandboxed from private source and repository caches, and
candidate test sources are compiled but never executed on the privileged runner.
Candidate scripts and plugins never run there. Private source, compiled products,
caches, and credentials are destroyed before the job exits.

Full release validation runs all package/API/security gates, Release tests for
all three packages, and fresh downstream builds for all six products:

```bash
Scripts/validate-release.sh
```

Release qualification uses private staging mirrors. It publishes the two
qualified provider tags and creates the root AFMKit tag only after all tests pass.
SwiftPM release tags reject SemVer build metadata. Prerelease GitHub releases are
created with `prerelease=true` and `make_latest=false`. See
[`docs/RELEASING.md`](docs/RELEASING.md).

## Provider boundaries

`AFMKitMLX` enters through `AFMMLXModel`, which implements `AFMModel` and the
provider-specific `AFMMLXOpenAIChatServing` facade. HTTP routing, Prometheus,
files, request orchestration, and benchmark compatibility stay in maclocal-api.
Advanced MLX engine types remain public for specialized consumers but are not the
stable cross-provider contract.

`AFMKitDwarfStar` enters through `AFMDwarfStarProviderFactory`,
`AFMDwarfStarModel`, and `AFMDwarfStarRuntimeConfiguration`. AFMKit pins an
unmodified `antirez/ds4` submodule and owns the Swift/C adapter, resources,
checkpoint projection, and provider lifecycle.

`AFMKitApple` is exposed by the manifest only with Xcode 27 or newer while the
package keeps its macOS 26 deployment floor. It owns reusable availability,
quota, signed-host entitlement validation, reasoning, and session behavior. The
host app owns signing, provisioning, entitlements, tools, user consent, UI, and
application state.

The downstream provider example is documented in
[`Examples/AFMKitQuickstart/README.md`](Examples/AFMKitQuickstart/README.md). The
durable migration plan is in [`docs/TRANSITION_PLAN.md`](docs/TRANSITION_PLAN.md).
