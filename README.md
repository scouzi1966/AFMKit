# AFMKit

AFMKit is a provider SDK for Swift applications. It supplies provider-neutral
model contracts, OpenAI-compatible DTOs, Apple Foundation Models integration,
and optional MLX and DwarfStar runtime adapters without importing the
maclocal-api server, CLI, WebUI, or transport layer.

The canonical boundaries, interfaces, dependency ledger, runtime diagrams, and
decisions are in [`Architecture/`](Architecture/README.md).

## Package architecture

Fourteen public modules are published across three real Swift package boundaries:

| Package | Public products | Dependency/authentication boundary |
| --- | --- | --- |
| `AFMKit` | `AFMKitCore`, `AFMOpenAICompat`, `AFMKitInference`, `AFMEvalKit`, `AFMKitApple`, `AFMKitEmbeddings`, `AFMKitSpeech`, `AFMKitSpeechSynthesis`, `AFMKitVision`, `AFMKitServices` | Zero SwiftPM dependencies. `AFMEvalKit` depends only on `AFMOpenAICompat`; Apple services are independently selectable and `AFMKitServices` is their compatibility umbrella. |
| `AFMKitDwarfStar` | `AFMKitDwarfStar`, `AFMKitFoundationModelsDwarfStar` | Exact AFMKit release plus public, exact-pinned Hub/Xet dependencies, the AFM-owned ds4 adapter/resources, and an Xcode 27-only `LanguageModel` bridge. |
| `AFMKitMLX` | `AFMKitMLX`, `AFMKitFoundationModelsMLX` | Exact AFMKit release plus the exact private AFM-compatible MLX graph. |

All three manifests use Swift tools 6.1 and keep a macOS 26 deployment floor.
With Xcode 26 (Swift 6.3), the root package exposes `AFMKitCore`,
`AFMOpenAICompat`, `AFMKitInference`, `AFMEvalKit`, and the five service products, and the MLX package exposes `AFMKitMLX`. Xcode 27 (Swift 6.4)
also exposes `AFMKitApple`, `AFMKitFoundationModelsMLX`, and
`AFMKitFoundationModelsDwarfStar`; those products import
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

## Apple service host requirements

Service products carry no bundled resources or third-party dependencies. A host
using `AFMKitSpeech` must provide `NSSpeechRecognitionUsageDescription` in its
own Info.plist and the user must authorize Speech Recognition. NaturalLanguage
embeddings and speech/voice services may download Apple-managed assets.

`AFMKitSpeechSynthesis` writes WAV and CAF in-process. AAC currently invokes the
system `/usr/bin/afconvert` tool after synthesis. Sandboxed applications that
cannot launch subprocesses should request WAV or CAF; replacing that AAC path
with an in-process resampler/encoder is tracked separately.

## Provider-free evaluation contracts

`AFMEvalKit` contains evaluation schemas, strict validation, deterministic
scoring, throughput metrics, aggregate output-token budgets, report models and
HTML/JSON rendering. It depends only on `AFMOpenAICompat` and Foundation and
does not load or select a model. Hosts continue to own suite discovery and
defaults, bundled suites, CLI planning, execution, signals, persistence paths,
and browser launching.

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
`docs/api-baselines/toolchain.env`. Baseline coverage discovers all fourteen modules
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
all three packages, and fresh downstream builds for all fourteen products:

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

On Xcode 27, `AFMKitFoundationModelsDwarfStar` exposes that provider through
Apple's `LanguageModel` and `LanguageModelExecutor` protocols. Provider-neutral
transcript and event translation lives in `AFMKitApple`, so DwarfStar does not
depend on MLX. Runtime leases retain a shared compatible DwarfStar engine until
the final executor releases it.

`AFMKitApple` is exposed by the manifest only with Xcode 27 or newer while the
package keeps its macOS 26 deployment floor. It owns reusable availability,
quota, signed-host entitlement validation, reasoning, and session behavior. The
host app owns signing, provisioning, entitlements, tools, user consent, UI, and
application state.

The downstream provider example is documented in
[`Examples/AFMKitQuickstart/README.md`](Examples/AFMKitQuickstart/README.md). The
durable migration plan is in [`docs/TRANSITION_PLAN.md`](docs/TRANSITION_PLAN.md).
