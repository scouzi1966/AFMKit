# AFMKit

AFMKit is a provider SDK for Swift applications. It supplies provider-neutral
model contracts, OpenAI-compatible DTOs, Apple Foundation Models integration,
and optional MLX and DwarfStar runtime adapters without importing the
maclocal-api server, CLI, WebUI, or transport layer.

The canonical boundaries, interfaces, dependency ledger, runtime diagrams, and
decisions are in [`Architecture/`](Architecture/README.md).

## Package architecture

Fifteen public modules are published from one Swift package and one tag:

| Area | Public products | Dependency boundary |
| --- | --- | --- |
| Core and Apple | `AFMKitCore`, `AFMOpenAICompat`, `AFMKitInference`, `AFMEvalKit`, `AFMKitApple`, `AFMKitEmbeddings`, `AFMKitSpeech`, `AFMKitSpeechSynthesis`, `AFMKitVision`, `AFMKitServices` | Provider-neutral contracts and independently selectable Apple services. |
| DwarfStar | `AFMKitDwarfStar`, `AFMKitFoundationModelsDwarfStar` | Exact-pinned Hub/Xet dependencies, the AFM-owned ds4 adapter/resources, and an Xcode 27-only `LanguageModel` bridge. |
| MLX | `AFMKitMLX`, `AFMKitMLXAudio`, `AFMKitFoundationModelsMLX` | The exact AFM-compatible MLX graph, provider-neutral local audio synthesis, and the Xcode 27 bridge. |

The root manifest uses Swift tools 6.1, keeps a macOS 26 deployment floor, and
declares an iOS 16 deployment floor for the basic provider-neutral layer.
With Xcode 26 (Swift 6.3), the root package exposes `AFMKitCore`,
`AFMOpenAICompat`, `AFMKitInference`, `AFMEvalKit`, the five service products,
`AFMKitMLX`, and `AFMKitMLXAudio`. Xcode 27 (Swift 6.4)
also exposes `AFMKitApple`, `AFMKitFoundationModelsMLX`, and
`AFMKitFoundationModelsDwarfStar`; those products import
macOS 27 Foundation Models APIs and remain runtime-gated to macOS 27. CI checks
both product matrices with `Scripts/check-sdk-product-exposure.sh`.

The supported iOS surface is intentionally narrow: `AFMKitCore`,
`AFMOpenAICompat`, and `AFMKitInference`. An arm64 iOS Simulator consumer
compiles these three products in CI without compiling any concrete provider or
service module. Speech, Vision, MLXAudio, Services, DwarfStar, MLX, Apple
Foundation Models, and their bridges have not completed independent iOS audits
and must not yet be selected by iOS targets. Their macOS behavior is unchanged.
Broader iOS work is tracked in
[#23](https://github.com/scouzi1966/AFMKit/issues/23).

Provider source remains organized under `Packages/`, but those directories are
targets of the root manifest rather than independently versioned packages.
Existing Swift imports and product names remain compatible.

A Core-only consumer needs only the public root package:

```swift
.package(url: "https://github.com/scouzi1966/AFMKit.git", exact: "0.1.0")
```

Consumers select runtime products from that same package dependency:

```swift
.product(name: "AFMKitDwarfStar", package: "AFMKit")
.product(name: "AFMKitMLX", package: "AFMKit")
.product(name: "AFMKitMLXAudio", package: "AFMKit")
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
and browser launching. Validation applies equally to decoded and
programmatically constructed suites, including bounded nested generation
configuration, tool names, response formats, stop strings, and expectations.

## Local development

```bash
git clone --recurse-submodules https://github.com/scouzi1966/AFMKit.git
cd AFMKit
swift test
Scripts/test-api-gate.sh
Scripts/check-api-baselines.sh
```

The AFM-compatible MLX, MLX C, Swift bindings, and language-model sources live
under `vendor/MLX`. They are ordinary source snapshots with their upstream
licenses and provenance, not submodules or separately authenticated packages.
Release qualification accepts only these two repository-relative package paths;
all other dependencies remain exact public HTTPS pins.

## Qualification

The checked-in API baselines use Xcode 27 build `27A5228h`, macOS SDK 27.0
build `26A5388f`, and the compiler identity in
`docs/api-baselines/toolchain.env`. Baseline coverage discovers all fifteen modules
from the root manifest. The current DwarfStar baseline contains 42 normalized
public symbols.

Public CI also proves that a fresh Core consumer builds without credentials and
that a minimal arm64 iOS Simulator consumer compiles the three supported basic
products at the iOS 16 deployment floor.
Trusted PR qualification consumes only an immutable artifact
from the exact successful workflow run. Candidate manifests and locks are inert
inputs whose dependency graph must equal the trusted default-branch graph before
the trusted qualification manifest compiles the candidate. The allowlisted artifact
contains the vendored MLX sources and licenses required by that graph. Candidate
compiler processes are sandboxed from unrelated repository caches, and
candidate test sources are compiled but never executed on the privileged runner.
Candidate scripts and plugins never run there. Compiled products, caches, and the
downloaded candidate artifact are destroyed before the job exits.

Full release validation runs all package/API/security gates, the root Release
tests, and fresh downstream builds for all fifteen products:

```bash
Scripts/validate-release.sh
```

The release dependency gate verifies that all external dependencies remain
exact-pinned and match `Package.resolved` without forcing unrelated products
into `AFMKitMLX` consumers. Clean downstream MLX build measurements and their
minimal fixture are documented in
[`docs/MLX_CONSUMER_BUILD.md`](docs/MLX_CONSUMER_BUILD.md).

Release qualification creates one AFMKit tag only after all tests pass.
SwiftPM release tags reject SemVer build metadata. Prerelease GitHub releases are
created with `prerelease=true` and `make_latest=false`. See
[`docs/RELEASING.md`](docs/RELEASING.md).

## Provider boundaries

`AFMKitMLX` enters through `AFMMLXModel`, which implements `AFMModel` and the
provider-specific `AFMMLXOpenAIChatServing` facade. HTTP routing, Prometheus,
files, request orchestration, and benchmark compatibility stay in maclocal-api.
Advanced MLX engine types remain public for specialized consumers but are not the
stable cross-provider contract.

`AFMKitMLXAudio` owns MLX audio model discovery, explicit download with progress,
shared-cache resolution, deletion, loading, synthesis, streaming, cancellation,
telemetry, and WAV encoding. `AFMMLXAudioRuntime.load` is local-only by default;
hosts retain control of user consent by calling `AFMMLXAudioModelStore.download`
or opting into `downloadIfNeeded` only after approval.

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
