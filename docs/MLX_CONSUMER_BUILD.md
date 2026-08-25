# AFMKitMLX downstream build cost

AFMKit keeps every external release dependency declared at an exact version and
checks those declarations against `Package.resolved`. Exact-version policy is a
resolution and release concern; it must not make unrelated dependency products
build-reachable from `AFMKitMLX`.

`AFMKitMLXReleaseGraph` previously made 26 pinned products reachable from every
MLX consumer. In particular, the consumer compiled a second BoringSSL/Crypto
stack, certificate handling, structured headers, and service-lifecycle modules
that are not required by the MLX runtime.

The minimal external-consumer fixture is under
`Tests/Fixtures/AFMKitMLXConsumer`. Measure a clean Release build with the
qualified Xcode toolchain by running:

```bash
Scripts/measure-mlx-consumer-build.sh
```

Set `AFMKIT_KEEP_MLX_CONSUMER_BUILD=1` to retain the scratch directory, or set
`AFMKIT_MLX_CONSUMER_REPORT=/absolute/path/report.json` to copy the JSON summary
before cleanup. The comparison defaults to 12 build jobs; override it with
`AFMKIT_MLX_CONSUMER_JOBS` only when recording a separately identified run.

## Xcode 27 baseline

The baseline was captured from commit `54ad34d` (v0.1.4) with Xcode 27 beta 4
build `27A5228h`, 12 build jobs, warm shared SwiftPM repository caches, and an
otherwise clean scratch directory:

| Metric | Forced graph | Lean graph | Change |
| --- | ---: | ---: | ---: |
| Maximum reported build actions | 1,159 | 744 | -35.8% |
| Consumer dependency targets | 97 | 75 | -22.7% |
| Clean Release build time | 223.55 seconds | 185.46 seconds | -17.0% |
| Scratch directory | 1,988,360 KiB | 1,803,664 KiB | -9.3% |
| Release products | 1,539,608 KiB | 1,357,032 KiB | -11.9% |
| Dependency checkouts | 204,724 KiB | 204,724 KiB | unchanged |

Wall-clock results depend on machine load and repository-cache state. Reported
actions and the downstream target closure are the primary structural regression
signals. The script fails if `AFMKitMLXReleaseGraph` becomes reachable again.
Checkout size is unchanged by design: SwiftPM still resolves the exact manifest
pins, but it no longer compiles unrelated products from those packages.

## Qwen3.8 runtime regression check

The v0.1.4 baseline and lean graph were each built in a separate Release scratch
directory, then run against the same locally cached
`mlx-community/Qwen3.8-27B-4bit` snapshot. Both runs used native Metal kernels,
greedy decoding, seed 19, disabled prefix caching, disabled MTP, and the same
128-token limit. No model download occurred.

| Metric | v0.1.4 forced graph | Lean graph |
| --- | ---: | ---: |
| Prompt tokens | 40 | 40 |
| Completion tokens | 12 | 12 |
| First generation | 14.832 tok/s | 15.048 tok/s |
| Warm generation | 14.964 tok/s | 15.019 tok/s |

The warmed result changed by +0.4%, within normal run-to-run variation; no
generation-performance regression was observed. The AFMKitMLX public symbol
graph also matches its checked-in API baseline exactly.
