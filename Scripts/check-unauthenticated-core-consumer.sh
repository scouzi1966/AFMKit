#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${AFMKIT_BUILD_ROOT:-$ROOT/.build}"
mkdir -p "$BUILD_ROOT"
QUALIFICATION_ROOT="$(mktemp -d "$BUILD_ROOT/unauthenticated-core.XXXXXX")"

cleanup() {
    rm -rf "$QUALIFICATION_ROOT"
}
trap cleanup EXIT

HOME_DIR="$QUALIFICATION_ROOT/home"
CONSUMER="$QUALIFICATION_ROOT/consumer"
SCRATCH="$QUALIFICATION_ROOT/build"
mkdir -p "$HOME_DIR" "$CONSUMER/Sources/CoreConsumer"

cat > "$CONSUMER/Package.swift" <<SWIFT
// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "UnauthenticatedAFMKitCoreConsumer",
    platforms: [.macOS("26.0")],
    products: [
        .executable(name: "CoreConsumer", targets: ["CoreConsumer"])
    ],
    dependencies: [
        .package(name: "AFMKit", path: "$ROOT")
    ],
    targets: [
        .executableTarget(
            name: "CoreConsumer",
            dependencies: [.product(name: "AFMKitCore", package: "AFMKit")]
        )
    ]
)
SWIFT
cat > "$CONSUMER/Sources/CoreConsumer/main.swift" <<'SWIFT'
import AFMKitCore

let providerID: AFMProviderID = "consumer"
print("AFMKitCore provider: \(providerID)")
SWIFT

UNAUTHENTICATED_ENV=(
    env
    -u GH_TOKEN
    -u GITHUB_TOKEN
    HOME="$HOME_DIR"
    GIT_CONFIG_GLOBAL=/dev/null
    GIT_CONFIG_NOSYSTEM=1
    GIT_TERMINAL_PROMPT=0
)

"${UNAUTHENTICATED_ENV[@]}" /usr/bin/xcrun --toolchain XcodeDefault swift package dump-package \
    --package-path "$ROOT" \
    | /usr/bin/python3 -c '
import json
import sys

package = json.load(sys.stdin)
expected_vendored = {
    "mlx-swift": "/vendor/MLX/mlx-swift",
    "mlx-swift-lm": "/vendor/MLX/mlx-swift-lm",
}
seen_vendored = set()
for dependency in package.get("dependencies", []):
    file_system = dependency.get("fileSystem", [])
    if file_system:
        if len(file_system) != 1:
            raise SystemExit("AFMKit has an ambiguous vendored dependency.")
        entry = file_system[0]
        identity = entry.get("identity", "")
        expected_suffix = expected_vendored.get(identity)
        if expected_suffix is None or not entry.get("path", "").endswith(expected_suffix):
            raise SystemExit(f"AFMKit has an unexpected local dependency: {identity}")
        seen_vendored.add(identity)
        continue
    source_control = dependency.get("sourceControl", [])
    if len(source_control) != 1:
        raise SystemExit("AFMKit dependencies must be remote or approved vendored packages.")
    entry = source_control[0]
    locations = entry.get("location", {}).get("remote", [])
    if not locations or not all(item.get("urlString", "").startswith("https://") for item in locations):
        raise SystemExit("AFMKit dependencies must use public HTTPS locations.")
    if set(entry.get("requirement", {})) != {"exact"}:
        raise SystemExit("AFMKit dependencies must use exact versions.")
if seen_vendored != set(expected_vendored):
    raise SystemExit("AFMKit must expose both vendored MLX packages.")
products = {product.get("name") for product in package.get("products", [])}
expected = {
    "AFMKitCore", "AFMOpenAICompat", "AFMKitInference", "AFMKitApple",
    "AFMKitEmbeddings", "AFMKitSpeech", "AFMKitSpeechSynthesis",
    "AFMKitVision", "AFMKitServices", "AFMEvalKit", "AFMKitMLX",
    "AFMKitFoundationModelsMLX", "AFMKitDwarfStar",
    "AFMKitFoundationModelsDwarfStar",
}
if products != expected:
    raise SystemExit(f"Unexpected public package products: {sorted(products)}")
'

"${UNAUTHENTICATED_ENV[@]}" /usr/bin/xcrun --toolchain XcodeDefault swift package resolve \
    --package-path "$CONSUMER" \
    --scratch-path "$SCRATCH"
"${UNAUTHENTICATED_ENV[@]}" /usr/bin/xcrun --toolchain XcodeDefault swift build \
    --package-path "$CONSUMER" \
    --scratch-path "$SCRATCH" \
    --disable-automatic-resolution \
    --product CoreConsumer

echo "Clean unauthenticated single-repository AFMKitCore consumer passed."
