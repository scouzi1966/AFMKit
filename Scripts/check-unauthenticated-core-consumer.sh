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
    -u AFMKIT_DEPENDENCY_TOKEN
    -u AFMKIT_MLX_SWIFT_PATH
    -u AFMKIT_MLX_SWIFT_LM_PATH
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
if package.get("dependencies"):
    raise SystemExit("The public AFMKit package must not resolve external dependencies.")
products = {product.get("name") for product in package.get("products", [])}
expected = {
    "AFMKitCore", "AFMOpenAICompat", "AFMKitInference", "AFMKitApple",
    "AFMKitEmbeddings", "AFMKitSpeech", "AFMKitSpeechSynthesis",
    "AFMKitVision", "AFMKitServices", "AFMEvalKit",
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

if find "$QUALIFICATION_ROOT" -iname '*mlx*' -print -quit | grep -q .; then
    echo "Core-only consumer unexpectedly materialized an MLX path." >&2
    exit 1
fi

echo "Clean unauthenticated AFMKitCore consumer passed."
