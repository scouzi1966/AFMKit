#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${AFMKIT_BUILD_ROOT:-$ROOT/.build}"
mkdir -p "$BUILD_ROOT"
SANDBOX="$(mktemp -d "$BUILD_ROOT/service-products.XXXXXX")"

cleanup() {
    find "$SANDBOX" -depth -delete
}
trap cleanup EXIT

FEATURE_MODULES=(AFMKitEmbeddings AFMKitSpeech AFMKitSpeechSynthesis AFMKitVision)
PRODUCTS=("${FEATURE_MODULES[@]}" AFMKitServices)

for PRODUCT in "${PRODUCTS[@]}"; do
    CONSUMER="$SANDBOX/$PRODUCT"
    SCRATCH="$SANDBOX/build-$PRODUCT"
    mkdir -p "$CONSUMER/Sources/Consumer"
    cat > "$CONSUMER/Package.swift" <<SWIFT
// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "${PRODUCT}IsolationConsumer",
    platforms: [.macOS("26.0")],
    dependencies: [.package(name: "AFMKit", path: "$ROOT")],
    targets: [
        .executableTarget(
            name: "Consumer",
            dependencies: [.product(name: "$PRODUCT", package: "AFMKit")]
        )
    ]
)
SWIFT
    cat > "$CONSUMER/Sources/Consumer/main.swift" <<SWIFT
import $PRODUCT
print("$PRODUCT loaded")
SWIFT

    /usr/bin/xcrun --toolchain XcodeDefault swift build \
        --package-path "$CONSUMER" \
        --scratch-path "$SCRATCH" \
        --disable-automatic-resolution \
        --product Consumer

    find "$SCRATCH" -name "$PRODUCT.swiftmodule" -print -quit | grep -q . || {
        echo "$PRODUCT consumer did not compile its selected module." >&2
        exit 1
    }

    if [[ "$PRODUCT" != "AFMKitServices" ]]; then
        for OTHER in "${FEATURE_MODULES[@]}"; do
            [[ "$OTHER" == "$PRODUCT" ]] && continue
            if find "$SCRATCH" -name "$OTHER.swiftmodule" -print -quit | grep -q .; then
                echo "$PRODUCT consumer unexpectedly compiled $OTHER." >&2
                exit 1
            fi
        done
    fi
done

echo "Every Apple service product builds independently; the umbrella builds all four."
