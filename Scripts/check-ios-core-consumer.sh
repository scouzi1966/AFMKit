#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${AFMKIT_BUILD_ROOT:-${RUNNER_TEMP:-/tmp}}"
JOBS="${AFMKIT_IOS_CORE_JOBS:-12}"

if [[ ! "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "AFMKIT_IOS_CORE_JOBS must be a positive integer." >&2
    exit 64
fi

mkdir -p "$BUILD_ROOT"
SANDBOX="$(mktemp -d "$BUILD_ROOT/afmkit-ios-core.XXXXXX")"
cleanup() {
    find "$SANDBOX" -depth -delete
}
trap cleanup EXIT

# shellcheck source=/dev/null
source "$ROOT/Scripts/verify-qualified-toolchain.sh"
afmkit_verify_qualified_toolchain "$ROOT"

IOS_SDK="$($AFMKIT_XCRUN_EXECUTABLE --sdk iphonesimulator --show-sdk-path)"
IOS_SDK_VERSION="$($AFMKIT_XCRUN_EXECUTABLE --sdk iphonesimulator --show-sdk-version)"
CONSUMER="$SANDBOX/consumer"
SCRATCH="$SANDBOX/build"

cp -R "$ROOT/Tests/Fixtures/AFMKitIOSCoreConsumer" "$CONSUMER"
cp "$ROOT/Package.resolved" "$CONSUMER/Package.resolved"

AFMKIT_PACKAGE_PATH="$ROOT" afmkit_run_qualified_swift build \
    --package-path "$CONSUMER" \
    --scratch-path "$SCRATCH" \
    --build-system native \
    --disable-automatic-resolution \
    --triple arm64-apple-ios16.0-simulator \
    --sdk "$IOS_SDK" \
    --target AFMKitIOSCoreConsumer \
    --jobs "$JOBS"

for module in AFMKitCore AFMOpenAICompat AFMKitInference AFMKitIOSCoreConsumer; do
    if [[ -z "$(find "$SCRATCH" -name "$module.swiftmodule" -print -quit)" ]]; then
        echo "The iOS consumer build did not emit $module.swiftmodule." >&2
        exit 1
    fi
done

unsupported_modules=(
    AFMKitApple
    AFMKitDwarfStar
    AFMKitEmbeddings
    AFMKitFoundationModelsDwarfStar
    AFMKitFoundationModelsMLX
    AFMKitMLX
    AFMKitMLXAudio
    AFMKitServices
    AFMKitSpeech
    AFMKitSpeechSynthesis
    AFMKitVision
)
for module in "${unsupported_modules[@]}"; do
    if [[ -n "$(find "$SCRATCH" -name "$module.swiftmodule" -print -quit)" ]]; then
        echo "The basic iOS consumer unexpectedly compiled unsupported module $module." >&2
        exit 1
    fi
done

printf 'Basic iOS consumer passed (arm64 simulator, iOS 16 target, iOS SDK %s).\n' \
    "$IOS_SDK_VERSION"
