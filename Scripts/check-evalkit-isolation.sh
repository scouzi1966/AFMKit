#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${AFMKIT_BUILD_ROOT:-${RUNNER_TEMP:-/tmp}}"
mkdir -p "$BUILD_ROOT"
SANDBOX="$(mktemp -d "$BUILD_ROOT/evalkit-isolation.XXXXXX")"
trap 'find "$SANDBOX" -depth -delete' EXIT

if grep -ERn 'import (AFMKitCore|AFMKitInference|AFMKitApple|AFMKitMLX|AFMKitDwarfStar|AFMKitServices)' \
    "$ROOT/Sources/AFMEvalKit"; then
    echo "AFMEvalKit imports a provider or service module." >&2
    exit 1
fi

mkdir -p "$SANDBOX/Sources/Consumer"
cat > "$SANDBOX/Package.swift" <<EOF
// swift-tools-version: 6.1
import PackageDescription
let package = Package(
    name: "EvalKitIsolation",
    platforms: [.macOS("26.0")],
    dependencies: [.package(name: "AFMKit", path: "$ROOT")],
    targets: [
        .executableTarget(
            name: "Consumer",
            dependencies: [.product(name: "AFMEvalKit", package: "AFMKit")])
    ])
EOF
cat > "$SANDBOX/Sources/Consumer/main.swift" <<'EOF'
import AFMEvalKit
let suite = AFMEvaluationSuite(
    name: "isolation",
    description: "Provider-free consumer.",
    cases: [.init(id: "one", prompt: "one")])
try AFMEvaluationValidator.validate(suite)
print(suite.name)
EOF

SCRATCH="$SANDBOX/build"
cp "$ROOT/Package.resolved" "$SANDBOX/Package.resolved"
/usr/bin/xcrun --toolchain XcodeDefault swift build \
    --package-path "$SANDBOX" \
    --scratch-path "$SCRATCH" \
    --disable-automatic-resolution \
    --product Consumer

for forbidden in AFMKitCore AFMKitInference AFMKitApple AFMKitEmbeddings \
    AFMKitSpeech AFMKitSpeechSynthesis AFMKitVision AFMKitServices; do
    if find "$SCRATCH" -name "$forbidden.swiftmodule" -print -quit | grep -q .; then
        echo "AFMEvalKit consumer compiled forbidden module $forbidden." >&2
        exit 1
    fi
done

find "$SCRATCH" -name AFMEvalKit.swiftmodule -print -quit | grep -q . || {
    echo "AFMEvalKit consumer did not compile AFMEvalKit." >&2
    exit 1
}
find "$SCRATCH" -name AFMOpenAICompat.swiftmodule -print -quit | grep -q . || {
    echo "AFMEvalKit consumer did not compile its sole AFMOpenAICompat dependency." >&2
    exit 1
}

echo "AFMEvalKit builds with only AFMOpenAICompat and Foundation."
