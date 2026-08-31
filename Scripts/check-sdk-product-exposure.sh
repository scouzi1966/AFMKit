#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^xcode(26|27)$ ]]; then
    echo "Usage: ${0##*/} xcode26|xcode27" >&2
    exit 64
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="$1"
BUILD_ROOT="${AFMKIT_BUILD_ROOT:-${RUNNER_TEMP:-/tmp}}"
mkdir -p "$BUILD_ROOT"
SANDBOX="$(mktemp -d "$BUILD_ROOT/afmkit-sdk-exposure.XXXXXX")"

cleanup() {
    find "$SANDBOX" -depth -delete
}
trap cleanup EXIT

SWIFT="$(/usr/bin/xcrun --toolchain XcodeDefault --find swift)"
"$SWIFT" --version > "$SANDBOX/swift-version.txt"
"$SWIFT" package dump-package --package-path "$ROOT" \
    > "$SANDBOX/root.json"

/usr/bin/python3 - "$MODE" "$SANDBOX" <<'PY'
import json
import pathlib
import re
import sys

mode = sys.argv[1]
root = pathlib.Path(sys.argv[2])
version_text = (root / "swift-version.txt").read_text(encoding="utf-8")
match = re.search(r"Apple Swift version (\d+)\.(\d+)", version_text)
if not match:
    raise SystemExit(f"Could not parse selected Swift compiler: {version_text.strip()}")
compiler = tuple(map(int, match.groups()))
if mode == "xcode26" and compiler >= (6, 4):
    raise SystemExit(f"Xcode 26 gate selected unexpected Swift {compiler}.")
if mode == "xcode27" and compiler < (6, 4):
    raise SystemExit(f"Xcode 27 gate requires Swift 6.4 or newer, found {compiler}.")

document = json.loads((root / "root.json").read_text(encoding="utf-8"))
expected = {
    "xcode26": {"AFMKitCore", "AFMOpenAICompat", "AFMKitInference", "AFMKitEmbeddings", "AFMKitSpeech", "AFMKitSpeechSynthesis", "AFMKitVision", "AFMKitServices", "AFMEvalKit", "AFMKitDwarfStar", "AFMKitMLX", "AFMKitMLXAudio", "AFMKitMLXImage"},
    "xcode27": {"AFMKitCore", "AFMOpenAICompat", "AFMKitInference", "AFMKitEmbeddings", "AFMKitSpeech", "AFMKitSpeechSynthesis", "AFMKitVision", "AFMKitServices", "AFMEvalKit", "AFMKitApple", "AFMKitDwarfStar", "AFMKitFoundationModelsDwarfStar", "AFMKitMLX", "AFMKitMLXAudio", "AFMKitMLXImage", "AFMKitFoundationModelsMLX"},
}[mode]

products = {product["name"] for product in document.get("products", [])}
if products != expected:
    raise SystemExit(f"{mode} products are {sorted(products)}, expected {sorted(expected)}.")
platforms = document.get("platforms", [])
if not any(
    platform.get("platformName") == "macos" and platform.get("version") == "26.0"
    for platform in platforms
):
    raise SystemExit("The root package does not preserve the macOS 26 deployment floor.")
if not any(
    platform.get("platformName") == "ios" and platform.get("version") == "16.0"
    for platform in platforms
):
    raise SystemExit("The root package does not declare the iOS 16 deployment floor.")

targets = {target["name"] for target in document.get("targets", [])}
macos27_targets = {"AFMKitApple", "AFMKitAppleTests"}
fmmlx_targets = {"AFMKitFoundationModelsMLX", "AFMKitFoundationModelsMLXTests"}
fmdwarf_targets = {
    "AFMKitFoundationModelsDwarfStar",
    "AFMKitFoundationModelsDwarfStarTests",
}
if mode == "xcode26":
    if targets & (macos27_targets | fmmlx_targets | fmdwarf_targets):
        raise SystemExit("Xcode 26 manifest exposes Xcode 27-only targets.")
else:
    if not macos27_targets.issubset(targets):
        raise SystemExit("Xcode 27 root manifest omits AFMKitApple targets.")
    if not fmmlx_targets.issubset(targets) or not fmdwarf_targets.issubset(targets):
        raise SystemExit("Xcode 27 root manifest omits provider Foundation Models bridge targets.")

print(f"{mode} product exposure matches the macOS and basic iOS compatibility contract.")
PY

if [[ "$MODE" == "xcode26" ]]; then
    "$SWIFT" build \
        --package-path "$ROOT" \
        --scratch-path "$SANDBOX/build" \
        --disable-automatic-resolution
fi
