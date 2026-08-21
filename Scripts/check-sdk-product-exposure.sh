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
"$SWIFT" package dump-package --package-path "$ROOT/Packages/AFMKitDwarfStar" \
    > "$SANDBOX/dwarfstar.json"
"$SWIFT" package dump-package --package-path "$ROOT/Packages/AFMKitMLX" \
    > "$SANDBOX/mlx.json"

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

documents = {
    name: json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
    for name in ("root", "dwarfstar", "mlx")
}
expected = {
    "xcode26": {
        "root": {"AFMKitCore", "AFMOpenAICompat"},
        "dwarfstar": {"AFMKitDwarfStar"},
        "mlx": {"AFMKitMLX"},
    },
    "xcode27": {
        "root": {"AFMKitCore", "AFMOpenAICompat", "AFMKitApple"},
        "dwarfstar": {"AFMKitDwarfStar"},
        "mlx": {"AFMKitMLX", "AFMKitFoundationModelsMLX"},
    },
}[mode]

for name, document in documents.items():
    products = {product["name"] for product in document.get("products", [])}
    if products != expected[name]:
        raise SystemExit(
            f"{mode} {name} products are {sorted(products)}, expected {sorted(expected[name])}."
        )
    platforms = document.get("platforms", [])
    if not any(
        platform.get("platformName") == "macos" and platform.get("version") == "26.0"
        for platform in platforms
    ):
        raise SystemExit(f"{name} does not preserve the macOS 26 deployment floor.")

targets = {
    name: {target["name"] for target in document.get("targets", [])}
    for name, document in documents.items()
}
macos27_targets = {"AFMKitApple", "AFMKitAppleTests"}
fmmlx_targets = {"AFMKitFoundationModelsMLX", "AFMKitFoundationModelsMLXTests"}
if mode == "xcode26":
    if targets["root"] & macos27_targets or targets["mlx"] & fmmlx_targets:
        raise SystemExit("Xcode 26 manifest exposes Xcode 27-only targets.")
else:
    if not macos27_targets.issubset(targets["root"]):
        raise SystemExit("Xcode 27 root manifest omits AFMKitApple targets.")
    if not fmmlx_targets.issubset(targets["mlx"]):
        raise SystemExit("Xcode 27 MLX manifest omits Foundation Models bridge targets.")

print(f"{mode} product exposure matches the macOS 26/27 compatibility contract.")
PY

if [[ "$MODE" == "xcode26" ]]; then
    "$SWIFT" build \
        --package-path "$ROOT" \
        --scratch-path "$SANDBOX/build" \
        --disable-automatic-resolution
fi
