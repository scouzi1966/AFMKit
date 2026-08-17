#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODULE="${1:-AFMKitCore}"
BUILD_DIR="$ROOT/.build"
PRODUCTS_DIR="$BUILD_DIR/out/Products/Debug"
BASELINE="$ROOT/docs/api-baselines/$MODULE.symbols.json"
CURRENT_DIR="$BUILD_DIR/api-current"
RAW_CURRENT_DIR="$BUILD_DIR/api-current-raw"
MODULE_CACHE="$BUILD_DIR/api-module-cache"
SDK="$(xcrun --sdk macosx --show-sdk-path)"
ARCH="$(uname -m)"
NUMERICS_SHIMS="$BUILD_DIR/checkouts/swift-numerics/Sources/_NumericsShims/include"
ATOMICS_SHIMS="$BUILD_DIR/checkouts/swift-atomics/Sources/_AtomicsShims/include"
SYSTEM_SHIMS="$BUILD_DIR/checkouts/swift-system/Sources/CSystem/include"
NIO_WINDOWS="$BUILD_DIR/checkouts/swift-nio/Sources/CNIOWindows/include"
MLX_SWIFT_ROOT="${AFMKIT_MLX_SWIFT_PATH:-$BUILD_DIR/checkouts/mlx-swift-afm}"
CMLX="$MLX_SWIFT_ROOT/Source/Cmlx/include"
AFM_XGRAMMAR="$ROOT/Sources/CXGrammar/include"

EXTRACTOR_FLAGS=()
for SHIMS_DIR in "$NUMERICS_SHIMS" "$ATOMICS_SHIMS" "$SYSTEM_SHIMS" "$NIO_WINDOWS" "$CMLX" "$AFM_XGRAMMAR"; do
    [[ -f "$SHIMS_DIR/module.modulemap" ]] || continue
    EXTRACTOR_FLAGS+=(
        -Xcc "-fmodule-map-file=$SHIMS_DIR/module.modulemap"
        -Xcc "-I$SHIMS_DIR"
    )
done

GENERATED_MODULE_MAPS="$BUILD_DIR/out/Intermediates.noindex/GeneratedModuleMaps"
for C_MODULE in CAsyncHTTPClient CDwarfStar CNIOAtomics CNIOBoringSSLShims CNIODarwin CNIOExtrasZlib CNIOFreeBSD CNIOLLHTTP CNIOLinux CNIOOpenBSD CNIOPosix CNIOWASI CNIOWindows yyjson; do
    MODULE_MAP="$GENERATED_MODULE_MAPS/$C_MODULE.modulemap"
    [[ -f "$MODULE_MAP" ]] || continue
    EXTRACTOR_FLAGS+=(-Xcc "-fmodule-map-file=$MODULE_MAP")
done

export SWIFTPM_MODULECACHE_OVERRIDE="$BUILD_DIR/swiftpm-module-cache"
export CLANG_MODULE_CACHE_PATH="$BUILD_DIR/clang-module-cache"

cd "$ROOT"
swift build --target "$MODULE"

rm -rf "$CURRENT_DIR" "$RAW_CURRENT_DIR"
mkdir -p "$CURRENT_DIR" "$RAW_CURRENT_DIR" "$MODULE_CACHE"

xcrun swift-symbolgraph-extract \
    -module-name "$MODULE" \
    -I "$PRODUCTS_DIR" \
    -output-dir "$RAW_CURRENT_DIR" \
    -minimum-access-level public \
    -skip-synthesized-members \
    -skip-inherited-docs \
    -pretty-print \
    -sdk "$SDK" \
    -target "${ARCH}-apple-macos26.0" \
    -module-cache-path "$MODULE_CACHE" \
    "${EXTRACTOR_FLAGS[@]}"

python3 - "$RAW_CURRENT_DIR/$MODULE.symbols.json" "$CURRENT_DIR/$MODULE.symbols.json" <<'PY'
import json
import sys

raw_path, normalized_path = sys.argv[1:3]
VOLATILE_KEYS = {"generator", "location", "uri", "range"}


def normalize(value):
    if isinstance(value, dict):
        return {
            key: normalize(value[key])
            for key in sorted(value)
            if key not in VOLATILE_KEYS
        }
    if isinstance(value, list):
        normalized = [normalize(item) for item in value]
        if all(isinstance(item, dict) for item in normalized):
            return sorted(
                normalized,
                key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
            )
        return normalized
    return value


with open(raw_path, "r", encoding="utf-8") as handle:
    raw = json.load(handle)

with open(normalized_path, "w", encoding="utf-8") as handle:
    json.dump(normalize(raw), handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

if [[ ! -f "$BASELINE" ]]; then
    echo "$MODULE has no checked-in API baseline at $BASELINE" >&2
    echo "Review $CURRENT_DIR/$MODULE.symbols.json, then add it intentionally." >&2
    exit 1
fi

if ! cmp -s "$BASELINE" "$CURRENT_DIR/$MODULE.symbols.json"; then
    echo "$MODULE public API differs from $BASELINE" >&2
    echo "Review the API change, then replace the baseline intentionally." >&2
    diff -u "$BASELINE" "$CURRENT_DIR/$MODULE.symbols.json" || true
    exit 1
fi

echo "$MODULE public API matches its checked-in baseline."
