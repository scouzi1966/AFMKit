#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODULE="${1:-AFMKitCore}"
BUILD_DIR="$ROOT/.build"
BASELINE="$ROOT/docs/api-baselines/$MODULE.symbols.json"
TOOLCHAIN_PROVENANCE="$ROOT/docs/api-baselines/toolchain.env"
CURRENT_DIR="$BUILD_DIR/api-current"
RAW_CURRENT_DIR="$BUILD_DIR/api-current-raw"
MODULE_CACHE="$BUILD_DIR/api-module-cache"

if [[ "${AFMKIT_API_SKIP_BUILD:-0}" != "0" ]]; then
    echo "AFMKIT_API_SKIP_BUILD is no longer supported." >&2
    echo "The API gate must rebuild the requested target so stale or unrelated .swiftmodule artifacts cannot be accepted." >&2
    exit 1
fi

if [[ ! -f "$TOOLCHAIN_PROVENANCE" ]]; then
    echo "Missing API baseline toolchain provenance: $TOOLCHAIN_PROVENANCE" >&2
    exit 1
fi

# shellcheck source=/dev/null
source "$TOOLCHAIN_PROVENANCE"
for REQUIRED_VARIABLE in \
    API_BASELINE_XCODE_VERSION \
    API_BASELINE_XCODE_BUILD \
    API_BASELINE_MACOS_SDK_VERSION \
    API_BASELINE_MACOS_SDK_BUILD; do
    if [[ -z "${!REQUIRED_VARIABLE:-}" ]]; then
        echo "Missing $REQUIRED_VARIABLE in $TOOLCHAIN_PROVENANCE" >&2
        exit 1
    fi
done

XCODE_VERSION_OUTPUT="$(xcodebuild -version)"
ACTUAL_XCODE_VERSION="$(printf '%s\n' "$XCODE_VERSION_OUTPUT" | sed -n 's/^Xcode //p')"
ACTUAL_XCODE_BUILD="$(printf '%s\n' "$XCODE_VERSION_OUTPUT" | sed -n 's/^Build version //p')"
ACTUAL_SDK_VERSION="$(xcrun --sdk macosx --show-sdk-version)"
ACTUAL_SDK_BUILD="$(xcrun --sdk macosx --show-sdk-build-version)"

if [[ "$ACTUAL_XCODE_VERSION" != "$API_BASELINE_XCODE_VERSION" ]] || \
   [[ "$ACTUAL_XCODE_BUILD" != "$API_BASELINE_XCODE_BUILD" ]] || \
   [[ "$ACTUAL_SDK_VERSION" != "$API_BASELINE_MACOS_SDK_VERSION" ]] || \
   [[ "$ACTUAL_SDK_BUILD" != "$API_BASELINE_MACOS_SDK_BUILD" ]]; then
    cat >&2 <<EOF
API baseline toolchain mismatch.
Required: Xcode $API_BASELINE_XCODE_VERSION ($API_BASELINE_XCODE_BUILD), macOS SDK $API_BASELINE_MACOS_SDK_VERSION ($API_BASELINE_MACOS_SDK_BUILD)
Current:  Xcode ${ACTUAL_XCODE_VERSION:-unknown} (${ACTUAL_XCODE_BUILD:-unknown}), macOS SDK ${ACTUAL_SDK_VERSION:-unknown} (${ACTUAL_SDK_BUILD:-unknown})
Select Xcode 27 Beta 3, for example:
  export DEVELOPER_DIR=/Applications/Xcode_27_beta_3.app/Contents/Developer
Then rerun the gate. To qualify a different toolchain, intentionally regenerate and review every API baseline and update:
  $TOOLCHAIN_PROVENANCE
EOF
    exit 1
fi

SDK="$(xcrun --sdk macosx --show-sdk-path)"
ARCH="$(uname -m)"

export SWIFTPM_MODULECACHE_OVERRIDE="$BUILD_DIR/swiftpm-module-cache"
export CLANG_MODULE_CACHE_PATH="$BUILD_DIR/clang-module-cache"

cd "$ROOT"
swift build --target "$MODULE"
PRODUCTS_DIR="$(swift build --show-bin-path)"

if [[ ! -d "$PRODUCTS_DIR/$MODULE.swiftmodule" ]]; then
    echo "SwiftPM completed without producing $PRODUCTS_DIR/$MODULE.swiftmodule" >&2
    exit 1
fi

# Dependency checkouts and generated module maps do not exist until the first
# clean build has completed. Discover them only after SwiftPM materializes them.
NUMERICS_SHIMS="$BUILD_DIR/checkouts/swift-numerics/Sources/_NumericsShims/include"
ATOMICS_SHIMS="$BUILD_DIR/checkouts/swift-atomics/Sources/_AtomicsShims/include"
SYSTEM_SHIMS="$BUILD_DIR/checkouts/swift-system/Sources/CSystem/include"
NIO_WINDOWS="$BUILD_DIR/checkouts/swift-nio/Sources/CNIOWindows/include"
MLX_SWIFT_ROOT="${AFMKIT_MLX_SWIFT_PATH:-$BUILD_DIR/checkouts/mlx-swift-afm}"
CMLX="$MLX_SWIFT_ROOT/Source/Cmlx/include"
AFM_XGRAMMAR="$ROOT/Sources/CXGrammar/include"
SHIMS_DIRS=(
    "$NUMERICS_SHIMS"
    "$ATOMICS_SHIMS"
    "$SYSTEM_SHIMS"
    "$NIO_WINDOWS"
    "$CMLX"
    "$AFM_XGRAMMAR"
)

case "$MODULE" in
    AFMKitMLX|AFMKitFoundationModelsMLX)
        for SHIMS_DIR in "${SHIMS_DIRS[@]}"; do
            if [[ ! -f "$SHIMS_DIR/module.modulemap" ]]; then
                echo "$MODULE API extraction requires $SHIMS_DIR/module.modulemap after its SwiftPM build." >&2
                exit 1
            fi
        done
        ;;
esac

EXTRACTOR_FLAGS=()
for SHIMS_DIR in "${SHIMS_DIRS[@]}"; do
    [[ -f "$SHIMS_DIR/module.modulemap" ]] || continue
    EXTRACTOR_FLAGS+=(
        -I "$SHIMS_DIR"
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

rm -rf "$CURRENT_DIR" "$RAW_CURRENT_DIR" "$MODULE_CACHE"
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
