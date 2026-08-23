#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODULE="${1:-AFMKitCore}"
BUILD_DIR="${AFMKIT_BUILD_ROOT:-$ROOT/.build}"
BASELINE_ROOT="${AFMKIT_API_BASELINE_ROOT:-$ROOT/docs/api-baselines}"
BASELINE="$BASELINE_ROOT/$MODULE.symbols.json"
CURRENT_DIR="$BUILD_DIR/api-current"
RAW_CURRENT_DIR="$BUILD_DIR/api-current-raw"
MODULE_CACHE="$BUILD_DIR/api-module-cache"
TOOLCHAIN_HELPER="$ROOT/Scripts/verify-qualified-toolchain.sh"
NORMALIZER="$ROOT/Scripts/normalize-symbol-graph.py"

if [[ -n "${AFMKIT_API_PACKAGE_ROOT:-}" ]]; then
    PACKAGE_ROOT="$AFMKIT_API_PACKAGE_ROOT"
    PACKAGE_BUILD_DIR="$BUILD_DIR"
else
case "$MODULE" in
    AFMKitCore|AFMOpenAICompat|AFMKitInference|AFMKitApple|AFMKitEmbeddings|AFMKitSpeech|AFMKitSpeechSynthesis|AFMKitVision|AFMKitServices|AFMEvalKit|AFMKitDwarfStar|AFMKitFoundationModelsDwarfStar|AFMKitMLX|AFMKitFoundationModelsMLX)
        PACKAGE_ROOT="$ROOT"
        PACKAGE_BUILD_DIR="$BUILD_DIR/public"
        ;;
    *)
        echo "Unknown public AFMKit module: $MODULE" >&2
        exit 64
        ;;
esac
fi

if [[ "${AFMKIT_API_SKIP_BUILD:-0}" != "0" ]]; then
    echo "AFMKIT_API_SKIP_BUILD is no longer supported." >&2
    echo "The API gate must rebuild the requested target so stale or unrelated .swiftmodule artifacts cannot be accepted." >&2
    exit 1
fi

if [[ ! -f "$TOOLCHAIN_HELPER" ]]; then
    echo "Missing qualified toolchain helper: $TOOLCHAIN_HELPER" >&2
    exit 1
fi

# shellcheck source=/dev/null
source "$TOOLCHAIN_HELPER"
afmkit_verify_qualified_toolchain "$ROOT"

SDK="$($AFMKIT_XCRUN_EXECUTABLE --sdk macosx --show-sdk-path)"
ARCH="$(uname -m)"

export SWIFTPM_MODULECACHE_OVERRIDE="$BUILD_DIR/swiftpm-module-cache"
export CLANG_MODULE_CACHE_PATH="$BUILD_DIR/clang-module-cache"

cd "$PACKAGE_ROOT"
afmkit_run_qualified_swift build \
    --package-path "$PACKAGE_ROOT" \
    --scratch-path "$PACKAGE_BUILD_DIR" \
    --build-system native \
    --disable-automatic-resolution \
    --target "$MODULE"
PRODUCTS_DIR="$(
    afmkit_run_qualified_swift build \
        --package-path "$PACKAGE_ROOT" \
        --scratch-path "$PACKAGE_BUILD_DIR" \
        --build-system native \
        --disable-automatic-resolution \
        --show-bin-path
)"

MODULE_SEARCH_DIR="$PRODUCTS_DIR"
if [[ -e "$PRODUCTS_DIR/Modules/$MODULE.swiftmodule" ]]; then
    MODULE_SEARCH_DIR="$PRODUCTS_DIR/Modules"
elif [[ ! -e "$PRODUCTS_DIR/$MODULE.swiftmodule" ]]; then
    echo "SwiftPM completed without producing a $MODULE.swiftmodule under $PRODUCTS_DIR" >&2
    exit 1
fi

# Dependency checkouts and generated module maps do not exist until the first
# clean build has completed. Discover them only after SwiftPM materializes them.
NUMERICS_SHIMS="$PACKAGE_BUILD_DIR/checkouts/swift-numerics/Sources/_NumericsShims/include"
ATOMICS_SHIMS="$PACKAGE_BUILD_DIR/checkouts/swift-atomics/Sources/_AtomicsShims/include"
SYSTEM_SHIMS="$PACKAGE_BUILD_DIR/checkouts/swift-system/Sources/CSystem/include"
NIO_WINDOWS="$PACKAGE_BUILD_DIR/checkouts/swift-nio/Sources/CNIOWindows/include"
MLX_SWIFT_ROOT="$ROOT/vendor/MLX/mlx-swift"
CMLX="$MLX_SWIFT_ROOT/Source/Cmlx/include"
AFM_XGRAMMAR="${AFMKIT_API_XGRAMMAR_ROOT:-$ROOT/Packages/AFMKitMLX/Sources/CXGrammar/include}"
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

GENERATED_MODULE_MAPS="$PACKAGE_BUILD_DIR/out/Intermediates.noindex/GeneratedModuleMaps"
for C_MODULE in CAsyncHTTPClient CDwarfStar CNIOAtomics CNIOBoringSSLShims CNIODarwin CNIOExtrasZlib CNIOFreeBSD CNIOLLHTTP CNIOLinux CNIOOpenBSD CNIOPosix CNIOWASI CNIOWindows yyjson; do
    MODULE_MAP="$PRODUCTS_DIR/$C_MODULE.build/module.modulemap"
    if [[ ! -f "$MODULE_MAP" ]]; then
        MODULE_MAP="$GENERATED_MODULE_MAPS/$C_MODULE.modulemap"
    fi
    [[ -f "$MODULE_MAP" ]] || continue
    EXTRACTOR_FLAGS+=(-Xcc "-fmodule-map-file=$MODULE_MAP")
done

rm -rf "$CURRENT_DIR" "$RAW_CURRENT_DIR" "$MODULE_CACHE"
mkdir -p "$CURRENT_DIR" "$RAW_CURRENT_DIR" "$MODULE_CACHE"

env -u TOOLCHAINS "$AFMKIT_SWIFT_SYMBOLGRAPH_EXECUTABLE" \
    -module-name "$MODULE" \
    -I "$MODULE_SEARCH_DIR" \
    -output-dir "$RAW_CURRENT_DIR" \
    -minimum-access-level public \
    -skip-synthesized-members \
    -skip-inherited-docs \
    -pretty-print \
    -sdk "$SDK" \
    -target "${ARCH}-apple-macos26.0" \
    -module-cache-path "$MODULE_CACHE" \
    "${EXTRACTOR_FLAGS[@]}"

/usr/bin/python3 "$NORMALIZER" \
    "$RAW_CURRENT_DIR/$MODULE.symbols.json" \
    "$CURRENT_DIR/$MODULE.symbols.json"

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
