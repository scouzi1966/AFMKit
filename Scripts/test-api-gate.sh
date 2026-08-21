#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKER="$ROOT/Scripts/check-afmkit-core-api.sh"
SANDBOX_ROOT="$ROOT/.build/api-gate-tests.$$"
PASSED=0

cleanup() {
    rm -rf "$SANDBOX_ROOT"
}
trap cleanup EXIT

fail() {
    echo "API gate regression failed: $1" >&2
    exit 1
}

copy_gate_fixture() {
    local destination="$1"
    mkdir -p "$destination/Scripts" "$destination/docs/api-baselines"
    cp "$CHECKER" "$destination/Scripts/check-afmkit-core-api.sh"
    cp "$ROOT/docs/api-baselines/toolchain.env" "$destination/docs/api-baselines/toolchain.env"
}

mkdir -p "$SANDBOX_ROOT"

SKIP_FIXTURE="$SANDBOX_ROOT/skip-build"
copy_gate_fixture "$SKIP_FIXTURE"
mkdir -p "$SKIP_FIXTURE/.build/out/Products/Debug/AFMKitCore.swiftmodule"
printf 'unrelated artifact\n' > "$SKIP_FIXTURE/.build/out/Products/Debug/AFMKitCore.swiftmodule/arm64.swiftmodule"
if AFMKIT_API_SKIP_BUILD=1 \
    "$SKIP_FIXTURE/Scripts/check-afmkit-core-api.sh" AFMKitCore \
    > "$SANDBOX_ROOT/skip-build.log" 2>&1; then
    fail "AFMKIT_API_SKIP_BUILD accepted a pre-existing .swiftmodule"
fi
grep -q "AFMKIT_API_SKIP_BUILD is no longer supported" "$SANDBOX_ROOT/skip-build.log" \
    || fail "skip-build rejection was not actionable"
PASSED=$((PASSED + 1))

MISMATCH_FIXTURE="$SANDBOX_ROOT/toolchain-mismatch"
copy_gate_fixture "$MISMATCH_FIXTURE"
sed 's/API_BASELINE_XCODE_BUILD=.*/API_BASELINE_XCODE_BUILD=not-the-current-build/' \
    "$ROOT/docs/api-baselines/toolchain.env" \
    > "$MISMATCH_FIXTURE/docs/api-baselines/toolchain.env"
if "$MISMATCH_FIXTURE/Scripts/check-afmkit-core-api.sh" AFMKitCore \
    > "$SANDBOX_ROOT/toolchain-mismatch.log" 2>&1; then
    fail "a mismatched API baseline toolchain was accepted"
fi
grep -q "API baseline toolchain mismatch" "$SANDBOX_ROOT/toolchain-mismatch.log" \
    || fail "toolchain mismatch did not explain the failure"
grep -q "Select Xcode 27 Beta 3" "$SANDBOX_ROOT/toolchain-mismatch.log" \
    || fail "toolchain mismatch did not provide an actionable selection"
PASSED=$((PASSED + 1))

CLEAN_FIXTURE="$SANDBOX_ROOT/clean-first-run"
FAKE_BIN="$SANDBOX_ROOT/fake-bin"
copy_gate_fixture "$CLEAN_FIXTURE"
mkdir -p "$FAKE_BIN"
printf '{}\n' > "$CLEAN_FIXTURE/docs/api-baselines/AFMKitFoundationModelsMLX.symbols.json"

cat > "$FAKE_BIN/xcodebuild" <<'SH'
#!/bin/bash
printf 'Xcode 27.0\nBuild version 27A5218g\n'
SH

cat > "$FAKE_BIN/swift" <<'SH'
#!/bin/bash
set -euo pipefail

if [[ "$1" == "build" && "$2" == "--target" ]]; then
    PRODUCTS="$FAKE_ROOT/.build/out/Products/Debug"
    mkdir -p "$PRODUCTS/$3.swiftmodule"
    SHIMS=(
        ".build/checkouts/swift-numerics/Sources/_NumericsShims/include"
        ".build/checkouts/swift-atomics/Sources/_AtomicsShims/include"
        ".build/checkouts/swift-system/Sources/CSystem/include"
        ".build/checkouts/swift-nio/Sources/CNIOWindows/include"
        ".build/checkouts/mlx-swift-afm/Source/Cmlx/include"
        "Sources/CXGrammar/include"
    )
    for SHIMS_DIR in "${SHIMS[@]}"; do
        mkdir -p "$FAKE_ROOT/$SHIMS_DIR"
        : > "$FAKE_ROOT/$SHIMS_DIR/module.modulemap"
    done
    exit 0
fi

if [[ "$1" == "build" && "$2" == "--show-bin-path" ]]; then
    printf '%s\n' "$FAKE_ROOT/.build/out/Products/Debug"
    exit 0
fi

exit 64
SH

cat > "$FAKE_BIN/xcrun" <<'SH'
#!/bin/bash
set -euo pipefail

if [[ "$1" == "--sdk" && "$2" == "macosx" ]]; then
    case "$3" in
        --show-sdk-version) printf '27.0\n' ;;
        --show-sdk-build-version) printf '26A5378i\n' ;;
        --show-sdk-path) printf '/fake/MacOSX.sdk\n' ;;
        *) exit 64 ;;
    esac
    exit 0
fi

if [[ "$1" != "swift-symbolgraph-extract" ]]; then
    exit 64
fi

NUMERICS="$FAKE_ROOT/.build/checkouts/swift-numerics/Sources/_NumericsShims/include"
SAW_INCLUDE=0
SAW_MODULE_MAP=0
OUTPUT_DIR=""
MODULE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        "$NUMERICS") SAW_INCLUDE=1 ;;
        "-fmodule-map-file=$NUMERICS/module.modulemap") SAW_MODULE_MAP=1 ;;
        -output-dir)
            shift
            OUTPUT_DIR="$1"
            ;;
        -module-name)
            shift
            MODULE="$1"
            ;;
    esac
    shift
done

[[ "$SAW_INCLUDE" == "1" ]] || exit 91
[[ "$SAW_MODULE_MAP" == "1" ]] || exit 92
mkdir -p "$OUTPUT_DIR"
printf '{}\n' > "$OUTPUT_DIR/$MODULE.symbols.json"
SH

chmod +x "$FAKE_BIN/xcodebuild" "$FAKE_BIN/swift" "$FAKE_BIN/xcrun"
PATH="$FAKE_BIN:$PATH" FAKE_ROOT="$CLEAN_FIXTURE" \
    "$CLEAN_FIXTURE/Scripts/check-afmkit-core-api.sh" AFMKitFoundationModelsMLX \
    > "$SANDBOX_ROOT/clean-first-run.log" 2>&1 \
    || fail "clean first-run extraction did not discover _NumericsShims after build"
grep -q "public API matches its checked-in baseline" "$SANDBOX_ROOT/clean-first-run.log" \
    || fail "clean first-run extraction did not complete"
PASSED=$((PASSED + 1))

echo "$PASSED API gate regression tests passed."
