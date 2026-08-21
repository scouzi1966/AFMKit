#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKER="$ROOT/Scripts/check-afmkit-core-api.sh"
AGGREGATE="$ROOT/Scripts/check-api-baselines.sh"
COVERAGE_CHECKER="$ROOT/Scripts/check-api-baseline-coverage.sh"
MODULE_PARSER="$ROOT/Scripts/public-library-modules.py"
TOOLCHAIN_HELPER="$ROOT/Scripts/verify-qualified-toolchain.sh"
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
    cp "$TOOLCHAIN_HELPER" "$destination/Scripts/verify-qualified-toolchain.sh"
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

AMBIENT_BIN="$SANDBOX_ROOT/ambient-bin"
AMBIENT_FIXTURE="$SANDBOX_ROOT/ambient-provenance"
mkdir -p "$AMBIENT_BIN"
mkdir -p "$AMBIENT_FIXTURE/Scripts" "$AMBIENT_FIXTURE/docs/api-baselines"
cp "$TOOLCHAIN_HELPER" "$AMBIENT_FIXTURE/Scripts/verify-qualified-toolchain.sh"
XCODE_VERSION_OUTPUT="$(/usr/bin/xcodebuild -version)"
QUALIFIED_SWIFT="$(/usr/bin/xcrun --toolchain XcodeDefault --find swift)"
{
    printf 'API_BASELINE_XCODE_VERSION=%q\n' "$(printf '%s\n' "$XCODE_VERSION_OUTPUT" | sed -n 's/^Xcode //p')"
    printf 'API_BASELINE_XCODE_BUILD=%q\n' "$(printf '%s\n' "$XCODE_VERSION_OUTPUT" | sed -n 's/^Build version //p')"
    printf 'API_BASELINE_MACOS_SDK_VERSION=%q\n' "$(/usr/bin/xcrun --sdk macosx --show-sdk-version)"
    printf 'API_BASELINE_MACOS_SDK_BUILD=%q\n' "$(/usr/bin/xcrun --sdk macosx --show-sdk-build-version)"
    printf 'API_BASELINE_SWIFT_VERSION=%q\n' "$($QUALIFIED_SWIFT --version | sed -n '1p')"
    printf 'API_BASELINE_SWIFT_EXECUTABLE_SHA256=%q\n' "$(/usr/bin/shasum -a 256 "$QUALIFIED_SWIFT" | awk '{print $1}')"
} > "$AMBIENT_FIXTURE/docs/api-baselines/toolchain.env"
cat > "$AMBIENT_BIN/swift" <<'SH'
#!/bin/bash
: > "$FAKE_SWIFT_MARKER"
exit 97
SH
chmod +x "$AMBIENT_BIN/swift"
PATH="$AMBIENT_BIN:$PATH" FAKE_SWIFT_MARKER="$SANDBOX_ROOT/ambient-swift-used" \
    "$AMBIENT_FIXTURE/Scripts/verify-qualified-toolchain.sh" \
    > "$SANDBOX_ROOT/qualified-toolchain.log" 2>&1 \
    || fail "qualified toolchain verification failed with an ambient swift earlier on PATH"
[[ ! -e "$SANDBOX_ROOT/ambient-swift-used" ]] \
    || fail "qualified toolchain verification invoked ambient PATH swift"
grep -q "/Toolchains/XcodeDefault.xctoolchain/usr/bin/swift" \
    "$SANDBOX_ROOT/qualified-toolchain.log" \
    || fail "qualified toolchain verification did not report XcodeDefault swift"
PASSED=$((PASSED + 1))

CLEAN_FIXTURE="$SANDBOX_ROOT/clean-first-run"
FAKE_BIN="$SANDBOX_ROOT/fake-bin"
copy_gate_fixture "$CLEAN_FIXTURE"
mkdir -p "$FAKE_BIN"
printf '{}\n' > "$CLEAN_FIXTURE/docs/api-baselines/AFMKitFoundationModelsMLX.symbols.json"

cat > "$CLEAN_FIXTURE/Scripts/verify-qualified-toolchain.sh" <<'SH'
#!/bin/bash
afmkit_verify_qualified_toolchain() {
    AFMKIT_XCRUN_EXECUTABLE="$FAKE_BIN/xcrun"
    AFMKIT_SWIFT_EXECUTABLE="$FAKE_BIN/swift"
    AFMKIT_SWIFT_SYMBOLGRAPH_EXECUTABLE="$FAKE_BIN/swift-symbolgraph-extract"
    export AFMKIT_XCRUN_EXECUTABLE AFMKIT_SWIFT_EXECUTABLE AFMKIT_SWIFT_SYMBOLGRAPH_EXECUTABLE
}
afmkit_run_qualified_swift() {
    "$AFMKIT_SWIFT_EXECUTABLE" "$@"
}
SH

cat > "$FAKE_BIN/swift" <<'SH'
#!/bin/bash
set -euo pipefail

MODULE=""
SHOW_BIN_PATH=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)
            shift
            MODULE="$1"
            ;;
        --show-bin-path) SHOW_BIN_PATH=1 ;;
    esac
    shift
done

PRODUCTS="$FAKE_ROOT/.build/out/Products/Debug"
if [[ "$SHOW_BIN_PATH" == "1" ]]; then
    printf '%s\n' "$PRODUCTS"
    exit 0
fi

[[ -n "$MODULE" ]] || exit 64
mkdir -p "$PRODUCTS/$MODULE.swiftmodule"
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
SH

cat > "$FAKE_BIN/xcrun" <<'SH'
#!/bin/bash
if [[ "$1" == "--sdk" && "$2" == "macosx" && "$3" == "--show-sdk-path" ]]; then
    printf '/fake/MacOSX.sdk\n'
    exit 0
fi
exit 64
SH

cat > "$FAKE_BIN/swift-symbolgraph-extract" <<'SH'
#!/bin/bash
set -euo pipefail

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

chmod +x "$CLEAN_FIXTURE/Scripts/verify-qualified-toolchain.sh" "$FAKE_BIN"/*
FAKE_ROOT="$CLEAN_FIXTURE" FAKE_BIN="$FAKE_BIN" \
    "$CLEAN_FIXTURE/Scripts/check-afmkit-core-api.sh" AFMKitFoundationModelsMLX \
    > "$SANDBOX_ROOT/clean-first-run.log" 2>&1 \
    || fail "clean first-run extraction did not discover _NumericsShims after build"
grep -q "public API matches its checked-in baseline" "$SANDBOX_ROOT/clean-first-run.log" \
    || fail "clean first-run extraction did not complete"
PASSED=$((PASSED + 1))

AGGREGATE_FIXTURE="$SANDBOX_ROOT/aggregate"
mkdir -p "$AGGREGATE_FIXTURE/Scripts" "$AGGREGATE_FIXTURE/docs/api-baselines"
cp "$AGGREGATE" "$AGGREGATE_FIXTURE/Scripts/check-api-baselines.sh"
cp "$COVERAGE_CHECKER" "$AGGREGATE_FIXTURE/Scripts/check-api-baseline-coverage.sh"
cp "$MODULE_PARSER" "$AGGREGATE_FIXTURE/Scripts/public-library-modules.py"
cat > "$AGGREGATE_FIXTURE/Scripts/check-afmkit-core-api.sh" <<'SH'
#!/bin/bash
printf '%s\n' "$1" >> "$FAKE_CHECKER_CALLS"
SH
cat > "$AGGREGATE_FIXTURE/Package.swift" <<'SWIFT'
// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "APIGateFixture",
    products: [
        .library(name: "AFMKitCore", targets: ["AFMKitCore"]),
        .library(name: "AFMOpenAICompat", targets: ["AFMOpenAICompat"]),
        .library(name: "AFMKitApple", targets: ["AFMKitApple"]),
        .library(name: "AFMKitMLX", targets: ["AFMKitMLX"]),
        .library(name: "AFMKitFoundationModelsMLX", targets: ["AFMKitFoundationModelsMLX"]),
        .library(name: "AFMKitDwarfStar", targets: ["AFMKitDwarfStar"]),
        .executable(name: "AFMKitTool", targets: ["AFMKitTool"]),
    ],
    targets: [
        .target(name: "AFMKitCore"),
        .target(name: "AFMOpenAICompat"),
        .target(name: "AFMKitApple"),
        .target(name: "AFMKitMLX"),
        .target(name: "AFMKitFoundationModelsMLX"),
        .target(name: "AFMKitDwarfStar"),
        .executableTarget(name: "AFMKitTool"),
    ]
)
SWIFT
for MODULE in AFMKitCore AFMOpenAICompat AFMKitApple AFMKitMLX AFMKitFoundationModelsMLX AFMKitDwarfStar; do
    mkdir -p "$AGGREGATE_FIXTURE/Sources/$MODULE"
    printf '// fixture\n' > "$AGGREGATE_FIXTURE/Sources/$MODULE/Fixture.swift"
    printf '{}\n' > "$AGGREGATE_FIXTURE/docs/api-baselines/$MODULE.symbols.json"
done
mkdir -p "$AGGREGATE_FIXTURE/Sources/AFMKitTool"
printf 'print("fixture")\n' > "$AGGREGATE_FIXTURE/Sources/AFMKitTool/main.swift"
chmod +x "$AGGREGATE_FIXTURE/Scripts"/*
FAKE_CHECKER_CALLS="$AGGREGATE_FIXTURE/checker-calls" \
    "$AGGREGATE_FIXTURE/Scripts/check-api-baselines.sh" \
    > "$SANDBOX_ROOT/aggregate.log" 2>&1 \
    || fail "aggregate API gate rejected the complete public library product set"
EXPECTED_MODULES="$(
    /usr/bin/xcrun --toolchain XcodeDefault swift package dump-package \
        --package-path "$AGGREGATE_FIXTURE" \
        | /usr/bin/python3 "$MODULE_PARSER"
)"
ACTUAL_MODULES="$(sort -u "$AGGREGATE_FIXTURE/checker-calls")"
[[ "$EXPECTED_MODULES" == "$ACTUAL_MODULES" ]] \
    || fail "aggregate API gate did not invoke every public library module"
grep -qx "AFMKitApple" "$AGGREGATE_FIXTURE/checker-calls" \
    || fail "aggregate API gate omitted AFMKitApple"
grep -qx "AFMKitDwarfStar" "$AGGREGATE_FIXTURE/checker-calls" \
    || fail "aggregate API gate omitted AFMKitDwarfStar"
PASSED=$((PASSED + 1))

/usr/bin/python3 - "$AGGREGATE_FIXTURE/Package.swift" <<'PY'
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    package = handle.read()
package = package.replace(
    '        .executable(name: "AFMKitTool", targets: ["AFMKitTool"]),',
    '        .library(name: "AFMKitFuture", targets: ["AFMKitFuture"]),\n'
    '        .executable(name: "AFMKitTool", targets: ["AFMKitTool"]),',
)
package = package.replace(
    '        .executableTarget(name: "AFMKitTool"),',
    '        .target(name: "AFMKitFuture"),\n'
    '        .executableTarget(name: "AFMKitTool"),',
)
with open(path, "w", encoding="utf-8") as handle:
    handle.write(package)
PY
mkdir -p "$AGGREGATE_FIXTURE/Sources/AFMKitFuture"
printf '// fixture\n' > "$AGGREGATE_FIXTURE/Sources/AFMKitFuture/Fixture.swift"
rm -f "$AGGREGATE_FIXTURE/checker-calls"
if FAKE_CHECKER_CALLS="$AGGREGATE_FIXTURE/checker-calls" \
    "$AGGREGATE_FIXTURE/Scripts/check-api-baselines.sh" \
    > "$SANDBOX_ROOT/missing-public-baseline.log" 2>&1; then
    fail "aggregate API gate accepted a public module with no baseline"
fi
grep -q "Public library modules and API baselines are not one-to-one" \
    "$SANDBOX_ROOT/missing-public-baseline.log" \
    || fail "missing public module baseline failure was not actionable"
[[ ! -e "$AGGREGATE_FIXTURE/checker-calls" ]] \
    || fail "aggregate API gate ran partial comparisons before validating coverage"
PASSED=$((PASSED + 1))

echo "$PASSED API gate regression tests passed."
