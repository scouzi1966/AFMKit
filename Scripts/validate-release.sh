#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${AFMKIT_BUILD_ROOT:-$ROOT/.build}"

# shellcheck source=/dev/null
source "$ROOT/Scripts/release-qualification-guard.sh"
# shellcheck source=/dev/null
source "$ROOT/Scripts/verify-qualified-toolchain.sh"

clean_root_build_configuration() {
    local configuration="$1"
    rm -rf \
        "$BUILD_ROOT/arm64-apple-macosx/$configuration" \
        "$BUILD_ROOT/$configuration"
}

afmkit_release_begin_immutable_worktree "$ROOT"
qualification_exit() {
    local status=$?
    trap - EXIT
    if ! afmkit_release_verify_immutable_worktree "$ROOT"; then
        status=1
    fi
    exit "$status"
}
trap qualification_exit EXIT

afmkit_release_reject_local_overrides
afmkit_verify_qualified_toolchain "$ROOT"
afmkit_run_qualified_swift package dump-package --package-path "$ROOT" \
    | afmkit_release_validate_manifest
afmkit_release_validate_resolved_files "$ROOT/Package.resolved"

"$ROOT/Scripts/test-api-gate.sh"
"$ROOT/Scripts/test-private-dependency-auth.sh"
"$ROOT/Scripts/test-release-qualification.sh"
clean_root_build_configuration debug
"$ROOT/Scripts/check-api-baselines.sh"
clean_root_build_configuration debug
rm -rf \
    "$BUILD_ROOT/api-current" \
    "$BUILD_ROOT/api-current-raw" \
    "$BUILD_ROOT/api-module-cache" \
    "$BUILD_ROOT/swiftpm-module-cache" \
    "$BUILD_ROOT/clang-module-cache"
clean_root_build_configuration release
afmkit_run_qualified_swift test \
    --package-path "$ROOT" \
    --scratch-path "$BUILD_ROOT" \
    --build-system native \
    --disable-automatic-resolution \
    --disable-swift-testing \
    -c release
clean_root_build_configuration release
"$ROOT/Scripts/check-downstream-example.sh"

echo "AFMKit release validation passed."
