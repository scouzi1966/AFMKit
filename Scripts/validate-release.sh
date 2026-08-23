#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${AFMKIT_BUILD_ROOT:-$ROOT/.build}"

# shellcheck source=/dev/null
source "$ROOT/Scripts/release-qualification-guard.sh"
# shellcheck source=/dev/null
source "$ROOT/Scripts/verify-qualified-toolchain.sh"

clean_build_configuration() {
    local package_build_root="$1"
    local configuration="$2"
    rm -rf \
        "$package_build_root/arm64-apple-macosx/$configuration" \
        "$package_build_root/$configuration"
}

run_release_tests() {
    local package_root="$1"
    local package_build_root="$2"

    clean_build_configuration "$package_build_root" release
    afmkit_run_qualified_swift test \
        --package-path "$package_root" \
        --scratch-path "$package_build_root" \
        --build-system native \
        --disable-automatic-resolution \
        --disable-swift-testing \
        -c release
    clean_build_configuration "$package_build_root" release
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

afmkit_verify_qualified_toolchain "$ROOT"
afmkit_run_qualified_swift package dump-package --package-path "$ROOT" \
    | afmkit_release_validate_manifest
afmkit_release_validate_resolved_files "$ROOT/Package.resolved"

"$ROOT/Scripts/test-api-gate.sh"
"$ROOT/Scripts/test-release-qualification.sh"
node "$ROOT/Scripts/test-release-publication.js"
"$ROOT/Scripts/test-workflow-security.sh"
"$ROOT/Scripts/check-dwarfstar-resources.sh"
"$ROOT/Scripts/check-unauthenticated-core-consumer.sh"

clean_build_configuration "$BUILD_ROOT/public" debug
"$ROOT/Scripts/check-api-baselines.sh"
clean_build_configuration "$BUILD_ROOT/public" debug
rm -rf \
    "$BUILD_ROOT/api-current" \
    "$BUILD_ROOT/api-current-raw" \
    "$BUILD_ROOT/api-module-cache" \
    "$BUILD_ROOT/swiftpm-module-cache" \
    "$BUILD_ROOT/clang-module-cache"

run_release_tests "$ROOT" "$BUILD_ROOT/public"

"$ROOT/Scripts/check-downstream-example.sh"

echo "AFMKit single-package release validation passed."
