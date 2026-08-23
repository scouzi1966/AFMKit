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

afmkit_release_reject_local_overrides
afmkit_verify_qualified_toolchain "$ROOT"
afmkit_run_qualified_swift package dump-package --package-path "$ROOT" \
    | afmkit_release_validate_manifest
for provider_root in \
    "$ROOT/Packages/AFMKitDwarfStar" \
    "$ROOT/Packages/AFMKitMLX"; do
    afmkit_run_qualified_swift package dump-package --package-path "$provider_root" \
        | AFMKIT_ALLOW_LOCAL_PUBLIC_PACKAGE=1 afmkit_release_validate_manifest
done
afmkit_release_validate_resolved_files \
    "$ROOT/Packages/AFMKitDwarfStar/Package.resolved" \
    "$ROOT/Packages/AFMKitMLX/Package.resolved"

"$ROOT/Scripts/test-api-gate.sh"
"$ROOT/Scripts/test-private-dependency-auth.sh"
"$ROOT/Scripts/test-release-qualification.sh"
node "$ROOT/Scripts/test-release-publication.js"
"$ROOT/Scripts/test-provider-publication.sh"
"$ROOT/Scripts/test-workflow-security.sh"
"$ROOT/Scripts/check-dwarfstar-resources.sh"
"$ROOT/Scripts/check-unauthenticated-core-consumer.sh"

for package_build_root in \
    "$BUILD_ROOT/public" \
    "$BUILD_ROOT/dwarfstar" \
    "$BUILD_ROOT/mlx"; do
    clean_build_configuration "$package_build_root" debug
done
"$ROOT/Scripts/check-api-baselines.sh"
for package_build_root in \
    "$BUILD_ROOT/public" \
    "$BUILD_ROOT/dwarfstar" \
    "$BUILD_ROOT/mlx"; do
    clean_build_configuration "$package_build_root" debug
done
rm -rf \
    "$BUILD_ROOT/api-current" \
    "$BUILD_ROOT/api-current-raw" \
    "$BUILD_ROOT/api-module-cache" \
    "$BUILD_ROOT/swiftpm-module-cache" \
    "$BUILD_ROOT/clang-module-cache"

run_release_tests "$ROOT" "$BUILD_ROOT/public"
run_release_tests "$ROOT/Packages/AFMKitDwarfStar" "$BUILD_ROOT/dwarfstar"
run_release_tests "$ROOT/Packages/AFMKitMLX" "$BUILD_ROOT/mlx"

"$ROOT/Scripts/check-downstream-example.sh"

echo "AFMKit split-package release validation passed."
