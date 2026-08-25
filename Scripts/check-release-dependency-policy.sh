#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${AFMKIT_BUILD_ROOT:-${RUNNER_TEMP:-/tmp}}"
mkdir -p "$BUILD_ROOT"
SANDBOX="$(mktemp -d "$BUILD_ROOT/afmkit-release-pins.XXXXXX")"
trap 'find "$SANDBOX" -depth -delete' EXIT

# shellcheck source=/dev/null
source "$ROOT/Scripts/verify-qualified-toolchain.sh"
afmkit_verify_qualified_toolchain "$ROOT"
afmkit_run_qualified_swift package dump-package --package-path "$ROOT" \
    > "$SANDBOX/dump-package.json"

/usr/bin/python3 "$ROOT/Scripts/check-release-dependency-policy.py" \
    "$SANDBOX/dump-package.json" \
    "$ROOT/Package.resolved"
