#!/bin/bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
example="$root/Examples/AFMKitQuickstart"

# shellcheck source=/dev/null
source "$root/Scripts/verify-qualified-toolchain.sh"
afmkit_verify_qualified_toolchain "$root"

afmkit_run_qualified_swift build \
    --build-system native \
    --disable-automatic-resolution \
    -c release \
    --package-path "$example" \
    "$@"

echo "AFMKit downstream example built successfully."
