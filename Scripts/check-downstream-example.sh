#!/bin/bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
example="$root/Examples/AFMKitQuickstart"

swift build \
    -c release \
    --package-path "$example" \
    "$@"

echo "AFMKit downstream example built successfully."
