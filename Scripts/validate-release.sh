#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$ROOT/Scripts/test-api-gate.sh"
"$ROOT/Scripts/check-api-baselines.sh"
swift test --package-path "$ROOT" -c release --disable-swift-testing
"$ROOT/Scripts/check-downstream-example.sh"

echo "AFMKit release validation passed."
