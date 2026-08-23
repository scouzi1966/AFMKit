#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKER="$ROOT/Scripts/check-afmkit-core-api.sh"
COVERAGE_CHECKER="$ROOT/Scripts/check-api-baseline-coverage.sh"

MODULES=()
MODULE_OUTPUT="$("$COVERAGE_CHECKER" --list)"
while IFS= read -r MODULE; do
    MODULES+=("$MODULE")
done <<< "$MODULE_OUTPUT"

for MODULE in "${MODULES[@]}"; do
    "$CHECKER" "$MODULE"
done

echo "All ${#MODULES[@]} public API baselines match."
