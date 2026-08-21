#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODULES=(
    AFMKitCore
    AFMOpenAICompat
    AFMKitApple
    AFMKitMLX
    AFMKitFoundationModelsMLX
    AFMKitDwarfStar
)

for MODULE in "${MODULES[@]}"; do
    "$ROOT/Scripts/check-afmkit-core-api.sh" "$MODULE"
done

echo "All ${#MODULES[@]} public API baselines match."
