#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for module in \
  AFMKitCore \
  AFMOpenAICompat \
  AFMKitApple \
  AFMKitMLX \
  AFMKitDwarfStar; do
  "$ROOT/Scripts/check-afmkit-core-api.sh" "$module"
done

echo "All AFMKit provider API baselines match."
