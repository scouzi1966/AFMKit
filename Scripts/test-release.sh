#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METALLIB="$ROOT/Sources/AFMKitMLX/Resources/default.metallib"

cd "$ROOT"
swift test -c release list >/dev/null

test_bundles=0
while IFS= read -r -d '' bundle; do
  install -m 0644 "$METALLIB" "$bundle/Contents/MacOS/mlx.metallib"
  test_bundles=$((test_bundles + 1))
done < <(find "$ROOT/.build" -type d -path '*/Release/*.xctest' -print0)

if [[ "$test_bundles" -eq 0 ]]; then
  echo "No Release test bundles were produced." >&2
  exit 1
fi

swift test -c release --skip-build "$@"
