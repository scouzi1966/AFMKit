#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${AFMKIT_BUILD_ROOT:-$ROOT/.build/public-xctest-targets}"

# shellcheck source=/dev/null
source "$ROOT/Scripts/verify-qualified-toolchain.sh"
afmkit_verify_qualified_toolchain "$ROOT"

test_targets="$(
    cd "$ROOT"
    afmkit_run_qualified_swift package describe --type json \
        | /usr/bin/python3 -c '
import json
import sys

package = json.load(sys.stdin)
for target in package["targets"]:
    if target.get("type") == "test":
        print(target["name"])
'
)"

if [[ -z "$test_targets" ]]; then
    echo "No XCTest targets were discovered in Package.swift." >&2
    exit 1
fi

target_count=0
while IFS= read -r test_target; do
    [[ -n "$test_target" ]] || continue
    target_count=$((target_count + 1))
    echo "Running isolated XCTest target: $test_target"
    afmkit_run_qualified_swift test \
        --package-path "$ROOT" \
        --scratch-path "$BUILD_ROOT" \
        --build-system native \
        --disable-automatic-resolution \
        --disable-swift-testing \
        --filter "$test_target" \
        -c release
done <<< "$test_targets"

echo "Completed $target_count isolated XCTest targets."
