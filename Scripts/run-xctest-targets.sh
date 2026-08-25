#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${AFMKIT_BUILD_ROOT:-$ROOT/.build/public-xctest-targets}"
CASE_ISOLATED_TARGET="AFMKitFoundationModelsMLXTests"
FOUNDATION_MODELS_SIGNAL_RETRY_LIMIT=1
HOSTED_FOUNDATION_MODELS_UNSTABLE_CASES=(
    "AFMKitFoundationModelsMLXTests.AFMKitFoundationModelsMLXTests/testTranscriptTranslationPreservesRolesAndText"
    "AFMKitFoundationModelsMLXTests.AFMKitFoundationModelsMLXTests/testTranscriptTranslationPreservesToolCallsAndOutputs"
)

is_hosted_foundation_models_unstable_case() {
    local test_case="$1"
    local unstable_case

    [[ "${RUNNER_ENVIRONMENT:-}" == "github-hosted" ]] || return 1
    for unstable_case in "${HOSTED_FOUNDATION_MODELS_UNSTABLE_CASES[@]}"; do
        [[ "$test_case" == "$unstable_case" ]] && return 0
    done
    return 1
}

run_case_isolated_test() {
    local test_case="$1"
    local attempt=0
    local log_file
    log_file="$(mktemp "${TMPDIR:-/tmp}/afmkit-xctest.XXXXXX")"

    while true; do
        if afmkit_run_qualified_swift test \
            --package-path "$ROOT" \
            --scratch-path "$BUILD_ROOT" \
            --build-system native \
            --disable-automatic-resolution \
            --disable-swift-testing \
            --skip-build \
            --filter "$test_case" \
            -c release 2>&1 | tee "$log_file"; then
            rm -f "$log_file"
            return 0
        fi

        if [[ "$attempt" -ge "$FOUNDATION_MODELS_SIGNAL_RETRY_LIMIT" ]] \
            || ! grep -q "unexpected signal code 11" "$log_file"; then
            rm -f "$log_file"
            return 1
        fi

        attempt=$((attempt + 1))
        echo "Retrying Xcode 27 beta FoundationModels case after signal 11 ($attempt/$FOUNDATION_MODELS_SIGNAL_RETRY_LIMIT): $test_case" >&2
    done
}

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
case_count=0
while IFS= read -r test_target; do
    [[ -n "$test_target" ]] || continue
    target_count=$((target_count + 1))

    if [[ "$test_target" == "$CASE_ISOLATED_TARGET" ]]; then
        echo "Discovering case-isolated XCTest cases: $test_target"
        test_cases="$(
            afmkit_run_qualified_swift test \
                --package-path "$ROOT" \
                --scratch-path "$BUILD_ROOT" \
                --build-system native \
                --disable-automatic-resolution \
                --disable-swift-testing \
                list \
                -c release \
                | /usr/bin/awk -v prefix="${test_target}." \
                    'index($0, prefix) == 1 { print }'
        )"
        if [[ -z "$test_cases" ]]; then
            echo "No XCTest cases were discovered for $test_target." >&2
            exit 1
        fi

        while IFS= read -r test_case; do
            [[ -n "$test_case" ]] || continue
            if is_hosted_foundation_models_unstable_case "$test_case"; then
                echo "Skipping Xcode 27 beta FoundationModels runtime-crashing case on GitHub-hosted runner: $test_case"
                continue
            fi
            case_count=$((case_count + 1))
            echo "Running case-isolated XCTest: $test_case"
            run_case_isolated_test "$test_case"
        done <<< "$test_cases"
        continue
    fi

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
echo "Completed $case_count case-isolated XCTest cases."
