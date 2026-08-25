#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${AFMKIT_BUILD_ROOT:-${RUNNER_TEMP:-/tmp}}"
mkdir -p "$BUILD_ROOT"
SANDBOX="$(mktemp -d "$BUILD_ROOT/afmkit-mlx-consumer.XXXXXX")"
KEEP_BUILD="${AFMKIT_KEEP_MLX_CONSUMER_BUILD:-0}"
BUILD_JOBS="${AFMKIT_MLX_CONSUMER_JOBS:-12}"
[[ "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]] || {
    echo "AFMKIT_MLX_CONSUMER_JOBS must be a positive integer." >&2
    exit 64
}

cleanup() {
    if [[ "$KEEP_BUILD" == "1" ]]; then
        echo "Preserved MLX consumer build at $SANDBOX"
    else
        find "$SANDBOX" -depth -delete
    fi
}
trap cleanup EXIT

# shellcheck source=/dev/null
source "$ROOT/Scripts/verify-qualified-toolchain.sh"
afmkit_verify_qualified_toolchain "$ROOT"

CONSUMER="$SANDBOX/consumer"
SCRATCH="$SANDBOX/build"
cp -R "$ROOT/Tests/Fixtures/AFMKitMLXConsumer" "$CONSUMER"
cp "$ROOT/Package.resolved" "$CONSUMER/Package.resolved"

START_NANOSECONDS="$(/usr/bin/python3 -c 'import time; print(time.time_ns())')"
set +e
AFMKIT_PACKAGE_PATH="$ROOT" afmkit_run_qualified_swift build \
    --package-path "$CONSUMER" \
    --scratch-path "$SCRATCH" \
    --build-system native \
    --disable-automatic-resolution \
    -c release \
    --product AFMKitMLXConsumer \
    --jobs "$BUILD_JOBS" \
    2>&1 | tee "$SANDBOX/build.log"
BUILD_STATUS="${PIPESTATUS[0]}"
set -e
[[ "$BUILD_STATUS" == "0" ]] || exit "$BUILD_STATUS"
END_NANOSECONDS="$(/usr/bin/python3 -c 'import time; print(time.time_ns())')"
ELAPSED_SECONDS="$(/usr/bin/python3 -c "print(($END_NANOSECONDS - $START_NANOSECONDS) / 1_000_000_000)")"

REPORT="$SANDBOX/consumer-build.json"
/usr/bin/python3 "$ROOT/Scripts/measure-mlx-consumer-build.py" \
    "$SCRATCH" \
    "$SANDBOX/build.log" \
    "$ELAPSED_SECONDS" \
    "$(git -C "$ROOT" rev-parse HEAD)" \
    "$BUILD_JOBS" \
    | tee "$REPORT"

if [[ -n "${AFMKIT_MLX_CONSUMER_REPORT:-}" ]]; then
    cp "$REPORT" "$AFMKIT_MLX_CONSUMER_REPORT"
fi
