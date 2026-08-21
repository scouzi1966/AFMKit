#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${AFMKIT_BUILD_ROOT:-$ROOT/.build}"
GENERATOR="$ROOT/Scripts/generate-downstream-package.py"

# shellcheck source=/dev/null
source "$ROOT/Scripts/release-qualification-guard.sh"
# shellcheck source=/dev/null
source "$ROOT/Scripts/verify-qualified-toolchain.sh"

afmkit_release_reject_local_overrides
afmkit_verify_qualified_toolchain "$ROOT"
mkdir -p "$BUILD_ROOT"
QUALIFICATION_ROOT="$(mktemp -d "$BUILD_ROOT/downstream-qualification.XXXXXX")"

cleanup() {
    if [[ "${AFMKIT_KEEP_DOWNSTREAM_BUILD:-0}" == "1" ]]; then
        echo "Preserved downstream qualification at $QUALIFICATION_ROOT"
    else
        rm -rf "$QUALIFICATION_ROOT"
    fi
}
trap cleanup EXIT

SOURCE_SHA="${AFMKIT_DOWNSTREAM_SHA:-$(git -C "$ROOT" rev-parse HEAD)}"
TAG_NAME="${AFMKIT_DOWNSTREAM_TAG:-v0.0.0-afmkit-qualification.${SOURCE_SHA:0:12}}"
afmkit_release_validate_tag "$TAG_NAME"
VERSION="${TAG_NAME#v}"
DEPENDENCY_URL="${AFMKIT_DOWNSTREAM_URL:-}"

if [[ -z "$DEPENDENCY_URL" ]]; then
    BARE_REMOTE="$QUALIFICATION_ROOT/AFMKit.git"
    git clone --quiet --bare "$ROOT" "$BARE_REMOTE"
    git --git-dir="$BARE_REMOTE" tag "$TAG_NAME" "$SOURCE_SHA"
    DEPENDENCY_URL="file://$BARE_REMOTE"
elif [[ "$DEPENDENCY_URL" != https://* ]]; then
    echo "Explicit downstream qualification requires an HTTPS AFMKit remote." >&2
    exit 1
fi

CONSUMER="$QUALIFICATION_ROOT/consumer"
CONSUMER_BUILD="$QUALIFICATION_ROOT/build"
mkdir -p "$CONSUMER"
afmkit_run_qualified_swift package dump-package --package-path "$ROOT" \
    | /usr/bin/python3 "$GENERATOR" "$CONSUMER" "$DEPENDENCY_URL" "$VERSION"

if [[ -e "$CONSUMER/Package.resolved" ]]; then
    echo "Generated downstream consumer unexpectedly inherited a lockfile." >&2
    exit 1
fi

afmkit_run_qualified_swift package resolve \
    --package-path "$CONSUMER" \
    --scratch-path "$CONSUMER_BUILD"

/usr/bin/python3 - \
    "$CONSUMER/Package.resolved" "$DEPENDENCY_URL" "$VERSION" "$SOURCE_SHA" <<'PY'
import json
import sys

path, expected_url, expected_version, expected_revision = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    resolved = json.load(handle)

pins = [pin for pin in resolved.get("pins", []) if pin.get("identity") == "afmkit"]
if len(pins) != 1:
    raise SystemExit(f"Downstream graph resolved {len(pins)} AFMKit pins instead of one.")

pin = pins[0]
state = pin.get("state", {})
if pin.get("location") != expected_url:
    raise SystemExit(f"Downstream AFMKit pin used {pin.get('location')} instead of {expected_url}.")
if state.get("version") != expected_version:
    raise SystemExit(
        f"Downstream AFMKit pin used version {state.get('version')} instead of {expected_version}."
    )
if state.get("revision") != expected_revision:
    raise SystemExit(
        f"Downstream AFMKit pin used revision {state.get('revision')} instead of {expected_revision}."
    )
PY

PRODUCT_COUNT=0
while IFS= read -r PRODUCT; do
    [[ -n "$PRODUCT" ]] || continue
    afmkit_run_qualified_swift build \
        --package-path "$CONSUMER" \
        --scratch-path "$CONSUMER_BUILD" \
        --build-system native \
        --disable-automatic-resolution \
        -c release \
        --product "$PRODUCT"
    PRODUCT_COUNT=$((PRODUCT_COUNT + 1))
done < "$CONSUMER/validator-products.txt"

EXPECTED_COUNT="$("$ROOT/Scripts/check-api-baseline-coverage.sh" --list | wc -l | tr -d ' ')"
if [[ "$PRODUCT_COUNT" != "$EXPECTED_COUNT" ]]; then
    echo "Downstream qualification built $PRODUCT_COUNT products; expected $EXPECTED_COUNT." >&2
    exit 1
fi

echo "AFMKit downstream remote/tag graph built all $PRODUCT_COUNT public library products."
