#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${AFMKIT_BUILD_ROOT:-$ROOT/.build}"
GENERATOR="$ROOT/Scripts/generate-downstream-package.py"

source "$ROOT/Scripts/release-qualification-guard.sh"
source "$ROOT/Scripts/verify-qualified-toolchain.sh"

afmkit_verify_qualified_toolchain "$ROOT"
mkdir -p "$BUILD_ROOT"
QUALIFICATION_ROOT="$(mktemp -d "$BUILD_ROOT/downstream-qualification.XXXXXX")"
cleanup() {
    [[ "${AFMKIT_KEEP_DOWNSTREAM_BUILD:-0}" == "1" ]] \
        && echo "Preserved downstream qualification at $QUALIFICATION_ROOT" \
        || rm -rf "$QUALIFICATION_ROOT"
}
trap cleanup EXIT

SOURCE_SHA="${AFMKIT_DOWNSTREAM_SHA:-$(git -C "$ROOT" rev-parse HEAD)}"
TAG_NAME="${AFMKIT_DOWNSTREAM_TAG:-v0.0.0-afmkit-qualification.${SOURCE_SHA:0:12}}"
afmkit_release_validate_tag "$TAG_NAME"
VERSION="${TAG_NAME#v}"

ROOT_REMOTE="$QUALIFICATION_ROOT/AFMKit.git"
git clone --quiet --bare "$ROOT" "$ROOT_REMOTE"
if git --git-dir="$ROOT_REMOTE" rev-parse -q --verify "refs/tags/$TAG_NAME" >/dev/null; then
    EXISTING_TAG_SHA="$(git --git-dir="$ROOT_REMOTE" rev-parse "$TAG_NAME^{commit}")"
    [[ "$EXISTING_TAG_SHA" == "$SOURCE_SHA" ]] || {
        echo "Tag $TAG_NAME points to $EXISTING_TAG_SHA, not $SOURCE_SHA." >&2
        exit 1
    }
else
    git --git-dir="$ROOT_REMOTE" tag "$TAG_NAME" "$SOURCE_SHA"
fi

CONSUMER="$QUALIFICATION_ROOT/consumer"
CONSUMER_BUILD="$QUALIFICATION_ROOT/build"
mkdir -p "$CONSUMER"
afmkit_run_qualified_swift package dump-package --package-path "$ROOT" \
    | /usr/bin/python3 "$GENERATOR" "$CONSUMER" "file://$ROOT_REMOTE" "$VERSION" AFMKit
afmkit_run_qualified_swift package resolve --package-path "$CONSUMER" --scratch-path "$CONSUMER_BUILD"

/usr/bin/python3 - "$CONSUMER/Package.resolved" "$ROOT/Package.resolved" "$VERSION" "$SOURCE_SHA" <<'PY'
import json
import sys

actual_path, expected_path, version, revision = sys.argv[1:]
actual = json.load(open(actual_path, encoding="utf-8"))
expected = json.load(open(expected_path, encoding="utf-8"))
actual_pins = {pin["identity"]: pin for pin in actual["pins"]}
root = actual_pins.pop("afmkit", None)
if root is None or root["state"].get("version") != version or root["state"].get("revision") != revision:
    raise SystemExit("Fresh consumer did not resolve the exact qualified AFMKit tag.")

def normalized(pins):
    return {
        pin["identity"]: (
            pin.get("location"),
            pin.get("state", {}).get("version"),
            pin.get("state", {}).get("revision"),
        )
        for pin in pins
    }

if normalized(actual_pins.values()) != normalized(expected["pins"]):
    raise SystemExit("Fresh consumer dependency graph differs from the root release lock.")
PY

TOTAL=0
while IFS= read -r product; do
    [[ -n "$product" ]] || continue
    afmkit_run_qualified_swift build \
        --package-path "$CONSUMER" \
        --scratch-path "$CONSUMER_BUILD" \
        --build-system native \
        --disable-automatic-resolution \
        -c release \
        --product "$product"
    TOTAL=$((TOTAL + 1))
done < "$CONSUMER/validator-products.txt"

[[ "$TOTAL" == "15" ]] || {
    echo "Downstream qualification built $TOTAL products; expected 15." >&2
    exit 1
}
echo "AFMKit downstream qualification built all sixteen products from one exact tag."
