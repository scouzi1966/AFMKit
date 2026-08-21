#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${AFMKIT_BUILD_ROOT:-$ROOT/.build}"
GENERATOR="$ROOT/Scripts/generate-downstream-package.py"
MATERIALIZER="$ROOT/Scripts/materialize-provider-package.py"

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
PUBLIC_RELEASE_URL="${AFMKIT_PUBLIC_RELEASE_URL:-$(git -C "$ROOT" remote get-url origin)}"
if [[ "$PUBLIC_RELEASE_URL" != https://* ]]; then
    echo "Provider release manifests require an HTTPS AFMKit public URL." >&2
    exit 1
fi

ROOT_REMOTE="$QUALIFICATION_ROOT/AFMKit.git"
git clone --quiet --bare "$ROOT" "$ROOT_REMOTE"
if git --git-dir="$ROOT_REMOTE" rev-parse -q --verify "refs/tags/$TAG_NAME" > /dev/null; then
    EXISTING_TAG_SHA="$(git --git-dir="$ROOT_REMOTE" rev-parse "$TAG_NAME^{commit}")"
    if [[ "$EXISTING_TAG_SHA" != "$SOURCE_SHA" ]]; then
        echo "Tag $TAG_NAME already points to $EXISTING_TAG_SHA, not $SOURCE_SHA." >&2
        exit 1
    fi
else
    git --git-dir="$ROOT_REMOTE" tag "$TAG_NAME" "$SOURCE_SHA"
fi
ROOT_URL="file://$ROOT_REMOTE"

SOURCE_DATE="$(git -C "$ROOT" show -s --format=%aI "$SOURCE_SHA")"
materialize_provider() {
    local package_name="$1"
    local source_directory="$ROOT/Packages/$package_name"
    local repository="$QUALIFICATION_ROOT/$package_name"
    local bare_repository="$QUALIFICATION_ROOT/$package_name.git"

    /usr/bin/python3 "$MATERIALIZER" \
        --source "$source_directory" \
        --output "$repository" \
        --public-url "$PUBLIC_RELEASE_URL" \
        --version "$VERSION" \
        --source-sha "$SOURCE_SHA"
    git -C "$repository" init -q
    git -C "$repository" config user.email afmkit-release@example.invalid
    git -C "$repository" config user.name "AFMKit Release"
    git -C "$repository" add -A
    GIT_AUTHOR_DATE="$SOURCE_DATE" GIT_COMMITTER_DATE="$SOURCE_DATE" \
        git -C "$repository" commit -q \
            -m "$package_name $TAG_NAME from AFMKit $SOURCE_SHA"
    git -C "$repository" tag "$TAG_NAME"
    git clone --quiet --bare "$repository" "$bare_repository"
}

materialize_provider AFMKitDwarfStar
materialize_provider AFMKitMLX

validate_resolved_graph() {
    local resolved_path="$1"
    local package_identity="$2"
    local expected_revision="$3"
    local expected_lock="${4:-}"

    /usr/bin/python3 - \
        "$resolved_path" "$package_identity" "$VERSION" \
        "$expected_revision" "$expected_lock" <<'PY'
import json
import sys

resolved_path, package_identity, version, revision, expected_lock = sys.argv[1:]
with open(resolved_path, encoding="utf-8") as handle:
    resolved = json.load(handle)
pins = {pin["identity"]: pin for pin in resolved.get("pins", [])}
pin_identity = package_identity.lower()
primary = pins.get(pin_identity)
if primary is None:
    raise SystemExit(f"Fresh graph omitted {package_identity}.")
state = primary.get("state", {})
if state.get("version") != version or state.get("revision") != revision:
    raise SystemExit(
        f"Fresh {package_identity} pin was {state}, expected {version} at {revision}."
    )

if not expected_lock:
    if set(pins) != {pin_identity}:
        raise SystemExit(f"Public package unexpectedly resolved {sorted(set(pins) - {pin_identity})}.")
    raise SystemExit(0)

with open(expected_lock, encoding="utf-8") as handle:
    expected = json.load(handle)
expected_pins = {
    pin["identity"]: (
        pin.get("location"),
        pin.get("state", {}).get("version"),
        pin.get("state", {}).get("revision"),
    )
    for pin in expected.get("pins", [])
}
actual_pins = {
    identity: (
        pin.get("location"),
        pin.get("state", {}).get("version"),
        pin.get("state", {}).get("revision"),
    )
    for identity, pin in pins.items()
    if identity not in {pin_identity, "afmkit"}
}
if actual_pins != expected_pins:
    missing = sorted(set(expected_pins) - set(actual_pins))
    extra = sorted(set(actual_pins) - set(expected_pins))
    changed = sorted(
        identity
        for identity in set(actual_pins) & set(expected_pins)
        if actual_pins[identity] != expected_pins[identity]
    )
    raise SystemExit(
        f"Fresh no-lock graph differs from release lock; "
        f"missing={missing}, extra={extra}, changed={changed}."
    )
PY
}

TOTAL_PRODUCT_COUNT=0
qualify_package() {
    local package_name="$1"
    local package_root="$2"
    local dependency_url="$3"
    local package_identity="$4"
    local expected_revision="$5"
    local expected_lock="${6:-}"
    local consumer="$QUALIFICATION_ROOT/consumer-$package_name"
    local consumer_build="$QUALIFICATION_ROOT/build-$package_name"

    mkdir -p "$consumer"
    afmkit_run_qualified_swift package dump-package --package-path "$package_root" \
        | /usr/bin/python3 "$GENERATOR" \
            "$consumer" "$dependency_url" "$VERSION" "$package_identity"
    if [[ -e "$consumer/Package.resolved" ]]; then
        echo "$package_name consumer unexpectedly inherited a lockfile." >&2
        exit 1
    fi
    if [[ "$package_name" != "AFMKit" ]]; then
        (
            cd "$consumer"
            afmkit_run_qualified_swift package config set-mirror \
                --original "$PUBLIC_RELEASE_URL" \
                --mirror "$ROOT_URL"
        )
    fi
    afmkit_run_qualified_swift package resolve \
        --package-path "$consumer" \
        --scratch-path "$consumer_build"
    validate_resolved_graph \
        "$consumer/Package.resolved" "$package_identity" \
        "$expected_revision" "$expected_lock"

    while IFS= read -r product; do
        [[ -n "$product" ]] || continue
        afmkit_run_qualified_swift build \
            --package-path "$consumer" \
            --scratch-path "$consumer_build" \
            --build-system native \
            --disable-automatic-resolution \
            -c release \
            --product "$product"
        TOTAL_PRODUCT_COUNT=$((TOTAL_PRODUCT_COUNT + 1))
    done < "$consumer/validator-products.txt"
}

ROOT_REVISION="$(git --git-dir="$ROOT_REMOTE" rev-parse "$TAG_NAME^{commit}")"
DWARF_REVISION="$(git --git-dir="$QUALIFICATION_ROOT/AFMKitDwarfStar.git" rev-parse "$TAG_NAME^{commit}")"
MLX_REVISION="$(git --git-dir="$QUALIFICATION_ROOT/AFMKitMLX.git" rev-parse "$TAG_NAME^{commit}")"

qualify_package AFMKit "$ROOT" "$ROOT_URL" AFMKit "$ROOT_REVISION"
qualify_package \
    AFMKitDwarfStar "$QUALIFICATION_ROOT/AFMKitDwarfStar" \
    "file://$QUALIFICATION_ROOT/AFMKitDwarfStar.git" AFMKitDwarfStar \
    "$DWARF_REVISION" "$ROOT/Packages/AFMKitDwarfStar/Package.resolved"
qualify_package \
    AFMKitMLX "$QUALIFICATION_ROOT/AFMKitMLX" \
    "file://$QUALIFICATION_ROOT/AFMKitMLX.git" AFMKitMLX \
    "$MLX_REVISION" "$ROOT/Packages/AFMKitMLX/Package.resolved"

if [[ "$TOTAL_PRODUCT_COUNT" != "6" ]]; then
    echo "Downstream qualification built $TOTAL_PRODUCT_COUNT products; expected 6." >&2
    exit 1
fi

if [[ -n "${AFMKIT_RELEASE_MIRROR_OUTPUT:-}" ]]; then
    mkdir -p "$AFMKIT_RELEASE_MIRROR_OUTPUT"
    git --git-dir="$QUALIFICATION_ROOT/AFMKitDwarfStar.git" bundle create \
        "$AFMKIT_RELEASE_MIRROR_OUTPUT/AFMKitDwarfStar.bundle" --all
    git --git-dir="$QUALIFICATION_ROOT/AFMKitMLX.git" bundle create \
        "$AFMKIT_RELEASE_MIRROR_OUTPUT/AFMKitMLX.bundle" --all
    /usr/bin/python3 - \
        "$AFMKIT_RELEASE_MIRROR_OUTPUT/publication.json" \
        "$TAG_NAME" "$SOURCE_SHA" "$DWARF_REVISION" "$MLX_REVISION" <<'PY'
import json
import sys

path, tag, source_sha, dwarf_sha, mlx_sha = sys.argv[1:]
with open(path, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "packages": {
                "AFMKitDwarfStar": dwarf_sha,
                "AFMKitMLX": mlx_sha,
            },
            "source_sha": source_sha,
            "tag": tag,
        },
        handle,
        indent=2,
        sort_keys=True,
    )
    handle.write("\n")
PY
fi

echo "AFMKit downstream staging mirrors built all six public products from fresh graphs."
