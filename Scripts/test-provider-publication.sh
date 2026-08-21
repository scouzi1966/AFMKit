#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${AFMKIT_BUILD_ROOT:-$ROOT/.build}"
mkdir -p "$BUILD_ROOT"
SANDBOX="$(mktemp -d "$BUILD_ROOT/provider-publication-tests.XXXXXX")"
TAG_NAME="v1.2.3-rc.1"
SOURCE_SHA="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
PUBLIC_URL="https://github.com/example/AFMKit.git"

cleanup() {
    find "$SANDBOX" -depth -delete
}
trap cleanup EXIT

mkdir -p "$SANDBOX/mirrors"
for package_name in AFMKitDwarfStar AFMKitMLX; do
    repository="$SANDBOX/source-$package_name"
    remote="$SANDBOX/remote-$package_name.git"
    mkdir -p "$repository"
    git -C "$repository" init -q
    git -C "$repository" config user.email afmkit-tests@example.invalid
    git -C "$repository" config user.name "AFMKit Tests"
    /usr/bin/python3 - \
        "$repository/AFMKit.release.json" "$package_name" "$PUBLIC_URL" "$SOURCE_SHA" <<'PY'
import json
import sys

path, package, public_url, source_sha = sys.argv[1:]
with open(path, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "package": package,
            "public_dependency": {"url": public_url, "version": "1.2.3-rc.1"},
            "source_sha": source_sha,
        },
        handle,
        indent=2,
        sort_keys=True,
    )
    handle.write("\n")
PY
    git -C "$repository" add AFMKit.release.json
    git -C "$repository" commit -q -m "$package_name fixture"
    git -C "$repository" tag "$TAG_NAME"
    revision="$(git -C "$repository" rev-parse HEAD)"
    case "$package_name" in
        AFMKitDwarfStar) DWARFSTAR_REVISION="$revision" ;;
        AFMKitMLX) MLX_REVISION="$revision" ;;
    esac
    git -C "$repository" bundle create \
        "$SANDBOX/mirrors/$package_name.bundle" --all
    git init -q --bare "$remote"
done

/usr/bin/python3 - \
    "$SANDBOX/mirrors/publication.json" "$TAG_NAME" "$SOURCE_SHA" \
    "$DWARFSTAR_REVISION" "$MLX_REVISION" <<'PY'
import json
import sys

path, tag, source_sha, dwarf_revision, mlx_revision = sys.argv[1:]
with open(path, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "packages": {
                "AFMKitDwarfStar": dwarf_revision,
                "AFMKitMLX": mlx_revision,
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

publish_fixture() {
    AFMKIT_PROVIDER_PUBLISH_TESTING=1 \
    AFMKIT_PUBLIC_RELEASE_URL="$PUBLIC_URL" \
    AFMKIT_DWARFSTAR_PUBLISH_URL="file://$SANDBOX/remote-AFMKitDwarfStar.git" \
    AFMKIT_MLX_PUBLISH_URL="file://$SANDBOX/remote-AFMKitMLX.git" \
        "$ROOT/Scripts/publish-provider-mirrors.sh" \
            "$SANDBOX/mirrors" "$TAG_NAME" "$SOURCE_SHA"
}

publish_partial_fixture() {
    AFMKIT_PROVIDER_PUBLISH_TESTING=1 \
    AFMKIT_PUBLIC_RELEASE_URL="$PUBLIC_URL" \
    AFMKIT_DWARFSTAR_PUBLISH_URL="file://$SANDBOX/remote-AFMKitDwarfStar.git" \
    AFMKIT_MLX_PUBLISH_URL="file://$SANDBOX/unavailable-AFMKitMLX.git" \
        "$ROOT/Scripts/publish-provider-mirrors.sh" \
            "$SANDBOX/mirrors" "$TAG_NAME" "$SOURCE_SHA"
}

if publish_partial_fixture > "$SANDBOX/partial.log" 2>&1; then
    echo "Provider publication regression failed: unavailable second remote succeeded." >&2
    exit 1
fi
test "$(git --git-dir="$SANDBOX/remote-AFMKitDwarfStar.git" \
    rev-parse "$TAG_NAME^{commit}")" = "$DWARFSTAR_REVISION"
if git --git-dir="$SANDBOX/remote-AFMKitMLX.git" \
    rev-parse "$TAG_NAME^{commit}" > /dev/null 2>&1; then
    echo "Provider publication regression failed: MLX tag exists after partial failure." >&2
    exit 1
fi

publish_fixture > "$SANDBOX/partial-recovery.log"
grep -q "AFMKitDwarfStar $TAG_NAME already published" "$SANDBOX/partial-recovery.log"
for package_name in AFMKitDwarfStar AFMKitMLX; do
    case "$package_name" in
        AFMKitDwarfStar) expected_revision="$DWARFSTAR_REVISION" ;;
        AFMKitMLX) expected_revision="$MLX_REVISION" ;;
    esac
    test "$(git --git-dir="$SANDBOX/remote-$package_name.git" \
        rev-parse "$TAG_NAME^{commit}")" = "$expected_revision"
done

publish_fixture > "$SANDBOX/idempotent.log"
grep -q "already published at the qualified revision" "$SANDBOX/idempotent.log"

/usr/bin/python3 - "$SANDBOX/mirrors/publication.json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    publication = json.load(handle)
publication["source_sha"] = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
with open(path, "w", encoding="utf-8") as handle:
    json.dump(publication, handle)
PY
if publish_fixture > "$SANDBOX/provenance.log" 2>&1; then
    echo "Provider publication regression failed: mismatched provenance was accepted." >&2
    exit 1
fi
grep -q "provenance does not match" "$SANDBOX/provenance.log"

echo "5 provider publication regression scenarios passed."
