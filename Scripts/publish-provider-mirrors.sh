#!/bin/bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: ${0##*/} mirror-directory tag source-sha" >&2
    exit 64
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIRROR_ROOT="$(cd "$1" && pwd)"
TAG_NAME="$2"
SOURCE_SHA="$3"
PUBLIC_RELEASE_URL="${AFMKIT_PUBLIC_RELEASE_URL:-}"
DWARFSTAR_URL="${AFMKIT_DWARFSTAR_PUBLISH_URL:-}"
MLX_URL="${AFMKIT_MLX_PUBLISH_URL:-}"

# shellcheck source=/dev/null
source "$ROOT/Scripts/release-qualification-guard.sh"
afmkit_release_validate_tag "$TAG_NAME"
if [[ ! "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Provider publication requires a full source commit SHA." >&2
    exit 1
fi
if [[ "$PUBLIC_RELEASE_URL" != https://* ]]; then
    echo "Provider publication requires the production HTTPS AFMKit URL." >&2
    exit 1
fi

if [[ "${AFMKIT_PROVIDER_PUBLISH_TESTING:-0}" != "1" ]]; then
    for remote_url in "$DWARFSTAR_URL" "$MLX_URL"; do
        if [[ ! "$remote_url" =~ ^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\.git$ ]]; then
            echo "Provider publication accepts only explicit HTTPS GitHub repository URLs." >&2
            exit 1
        fi
    done
    if [[ -z "${AFMKIT_PROVIDER_PUBLISH_TOKEN:-}" ]]; then
        echo "AFMKIT_PROVIDER_PUBLISH_TOKEN is required to publish provider packages." >&2
        exit 78
    fi
else
    for remote_url in "$DWARFSTAR_URL" "$MLX_URL"; do
        if [[ "$remote_url" != file://* ]]; then
            echo "Provider publication tests require file remotes." >&2
            exit 1
        fi
    done
fi

PUBLICATION="$MIRROR_ROOT/publication.json"
/usr/bin/python3 - "$PUBLICATION" "$TAG_NAME" "$SOURCE_SHA" <<'PY'
import json
import re
import sys

path, tag, source_sha = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    publication = json.load(handle)
if publication.get("tag") != tag or publication.get("source_sha") != source_sha:
    raise SystemExit("Provider publication provenance does not match the qualified release.")
packages = publication.get("packages")
if set(packages or {}) != {"AFMKitDwarfStar", "AFMKitMLX"}:
    raise SystemExit("Provider publication must contain exactly the two provider packages.")
for package, revision in packages.items():
    if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise SystemExit(f"Provider publication has an invalid {package} revision.")
PY

umask 077
TEMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/afmkit-provider-publish.XXXXXX")"
AUTH_CONFIG="$TEMP_ROOT/gitconfig"
: > "$AUTH_CONFIG"
cleanup() {
    find "$TEMP_ROOT" -depth -delete
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "${AFMKIT_PROVIDER_PUBLISH_TESTING:-0}" != "1" ]]; then
    /usr/bin/python3 - "$AUTH_CONFIG" "$DWARFSTAR_URL" "$MLX_URL" <<'PY'
import base64
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
header = base64.b64encode(
    f"x-access-token:{os.environ['AFMKIT_PROVIDER_PUBLISH_TOKEN']}".encode("utf-8")
).decode("ascii")
lines = []
for url in sys.argv[2:]:
    lines.extend([f'[http "{url}"]', f"\textraheader = AUTHORIZATION: basic {header}"])
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
fi
unset AFMKIT_PROVIDER_PUBLISH_TOKEN

git_publish() {
    GIT_CONFIG_NOSYSTEM=1 \
    GIT_CONFIG_GLOBAL="$AUTH_CONFIG" \
    GIT_TERMINAL_PROMPT=0 \
        git "$@"
}

publish_package() {
    local package_name="$1"
    local remote_url="$2"
    local bundle="$MIRROR_ROOT/$package_name.bundle"
    local repository="$TEMP_ROOT/$package_name.git"
    local expected_revision bundle_revision remote_refs

    expected_revision="$(/usr/bin/python3 - "$PUBLICATION" "$package_name" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["packages"][sys.argv[2]])
PY
)"
    test -f "$bundle"
    git -C "$ROOT" bundle verify "$bundle" >/dev/null
    bundle_revision="$(git bundle list-heads "$bundle" "refs/tags/$TAG_NAME" | awk '{print $1}')"
    if [[ "$bundle_revision" != "$expected_revision" ]]; then
        echo "$package_name bundle tag resolves to $bundle_revision, expected $expected_revision." >&2
        exit 1
    fi

    git clone --quiet --bare "$bundle" "$repository"
    if [[ "$(git --git-dir="$repository" rev-parse "$TAG_NAME^{commit}")" != "$expected_revision" ]]; then
        echo "$package_name cloned bundle did not preserve the qualified tag." >&2
        exit 1
    fi
    git --git-dir="$repository" show "$TAG_NAME:AFMKit.release.json" \
        > "$TEMP_ROOT/$package_name.release.json"
    /usr/bin/python3 - \
        "$TEMP_ROOT/$package_name.release.json" \
        "$package_name" "$PUBLIC_RELEASE_URL" "${TAG_NAME#v}" "$SOURCE_SHA" <<'PY'
import json
import sys

path, package, public_url, version, source_sha = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    provenance = json.load(handle)
expected = {
    "package": package,
    "public_dependency": {"url": public_url, "version": version},
    "source_sha": source_sha,
}
if provenance != expected:
    raise SystemExit(f"{package} release manifest does not match qualified provenance.")
PY

    remote_refs="$(git_publish ls-remote "$remote_url" \
        "refs/tags/$TAG_NAME" "refs/tags/$TAG_NAME^{}")"
    if [[ -n "$remote_refs" ]]; then
        /usr/bin/python3 - "$TAG_NAME" "$expected_revision" "$remote_refs" <<'PY'
import sys

tag, expected, output = sys.argv[1:]
refs = dict(line.split("\t", 1)[::-1] for line in output.splitlines())
actual = refs.get(f"refs/tags/{tag}^{{}}", refs.get(f"refs/tags/{tag}"))
if actual != expected:
    raise SystemExit(f"Remote tag {tag} resolves to {actual}, expected {expected}.")
PY
        echo "$package_name $TAG_NAME already published at the qualified revision."
        return
    fi

    git_publish --git-dir="$repository" push "$remote_url" \
        "refs/tags/$TAG_NAME:refs/tags/$TAG_NAME"
    remote_refs="$(git_publish ls-remote "$remote_url" \
        "refs/tags/$TAG_NAME" "refs/tags/$TAG_NAME^{}")"
    if ! grep -q "^$expected_revision[[:space:]]" <<< "$remote_refs"; then
        echo "$package_name remote did not expose the qualified tag after publication." >&2
        exit 1
    fi
    echo "Published $package_name $TAG_NAME at $expected_revision."
}

publish_package AFMKitDwarfStar "$DWARFSTAR_URL"
publish_package AFMKitMLX "$MLX_URL"
