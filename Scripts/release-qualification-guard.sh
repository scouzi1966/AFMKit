#!/bin/bash

afmkit_release_validate_tag() {
    local tag_name="${1:-}"

    /usr/bin/python3 - "$tag_name" <<'PY'
import re
import sys

tag = sys.argv[1]
numeric = r"(?:0|[1-9][0-9]*)"
alphanumeric = r"(?:[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
prerelease_identifier = rf"(?:{numeric}|{alphanumeric})"
build_identifier = r"[0-9A-Za-z-]+"
pattern = re.compile(
    rf"^v{numeric}\.{numeric}\.{numeric}"
    rf"(?:-{prerelease_identifier}(?:\.{prerelease_identifier})*)?"
    rf"(?:\+{build_identifier}(?:\.{build_identifier})*)?$"
)

if not pattern.fullmatch(tag):
    raise SystemExit(
        "Release tag must be strict SemVer with a v prefix "
        "(for example, v1.2.3-rc.1+build.5)."
    )
PY
}

afmkit_release_reject_local_overrides() {
    local variable
    for variable in AFMKIT_MLX_SWIFT_PATH AFMKIT_MLX_SWIFT_LM_PATH; do
        if [[ -n "${!variable:-}" ]]; then
            echo "Release qualification rejects local dependency override $variable." >&2
            return 1
        fi
    done
}

afmkit_release_validate_manifest() {
    /usr/bin/python3 -c '
import json
import sys

package = json.load(sys.stdin)
for dependency in package.get("dependencies", []):
    source_control = dependency.get("sourceControl")
    if not source_control:
        raise SystemExit("Release qualification requires every root dependency to use remote source control.")
    for entry in source_control:
        locations = entry.get("location", {}).get("remote")
        if not locations or not all(item.get("urlString", "").startswith("https://") for item in locations):
            identity = entry.get("identity", "unknown")
            raise SystemExit(f"Release qualification rejected non-remote dependency {identity}.")
'
}

afmkit_release_validate_resolved_files() {
    local root="$1"
    /usr/bin/python3 - "$root/Package.resolved" "$root/Examples/AFMKitQuickstart/Package.resolved" <<'PY'
import json
import re
import sys

for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as handle:
        resolved = json.load(handle)
    origin_hash = resolved.get("originHash", "")
    if not re.fullmatch(r"[0-9a-f]{64}", origin_hash):
        raise SystemExit(f"{path} has no valid originHash.")
    pins = resolved.get("pins", [])
    if not pins:
        raise SystemExit(f"{path} contains no dependency pins.")
    identities = set()
    for pin in pins:
        identity = pin.get("identity", "")
        if identity in identities:
            raise SystemExit(f"{path} contains duplicate pin {identity}.")
        identities.add(identity)
        if pin.get("kind") != "remoteSourceControl":
            raise SystemExit(f"{path} contains non-remote pin {identity}.")
        if not pin.get("location", "").startswith("https://"):
            raise SystemExit(f"{path} contains non-HTTPS pin {identity}.")
        if not re.fullmatch(r"[0-9a-f]{40}", pin.get("state", {}).get("revision", "")):
            raise SystemExit(f"{path} contains unpinned dependency {identity}.")
PY
}

afmkit_release_begin_immutable_worktree() {
    local root="$1"
    local status

    status="$(git -C "$root" status --porcelain=v1 --untracked-files=all)"
    if [[ -n "$status" ]]; then
        echo "Release qualification requires a clean worktree." >&2
        printf '%s\n' "$status" >&2
        return 1
    fi

    AFMKIT_RELEASE_START_HEAD="$(git -C "$root" rev-parse HEAD)"
}

afmkit_release_verify_immutable_worktree() {
    local root="$1"
    local status current_head

    current_head="$(git -C "$root" rev-parse HEAD)"
    status="$(git -C "$root" status --porcelain=v1 --untracked-files=all)"
    if [[ "$current_head" != "${AFMKIT_RELEASE_START_HEAD:-}" ]] || [[ -n "$status" ]]; then
        echo "Release qualification mutated Package.resolved or the worktree." >&2
        [[ -z "$status" ]] || printf '%s\n' "$status" >&2
        return 1
    fi
}
