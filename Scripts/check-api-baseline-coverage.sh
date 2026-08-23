#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODULE_PARSER="$ROOT/Scripts/public-library-modules.py"

MODULES=()
PACKAGE_ROOTS=(
    "$ROOT"
)
for OPTIONAL_PACKAGE_ROOT in \
    "$ROOT/Packages/AFMKitDwarfStar" \
    "$ROOT/Packages/AFMKitMLX"; do
    [[ ! -f "$OPTIONAL_PACKAGE_ROOT/Package.swift" ]] \
        || PACKAGE_ROOTS+=("$OPTIONAL_PACKAGE_ROOT")
done
for PACKAGE_ROOT in "${PACKAGE_ROOTS[@]}"; do
    while IFS= read -r MODULE; do
        MODULES+=("$MODULE")
    done < <(
        /usr/bin/xcrun --toolchain XcodeDefault swift package dump-package \
            --package-path "$PACKAGE_ROOT" \
            | /usr/bin/python3 "$MODULE_PARSER"
    )
done

if [[ ${#MODULES[@]} -eq 0 ]]; then
    echo "No public library modules were discovered from the package manifests." >&2
    exit 1
fi

BASELINE_MODULES=()
while IFS= read -r BASELINE; do
    BASELINE_MODULES+=("$(basename "$BASELINE" .symbols.json)")
done < <(find "$ROOT/docs/api-baselines" -maxdepth 1 -name '*.symbols.json' -print | sort)

UNIQUE_MODULE_COUNT="$(printf '%s\n' "${MODULES[@]}" | sort -u | wc -l | tr -d ' ')"
if [[ "$UNIQUE_MODULE_COUNT" != "${#MODULES[@]}" ]]; then
    echo "A public library module is exposed by more than one package manifest." >&2
    printf '%s\n' "${MODULES[@]}" | sort | uniq -d >&2
    exit 1
fi

EXPECTED="$(printf '%s\n' "${MODULES[@]}" | sort)"
ACTUAL="$(printf '%s\n' "${BASELINE_MODULES[@]}" | sort -u)"
if [[ "$EXPECTED" != "$ACTUAL" ]]; then
    echo "Public library modules and API baselines are not one-to-one." >&2
    diff -u <(printf '%s\n' "$EXPECTED") <(printf '%s\n' "$ACTUAL") || true
    exit 1
fi

if [[ "${1:-}" == "--list" ]]; then
    printf '%s\n' "${MODULES[@]}" | sort
elif [[ $# -ne 0 ]]; then
    echo "Usage: ${0##*/} [--list]" >&2
    exit 64
else
    echo "All ${#MODULES[@]} public library modules have exactly one API baseline."
fi
