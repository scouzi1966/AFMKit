#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODULE_PARSER="$ROOT/Scripts/public-library-modules.py"

# shellcheck source=/dev/null
source "$ROOT/Scripts/verify-qualified-toolchain.sh"
afmkit_verify_qualified_toolchain "$ROOT"

MODULES=()
while IFS= read -r MODULE; do
    MODULES+=("$MODULE")
done < <(
    afmkit_run_qualified_swift package dump-package --package-path "$ROOT" \
        | /usr/bin/python3 "$MODULE_PARSER"
)

if [[ ${#MODULES[@]} -eq 0 ]]; then
    echo "No public library modules were discovered from Package.swift." >&2
    exit 1
fi

BASELINE_MODULES=()
while IFS= read -r BASELINE; do
    BASELINE_MODULES+=("$(basename "$BASELINE" .symbols.json)")
done < <(find "$ROOT/docs/api-baselines" -maxdepth 1 -name '*.symbols.json' -print | sort)

EXPECTED="$(printf '%s\n' "${MODULES[@]}" | sort -u)"
ACTUAL="$(printf '%s\n' "${BASELINE_MODULES[@]}" | sort -u)"
if [[ "$EXPECTED" != "$ACTUAL" ]]; then
    echo "Public library modules and API baselines are not one-to-one." >&2
    diff -u <(printf '%s\n' "$EXPECTED") <(printf '%s\n' "$ACTUAL") || true
    exit 1
fi

if [[ "${1:-}" == "--list" ]]; then
    printf '%s\n' "${MODULES[@]}"
elif [[ $# -ne 0 ]]; then
    echo "Usage: ${0##*/} [--list]" >&2
    exit 64
else
    echo "All ${#MODULES[@]} public library modules have exactly one API baseline."
fi
