#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -ne 1 ]]; then
    echo "Usage: ${0##*/} vMAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]" >&2
    exit 64
fi

# shellcheck source=/dev/null
source "$ROOT/Scripts/release-qualification-guard.sh"
afmkit_release_validate_tag "$1"
