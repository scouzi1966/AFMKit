#!/bin/bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: ${0##*/} command [args ...]" >&2
    exit 64
fi

if [[ -z "${AFMKIT_DEPENDENCY_TOKEN:-}" ]]; then
    echo "AFMKIT_DEPENDENCY_TOKEN is required for private AFM-compatible MLX dependencies." >&2
    echo "Token-independent validation should run separately; full package and API qualification cannot resolve the private graph without read access." >&2
    exit 78
fi

umask 077
AUTH_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/afmkit-git-auth.XXXXXX")"
AUTH_CONFIG="$AUTH_DIRECTORY/gitconfig"

cleanup() {
    rm -rf "$AUTH_DIRECTORY"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

git config --file "$AUTH_CONFIG" \
    "url.https://x-access-token:${AFMKIT_DEPENDENCY_TOKEN}@github.com/scouzi1966/mlx-swift-afm.insteadOf" \
    "https://github.com/scouzi1966/mlx-swift-afm"
git config --file "$AUTH_CONFIG" \
    "url.https://x-access-token:${AFMKIT_DEPENDENCY_TOKEN}@github.com/scouzi1966/mlx-swift-lm.git.insteadOf" \
    "https://github.com/scouzi1966/mlx-swift-lm.git"

unset AFMKIT_DEPENDENCY_TOKEN
GIT_CONFIG_GLOBAL="$AUTH_CONFIG" \
GIT_TERMINAL_PROMPT=0 \
    "$@"
