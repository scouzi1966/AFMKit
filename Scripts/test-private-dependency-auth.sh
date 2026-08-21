#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WRAPPER="$ROOT/Scripts/with-private-dependency-auth.sh"
SANDBOX="$ROOT/.build/private-auth-tests.$$"

cleanup() {
    rm -rf "$SANDBOX"
}
trap cleanup EXIT
mkdir -p "$SANDBOX/tmp"

if AFMKIT_DEPENDENCY_TOKEN= "$WRAPPER" true > "$SANDBOX/missing.log" 2>&1; then
    echo "Private auth regression failed: missing token was accepted." >&2
    exit 1
fi
grep -q "Token-independent validation should run separately" "$SANDBOX/missing.log" \
    || { echo "Private auth regression failed: missing-token guidance was not actionable." >&2; exit 1; }

set +e
TMPDIR="$SANDBOX/tmp" AFMKIT_DEPENDENCY_TOKEN="test-token-not-a-secret" \
    "$WRAPPER" bash -c '
        set -euo pipefail
        test -f "$GIT_CONFIG_GLOBAL"
        test -z "${AFMKIT_DEPENDENCY_TOKEN:-}"
        test "$(git config --global --get url.https://x-access-token:test-token-not-a-secret@github.com/.insteadof)" = "https://github.com/"
        exit 42
    '
STATUS=$?
set -e
[[ $STATUS -eq 42 ]] \
    || { echo "Private auth regression failed: wrapped failure status was not preserved." >&2; exit 1; }
if find "$SANDBOX/tmp" -mindepth 1 -print -quit | grep -q .; then
    echo "Private auth regression failed: temporary credential config survived a command failure." >&2
    exit 1
fi
if git config --global --get-regexp 'test-token-not-a-secret' >/dev/null 2>&1; then
    echo "Private auth regression failed: credential reached persistent global Git config." >&2
    exit 1
fi

echo "2 private dependency auth regression tests passed."
