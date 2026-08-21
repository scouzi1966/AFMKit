#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GUARD="$ROOT/Scripts/release-qualification-guard.sh"
BUILD_ROOT="${AFMKIT_BUILD_ROOT:-$ROOT/.build}"
TAG_VALIDATOR="$ROOT/Scripts/validate-release-tag.sh"
SANDBOX="$BUILD_ROOT/release-qualification-tests.$$"
PASSED=0

cleanup() {
    rm -rf "$SANDBOX"
}
trap cleanup EXIT
mkdir -p "$SANDBOX"

# shellcheck source=/dev/null
source "$GUARD"

if AFMKIT_MLX_SWIFT_PATH=/tmp/local-mlx \
    afmkit_release_reject_local_overrides > "$SANDBOX/override.log" 2>&1; then
    echo "Release qualification regression failed: local override was accepted." >&2
    exit 1
fi
grep -q "rejects local dependency override" "$SANDBOX/override.log"
PASSED=$((PASSED + 1))

VALID_TAGS=(
    v0.0.0
    v1.2.3
    v1.2.3-alpha
    v1.2.3-alpha.1
    v1.2.3-0.3.7
    v1.2.3-x.7.z.92
    v1.2.3+build.5-e
    v1.2.3-rc.1+build.001
)
for TAG in "${VALID_TAGS[@]}"; do
    "$TAG_VALIDATOR" "$TAG" \
        || { echo "Release qualification regression failed: valid tag $TAG was rejected." >&2; exit 1; }
done

INVALID_TAGS=(
    1.2.3
    v01.2.3
    v1.02.3
    v1.2.03
    v1.2
    v1.2.3.
    v1.2.3-
    v1.2.3-alpha.
    v1.2.3-alpha..1
    v1.2.3-alpha.01
    v1.2.3+
    v1.2.3+build.
    v1.2.3+build..5
    v1.2.3-alpha_beta
)
for TAG in "${INVALID_TAGS[@]}"; do
    if "$TAG_VALIDATOR" "$TAG" > "$SANDBOX/tag.log" 2>&1; then
        echo "Release qualification regression failed: invalid tag $TAG was accepted." >&2
        exit 1
    fi
    grep -q "strict SemVer" "$SANDBOX/tag.log"
done
PASSED=$((PASSED + 1))

printf '%s\n' '{"dependencies":[{"fileSystem":[{"identity":"local","path":"/tmp/local"}]}]}' \
    > "$SANDBOX/local-manifest.json"
if afmkit_release_validate_manifest < "$SANDBOX/local-manifest.json" \
    > "$SANDBOX/manifest.log" 2>&1; then
    echo "Release qualification regression failed: local manifest dependency was accepted." >&2
    exit 1
fi
grep -q "requires every root dependency to use remote source control" "$SANDBOX/manifest.log"
PASSED=$((PASSED + 1))

FIXTURE="$SANDBOX/worktree"
mkdir -p "$FIXTURE"
git -C "$FIXTURE" init -q
git -C "$FIXTURE" config user.email afmkit-tests@example.invalid
git -C "$FIXTURE" config user.name "AFMKit Tests"
printf 'pinned\n' > "$FIXTURE/Package.resolved"
git -C "$FIXTURE" add -- Package.resolved
git -C "$FIXTURE" commit -q -m fixture
afmkit_release_begin_immutable_worktree "$FIXTURE"
printf 'mutated\n' >> "$FIXTURE/Package.resolved"
if afmkit_release_verify_immutable_worktree "$FIXTURE" \
    > "$SANDBOX/mutation.log" 2>&1; then
    echo "Release qualification regression failed: Package.resolved mutation was accepted." >&2
    exit 1
fi
grep -q "mutated Package.resolved or the worktree" "$SANDBOX/mutation.log"
PASSED=$((PASSED + 1))

echo "$PASSED release qualification regression tests passed."
