#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GUARD="$ROOT/Scripts/release-qualification-guard.sh"
BUILD_ROOT="${AFMKIT_BUILD_ROOT:-$ROOT/.build}"
TAG_VALIDATOR="$ROOT/Scripts/validate-release-tag.sh"
DOWNSTREAM_GENERATOR="$ROOT/Scripts/generate-downstream-package.py"
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

GENERATOR_FIXTURE="$SANDBOX/downstream-generator"
mkdir -p "$GENERATOR_FIXTURE"
cat > "$SANDBOX/package.json" <<'JSON'
{
  "products": [
    {"name": "AFMKitCore", "targets": ["AFMKitCore"], "type": {"library": ["automatic"]}},
    {"name": "AFMOpenAICompat", "targets": ["AFMOpenAICompat"], "type": {"library": ["automatic"]}},
    {"name": "AFMKitApple", "targets": ["AFMKitApple"], "type": {"library": ["automatic"]}},
    {"name": "AFMKitMLX", "targets": ["AFMKitMLX"], "type": {"library": ["automatic"]}},
    {"name": "AFMKitFoundationModelsMLX", "targets": ["AFMKitFoundationModelsMLX"], "type": {"library": ["automatic"]}},
    {"name": "AFMKitDwarfStar", "targets": ["AFMKitDwarfStar"], "type": {"library": ["automatic"]}},
    {"name": "IgnoredTool", "targets": ["IgnoredTool"], "type": {"executable": null}}
  ]
}
JSON
/usr/bin/python3 "$DOWNSTREAM_GENERATOR" \
    "$GENERATOR_FIXTURE" "https://github.com/example/AFMKit.git" "1.2.3-rc.1+build.5" \
    < "$SANDBOX/package.json"
/usr/bin/python3 - "$GENERATOR_FIXTURE" <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
manifest = (root / "Package.swift").read_text(encoding="utf-8")
products = (root / "validator-products.txt").read_text(encoding="utf-8").splitlines()
assert len(products) == 6
assert products == sorted(products)
assert "IgnoredTool" not in manifest
assert 'exact: "1.2.3-rc.1+build.5"' in manifest
sources = list((root / "Sources").glob("*/main.swift"))
for product in products:
    public_product = product.removesuffix("Consumer")
    assert manifest.count(f'.product(name: "{public_product}", package: "AFMKit")') == 1
    assert any(f"import {public_product}" in path.read_text(encoding="utf-8") for path in sources)
PY
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

printf '%s\n' '{"dependencies":[{"sourceControl":[{"identity":"unstable","location":{"remote":[{"urlString":"https://github.com/example/unstable"}]},"requirement":{"revision":["0123456789012345678901234567890123456789"]}}]}]}' \
    > "$SANDBOX/revision-manifest.json"
if afmkit_release_validate_manifest < "$SANDBOX/revision-manifest.json" \
    > "$SANDBOX/revision-manifest.log" 2>&1; then
    echo "Release qualification regression failed: revision dependency was accepted." >&2
    exit 1
fi
grep -q "rejected unstable branch/revision dependency" "$SANDBOX/revision-manifest.log"
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
