#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${AFMKIT_BUILD_ROOT:-$ROOT/.build}"
mkdir -p "$BUILD_ROOT"
SANDBOX="$(mktemp -d "$BUILD_ROOT/workflow-security.XXXXXX")"

cleanup() {
    rm -rf "$SANDBOX"
}
trap cleanup EXIT

/usr/bin/python3 - "$ROOT/.github/workflows" <<'PY'
import pathlib
import re
import sys

workflows = pathlib.Path(sys.argv[1])
contents = {path.name: path.read_text(encoding="utf-8") for path in workflows.glob("*.yml")}
required = {"ci.yml", "private-ci.yml", "release-request.yml", "release.yml"}
missing = required - contents.keys()
assert not missing, f"Missing workflows: {sorted(missing)}"

for name, content in contents.items():
    for action, ref in re.findall(r"uses:\s+([^@\s]+)@([^\s#]+)", content):
        assert re.fullmatch(r"[0-9a-f]{40}", ref), (
            f"{name} uses mutable action ref {action}@{ref}"
        )
    assert "environment:" not in content, (
        f"{name} relies on an unavailable protected environment"
    )

public_ci = contents["ci.yml"]
assert "AFMKIT_DEPENDENCY_TOKEN" not in public_ci
assert "secrets." not in public_ci
assert "create-qualification-artifact.py" in public_ci
assert "private-qualification-${{ github.run_id }}" in public_ci
assert "github.event.pull_request.head.sha || github.sha" in public_ci

private_ci = contents["private-ci.yml"]
assert "workflow_run:" in private_ci
assert "github.rest.pulls.get" not in private_ci
assert "Check out exact candidate" not in private_ci
assert "candidate/Scripts" not in private_ci
assert "candidate/Package.swift" not in private_ci
assert "Download exact successful-run artifact" in private_ci
assert "prepare-private-qualification.py" in private_ci
assert "Qualification/PrivatePackage.swift" in private_ci
assert "AFMKitPrivateDependencySeed" in private_ci
assert "with-private-source-sandbox.sh" in private_ci
assert "AFMKIT_REQUIRE_SANDBOX_COMPILERS: swiftc,clang" in private_ci
assert "--build-tests" in private_ci
assert "xctest \"$TEST_BUNDLE\"" in private_ci
assert "swift test" not in private_ci
assert private_ci.index("Remove all private source") < private_ci.index(
    "Run prebuilt candidate tests"
)
assert "checkouts/mlx-swift-afm" in private_ci
assert "checkouts/mlx-swift-lm" in private_ci
assert "persist-credentials: false" in private_ci
assert private_ci.index("Prebuild private dependencies") < private_ci.index(
    "Compile candidate without private source access"
)
assert private_ci.index("Compile candidate without private source access") < private_ci.index(
    "Remove all private source"
)

request = contents["release-request.yml"]
assert "workflow_dispatch:" in request
assert "secrets." not in request
assert "contents: write" not in request

release = contents["release.yml"]
assert "workflow_run:" in release
assert "workflow_dispatch:" not in release
assert "github.event.workflow_run.head_branch == github.event.repository.default_branch" in release
assert "publishRelease" in release
assert "stage-tag:" not in release
assert "qualify-remote-tag:" not in release
assert "AFMKIT_RELEASE_MIRROR_OUTPUT" in release
assert "publish-provider-mirrors.sh" in release
assert release.index("Publish provider package tags idempotently") < release.index(
    "Publish root tag and GitHub release last"
)
assert "actions/github-script@ed597411d8f924073f98dfc5c65a23a2325f34cd" in release
PY

SHA="1111111111111111111111111111111111111111"
ARTIFACT="$SANDBOX/artifact"
EXTRACTED="$SANDBOX/extracted"
"$ROOT/Scripts/create-qualification-artifact.py" \
    --root "$ROOT" \
    --output "$ARTIFACT" \
    --run-id 17 \
    --sha "$SHA" \
    --repository owner/AFMKit
"$ROOT/Scripts/prepare-private-qualification.py" \
    --artifact "$ARTIFACT" \
    --destination "$EXTRACTED" \
    --expected-run-id 17 \
    --expected-sha "$SHA" \
    --expected-repository owner/AFMKit

if find "$EXTRACTED" -name 'Package*.swift' -o -path '*/Scripts/*' | grep -q .; then
    echo "Private qualification artifact included candidate executable control files." >&2
    exit 1
fi

if "$ROOT/Scripts/prepare-private-qualification.py" \
    --artifact "$ARTIFACT" \
    --destination "$SANDBOX/wrong-provenance" \
    --expected-run-id 18 \
    --expected-sha "$SHA" \
    --expected-repository owner/AFMKit \
    > "$SANDBOX/provenance.log" 2>&1; then
    echo "Private qualification accepted mismatched workflow-run provenance." >&2
    exit 1
fi
grep -q "provenance mismatch" "$SANDBOX/provenance.log"

cp "$ROOT/Qualification/PrivatePackage.swift" "$SANDBOX/Package.swift"
cp -R "$ROOT/Qualification/TrustedSeed" "$SANDBOX/TrustedSeed"
mv "$EXTRACTED" "$SANDBOX/Candidate"
/usr/bin/xcrun --toolchain XcodeDefault swift package dump-package \
    --package-path "$SANDBOX" > /dev/null

SANDBOX_BUILD="$SANDBOX/compiler-build"
mkdir -p \
    "$SANDBOX_BUILD/checkouts/mlx-swift-afm" \
    "$SANDBOX_BUILD/checkouts/mlx-swift-lm" \
    "$SANDBOX_BUILD/repositories"
printf '#define PRIVATE_VALUE 17\n' \
    > "$SANDBOX_BUILD/checkouts/mlx-swift-afm/private.h"
printf '#include "%s"\nPRIVATE_VALUE\n' \
    "$SANDBOX_BUILD/checkouts/mlx-swift-afm/private.h" \
    > "$SANDBOX/read-private.c"
if AFMKIT_REQUIRE_SANDBOX_COMPILERS=clang \
    "$ROOT/Scripts/with-private-source-sandbox.sh" \
        "$SANDBOX_BUILD" /bin/bash -c '"$CC" -E "$1"' _ \
        "$SANDBOX/read-private.c" \
        > "$SANDBOX/compiler-sandbox.log" 2>&1; then
    echo "Sandboxed candidate compiler read private dependency source." >&2
    exit 1
fi
grep -Eq "Operation not permitted|not found" "$SANDBOX/compiler-sandbox.log"

echo "Workflow trust boundaries and qualification artifact checks passed."
