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
assert "SDK exposure (${{ matrix.sdk }})" in public_ci
assert "vars.AFMKIT_XCODE26_RUNNER || 'macos-26'" in public_ci
assert "vars.AFMKIT_XCODE27_RUNNER || 'xcode-27'" in public_ci
assert "check-sdk-product-exposure.sh" in public_ci
assert "Run untrusted candidate XCTest suite" in public_ci
assert "--disable-swift-testing" in public_ci
assert "Run untrusted candidate Swift Testing suite" in public_ci
assert "--disable-xctest" in public_ci

private_ci = contents["private-ci.yml"]
assert "workflow_run:" in private_ci
assert "Check out exact candidate" not in private_ci
assert "candidate/Scripts" not in private_ci
assert "candidate/Package.swift" not in private_ci
assert "github.event.workflow_run.status == 'completed'" in private_ci
assert 'run.status !== "completed"' in private_ci
assert "github.rest.actions.getWorkflowRun" in private_ci
assert 'source.path !== ".github/workflows/ci.yml"' in private_ci
assert "source.repository.full_name !== currentRepository" in private_ci
assert "pull.base.ref !== defaultBranch" in private_ci
assert "pull.base.repo.full_name !== currentRepository" in private_ci
assert "pull.head.sha !== run.head_sha" in private_ci
assert "github.rest.git.getCommit" in private_ci
assert "github.rest.repos.compareCommitsWithBasehead" in private_ci
assert "vars.AFMKIT_XCODE27_RUNNER || 'xcode-27'" in private_ci
assert "pull-requests: read" in private_ci
assert "Download exact successful-run artifact" in private_ci
assert "prepare-private-qualification.py" in private_ci
assert "validate-private-qualification-graph.py" in private_ci
assert "Qualification/PrivatePackage.swift" in private_ci
assert "AFMKIT_DEPENDENCY_TOKEN" not in private_ci
assert "Compile candidate with vendored MLX sources" in private_ci
assert "--build-tests" in private_ci
assert "xctest" not in private_ci
assert "swift test" not in private_ci
assert "if: always() && steps.metadata.outputs.eligible == 'true'" in private_ci
assert "destroy-private-qualification.py" in private_ci
assert "--path \"$PRIVATE_BUILD_ROOT\"" in private_ci
assert "--path \"$HARNESS_ROOT\"" in private_ci
assert "persist-credentials: false" in private_ci
assert private_ci.index("Compile candidate with vendored MLX sources") < private_ci.index(
    "Destroy candidate sources, products, and harness"
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
assert "AFMKIT_RELEASE_MIRROR_OUTPUT" not in release
assert "publish-provider-mirrors.sh" not in release
assert "AFMKIT_DEPENDENCY_TOKEN" not in release
assert "with-private-dependency-auth.sh" not in release
assert "run: Scripts/validate-release.sh" in release
assert "vars.AFMKIT_XCODE27_RUNNER || 'xcode-27'" in release
assert "Record immutable publication intent" in release
assert "ensurePublicationIntent" in release
assert release.index("Record immutable publication intent") < release.index(
    "Publish root tag and GitHub release"
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

EXPECTED_GRAPH_INPUTS="$(cat <<'EOF'
Package.resolved
Package.swift
vendor/MLX/mlx-swift-lm/Package.swift
vendor/MLX/mlx-swift/Package.swift
EOF
)"
ACTUAL_GRAPH_INPUTS="$(find "$EXTRACTED" -type f \
    \( -name 'Package*.swift' -o -name 'Package.resolved' \) \
    -print | sed "s|^$EXTRACTED/||" | sort)"
if [[ "$ACTUAL_GRAPH_INPUTS" != "$EXPECTED_GRAPH_INPUTS" ]]; then
    echo "Private qualification artifact did not preserve the exact graph inputs." >&2
    diff -u <(printf '%s\n' "$EXPECTED_GRAPH_INPUTS") \
        <(printf '%s\n' "$ACTUAL_GRAPH_INPUTS") || true
    exit 1
fi
if find "$EXTRACTED" -path '*/Scripts/*' -o -name 'Package@swift-*.swift' | grep -q .; then
    echo "Private qualification artifact included candidate executable controls." >&2
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

"$ROOT/Scripts/validate-private-qualification-graph.py" \
    --candidate "$EXTRACTED" \
    --trusted "$ROOT" \
    --qualification-manifest "$ROOT/Qualification/PrivatePackage.swift" \
    > "$SANDBOX/graph.log"
grep -q "Candidate manifest and lock equal the trusted single-package graph" \
    "$SANDBOX/graph.log"

cp -R "$EXTRACTED" "$SANDBOX/tampered-graph"
printf '\n// candidate dependency control\n' \
    >> "$SANDBOX/tampered-graph/Package.swift"
if "$ROOT/Scripts/validate-private-qualification-graph.py" \
    --candidate "$SANDBOX/tampered-graph" \
    --trusted "$ROOT" \
    --qualification-manifest "$ROOT/Qualification/PrivatePackage.swift" \
    > "$SANDBOX/manifest-equality.log" 2>&1; then
    echo "Private qualification accepted a candidate-controlled manifest." >&2
    exit 1
fi
grep -q "differs from the trusted default-branch copy" \
    "$SANDBOX/manifest-equality.log"

cp -R "$EXTRACTED" "$SANDBOX/tampered-lock"
/usr/bin/python3 - "$SANDBOX/tampered-lock/Package.resolved" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    lock = json.load(handle)
lock["pins"][0]["state"]["revision"] = "f" * 40
with open(path, "w", encoding="utf-8") as handle:
    json.dump(lock, handle)
PY
if "$ROOT/Scripts/validate-private-qualification-graph.py" \
    --candidate "$SANDBOX/tampered-lock" \
    --trusted "$ROOT" \
    --qualification-manifest "$ROOT/Qualification/PrivatePackage.swift" \
    > "$SANDBOX/lock-equality.log" 2>&1; then
    echo "Private qualification accepted a candidate-controlled lock." >&2
    exit 1
fi
grep -q "differs from the trusted default-branch copy" "$SANDBOX/lock-equality.log"

cp "$ROOT/Qualification/PrivatePackage.swift" "$SANDBOX/Package.swift"
cp "$EXTRACTED/Package.resolved" "$SANDBOX/Package.resolved"
mv "$EXTRACTED" "$SANDBOX/Candidate"
/usr/bin/xcrun --toolchain XcodeDefault swift package dump-package \
    --package-path "$SANDBOX" > "$SANDBOX/private-package.json"
/usr/bin/python3 - "$SANDBOX/private-package.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    package = json.load(handle)

products = {product["name"] for product in package["products"]}
targets = {target["name"] for target in package["targets"]}
assert "AFMEvalKit" in products, "Private qualification omits AFMEvalKit product"
assert "AFMEvalKit" in targets, "Private qualification omits AFMEvalKit target"
assert "AFMEvalKitTests" in targets, "Private qualification omits AFMEvalKit tests"
PY

CLEANUP_ROOT="$SANDBOX/cleanup-root"
OUTSIDE_ROOT="$SANDBOX/outside-root"
mkdir -p "$CLEANUP_ROOT/private-products" "$OUTSIDE_ROOT"
printf 'must survive\n' > "$OUTSIDE_ROOT/sentinel"
ln -s "$OUTSIDE_ROOT" "$CLEANUP_ROOT/private-products/outside-link"
"$ROOT/Scripts/destroy-private-qualification.py" \
    --root "$CLEANUP_ROOT" \
    --path "$CLEANUP_ROOT/private-products"
test -f "$OUTSIDE_ROOT/sentinel"
test ! -e "$CLEANUP_ROOT/private-products"
if "$ROOT/Scripts/destroy-private-qualification.py" \
    --root "$CLEANUP_ROOT" \
    --path "$OUTSIDE_ROOT" \
    > "$SANDBOX/cleanup-boundary.log" 2>&1; then
    echo "Private cleanup accepted a path outside its runner root." >&2
    exit 1
fi
grep -q "outside cleanup root" "$SANDBOX/cleanup-boundary.log"

echo "Workflow trust boundaries and qualification artifact checks passed."
