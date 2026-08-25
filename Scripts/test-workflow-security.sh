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
assert "test-provider-publication.sh" not in public_ci
assert "create-qualification-artifact.py" in public_ci
assert "private-qualification-${{ github.run_id }}" in public_ci
assert "github.event.pull_request.head.sha || github.sha" in public_ci
assert "SDK exposure (${{ matrix.sdk }})" in public_ci
assert "vars.AFMKIT_XCODE26_RUNNER || 'macos-26'" in public_ci
assert "vars.AFMKIT_XCODE27_RUNNER || 'xcode-27'" in public_ci
assert "check-sdk-product-exposure.sh" in public_ci
assert public_ci.count("run: Scripts/check-release-dependency-policy.sh") == 1

xctest_step = public_ci.split(
    "- name: Run untrusted candidate XCTest suite", 1
)[1].split("\n      - name:", 1)[0]
swift_testing_step = public_ci.split(
    "- name: Run untrusted candidate Swift Testing suite", 1
)[1].split("\n      - name:", 1)[0]
assert "run: Scripts/run-xctest-targets.sh" in xctest_step
assert "--disable-xctest" in swift_testing_step
assert "--disable-swift-testing" not in swift_testing_step
assert "-c release" in swift_testing_step

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

release_validator = (workflows.parent.parent / "Scripts" / "validate-release.sh").read_text(
    encoding="utf-8"
)
isolated_xctest = (workflows.parent.parent / "Scripts" / "run-xctest-targets.sh").read_text(
    encoding="utf-8"
)
service_isolation = (
    workflows.parent.parent / "Scripts" / "check-service-product-isolation.sh"
).read_text(encoding="utf-8")
evalkit_isolation = (
    workflows.parent.parent / "Scripts" / "check-evalkit-isolation.sh"
).read_text(encoding="utf-8")
local_release = (
    workflows.parent.parent / "Scripts" / "release-local.js"
).read_text(encoding="utf-8")
assert release_validator.count("afmkit_run_qualified_swift test") == 1
assert release_validator.count("run-xctest-targets.sh") == 1
assert release_validator.count("--disable-xctest") == 1
assert release_validator.count("test-release-local.js") == 1
assert release_validator.count('"$ROOT/Scripts/check-release-dependency-policy.sh"') == 1
candidate_call = "await publication.assertReleaseCandidate(publicationArguments)"
validator_call = 'path.join(root, "Scripts/validate-release.sh")'
intent_call = "await publication.ensurePublicationIntent(publicationArguments)"
publish_call = "await publication.publishRelease(publicationArguments)"
verify_call = "await publication.validatePublishedRelease(publicationArguments)"
for snippet in (candidate_call, validator_call, intent_call, publish_call, verify_call):
    assert local_release.count(snippet) == 1
assert local_release.index(candidate_call) < local_release.index(validator_call)
assert local_release.index(validator_call) < local_release.index(intent_call)
assert local_release.index(intent_call) < local_release.index(publish_call)
assert local_release.index(publish_call) < local_release.index(verify_call)
assert 'delete clean.GH_TOKEN' in local_release
assert 'delete clean.GITHUB_TOKEN' in local_release
assert '"gh",\n            ["auth", "token"' in local_release
assert "git push" not in local_release
assert "git tag" not in local_release
assert 'target.get("type") == "test"' in isolated_xctest
assert '--filter "$test_target"' in isolated_xctest
isolated_targets_match = re.search(
    r'CASE_ISOLATED_TARGETS=\(\n(?P<body>.*?)\n\)',
    isolated_xctest,
    re.DOTALL,
)
assert isolated_targets_match is not None
isolated_target_lines = [
    line.strip()
    for line in isolated_targets_match.group("body").splitlines()
    if line.strip()
]
assert isolated_target_lines == [
    '"AFMKitAppleTests"',
    '"AFMKitFoundationModelsMLXTests"',
]
assert isolated_xctest.count("CASE_ISOLATED_TARGETS") == 2
assert isolated_xctest.count('if is_case_isolated_target "$test_target"; then') == 1
unstable_cases_match = re.search(
    r'HOSTED_FOUNDATION_MODELS_UNSTABLE_CASES=\(\n(?P<body>.*?)\n\)',
    isolated_xctest,
    re.DOTALL,
)
assert unstable_cases_match is not None
unstable_case_lines = [
    line.strip()
    for line in unstable_cases_match.group("body").splitlines()
    if line.strip()
]
assert unstable_case_lines == [
    '"AFMKitFoundationModelsMLXTests.AFMKitFoundationModelsMLXTests/'
    'testTranscriptTranslationPreservesRolesAndText"',
    '"AFMKitFoundationModelsMLXTests.AFMKitFoundationModelsMLXTests/'
    'testTranscriptTranslationPreservesToolCallsAndOutputs"',
    '"AFMKitAppleTests.FoundationStopSequenceFilterTests/'
    'testAcceptedTranscriptExcludesHiddenPostStopOutput"',
    '"AFMKitAppleTests.FoundationModelSessionCoordinatorTests/'
    'testDynamicProfileSessionReusesExactProviderAndSignature"',
    '"AFMKitAppleTests.FoundationModelSessionCoordinatorTests/'
    'testSimpleSessionReusesExactProviderAndSignature"',
    '"AFMKitAppleTests.FoundationModelSessionCoordinatorTests/'
    'testHistoryTransformDropsOrphanResponseBeforeFirstPrompt"',
    '"AFMKitAppleTests.FoundationModelSessionCoordinatorTests/'
    'testHistoryTransformKeepsConversationStartingAtFirstPrompt"',
    '"AFMKitAppleTests.FoundationNativeSessionRuntimeTests/'
    'testAppleOnDeviceReusesMatchingSession"',
    '"AFMKitAppleTests.FoundationPromptBuilderTests/'
    'testBuildsAttachmentPromptWithDefaultInstruction"',
    '"AFMKitAppleTests.FoundationSessionUsageTelemetryTests/'
    'testMapsLanguageModelSessionUsageIntoTelemetry"',
    '"AFMKitAppleTests.FoundationSessionUsageTelemetryTests/'
    'testSingleResponseTelemetryUsesCompletionAsFirstChunk"',
    '"AFMKitAppleTests.FoundationTranscriptWindowPlannerTests/'
    'testTrimsOldestPromptTurnsAndPreservesInstructions"',
    '"AFMKitAppleTests.FoundationTranscriptWindowPlannerTests/'
    'testThrowsWhenCurrentTurnCannotFit"',
    '"AFMKitAppleTests.FoundationTranscriptSnapshotParserTests/'
    'testRecordsToolArgumentsAndCompletedOutput"',
    '"AFMKitAppleTests.FoundationTranscriptSnapshotParserTests/'
    'testOutputWithoutMatchingIDUsesLatestRequestedToolWithSameName"',
    '"AFMKitAppleTests.FoundationTranscriptSnapshotParserTests/'
    'testReasoningContentReturnsReasoningAfterLatestPrompt"',
    '"AFMKitAppleTests.FoundationStructuredResponseCompleterTests/'
    'testCompletesRenderedContentToolSnapshotsAndTelemetry"',
]
assert isolated_xctest.count("HOSTED_FOUNDATION_MODELS_UNSTABLE_CASES") == 2
assert isolated_xctest.count(
    '[[ "${RUNNER_ENVIRONMENT:-}" == "github-hosted" ]] || return 1'
) == 1
assert isolated_xctest.count(
    'if is_hosted_foundation_models_unstable_case "$test_case"; then'
) == 1
assert '/usr/bin/awk -v prefix="${test_target}."' in isolated_xctest
assert 'run_case_isolated_test "$test_case"' in isolated_xctest
assert 'FOUNDATION_MODELS_SIGNAL_RETRY_LIMIT=1' in isolated_xctest
assert 'unexpected signal code 11' in isolated_xctest
assert isolated_xctest.count('--filter "$test_case"') == 1
assert "--skip-build" in isolated_xctest
for isolation_script, lock_copy in (
    (service_isolation, 'cp "$ROOT/Package.resolved" "$CONSUMER/Package.resolved"'),
    (evalkit_isolation, 'cp "$ROOT/Package.resolved" "$SANDBOX/Package.resolved"'),
):
    assert isolation_script.count(lock_copy) == 1
    assert isolation_script.count("swift build") == 1
    assert isolation_script.count("--disable-automatic-resolution") == 1
    assert isolation_script.index(lock_copy) < isolation_script.index("swift build")
assert "--disable-swift-testing" in isolated_xctest
assert "--disable-automatic-resolution" in isolated_xctest
assert "-c release" in isolated_xctest
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

/usr/bin/python3 - "$ROOT/Scripts/check-release-dependency-policy.py" "$SANDBOX" <<'PY'
import copy
import json
import pathlib
import subprocess
import sys

checker = pathlib.Path(sys.argv[1])
sandbox = pathlib.Path(sys.argv[2])
base_manifest = {
    "dependencies": [{
        "sourceControl": [{
            "identity": "example",
            "requirement": {"exact": ["1.2.3"]},
        }]
    }],
    "targets": [{"name": "AFMKitMLX", "dependencies": []}],
}
base_resolved = {
    "pins": [{"identity": "example", "state": {"version": "1.2.3"}}]
}

def expect_rejection(name, manifest, resolved, message):
    manifest_path = sandbox / f"{name}-manifest.json"
    resolved_path = sandbox / f"{name}-resolved.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    resolved_path.write_text(json.dumps(resolved), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(checker), str(manifest_path), str(resolved_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0, f"{name} unexpectedly passed"
    assert message in result.stdout + result.stderr, (name, result.stdout, result.stderr)

non_exact = copy.deepcopy(base_manifest)
non_exact["dependencies"][0]["sourceControl"][0]["requirement"] = {
    "range": [{"lowerBound": "1.0.0", "upperBound": "2.0.0"}]
}
expect_rejection("non-exact", non_exact, base_resolved, "not pinned to one exact version")

local_dependency = copy.deepcopy(base_manifest)
local_dependency["dependencies"] = [{"fileSystem": [{"identity": "local", "path": "../local"}]}]
expect_rejection("local", local_dependency, base_resolved, "Unsupported non-source-control")

duplicate = copy.deepcopy(base_manifest)
duplicate["dependencies"].append(copy.deepcopy(duplicate["dependencies"][0]))
expect_rejection("duplicate", duplicate, base_resolved, "declared more than once")

mismatch = copy.deepcopy(base_resolved)
mismatch["pins"][0]["state"]["version"] = "1.2.4"
expect_rejection("mismatch", base_manifest, mismatch, "mismatched=['example']")

missing = copy.deepcopy(base_resolved)
missing["pins"].append({"identity": "extra", "state": {"version": "4.5.6"}})
expect_rejection("missing", base_manifest, missing, "missing=['extra']")

forced_graph = copy.deepcopy(base_manifest)
forced_graph["targets"].append({"name": "AFMKitMLXReleaseGraph", "dependencies": []})
expect_rejection("forced-graph", forced_graph, base_resolved, "must not be present")
PY

echo "Workflow trust boundaries and qualification artifact checks passed."
