#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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

private_ci = contents["private-ci.yml"]
assert "workflow_run:" in private_ci
assert "Provision locked dependencies with trusted code" in private_ci
assert "Validate candidate API baselines without credentials" in private_ci
assert private_ci.index("Provision locked dependencies with trusted code") < private_ci.index(
    "Validate candidate API baselines without credentials"
)
assert "persist-credentials: false" in private_ci

request = contents["release-request.yml"]
assert "workflow_dispatch:" in request
assert "secrets." not in request
assert "contents: write" not in request

release = contents["release.yml"]
assert "workflow_run:" in release
assert "workflow_dispatch:" not in release
assert "github.event.workflow_run.head_branch == github.event.repository.default_branch" in release
assert "Qualify clean remote tag graph" in release
assert "publishRelease" in release
assert "actions/github-script@ed597411d8f924073f98dfc5c65a23a2325f34cd" in release

print(f"{len(contents)} workflow trust-boundary regression checks passed.")
PY
