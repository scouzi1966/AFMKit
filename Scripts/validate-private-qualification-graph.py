#!/usr/bin/env python3
import argparse
import json
import pathlib
import re
import urllib.parse


GRAPH_FILES = ("Package.swift", "Package.resolved")
EXACT_DEPENDENCY = re.compile(
    r'\.package\(\s*url:\s*"([^"]+)"\s*,\s*exact:\s*"([^"]+)"',
    re.MULTILINE,
)


def package_identity(url: str) -> str:
    name = pathlib.PurePosixPath(urllib.parse.urlparse(url).path.rstrip("/")).name
    return (name[:-4] if name.lower().endswith(".git") else name).lower()


def exact_dependencies(path: pathlib.Path) -> dict[str, tuple[str, str]]:
    dependencies = {
        package_identity(url): (url, version)
        for url, version in EXACT_DEPENDENCY.findall(path.read_text(encoding="utf-8"))
    }
    if not dependencies:
        raise SystemExit(f"{path} contains no exact remote dependencies.")
    return dependencies


def resolved_pins(path: pathlib.Path) -> dict[str, tuple[str, str, str]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    pins: dict[str, tuple[str, str, str]] = {}
    for pin in document.get("pins", []):
        identity = pin.get("identity", "")
        state = pin.get("state", {})
        location = pin.get("location", "")
        version = state.get("version", "")
        revision = state.get("revision", "")
        if identity in pins or not location.startswith("https://") or not version or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise SystemExit(f"{path} contains an invalid or duplicate pin {identity!r}.")
        pins[identity] = (location, version, revision)
    return pins


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=pathlib.Path, required=True)
    parser.add_argument("--trusted", type=pathlib.Path, required=True)
    parser.add_argument("--qualification-manifest", type=pathlib.Path, required=True)
    arguments = parser.parse_args()

    for relative in GRAPH_FILES:
        if (arguments.candidate / relative).read_bytes() != (arguments.trusted / relative).read_bytes():
            raise SystemExit(
                f"Candidate graph input {relative} differs from the trusted default-branch copy. "
                "Dependency-graph changes require a separately trusted infrastructure update."
            )

    root_dependencies = exact_dependencies(arguments.candidate / "Package.swift")
    qualification_dependencies = exact_dependencies(arguments.qualification_manifest)
    if qualification_dependencies != root_dependencies:
        raise SystemExit("Trusted private qualification dependencies do not equal the candidate root manifest.")

    pins = resolved_pins(arguments.candidate / "Package.resolved")
    for identity, (_, version) in root_dependencies.items():
        pin = pins.get(identity)
        if pin is None or pin[1] != version:
            raise SystemExit(f"Root lock does not pin exact dependency {identity} at {version}.")

    print(
        "Candidate manifest and lock equal the trusted single-package graph: "
        f"{len(root_dependencies)} direct dependencies, {len(pins)} resolved pins."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
