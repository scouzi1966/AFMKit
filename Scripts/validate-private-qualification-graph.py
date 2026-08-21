#!/usr/bin/env python3
import argparse
import json
import pathlib
import re
import urllib.parse


GRAPH_FILES = (
    "Package.swift",
    "Packages/AFMKitDwarfStar/Package.swift",
    "Packages/AFMKitDwarfStar/Package.resolved",
    "Packages/AFMKitMLX/Package.swift",
    "Packages/AFMKitMLX/Package.resolved",
)
EXACT_DEPENDENCY = re.compile(
    r"\.package\(\s*url:\s*\"([^\"]+)\"\s*,\s*exact:\s*\"([^\"]+)\"",
    re.MULTILINE,
)


def package_identity(url: str) -> str:
    path = urllib.parse.urlparse(url).path.rstrip("/")
    name = pathlib.PurePosixPath(path).name
    return (name[:-4] if name.lower().endswith(".git") else name).lower()


def exact_dependencies(path: pathlib.Path) -> dict[str, tuple[str, str]]:
    dependencies: dict[str, tuple[str, str]] = {}
    for url, version in EXACT_DEPENDENCY.findall(path.read_text(encoding="utf-8")):
        identity = package_identity(url)
        value = (url, version)
        if identity in dependencies and dependencies[identity] != value:
            raise SystemExit(f"{path} declares conflicting requirements for {identity}.")
        dependencies[identity] = value
    if not dependencies:
        raise SystemExit(f"{path} contains no exact remote dependencies.")
    return dependencies


def resolved_pins(path: pathlib.Path) -> dict[str, tuple[str, str, str]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("version") != 3 or not isinstance(document.get("pins"), list):
        raise SystemExit(f"{path} is not a version 3 SwiftPM lock.")
    pins: dict[str, tuple[str, str, str]] = {}
    for pin in document["pins"]:
        identity = pin.get("identity")
        location = pin.get("location")
        state = pin.get("state") or {}
        version = state.get("version")
        revision = state.get("revision")
        if not isinstance(identity, str) or not identity:
            raise SystemExit(f"{path} contains a pin without an identity.")
        if identity in pins:
            raise SystemExit(f"{path} contains duplicate pin {identity}.")
        if not isinstance(location, str) or not location.startswith("https://"):
            raise SystemExit(f"{path} pin {identity} does not use HTTPS.")
        if not isinstance(version, str) or not version:
            raise SystemExit(f"{path} pin {identity} does not have an exact version.")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise SystemExit(f"{path} pin {identity} does not have an immutable revision.")
        pins[identity] = (location, version, revision)
    return pins


def validate_direct_dependencies(
    manifest: pathlib.Path,
    lock: pathlib.Path,
) -> dict[str, tuple[str, str]]:
    dependencies = exact_dependencies(manifest)
    pins = resolved_pins(lock)
    for identity, (_, version) in dependencies.items():
        pin = pins.get(identity)
        if pin is None:
            raise SystemExit(f"{lock} does not pin direct dependency {identity}.")
        if pin[1] != version:
            raise SystemExit(
                f"{lock} pins {identity} at {pin[1]}, but {manifest} requires {version}."
            )
    return dependencies


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=pathlib.Path, required=True)
    parser.add_argument("--trusted", type=pathlib.Path, required=True)
    parser.add_argument("--qualification-manifest", type=pathlib.Path, required=True)
    arguments = parser.parse_args()

    candidate = arguments.candidate.resolve()
    trusted = arguments.trusted.resolve()
    for relative in GRAPH_FILES:
        candidate_path = candidate / relative
        trusted_path = trusted / relative
        if candidate_path.read_bytes() != trusted_path.read_bytes():
            raise SystemExit(
                f"Candidate graph input {relative} differs from the trusted default-branch copy. "
                "Dependency-graph changes require a separately trusted infrastructure update."
            )

    mlx_manifest = candidate / "Packages/AFMKitMLX/Package.swift"
    mlx_lock = candidate / "Packages/AFMKitMLX/Package.resolved"
    dwarf_manifest = candidate / "Packages/AFMKitDwarfStar/Package.swift"
    dwarf_lock = candidate / "Packages/AFMKitDwarfStar/Package.resolved"
    mlx_dependencies = validate_direct_dependencies(mlx_manifest, mlx_lock)
    dwarf_dependencies = validate_direct_dependencies(dwarf_manifest, dwarf_lock)
    qualification_dependencies = exact_dependencies(arguments.qualification_manifest)

    if qualification_dependencies != mlx_dependencies:
        raise SystemExit(
            "Trusted private qualification dependencies do not equal the candidate MLX manifest."
        )
    if not set(dwarf_dependencies.items()).issubset(set(mlx_dependencies.items())):
        raise SystemExit("DwarfStar direct dependencies are not a subset of the MLX graph.")

    mlx_pins = resolved_pins(mlx_lock)
    dwarf_pins = resolved_pins(dwarf_lock)
    for identity, dwarf_pin in dwarf_pins.items():
        if mlx_pins.get(identity) != dwarf_pin:
            raise SystemExit(
                f"Provider locks disagree on shared dependency {identity}."
            )

    print(
        "Candidate manifests and locks equal the trusted graph: "
        f"{len(mlx_dependencies)} direct dependencies, {len(mlx_pins)} resolved pins."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
