#!/usr/bin/env python3
"""Validate exact package pins without forcing their products into consumers."""

from __future__ import annotations

import json
import pathlib
import sys


def fail(message: str) -> None:
    raise SystemExit(message)


def dependency_pins(document: dict) -> dict[str, str]:
    pins: dict[str, str] = {}
    for entry in document.get("dependencies", []):
        values = entry.get("sourceControl")
        if not values or len(values) != 1:
            fail(f"Unsupported non-source-control dependency entry: {entry!r}")
        dependency = values[0]
        identity = dependency.get("identity")
        exact = dependency.get("requirement", {}).get("exact")
        if not identity or not isinstance(exact, list) or len(exact) != 1:
            fail(f"Dependency {identity or entry!r} is not pinned to one exact version.")
        if identity in pins:
            fail(f"Dependency {identity} is declared more than once.")
        pins[identity] = exact[0]
    return pins


def resolved_pins(document: dict) -> dict[str, str]:
    pins: dict[str, str] = {}
    for entry in document.get("pins", []):
        identity = entry.get("identity")
        version = entry.get("state", {}).get("version")
        if not identity or not version:
            fail(f"Resolved dependency lacks an identity or semantic version: {entry!r}")
        if identity in pins:
            fail(f"Resolved dependency {identity} appears more than once.")
        pins[identity] = version
    return pins


def target_named(document: dict, name: str) -> dict:
    matches = [target for target in document.get("targets", []) if target.get("name") == name]
    if len(matches) != 1:
        fail(f"Expected exactly one {name} target, found {len(matches)}.")
    return matches[0]


def dependency_names(target: dict) -> set[str]:
    names: set[str] = set()
    for dependency in target.get("dependencies", []):
        if "byName" in dependency:
            names.add(dependency["byName"][0])
        elif "target" in dependency:
            names.add(dependency["target"][0])
        elif "product" in dependency:
            names.add(dependency["product"][0])
    return names


def main() -> int:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} dump-package.json Package.resolved", file=sys.stderr)
        return 64

    manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
    resolved = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))

    manifest_pins = dependency_pins(manifest)
    lock_pins = resolved_pins(resolved)
    if manifest_pins != lock_pins:
        missing = sorted(lock_pins.keys() - manifest_pins.keys())
        extra = sorted(manifest_pins.keys() - lock_pins.keys())
        mismatched = sorted(
            identity
            for identity in manifest_pins.keys() & lock_pins.keys()
            if manifest_pins[identity] != lock_pins[identity]
        )
        fail(
            "Package.swift exact pins differ from Package.resolved: "
            f"missing={missing}, extra={extra}, mismatched={mismatched}."
        )

    target_names = {target.get("name") for target in manifest.get("targets", [])}
    if "AFMKitMLXReleaseGraph" in target_names:
        fail("AFMKitMLXReleaseGraph must not be present in the package target graph.")

    mlx_dependencies = dependency_names(target_named(manifest, "AFMKitMLX"))
    if "AFMKitMLXReleaseGraph" in mlx_dependencies:
        fail("AFMKitMLX must not depend on the forced release graph target.")

    print(
        f"Validated {len(manifest_pins)} exact release pins; "
        "AFMKitMLX has no forced release-graph dependency."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
