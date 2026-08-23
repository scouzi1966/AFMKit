#!/usr/bin/env python3
import argparse
import json
import os
import pathlib
import re
import shutil
import stat


LOCAL_DEPENDENCY_PATTERN = re.compile(
    r"if let publicPackagePath = ProcessInfo\.processInfo\.environment"
    r"\[\"AFMKIT_PUBLIC_PATH\"\],\n"
    r"   !publicPackagePath\.isEmpty \{\n"
    r"    publicPackageDependency = \.package\(name: \"AFMKit\", path: publicPackagePath\)\n"
    r"\} else \{\n"
    r"    publicPackageDependency = \.package\(name: \"AFMKit\", path: \"\.\./\.\.\"\)\n"
    r"\}\n"
)


def copy_package(source: pathlib.Path, destination: pathlib.Path) -> None:
    ignored = {".build", ".swiftpm", "Package.resolved"}
    for directory, directory_names, file_names in os.walk(source):
        directory_names[:] = sorted(name for name in directory_names if name not in ignored)
        file_names.sort()
        directory_path = pathlib.Path(directory)
        relative_directory = directory_path.relative_to(source)
        output_directory = destination / relative_directory
        output_directory.mkdir(parents=True, exist_ok=True)
        for name in directory_names + file_names:
            path = directory_path / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise SystemExit(f"Provider package materialization rejects symlink: {path}")
        for name in file_names:
            if name in ignored:
                continue
            source_file = directory_path / name
            if not stat.S_ISREG(source_file.lstat().st_mode):
                raise SystemExit(f"Provider package requires regular file: {source_file}")
            shutil.copy2(source_file, output_directory / name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--public-url", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-sha", required=True)
    arguments = parser.parse_args()

    source = arguments.source.resolve()
    output = arguments.output.resolve()
    if output.exists():
        raise SystemExit(f"Provider package output already exists: {output}")
    if not arguments.public_url.startswith("https://"):
        raise SystemExit("Published provider packages require an HTTPS AFMKit URL.")
    if not re.fullmatch(r"[0-9a-f]{40}", arguments.source_sha):
        raise SystemExit("Provider package source SHA must be a full commit SHA.")
    if "+" in arguments.version:
        raise SystemExit("SwiftPM release versions reject SemVer build metadata.")

    copy_package(source, output)
    manifest_path = output / "Package.swift"
    manifest = manifest_path.read_text(encoding="utf-8")
    replacement = (
        "publicPackageDependency = .package(\n"
        f"    url: {json.dumps(arguments.public_url)},\n"
        f"    exact: {json.dumps(arguments.version)}\n"
        ")\n"
    )
    manifest, replacement_count = LOCAL_DEPENDENCY_PATTERN.subn(replacement, manifest)
    if replacement_count != 1:
        raise SystemExit(
            f"Expected one local AFMKit dependency block; replaced {replacement_count}."
        )
    manifest_path.write_text(manifest, encoding="utf-8")

    provenance = {
        "package": source.name,
        "public_dependency": {
            "url": arguments.public_url,
            "version": arguments.version,
        },
        "source_sha": arguments.source_sha,
    }
    (output / "AFMKit.release.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
