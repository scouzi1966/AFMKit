#!/usr/bin/env python3
import argparse
import json
import pathlib
import re
import tarfile


ALLOWED_PREFIXES = (
    "Sources/AFMKitCore/",
    "Sources/AFMOpenAICompat/",
    "Sources/AFMKitApple/",
    "Tests/AFMKitCoreTests/",
    "Tests/AFMOpenAICompatTests/",
    "Tests/AFMKitAppleTests/",
    "Packages/AFMKitDwarfStar/Sources/",
    "Packages/AFMKitDwarfStar/Tests/",
    "Packages/AFMKitDwarfStar/vendor/",
    "Packages/AFMKitMLX/Sources/",
    "Packages/AFMKitMLX/Tests/",
    "docs/api-baselines/",
)
MAX_FILE_SIZE = 16 * 1024 * 1024
MAX_TOTAL_SIZE = 96 * 1024 * 1024


def validated_name(raw_name: str) -> pathlib.PurePosixPath:
    if raw_name.startswith("/") or "\\" in raw_name:
        raise SystemExit(f"Unsafe qualification archive path: {raw_name}")
    path = pathlib.PurePosixPath(raw_name)
    if not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise SystemExit(f"Unsafe qualification archive path: {raw_name}")
    normalized = path.as_posix()
    if not any(normalized.startswith(prefix) for prefix in ALLOWED_PREFIXES):
        raise SystemExit(f"Unexpected qualification archive path: {raw_name}")
    if path.name == "Package.swift" or re.fullmatch(r"Package@swift-.*\.swift", path.name):
        raise SystemExit(f"Candidate package manifests are not qualification inputs: {raw_name}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=pathlib.Path, required=True)
    parser.add_argument("--destination", type=pathlib.Path, required=True)
    parser.add_argument("--expected-run-id", type=int, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--expected-repository", required=True)
    arguments = parser.parse_args()

    artifact = arguments.artifact.resolve()
    destination = arguments.destination.resolve()
    metadata = json.loads((artifact / "metadata.json").read_text(encoding="utf-8"))
    expected = {
        "repository": arguments.expected_repository,
        "run_id": arguments.expected_run_id,
        "sha": arguments.expected_sha,
    }
    if metadata != expected:
        raise SystemExit(f"Qualification artifact provenance mismatch: {metadata!r}")
    if not re.fullmatch(r"[0-9a-f]{40}", arguments.expected_sha):
        raise SystemExit("Expected qualification SHA must be a full commit SHA.")

    destination.mkdir(parents=True, exist_ok=False)
    seen: set[str] = set()
    total_size = 0
    with tarfile.open(artifact / "candidate.tar", "r:") as archive:
        for member in archive:
            if not member.isfile():
                raise SystemExit(
                    f"Qualification archives permit regular files only: {member.name}"
                )
            relative = validated_name(member.name)
            normalized = relative.as_posix()
            if normalized in seen:
                raise SystemExit(f"Duplicate qualification archive path: {normalized}")
            seen.add(normalized)
            if member.size > MAX_FILE_SIZE:
                raise SystemExit(f"Oversized qualification archive file: {normalized}")
            total_size += member.size
            if total_size > MAX_TOTAL_SIZE:
                raise SystemExit("Qualification archive exceeds the uncompressed size limit.")
            source = archive.extractfile(member)
            if source is None:
                raise SystemExit(f"Unreadable qualification archive file: {normalized}")
            output = destination / pathlib.Path(*relative.parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("xb") as handle:
                handle.write(source.read())
            output.chmod(0o644)

    required = {
        "Sources/AFMKitCore/AFMCoreTypes.swift",
        "Packages/AFMKitDwarfStar/Sources/AFMKitDwarfStar/AFMDwarfStarProvider.swift",
        "Packages/AFMKitMLX/Sources/AFMKitMLX/AFMMLXProvider.swift",
        "docs/api-baselines/AFMKitCore.symbols.json",
        "docs/api-baselines/AFMKitDwarfStar.symbols.json",
        "docs/api-baselines/AFMKitMLX.symbols.json",
    }
    missing = required - seen
    if missing:
        raise SystemExit(f"Qualification artifact is incomplete: {sorted(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
