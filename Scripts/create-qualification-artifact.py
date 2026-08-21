#!/usr/bin/env python3
import argparse
import json
import os
import pathlib
import re
import stat
import tarfile


ARCHIVE_ROOTS = (
    "Sources/AFMKitCore",
    "Sources/AFMOpenAICompat",
    "Sources/AFMKitApple",
    "Tests/AFMKitCoreTests",
    "Tests/AFMOpenAICompatTests",
    "Tests/AFMKitAppleTests",
    "Packages/AFMKitDwarfStar/Sources",
    "Packages/AFMKitDwarfStar/Tests",
    "Packages/AFMKitDwarfStar/vendor",
    "Packages/AFMKitMLX/Sources",
    "Packages/AFMKitMLX/Tests",
    "docs/api-baselines",
)


def regular_files(root: pathlib.Path) -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for relative_root in ARCHIVE_ROOTS:
        source_root = root / relative_root
        if not source_root.is_dir():
            raise SystemExit(f"Missing qualification source root: {relative_root}")
        for directory, directory_names, file_names in os.walk(source_root):
            directory_names.sort()
            file_names.sort()
            directory_path = pathlib.Path(directory)
            for name in directory_names + file_names:
                path = directory_path / name
                mode = path.lstat().st_mode
                if stat.S_ISLNK(mode):
                    raise SystemExit(f"Qualification artifacts reject symlink: {path}")
                if name in file_names:
                    if not stat.S_ISREG(mode):
                        raise SystemExit(f"Qualification artifacts require regular files: {path}")
                    files.append(path)
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--repository", required=True)
    arguments = parser.parse_args()

    root = arguments.root.resolve()
    output = arguments.output.resolve()
    if not root.is_dir():
        raise SystemExit(f"Missing repository root: {root}")
    if not re.fullmatch(r"[0-9a-f]{40}", arguments.sha):
        raise SystemExit("Qualification artifact SHA must be a full commit SHA.")

    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "repository": arguments.repository,
        "run_id": arguments.run_id,
        "sha": arguments.sha,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with tarfile.open(output / "candidate.tar", "w", format=tarfile.PAX_FORMAT) as archive:
        for path in regular_files(root):
            relative = path.relative_to(root)
            info = archive.gettarinfo(str(path), arcname=relative.as_posix())
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            info.mode = 0o644
            with path.open("rb") as handle:
                archive.addfile(info, handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
