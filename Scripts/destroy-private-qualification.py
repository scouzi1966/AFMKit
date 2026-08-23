#!/usr/bin/env python3
import argparse
import os
import pathlib
import shutil
import stat


def remove(path: pathlib.Path) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        path.unlink()
        return

    def make_writable(function, target, _error) -> None:
        os.chmod(target, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        function(target)

    shutil.rmtree(path, onerror=make_writable)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--path", type=pathlib.Path, action="append", required=True)
    arguments = parser.parse_args()

    root = arguments.root.resolve(strict=True)
    for raw_path in arguments.path:
        path = raw_path.resolve(strict=False)
        try:
            path.relative_to(root)
        except ValueError:
            raise SystemExit(f"Refusing to remove path outside cleanup root: {path}")
        if path == root:
            raise SystemExit("Refusing to remove the cleanup root itself.")
        remove(path)
        if path.exists() or path.is_symlink():
            raise SystemExit(f"Private qualification path survived cleanup: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
