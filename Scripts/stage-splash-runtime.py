#!/usr/bin/env python3
"""Stage the pinned upstream binary release. Never starts Splash or downloads models."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PIN = ROOT / "Sources/AFMKitSplash/Resources/splash-release.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(root, pin):
    identity = json.loads((root / "release.json").read_text())
    if identity["version"] != pin["version"]:
        raise ValueError("Splash release version mismatch")
    for path, field in [("engine/splash", "binary_sha256"), ("engine/splash.metallib", "metallib_sha256")]:
        if sha256(root / path) != identity[field]:
            raise ValueError(f"Splash checksum mismatch: {path}")
    for path in ["python/bin/python3", "install/launcher.py", "LICENSE"]:
        if not (root / path).is_file():
            raise ValueError(f"Splash distribution missing {path}")


def stage(destination, cache, archive=None):
    pin = json.loads(PIN.read_text())
    destination = destination.resolve()
    marker = destination / "afm-release-pin.json"
    if marker.is_file() and json.loads(marker.read_text()) == pin:
        verify(destination, pin)
        print(f"Splash {pin['version']} already staged: {destination}")
        return
    if destination.exists():
        raise ValueError(f"Refusing to replace an unverified/different runtime at {destination}; remove it explicitly first")
    cache.mkdir(parents=True, exist_ok=True)
    downloaded = archive or cache / f"{pin['directory']}.tar.gz"
    if not downloaded.is_file() or sha256(downloaded) != pin["sha256"]:
        if archive:
            raise ValueError("Supplied Splash archive checksum mismatch")
        with tempfile.TemporaryDirectory(dir=cache, prefix="download-") as temporary:
            candidate = Path(temporary) / "release.tar.gz"
            subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--output", str(candidate), pin["url"]], check=True)
            if sha256(candidate) != pin["sha256"]:
                raise ValueError("Downloaded Splash archive checksum mismatch")
            candidate.replace(downloaded)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".splash-stage-") as temporary:
        temporary = Path(temporary)
        with tarfile.open(downloaded) as source:
            # data_filter prevents traversal and escaping symlinks, while allowing
            # the bundled Python's internal relative symlinks.
            if not hasattr(tarfile, "data_filter"):
                raise RuntimeError("Staging Splash requires Python with tarfile.data_filter (Python 3.12+ or a security backport)")
            source.extractall(temporary, filter="data")
        root = temporary / pin["directory"]
        verify(root, pin)
        shutil.copyfile(PIN, root / "afm-release-pin.json")
        root.rename(destination)
    print(f"Staged Splash {pin['version']} ({pin['revision']}) at {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--cache", type=Path, default=ROOT / ".build-splash-downloads")
    parser.add_argument("--archive", type=Path, help="Use a pre-downloaded archive; checksum still mandatory")
    args = parser.parse_args()
    stage(args.destination, args.cache, args.archive)
