"""CPU-only release staging tests; no network and no executable is launched."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("staging", ROOT / "Scripts/stage-splash-runtime.py")
staging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(staging)


class StagingTests(unittest.TestCase):
    def setUp(self):
        work = ROOT / ".build-splash-staging-tests"
        work.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=work)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.payload = self.root / "splash-fixture"
        for name in ["engine/splash", "engine/splash.metallib", "python/bin/python3", "install/launcher.py", "LICENSE"]:
            path = self.payload / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("inert fixture")
        digest = hashlib.sha256(b"inert fixture").hexdigest()
        (self.payload / "release.json").write_text(json.dumps(dict(version="fixture", binary_sha256=digest, metallib_sha256=digest)))
        self.archive = self.root / "release.tar.gz"
        with tarfile.open(self.archive, "w:gz") as archive:
            archive.add(self.payload, arcname="splash-fixture")
        self.pin = dict(version="fixture", revision="a" * 40, protocolVersion=5, url="https://invalid.invalid", sha256=staging.sha256(self.archive), directory="splash-fixture")
        self.pin_path = self.root / "pin.json"
        self.pin_path.write_text(json.dumps(self.pin))
        original = staging.PIN
        staging.PIN = self.pin_path
        self.addCleanup(setattr, staging, "PIN", original)
        self.destination = self.root / "installed"

    def stage(self):
        staging.stage(self.destination, self.root / "cache", self.archive)

    def testVerifiedArchiveStagesAndIsIdempotent(self):
        self.stage()
        self.stage()
        self.assertEqual(json.loads((self.destination / "afm-release-pin.json").read_text()), self.pin)

    def testRejectsChangedArchive(self):
        self.archive.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.stage()
        self.assertFalse(self.destination.exists())

    def testRejectsChangedEngineOnReuse(self):
        self.stage()
        (self.destination / "engine/splash").write_text("changed")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.stage()

    def testRefusesToOverwriteUnrelatedDirectory(self):
        self.destination.mkdir()
        with self.assertRaisesRegex(ValueError, "Refusing to replace"):
            self.stage()


if __name__ == "__main__":
    unittest.main()
