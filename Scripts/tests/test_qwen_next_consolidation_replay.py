"""CPU-only consolidation replay checks; no inference or file mutation."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "replay", Path(__file__).parents[1] / "qwen-next-consolidation-replay.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class ConsolidationReplayTests(unittest.TestCase):
    def test_only_binary_changes(self):
        argv = ["/usr/bin/env", "AFM_QWEN_MTP_TEST=1", "/old/afm", "mlx",
                "-m", "/exact/checkpoint", "--concurrent", "1", "--mtp", "--mtp-depth", "3"]
        result = replay.replacement_command(argv, Path("/new/afm"))
        self.assertEqual(argv[2], "/old/afm")
        self.assertEqual(result, argv[:2] + ["/new/afm"] + argv[3:])

    def test_instrumentation_cache_and_concurrency_rejected(self):
        base = ["/old/afm", "mlx", "--concurrent", "1"]
        for argv in (["/usr/bin/env", "AFM_PERF=1"] + base,
                     base + ["--enable-prefix-caching"], base[:-1] + ["15"]):
            with self.assertRaises(ValueError):
                replay.replacement_command(argv, Path("/new/afm"))

    def test_ambiguous_executable_rejected(self):
        with self.assertRaises(ValueError):
            replay.replacement_command(["/one/afm", "/two/afm"], Path("/new/afm"))


if __name__ == "__main__":
    unittest.main()
