"""CPU-only regressions for the paired prefill screen; no files or GPU needed."""
import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("quality", Path(__file__).parents[1] / "qwen-next-prefill-quality.py")
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


class PrefillQualityTests(unittest.TestCase):
    fixture = {"request_id": "task-a", "file": "cache.swift"}

    def response(self, **changes):
        return {**self.fixture, "diagnosis": "Cause", "fix": "Repair", "test": "Regression", **changes}

    def test_correct_object(self):
        self.assertTrue(quality.score(json.dumps(self.response()), self.fixture)["strict_ok"])

    def test_wrong_file_and_identity_are_not_quality_passes(self):
        for changes in ({"file": "other.swift"}, {"request_id": "task-b"}):
            result = quality.score(json.dumps(self.response(**changes)), self.fixture)
            self.assertFalse(result["strict_ok"])
            self.assertFalse(result["identity_ok"])

    def test_duplicate_keys_fail_even_when_identity_matches(self):
        text = json.dumps(self.response())[:-1] + ', "fix": "Second repair"}'
        result = quality.score(text, self.fixture)
        self.assertTrue(result["identity_ok"])
        self.assertFalse(result["strict_ok"])

    def test_fences_missing_keys_and_nonstring_values_fail(self):
        missing = self.response()
        del missing["test"]
        for text in ("```json\n" + json.dumps(self.response()) + "\n```",
                     json.dumps(missing), json.dumps(self.response(test=[])),
                     json.dumps(self.response(fix="")), "[]", "null"):
            self.assertFalse(quality.score(text, self.fixture)["strict_ok"])

    def test_frozen_launch_only_changes_binary_and_adds_step(self):
        frozen = ["/usr/bin/env", "AFM_TEST=1", "/old/afm", "mlx", "-m", "/exact/model",
                  "--concurrent", "1", "--mtp", "--mtp-depth", "3"]
        original = frozen.copy()
        changed = quality.launch_argv(frozen, Path("/new/afm"), 8192)
        self.assertEqual(frozen, original)
        self.assertEqual(changed[2], "/new/afm")
        self.assertEqual(changed[3:-2], original[3:])
        self.assertEqual(changed[-2:], ["--prefill-step-size", "8192"])

    def test_cache_concurrency_and_existing_step_rejected(self):
        base = ["/old/afm", "mlx", "--concurrent", "1"]
        for frozen in (base + ["--enable-prefix-caching"], base[:-1] + ["15"],
                       base + ["--prefill-step-size", "4096"]):
            with self.assertRaises(ValueError):
                quality.launch_argv(frozen, Path("/new/afm"), 8192)


if __name__ == "__main__":
    unittest.main()
