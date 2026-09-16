"""CPU-only tests for paired scoring and metric separation."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("audit", Path(__file__).parents[1] / "audit-qwen-next-prefill-quality.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def row(index, strict, *, kind="sampled", seconds=2, tokens=10):
    return {"case": {"task": index, "seed": index, "kind": kind},
            "payload": {"seed": index}, "text": str(index), "strict_ok": strict,
            "identity_ok": strict, "runtime_ok": True, "seconds": seconds,
            "usage": {"completion_tokens": tokens}, "ttft": 1, "decode_tps": 10}


class PrefillAuditTests(unittest.TestCase):
    def test_pairing_records_both_gains_and_losses(self):
        old = [row(0, True), row(1, False), row(2, False), row(3, True)]
        new = [row(0, False), row(1, True), row(2, False), row(3, True)]
        result = audit.paired(old, new)
        self.assertEqual(len(result["strict_gained"]), 1)
        self.assertEqual(len(result["strict_lost"]), 1)
        self.assertEqual(result["both_strict"], 1)
        self.assertEqual(result["neither_strict"], 1)

    def test_pairing_rejects_different_payloads(self):
        old = [row(0, True)]
        new = copy.deepcopy(old)
        new[0]["payload"]["seed"] = 999
        with self.assertRaises(AssertionError):
            audit.paired(old, new)

    def test_greedy_controls_not_counted_as_sampled_evidence(self):
        rows = [row(0, True, kind="greedy"), row(1, False)]
        result = audit.paired(rows, rows)
        self.assertEqual(result["sampled_pairs"], 1)
        self.assertEqual(result["both_strict"], 0)

    def test_wall_throughput_uses_ratio_of_totals(self):
        rows = [row(0, True, kind="greedy"), row(0, True, seconds=1, tokens=100),
                row(1, False, seconds=9, tokens=90)]
        sampled = audit.q.summarize(rows)["sampled"]
        self.assertEqual(sampled["output_tokens_per_wall_second_including_prefill"], 19)
        self.assertEqual(sampled["strict_tasks_per_second"], 0.1)
        self.assertEqual(sampled["total"], 2)


if __name__ == "__main__":
    unittest.main()
