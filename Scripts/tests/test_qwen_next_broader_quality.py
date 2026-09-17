"""CPU-only scorer/fixture/launch checks; generated responses are never executed."""
import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("broader", Path(__file__).parents[1] / "qwen-next-broader-quality.py")
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


class BroaderQualityTests(unittest.TestCase):
    def response(self, case):
        return {"request_id": case["request_id"], "answer": case["answer"], "evidence": [case["record"]]}

    def test_all_oracles_pass(self):
        for case in q.cases():
            with self.subTest(case=case["request_id"]):
                self.assertTrue(q.score(json.dumps(self.response(case)), case)["semantic_ok"])

    def test_cases_have_disjoint_ids_and_repeat_deterministically(self):
        cases = q.cases()
        self.assertEqual(len(cases), 45)
        self.assertEqual(len({c["request_id"] for c in cases}), 45)
        self.assertEqual(len({c["task"] for c in cases}), 15)
        self.assertEqual(sum(c["kind"] == "greedy" for c in cases), 15)
        self.assertEqual(cases, q.cases())

    def test_wrong_answer_does_not_pass_on_structure_alone(self):
        case = q.cases()[0]
        response = self.response(case)
        response["answer"] = {"hits": [True, True, True], "values": ["A", "A", "B"]}
        result = q.score(json.dumps(response), case)
        self.assertTrue(result["structure_ok"])
        self.assertTrue(result["identity_ok"])
        self.assertFalse(result["semantic_ok"])

    def test_cross_slot_identity_and_wrong_evidence_fail(self):
        case = q.cases()[0]
        for changes in ({"request_id": "other-slot"}, {"evidence": ["R02"]},
                        {"evidence": ["R01", "R02"]}, {"evidence": ["R01", "R01"]}):
            self.assertFalse(q.score(json.dumps({**self.response(case), **changes}), case)["semantic_ok"])

    def test_duplicate_keys_at_either_depth_fail(self):
        case = q.cases()[0]
        valid = json.dumps(self.response(case))
        self.assertFalse(q.score(valid[:-1] + ', "answer": {}}', case)["semantic_ok"])
        self.assertFalse(q.score(valid.replace('"hits":', '"hits": [], "hits":'), case)["semantic_ok"])

    def test_fences_missing_extra_keys_and_nonfinite_fail(self):
        case = q.cases()[0]
        valid = json.dumps(self.response(case))
        for text in ("```json\n" + valid + "\n```", "[]", "null", "NaN",
                     valid[:-1], valid[:-1] + ',"extra":true}',
                     '{"request_id":"review-0-00","answer":NaN,"evidence":["R01"]}'):
            self.assertFalse(q.score(text, case)["semantic_ok"])

    def test_boolean_is_not_numeric_one(self):
        case = q.cases()[12]
        response = self.response(case)
        response["answer"] = {"valid": 0, "reuse": 0}
        self.assertFalse(q.score(json.dumps(response), case)["semantic_ok"])

    def test_nested_object_order_irrelevant_array_order_significant(self):
        case = q.cases()[1]
        response = self.response(case)
        response["answer"] = dict(reversed(list(case["answer"].items())))
        self.assertTrue(q.score(json.dumps(response), case)["semantic_ok"])
        response["answer"] = {**case["answer"], "residents": ["B", "C"]}
        self.assertFalse(q.score(json.dumps(response), case)["semantic_ok"])

    def test_launch_preserves_mtp_settings_and_only_changes_requested_fields(self):
        frozen = ["/usr/bin/env", "AFM_TEST=1", "/old/afm", "mlx", "-m", "/exact/model",
                  "--concurrent", "1", "--mtp", "--mtp-depth", "3"]
        old = frozen.copy()
        command = q.command(frozen, Path("/new/afm"), 15, True)
        expected = [*old]
        expected[2], expected[7] = "/new/afm", "15"
        self.assertEqual(command, expected + ["--enable-prefix-caching"])
        self.assertEqual(old, frozen)

    def test_uncontrolled_baselines_rejected(self):
        base = ["/old/afm", "mlx", "--concurrent", "1"]
        for command in (base + ["--enable-prefix-caching"], base[:-1] + ["15"],
                        ["AFM_DEBUG=1", *base], [*base, "/other/afm"]):
            with self.assertRaises(ValueError):
                q.command(command, Path("/new/afm"), 1, False)

    def test_aggregate_uses_phase_wall_not_sum_of_request_times(self):
        rows = [{"runtime_ok": True, **{k: True for k in q.score("", {})},
                 "usage": {"completion_tokens": 100}, "seconds": 2,
                 "ttft": 1, "decode_tps": 99, "finish_reasons": ["stop"]} for _ in range(15)]
        summary = q.summarize(rows, 2)
        self.assertEqual(summary["aggregate_output_tok_s"], 750)
        self.assertEqual(summary["semantic_tasks_s"], 7.5)
        for row in rows:
            row["started_monotonic"] = 10
        self.assertEqual(q.summarize(rows, 2)["max_overlapping_client_requests"], 15)
        for index, row in enumerate(rows):
            row["started_monotonic"] = index * 2
        self.assertEqual(q.summarize(rows, 30)["max_overlapping_client_requests"], 1)

    def test_reference_changes_only_binary_and_mtp_switch(self):
        original = ["/old/ref", "--model", "/exact/model", "--mtp", "--prefix-cache-entries", "0",
                    "--prefix-cache-disk", "off", "--tokenize-cache-entries", "0", "--kv-quant", "off",
                    "--max-concurrent", "1", "--top-k", "0", "--mtp-depth", "3"]
        for mtp in (False, True):
            expected = [*original]
            expected[0], expected[3] = "/new/ref", "--mtp" if mtp else "--no-mtp"
            self.assertEqual(q.reference_command(original, Path("/new/ref"), mtp), expected)
        invalid = [*original]
        invalid[5] = "10"
        with self.assertRaises(ValueError):
            q.reference_command(invalid, Path("/new/ref"), False)


if __name__ == "__main__":
    unittest.main()
