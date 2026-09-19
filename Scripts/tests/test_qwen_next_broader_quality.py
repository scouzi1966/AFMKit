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

    def test_serial_replay_is_explicit_and_ar_only(self):
        base = ["/usr/bin/env", "AFM_TEST=1", "/old/afm", "mlx", "--concurrent", "1"]
        enabled = q.command(base, Path("/new/afm"), 1, True, True)
        self.assertEqual(enabled[1], "AFM_PREFIX_REPLAY_BOUNDARIES=1")
        self.assertNotIn("AFM_PREFIX_REPLAY_BOUNDARIES=1", q.command(base, Path("/new/afm"), 1, True))
        for argv, concurrency, prefix in ((base, 15, True), (base, 1, False), (base + ["--mtp"], 1, True)):
            with self.assertRaises(ValueError):
                q.command(argv, Path("/new/afm"), concurrency, prefix, True)

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

    def test_default_groups_keep_original_phase_order(self):
        groups = q.request_groups(q.cases(), True)
        self.assertEqual([(p, k, len(c)) for p, k, c in groups],
                         [("first", "greedy", 15), ("first", "sampled", 30),
                          ("repeat", "greedy", 15), ("repeat", "sampled", 30)])

    def test_frozen_ar_profile_only_replaces_binary(self):
        base = ['/usr/bin/env', 'AFM_QWEN_BATCH_BANKED_ATTENTION=1', '/old/afm',
                'mlx', '-m', '/exact/model', '--concurrent', '15', '--port', '9998',
                '--no-think', '--enable-prefix-caching', '--prefill-step-size', '8192']
        expected = [*base]
        expected[2] = '/new/afm'
        self.assertEqual(q.profile_command(base, Path('/new/afm'), Path('/exact/model'),
                                          False, 15, True), expected)
        for model, mtp, concurrency, prefix in (('/other/model', False, 15, True),
                ('/exact/model', True, 15, True), ('/exact/model', False, 1, True),
                ('/exact/model', False, 15, False)):
            with self.assertRaises(ValueError):
                q.profile_command(base, Path('/new/afm'), Path(model), mtp, concurrency, prefix)

    def test_reference_c15_cache_budget_is_explicit(self):
        base = ['/old/ref', '--mtp', '--prefix-cache-entries', '0', '--prefix-cache-disk', 'off',
                '--tokenize-cache-entries', '0', '--kv-quant', 'off', '--max-concurrent', '1',
                '--top-k', '0', '--mtp-depth', '3']
        result = q.reference_command(base, Path('/new/ref'), True, 15, True)
        self.assertEqual(result[result.index('--max-concurrent') + 1], '15')
        self.assertEqual(result[result.index('--prefix-cache-entries') + 1], '16')
        self.assertEqual(result[-2:], ['--prefix-cache-mem', '4GB'])
        self.assertEqual(base[3], '0')

    def test_two_slot_server_does_not_create_concurrent_client_requests(self):
        base = ['/usr/bin/env', 'AFM_QWEN_MTP_SCHEDULER=1', '/old/afm', 'mlx',
                '--concurrent', '1', '--mtp', '--mtp-depth', '3']
        result = q.command(base, Path('/new/afm'), 2, True)
        self.assertEqual(result[result.index('--concurrent') + 1], '2')
        self.assertEqual(result[-1], '--enable-prefix-caching')
        self.assertEqual(base[5], '1')
        self.assertEqual(len(q.cases()), 45)

    def test_windowed_replay_preserves_every_payload_and_seed(self):
        cases = q.cases()
        groups = q.request_groups(cases, True, 15)
        self.assertEqual(len(groups), 6)
        for start in (0, 2, 4):
            first, repeated = groups[start:start + 2]
            self.assertEqual(first[0], "first")
            self.assertEqual(repeated[0], "repeat")
            self.assertEqual(first[1:], repeated[1:])
            self.assertEqual(len(first[2]), 15)
        for phase in ("first", "repeat"):
            self.assertEqual([c for p, k, group in groups if p == phase for c in group], cases)
        for repeat, size in ((False, 15), (True, 16), (True, -1)):
            with self.assertRaises(ValueError):
                q.request_groups(cases, repeat, size)


if __name__ == "__main__":
    unittest.main()
