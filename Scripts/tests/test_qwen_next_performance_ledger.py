"""CPU-only ledger tests; fixtures live on the external test-artifact volume."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('ledger', Path(__file__).parents[1] / 'qwen-next-performance-ledger.py')
ledger = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ledger)


class PerformanceLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ['AFM_LEDGER_TEST_DIRECTORY'])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_fixture(self, name, rates=(50, 51, 100), *, environment=(), depth=3, prompt='prompt-a', binary='a'*64):
        run = self.root / name
        run.mkdir(parents=True)
        launch = {'argv': ['/usr/bin/env', *environment, '/build/afm', 'mlx', '-m', '/models/exact',
                           '--port', '9998', '--no-think', '--mtp', '--mtp-depth', str(depth)],
                  'request': {'temperature':0.6, 'top_p':1.0, 'seed':42},
                  'binary_sha256':binary, 'time':'2026-09-15'}
        (run / 'launch.json').write_text(json.dumps(launch))
        (run / 'complete.json').write_text('{}')
        for i, rate in enumerate(rates, 1):
            row = {'result': {'usage': {'completion_tokens':128, 'prompt_tokens':493},
                             'generated_text':'A nonempty response', 'reasoning_text':'', 'warmup':False,
                             'cached_tokens':0, 'prompt_sha256':prompt, 'finish_reasons':['length'],
                             'client_prefill_tps':1000, 'client_decode_tps':rate, 'context':'0.5', 'trial':i}}
            (run / f'trial-{i}-0.5k.json').write_text(json.dumps(row))
        return run

    def test_peak_is_run_median_not_fastest_token_rate(self):
        old, _ = ledger.index_run(self.run_fixture('old'))
        new, _ = ledger.index_run(self.run_fixture('current/new', rates=(40, 40, 40)))
        comparison = ledger.compare(old + new, self.root / 'current')
        decode = next(r for r in comparison if r['metric'] == 'client_decode_tps')
        peak = decode['same_explicit_launch']
        self.assertEqual(peak['best_median'], 51)
        self.assertEqual(peak['fastest_individual'], 100)
        self.assertAlmostEqual(peak['change_percent'], 100*(40/51-1))

    def test_different_depth_is_envelope_not_strict_match(self):
        old, _ = ledger.index_run(self.run_fixture('old', depth=4))
        new, _ = ledger.index_run(self.run_fixture('current/new', depth=3))
        for row in ledger.compare(old + new, self.root / 'current'):
            self.assertIsNotNone(row['historical_configuration_envelope'])
            self.assertIsNone(row['same_explicit_launch'])

    def test_different_prompt_cannot_claim_peak_regression(self):
        old, _ = ledger.index_run(self.run_fixture('old', prompt='other'))
        new, _ = ledger.index_run(self.run_fixture('current/new'))
        for row in ledger.compare(old + new, self.root / 'current'):
            self.assertIsNone(row['historical_configuration_envelope'])

    def test_instrumented_run_is_excluded(self):
        cells, reasons = ledger.index_run(self.run_fixture('trace', environment=['AFM_PERF=1']))
        self.assertEqual(cells, [])
        self.assertTrue(all('instrumented launch' in r['reasons'] for r in reasons))

    def test_cached_and_short_outputs_are_excluded(self):
        run = self.run_fixture('invalid')
        path = run / 'trial-1-0.5k.json'
        data = json.loads(path.read_text())
        data['result']['cached_tokens'] = 200
        data['result']['usage']['completion_tokens'] = 16
        path.write_text(json.dumps(data))
        cells, reasons = ledger.index_run(run)
        self.assertEqual(cells[0]['n'], 2)
        self.assertIn('prefix reuse', reasons[0]['reasons'])
        self.assertIn('not the 128-token context workload', reasons[0]['reasons'])

    def test_single_fast_run_cannot_set_repeatable_peak(self):
        old, _ = ledger.index_run(self.run_fixture('old', rates=(999,)))
        new, _ = ledger.index_run(self.run_fixture('current/new'))
        for row in ledger.compare(old + new, self.root / 'current'):
            self.assertIsNone(row['historical_configuration_envelope'])

    def test_actual_inference_hash_takes_precedence_over_launcher(self):
        run = self.run_fixture('hash', binary='9e3f338ac5a436322acb4821d6023faa48df0a54f12e23f806c59316226dec45')
        self.assertFalse(ledger.index_run(run)[0])
        path = run / 'launch.json'
        launch = json.loads(path.read_text())
        launch['actual_inference_binary_sha256'] = 'b'*64
        path.write_text(json.dumps(launch))
        self.assertEqual(ledger.index_run(run)[0][0]['binary_sha256'], 'b'*64)

    def test_absolute_non_model_flag_paths_are_part_of_strict_identity(self):
        launch = {'argv':['/build/afm','mlx','-m','/models/exact','--ngram-path','/sidecars/a']}
        self.assertIn('/sidecars/a', ledger.configuration(launch)[1]['flags'])


if __name__ == '__main__':
    unittest.main()
