import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('retained_lifecycle',
    Path(__file__).resolve().parents[1] / 'qwen-next-retained-lifecycle.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class RetainedLifecycleTests(unittest.TestCase):
    def row(self):
        return dict(text='OWNER_03_ISOLATED\n1\n2', usage=dict(completion_tokens=12,
                    prompt_tokens=400, prompt_tokens_details=dict(cached_tokens=400)),
                    chunks=[dict(choices=[dict(finish_reason='length', logprobs=None)])])

    def test_complete_state_and_own_identity(self):
        self.assertTrue(all(gate.checks(self.row(), 96, full_replay=True,
                                       identity='OWNER_03_ISOLATED').values()))

    def test_foreign_owner_rejected_even_with_own_identity(self):
        row = self.row()
        row['text'] += '\nOWNER_04_ISOLATED'
        self.assertFalse(gate.checks(row, 96, identity='OWNER_03_ISOLATED')['identity'])

    def test_cache_flag_alone_not_enough(self):
        row = self.row()
        row['usage']['prompt_tokens_details']['cached_tokens'] = 0
        self.assertFalse(gate.checks(row, 96, full_replay=True)['full_prompt_replay'])

    def test_partial_replay_not_complete_state(self):
        row = self.row()
        row['usage']['prompt_tokens_details']['cached_tokens'] = 399
        self.assertFalse(gate.checks(row, 96, full_replay=True)['full_prompt_replay'])

    def test_explicit_earlier_boundary_requires_exact_expected_position(self):
        row = self.row()
        for cached, expected in [(0, False), (368, False), (369, True), (370, False), (400, False)]:
            row['usage']['prompt_tokens_details']['cached_tokens'] = cached
            self.assertEqual(gate.checks(row, 96, full_replay=True, snapshot_backoff_tokens=31)
                             ['prompt_boundary_replay'], expected)

    def test_short_prompt_backoff_retains_endpoint(self):
        row = self.row()
        row['usage'].update(prompt_tokens=20, prompt_tokens_details=dict(cached_tokens=20))
        self.assertTrue(gate.checks(row, 96, full_replay=True, snapshot_backoff_tokens=31)
                        ['prompt_boundary_replay'])

    def test_endpoint_promotion_does_not_accept_arbitrary_partial_hit(self):
        row = self.row()
        for cached, expected in [(0, False), (368, False), (369, True), (370, False), (400, True)]:
            row['usage']['prompt_tokens_details']['cached_tokens'] = cached
            self.assertEqual(gate.checks(row, 96, full_replay=True, snapshot_backoff_tokens=31,
                                        allow_endpoint_promotion=True)['prompt_boundary_replay'], expected)

    def test_missing_finish_and_excess_tokens_rejected(self):
        row = self.row()
        row['chunks'] = []
        self.assertFalse(gate.checks(row, 8)['token_cap'])
        self.assertFalse(gate.checks(row, 96)['finished'])


if __name__ == '__main__':
    unittest.main()
