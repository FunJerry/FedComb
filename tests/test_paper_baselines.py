"""论文基线：一次随机抽样、固定起点扩展，以及拒绝后不训练。"""
from pathlib import Path
from types import SimpleNamespace
import json
import random
import tempfile
import unittest
from unittest.mock import Mock, call, patch
import torch
from fedcomb.config import load_config, apply_overrides
from fedcomb.core.types import Cost, Resources
from fedcomb.federation.round import RoundRunner
from fedcomb.experiment import Experiment
from fedcomb.simulation.resources import ResourceState
from fedcomb.strategies.allocation import Allocator
from fedcomb.tracking.progress import ProgressTracker

ROOT = Path(__file__).resolve().parents[1]
RESOURCE = Resources(100, 1e9, 1e6)


class PaperBaselineTests(unittest.TestCase):
    def allocator(self, policy):
        cfg = apply_overrides(load_config(ROOT / 'configs/smoke.json'), {'allocation.policy': policy})
        return Allocator(ProgressTracker(cfg.federation.clients), {}, cfg)

    def test_random_rejects_one_draw_without_screening_or_retry(self):
        allocator = self.allocator('random')
        allocator.cost = Mock(side_effect=lambda plan, resource: Cost(
            40 if plan.end - plan.start + 1 == 3 else 200, 0, 0, 0))
        allocator.candidates = Mock(side_effect=AssertionError('不得先筛可行域'))
        rng = Mock()
        rng.randint.side_effect = [16, 0]
        self.assertIsNone(allocator.select(0, RESOURCE, rng))
        self.assertEqual(rng.randint.call_args_list, [call(3, 16), call(0, 0)])
        self.assertEqual(allocator.cost.call_count, 1)
        self.assertEqual(allocator.last_rejection['reason'], 'random_interval_exceeds_reported_budget')
        self.assertEqual(allocator.last_rejection['plan']['end'], 15)
        rng.randint.side_effect = [3, 4]
        selected = allocator.select(0, RESOURCE, rng)
        self.assertEqual((selected.plan.start, selected.plan.end), (4, 6))
        self.assertIsNotNone(selected.plan.student_kind)
        self.assertIsNone(selected.feasible_count)
        self.assertIsNone(allocator.last_rejection)

    def test_random_matches_author_length_then_start_sampling(self):
        allocator = self.allocator('random')
        allocator.cost = Mock(return_value=Cost(1, 0, 0, 0))
        for seed in range(20):
            expected = random.Random(seed)
            length = expected.randint(3, 16)
            start = expected.randint(0, 16 - length)
            selected = allocator.select(0, RESOURCE, random.Random(seed))
            self.assertEqual((selected.plan.start, selected.plan.end), (start, start + length - 1))

    def test_greedy_keeps_start_and_maximizes_bytes_not_length(self):
        allocator = self.allocator('greedy')
        # 模型包成本不必单调：端点 11 超预算，端点 15 仍可行；端点 10 的包更大。
        allocator.cost = Mock(side_effect=lambda plan, resource: Cost(
            {10: 90, 11: 200, 15: 50}.get(plan.end, 40), 0, 0, 0))
        rng = Mock()
        rng.randint.return_value = 8
        selected = allocator.select(0, RESOURCE, rng)
        rng.randint.assert_called_once_with(0, 15)
        self.assertEqual((selected.plan.start, selected.plan.end), (8, 10))
        self.assertEqual(selected.cost.weight_bytes, 90)
        self.assertTrue(all(args[0].start == 8 for args, _ in allocator.cost.call_args_list))

    def test_greedy_no_feasible_interval_does_not_change_start(self):
        allocator = self.allocator('greedy')
        allocator.cost = Mock(side_effect=lambda plan, resource: Cost(
            200 if plan.start == 13 else 1, 0, 0, 0))
        rng = Mock()
        rng.randint.return_value = 13
        self.assertIsNone(allocator.select(0, RESOURCE, rng))
        rng.randint.assert_called_once_with(0, 15)
        self.assertEqual(allocator.last_rejection['start'], 13)

    def test_rejected_client_is_not_trained_or_counted(self):
        allocator = self.allocator('random')
        server = SimpleNamespace(bank=None, progress=allocator.progress, device=torch.device('cpu'))
        runner = RoundRunner(allocator.config, server, {}, {})
        runner.resources[0] = ResourceState(RESOURCE, RESOURCE)
        runner.allocation_rng = Mock()
        runner.allocation_rng.randint.side_effect = [16, 0]
        runner.allocator.cost = Mock(return_value=Cost(200, 0, 0, 0))
        with patch.object(runner.trainer, 'train') as train, patch.object(runner.allocator, 'commit') as commit:
            updates, records, skipped, _ = runner.train_selected_clients([0], 0)
        train.assert_not_called()
        commit.assert_not_called()
        self.assertEqual((updates, records), ([], []))
        self.assertEqual(skipped[0]['reason'], 'random_interval_exceeds_reported_budget')
        self.assertEqual(server.progress.counts.sum(), 0)

    def test_ipc_still_uses_explicit_feasible_sampling(self):
        cfg = load_config(ROOT / 'configs/smoke.json')
        for aggregation in ('ipc', 'gradient_mean'):
            with self.assertRaises(ValueError):
                apply_overrides(cfg, {'allocation.policy': 'random',
                    'aggregation.method': aggregation, 'environment.reporting': 'fresh'})
            valid = apply_overrides(cfg, {'allocation.policy': 'random_feasible',
                'aggregation.method': aggregation, 'environment.reporting': 'fresh'})
            allocator = Allocator(ProgressTracker(valid.federation.clients), {}, valid)
            allocator.cost = Mock(return_value=Cost(1, 0, 0, 0))
            selected = allocator.select(0, RESOURCE, random.Random(42))
            self.assertEqual(selected.feasible_count, 105)
            self.assertTrue(0 < selected.probabilities[selected.plan.start] <= 1)

    def test_both_baselines_complete_small_training(self):
        # 小模型合成数据仅验证路径可执行，不作为 CIFAR-10 性能实验。
        for policy in ('random', 'greedy'):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as directory:
                cfg = apply_overrides(load_config(ROOT / 'configs/smoke.json'), {
                    'allocation.policy': policy, 'federation.rounds': 2})
                output = Path(directory) / policy
                summary = Experiment(cfg, output, ROOT).run()
                self.assertEqual(summary['rounds'], 2)
                self.assertGreater(summary['training_updates'], 0)
                rows = [json.loads(line) for line in (output / 'rounds.jsonl').read_text().splitlines()]
                for row in rows:
                    for client in row['clients']:
                        minimum = 1 if policy == 'greedy' else 3
                        self.assertGreaterEqual(client['plan']['end'] - client['plan']['start'] + 1, minimum)
                        self.assertGreater(client['task_steps'], 0)


if __name__ == '__main__':
    unittest.main()
