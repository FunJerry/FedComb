"""修正型原算法：独立数值断言、旧模型隔离，以及资源/Greedy 边界。"""
from dataclasses import astuple
from pathlib import Path
from types import SimpleNamespace
import random
import unittest
from unittest.mock import Mock, patch
import numpy as np
import torch
from fedcomb.config import load_config, apply_overrides
from fedcomb.core.types import Plan, Cost, Resources
from fedcomb.federation.client import ClientTrainer, ClientUpdate, serialize_components
from fedcomb.federation.round import RoundRunner
from fedcomb.models.network import ModelBank
from fedcomb.simulation.resources import ResourceState, fluctuate
from fedcomb.simulation.costs import ComponentCost
from fedcomb.strategies.allocation import Allocator, Allocation
from fedcomb.strategies.aggregation import aggregate
from fedcomb.tracking.progress import ProgressTracker


ROOT = Path(__file__).resolve().parents[1]


class CorrectedAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT / 'configs/smoke.json')
        self.resource = Resources(100, 1e9, 1e6)

    def test_resource_joint_probability_amplitudes_and_no_clipping(self):
        config = self.cfg.environment
        original = Resources(config.memory_max_mb * 1e6, 10e9, 20e6 / 8)
        state = ResourceState(original, original)
        rng = Mock()
        rng.random.return_value = .19
        rng.uniform.side_effect = [1.2, 1.3, 1.25]
        self.assertTrue(fluctuate(state, config, rng))
        np.testing.assert_allclose(astuple(state.actual), [v * m for v, m in zip(astuple(original), [1.2, 1.3, 1.25])])
        self.assertEqual(state.reported, original)  # 资源变化不冒充服务器已收到报告。
        self.assertGreater(state.actual.memory_bytes, config.memory_max_mb * 1e6)
        self.assertEqual([c.args for c in rng.uniform.call_args_list], [(.8, 1.2), (.7, 1.3), (.75, 1.25)])
        rng.random.return_value = .2
        unchanged = state.actual
        self.assertFalse(fluctuate(state, config, rng))
        self.assertEqual(state.actual, unchanged)
        cfg = apply_overrides(self.cfg, {'environment.change_probability': 0})
        rng = Mock()
        self.assertFalse(fluctuate(state, cfg.environment, rng))
        rng.random.assert_not_called()

    def test_only_selected_resources_change_before_allocation(self):
        server = SimpleNamespace(bank=None, progress=ProgressTracker(4), device=torch.device('cpu'))
        runner = RoundRunner(self.cfg, server, {}, {})
        runner.participant_rng = Mock()
        runner.participant_rng.sample.return_value = [2, 0]
        with patch('fedcomb.federation.round.fluctuate') as fluctuate_mock:
            self.assertEqual(runner.update_resources_and_select_clients(), [2, 0])
        self.assertEqual([c.args[0] for c in fluctuate_mock.call_args_list],
                         [runner.resources[2], runner.resources[0]])

    def test_greedy_last_start_short_intervals_and_stable_tie(self):
        cfg = apply_overrides(self.cfg, {'allocation.policy': 'greedy'})
        allocator = Allocator(ProgressTracker(4), {}, cfg)
        allocator.cost = Mock(return_value=Cost(50, 0, 0, 0))
        rng = Mock()
        rng.randint.return_value = 15
        selected = allocator.select(0, self.resource, rng)
        self.assertEqual((selected.plan.start, selected.plan.end), (15, 15))
        rng.randint.assert_called_once_with(0, 15)
        rng.randint.return_value = 14
        # 两个区间包大小相等时，与原作者的严格 > 一致，保留先扫描到的区间。
        selected = allocator.select(0, self.resource, rng)
        self.assertEqual((selected.plan.start, selected.plan.end), (14, 14))

    def test_maturity_score_uses_queue_mean_and_counts_extra_student(self):
        cfg = apply_overrides(self.cfg, {'distillation.weight_mode': 'fixed'})
        state = ProgressTracker(4)
        state.maturity[:] = .2
        state.maturity_queue[:] = 2
        allocator = Allocator(state, {}, cfg)
        allocator.normalizer = 10
        plan = Plan(3, 5, 'tail', 4)
        terms = allocator.score_terms(0, self.resource, plan, Cost(60, 0, 1, 10))
        # delta=mean(M)=2；3 个 B：2*(1.8-.5)，3 个 H/T：2*(1-.25)。
        self.assertAlmostEqual(terms['maturity'], 2 * (3 * 2.6 + 3 * 1.5) / 6)
        allocator.commit(plan, student_trained=True)
        self.assertEqual(allocator.counts[32 + 4], .5)
        np.testing.assert_array_equal(allocator.covered[3:6], [1, 1, 1])
        allocator.begin_round()
        self.assertEqual(allocator.covered.sum(), 0)
        self.assertEqual(allocator.accepted, 0)

    def test_coverage_rewards_underserved_shards_not_missing_them(self):
        state = ProgressTracker(4)
        state.coverage_queue[3], state.coverage_queue[8] = 10, 1
        allocator = Allocator(state, {}, self.cfg)
        allocator.normalizer = 1
        cost = Cost(1, 0, 0, 0)
        a = allocator.score_terms(0, self.resource, Plan(3, 5), cost)['coverage']
        b = allocator.score_terms(0, self.resource, Plan(8, 10), cost)['coverage']
        self.assertLess(a, b)
        self.assertAlmostEqual(a, 6 * (.5 * 11 - 10))
        allocator.commit(Plan(3, 5))
        self.assertEqual(allocator.covered[3], 1)

    def test_time_reference_is_common_and_first_compute_is_nonzero(self):
        table = {'stem': ComponentCost(4, 2), 'classifier': ComponentCost(4, 2)}
        for kind in ('block', 'head', 'tail'):
            table.update({f'{kind}.{i}': ComponentCost(4, 2) for i in range(16)})
        a = Allocator(ProgressTracker(4), table, self.cfg)
        expected = 2 * 72 / (self.cfg.environment.bandwidth_min_mbps * 1e6 / 8)
        expected += 3 * 8 * 36 / (self.cfg.environment.compute_min_gflops * 1e9)
        self.assertAlmostEqual(a.normalizer, expected)
        self.assertGreater(a.cost(Plan(0, 2), self.resource).compute_seconds, 0)
        b = Allocator(ProgressTracker(4), table, self.cfg)
        a.select(0, Resources(1e9, 1e9, 1e6), random.Random(1))
        b.select(1, Resources(1e9, 2e9, 2e6), random.Random(1))
        self.assertEqual(a.normalizer, b.normalizer)
        observed = a.observed_time
        old = a.normalizer
        a.finish_round()
        self.assertAlmostEqual(a.normalizer, .9 * old + .1 * observed)

    def test_intra_training_loss_mean_student_counts_and_selected_denominator(self):
        state = ProgressTracker(4)
        plan = Plan(3, 5, 'tail', 4)
        def update(client, delta, kd_steps):
            return SimpleNamespace(client_id=client, plan=plan, samples=8, task_loss=999,
                delta_loss=delta, distillation_steps=kd_steps,
                cost=Cost(50, 0, 0, 0), resources=self.resource)
        state.update([update(0, 1, 1), update(1, 3, 1)], self.cfg.progress, participants=4)
        self.assertAlmostEqual(state.ema[3], .9 * 2)  # 不再额外 / 2，也不依赖跨轮历史。
        self.assertEqual(state.counts[32 + 4], 1)
        self.assertEqual(state.coverage_queue[3], 0)  # 2/4 == target .5。
        state = ProgressTracker(4)
        state.update([update(0, 1, 0)], self.cfg.progress, participants=4)
        self.assertEqual(state.counts[32 + 4], 0)
        self.assertEqual(state.ema[32 + 4], 0)
        self.assertEqual(state.coverage_queue[3], .25)
        counts = state.counts.copy()
        state.update([], self.cfg.progress, participants=4)
        np.testing.assert_equal(counts, state.counts)
        self.assertEqual(state.coverage_queue[3], .75)

    def test_previous_success_then_rejection_does_not_reaggregate_old_model(self):
        state = ProgressTracker(4)
        module = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            module.weight.zero_()
        bank = SimpleNamespace(components=lambda: {'stem': module})
        server = SimpleNamespace(bank=bank, progress=state, device=torch.device('cpu'))
        runner = RoundRunner(self.cfg, server, {}, {0: None})
        runner.resources[0] = ResourceState(self.resource, self.resource)
        plan, cost = Plan(0, 2), Cost(50, 0, 0, 0)
        update = ClientUpdate(0, plan, {'stem': {'weight': torch.tensor([[2.]])}}, {},
                              1, None, 8, 2, 0, 0, 0, cost, self.resource, None)
        runner.allocator.select = Mock(side_effect=[Allocation(plan, cost, 1, {}), None])
        runner.allocator.last_rejection = {'reason': 'no_feasible_candidate'}
        runner.allocator.cost = Mock(return_value=cost)
        runner.trainer.train = Mock(return_value=update)
        first, _, _, _ = runner.train_selected_clients([0], 0)
        aggregate(bank, first, self.cfg.aggregation)
        before = module.weight.detach().clone()
        second, records, skipped, _ = runner.train_selected_clients([0], 1)
        self.assertEqual((second, records), ([], []))
        self.assertEqual(len(skipped), 1)
        aggregate(bank, second, self.cfg.aggregation)
        self.assertTrue(torch.equal(before, module.weight))
        runner.trainer.train.assert_called_once()

    def test_client_delta_is_first_last_batch_not_average_or_previous_round(self):
        class Dataset:
            def __len__(self):
                return 8
        class Loader:
            dataset = Dataset()
            batch_sampler = SimpleNamespace(generator=torch.Generator())
            def __iter__(self):
                return iter([(None, torch.zeros(4, dtype=torch.long))] * 2)
        trainer = ClientTrainer(None, None, self.cfg, torch.device('cpu'))
        trainer.prepare = Mock(return_value=SimpleNamespace(weight=0))
        trainer.train_batch = Mock(side_effect=[(1., None, {}), (3., None, {})])
        allocation = SimpleNamespace(plan=Plan(0, 2), probabilities=None)
        with patch('fedcomb.federation.client.serialize_components', return_value={}):
            update = trainer.train(0, allocation, Loader(), 0, Cost(1, 0, 0, 0), self.resource)
        self.assertEqual(update.delta_loss, 2)
        self.assertEqual(update.task_loss, 2)
        self.assertEqual(update.steps, 2)

    def test_zero_weight_student_not_uploaded_as_an_update(self):
        module = torch.nn.Linear(1, 1)
        model = SimpleNamespace(plan=Plan(0, 2, 'tail', 1), component=lambda key: module)
        session = SimpleNamespace(model=model, student=module, weight=0)
        self.assertNotIn('tail.1', serialize_components(session))
        session.weight = 1
        self.assertIn('tail.1', serialize_components(session))

    def test_short_greedy_paths_really_train_with_head_and_tail_students(self):
        torch.set_num_threads(2)
        cfg = apply_overrides(self.cfg, {'distillation.weight_mode': 'fixed'})
        bank = ModelBank(cfg.model)
        trainer = ClientTrainer(bank, ProgressTracker(4), cfg, torch.device('cpu'))
        images, labels = torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 2, 0])
        for plan in (Plan(15, 15, 'tail', 15), Plan(14, 15, 'head', 14), Plan(0, 0, 'head', 0)):
            with self.subTest(plan=plan):
                session = trainer.prepare(plan, 0)
                task, kd, _ = trainer.train_batch(session, images, labels)
                self.assertTrue(np.isfinite(task) and np.isfinite(kd))
                self.assertIn(plan.student_key(), serialize_components(session))


if __name__ == '__main__':
    unittest.main()
