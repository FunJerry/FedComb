"""实验组织与关键学习机制测试；不下载真实数据，不需要 GPU。"""
from dataclasses import asdict
from pathlib import Path
import copy
import json
import random
import tempfile
import unittest
import numpy as np
import torch
from torch import nn
from fedcomb.config import load_config, apply_overrides, from_dict
from fedcomb.core.types import Plan, Resources
from fedcomb.models.network import ModelBank, LocalModel
from fedcomb.data.datasets import load_datasets, equal_size_dirichlet, make_loaders, SeededSubset
from fedcomb.strategies.distillation import select_student, compute_student_loss
from fedcomb.strategies.aggregation import aggregate_ipc_shard
from fedcomb.strategies.allocation import Allocator, Allocation, inclusion_probabilities
from fedcomb.tracking.progress import ProgressTracker
from fedcomb.simulation.costs import profile_bank
from fedcomb.federation.client import ClientTrainer
from fedcomb.experiment import Experiment
from fedcomb.suite import expand_suite


ROOT = Path(__file__).resolve().parents[1]


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cfg = load_config(ROOT / 'configs/smoke.json')
        cls.bank = ModelBank(cls.cfg.model)
        cls.table = profile_bank(cls.bank, 32)
        cls.resource = Resources(1e9, 1e9, 1e6)

    def test_grouped_configuration_and_invalid_combinations(self):
        original = asdict(self.cfg)
        changed = apply_overrides(self.cfg, {'distillation.mode': 'none', 'allocation.time_weight': 6})
        self.assertEqual(changed.distillation.mode, 'none')
        self.assertEqual(asdict(self.cfg), original)
        for overrides in [{'aggregation.method': 'ipc'}, {'data.unknown': 1},
                          {'training.local_epochs': 1.2}, {'data.download': 'true'},
                          {'distillation.temperature': float('nan')}]:
            with self.assertRaises(ValueError):
                apply_overrides(self.cfg, overrides)

    def test_suite_expansion_seeds_grid_and_no_mutation(self):
        cases = expand_suite(ROOT / 'experiments/smoke_matrix.json')
        self.assertEqual(len(cases), 18)
        self.assertEqual(len(set(name for name, _ in cases)), 18)
        self.assertEqual(cases[0][1].experiment.seed, 42)
        self.assertEqual(cases[1][1].experiment.seed, 43)
        self.assertEqual(len(expand_suite(ROOT / 'experiments/ablations.json')), 33)
        sweep = expand_suite(ROOT / 'experiments/time_weight_sweep.json')
        self.assertEqual(len(sweep), 9)
        self.assertEqual(sorted(set(c.allocation.time_weight for _, c in sweep)), [1, 3, 6])

    def test_student_modes_and_real_budget_removal(self):
        progress = ProgressTracker(4)
        for mode in ['adaptive', 'head', 'tail', 'none']:
            cfg = apply_overrides(self.cfg, {'distillation.mode': mode})
            kind, p = select_student(3, 6, progress.maturity, cfg.distillation)
            if mode == 'none':
                self.assertIsNone(kind)
            elif mode in ('head', 'tail'):
                self.assertEqual(kind, mode)
            else:
                self.assertEqual((kind, p), ('tail', 3))
        with_student = Allocator(progress, self.table, self.cfg)
        no_student_cfg = apply_overrides(self.cfg, {'distillation.mode': 'none'})
        without_student = Allocator(progress, self.table, no_student_cfg)
        a, b = Plan(3, 6, 'tail', 4), Plan(3, 6)
        self.assertEqual(with_student.cost(a, self.resource).weight_bytes
                         - without_student.cost(b, self.resource).weight_bytes,
                         self.table['tail.4'].weight_bytes)

    def test_all_105_intervals_and_feature_shapes(self):
        images = torch.randn(2, 3, 32, 32)
        progress = ProgressTracker(4)
        allocator = Allocator(progress, self.table, self.cfg)
        self.assertEqual(len(allocator.candidates(self.resource)), 105)
        self.assertIsNone(allocator.select(0, Resources(1, 1e9, 1e6), random.Random(42)))
        self.assertIsNot(self.bank.blocks[15], self.bank.tails[15])
        with torch.no_grad():
            for start in range(16):
                for end in range(start + 2, 16):
                    for kind in ('head', 'tail'):
                        plan = Plan(start, end, kind, (start + end)//2)
                        local = LocalModel(self.bank, plan).eval()
                        logits, stem, target = local(images)
                        student = self.bank.components()[plan.student_key()].eval()
                        result = student(stem if kind == 'head' else target)
                        self.assertEqual(logits.shape, (2, 3))
                        if kind == 'head':
                            self.assertEqual(result.shape, target.shape)
                        else:
                            self.assertEqual(local.component('classifier')(result).shape, logits.shape)

    def test_score_ablation_changes_only_one_term(self):
        progress = ProgressTracker(4)
        progress.resource_queue[:] = 1
        progress.maturity_queue[:] = 1
        progress.coverage_queue[:] = 1
        allocator = Allocator(progress, self.table, self.cfg)
        allocator.normalizer = 10
        plan = Plan(3, 6, 'tail', 4)
        cost = allocator.cost(plan, self.resource)
        terms = allocator.score_terms(0, self.resource, plan, cost)
        cfg = apply_overrides(self.cfg, {'allocation.coverage_weight': 0})
        ablated = Allocator(progress, self.table, cfg)
        ablated.normalizer = 10
        changed = ablated.score_terms(0, self.resource, plan, cost)
        self.assertEqual(changed['coverage'], 0)
        for key in ['resource', 'maturity', 'time']:
            self.assertEqual(terms[key], changed[key])
        allocator.commit(plan)
        self.assertEqual(allocator.covered.tolist(), [0]*3 + [1]*4 + [0]*9)

    def test_distillation_gradient_isolation(self):
        images = torch.randn(2, 3, 32, 32)
        for kind in ['head', 'tail']:
            plan = Plan(3, 6, kind, 4)
            local = LocalModel(self.bank, plan).train()
            student = copy.deepcopy(self.bank.components()[plan.student_key()]).train()
            classifier = copy.deepcopy(local.component('classifier')).eval().requires_grad_(False)
            logits, stem, target = local(images)
            compute_student_loss(student, classifier, plan, stem, target, logits,
                                 self.cfg.distillation).backward()
            self.assertTrue(all(p.grad is None for p in local.parameters()))
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters()))
            nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
            for key in plan.task_keys():
                self.assertTrue(all(p.grad is not None for p in local.component(key).parameters()))

    def test_real_head_tail_training_and_server_snapshot(self):
        train, test, labels = load_datasets(self.cfg.data, self.cfg.model, 4, 42, ROOT)
        indices = equal_size_dirichlet(labels, 4, 8, .5, 42)
        loaders, _ = make_loaders(train, test, indices, self.cfg.data, 42)
        for kind in ['head', 'tail']:
            cfg = apply_overrides(self.cfg, {'distillation.mode': kind, 'distillation.weight_mode': 'fixed'})
            tracker = ProgressTracker(4)
            trainer = ClientTrainer(self.bank, tracker, cfg, torch.device('cpu'))
            plan = Plan(3, 6, kind, 4)
            cost = Allocator(tracker, self.table, cfg).cost(plan, self.resource)
            before = copy.deepcopy(self.bank.state_dict())
            update = trainer.train(0, Allocation(plan, cost, 105, {}), loaders[0], 0, cost, self.resource)
            self.assertEqual(update.distillation_steps, 2)
            self.assertEqual(update.lambda_distill, 1)
            self.assertTrue(all(torch.equal(value, before[name]) for name, value in self.bank.state_dict().items()))
            original = self.bank.components()[plan.student_key()].state_dict()
            self.assertTrue(any(not torch.equal(value, update.states[plan.student_key()][name])
                                for name, value in original.items()))

    def test_ipc_fixed_update_expectation_and_weight(self):
        candidates = [(Plan(s, e), None) for s in range(4) for e in range(s, 4)]
        probability = inclusion_probabilities(candidates)
        for shard in range(4):
            expectation = sum(1/probability[shard] if p.start <= shard <= p.end else 0
                              for p, _ in candidates) / 10
            self.assertAlmostEqual(expectation, 1)
        from types import SimpleNamespace
        module = nn.Linear(1, 1, bias=False)
        module.weight.data.fill_(1)
        result = SimpleNamespace(samples=10, probabilities=[.25]*16,
            first_gradients={'block.0': {'weight': torch.tensor([[2.]])}},
            states={'block.0': module.state_dict()})
        other = SimpleNamespace(samples=10, first_gradients={})
        with torch.no_grad():
            aggregate_ipc_shard(module, 'block.0', 0, [result, other], .1, .1)
        self.assertAlmostEqual(module.weight.item(), .6, places=6)

    def test_data_augmentation_independent_of_training_random_stream(self):
        class RandomImageDataset:
            def __getitem__(self, index):
                return torch.rand(3, 4, 4), index
        subset = SeededSubset(RandomImageDataset(), [1, 2], 42)
        a = subset[0][0]
        torch.rand(100)
        b = subset[0][0]
        self.assertTrue(torch.equal(a, b))
        subset.epoch = 1
        self.assertFalse(torch.equal(a, subset[0][0]))

    def test_initialization_resource_pairing_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = apply_overrides(self.cfg, {'federation.rounds': 1})
            a = Experiment(cfg, Path(directory)/'a', ROOT)
            bcfg = apply_overrides(cfg, {'distillation.mode': 'none'})
            b = Experiment(bcfg, Path(directory)/'b', ROOT)
            a.prepare()
            b.prepare()
            ma = json.loads((a.directory/'manifest.json').read_text())
            mb = json.loads((b.directory/'manifest.json').read_text())
            for key in ['initial_model_sha256', 'partition_sha256', 'initial_resources']:
                self.assertEqual(ma[key], mb[key])
            self.assertEqual(a.rounds.update_resources_and_select_clients(),
                             b.rounds.update_resources_and_select_clients())
            self.assertEqual([asdict(r.actual) for r in a.rounds.resources],
                             [asdict(r.actual) for r in b.rounds.resources])
            with self.assertRaises(FileExistsError):
                a.prepare()


if __name__ == '__main__':
    unittest.main(verbosity=2)
