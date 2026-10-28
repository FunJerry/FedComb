"""三方法公平配置与中断恢复测试；不在本机运行 1000 轮。"""
from dataclasses import asdict
from pathlib import Path
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch
import torch
from fedcomb.config import load_config, apply_overrides
from fedcomb.experiment import Experiment
from fedcomb.federation.round import RoundRunner


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cloud_training', ROOT / 'scripts/train_cifar10.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CloudTrainingTests(unittest.TestCase):
    def test_three_methods_differ_only_in_name_and_policy(self):
        comparable = []
        for method in module.METHODS:
            cfg = module.method_config(method, 'paper', 42)
            self.assertEqual(cfg.data.dataset, 'cifar10')
            self.assertEqual(cfg.federation.rounds, 1000)
            self.assertEqual(cfg.model.width, 64)
            self.assertEqual(cfg.aggregation.method, 'mean')
            value = asdict(cfg)
            del value['experiment']['name']
            del value['allocation']['policy']
            comparable.append(value)
        self.assertEqual(comparable[0], comparable[1])
        self.assertEqual(comparable[1], comparable[2])

    def test_resume_replays_after_checkpoint_and_matches_uninterrupted_run(self):
        cfg = apply_overrides(load_config(ROOT / 'configs/smoke.json'), {
            'federation.rounds': 3, 'federation.checkpoint_every': 1})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full, stopped = root / 'full', root / 'stopped'
            Experiment(cfg, full, ROOT).run()
            original = RoundRunner.run

            def interrupt(runner, index, loader):
                if index == 2:
                    raise KeyboardInterrupt('测试中断')
                return original(runner, index, loader)

            with patch.object(RoundRunner, 'run', interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    Experiment(cfg, stopped, ROOT).run()
            # 模拟更晚的完整日志已落盘、但检查点未更新；恢复必须备份并回滚。
            with (stopped / 'rounds.jsonl').open('a') as handle:
                handle.write((full / 'rounds.jsonl').read_text().splitlines()[-1] + '\n')
            Experiment(cfg, stopped, ROOT, resume=True).run()
            a = torch.load(full / 'checkpoint_0002.pt', weights_only=True)
            b = torch.load(stopped / 'checkpoint_0002.pt', weights_only=True)
            self.assertTrue(all(torch.equal(value, b['model'][name]) for name, value in a['model'].items()))
            self.assertEqual(a['progress'], b['progress'])
            self.assertEqual(a['round_state'], b['round_state'])
            records = [json.loads(line) for line in (stopped / 'rounds.jsonl').read_text().splitlines()]
            self.assertEqual([r['round'] for r in records], [0, 1, 2])
            self.assertTrue(list(stopped.glob('rounds.jsonl.before_resume_*')))


if __name__ == '__main__':
    unittest.main(verbosity=2)
