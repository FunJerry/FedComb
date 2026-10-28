"""运行单次实验：准备一次，执行轮次，保存结果。与具体消融策略无关。"""
from dataclasses import asdict
from pathlib import Path
import torch
import numpy as np
import json
import hashlib
from .config import Config
from .core.randomness import seed_model, select_device
from .data.datasets import load_datasets, equal_size_dirichlet, make_loaders
from .federation.server import FederatedServer
from .federation.round import RoundRunner
from .simulation.costs import profile_bank
from .tracking.artifacts import ArtifactWriter


class Experiment:
    def __init__(self, config: Config, directory, project_root, resume=False):
        self.config, self.directory, self.project_root = config, Path(directory), Path(project_root)
        self.resume, self.start_round = resume, 0

    def prepare(self):
        """所有变体都重置初始化；禁止把前一个变体训练好的模型传给后一个。"""
        cfg = self.config
        cfg.validate()
        if self.directory.exists() and any(self.directory.iterdir()) and not self.resume:
            raise FileExistsError(f'结果目录已存在：{self.directory}')
        seed_model(cfg.experiment.seed)
        torch.set_num_threads(cfg.experiment.threads)
        device = select_device(cfg.experiment.device)
        train, test, labels = load_datasets(cfg.data, cfg.model, cfg.federation.clients,
                                            cfg.experiment.seed, self.project_root)
        indices = equal_size_dirichlet(labels, cfg.federation.clients, cfg.data.samples_per_client,
                                       cfg.data.dirichlet_alpha, cfg.experiment.seed)
        self.loaders, self.test_loader = make_loaders(train, test, indices, cfg.data, cfg.experiment.seed)
        self.server = FederatedServer(cfg, device)
        table = profile_bank(self.server.bank, cfg.data.input_size)
        self.rounds = RoundRunner(cfg, self.server, table, self.loaders)
        self.writer = ArtifactWriter(self.directory, cfg, self.server.bank, indices, labels, table,
                                     [asdict(state.actual) for state in self.rounds.resources], resume=self.resume)
        if self.resume:
            checkpoints = sorted(self.directory.glob('checkpoint_*.pt'))
            if not checkpoints:
                raise FileNotFoundError('没有完整检查点，请用新目录重跑')
            snapshot = torch.load(checkpoints[-1], map_location='cpu', weights_only=True)
            if 'round_state' not in snapshot:
                raise ValueError('旧版检查点缺少资源/RNG，不能精确恢复')
            manifest = json.loads((self.directory / 'manifest.json').read_text())
            from .tracking.artifacts import model_digest
            if (manifest['initial_model_sha256'] != model_digest(self.server.bank)
                or json.loads((self.directory / 'partition.json').read_text()) != indices):
                raise ValueError('初始化或数据划分改变，拒绝恢复')
            source = Path(__file__).resolve().parent
            hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(source.rglob('*.py'))}
            if manifest['source_sha256'] != hashes:
                raise ValueError('源码改变，不能混入同一实验恢复')
            self.server.bank.load_state_dict(snapshot['model'])
            for name, values in snapshot['progress'].items():
                setattr(self.server.progress, name, np.asarray(values, dtype=float))
            self.rounds.allocator.normalizer = snapshot['time_normalizer']
            self.rounds.restore(snapshot['round_state'])
            self.start_round = snapshot['round'] + 1
            self.writer.rewind_to_checkpoint(snapshot['round'])

    def checkpoint(self, round_index):
        snapshot = self.server.checkpoint(round_index, self.rounds.allocator)
        snapshot['round_state'] = self.rounds.snapshot()
        return snapshot

    def run(self):
        self.prepare()
        cfg = self.config
        for round_index in range(self.start_round, cfg.federation.rounds):
            record = self.rounds.run(round_index, self.test_loader)
            improved = self.writer.record_round(record)
            if improved:
                self.writer.save_best(self.checkpoint(round_index))
            if (round_index + 1) % cfg.federation.checkpoint_every == 0 or round_index == cfg.federation.rounds - 1:
                self.writer.save_checkpoint(self.checkpoint(round_index))
            # 原作者风格：累计模拟秒数、累计双向通信字节、测试损失、百分制精度。
            # 直接用 writer 累计值，恢复后仍连续；不把实测耗时冒充模拟训练时间。
            print(f'Epoch {round_index}, Training Time: {self.writer.simulated}, '
                  f'Training Comm: {int(self.writer.communication)}, '
                  f'Test Loss: {record["evaluation"]["loss"]}, '
                  f'Test Accuracy: {100 * record["evaluation"]["accuracy"]}', flush=True)
            print(f'{cfg.experiment.name} | seed={cfg.experiment.seed} | '
                  f'轮 {round_index + 1}/{cfg.federation.rounds} | '
                  f'成功 {len(record["clients"])} | 准确率 {record["evaluation"]["accuracy"]:.3f}', flush=True)
        return self.writer.finish(cfg)
