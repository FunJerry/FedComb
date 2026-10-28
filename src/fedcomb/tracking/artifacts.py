"""每次实验有独立目录：配置、原始轮记录、简表、检查点、摘要。

拒绝覆盖非空目录。汇总只读取已记录结果，不把最佳精度冒充最后精度。
"""
from dataclasses import asdict
from pathlib import Path
import csv
import hashlib
import json
import platform
import time
import shutil
import torch
import numpy as np


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def model_digest(bank):
    digest = hashlib.sha256()
    for name, tensor in sorted(bank.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class ArtifactWriter:
    def __init__(self, directory, config, bank, indices, labels, table, initial_resources, resume=False):
        self.directory = Path(directory)
        if self.directory.exists() and any(self.directory.iterdir()):
            if not resume:
                raise FileExistsError(f'不覆盖已有实验：{self.directory}')
            if json.loads((self.directory / 'config.json').read_text()) != asdict(config):
                raise ValueError('恢复必须使用原实验完整配置')
            self.simulated = self.communication = self.wall = 0.0
            self.best, self.last, self.training_updates = 0.0, None, 0
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        write_json(self.directory / 'config.json', asdict(config))
        source = Path(__file__).resolve().parents[1]
        source_hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(source.rglob('*.py'))}
        write_json(self.directory / 'manifest.json', {
            'python': platform.python_version(), 'torch': str(torch.__version__), 'numpy': np.__version__,
            'synthetic': config.data.dataset == 'synthetic', 'initial_model_sha256': model_digest(bank),
            'partition_sha256': hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
            'source_sha256': source_hashes, 'initial_resources': initial_resources,
            'scope': 'single-process simulation; parameter memory; approximate FLOPs; not real network'})
        write_json(self.directory / 'partition.json', indices)
        write_json(self.directory / 'label_counts.json', [
            np.bincount(labels[index], minlength=config.model.classes).tolist() for index in indices])
        write_json(self.directory / 'component_costs.json', {k: asdict(v) for k, v in table.items()})
        self.simulated = self.communication = self.wall = 0.0
        self.best = 0.0
        self.last = None
        self.training_updates = 0

    def record_round(self, record):
        improved = self.last is None or record['evaluation']['accuracy'] > self.best
        self.simulated += record['simulated_seconds']
        self.communication += record['communication_bytes']
        self.wall += record['wall_seconds']
        self.best = max(self.best, record['evaluation']['accuracy'])
        self.last = record
        self.training_updates += len(record['clients'])
        record['cumulative_simulated_seconds'] = self.simulated
        record['cumulative_communication_bytes'] = self.communication
        with (self.directory / 'rounds.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        row = {'round': record['round'], 'accuracy': record['evaluation']['accuracy'],
               'task_loss': record['evaluation']['loss'], 'accepted': len(record['clients']),
               'skipped': len(record['skipped']), 'simulated_seconds': record['simulated_seconds'],
               'communication_bytes': record['communication_bytes'], 'wall_seconds': record['wall_seconds']}
        path = self.directory / 'metrics.csv'
        exists = path.exists()
        with path.open('a', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        return improved

    def rewind_to_checkpoint(self, round_index):
        """备份中断后日志，再重放到完整检查点；不重复计入多出的轮次。"""
        path = self.directory / 'rounds.jsonl'
        rows = []
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            if row['round'] <= round_index:
                rows.append(row)
        if [r['round'] for r in rows] != list(range(round_index + 1)):
            raise ValueError('检查点之前的轮记录不完整，不能恢复')
        backup = f'.before_resume_{time.time_ns()}'
        for filename in ['rounds.jsonl', 'metrics.csv', 'summary.json', 'best.pt']:
            file = self.directory / filename
            if file.exists():
                file.rename(file.with_name(file.name + backup))
        self.simulated = self.communication = self.wall = 0.0
        self.best, self.last, self.training_updates = 0.0, None, 0
        for row in rows:
            self.record_round(row)
        best = self.directory / 'best_at_checkpoint.pt'
        if best.exists():
            shutil.copy2(best, self.directory / 'best.pt')

    def save_checkpoint(self, checkpoint):
        path = self.directory / f'checkpoint_{checkpoint["round"]:04d}.pt'
        temporary = path.with_suffix('.pt.tmp')
        torch.save(checkpoint, temporary)
        temporary.replace(path)
        best = self.directory / 'best.pt'
        if best.exists():
            shutil.copy2(best, self.directory / 'best_at_checkpoint.pt')

    def save_best(self, checkpoint):
        path = self.directory / 'best.pt'
        temporary = path.with_suffix('.pt.tmp')
        torch.save(checkpoint, temporary)
        temporary.replace(path)

    def finish(self, config):
        summary = {'name': config.experiment.name, 'seed': config.experiment.seed,
                   'synthetic': config.data.dataset == 'synthetic',
                   'last_accuracy': self.last['evaluation']['accuracy'], 'best_accuracy': self.best,
                   'simulated_seconds': self.simulated, 'communication_bytes': self.communication,
                   'wall_seconds': self.wall, 'rounds': self.last['round'] + 1}
        summary['training_updates'] = self.training_updates
        write_json(self.directory / 'summary.json', summary)
        return summary
