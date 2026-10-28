"""分组配置：实验、数据、模型、训练、策略各有明确归属。

只允许已声明字段和有效组合。消融使用 dotted overrides，例如
distillation.mode=none，不复制模型/训练代码，不通过魔法字符串传隐式参数。
"""
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
import copy
import json
import math
import re


@dataclass
class ExperimentConfig:
    name: str = 'paper_dynamic'
    seed: int = 42
    device: str = 'auto'
    threads: int = 2


@dataclass
class DataConfig:
    dataset: str = 'cifar10'
    samples_per_client: int = 500
    batch_size: int = 100
    dirichlet_alpha: float = 0.5
    input_size: int = 32
    download: bool = False


@dataclass
class ModelConfig:
    width: int = 64  # 64 为论文 ResNet-50；缩小仅作机制测试。
    classes: int = 10
    drop_path: float = 0.05
    classifier_dropout: float = 0.0  # 原作者抽出的分类器实际未用 dropout。


@dataclass
class FederationConfig:
    clients: int = 100
    participants: int = 10
    rounds: int = 1000
    checkpoint_every: int = 50


@dataclass
class TrainingConfig:
    local_epochs: int = 5
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 0.0001
    decay: float = 0.85
    decay_rounds: int = 100


@dataclass
class DistillationConfig:
    mode: str = 'adaptive'  # adaptive / head / tail / none；每客户端仍只选一个学生。
    weight_mode: str = 'adaptive'  # adaptive / fixed
    fixed_weight: float = 1.0
    learning_rate: float = 0.005
    temperature: float = 2.0


@dataclass
class AllocationConfig:
    policy: str = 'dynamic'  # dynamic / random / greedy / random_feasible / full / fixed
    min_shards: int = 3  # dynamic/random 等使用；原作者 Greedy 单独允许 1～2 片。
    fixed_start: int = 3
    fixed_end: int = 6
    # 消融项显式以 0 去掉，不关闭日志中的队列诊断。
    resource_weight: float = 1.0
    maturity_weight: float = 2.0
    coverage_weight: float = 6.0
    time_weight: float = 3.0
    max_comm_seconds: float | None = None
    max_compute_seconds: float | None = None


@dataclass
class AggregationConfig:
    method: str = 'mean'  # mean / gradient_mean / ipc；后两者使用同一首批梯度载荷。
    old_weight: float = 0.1
    ipc_learning_rate: float = 0.1


@dataclass
class ProgressConfig:
    aux_count_weight: float = 0.5
    eta: float = 0.5
    ema_old_weight: float = 0.1
    coverage_target: float = 0.5
    utilization_target: float = 0.95
    time_ema_old_weight: float = 0.9


@dataclass
class EnvironmentConfig:
    memory_min_mb: float = 20.0
    memory_max_mb: float = 120.0
    compute_min_gflops: float = 3.0
    compute_max_gflops: float = 12.5
    bandwidth_min_mbps: float = 1.0
    bandwidth_max_mbps: float = 50.0
    change_probability: float = 0.2  # 每轮被选中设备发生一次联合资源变化的概率；0 为静态消融。
    memory_fluctuation: float = 0.2
    compute_fluctuation: float = 0.3
    bandwidth_fluctuation: float = 0.25
    report_threshold: float = 0.1
    reporting: str = 'event'  # event / fresh；协议也是显式实验条件。


@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    federation: FederationConfig = field(default_factory=FederationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    allocation: AllocationConfig = field(default_factory=AllocationConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    progress: ProgressConfig = field(default_factory=ProgressConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)

    def validate(self):
        """先验证标量类型和区间，再验证实验策略组合。"""
        for group in vars(self).values():
            for item in fields(group):
                value = getattr(group, item.name)
                if item.type in (int, bool, str) and type(value) is not item.type:
                    raise ValueError(f'{type(group).__name__}.{item.name} 类型错误')
                if item.type is float and (type(value) not in (int, float) or not math.isfinite(value)):
                    raise ValueError(f'{type(group).__name__}.{item.name} 必须是有限数')
        if not re.fullmatch(r'[A-Za-z0-9_-]+', self.experiment.name):
            raise ValueError('实验名仅使用字母、数字、下划线、短横线')
        positive = [self.experiment.threads, self.data.samples_per_client, self.data.batch_size,
                    self.data.input_size, self.data.dirichlet_alpha, self.model.width, self.model.classes,
                    self.federation.clients, self.federation.participants, self.federation.rounds,
                    self.federation.checkpoint_every, self.training.local_epochs,
                    self.training.learning_rate, self.training.decay_rounds,
                    self.distillation.learning_rate, self.distillation.temperature,
                    self.aggregation.ipc_learning_rate]
        if any(value <= 0 for value in positive):
            raise ValueError('数量、学习率、温度、Dirichlet 参数必须为正')
        if self.experiment.seed < 0:
            raise ValueError('种子不能为负')
        if self.data.batch_size < 2 or self.data.samples_per_client < 2:
            raise ValueError('BatchNorm 训练至少需要两样本')
        if self.federation.participants > self.federation.clients:
            raise ValueError('参与客户端数大于总数')
        if self.data.dataset not in ('synthetic', 'cifar10', 'cifar100', 'svhn'):
            raise ValueError('未知数据集')
        classes = {'cifar10': 10, 'cifar100': 100, 'svhn': 10}
        if self.data.dataset in classes and (self.model.classes != classes[self.data.dataset]
                                             or self.data.input_size != 32):
            raise ValueError('真实数据类别数或输入尺寸错误')
        if self.experiment.device not in ('auto', 'cpu', 'cuda', 'mps'):
            raise ValueError('未知计算设备')
        if self.allocation.policy not in ('dynamic', 'random', 'greedy', 'random_feasible', 'full', 'fixed'):
            raise ValueError('未知分配策略')
        if self.distillation.mode not in ('adaptive', 'head', 'tail', 'none'):
            raise ValueError('未知蒸馏模式')
        if self.distillation.weight_mode not in ('adaptive', 'fixed'):
            raise ValueError('未知蒸馏权重模式')
        if self.aggregation.method not in ('mean', 'gradient_mean', 'ipc'):
            raise ValueError('未知聚合策略')
        if self.environment.reporting not in ('event', 'fresh'):
            raise ValueError('未知资源上报协议')
        if self.aggregation.method in ('ipc', 'gradient_mean') and (
            self.allocation.policy != 'random_feasible' or self.environment.reporting != 'fresh'):
            raise ValueError('首批梯度对照须使用 random_feasible 分配与 fresh 资源快照')
        if not 1 <= self.allocation.min_shards <= 16:
            raise ValueError('最小区间长度必须在 1～16')
        if not 0 <= self.allocation.fixed_start <= self.allocation.fixed_end < 16:
            raise ValueError('固定区间越界')
        if self.allocation.policy == 'fixed' and (
            self.allocation.fixed_end - self.allocation.fixed_start + 1 < self.allocation.min_shards):
            raise ValueError('固定区间短于最小分片数')
        unit_values = [self.progress.eta, self.progress.ema_old_weight, self.progress.aux_count_weight,
                       self.progress.coverage_target, self.progress.time_ema_old_weight,
                       self.aggregation.old_weight, self.environment.report_threshold,
                       self.environment.change_probability]
        if any(not 0 <= value <= 1 for value in unit_values):
            raise ValueError('比例参数必须在 [0,1]')
        if not 0 < self.progress.utilization_target <= 1 or not 0 < self.training.decay <= 1:
            raise ValueError('使用目标、学习率衰减必须在 (0,1]')
        if any(not 0 <= value < 1 for value in [self.environment.memory_fluctuation,
                                               self.environment.compute_fluctuation,
                                               self.environment.bandwidth_fluctuation,
                                               self.model.drop_path, self.model.classifier_dropout]):
            raise ValueError('波动、dropout 必须在 [0,1)')
        nonnegative = [self.allocation.resource_weight, self.allocation.maturity_weight,
                       self.allocation.coverage_weight, self.allocation.time_weight,
                       self.distillation.fixed_weight, self.training.weight_decay, self.training.momentum]
        if any(value < 0 for value in nonnegative):
            raise ValueError('权重、动量不能为负')
        for prefix, unit in [('memory', 'mb'), ('compute', 'gflops'), ('bandwidth', 'mbps')]:
            lo = getattr(self.environment, f'{prefix}_min_{unit}')
            hi = getattr(self.environment, f'{prefix}_max_{unit}')
            if not 0 < lo <= hi:
                raise ValueError('资源上下界错误')
        for value in [self.allocation.max_comm_seconds, self.allocation.max_compute_seconds]:
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
                raise ValueError('时延上限必须为有限正数或 null')


def from_dict(values):
    """只覆盖声明的配置字段；未知组/字段直接报错。"""
    config = Config()
    for name, contents in values.items():
        if name not in vars(config) or not isinstance(contents, dict):
            raise ValueError(f'未知配置组：{name}')
        group = getattr(config, name)
        for key, value in contents.items():
            if key not in vars(group):
                raise ValueError(f'未知字段：{name}.{key}')
            setattr(group, key, value)
    config.validate()
    return config


def load_config(path):
    return from_dict(json.loads(Path(path).read_text(encoding='utf-8')))


def apply_overrides(config, overrides):
    """每个变体独立复制配置，不能把消融开关泄漏到下一个实验。"""
    values = copy.deepcopy(asdict(config))
    for dotted_key, value in overrides.items():
        parts = dotted_key.split('.')
        if len(parts) != 2 or parts[0] not in values or parts[1] not in values[parts[0]]:
            raise ValueError(f'未知实验覆盖项：{dotted_key}')
        values[parts[0]][parts[1]] = value
    return from_dict(values)
