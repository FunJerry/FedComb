"""Eq.3–6、38：学习状态与虚拟队列。与候选评分分开，便于审查。"""
import numpy as np
from ..core.types import component_id


def key_id(key):
    """E/D 不参与成熟度；其它名称映射到 B/H/T 的编号。"""
    if key in ('stem', 'classifier'):
        return None
    kind, index = key.split('.')
    return component_id(kind, int(index))


def count_increment(plan, config, include_student=True):
    """B 增 1，参与的边界辅助增 aux_count_weight；学生只有实际训练才增加。"""
    increment = np.zeros(48)
    keys = plan.keys() if include_student else plan.task_keys()
    for key in keys:
        index = key_id(key)
        if index is not None:
            increment[index] += 1.0 if key.startswith('block.') else config.aux_count_weight
    return increment


def maturity_scores(counts, ema, config):
    """零分母单独处理，不产生 NaN；稳定分数仍遵循论文启发式。"""
    experience = counts / max(float(counts.max()), 1.0)
    scale = float(ema.max())
    stability = np.ones_like(ema) if scale == 0 else 1 - ema / scale
    return config.eta * experience + (1 - config.eta) * stability


class ProgressTracker:
    def __init__(self, clients):
        self.counts = np.zeros(48)
        self.ema = np.zeros(48)
        self.maturity = np.full(48, 1e-8)
        self.resource_queue = np.zeros(clients)
        self.maturity_queue = np.zeros(48)
        self.coverage_queue = np.zeros(16)

    def distillation_weight(self, plan):
        if plan.student_key() is None:
            return 0.0
        teacher_ids = [key_id(key) for key in plan.task_keys() if key_id(key) is not None]
        teacher = float(self.maturity[teacher_ids].mean())
        student = float(self.maturity[key_id(plan.student_key())])
        # 均匀初值的浮点 mean 可能产生 ~1e-16 假差距；避免首轮暗中 step。
        if np.isclose(teacher, student, rtol=1e-12, atol=0):
            return 0.0
        return abs(teacher - student) / max(teacher, student, 1e-8)

    def update(self, results, config, participants):
        """只接收成功训练的客户端，失败/无候选不产生训练次数。

        对齐原作者：损失变化来自一次本地训练内首/末 batch，不是跨轮平均损失差。
        所有实际训练的组件使用该客户端任务损失变化作代理，不声称是组件独立损失。
        覆盖率分母是本轮选中设备数，失败设备贡献 0，不能从分母中消失。
        """
        if not results:
            self.coverage_queue += config.coverage_target
            return  # 空轮只有覆盖缺口增加，不增加学习次数或假装产生损失变化。
        old = self.maturity.copy()
        changes = [[] for _ in range(48)]
        coverage = np.zeros(16)
        for result in results:
            student_trained = result.distillation_steps > 0
            self.counts += count_increment(result.plan, config, include_student=student_trained)
            coverage[result.plan.start:result.plan.end + 1] += 1
            self.resource_queue[result.client_id] = max(0.0,
                self.resource_queue[result.client_id] + result.cost.weight_bytes
                / result.resources.memory_bytes - config.utilization_target)
            keys = result.plan.keys() if student_trained else result.plan.task_keys()
            for key in keys:
                index = key_id(key)
                if index is None:
                    continue
                changes[index].append((result.delta_loss, result.samples))
        for index, entries in enumerate(changes):
            if entries:
                total_samples = sum(samples for _, samples in entries)
                # 样本权重已经归一化，不再额外除客户端数量；等样本时与原作者均值一致。
                delta = sum(change * samples / total_samples for change, samples in entries)
                beta = config.ema_old_weight
                self.ema[index] = beta * self.ema[index] + (1 - beta) * delta
        self.maturity = maturity_scores(self.counts, self.ema, config)
        deficit = np.maximum(0, old.mean() - old)
        self.maturity_queue = np.maximum(0, self.maturity_queue + deficit - (self.maturity - old))
        self.coverage_queue = np.maximum(0, self.coverage_queue + config.coverage_target
                                          - coverage / participants)

    def snapshot(self):
        """JSON 可序列化状态，用于日志、检查点和论文作图。"""
        return {name: getattr(self, name).tolist() for name in
                ('counts', 'ema', 'maturity', 'resource_queue', 'maturity_queue', 'coverage_queue')}
