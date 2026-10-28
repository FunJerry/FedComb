"""论文对比策略与动态评分分开：Random 抽一次再验预算，Greedy 固定随机起点。

random_feasible 保留先筛可行区间再均匀采样，专供已有 IPC/首批梯度对照。
所有策略共用成本模型与额外学生选择；Greedy 按原作者规则单独允许 1～2 片。
"""
from dataclasses import dataclass, asdict
import numpy as np
from ..core.types import Plan, Cost, Resources
from ..simulation.costs import estimate_cost
from ..tracking.progress import count_increment
from .distillation import select_student, student_weight


@dataclass
class Allocation:
    plan: Plan
    cost: Cost
    feasible_count: int | None  # Random 不枚举可行域，因此记录 None。
    score_terms: dict[str, float]
    probabilities: list[float] | None = None


def inclusion_probabilities(candidates):
    counts = np.zeros(16)
    for plan, _ in candidates:
        counts[plan.start:plan.end + 1] += 1
    return (counts / len(candidates)).tolist() if candidates else [0.0] * 16


class Allocator:
    """本轮预测计数与已成功学习计数分离。选择本身不产生更新。"""
    def __init__(self, progress, table, config):
        self.progress, self.table, self.config = progress, table, config
        self.normalizer = None
        if table:
            # 用公共的完整主模型、初始最低算力/带宽建立时间参考，不依赖首个设备是谁。
            # 这是归一化参考，不保证覆盖包含辅助学生的所有包，更不是实际测速。
            environment = config.environment
            reference = Resources(environment.memory_max_mb * 1e6,
                                  environment.compute_min_gflops * 1e9,
                                  environment.bandwidth_min_mbps * 1e6 / 8)
            self.normalizer = max(self.cost(Plan(0, 15), reference).round_seconds, 1e-8)
        self.begin_round()

    def begin_round(self):
        self.covered = np.zeros(16)
        self.counts = self.progress.counts.copy()
        self.accepted = 0
        self.observed_time = 0.0

    def cost(self, plan, resources):
        cfg = self.config
        return estimate_cost(plan, self.table, resources, cfg.data.samples_per_client,
                             cfg.training.local_epochs, ipc=cfg.aggregation.method in ('ipc', 'gradient_mean'))

    def is_feasible(self, cost, resource):
        cfg = self.config.allocation
        return (cost.weight_bytes <= resource.memory_bytes
                and (cfg.max_comm_seconds is None or cost.communication_seconds <= cfg.max_comm_seconds)
                and (cfg.max_compute_seconds is None or cost.compute_seconds <= cfg.max_compute_seconds))

    def candidates(self, resources):
        """供 dynamic/random_feasible/fixed/full 枚举可行域。"""
        cfg = self.config
        intervals = [(s, e) for s in range(16)
                     for e in range(s + cfg.allocation.min_shards - 1, 16)]
        if cfg.allocation.policy == 'full':
            intervals = [(0, 15)]
        elif cfg.allocation.policy == 'fixed':
            intervals = [(cfg.allocation.fixed_start, cfg.allocation.fixed_end)]
        candidates = []
        for start, end in intervals:
            plan = self.make_plan(start, end)
            cost = self.cost(plan, resources)
            if self.is_feasible(cost, resources):
                candidates.append((plan, cost))
        return candidates

    def make_plan(self, start, end):
        """每个区间先配套选定额外学生，再统一计算整个模型包的成本。"""
        kind, p = select_student(start, end, self.progress.maturity, self.config.distillation)
        return Plan(start, end, kind, p)

    def select_baseline(self, resources, rng):
        """论文消融流程：不按容量重抽 Random，也不为 Greedy 更换随机起点。"""
        minimum = self.config.allocation.min_shards
        if self.config.allocation.policy == 'random':
            # 沿用原作者 Random：长度均匀抽样，再在该长度的合法起点中均匀抽样。
            # 这不是“全部区间均匀”，更不是“可行区间均匀”。失败直接剔除。
            length = rng.randint(minimum, 16)
            start = rng.randint(0, 16 - length)
            plan = self.make_plan(start, start + length - 1)
            cost = self.cost(plan, resources)
            if not self.is_feasible(cost, resources):
                self.last_rejection = {'reason': 'random_interval_exceeds_reported_budget',
                    'plan': asdict(plan), 'weight_bytes': cost.weight_bytes,
                    'reported_memory_bytes': resources.memory_bytes}
                return None
            return Allocation(plan, cost, None, {})
        # 用户确认对齐原作者 Greedy：不使用 dynamic/random 的最少 3 片约束。
        start = rng.randint(0, 15)
        feasible = []
        for end in range(start, 16):
            plan = self.make_plan(start, end)
            cost = self.cost(plan, resources)
            if self.is_feasible(cost, resources):
                feasible.append((plan, cost))
        if not feasible:
            self.last_rejection = {'reason': 'no_feasible_interval_at_random_start', 'start': start}
            return None
        # 原作者比较完整参数包字节数，而不是片数；相同时保留先扫描到的较短区间。
        # 辅助尾和学生随端点变化，所以不能遇到第一次超预算就停止扫描。
        plan, cost = max(feasible, key=lambda item: item[1].weight_bytes)
        return Allocation(plan, cost, len(feasible), {})

    def learning_increment(self, plan, student_trained=None):
        """预测时检查学生权重，提交时使用实际蒸馏步数；不把零权重学生算作训练。"""
        if student_trained is None:
            student_trained = student_weight(self.progress, plan, self.config.distillation) > 0
        return count_increment(plan, self.config.progress, include_student=student_trained)

    def score_terms(self, client, resource, plan, cost):
        """修正型原算法：资源惩罚 + 成熟度队列项 + 覆盖缺口惩罚 + 时间惩罚。

        成熟度阈值 delta 对齐原实现的 mean(M)，但不复刻缺失覆盖计数/学生计数。
        覆盖项对全部 16 片计算，正号惩罚缺口，避免原负号反而奖励漏覆盖。
        """
        cfg, state = self.config, self.progress
        increment = self.learning_increment(plan)
        future = self.counts + increment
        gain = cfg.progress.eta * (future / max(float(future.max()), 1)
                                  - self.counts / max(float(self.counts.max()), 1))
        indices = np.flatnonzero(increment)
        delta = float(state.maturity_queue.mean())
        deficit = np.full(48, cfg.progress.aux_count_weight * delta)
        deficit[:16] = np.maximum(0, delta - state.maturity[:16])
        section = slice(plan.start, plan.end + 1)
        coverage = self.covered.copy()
        coverage[section] += 1
        coverage /= self.accepted + 1
        return {
            'resource': cfg.allocation.resource_weight * state.resource_queue[client]
                        * (cost.weight_bytes / resource.memory_bytes - cfg.progress.utilization_target),
            'maturity': cfg.allocation.maturity_weight * float((state.maturity_queue[indices]
                         * (deficit[indices] - gain[indices])).mean()),
            'coverage': cfg.allocation.coverage_weight * float((state.coverage_queue
                         * (cfg.progress.coverage_target - coverage)).sum()),
            'time': cfg.allocation.time_weight * cost.round_seconds / max(self.normalizer, 1e-8)}

    def select(self, client, resources, rng):
        self.last_rejection = None
        if self.config.allocation.policy in ('random', 'greedy'):
            return self.select_baseline(resources, rng)
        candidates = self.candidates(resources)
        if not candidates:
            return None
        maximum = max(cost.round_seconds for _, cost in candidates)
        self.observed_time = max(maximum, self.observed_time)
        if self.normalizer is None:
            self.normalizer = max(maximum, 1e-8)
        policy, terms, probabilities = self.config.allocation.policy, {}, None
        if policy == 'random_feasible':
            plan, cost = candidates[rng.randrange(len(candidates))]
            if self.config.aggregation.method in ('ipc', 'gradient_mean'):
                probabilities = inclusion_probabilities(candidates)
        elif policy in ('fixed', 'full'):
            plan, cost = candidates[0]
        else:
            scored = [(self.score_terms(client, resources, plan, cost), plan, cost)
                      for plan, cost in candidates]
            terms, plan, cost = min(scored, key=lambda item: sum(item[0].values()))
        return Allocation(plan, cost, len(candidates), terms, probabilities)

    def commit(self, plan, student_trained=None):
        self.covered[plan.start:plan.end + 1] += 1
        self.counts += self.learning_increment(plan, student_trained)
        self.accepted += 1

    def finish_round(self):
        if self.observed_time > 0:
            old = self.config.progress.time_ema_old_weight
            self.normalizer = old * self.normalizer + (1 - old) * self.observed_time
