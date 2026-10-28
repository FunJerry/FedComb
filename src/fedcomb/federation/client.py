"""客户端只负责本地训练，不挑参与者、不选择区间、不聚合。

读 train → prepare → train_batch，即可看到旧主脚本的一个客户端训练流程。
策略变化全部通过 strategies/distillation.py 和分组配置传入。
"""
from dataclasses import dataclass
import copy
import time
import torch
from torch.nn import functional as F
from ..models.network import LocalModel
from ..core.types import Plan, Cost, Resources
from ..core.randomness import client_seed
from ..strategies.distillation import student_weight, compute_student_loss


@dataclass
class LocalSession:
    model: LocalModel
    student: object
    frozen_classifier: object
    task_optimizer: object
    student_optimizer: object
    weight: float


@dataclass
class ClientUpdate:
    client_id: int
    plan: Plan
    states: dict
    first_gradients: dict
    task_loss: float
    student_loss: float | None
    samples: int
    steps: int
    distillation_steps: int
    lambda_distill: float
    wall_seconds: float
    cost: Cost
    resources: Resources
    probabilities: list[float] | None
    cuda_peak_bytes: int = 0
    delta_loss: float = 0.0  # 本次本地训练首个/最后一个任务 batch 的损失绝对差。


def capture_gradients(model, plan):
    """IPC 取首批更新前的主任务参数梯度，不混入动量、衰减或多步差。"""
    return {f'block.{i}': {name: p.grad.detach().cpu().clone()
                          for name, p in model.component(f'block.{i}').named_parameters()
                          if p.grad is not None}
            for i in range(plan.start, plan.end + 1)}


def serialize_components(session):
    """回传 CPU 张量；轮末不需要一直保留所有客户端 GPU 模型。"""
    plan = session.model.plan
    modules = {key: session.model.component(key) for key in plan.task_keys()}
    if session.student is not None and session.weight > 0:
        modules[plan.student_key()] = session.student
    return {key: {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}
            for key, module in modules.items()}


class ClientTrainer:
    def __init__(self, bank, progress, config, device):
        self.bank, self.progress, self.config, self.device = bank, progress, config, device

    def optimizer(self, parameters, learning_rate):
        cfg = self.config.training
        return torch.optim.SGD(parameters, lr=learning_rate, momentum=cfg.momentum,
                               weight_decay=cfg.weight_decay)

    def prepare(self, plan, round_index):
        """任务模型与额外学生分开注册、分开优化，边界头尾属于任务模型。"""
        cfg = self.config
        decay = cfg.training.decay ** (round_index // cfg.training.decay_rounds)
        model = LocalModel(self.bank, plan).to(self.device).train()
        task_optimizer = self.optimizer(model.parameters(), cfg.training.learning_rate * decay)
        student = classifier = student_optimizer = None
        if plan.student_key() is not None:
            student = copy.deepcopy(self.bank.components()[plan.student_key()]).to(self.device).train()  #创建学生
            classifier = copy.deepcopy(model.component('classifier')).eval().requires_grad_(False)
            student_optimizer = self.optimizer(student.parameters(), cfg.distillation.learning_rate * decay)
        return LocalSession(model, student, classifier, task_optimizer, student_optimizer,
                            student_weight(self.progress, plan, cfg.distillation))

    def train_batch(self, session, images, labels, capture=False):
        """一个 batch：任务前向/反向 → 构建学生目标 → 任务 step → 学生 step。"""
        images, labels = images.to(self.device), labels.to(self.device)
        session.task_optimizer.zero_grad(set_to_none=True)  #清空任务梯度
        logits, stem, target = session.model(images)  #主任务前向传播，得到logits、 stem、target
        task_loss = F.cross_entropy(logits, labels) #计算任务交叉熵
        if not torch.isfinite(task_loss):
            raise FloatingPointError('任务损失非有限值')
        task_loss.backward()    #分类损失
        gradients = capture_gradients(session.model, session.model.plan) if capture else {}
        kd_loss = None
        if session.student is not None and session.weight > 0:
            session.student_optimizer.zero_grad(set_to_none=True)
            if session.model.plan.student_kind == 'tail':
                session.frozen_classifier.load_state_dict(session.model.component('classifier').state_dict())
            # 额外学生的随机数不推进任务随机流，避免无蒸馏对照暗中改变后续 task DropPath。
            cuda_devices = [self.device.index or 0] if self.device.type == 'cuda' else []
            with torch.random.fork_rng(devices=cuda_devices):
                kd_loss = compute_student_loss(session.student, session.frozen_classifier,
                    session.model.plan, stem, target, logits, self.config.distillation)
                if not torch.isfinite(kd_loss):
                    raise FloatingPointError('蒸馏损失非有限值')
        session.task_optimizer.step()   #任务优化器step
        if kd_loss is not None:
            (session.weight * kd_loss).backward()
            torch.nn.utils.clip_grad_norm_(session.student.parameters(), 1.0)
            session.student_optimizer.step()
        return float(task_loss.detach()), None if kd_loss is None else float(kd_loss.detach()), gradients

    def train(self, client_id, allocation, loader, round_index, cost, resources):
        started = time.perf_counter()
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        seed = client_seed(self.config.experiment.seed, round_index, client_id)
        torch.manual_seed(seed)
        loader.batch_sampler.generator.manual_seed(seed)
        session = self.prepare(allocation.plan, round_index)
        task_sum = kd_sum = 0.0
        seen = kd_seen = steps = kd_steps = 0
        gradients = {}
        first_task = last_task = None
        for epoch in range(self.config.training.local_epochs):
            loader.dataset.round_index, loader.dataset.epoch = round_index, epoch
            for images, labels in loader:
                task, kd, first = self.train_batch(session, images, labels,
                    capture=steps == 0 and self.config.aggregation.method in ('ipc', 'gradient_mean'))
                if first:
                    gradients = first
                task_sum += task * len(labels)
                if first_task is None:
                    first_task = task
                last_task = task
                seen += len(labels)
                steps += 1
                if kd is not None:
                    kd_sum += kd * len(labels)
                    kd_seen += len(labels)
                    kd_steps += 1
        states = serialize_components(session)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        return ClientUpdate(client_id, allocation.plan, states, gradients, task_sum / seen,
            kd_sum / kd_seen if kd_seen else None, len(loader.dataset), steps, kd_steps,
            session.weight, time.perf_counter() - started, cost, resources, allocation.probabilities,
            cuda_peak_bytes=torch.cuda.max_memory_allocated(self.device) if self.device.type == 'cuda' else 0,
            delta_loss=abs(first_task - last_task))
