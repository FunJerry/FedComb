"""实验主线就在这一个文件。所有变体共用 run，不复制循环。"""
from dataclasses import asdict
import random
import time
from ..simulation.resources import create_resources, fluctuate, report_if_triggered
from ..strategies.allocation import Allocator
from .client import ClientTrainer
from ..core.types import Resources
from ..simulation.resources import ResourceState


class RoundRunner:
    def __init__(self, config, server, table, client_loaders):
        self.config, self.server, self.loaders = config, server, client_loaders
        seed = config.experiment.seed
        self.resource_rng = random.Random(seed + 10000)
        self.participant_rng = random.Random(seed + 20000)
        self.allocation_rng = random.Random(seed + 30000)
        self.resources = create_resources(config.environment, config.federation.clients, self.resource_rng)
        self.allocator = Allocator(server.progress, table, config)
        self.trainer = ClientTrainer(server.bank, server.progress, config, server.device)

    def snapshot(self):
        """轮边界状态：下一轮所需 RNG 和实际/已报告资源。"""
        return {'resource_rng': self.resource_rng.getstate(),
                'participant_rng': self.participant_rng.getstate(),
                'allocation_rng': self.allocation_rng.getstate(),
                'resources': [{'actual': asdict(s.actual), 'reported': asdict(s.reported)}
                              for s in self.resources]}

    def restore(self, state):
        self.resource_rng.setstate(state['resource_rng'])
        self.participant_rng.setstate(state['participant_rng'])
        self.allocation_rng.setstate(state['allocation_rng'])
        self.resources = [ResourceState(Resources(**s['actual']), Resources(**s['reported']))
                          for s in state['resources']]

    def update_resources_and_select_clients(self):
        """步骤 1：先抽参与者，再更新这些设备的资源，随后才开始分片分配。

        即使某个设备随后无可行区间，也不撤销已发生的资源变化，避免选择反馈偏置。
        参与者/资源随机流继续隔离；事件报告与实际内存复核机制保留。
        """
        selected = self.participant_rng.sample(range(self.config.federation.clients),
                                               self.config.federation.participants)
        for client in selected:
            fluctuate(self.resources[client], self.config.environment, self.resource_rng)
        return selected

    def train_selected_clients(self, selected, round_index):
        """步骤 2/3：逐客户端分配、接受预算、训练、回传，尚不更新全局权重。"""
        updates, records, skipped, reports = [], [], [], []
        for client in selected:
            state = self.resources[client]
            if self.config.environment.reporting == 'fresh':
                # IPC 的实际采样集合必须与概率计算集合一致，使用新鲜资源快照。
                state.reported = state.actual
            reported = state.reported
            allocation = self.allocator.select(client, reported, self.allocation_rng)
            if allocation is None:
                if report_if_triggered(state, self.config.environment.report_threshold):
                    reports.append({'client': client, 'reason': 'no_candidate'})
                # Random 抽到超预算区间并不等于整个可行域为空，明确保留剔除原因。
                skipped.append({'client': client, **(self.allocator.last_rejection or
                                                     {'reason': 'no_feasible_candidate'})})
                continue
            actual_cost = self.allocator.cost(allocation.plan, state.actual)
            if not self.allocator.is_feasible(actual_cost, state.actual):
                state.reported = state.actual
                reports.append({'client': client, 'reason': 'budget_rejected'})
                skipped.append({'client': client, 'reason': 'actual_budget_changed'})
                continue
            # 沿用原作者的 0 基分片编号；Aux/P 指额外学生，不是任务路径的边界头尾。
            plan = allocation.plan
            print(f'Client {client} Model Parts: {plan.start}-{plan.end}, '
                  f'Aux Type: {plan.student_kind}, P Index: {plan.student_index}', flush=True)
            update = self.trainer.train(client, allocation, self.loaders[client],
                                         round_index, actual_cost, state.actual)
            # Epoch 为 0 基全局轮次，不是本地 epoch；复用已有损失，不增加前向计算。
            print(f'Client {client} Epoch {round_index} Loss: {update.task_loss}', flush=True)
            self.allocator.commit(allocation.plan, student_trained=update.distillation_steps > 0)
            updates.append(update)
            if report_if_triggered(state, self.config.environment.report_threshold):
                reports.append({'client': client, 'reason': 'normal_upload'})
            records.append({'client': client, 'plan': asdict(update.plan), 'cost': asdict(actual_cost),
                'reported_resources': asdict(reported), 'actual_resources': asdict(state.actual),
                'score_terms': allocation.score_terms, 'feasible_count': allocation.feasible_count,
                'probabilities': update.probabilities, 'task_loss': update.task_loss,
                'student_loss': update.student_loss, 'lambda': update.lambda_distill,
                'task_steps': update.steps, 'student_steps': update.distillation_steps,
                'delta_loss': update.delta_loss,
                'wall_seconds': update.wall_seconds, 'cuda_peak_bytes': update.cuda_peak_bytes})
        return updates, records, skipped, reports

    def run(self, round_index, test_loader):
        """一轮固定顺序。读这 6 步即可定位每种实验变化的作用点。"""
        started = time.perf_counter()
        # 1. 选参与者，仅更新这些设备的模拟资源。
        selected = self.update_resources_and_select_clients()
        # 2. 初始化本轮预测计数。
        self.allocator.begin_round()
        # 3. 分配连续区间，各客户端独立训练。
        updates, clients, skipped, reports = self.train_selected_clients(selected, round_index)
        # 4. 服务端聚合，更新实际成熟度与虚拟队列。
        self.server.apply_updates(updates)
        self.allocator.finish_round()
        # 5. 测试完整主模型，不测试局部辅助组合。
        evaluation = self.server.evaluate(test_loader)
        # 6. 回传记录。写文件由 tracking/artifacts 负责。
        return {'round': round_index, 'selected': selected, 'clients': clients, 'skipped': skipped,
                'reports': reports, 'resources': [asdict(state.actual) for state in self.resources],
                'evaluation': evaluation, 'progress': self.server.progress.snapshot(),
                'simulated_seconds': max((u.cost.round_seconds for u in updates), default=0.0),
                'communication_bytes': sum(u.cost.communication_bytes for u in updates),
                'wall_seconds': time.perf_counter() - started}
