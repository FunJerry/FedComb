"""服务器职责：维护全局组件/进度、轮末聚合、测试完整模型。"""
import torch
from torch.nn import functional as F
from ..models.network import ModelBank
from ..tracking.progress import ProgressTracker
from ..strategies.aggregation import aggregate


class FederatedServer:
    def __init__(self, config, device):
        self.config, self.device = config, device
        self.bank = ModelBank(config.model)
        self.progress = ProgressTracker(config.federation.clients)

    def apply_updates(self, updates):
        """所有客户端完成后才更新权重，因此同轮从同一模型快照出发。"""
        aggregate(self.bank, updates, self.config.aggregation)
        self.progress.update(updates, self.config.progress, self.config.federation.participants)

    @torch.no_grad()
    def evaluate(self, loader):
        """只测试 E→B0…B15→D，辅助库不参与测试也不移动到 GPU。"""
        modules = [self.bank.stem, *self.bank.blocks, self.bank.classifier]
        for module in modules:
            module.to(self.device).eval()
        total_loss = correct = samples = 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            logits = self.bank(images)
            total_loss += float(F.cross_entropy(logits, labels, reduction='sum'))
            correct += int((logits.argmax(1) == labels).sum())
            samples += len(labels)
        for module in modules:
            module.cpu()
        return {'loss': total_loss / samples, 'accuracy': correct / samples}

    def checkpoint(self, round_index, allocator):
        """学习快照；Experiment 额外补入轮次 RNG 与资源状态。"""
        return {'round': round_index, 'model': self.bank.state_dict(),
                'progress': self.progress.snapshot(),
                'time_normalizer': allocator.normalizer}
