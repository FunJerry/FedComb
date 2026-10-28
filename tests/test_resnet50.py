"""width=64 参数量与实际任务反向；不是合成精度评估。"""
import unittest
import torch
from fedcomb.config import ModelConfig
from fedcomb.models.network import ModelBank, LocalModel
from fedcomb.core.types import Plan


class ResNet50Tests(unittest.TestCase):
    def test_paper_parameter_counts_and_backward(self):
        torch.set_num_threads(2)
        bank = ModelBank(ModelConfig(drop_path=0)).eval()
        counts = [75008, 70400, 70400, 379392, 280064, 280064, 280064,
                  1512448, 1117184, 1117184, 1117184, 1117184, 1117184,
                  6039552, 4462592, 4462592]
        self.assertEqual([sum(p.numel() for p in block.parameters()) for block in bank.blocks], counts)
        total = sum(p.numel()*4 for module in [bank.stem, *bank.blocks, bank.classifier]
                    for p in module.parameters())
        self.assertEqual(total, 94_114_088)
        images = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            self.assertEqual(bank(images).shape, (2, 10))
        local = LocalModel(bank, Plan(3, 6)).train()
        logits, _, _ = local(images)
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
        loss.backward()
        torch.optim.SGD(local.parameters(), lr=.01).step()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None for p in local.parameters()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
