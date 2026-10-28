"""实际构造后测量参数量和前向形状；不沿用 batch=100 的硬编码表。"""
from dataclasses import dataclass, asdict
import torch
from torch import nn
from ..core.types import Cost, Plan, Resources


@dataclass(frozen=True)
class ComponentCost:
    weight_bytes: int
    forward_flops: float  # 每样本；卷积/线性乘加记 2 FLOPs。


def profile_component(module, example):
    """用 hook 统计常见算子。近似值，不是运行时长或训练峰值内存。"""
    total = 0

    def count(layer, inputs, output):
        nonlocal total
        if isinstance(layer, nn.Conv2d):
            total += output.numel() * (2 * layer.in_channels // layer.groups
                                       * layer.kernel_size[0] * layer.kernel_size[1])
        elif isinstance(layer, nn.Linear):
            total += output.numel() * 2 * layer.in_features
        elif isinstance(layer, nn.BatchNorm2d):
            total += 2 * output.numel()
        elif isinstance(layer, nn.ReLU):
            total += output.numel()
        elif isinstance(layer, nn.MaxPool2d):
            kernel = layer.kernel_size
            total += output.numel() * (kernel * kernel if isinstance(kernel, int) else kernel[0] * kernel[1])
        elif isinstance(layer, nn.AdaptiveAvgPool2d):
            total += inputs[0].numel()

    hooks = [layer.register_forward_hook(count) for layer in module.modules()]
    try:
        with torch.no_grad():
            output = module(example)
    finally:
        for hook in hooks:
            hook.remove()
    size = sum(parameter.numel() * parameter.element_size() for parameter in module.parameters())
    return ComponentCost(size, float(total)), output


def profile_bank(bank, input_size):
    """返回组件成本表；所有输入为单样本，推理态避免改变 BN 统计。"""
    was_training = bank.training
    bank.eval()
    table = {}
    image = torch.zeros(1, 3, input_size, input_size)
    table['stem'], stem = profile_component(bank.stem, image)
    block_inputs, x = [], stem
    for i, block in enumerate(bank.blocks):
        block_inputs.append(x)
        table[f'block.{i}'], x = profile_component(block, x)
    table['classifier'], _ = profile_component(bank.classifier, x)
    for i in range(len(bank.blocks)):
        table[f'head.{i}'], _ = profile_component(bank.heads[i], stem)
        table[f'tail.{i}'], _ = profile_component(bank.tails[i], block_inputs[i])
    bank.train(was_training)
    return table


def estimate_cost(plan: Plan, table, resource: Resources, samples: int, local_epochs: int, ipc=False):
    """Eq.34–36 的可解释口径：含 E/D 和学生，全部单位转换在资源模块做。

    3×前向计算量近似前向+反向；采用实际本地样本量，而非旧批次数。
    不含教师重复评估、控制报文和完整梯度/优化器内存，见对应说明。
    """
    size = sum(table[key].weight_bytes for key in plan.keys())
    flops = sum(table[key].forward_flops for key in plan.keys())
    # IPC 同时保留辅助任务状态上传和额外梯度上传，不能漏计后一份载荷。
    gradient_bytes = sum(table[f'block.{i}'].weight_bytes for i in range(plan.start, plan.end + 1)) if ipc else 0
    return Cost(size, flops, size / resource.bandwidth_bytes_per_second,
                3 * samples * local_epochs * flops / resource.compute_flops_per_second,
                gradient_bytes, gradient_bytes / resource.bandwidth_bytes_per_second)


def cost_table_json(table):
    return {key: asdict(value) for key, value in table.items()}
