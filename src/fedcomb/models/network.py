"""ResNet-50 与辅助库：忠实保留原作者的跨块 Bottleneck 构造思想。

H_p 用首块的输入/中间通道，跨度覆盖 B0..Bp。
T_p 用 Bp 的输入/中间通道，跨度覆盖 Bp..B15。
每个辅助模块是独立深拷贝，避免原代码 T15 与 B15 共享对象。
"""
import copy
import torch
from torch import nn
from ..core.types import Plan


class Bottleneck(nn.Module):
    """1×1降维 → 3×3处理/下采样 → 1×1升维，加残差，再 ReLU。"""
    def __init__(self, in_channels, middle, out_channels, stride=1, drop_path=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, middle, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(middle)
        self.conv2 = nn.Conv2d(middle, middle, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(middle)
        self.conv3 = nn.Conv2d(middle, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = self.make_shortcut(in_channels, out_channels, stride)
        self.drop_path = drop_path

    @staticmethod
    def make_shortcut(in_channels, out_channels, stride):
        if in_channels == out_channels and stride == 1:
            return nn.Identity()
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                             nn.BatchNorm2d(out_channels))

    def forward(self, x):
        shortcut = self.downsample(x)
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        if self.training and self.drop_path > 0:
            # 与 stochastic depth 一致：按样本丢弃残差分支，不丢主干捷径。
            keep = 1 - self.drop_path
            mask = torch.empty((y.shape[0], 1, 1, 1), device=y.device).bernoulli_(keep)
            y = y * mask / keep
        return self.relu(y + shortcut)


def make_auxiliary(source: Bottleneck, stride: int, out_channels: int) -> Bottleneck:
    """参考旧 modify_block：只替换跨度改变所需的 conv2、conv3、shortcut。"""
    aux = copy.deepcopy(source)
    old_stride = source.conv2.stride[0]
    middle = source.conv2.in_channels
    old_output = source.conv3.out_channels
    in_channels = source.conv1.in_channels
    if stride != old_stride:
        aux.conv2 = nn.Conv2d(middle, middle, 3, stride=stride, padding=1, bias=False)
    if out_channels != old_output:
        aux.conv3 = nn.Conv2d(middle, out_channels, 1, bias=False)
        aux.bn3 = nn.BatchNorm2d(out_channels)
    if in_channels != out_channels or stride != old_stride:
        aux.downsample = Bottleneck.make_shortcut(in_channels, out_channels, stride)
    return aux


class ModelBank(nn.Module):
    """服务器：公共 E/D、16 主块、16 辅助头、16 辅助尾。"""
    def __init__(self, config):
        super().__init__()
        width = config.width
        self.stem = nn.Sequential(nn.Conv2d(3, width, 7, stride=2, padding=3, bias=False),
                                  nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(3, 2, 1))
        blocks, in_channels = [], width
        for stage, repeats in enumerate((3, 4, 6, 3)):
            middle = width * 2 ** stage
            for position in range(repeats):
                stride = 2 if stage > 0 and position == 0 else 1
                blocks.append(Bottleneck(in_channels, middle, 4 * middle, stride,
                                         config.drop_path * len(blocks) / 15))
                in_channels = 4 * middle
        self.blocks = nn.ModuleList(blocks)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                        nn.Dropout(config.classifier_dropout),
                                        nn.Linear(in_channels, config.classes))
        # 原 ResNet 的主块使用 Kaiming 初始化、末端 BN 零初始化。
        # 辅助替换卷积则保持新建 Conv2d 的默认初始化，接近原 modify_block。
        self.initialize_main()
        strides = [block.conv2.stride[0] for block in blocks]
        self.heads = nn.ModuleList()
        self.tails = nn.ModuleList()
        for p in range(len(blocks)):
            prefix_stride = suffix_stride = 1
            for stride in strides[:p + 1]:
                prefix_stride *= stride
            for stride in strides[p:]:
                suffix_stride *= stride
            # H0 在原作者结构中是首主块副本，不重置它的 downsample。
            self.heads.append(copy.deepcopy(blocks[0]) if p == 0 else
                              make_auxiliary(blocks[0], prefix_stride,
                                             blocks[p].conv3.out_channels))
            self.tails.append(make_auxiliary(blocks[p], suffix_stride,
                                             blocks[-1].conv3.out_channels))

    def initialize_main(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        for block in self.blocks:
            nn.init.zeros_(block.bn3.weight)

    def components(self) -> dict[str, nn.Module]:
        """人可读的稳定名称用于成本表、日志和聚合，不依赖偏移魔法数。"""
        result = {'stem': self.stem, 'classifier': self.classifier}
        for kind, modules in (('block', self.blocks), ('head', self.heads), ('tail', self.tails)):
            result.update({f'{kind}.{i}': module for i, module in enumerate(modules)})
        return result

    def forward(self, images):
        """完整模型测试：不用 heads/tails。"""
        x = self.stem(images)
        for block in self.blocks:
            x = block(x)
        return self.classifier(x)


class LocalModel(nn.Module):
    """客户端任务路径的独立副本；额外 student 不注册在此模型中。"""
    def __init__(self, bank: ModelBank, plan: Plan):
        super().__init__()
        self.plan = plan
        self.parts = nn.ModuleDict({key.replace('.', '_'): copy.deepcopy(bank.components()[key])
                                    for key in plan.task_keys(len(bank.blocks))})

    def component(self, key):
        return self.parts[key.replace('.', '_')]

    def forward(self, images):
        """同时返回 logits、E 后特征、选定 p 前/后特征，供头尾蒸馏使用。"""
        plan = self.plan
        stem_features = self.component('stem')(images)
        x = stem_features
        if plan.start > 0:
            x = self.component(f'head.{plan.start - 1}')(x)
        target = None
        for i in range(plan.start, plan.end + 1):
            if plan.student_kind == 'tail' and i == plan.student_index:
                target = x  # T_p 从 Bp 之前出发。
            x = self.component(f'block.{i}')(x)
            if plan.student_kind == 'head' and i == plan.student_index:
                target = x  # H_p 到达 Bp 之后。
        if plan.end < 15:
            x = self.component(f'tail.{plan.end + 1}')(x)
        return self.component('classifier')(x), stem_features, target
