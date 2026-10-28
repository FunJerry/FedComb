"""数据加载与等样本量 Dirichlet 划分；无放回、无重复、无客户端空集。"""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset


def load_cifar10_png(root, train_transform, test_transform):
    """兼容 RingSFL 的 cifar10/{train,test}/{类别}/*.png，不重新下载。

    ImageFolder 自带 targets，与后续 Dirichlet 划分接口一致；明确检查类别
    顺序和训练/测试映射，避免两种存储形式导致标签错位。
    """
    from torchvision.datasets import ImageFolder
    expected = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']
    train = ImageFolder(Path(root) / 'cifar10/train', transform=train_transform)
    test = ImageFolder(Path(root) / 'cifar10/test', transform=test_transform)
    if train.classes != expected or test.class_to_idx != train.class_to_idx:
        raise ValueError('本地 CIFAR-10 PNG 类别或训练/测试标签映射不一致')
    return train, test, np.asarray(train.targets)


def load_datasets(config, model, clients, seed, project_root):
    """真实数据默认不下载；synthetic 仅检验流程，准确率没有研究意义。"""
    if config.dataset == 'synthetic':
        generator = torch.Generator().manual_seed(seed)
        def make(n):
            images = torch.randn(n, 3, config.input_size, config.input_size, generator=generator)
            labels = torch.arange(n) % model.classes
            return TensorDataset(images, labels), labels.numpy()
        train, labels = make(clients * config.samples_per_client)
        test, _ = make(max(16, config.batch_size * 2))
        return train, test, labels
    # 仅真实数据入口依赖 torchvision，包导入不触发数据 IO。
    from torchvision import datasets, transforms
    root = Path(project_root) / 'data'
    stats = {'cifar10': ((.4914, .4822, .4465), (.2023, .1994, .2010)),
             'cifar100': ((.5071, .4867, .4408), (.2675, .2565, .2761)),
             'svhn': ((.4377, .4438, .4728), (.1980, .2010, .1970))}
    mean, std = stats[config.dataset]
    basic = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    augmentation = [] if config.dataset == 'svhn' else [
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    train_transform = transforms.Compose(augmentation + basic)
    test_transform = transforms.Compose(basic)
    if config.dataset == 'cifar10' and (root / 'cifar10').is_dir():
        return load_cifar10_png(root, train_transform, test_transform)
    if config.dataset == 'svhn':
        train = datasets.SVHN(root, split='train', transform=train_transform, download=config.download)
        test = datasets.SVHN(root, split='test', transform=test_transform, download=config.download)
        labels = np.asarray(train.labels)
    else:
        dataset_class = datasets.CIFAR10 if config.dataset == 'cifar10' else datasets.CIFAR100
        train = dataset_class(root, train=True, transform=train_transform, download=config.download)
        test = dataset_class(root, train=False, transform=test_transform, download=config.download)
        labels = np.asarray(train.targets)
    return train, test, labels


def equal_size_dirichlet(labels, clients, samples, alpha, seed):
    """先给每客户端抽类别先验，再从未耗尽类别池无放回取样。

    保留原作者“各设备样本数相同、类别比例不同”的思想。
    类别池耗尽时把对应概率置零，避免旧 while 重采样可能极慢的问题。
    """
    if clients * samples > len(labels) or clients <= 0 or samples <= 0 or alpha <= 0:
        raise ValueError('样本不足或 Dirichlet 划分参数不合法')
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    pools = [rng.permutation(np.flatnonzero(labels == label)).tolist() for label in classes]
    prior = rng.dirichlet(np.full(len(classes), alpha), size=clients)
    result = [[] for _ in range(clients)]
    # 交错分配，避免先处理的客户端独占稀缺类别。
    for _ in range(samples):
        for client in rng.permutation(clients):
            probabilities = prior[client] * np.array([bool(pool) for pool in pools])
            if probabilities.sum() == 0:
                probabilities = np.array([bool(pool) for pool in pools], dtype=float)
            probabilities /= probabilities.sum()
            category = rng.choice(len(classes), p=probabilities)
            result[client].append(pools[category].pop())
    return result


class MergeLastBatch:
    """最后若只剩 1 个样本，并入前一批；不丢样本，避免末端 BN 报错。"""
    def __init__(self, samples, batch_size, seed):
        self.samples, self.batch_size = samples, batch_size
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self):
        indices = torch.randperm(self.samples, generator=self.generator).tolist()
        batches = [indices[i:i + self.batch_size] for i in range(0, self.samples, self.batch_size)]
        if len(batches) > 1 and len(batches[-1]) == 1:
            batches[-2].extend(batches.pop())
        return iter(batches)

    def __len__(self):
        count = (self.samples + self.batch_size - 1) // self.batch_size
        return count - int(count > 1 and self.samples % self.batch_size == 1)


class SeededSubset(Subset):
    """数据增强按(种子,客户端,轮次,epoch,样本)确定，不受蒸馏随机数消耗影响。"""
    def __init__(self, dataset, indices, seed):
        super().__init__(dataset, indices)
        self.seed = seed
        self.round_index = self.epoch = 0

    def __getitem__(self, position):
        seed = (self.seed + self.round_index * 100003 + self.epoch * 1009
                + int(self.indices[position])) % (2**32)
        # 只设置 CPU generator；torch.manual_seed 会额外重置 CUDA generator。
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            return self.dataset[self.indices[position]]

    def __getitems__(self, positions):
        # 覆盖 Subset 批量捷径，否则新版 DataLoader 可能绕过上述逐样本播种。
        return [self[position] for position in positions]


def make_loaders(train, test, indices, config, seed):
    loaders = [DataLoader(SeededSubset(train, subset, seed + client), num_workers=0,
                          batch_sampler=MergeLastBatch(len(subset), config.batch_size,
                                                       seed + client))
               for client, subset in enumerate(indices)]
    return loaders, DataLoader(test, batch_size=config.batch_size, shuffle=False, num_workers=0)
