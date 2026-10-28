"""云端运行前检查：真实 CIFAR-10、GPU 和依赖，不启动训练。"""
from pathlib import Path
import argparse
import json
import platform
import shutil
import torch
import torchvision
from fedcomb.config import Config
from fedcomb.data.datasets import load_datasets


ROOT = Path(__file__).resolve().parents[1]


def check(require_cuda=False):
    cfg = Config()
    train, test, labels = load_datasets(cfg.data, cfg.model, 100, 42, ROOT)
    if (len(train), len(test)) != (50000, 10000) or set(labels.tolist()) != set(range(10)):
        raise ValueError('不是完整的 CIFAR-10 训练/测试集')
    if train[0][0].shape != (3, 32, 32):
        raise ValueError('CIFAR-10 输入尺寸错误')
    cuda = torch.cuda.is_available()
    if require_cuda and not cuda:
        raise RuntimeError('要求 NVIDIA GPU，但 torch.cuda.is_available() 为 False')
    result = {'python': platform.python_version(), 'torch': str(torch.__version__),
              'torchvision': torchvision.__version__, 'cuda_available': cuda,
              'cuda_version': torch.version.cuda,
              'gpu': torch.cuda.get_device_name(0) if cuda else None,
              'gpu_memory_gib': torch.cuda.get_device_properties(0).total_memory / 2**30 if cuda else 0,
              'cifar10_train': len(train), 'cifar10_test': len(test),
              'data_format': type(train).__name__, 'disk_free_gib': shutil.disk_usage(ROOT).free / 2**30}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-cuda', action='store_true')
    check(parser.parse_args().require_cuda)
