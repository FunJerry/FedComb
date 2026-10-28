"""把资源、参与者、批次和训练随机性分开，支持公平对照。"""
import random
import numpy as np
import torch


def seed_model(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def client_seed(seed, round_index, client_id):
    """客户端是否在其它轮参与，不影响本轮的随机种子。"""
    return (seed + 100003 * round_index + 1009 * client_id) % (2**32)


def select_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA 不可用')
    if name == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS 不可用')
    return torch.device(name)
