#!/usr/bin/env bash
# 仅补齐工程依赖；复用镜像中的 CUDA torch/torchvision，不安装 macOS 锁文件。
set -euo pipefail
cd "$(dirname "$0")/.."
python -c 'import sys, torch, torchvision; assert sys.version_info >= (3,10), "需要 Python 3.10+"; assert torch.cuda.is_available(), "镜像中的 CUDA PyTorch 尚不可用"'
if [ ! -d .venv ]; then
  python -m venv --system-site-packages .venv
fi
.venv/bin/python -m pip install --disable-pip-version-check -e . --no-deps --no-build-isolation
# 只安装缺失的可选库，不升级已可用的 GPU 栈。允许用户切换到自己的可信镜像。
task_missing=$(.venv/bin/python -c 'import importlib.util; print(" ".join(requirement for name, requirement in [("scipy", "scipy>=1.11,<2"), ("matplotlib", "matplotlib>=3.7,<4")] if importlib.util.find_spec(name) is None))')
if [ -n "$task_missing" ]; then
  read -r -a task_dependencies <<< "$task_missing"
  .venv/bin/python -m pip install --disable-pip-version-check --index-url "${FEDCOMB_PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" "${task_dependencies[@]}"
fi
.venv/bin/python -c 'import torch, torchvision, numpy, scipy, matplotlib; print("训练与绘图依赖已就绪，CUDA torch 未替换")'
.venv/bin/python scripts/check_environment.py --require-cuda
