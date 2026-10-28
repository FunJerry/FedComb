#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -u scripts/train_cifar10.py --method fedcomb-greedy "$@"
