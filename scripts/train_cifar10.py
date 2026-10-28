"""三方法共用入口。只改变 policy，不偷换数据、蒸馏和聚合。

random 先抽后验；greedy 对齐原作者的随机 0～15 起点、最大参数包（允许 1～2 片）。
dynamic 使用修正型原算法评分；旧结果不能用本版源码恢复。可行域随机仍名为 random_feasible。

默认 paper 是 1000 轮；smoke 仅云端机制验证；quick 保留论文规模、缩短轮数。
后台模式只是启动独立进程，不创建守护服务、不调整付费资源。
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import datetime
import json
import os
import subprocess
import sys
from fedcomb.config import load_config, apply_overrides


ROOT = Path(__file__).resolve().parents[1]
METHODS = {'fedcomb': 'dynamic', 'fedcomb-random': 'random', 'fedcomb-greedy': 'greedy'}


def method_config(method, profile, seed, device=None, rounds=None):
    config = load_config(ROOT / 'configs/cifar10_cloud.json')
    changes = {'experiment.name': method.replace('-', '_'), 'experiment.seed': seed,
               'allocation.policy': METHODS[method]}
    if profile == 'smoke':
        changes.update({'federation.clients': 4, 'federation.participants': 2,
                        'federation.rounds': 2, 'federation.checkpoint_every': 1,
                        'data.samples_per_client': 16, 'data.batch_size': 16,
                        'training.local_epochs': 1})
    elif profile == 'quick':
        changes['federation.rounds'] = 3
        changes['federation.checkpoint_every'] = 1
    if device is not None:
        changes['experiment.device'] = device
    if rounds is not None:
        changes['federation.rounds'] = rounds
    return apply_overrides(config, changes)


def write_status(path, payload):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 FedComb 三方法真实训练')
    parser.add_argument('--method', choices=['all', *METHODS], default='all')
    parser.add_argument('--profile', choices=['paper', 'quick', 'smoke'], default='paper')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'])
    parser.add_argument('--rounds', type=int, help='覆盖全局轮数；日志保存实际配置')
    parser.add_argument('--output', required=True, help='results 内的新任务目录')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--background', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='只显示三组配置，不加载数据/训练')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    methods = list(METHODS) if args.method == 'all' else [args.method]
    configs = {name: method_config(name, args.profile, args.seed, args.device, args.rounds) for name in methods}
    if args.dry_run:
        print(json.dumps({name: asdict(cfg) for name, cfg in configs.items()}, ensure_ascii=False, indent=2))
        return
    directory = (ROOT / 'results' / args.output).resolve()
    if not directory.is_relative_to(ROOT / 'results') or directory == ROOT / 'results':
        raise ValueError('输出必须为 results 内的独立子目录')
    if directory.exists() and any(directory.iterdir()) and not (args.resume or args.worker):
        raise FileExistsError('任务目录已有内容，请换 --output，或显式 --resume')
    # 恢复必须先确认旧任务已退出，避免两份训练同时写同一份结果。
    process_file = directory / 'process.json'
    if args.resume and not args.worker and process_file.exists():
        pid = json.loads(process_file.read_text())['pid']
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError(f'旧任务 PID={pid} 仍存在，请确认它已退出后恢复')
    if args.background:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / 'console.log').open('a', encoding='utf-8') as log:
            command = [sys.executable, '-u', str(Path(__file__).resolve()),
                       *[arg for arg in sys.argv[1:] if arg != '--background'], '--worker']
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        write_status(directory / 'process.json', {'pid': process.pid, 'command': command,
            'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat()})
        print(f'后台 PID={process.pid}\n日志：{directory / "console.log"}')
        return
    # 一个 GPU 上顺序执行，避免三组互抢显存/算力，使实测时间更易比较。
    from fedcomb.experiment import Experiment
    directory.mkdir(parents=True, exist_ok=True)
    state = {'status': 'running', 'profile': args.profile, 'methods': {}, 'pid': os.getpid()}
    write_status(directory / 'job_status.json', state)
    try:
        for name, config in configs.items():
            target = directory / name / f'seed-{args.seed}'
            summary = target / 'summary.json'
            if args.resume and summary.exists():
                saved = json.loads((target / 'config.json').read_text())
                if saved != asdict(config):
                    raise ValueError(f'{name} 的恢复配置不同')
                if json.loads(summary.read_text())['rounds'] == config.federation.rounds:
                    state['methods'][name] = 'completed'
                    continue
            state['methods'][name] = 'running'
            write_status(directory / 'job_status.json', state)
            resume = args.resume and (target / 'manifest.json').exists()
            Experiment(config, target, ROOT, resume=resume).run()
            state['methods'][name] = 'completed'
            write_status(directory / 'job_status.json', state)
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                     error=f'{type(error).__name__}: {error}')
        write_status(directory / 'job_status.json', state)
        raise
    state['status'] = 'completed'
    write_status(directory / 'job_status.json', state)
    print(f'任务完成。查看：python scripts/show_results.py {directory}')


if __name__ == '__main__':
    main()
