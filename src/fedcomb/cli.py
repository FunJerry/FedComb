"""两个明确入口：run 单实验，suite 批量实验。--list 只展开，不训练。"""
from dataclasses import asdict
from pathlib import Path
import argparse
import json
from .config import load_config
from .suite import expand_suite


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description='FedComb 实验与消融工程')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('run', 'suite'):
        command = sub.add_parser(name)
        command.add_argument('config', help='单实验或实验表 JSON')
        command.add_argument('--output', help='results 下的新目录名，不覆盖旧结果')
        if name == 'suite':
            command.add_argument('--list', action='store_true', help='只显示展开配置，不训练')
    args = parser.parse_args()
    if args.command == 'suite' and args.list:
        print(json.dumps([{'id': name, 'config': asdict(config)} for name, config in expand_suite(args.config)],
                         ensure_ascii=False, indent=2))
        return
    output_name = args.output or Path(args.config).stem
    directory = (PROJECT_ROOT / 'results' / output_name).resolve()
    if not directory.is_relative_to(PROJECT_ROOT / 'results'):
        raise ValueError('输出目录只能位于本工程 results 内')
    if args.command == 'run':
        from .experiment import Experiment
        Experiment(load_config(args.config), directory, PROJECT_ROOT).run()
    else:
        from .suite import run_suite
        run_suite(args.config, directory, PROJECT_ROOT)


if __name__ == '__main__':
    main()
