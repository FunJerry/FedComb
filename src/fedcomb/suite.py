"""实验表驱动：变体 × 参数网格 × 种子；公共流程只调用 Experiment.run。"""
from dataclasses import asdict
from pathlib import Path
import itertools
import json
import statistics
from .config import load_config, apply_overrides


def expand_suite(path):
    """返回确定顺序的(case_id, Config)，--list 可直接审查实际变化项。"""
    path = Path(path)
    spec = json.loads(path.read_text(encoding='utf-8'))
    if set(spec) - {'base', 'variants', 'grid', 'seeds'}:
        raise ValueError('实验表包含未知字段')
    base = load_config(path.parent / spec['base'])
    variants = spec.get('variants', [{'name': base.experiment.name, 'overrides': {}}])
    grid = spec.get('grid', {})
    seeds = spec.get('seeds', [base.experiment.seed])
    if not variants or not seeds or any(not values for values in grid.values()):
        raise ValueError('实验表不能含空列表')
    cases, identifiers = [], set()
    combinations = list(itertools.product(*grid.values()))
    for variant in variants:
        if set(variant) - {'name', 'overrides'}:
            raise ValueError('变体包含未知字段')
        for grid_index, values in enumerate(combinations):
            grid_changes = dict(zip(grid, values))
            variant_changes = variant.get('overrides', {})
            if {'experiment.name', 'experiment.seed'} & (set(grid_changes) | set(variant_changes)):
                raise ValueError('实验名由 variants.name 指定，种子由 seeds 指定，不能重复覆盖')
            if set(grid_changes) & set(variant_changes):
                raise ValueError('网格和变体不能同时覆盖同一个字段')
            for seed in seeds:
                name = variant['name'] + (f'_grid{grid_index:03d}' if grid else '')
                changes = {**variant_changes, **grid_changes, 'experiment.name': name, 'experiment.seed': seed}
                config = apply_overrides(base, changes)
                identifier = f'{name}/seed-{seed}'
                if identifier in identifiers:
                    raise ValueError('重复实验/种子，可能覆盖结果')
                identifiers.add(identifier)
                cases.append((identifier, config))
    return cases


def summarize(summaries, directory):
    """按同一变体跨种子汇总；样本标准差，单种子 std=null，不伪装误差棒。"""
    groups = {}
    for summary in summaries:
        groups.setdefault(summary['name'], []).append(summary)
    result = []
    for name, entries in groups.items():
        row = {'name': name, 'seeds': [entry['seed'] for entry in entries],
               'synthetic': entries[0]['synthetic']}
        for key in ['last_accuracy', 'best_accuracy', 'simulated_seconds', 'communication_bytes', 'wall_seconds']:
            values = [entry[key] for entry in entries]
            row[key] = {'mean': statistics.mean(values),
                        'std': statistics.stdev(values) if len(values) > 1 else None}
        result.append(row)
    from .tracking.artifacts import write_json
    write_json(Path(directory) / 'suite_summary.json', result)
    return result


def run_suite(path, directory, project_root):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f'不覆盖实验表结果：{directory}')
    cases = expand_suite(path)  # 开始任何训练前，验证全部组合。
    directory.mkdir(parents=True, exist_ok=True)
    from .tracking.artifacts import write_json
    from .experiment import Experiment
    write_json(directory / 'resolved_cases.json', [{'id': name, 'config': asdict(config)} for name, config in cases])
    summaries = [Experiment(config, directory / name, project_root).run() for name, config in cases]
    return summarize(summaries, directory)
