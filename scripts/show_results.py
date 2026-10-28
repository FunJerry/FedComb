"""读取已有实验结果并绘图，不导入训练模型、不修改原始日志。

指标只从 metrics.csv 重算；summary.json 用于交叉核对。
实测轮耗时、模拟耗时、估算通信量分别作图，不能互相替代。
"""
from pathlib import Path
import argparse
import csv
import json
import math


METHOD_NAMES = {'fedcomb': 'FedComb', 'fedcomb_random': 'FedComb-Random',
                'fedcomb_greedy': 'FedComb-Greedy'}
FIELDS = ('accuracy', 'task_loss', 'accepted', 'skipped', 'simulated_seconds',
          'communication_bytes', 'wall_seconds')
# 相同方法在每张图中保持同色；线型使黑白打印也能区分。
STYLES = {'fedcomb': ('#2463A6', '-'), 'fedcomb_random': ('#C47719', '--'),
          'fedcomb_greedy': ('#6D793A', '-.')}


def collect(directory):
    records = []
    for path in sorted(Path(directory).rglob('metrics.csv')):
        folder = path.parent
        config = json.loads((folder / 'config.json').read_text())
        with path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        for index, row in enumerate(rows):
            row['round'] = int(row['round'])
            if row['round'] != index:
                raise ValueError(f'{path}: 轮次缺失、重复或乱序')
            for key in FIELDS:
                row[key] = int(row[key]) if key in ('accepted', 'skipped') else float(row[key])
                if not math.isfinite(row[key]) or row[key] < 0:
                    raise ValueError(f'{path}: 第 {index + 1} 轮的 {key} 无效')
            if row['accuracy'] > 1:
                raise ValueError(f'{path}: accuracy 应为 0～1，而不是百分数')
            if row['accepted'] + row['skipped'] != config['federation']['participants']:
                raise ValueError(f'{path}: 成功/跳过数量与本轮参与者数不符')
        last, best = rows[-1]['accuracy'], max(row['accuracy'] for row in rows)
        summary_path = folder / 'summary.json'
        completed = False
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            if (summary['rounds'] != len(rows)
                or not math.isclose(summary['last_accuracy'], last)
                or not math.isclose(summary['best_accuracy'], best)):
                raise ValueError(f'{folder}: summary 与原始轮次指标不一致')
            for key in ('wall_seconds', 'simulated_seconds', 'communication_bytes'):
                if not math.isclose(summary[key], sum(row[key] for row in rows), rel_tol=1e-7):
                    raise ValueError(f'{folder}: summary 的 {key} 与逐轮求和不一致')
            completed = len(rows) == config['federation']['rounds']
        records.append({'path': folder, 'config': config, 'rows': rows,
                        'completed': completed, 'last': last, 'best': best,
                        'updates': sum(row['accepted'] for row in rows)})
    return sorted(records, key=lambda r: (list(METHOD_NAMES).index(r['config']['experiment']['name'])
                  if r['config']['experiment']['name'] in METHOD_NAMES else 99,
                  r['config']['experiment']['name'], r['config']['experiment']['seed']))


def trailing_mean(values, window):
    """只用当前及之前轮次；前 window-1 个点使用已有样本，不补零。"""
    if window < 1:
        raise ValueError('平滑窗口必须至少为 1')
    result, total = [], 0.0
    for i, value in enumerate(values):
        total += value
        if i >= window:
            total -= values[i - window]
        result.append(total / min(i + 1, window))
    return result


def cumulative(rows, key, scale):
    result, total = [], 0.0
    for row in rows:
        total += row[key]
        result.append(total / scale)
    return result


def chart_series(record):
    rows = record['rows']
    rounds = [row['round'] + 1 for row in rows]
    accuracy = [100 * row['accuracy'] for row in rows]
    return [
        (rounds, accuracy),
        (rounds, [row['task_loss'] for row in rows]),
        (cumulative(rows, 'wall_seconds', 3600), accuracy),
        (cumulative(rows, 'simulated_seconds', 3600), accuracy),
        (cumulative(rows, 'communication_bytes', 2**30), accuracy),
        (rounds, [row['accepted'] for row in rows]),
    ]


def summary_rows(records):
    """最高、最后、末 50 轮均值分别报告；单种子不生成虚假的误差条。"""
    return [{
        'method': METHOD_NAMES.get(r['config']['experiment']['name'], r['config']['experiment']['name']),
        'seed': r['config']['experiment']['seed'], 'completed': r['completed'],
        'rounds': len(r['rows']), 'target_rounds': r['config']['federation']['rounds'],
        'last_accuracy_pct': 100 * r['last'], 'best_accuracy_pct': 100 * r['best'],
        'best_round': max(r['rows'], key=lambda row: row['accuracy'])['round'] + 1,
        'last50_accuracy_pct': 100 * sum(row['accuracy'] for row in r['rows'][-50:]) / min(50, len(r['rows'])),
        'measured_round_hours': sum(row['wall_seconds'] for row in r['rows']) / 3600,
        'simulated_hours': sum(row['simulated_seconds'] for row in r['rows']) / 3600,
        'estimated_communication_gib': sum(row['communication_bytes'] for row in r['rows']) / 2**30,
        'accepted_updates': r['updates'],
        'acceptance_pct': 100 * r['updates'] / sum(row['accepted'] + row['skipped'] for row in r['rows']),
    } for r in records]


CHARTS = [
    ('accuracy_round', 'Test accuracy vs. round', 'Global round', 'Test accuracy (%)'),
    ('loss_round', 'Test loss vs. round', 'Global round', 'Test cross-entropy'),
    ('accuracy_measured_time', 'Accuracy vs. measured round time', 'Cumulative measured round time (h)', 'Test accuracy (%)'),
    ('accuracy_simulated_time', 'Accuracy vs. simulated time', 'Cumulative simulated time (h)', 'Test accuracy (%)'),
    ('accuracy_communication', 'Accuracy vs. estimated communication', 'Cumulative estimated communication (GiB)', 'Test accuracy (%)'),
    ('accepted_clients', 'Successful client updates per round', 'Global round', 'Accepted client updates'),
]


def plot(records, directory, window=20):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    seeds = sorted({r['config']['experiment']['seed'] for r in records})
    note = (f'Seeds: {seeds}. Pale lines: raw; dark lines: trailing {window}-round mean. '
            'No cross-seed uncertainty estimated.')

    def draw(axis, index):
        for record in records:
            name = record['config']['experiment']['name']
            color, style = STYLES.get(name, ('#555555', '-'))
            label = METHOD_NAMES.get(name, name) + f" / seed {record['config']['experiment']['seed']}"
            if not record['completed']:
                label += ' (incomplete)'
            x, y = chart_series(record)[index]
            if window > 1:
                axis.plot(x, y, color=color, alpha=.18, linewidth=.7)
            axis.plot(x, trailing_mean(y, window), color=color, linestyle=style,
                      linewidth=1.6, label=label)
        _, title, xlabel, ylabel = CHARTS[index]
        axis.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(0, None))
        axis.set_ylim(0, 100) if index in (0, 2, 3, 4) else axis.set_ylim(bottom=0)
        axis.grid(alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)
        axis.legend(fontsize=9, loc='best')

    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titleweight': 'medium', 'savefig.facecolor': 'white'}):
        overview, axes = plt.subplots(2, 3, figsize=(17, 9))
        overview.suptitle('Federated learning comparison — recorded experiment results', fontsize=16)
        for index, axis in enumerate(axes.flat):
            draw(axis, index)
        overview.text(.5, .015, note, ha='center', fontsize=9)
        overview.tight_layout(rect=(0, .045, 1, .95))
        for extension in ('png', 'pdf'):
            overview.savefig(directory / f'comparison.{extension}', dpi=180)
        plt.close(overview)
        for index, (name, _, _, _) in enumerate(CHARTS):
            figure, axis = plt.subplots(figsize=(8.4, 5.5))
            draw(axis, index)
            figure.text(.5, .015, note, ha='center', fontsize=8)
            figure.tight_layout(rect=(0, .06, 1, 1))
            for extension in ('png', 'pdf'):
                figure.savefig(directory / f'{name}.{extension}', dpi=180)
            plt.close(figure)
    summaries = summary_rows(records)
    with (directory / 'summary_comparison.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f'图表与汇总表：{directory.resolve()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', help='仅选择需要比较的实验根目录，避免混入机制验证数据')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--output-dir', help='图片输出目录，默认与实验根目录相同')
    parser.add_argument('--window', type=int, default=20, help='后向滑动平均窗口；1 表示只画原始值')
    args = parser.parse_args()
    if args.window < 1:
        parser.error('--window 必须至少为 1')
    records = collect(args.directory)
    if not records:
        parser.error('没有可读取的轮次指标，请检查目录')
    print('方法 / seed | 状态 | 完成轮次/目标 | 最后精度 | 最佳精度 | 末50轮均值')
    for row in summary_rows(records):
        state = '完成' if row['completed'] else '未完成'
        print(f"{row['method']} / {row['seed']} | {state} | {row['rounds']}/{row['target_rounds']} | "
              f"{row['last_accuracy_pct']:.2f}% | {row['best_accuracy_pct']:.2f}% | {row['last50_accuracy_pct']:.2f}%")
    if args.plot:
        plot(records, args.output_dir or args.directory, args.window)


if __name__ == '__main__':
    main()
