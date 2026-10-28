"""只读原作者 .log，重新生成参考图；不导入、运行或修改原训练工程。

完整日志和论文消融日志是不同批次，分别绘图，不拼接时延或通信量。
用法：python scripts/plot_original_results.py --source /path/to/fedshtd
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

from show_results import STYLES, trailing_mean


METHODS = [('fedcomb', 'FedSHTD', 'log.log'),
           ('fedcomb_random', 'FedSHTD-Random', 'log_random.log'),
           ('fedcomb_greedy', 'FedSHTD-Greedy', 'log_greedy.log')]
NUMBER = r'([\d.eE+-]+)'
ROUND = re.compile(r'^Epoch (\d+), Training Time: ' + NUMBER
                   + r',(?: Training Comm: ' + NUMBER + r',)? Test Loss: '
                   + NUMBER + r', Test Accuracy: ' + NUMBER + r'\s*$')
GROUPS = [('noniid_full', 'noniid/fedshtd/plot', 'Non-IID: full original logs'),
          ('noniid_ablation', 'noniid/logs_ablation', 'Non-IID: original ablation logs'),
          ('iid_ablation', 'iid/logs_ablation', 'IID-directory: original ablation logs')]


def parse_lines(lines):
    """只接收完整轮末记录；未写完的末轮不补齐。轮次在导出时转为 1 起始。"""
    rows, accepted, skipped = [], 0, 0
    for line in lines:
        accepted += bool(re.match(r'^Client \d+ Model Parts:', line))
        skipped += bool(re.match(r'^Client \d+ No model parts selected', line))
        match = ROUND.match(line)
        if not match:
            continue
        index, seconds, communication, loss, accuracy = match.groups()
        values = [float(seconds), float(loss), float(accuracy)]
        if (int(index) != len(rows) or accepted + skipped != 10
                or any(not math.isfinite(v) or v < 0 for v in values)
                or float(accuracy) > 100):
            raise ValueError(f'轮次、数值或客户端数量异常：{line.strip()}')
        comm = None if communication is None else float(communication)
        if comm is not None and (not math.isfinite(comm) or comm < 0):
            raise ValueError('通信量无效')
        if rows and (float(seconds) < rows[-1]['simulated_seconds']
                     or (comm is not None and rows[-1]['logged_communication_bytes'] is not None
                         and comm < rows[-1]['logged_communication_bytes'])):
            raise ValueError('累计时间或通信量倒退')
        rows.append({'round': int(index) + 1, 'accuracy_pct': float(accuracy),
                     'test_loss': float(loss), 'simulated_seconds': float(seconds),
                     'logged_communication_bytes': comm, 'new_client_trainings': accepted,
                     'skipped_clients': skipped})
        accepted = skipped = 0
    if not rows:
        raise ValueError('未找到完整训练轮次')
    return rows


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(records, output, title, author_view):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    communication = all(all(r['logged_communication_bytes'] is not None for r in rows)
                        for _, _, rows in records)
    note = ('Pale: raw values; dark: trailing 20-round mean. Original saved runs; no multi-seed intervals.\n'
            'Time is simulated. Communication, where available, follows original single-payload counting.')

    def draw(axis, metric):
        for key, name, rows in records:
            color, style = STYLES[key]
            y = [r['test_loss' if metric == 1 else 'accuracy_pct'] for r in rows]
            x = [r['round'] for r in rows]
            xlabel, ylabel = 'Global round', 'Test accuracy (%)'
            if metric == 1:
                ylabel = 'Test cross-entropy'
            elif metric == 2:
                x = [r['simulated_seconds'] / 3600 for r in rows]
                xlabel = 'Cumulative logged simulated time (h)'
            elif metric == 3 and communication:
                x = [r['logged_communication_bytes'] / 2**30 for r in rows]
                xlabel = 'Cumulative logged model payload (GiB)'
            elif metric == 3:
                y = [r['new_client_trainings'] for r in rows]
                ylabel = 'New client trainings per round'
            axis.plot(x, y, color=color, alpha=.18, linewidth=.6)
            axis.plot(x, trailing_mean(y, 20), color=color, linestyle=style,
                      linewidth=1.7, label=f'{name} ({len(rows)} rounds)')
        axis.set(xlabel=xlabel, ylabel=ylabel, xlim=(0, None), ylim=(0, None))
        if metric in (0, 2) or (metric == 3 and communication):
            axis.set_ylim(0, 100)
        elif metric == 3:
            axis.set_ylim(0, 10.5)
        axis.grid(alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)
        axis.legend(fontsize=9)

    def save(figure, name):
        for extension in ('png', 'pdf'):
            figure.savefig(output / f'{name}.{extension}', dpi=180, facecolor='white')
        plt.close(figure)

    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10}):
        figure, axes = plt.subplots(2, 2, figsize=(13, 8))
        figure.suptitle(title, fontsize=16)
        for metric, axis in enumerate(axes.flat):
            draw(axis, metric)
        figure.text(.5, .015, note, ha='center', fontsize=8)
        figure.tight_layout(rect=(0, .065, 1, .95))
        save(figure, 'comparison')
        figure, axis = plt.subplots(figsize=(9, 5.6))
        draw(axis, 0)
        axis.set_title(title)
        figure.text(.5, .015, note.split('\n')[0], ha='center', fontsize=8)
        figure.tight_layout(rect=(0, .045, 1, 1))
        save(figure, 'accuracy_round')
        if author_view:
            from scipy.signal import savgol_filter
            figure, axis = plt.subplots(figsize=(9, 5.6))
            for key, name, rows in records:
                values = [r['accuracy_pct'] for r in rows]
                color, style = STYLES[key]
                # 与原消融 notebook 一致：先平滑完整序列，再取前 610 个点。
                y = savgol_filter(values, 99, 3)[:610]
                axis.plot(range(len(y)), y, color=color, linestyle=style, linewidth=1.7, label=name)
            axis.set(title=title + ' — author-style view', xlabel='Original epoch index (0-based)',
                     ylabel='Smoothed test accuracy (%)', xlim=(0, 609), ylim=(0, 100))
            axis.grid(alpha=.2)
            axis.legend()
            axis.spines[['top', 'right']].set_visible(False)
            figure.text(.5, .015, 'Savitzky-Golay: window 99, order 3; first 610 points only.\n'
                        'Offline smoothing uses future points; not raw accuracy or a confidence interval.',
                        ha='center', fontsize=8)
            figure.tight_layout(rect=(0, .075, 1, 1))
            save(figure, 'author_style_first610')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('results/original_reference'))
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.source.resolve()):
        parser.error('不能向原作者参考工程内写入输出')
    args.output.mkdir(parents=True, exist_ok=True)
    summaries, sources = [], []
    for group, relative, title in GROUPS:
        output = args.output / group
        output.mkdir(exist_ok=True)
        records = []
        for key, name, filename in METHODS:
            path = args.source / relative / filename
            rows = parse_lines(path.read_text().splitlines())
            records.append((key, name, rows))
            write_csv(output / f'{name}_metrics.csv', rows)
            summaries.append({'group': group, 'method': name, 'rounds': len(rows),
                              'last_accuracy_pct': rows[-1]['accuracy_pct'],
                              'best_accuracy_pct': max(r['accuracy_pct'] for r in rows),
                              'last50_accuracy_pct': sum(r['accuracy_pct'] for r in rows[-50:]) / min(50, len(rows)),
                              'new_client_trainings': sum(r['new_client_trainings'] for r in rows)})
            sources.append({'group': group, 'method': name, 'path': str(path.resolve()),
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'rounds': len(rows)})
        render(records, output, title, author_view=group.endswith('ablation'))
    write_csv(args.output / 'summary_comparison.csv', summaries)
    (args.output / 'sources.json').write_text(json.dumps(sources, ensure_ascii=False, indent=2))
    print(f'原日志参考图已保存：{args.output.resolve()}')


if __name__ == '__main__':
    main()
