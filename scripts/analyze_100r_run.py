"""Export descriptive statistics and a dependency-light plot of the 100-round run."""
import csv
import json
import statistics as st
from pathlib import Path
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parents[1]
source = root / 'results/paper_dynamic_100r_01'
out = source / 'analysis'
out.mkdir(exist_ok=True)
records = [json.loads(line) for line in (source / 'rounds.jsonl').read_text().splitlines() if line.strip()]
rows = []
coverage = [0] * 16
first_full = None
for r in records:
    clients = r['clients']
    for c in clients:
        for i in range(c['plan']['start'], c['plan']['end'] + 1):
            coverage[i] += 1
    if first_full is None and all(coverage):
        first_full = r['round'] + 1
    rows.append(dict(round=r['round'] + 1, accuracy=100 * r['evaluation']['accuracy'],
                     test_loss=r['evaluation']['loss'],
                     client_mean_task_loss=st.mean(c['task_loss'] for c in clients) if clients else None,
                     accepted=len(clients), skipped=len(r['skipped'])))
acc = [r['accuracy'] for r in rows]
loss = [r['test_loss'] for r in rows]
report = dict(rounds=len(rows), best_round=rows[acc.index(max(acc))]['round'],
              best_accuracy=max(acc), final_accuracy=acc[-1],
              first_all_blocks_covered_round=first_full, block_client_update_counts=coverage,
              skipped=sum(r['skipped'] for r in rows), windows=[])
for threshold in (15, 20, 30, 40, 50):
    report[f'first_round_above_{threshold}_percent'] = next((r['round'] for r in rows if r['accuracy'] > threshold), None)
    report[f'first_5_consecutive_rounds_above_{threshold}_percent'] = next((rows[i]['round'] for i in range(len(rows)-4) if min(acc[i:i+5]) > threshold), None)
for start in range(0, len(rows), 10):
    segment = rows[start:start+10]
    report['windows'].append(dict(rounds=f'{segment[0]["round"]}-{segment[-1]["round"]}',
        mean_accuracy=st.mean(r['accuracy'] for r in segment),
        min_accuracy=min(r['accuracy'] for r in segment), max_accuracy=max(r['accuracy'] for r in segment),
        mean_test_loss=st.mean(r['test_loss'] for r in segment)))
(out / 'statistics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
with (out / 'round_metrics.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
im = Image.new('RGB', (1400, 940), 'white')
d = ImageDraw.Draw(im)
def panel(box, series, title, ymax, baseline=None):
    x0,y0,x1,y1=box
    d.text((x0,y0-30), title, fill='black', font_size=21)
    for j in range(6):
        y=y1-(y1-y0)*j/5
        d.line((x0,y,x1,y), fill='#e2e8f0')
        d.text((x0-52,y-9), f'{ymax*j/5:.1f}', fill='#475569', font_size=15)
    def pt(i,v): return (x0+(x1-x0)*i/(len(series)-1), y1-(y1-y0)*v/ymax)
    if baseline is not None:
        y=pt(0,baseline)[1]; d.line((x0,y,x1,y), fill='#f59e0b', width=2)
    d.line([pt(i,v) for i,v in enumerate(series)], fill='#93b5e8', width=2)
    d.line([pt(i,st.mean(series[i-4:i+1])) for i in range(4,len(series))], fill='#1755a4', width=4)
    for i in (0,19,39,59,79,99):
        x=pt(i,0)[0]; d.text((x-10,y1+8), str(i+1), fill='#475569', font_size=15)
    d.text((x1-130,y1+29),'Global round',fill='#475569',font_size=16)
d.text((85,18),'Full global model evaluation | seed 42 | dynamic allocation', fill='black',font_size=25)
d.text((85,55),'Light: per round    Dark: trailing 5-round mean    Orange: 10% reference',fill='#475569',font_size=17)
panel((90,130,1330,440),acc,'Test accuracy (%)',60,10)
panel((90,555,1330,865),loss,'Test cross-entropy',max(loss)*1.05)
im.save(out / 'learning_curves.png')
print(json.dumps(report,indent=2))
