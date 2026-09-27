"""Collect every run into one table (used for the report).

python summarize.py
"""

import json
from pathlib import Path

ROWS = []
for path in sorted(Path('runs').glob('*/metrics.json')):
    metrics = json.loads(path.read_text())
    config = metrics['config']
    candidates = metrics.get('candidate_validation_bpb', {})
    curve = ', '.join(f"{v['step']//1000}k:{v['bpb']:.3f}" for v in metrics.get('validation_history', []))
    ROWS.append(dict(
        run=path.parent.name,
        tag=metrics.get('tag', ''),
        arch=f"w{config['width']}/d{config['depth']}/{config.get('norm', 'ln')[:3]}"
             f"/{config.get('pos', 'learned')[:4]}/{config.get('mlp', 'gelu')[:3]}",
        params=f"{metrics['parameters']/1e6:.2f}M",
        optimizer=metrics.get('optimizer', 'adamw'),
        steps=metrics.get('steps', 1200),
        targets=f"{metrics['train_tokens']/1e6:.1f}M",
        val_bpb=metrics['validation']['bpb'],
        val_ppl=metrics['validation']['token_ppl'],
        candidates=' '.join(f"{k}={v:.4f}" for k, v in candidates.items()),
        selected=metrics.get('selected', '-'),
        train_min=round(metrics['train_seconds'] / 60, 1),
        curve=curve,
    ))

print(f"{'run':22s} {'tag':18s} {'arch':24s} {'params':7s} {'opt':6s} {'steps':6s} "
      f"{'targets':8s} {'val_bpb':8s} {'ppl':7s} {'min':5s}")
for row in ROWS:
    print(f"{row['run']:22s} {row['tag']:18s} {row['arch']:24s} {row['params']:7s} "
          f"{row['optimizer']:6s} {row['steps']:<6d} {row['targets']:8s} "
          f"{row['val_bpb']:<8.4f} {row['val_ppl']:<7.2f} {row['train_min']:<5}")
print()
for row in ROWS:
    if row['curve']:
        print(f"{row['run']:22s} {row['curve']}")
