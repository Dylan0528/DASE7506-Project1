"""Time candidate configurations with the *real* scorer on CPU (4 threads).

python bench_real.py configs/student_c.json configs/student_d.json
"""

import argparse
import importlib
import json
from pathlib import Path

import torch
from common import ROOT, load_data, setup
from evaluate import score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('configs', nargs='+', type=Path)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--modules', nargs='*', default=None)
    p.add_argument('--split', default='validation')
    args = p.parse_args()
    device, _ = setup('cpu', 'fp32', args.threads)
    data = load_data()
    reference = None
    for index, path in enumerate(args.configs):
        config = json.loads(path.read_text())
        module = 'model' if path.name == 'baseline.json' else (
            args.modules[index] if args.modules else 'student')
        torch.manual_seed(0)
        model = importlib.import_module(module).build_model(config)
        result = score(model, *data[args.split], device, 'fp32')
        ratio = 1. if reference is None else result['seconds'] / reference
        if reference is None:
            reference = result['seconds']
        print(f'{path.name:26s} params={sum(q.numel() for q in model.parameters())/1e6:6.2f}M  '
              f'{result["seconds"]:6.2f}s  ratio={ratio:5.2f}x', flush=True)


if __name__ == '__main__':
    main()
