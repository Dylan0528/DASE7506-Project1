"""Measure CPU scoring time of a configuration relative to the baseline GPT.

python bench_cpu.py configs/student_a.json
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from common import ROOT, load_data, make_model


def time_model(model, x, repeats=3):
    model.eval()
    with torch.no_grad():
        model.predict_log_probs(x)  # warm-up
        best = None
        for _ in range(repeats):
            started = time.perf_counter()
            model.predict_log_probs(x)
            best = time.perf_counter() - started if best is None else min(best, time.perf_counter() - started)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('configs', nargs='*', type=Path)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--windows', type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision('highest')
    tokens = torch.randint(0, 2048, (args.windows * 256,))
    x = tokens.view(args.windows, 256)
    reference = None
    for path in [ROOT / 'configs/baseline.json'] + list(args.configs):
        config = json.loads(path.read_text())
        torch.manual_seed(0)
        module = 'model' if path.name == 'baseline.json' else 'student'
        import importlib
        model = importlib.import_module(module).build_model(config)
        params = sum(p.numel() for p in model.parameters())
        seconds = time_model(model, x)
        ratio = 1. if reference is None else seconds / reference
        if reference is None:
            reference = seconds
        print(f'{path.name:24s} params={params/1e6:6.3f}M  '
              f'{seconds*1000/args.windows:7.2f} ms/window  ratio={ratio:5.2f}x', flush=True)


if __name__ == '__main__':
    main()
