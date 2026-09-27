"""Build the submitted checkpoint bundle: average chosen weight sets, fix metadata, verify.

python finalize.py runs/e5_muon_long/checkpoint_ema.pt runs/e5_muon_long/checkpoint_swa.pt \
    --out runs/final --train-tokens 163840000
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from common import PROTOCOL, load_data, make_model, setup
from evaluate import score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoints', nargs='+')
    p.add_argument('--out', default='runs/final')
    p.add_argument('--train-tokens', type=int, default=0)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--device', default='cpu')
    p.add_argument('--split', default='validation')
    args = p.parse_args()

    states, config, implementation = [], None, None
    for path in args.checkpoints:
        state = torch.load(path, map_location='cpu', weights_only=True)
        states.append(state['model'])
        config, implementation = state['config'], state['implementation']
    weights = {k: sum(s[k].float() for s in states) / len(states) for k in states[0]}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / 'checkpoint.pt'
    torch.save({'protocol': PROTOCOL, 'implementation': implementation, 'config': config,
                'model': {k: v.cpu() for k, v in weights.items()}, 'seed': args.seed,
                'train_tokens': args.train_tokens}, checkpoint)

    device, _ = setup(args.device, 'fp32', 4)
    model, _ = make_model(implementation, config, device)
    model.load_state_dict({k: v.to(next(model.parameters()).dtype) for k, v in weights.items()})
    data = load_data()
    result = score(model, *data[args.split], device, 'fp32')
    result.pop('window_nll_nats')
    parameters = sum(p.numel() for p in model.parameters())
    summary = {'checkpoint': str(checkpoint), 'sources': args.checkpoints,
               'parameters': parameters,
               'checkpoint_bytes': checkpoint.stat().st_size,
               'split': args.split, **result}
    (out / f'{args.split}_cpu_fp32.json').write_text(json.dumps(summary, indent=2) + '\n')
    shutil.copy(checkpoint, out / 'checkpoint_copy.pt') if False else None
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
