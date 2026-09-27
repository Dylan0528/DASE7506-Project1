"""Average several checkpoints ("model soup") and score the result on validation.

python soup.py runs/e5_muon_long/checkpoint_ema.pt runs/e5_muon_long/checkpoint_swa.pt \
    --out runs/e5_muon_long/checkpoint_soup.pt
"""

import argparse
import json
import torch
from common import PROTOCOL, load_data, make_model, setup
from evaluate import score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoints', nargs='+')
    p.add_argument('--out')
    p.add_argument('--device', default='cuda')
    p.add_argument('--split', default='validation')
    args = p.parse_args()
    device, _ = setup(args.device, 'fp32', 4)
    states, config, implementation = [], None, None
    for path in args.checkpoints:
        state = torch.load(path, map_location='cpu', weights_only=True)
        states.append(state['model'])
        config, implementation = state['config'], state['implementation']
    averaged = {k: sum(s[k].float() for s in states) / len(states) for k in states[0]}
    model, _ = make_model(implementation, config, device)
    model.load_state_dict({k: v.to(next(model.parameters()).dtype) for k, v in averaged.items()})
    data = load_data()
    result = score(model, *data[args.split], device, 'fp32')
    result.pop('window_nll_nats')
    print(json.dumps({'checkpoints': args.checkpoints, **result}, indent=2))
    if args.out:
        torch.save({'protocol': PROTOCOL, 'implementation': implementation, 'config': config,
                    'model': {k: v.cpu() for k, v in averaged.items()}, 'seed': 17,
                    'train_tokens': 0}, args.out)


if __name__ == '__main__':
    main()
