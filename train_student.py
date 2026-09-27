"""Student training recipe (see student.py for the architecture).

Extra capabilities over the default trainer in ``train.py``:

* AdamW or Muon (``optim.py``), configurable learning-rate schedule,
* dropout and weight-decay control,
* weight averaging: exponential moving average (EMA) and stochastic weight
  averaging (SWA) over the tail of training,
* periodic validation scoring and automatic selection of the best of
  {final, EMA, SWA} weights on validation.

Example
-------
python train_student.py --config configs/student_a.json --steps 6000 \
    --batch-size 32 --optimizer muon --lr 0.02 --run-dir runs/a
"""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score
from optim import Muon


def learning_rate_at(step, total, warmup, peak, min_ratio):
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (min_ratio + (1. - min_ratio) * .5 * (1. + math.cos(math.pi * progress)))


def build_optimizer(model, args):
    if args.optimizer == 'muon':
        return Muon(model, muon_lr=args.lr, muon_momentum=args.momentum,
                    ns_steps=args.ns_steps, adam_lr=args.embedding_lr,
                    weight_decay=args.weight_decay, adam_weight_decay=args.embedding_weight_decay)
    hidden, other = [], []
    for name, param in model.named_parameters():
        (other if 'token' in name or 'pos' in name or 'norm' in name else hidden).append(param)
    groups = [{'params': hidden, 'weight_decay': args.weight_decay, 'kind': 'hidden'},
              {'params': other, 'weight_decay': args.embedding_weight_decay, 'kind': 'other'}]
    return torch.optim.AdamW(groups, lr=args.lr, betas=(args.beta1, args.beta2), eps=1e-8)


class Averager:
    """Keeps an averaged copy of the weights (EMA or uniform SWA) on CPU."""

    def __init__(self, model, device):
        self.shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        self.sum = None
        self.count = 0
        self.decay = None
        self.device = device

    def _flat(self, model):
        # clone: the averaged copy must not share storage with the live parameters
        return {k: v.detach().to(torch.float32).clone() for k, v in model.state_dict().items()}

    def start(self, model, decay=None):
        self.decay = decay
        self.sum = self._flat(model) if decay is None else self._flat(model)
        self.count = 1

    def update(self, model):
        current = self._flat(model)
        if self.decay is None:
            for key in self.sum:
                self.sum[key].add_(current[key])
            self.count += 1
        else:
            for key in self.sum:
                self.sum[key].mul_(self.decay).add_(current[key], alpha=1. - self.decay)

    def weights(self):
        if self.decay is None:
            return {k: (v / self.count) for k, v in self.sum.items()}
        return dict(self.sum)


def save_checkpoint(path, model, config, args, tokens, weights=None):
    state = model.state_dict()
    if weights is not None:
        state = {k: weights[k].to(v.dtype) for k, v in state.items()}
    torch.save({'protocol': PROTOCOL, 'implementation': args.implementation, 'config': config,
                'model': {k: v.cpu() for k, v in state.items()}, 'seed': args.seed,
                'train_tokens': tokens}, path)


def main():
    total_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, default=ROOT / 'configs/student_a.json')
    p.add_argument('--run-dir', type=Path, default=ROOT / 'runs/student')
    p.add_argument('--device', default='cuda')
    p.add_argument('--precision', choices=['auto', 'fp32', 'bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--optimizer', choices=['adamw', 'muon'], default='adamw')
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--embedding-lr', type=float, default=3e-4)
    p.add_argument('--min-lr-ratio', type=float, default=.1)
    p.add_argument('--warmup', type=int, default=200)
    p.add_argument('--weight-decay', type=float, default=.1)
    p.add_argument('--embedding-weight-decay', type=float, default=0.)
    p.add_argument('--grad-clip', type=float, default=1.)
    p.add_argument('--beta1', type=float, default=.9)
    p.add_argument('--beta2', type=float, default=.95)
    p.add_argument('--momentum', type=float, default=.95)
    p.add_argument('--ns-steps', type=int, default=5)
    p.add_argument('--dropout', type=float, default=None, help='Override config dropout.')
    p.add_argument('--eval-every', type=int, default=0)
    p.add_argument('--ema-decay', type=float, default=0., help='0 disables EMA.')
    p.add_argument('--ema-start', type=float, default=.0, help='Fraction of steps before EMA starts.')
    p.add_argument('--swa-start', type=float, default=.0, help='Fraction of steps before SWA starts.')
    p.add_argument('--swa-every', type=int, default=100)
    p.add_argument('--save-every', type=int, default=0, help='Periodic checkpoint for safety.')
    p.add_argument('--save-all', action='store_true', help='Also save raw/EMA/SWA checkpoints.')
    p.add_argument('--teacher-checkpoint', type=Path, default=None,
                   help='Frozen teacher checkpoint for knowledge distillation.')
    p.add_argument('--teacher-config', type=Path, default=None,
                   help='Override the teacher configuration.')
    p.add_argument('--alpha', type=float, default=.5,
                   help='Weight of the distillation term (0 = hard labels only).')
    p.add_argument('--temperature', type=float, default=1.)
    p.add_argument('--tag', default='')
    args = p.parse_args()
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Run directory already contains results. Use a new --run-dir.')

    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data = load_data()
    config = json.loads(args.config.read_text())
    if args.dropout is not None:
        config['dropout'] = args.dropout
    model, implementation_sha = make_model(args.implementation, config, device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    optimizer = build_optimizer(model, args)
    teacher = None
    if args.teacher_checkpoint is not None:
        teacher_state = torch.load(args.teacher_checkpoint, map_location='cpu', weights_only=True)
        teacher_config = json.loads(args.teacher_config.read_text()) if args.teacher_config \
            else teacher_state['config']
        teacher, _ = make_model(teacher_state['implementation'], teacher_config, device)
        teacher.load_state_dict(teacher_state['model'])
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter() - prepared

    ema = Averager(model, device) if args.ema_decay > 0 else None
    swa = Averager(model, device) if args.swa_start > 0 else None
    ema_step = int(args.ema_start * args.steps)
    swa_step = int(args.swa_start * args.steps)

    history, validation_history, intermediate_seconds = [], [], 0.
    started = time.perf_counter()
    for step in range(args.steps):
        starts = torch.randint(len(tokens) - 257, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:, None] + torch.arange(257, device=device)]
        lr = learning_rate_at(step, args.steps, args.warmup, args.lr, args.min_lr_ratio)
        schedule_factor = lr / max(args.lr, 1e-12)
        for group in optimizer.param_groups:
            base = args.lr if group.get('kind', 'hidden') in ('muon', 'hidden') else args.embedding_lr
            group['lr'] = base * schedule_factor
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            logits = model(batch[:, :-1]).float()
            loss = F.cross_entropy(logits.flatten(0, 1), batch[:, 1:].flatten())
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(batch[:, :-1]).float()
                # summed KL per token, averaged over all targets in the batch
                soft = F.log_softmax(logits / args.temperature, dim=-1)
                target = F.softmax(teacher_logits / args.temperature, dim=-1)
                distillation = (target * (target.log().clamp_min(-1e9) - soft)).sum(-1).mean() \
                    * args.temperature ** 2
                loss = (1. - args.alpha) * loss + args.alpha * distillation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if ema is not None and step >= ema_step:
            if step == ema_step:
                ema.start(model, args.ema_decay)
            else:
                ema.update(model)
        if swa is not None and step >= swa_step:
            if step == swa_step:
                swa.start(model)
            elif (step - swa_step) % args.swa_every == 0:
                swa.update(model)
        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            save_checkpoint(args.run_dir / f'step{step + 1}.pt', model, config, args,
                            (step + 1) * args.batch_size * 256)

        if (step + 1) % 100 == 0 or step + 1 == args.steps:
            row = {'step': step + 1, 'loss': loss.item(),
                   'seconds': time.perf_counter() - started - intermediate_seconds, 'lr': lr}
            history.append(row)
            print(json.dumps(row), flush=True)
        if args.eval_every > 0 and (step + 1) % args.eval_every == 0:
            result = score(model, *data['validation'], device, 'fp32')
            result.pop('window_nll_nats')
            intermediate_seconds += result['seconds']
            validation_history.append({'step': step + 1, **result})
            print(json.dumps({'validation': validation_history[-1]}), flush=True)

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started - intermediate_seconds
    train_tokens = args.steps * args.batch_size * 256

    # ---- candidate weight sets: final, EMA, SWA -------------------------------
    candidates = {'final': None}
    if ema is not None:
        candidates['ema'] = ema.weights()
    if swa is not None:
        candidates['swa'] = swa.weights()
    evaluated = {}
    for name, weights in candidates.items():
        if weights is not None:
            model.load_state_dict({k: v.to(next(model.parameters()).dtype) for k, v in weights.items()})
        result = score(model, *data['validation'], device, 'fp32')
        result.pop('window_nll_nats')
        evaluated[name] = result['bpb']
        print(json.dumps({'candidate': name, 'validation_bpb': result['bpb']}), flush=True)
    best_name = min(evaluated, key=evaluated.get)
    best_weights = candidates[best_name]
    if best_weights is not None:
        model.load_state_dict({k: v.to(next(model.parameters()).dtype) for k, v in best_weights.items()})
    final_validation = score(model, *data['validation'], device, 'fp32')
    final_validation.pop('window_nll_nats')

    checkpoint = args.run_dir / 'checkpoint.pt'
    save_checkpoint(checkpoint, model, config, args, train_tokens, best_weights)
    if args.save_all:
        for name, weights in candidates.items():
            if weights is not None:
                save_checkpoint(args.run_dir / f'checkpoint_{name}.pt', model, config, args,
                                train_tokens, weights)

    result = {'protocol': PROTOCOL, 'implementation': args.implementation, 'config': config,
              'seed': args.seed, 'tag': args.tag,
              'parameters': sum(p.numel() for p in model.parameters()),
              'precision': precision, 'train_tokens': train_tokens, 'steps': args.steps,
              'batch_size': args.batch_size, 'optimizer': args.optimizer, 'lr': args.lr,
              'embedding_lr': args.embedding_lr, 'warmup': args.warmup,
              'min_lr_ratio': args.min_lr_ratio, 'weight_decay': args.weight_decay,
              'grad_clip': args.grad_clip, 'ema_decay': args.ema_decay,
              'ema_start': args.ema_start, 'swa_start': args.swa_start,
              'teacher': str(args.teacher_checkpoint) if args.teacher_checkpoint else None,
              'teacher_parameters': (sum(p.numel() for p in teacher.parameters()) if teacher else 0),
              'alpha': args.alpha, 'temperature': args.temperature,
              'preparation_seconds': preparation_seconds, 'train_seconds': train_seconds,
              'validation': final_validation, 'candidate_validation_bpb': evaluated,
              'selected': best_name, 'history': history,
              'validation_history': validation_history,
              'intermediate_validation_seconds': intermediate_seconds,
              'process_seconds': time.perf_counter() - total_started,
              'torch_version': str(torch.__version__), 'threads': args.threads,
              'checkpoint_sha256': sha(checkpoint), 'implementation_sha256': implementation_sha,
              **device_metrics(device)}
    (args.run_dir / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result | {'history': [], 'validation_history': []}, indent=2), flush=True)


if __name__ == '__main__':
    main()
