"""Optimizers used by the student recipe: AdamW and Muon (orthogonalised momentum).

Muon follows the public reference implementation (Keller Jordan et al.,
"Muon: An optimizer for hidden layers in neural networks", 2024): the momentum
buffer is replaced by its orthogonal polar factor, computed with a few
Newton-Schulz iterations, so that every update direction has unit singular
values.  Empirically this makes training noticeably more sample efficient for
small transformers.  Embeddings, gains and the (tied) output head keep AdamW
updates, as recommended by the reference implementation.
"""

import math

import torch


@torch.no_grad()
def newton_schulz5(matrix, steps=5, eps=1e-7):
    """Approximate the orthogonal polar factor of a 2-D matrix."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = matrix.float()
    transpose = x.size(-2) > x.size(-1)
    if transpose:
        x = x.mT
    x = x / (x.norm() + eps)
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.mT if transpose else x


class Muon(torch.optim.Optimizer):
    """Muon for hidden weight matrices + AdamW for the remaining parameters."""

    def __init__(self, model, muon_lr=.02, muon_momentum=.95, ns_steps=5,
                 adam_lr=3e-4, adam_betas=(.9, .95), adam_eps=1e-8,
                 weight_decay=.0, adam_weight_decay=.0):
        hidden, other = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            (hidden if (param.ndim == 2 and 'token' not in name and 'head' not in name
                        and 'pos' not in name and 'norm' not in name) else other).append(param)
        groups = []
        if hidden:
            groups.append({'params': hidden, 'kind': 'muon', 'lr': muon_lr,
                           'momentum': muon_momentum, 'ns_steps': ns_steps,
                           'weight_decay': weight_decay})
        if other:
            groups.append({'params': other, 'kind': 'adam', 'lr': adam_lr,
                           'betas': adam_betas, 'eps': adam_eps,
                           'weight_decay': adam_weight_decay})
        super().__init__(groups, defaults={})
        self.hidden_count = len(hidden)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.dtype != torch.float32:
                    grad = grad.float()
                state = self.state[param]
                if 'buffer' not in state:
                    state['buffer'] = torch.zeros_like(param)
                if group['kind'] == 'muon':
                    state['buffer'].mul_(group['momentum']).add_(grad)
                    update = grad + group['momentum'] * state['buffer']  # Nesterov
                    update = newton_schulz5(update, group['ns_steps'])
                    update = update * max(1., update.size(-2) / update.size(-1)) ** .5
                    if group['weight_decay']:
                        param.mul_(1. - group['lr'] * group['weight_decay'])
                    param.add_(update, alpha=-group['lr'])
                else:
                    beta1, beta2 = group['betas']
                    state.setdefault('step', 0)
                    state['step'] += 1
                    state['buffer'].mul_(beta1).add_(grad, alpha=1. - beta1)
                    if 'square' not in state:
                        state['square'] = torch.zeros_like(param)
                    state['square'].mul_(beta2).addcmul_(grad, grad, value=1. - beta2)
                    bias1 = 1. - beta1 ** state['step']
                    bias2 = 1. - beta2 ** state['step']
                    step_size = group['lr'] / bias1
                    denom = state['square'].div(bias2).sqrt_().add_(group['eps'])
                    if group['weight_decay']:
                        param.mul_(1. - group['lr'] * group['weight_decay'])
                    param.addcdiv_(state['buffer'], denom, value=-step_size)
