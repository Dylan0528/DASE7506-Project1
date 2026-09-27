"""Student model for MP1 (protocol 7506-mp1-wt2-v2).

A small decoder-only transformer with several structural changes over the
classroom baseline in ``model.py``:

* pre-norm **RMSNorm** instead of LayerNorm,
* **rotary position embeddings (RoPE)** instead of learned absolute positions,
* **SwiGLU** feed-forward blocks instead of GELU MLPs,
* a deeper / wider stack chosen to fit the evaluation compute budget,
* **dropout** on attention output and residual branches (the training set is small),
* depth-scaled initialisation of the residual branches.

Every change can be switched off through the configuration, so the same file
also implements the ablations (LayerNorm + learned positions + GELU MLP).

Interfaces (fixed by the protocol):
    forward(ids)              -> unnormalised logits [batch, time, 2048]
    predict_log_probs(ids)    -> normalised natural-log probabilities
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """Root-mean-square normalisation with a learnable gain."""

    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        xf = x.float()
        scale = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * scale * self.weight.float()).to(dtype)


def make_norm(kind, width):
    return RMSNorm(width) if kind == 'rmsnorm' else nn.LayerNorm(width)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, base=10000.0):
    """Rotate queries and keys by their absolute position (RoPE)."""
    length, dim = q.shape[-2], q.shape[-1]
    device = q.device
    inv_freq = base ** (-torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    angles = torch.outer(torch.arange(length, device=device, dtype=torch.float32), inv_freq)
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1).to(q.dtype)
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1).to(q.dtype)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class MLP(nn.Module):
    def __init__(self, width, kind='swiglu', ratio=8.0 / 3.0):
        super().__init__()
        self.kind = kind
        hidden = int(round(width * ratio))
        if kind == 'swiglu':
            self.gate = nn.Linear(width, hidden, bias=False)
            self.up = nn.Linear(width, hidden, bias=False)
            self.down = nn.Linear(hidden, width, bias=False)
            self.weights = (self.gate.weight, self.up.weight)
        else:  # plain GELU MLP; ratio multiplies the width (baseline uses 4)
            self.fc = nn.Linear(width, hidden, bias=False)
            self.down = nn.Linear(hidden, width, bias=False)
            self.weights = (self.fc.weight,)

    def forward(self, x):
        if self.kind == 'swiglu':
            return self.down(F.silu(self.gate(x)) * self.up(x))
        return self.down(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, width, heads, config, init_std=.02, residual_std=.02, dropout=0.):
        super().__init__()
        norm = config.get('norm', 'rmsnorm')
        self.heads = heads
        self.head_dim = width // heads
        self.rope = config.get('pos', 'rope') == 'rope'
        self.rope_base = float(config.get('rope_base', 10000.))
        self.norm1, self.norm2 = make_norm(norm, width), make_norm(norm, width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.mlp = MLP(width, config.get('mlp', 'swiglu'), config.get('mlp_ratio', 8. / 3.))
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.qkv.weight, std=init_std)
        for weight in self.mlp.weights:
            nn.init.normal_(weight, std=init_std)
        nn.init.normal_(self.proj.weight, std=residual_std)
        nn.init.normal_(self.mlp.down.weight, std=residual_std)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        if self.rope:
            q, k = apply_rope(q, k, self.rope_base)
        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout.p if self.training else 0.)
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        x = x + self.dropout(self.proj(attended))
        return x + self.dropout(self.mlp(self.norm2(x)))


class StudentLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        width, depth, heads = config['width'], config['depth'], config['heads']
        self.context = config['context']
        vocab = config['vocab']
        self.rope = config.get('pos', 'rope') == 'rope'
        self.tie = config.get('tie', True)
        self.dropout_p = float(config.get('dropout', 0.))
        self.token = nn.Embedding(vocab, width)
        if not self.rope:
            self.pos = nn.Embedding(self.context, width)
        residual_std = .02 / math.sqrt(2. * depth)
        self.blocks = nn.ModuleList([Block(width, heads, config, .02, residual_std, self.dropout_p)
                                     for _ in range(depth)])
        self.norm = make_norm(config.get('norm', 'rmsnorm'), width)
        self.head = nn.Linear(width, vocab, bias=False)
        if self.tie:
            self.head.weight = self.token.weight
        self.apply(self.initialize)

    @staticmethod
    def initialize(module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=.02)
        elif isinstance(module, nn.Linear):
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        x = self.token(ids)
        if not self.rope:
            x = x + self.pos(torch.arange(ids.shape[1], device=ids.device))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, ids):
        """Training interface: unnormalised next-token logits [batch, time, vocab]."""
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        """Evaluation interface: normalised log probabilities, stateless per window."""
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    """Build the student model. ``config`` comes from configs/*.json."""
    return StudentLM(config)
