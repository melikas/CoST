"""Sequence backbones behind one contract, and the temporal encoding added before them.

Contract: a backbone maps the projected window (B, T, H) to (B, T, D) and keeps T. It is
built by name,

    build_backbone(name, width=H, output_dims=D, seq_len=T,
                   tcn_depth=..., n_layers=..., n_heads=..., bidirectional=...)

and a new one is added with `@register("name")` on a factory taking the same keywords.
Each backbone reads only its own options:

    tcn          tcn_depth                  dilation levels; None = smallest RF >= T
    transformer  n_layers, n_heads
    mamba        n_layers, bidirectional    official mamba-ssm CUDA kernels

The heads, losses and evaluation never branch on the backbone. Transformer and Mamba are
RQ4 building blocks, not parameter-matched to the TCN.
"""
import math

import torch
from torch import nn

from models.dilated_conv import TCNBackbone

BACKBONES = {}


def register(name):
    """Decorator adding a backbone factory under `name`."""
    def add(factory):
        if name in BACKBONES:
            raise ValueError(f"backbone {name!r} is already registered")
        BACKBONES[name] = factory
        return factory
    return add


def build_backbone(name, *, width, output_dims, seq_len, tcn_depth=None, n_layers=4, n_heads=4,
                   bidirectional=True):
    if name not in BACKBONES:
        raise ValueError(f"unknown backbone {name!r}; registered: {sorted(BACKBONES)}")
    return BACKBONES[name](width=width, output_dims=output_dims, seq_len=seq_len,
                           tcn_depth=tcn_depth, n_layers=n_layers, n_heads=n_heads,
                           bidirectional=bidirectional)


class TemporalEncoding(nn.Module):
    """Add no encoding, standard sinusoidal positions, or learned Time2Vec in days."""

    def __init__(self, kind, width, bins_per_day):
        super().__init__()
        if kind not in ("none", "sinusoidal", "time2vec"):
            raise ValueError(f"unsupported temporal encoding: {kind}")
        self.kind, self.width, self.bins_per_day = kind, width, bins_per_day
        if kind == "time2vec":
            self.frequency = nn.Parameter(torch.randn(width))
            self.phase = nn.Parameter(torch.zeros(width))

    def forward(self, x):
        if self.kind == "none":
            return x
        position = torch.arange(x.size(1), device=x.device, dtype=x.dtype)[:, None]
        if self.kind == "time2vec":
            angle = (position / self.bins_per_day) * self.frequency + self.phase
            encoding = torch.cat([angle[:, :1], angle[:, 1:].sin()], dim=-1)
        else:
            frequency = torch.exp(torch.arange(0, self.width, 2, device=x.device,
                                                dtype=x.dtype) * (-math.log(10000.) / self.width))
            encoding = torch.zeros(x.size(1), self.width, device=x.device, dtype=x.dtype)
            encoding[:, 0::2] = torch.sin(position * frequency)
            encoding[:, 1::2] = torch.cos(position * frequency[:self.width // 2])
        return x + encoding[None]


class TransformerBackbone(nn.Module):
    """Full-resolution self-attention; no patching or extra input information."""

    def __init__(self, width, output_dims, layers, heads=4):
        super().__init__()
        if width % heads:
            raise ValueError("Transformer width must be divisible by attention heads")
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(width, heads, dim_feedforward=4 * width,
                                       dropout=.1, activation="gelu", batch_first=True,
                                       norm_first=True) for _ in range(layers)])
        self.output = nn.Linear(width, output_dims)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


class MambaBackbone(nn.Module):
    """Residual official Mamba-1 blocks. With `bidirectional`, each block also scans the
    time-reversed sequence (the window is fully observed). Requires CUDA and mamba-ssm."""

    def __init__(self, width, output_dims, layers, bidirectional=True):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError as exc:
            raise ImportError("Mamba requires the official mamba-ssm CUDA package; no substitute is used") from exc

        def block():
            return Mamba(d_model=width, d_state=16, d_conv=4, expand=2)

        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.forward_layers = nn.ModuleList([block() for _ in range(layers)])
        self.backward_layers = (nn.ModuleList([block() for _ in range(layers)])
                                if bidirectional else None)
        self.output = nn.Linear(width, output_dims)

    def forward(self, x):
        if not x.is_cuda:
            raise ValueError("this official Mamba backend requires CUDA; use TCN for CPU smoke tests")
        for i, norm in enumerate(self.norms):
            h = norm(x)
            y = self.forward_layers[i](h)
            if self.backward_layers is not None:
                y = y + self.backward_layers[i](h.flip(1)).flip(1)
            x = x + y
        return self.output(x)


@register("tcn")
def _tcn(*, width, output_dims, seq_len, tcn_depth, n_layers, n_heads, bidirectional):
    return TCNBackbone(width, output_dims, seq_len, depth=tcn_depth)


@register("transformer")
def _transformer(*, width, output_dims, seq_len, tcn_depth, n_layers, n_heads, bidirectional):
    return TransformerBackbone(width, output_dims, n_layers, n_heads)


@register("mamba")
def _mamba(*, width, output_dims, seq_len, tcn_depth, n_layers, n_heads, bidirectional):
    return MambaBackbone(width, output_dims, n_layers, bidirectional)
