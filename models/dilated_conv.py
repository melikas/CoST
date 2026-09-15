"""Dilated temporal convolution backbone (TS2Vec, via CoST; MIT; see NOTICE).
"""
from __future__ import annotations

import torch.nn.functional as F
from torch import nn


def receptive_field(depth: int, kernel: int = 3) -> int:
    """Receptive field of `DilatedConvEncoder` at `depth`: depth + 1 blocks of two
    convolutions at dilation 2^i, so RF = 1 + 2*(k-1)*(2^(depth+1) - 1)."""
    return 1 + 2 * (kernel - 1) * (2 ** (depth + 1) - 1)


def depth_for_window(seq_len: int, kernel: int = 3) -> int:
    """Smallest depth whose receptive field covers the window: 7 on HRD, 4 on GLOBEM."""
    depth = 0
    while receptive_field(depth, kernel) < seq_len:
        depth += 1
    return depth


class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.receptive_field // 2,
                              dilation=dilation, groups=groups)
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.remove] if self.remove > 0 else out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = (nn.Conv1d(in_channels, out_channels, 1)
                          if in_channels != out_channels or final else None)

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = self.conv1(F.gelu(x))
        x = self.conv2(F.gelu(x))
        return x + residual


class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(channels[i - 1] if i > 0 else in_channels, channels[i],
                      kernel_size=kernel_size, dilation=2 ** i,
                      final=(i == len(channels) - 1))
            for i in range(len(channels))
        ])

    def forward(self, x):
        return self.net(x)


class TCNBackbone(DilatedConvEncoder):
    """Dilated TCN behind the backbone contract (B, T, H) -> (B, T, D).

    `depth` counts dilation levels: depth + 1 residual blocks at dilations 2^0 .. 2^depth.
    None picks the smallest depth whose receptive field covers the window. Subclassing
    DilatedConvEncoder keeps the state_dict keys of earlier checkpoints.
    """

    def __init__(self, width, output_dims, seq_len, depth=None, kernel_size=3):
        depth = depth_for_window(seq_len, kernel_size) if depth is None else int(depth)
        super().__init__(width, [width] * depth + [output_dims], kernel_size)
        self.depth = depth
        self.receptive_field = receptive_field(depth, kernel_size)

    def forward(self, x):                                   # B x T x H -> B x T x D
        return self.net(x.transpose(1, 2)).transpose(1, 2)
