import math
from typing import List

import torch
from torch import nn
import torch.nn.functional as F
import torch.fft as fft
from einops import reduce, rearrange, repeat

import numpy as np

from .dilated_conv import DilatedConvEncoder
from .positional_encoding import (
    SUPPORTED_PES,
    Time2VecPE,
    PETransformerEncoderLayer,
    add_absolute_pe,
)


def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            res[i, t:t+l] = False
    return res


def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


class BandedFourierLayer(nn.Module):
    def __init__(self, in_channels, out_channels, band, num_bands, length=201):
        super().__init__()

        self.length = length
        self.total_freqs = (self.length // 2) + 1

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.band = band  # zero indexed
        self.num_bands = num_bands

        self.num_freqs = self.total_freqs // self.num_bands + (self.total_freqs % self.num_bands if self.band == self.num_bands - 1 else 0)

        self.start = self.band * (self.total_freqs // self.num_bands)
        self.end = self.start + self.num_freqs


        # case: from other frequencies
        self.weight = nn.Parameter(torch.empty((self.num_freqs, in_channels, out_channels), dtype=torch.cfloat))
        self.bias = nn.Parameter(torch.empty((self.num_freqs, out_channels), dtype=torch.cfloat))
        self.reset_parameters()

    def forward(self, input):
        # input - b t d
        b, t, _ = input.shape
        with torch.autocast(device_type='cuda', enabled=False):
            input_fft = fft.rfft(input.float(), dim=1)
            output_fft = torch.zeros(b, t // 2 + 1, self.out_channels, device=input.device, dtype=torch.cfloat)
            output_fft[:, self.start:self.end] = self._forward(input_fft)
            return fft.irfft(output_fft, n=input.size(1), dim=1)

    def _forward(self, input):
        output = torch.einsum('bti,tio->bto', input[:, self.start:self.end], self.weight)
        return output + self.bias

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)


class TransformerFeatureExtractor(nn.Module):
    """Transformer backbone alternative to the dilated-conv encoder (CoST Table 4).

    Keeps the same channels-first interface as DilatedConvEncoder, mapping
    (B, hidden_dims, T) -> (B, output_dims, T), so the rest of CoSTEncoder
    (TFD/SFD) is unchanged. Convolutions are position-aware on their own, so the
    Transformer needs an explicit positional encoding; ``pe`` selects which one
    (see ``models.positional_encoding.SUPPORTED_PES``). Absolute PEs are added to
    the embeddings; attention PEs act inside every self-attention layer.
    """
    def __init__(self, hidden_dims, output_dims, depth=10, n_heads=8,
                 dropout=0.1, max_len=2048, pe='sinusoidal'):
        super().__init__()
        pe = pe.lower()
        if pe not in SUPPORTED_PES:
            raise ValueError(f"pe must be one of {SUPPORTED_PES}, got: {pe}")
        n_heads = n_heads if output_dims % n_heads == 0 else 1
        self.pe = pe
        self.d_model = output_dims
        self.max_len = max_len
        self.input_proj = nn.Linear(hidden_dims, output_dims)
        self.in_drop = nn.Dropout(dropout)
        # parameters used only by specific absolute PEs
        self.lpe = nn.Embedding(max_len, output_dims) if pe == 'learnable' else None
        self.t2v = Time2VecPE(output_dims, max_len) if pe == 'time2vec' else None
        self.layers = nn.ModuleList([
            PETransformerEncoderLayer(pe, output_dims, n_heads, max_len, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(output_dims)

    def forward(self, x):                    # x: B x hidden_dims x T
        x = x.transpose(1, 2)                # B x T x hidden_dims
        x = self.input_proj(x)               # B x T x output_dims
        x = add_absolute_pe(x, self.pe, self.d_model,
                            learnable_pe=self.lpe, time2vec_pe=self.t2v)
        x = self.in_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x.transpose(1, 2)             # B x output_dims x T


class CoSTEncoder(nn.Module):
    def __init__(self, input_dims, output_dims,
                 kernels: List[int],
                 length: int,
                 hidden_dims=64, depth=10,
                 mask_mode='binomial', backbone='tcn', pe='sinusoidal'):
        super().__init__()

        component_dims = output_dims // 2

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.component_dims = component_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.backbone = backbone
        self.pe = pe.lower()
        self.input_fc = nn.Linear(input_dims, hidden_dims)

        # The TCN is position-aware through its convolutions, so it takes no PE by
        # default ('none'); Time2Vec can still be added on the hidden stream. The
        # Transformer always needs a PE and selects it via the `pe` argument.
        self.tcn_time2vec = None
        if backbone == 'transformer':
            self.feature_extractor = TransformerFeatureExtractor(
                hidden_dims, output_dims, depth=depth, pe=self.pe
            )
        elif backbone == 'tcn':
            if self.pe not in ('none', 'time2vec'):
                raise ValueError(
                    f"TCN backbone supports pe in ('none', 'time2vec'), got: {self.pe}"
                )
            if self.pe == 'time2vec':
                self.tcn_time2vec = Time2VecPE(hidden_dims, length)
            self.feature_extractor = DilatedConvEncoder(
                hidden_dims,
                [hidden_dims] * depth + [output_dims],
                kernel_size=3
            )
        else:
            raise ValueError(f"backbone must be 'tcn' or 'transformer', got: {backbone}")

        self.repr_dropout = nn.Dropout(p=0.1)

        self.kernels = kernels

        self.tfd = nn.ModuleList(
            [nn.Conv1d(output_dims, component_dims, k, padding=k-1) for k in kernels]
        )

        self.sfd = nn.ModuleList(
            [BandedFourierLayer(output_dims, component_dims, b, 1, length=length) for b in range(1)]
        )

    def forward(self, x, tcn_output=False, mask='all_true'):  # x: B x T x input_dims
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch

        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'

        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False

        mask &= nan_mask
        x[~mask] = 0

        # optional Time2Vec time-encoding for the TCN backbone (the Transformer
        # adds its own positional encoding inside the feature extractor)
        if self.tcn_time2vec is not None:
            x = x + self.tcn_time2vec(x.size(1), x.device, x.dtype).unsqueeze(0)

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.feature_extractor(x)  # B x Co x T

        if tcn_output:
            return x.transpose(1, 2)

        trend = []
        for idx, mod in enumerate(self.tfd):
            out = mod(x)  # b d t
            if self.kernels[idx] != 1:
                out = out[..., :-(self.kernels[idx] - 1)]
            trend.append(out.transpose(1, 2))  # b t d
        trend = reduce(
            rearrange(trend, 'list b t d -> list b t d'),
            'list b t d -> b t d', 'mean'
        )

        x = x.transpose(1, 2)  # B x T x Co

        season = []
        for mod in self.sfd:
            out = mod(x)  # b t d
            season.append(out)
        season = season[0]

        return trend, self.repr_dropout(season)
