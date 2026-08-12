import math
from typing import List

import torch
from torch import nn
import torch.nn.functional as F
import torch.fft as fft
from einops import reduce, rearrange

import numpy as np

from .dilated_conv import DilatedConvEncoder
from .positional_encoding import (
    CALENDAR_PES,
    SUPPORTED_PES,
    CircularCalendarPE,
    FactorizedCalendarPE,
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


# Training-time timestep-masking augmentation (inherited from TS2Vec). 'none' applies no
# mask and is the DEFAULT: upstream salesforce/CoST hard-defaulted the encoder's `mask`
# argument to 'all_true' and never passed one from the training loop, so its `mask_mode`
# was unreachable and no CoST result was ever produced with masking on. 'none' reproduces
# that exactly; 'binomial'/'continuous' are opt-in (--mask-mode) rather than silently
# implied. See the note on `mask_prob` in CoSTEncoder.__init__ before enabling either.
SUPPORTED_MASK_MODES = ('none', 'binomial', 'continuous')


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
    the embeddings; attention PEs act inside every self-attention layer. With
    ``pe='time2vec'`` the attention is vanilla and NO PE is added here -- Time2Vec
    is fed as an input feature upstream in CoSTEncoder (Kazemi et al. 2019).
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
        self.layers = nn.ModuleList([
            PETransformerEncoderLayer(pe, output_dims, n_heads, max_len, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(output_dims)

    def forward(self, x, cal_pe=None):       # x: B x hidden_dims x T
        x = x.transpose(1, 2)                # B x T x hidden_dims
        x = self.input_proj(x)               # B x T x output_dims
        x = add_absolute_pe(x, self.pe, self.d_model, learnable_pe=self.lpe)
        if cal_pe is not None:               # calendar PE: same site/width as the absolute PEs
            x = x + cal_pe
        x = self.in_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x.transpose(1, 2)             # B x output_dims x T


class LearnableFourierMultiDim(nn.Module):
    """Multi-dimensional Learnable Fourier positional encoding
    (Li et al., "Learnable Fourier Features for Multi-Dimensional Spatial Positional
    Encoding", NeurIPS 2021 -- Algorithm 1) for a metric M-dimensional position.

    Given positions ``X`` of shape ``(T, M)``, a SINGLE shared learnable projection
    ``Wr in R^{F x M}`` (initialised ``N(0, gamma^-2)``, Eq. 4) produces the Fourier
    features ``r = (1/sqrt(F)) [cos(X Wr^T) || sin(X Wr^T)]`` of shape ``(T, 2F)`` (Eq. 2),
    which a small MLP maps to ``d_model`` (Eq. 6). Because all M dimensions share ONE Wr
    and are encoded JOINTLY (a single positional group, G=1), the dot product of two
    encodings depends on ``X_i - X_j`` across every dimension together -- the paper's
    L2-distance / shift-invariant inductive bias -- rather than encoding each dimension
    separately and adding, which the paper shows is weaker (Sec. 5).

    For the ViT-2D backbone this encodes each timestep's metric TIME position as
    ``[within-day phase, day-of-week]`` (M=2), a shift-invariant multi-scale (daily +
    weekly) circadian code. ``M=1`` recovers the plain 1-D learnable-Fourier time encoding.
    The non-metric CHANNEL axis is left to a discrete embedding (the L2 prior only holds
    on a metric axis).
    """

    def __init__(self, d_model, n_dims=2, n_freqs=None, gamma=1.0, mlp_hidden=None):
        super().__init__()
        n_freqs = n_freqs if n_freqs is not None else max(d_model // 2, 1)
        mlp_hidden = mlp_hidden if mlp_hidden is not None else d_model
        # W_r in R^{F x M}: N(0, gamma^-2), i.e. std = 1/gamma (Li et al. 2021, Eq. 4)
        self.Wr = nn.Parameter(torch.randn(n_freqs, n_dims) / max(gamma, 1e-6))
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_freqs, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, d_model)
        )
        self.scale = 1.0 / math.sqrt(n_freqs)

    def forward(self, pos):                              # pos: (T, M) float positions
        proj = pos @ self.Wr.t()                         # (T, F) = X Wr^T
        r = torch.cat([proj.cos(), proj.sin()], dim=-1) * self.scale   # (T, 2F), Eq. 2
        return self.mlp(r)                               # (T, d_model), Eq. 6


class _AxialBlock(nn.Module):
    """One factorized 2-D attention block: channel-axis attention (mixing the few
    sensors) followed by time-axis attention. Both are the project's vanilla
    pre-norm Transformer blocks (``method='none'`` -> no in-attention PE); the
    channel/time positions are injected once into the tokens upstream."""

    def __init__(self, d_model, n_heads, n_channels, max_len, dropout):
        super().__init__()
        self.chan = PETransformerEncoderLayer('none', d_model, n_heads, max(n_channels, 2), dropout)
        self.time = PETransformerEncoderLayer('none', d_model, n_heads, max_len, dropout)

    def forward(self, x):                                   # x: (B, C, T, d)
        B, C, T, d = x.shape
        xc = x.permute(0, 2, 1, 3).reshape(B * T, C, d)     # channel-axis attention
        xc = self.chan(xc).reshape(B, T, C, d).permute(0, 2, 1, 3)
        xt = xc.reshape(B * C, T, d)                        # time-axis attention
        xt = self.time(xt).reshape(B, C, T, d)
        return xt


class ViT2DFeatureExtractor(nn.Module):
    """SensorLM-style 2-D encoder over the raw ``[sensor-channel x time]`` grid,
    kept interface-compatible with the TCN / Transformer backbones:
    ``(B, C_in, T) -> (B, output_dims, T)``.

    Each ``(channel, time)`` cell is embedded to a token. Positions follow the
    metric/non-metric split: the CHANNEL axis (HR, steps, ... have no meaningful
    ordering or distance) gets a DISCRETE learnable embedding, while the metric TIME
    position of each timestep -- taken as the MULTI-DIMENSIONAL point
    ``[within-day phase, day-of-week]`` -- gets the multi-dimensional Learnable Fourier
    encoding (Li et al. 2021, Algorithm 1), which encodes both time dimensions JOINTLY
    (one shared Wr) so its shift-invariant / L2-kernel prior spans the daily + weekly
    circadian scales. The L2 prior is only valid on such metric axes, hence the channel
    axis stays discrete. Attention is factorized (channel-axis then time-axis) -- cheap because
    there are only a handful of sensor channels -- and the time resolution T is
    preserved (temporal patch = 1), so the CoST TFD/SFD heads and the plain-SSL
    read-out downstream are unchanged.
    """

    def __init__(self, n_channels, output_dims, depth=4, n_heads=8, dropout=0.1,
                 max_len=2048, lf_freqs=None, lf_gamma=1.0, bins_per_day=96):
        super().__init__()
        d = output_dims
        n_heads = n_heads if d % n_heads == 0 else 1
        self.n_channels = n_channels
        self.d_model = d
        self.bins_per_day = int(bins_per_day)
        self.value_embed = nn.Linear(1, d)                      # scalar cell -> token
        self.chan_embed = nn.Embedding(n_channels, d)           # discrete (non-metric) channel position
        # multi-dimensional Learnable Fourier PE on the metric TIME position
        # [within-day phase, day-of-week] (Li et al. 2021, Algorithm 1, M=2)
        self.time_pe = LearnableFourierMultiDim(d, n_dims=2, n_freqs=lf_freqs, gamma=lf_gamma)
        self.in_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            _AxialBlock(d, n_heads, n_channels, max_len, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d)

    def _time_positions(self, T, device):
        """Metric 2-D time position per timestep: [within-day phase, day-of-week index],
        each normalised to [0,1) so the learnable Fourier frequencies see comparable scales."""
        t = torch.arange(T, device=device, dtype=torch.float32)
        bpd = float(self.bins_per_day)
        within_day = torch.remainder(t, bpd) / bpd                       # clock phase, [0,1)
        n_days = max(1.0, T / bpd)
        day = torch.div(t, bpd, rounding_mode="floor") / n_days          # day index, [0,1)
        return torch.stack([within_day, day], dim=-1)                    # (T, 2)

    def forward(self, x):                                       # x: (B, C_in, T) raw sensor grid
        B, C, T = x.shape
        tok = self.value_embed(x.unsqueeze(-1))                 # (B, C, T, d)
        chan = self.chan_embed(torch.arange(C, device=x.device))    # (C, d)  discrete channel PE
        time = self.time_pe(self._time_positions(T, x.device)).to(tok.dtype)   # (T, d) multi-dim LFF
        tok = tok + chan[None, :, None, :] + time[None, None, :, :]
        tok = self.in_drop(tok)
        for blk in self.blocks:
            tok = blk(tok)
        tok = self.norm(tok)
        tok = tok.mean(dim=1)                                   # aggregate channels -> (B, T, d)
        return tok.transpose(1, 2)                              # (B, output_dims, T)


class PlainViTFeatureExtractor(nn.Module):
    """Vanilla ViT (Dosovitskiy et al. 2021) over the raw ``[sensor-channel x time]``
    grid, kept interface-compatible with the other backbones: ``(B, C_in, T) ->
    (B, output_dims, T)``. Sits ALONGSIDE the SensorLM-style axial ``ViT2DFeatureExtractor``
    (``backbone='vit'``) as a literal reading of the original paper's design, rather than
    that backbone's intrinsic multi-dim Learnable-Fourier + axial-attention scheme:

      * Patch embedding: each ``(channel, timestep)`` scalar cell is linearly projected to
        ``d_model`` -- this project's temporal resolution T must be preserved for the
        downstream TFD (Conv1d) / SFD (FFT) heads, so there is no time-axis patching
        (patch size 1), same tokenisation as ``backbone='vit'``.
      * Positional encoding: ONE shared 1-D learnable absolute embedding over the
        FLATTENED ``(channel, time)`` sequence of length ``C*T`` -- reuses the exact
        ``add_absolute_pe(..., 'learnable', ...)`` mechanism the plain Transformer
        backbone uses -- exactly the original ViT's single learned position-embedding
        table, added once before the encoder stack. No metric/non-metric axis split,
        no Learnable-Fourier.
      * Attention: standard JOINT multi-head self-attention over all ``C*T`` tokens
        (``PETransformerEncoderLayer`` with ``method='none'``), NOT the axial
        (channel-then-time) factorisation ``backbone='vit'`` uses -- faithful to the
        original paper's full self-attention, at the cost of ``O((C*T)^2)`` instead of
        the axial backbone's much cheaper ``O(T*(C^2+T))``.
      * No CLS token: the paper's classification token has no role here since the
        encoder must emit a PER-TIMESTEP sequence for the trend/seasonal decomposition
        heads downstream, not a single global vector, so it is intentionally omitted.

    WARNING: full O((C*T)^2) attention over C=n_channels x T=length tokens is far more
    memory-hungry than the axial 'vit' backbone at the same T -- use a much smaller
    batch size (see VIT_PLAIN_BATCH_SIZE in scripts/run.sh).
    """

    def __init__(self, n_channels, output_dims, depth=4, n_heads=8, dropout=0.1, max_len=2048):
        super().__init__()
        d = output_dims
        n_heads = n_heads if d % n_heads == 0 else 1
        self.n_channels = n_channels
        self.d_model = d
        self.value_embed = nn.Linear(1, d)                       # "patch" (1x1 cell) embedding
        # single flat learnable position-embedding table over the whole (channel, time)
        # sequence -- the original ViT's one learned positional embedding, not split by axis
        self.pos_embed = nn.Embedding(n_channels * max_len, d)
        self.in_drop = nn.Dropout(dropout)
        seq_len = n_channels * max_len
        self.blocks = nn.ModuleList([
            PETransformerEncoderLayer('none', d, n_heads, seq_len, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d)

    def forward(self, x):                                       # x: (B, C_in, T) raw sensor grid
        B, C, T = x.shape
        tok = self.value_embed(x.unsqueeze(-1))                 # (B, C, T, d)
        tok = tok.reshape(B, C * T, self.d_model)               # ONE flat sequence, ViT-style
        tok = add_absolute_pe(tok, "learnable", self.d_model, learnable_pe=self.pos_embed)
        tok = self.in_drop(tok)
        for blk in self.blocks:
            tok = blk(tok)                                      # full joint self-attention
        tok = self.norm(tok)
        tok = tok.reshape(B, C, T, self.d_model).mean(dim=1)    # aggregate channels -> (B, T, d)
        return tok.transpose(1, 2)                              # (B, output_dims, T)


class CoSTEncoder(nn.Module):
    def __init__(self, input_dims, output_dims,
                 kernels: List[int],
                 length: int,
                 hidden_dims=64, depth=10,
                 mask_mode='none', backbone='tcn', pe='sinusoidal',
                 n_time_features=0, time2vec_dim=16, disentangle=True,
                 bins_per_day=96, mask_prob=0.5):
        super().__init__()

        component_dims = output_dims // 2

        self.input_dims = input_dims
        self.n_time_features = n_time_features
        self.n_sensor_dims = input_dims - n_time_features
        self.output_dims = output_dims
        self.component_dims = component_dims
        self.hidden_dims = hidden_dims
        mask_mode = mask_mode.lower()
        if mask_mode not in SUPPORTED_MASK_MODES:
            raise ValueError(f"mask_mode must be one of {SUPPORTED_MASK_MODES}, got: {mask_mode}")
        self.mask_mode = mask_mode
        # Keep-probability of the binomial training mask (0.5 -> ~50% of timesteps zeroed).
        # Higher = LESS masking. Only read when mask_mode='binomial'; inert otherwise.
        # CAUTION for this project: the mask stacks on top of nan_mask (real non-wear gaps)
        # and the SFD branch contrasts the rFFT AMPLITUDE and PHASE of the representation,
        # so random timestep dropout injects broadband noise into the exact quantity the
        # seasonal objective is fitting. A/B one seed before trusting a masked run.
        self.mask_prob = mask_prob
        self.backbone = backbone
        self.pe = pe.lower()
        # disentangle=False -> PLAIN encoder: a single representation of dim
        # `output_dims` straight from the backbone (no TFD trend head, no SFD
        # seasonal Fourier head). Same backbone/PE/augmentations as CoST, so the
        # only difference is the ABSENCE of the trend/seasonal split entirely.
        self.disentangle = disentangle

        # Time2Vec (Kazemi et al. 2019) is fed as an INPUT feature: t2v(tau) of
        # size `time2vec_dim` (1 linear + (time2vec_dim-1) learnable-frequency
        # sines) is concatenated to the sensor channels BEFORE input_fc, exactly
        # as the paper's x'_j = [x_j ; t2v(tau)]. It is NOT added as a PE, and it
        # works with either backbone (the Transformer then uses vanilla attention).
        self.t2v = Time2VecPE(time2vec_dim, length) if self.pe == 'time2vec' else None
        t2v_in = time2vec_dim if self.t2v is not None else 0
        # input_fc sees the sensor channels (+ Time2Vec when enabled). The appended
        # clock/time-feature channels (time-of-day / day-of-week) are projected
        # separately and added as a temporal encoding (self.time_fc), so they
        # neither pass through input_fc nor get decomposed by the seasonal branch.
        self.input_fc = nn.Linear(self.n_sensor_dims + t2v_in, hidden_dims)
        # The calendar PEs replace the linear time_fc with a wall-clock encoding of the 2 raw
        # [tod, dow] index channels: 'factorized' = two learnable lookup tables, 'circular' =
        # a fixed sin/cos basis. Both sit at the SAME place as the backbone's other absolute
        # PEs -- inside the Transformer at output_dims, so `learnable` vs `factorized` differs
        # only in the ANCHOR (index vs calendar) -- while the TCN, which has no PE site, gets
        # them at the input projection. Same site and width for both, so 'circular' vs
        # 'factorized' isolates the encoder from the reference frame.
        cal_dim = output_dims if backbone == 'transformer' else hidden_dims
        self.cal_pe = (FactorizedCalendarPE(bins_per_day, cal_dim) if self.pe == 'factorized'
                       else CircularCalendarPE(bins_per_day, cal_dim) if self.pe == 'circular'
                       else None)
        self.time_fc = (nn.Linear(n_time_features, hidden_dims)
                        if n_time_features > 0 and self.cal_pe is None else None)

        # The TCN is position-aware through its convolutions, so it takes no PE
        # ('none'); the Transformer always needs a PE and selects it via `pe`.
        if backbone == 'transformer':
            self.feature_extractor = TransformerFeatureExtractor(
                hidden_dims, output_dims, depth=depth, pe=self.pe
            )
        elif backbone == 'vit':
            # ViT-2D over the raw [sensor-channel x time] grid. It builds its own
            # token/positional embedding (discrete channel + MULTI-DIMENSIONAL Learnable
            # Fourier on the [within-day, day-of-week] time position), so it bypasses
            # input_fc / Time2Vec and takes no swappable PE.
            if self.pe != 'none':
                raise ValueError(
                    "vit backbone uses an intrinsic multi-dimensional Learnable-Fourier "
                    "(time) + discrete(channel) encoding; use pe='none'."
                )
            if n_time_features > 0:
                raise ValueError(
                    "vit backbone does not support appended clock/time features "
                    "(run without --with-clock-features)."
                )
            self.feature_extractor = ViT2DFeatureExtractor(
                self.n_sensor_dims, output_dims, depth=depth, max_len=length,
                bins_per_day=bins_per_day
            )
        elif backbone == 'vit_plain':
            # Vanilla ViT (Dosovitskiy et al. 2021): one flat learnable positional
            # embedding + full joint self-attention over the raw [sensor-channel x time]
            # grid -- see PlainViTFeatureExtractor's docstring for how this differs from
            # backbone='vit'. It also bypasses input_fc / Time2Vec and takes only the
            # single 'learnable' PE (the original paper's standard configuration).
            if self.pe != 'learnable':
                raise ValueError(
                    "vit_plain backbone uses the original ViT's single learnable "
                    "absolute positional embedding; use pe='learnable'."
                )
            if n_time_features > 0:
                raise ValueError(
                    "vit_plain backbone does not support appended clock/time features "
                    "(run without --with-clock-features)."
                )
            self.feature_extractor = PlainViTFeatureExtractor(
                self.n_sensor_dims, output_dims, depth=depth, max_len=length
            )
        elif backbone == 'tcn':
            # The calendar PEs are allowed: convolutions carry RELATIVE position but no
            # absolute calendar phase, which is what they supply (added at input_fc). Both are
            # input-side encodings, so unlike the attention PEs they run on either backbone --
            # which is what lets a reference-frame contrast be read free of the backbone.
            if self.pe not in ('none', 'time2vec') + CALENDAR_PES:
                raise ValueError(
                    "TCN backbone supports pe in ('none', 'time2vec', 'factorized', "
                    f"'circular'), got: {self.pe}"
                )
            self.feature_extractor = DilatedConvEncoder(
                hidden_dims,
                [hidden_dims] * depth + [output_dims],
                kernel_size=3
            )
        else:
            raise ValueError(
                f"backbone must be one of 'tcn', 'transformer', 'vit', 'vit_plain', got: {backbone}"
            )

        self.repr_dropout = nn.Dropout(p=0.1)

        self.kernels = kernels

        # Trend (TFD) and Seasonal (SFD) disentangling heads exist ONLY when
        # disentangling. In plain mode the backbone output is the representation.
        if self.disentangle:
            self.tfd = nn.ModuleList(
                [nn.Conv1d(output_dims, component_dims, k, padding=k-1) for k in kernels]
            )
            self.sfd = nn.ModuleList(
                [BandedFourierLayer(output_dims, component_dims, b, 1, length=length) for b in range(1)]
            )

    def forward(self, x, tcn_output=False, mask=None):  # x: B x T x input_dims
        # peel off the appended clock/time channels; they are injected below as an
        # additive temporal encoding rather than mixed through input_fc and
        # decomposed by the seasonal branch.
        x_time = None
        if self.n_time_features > 0:
            x_time = x[..., self.n_sensor_dims:]
            x = x[..., :self.n_sensor_dims]

        nan_mask = ~x.isnan().any(axis=-1)
        x = x.clone()                                  # don't mutate the caller's tensor / a view
        x[~nan_mask] = 0

        # The ViT-2D / vit_plain backbones consume the RAW [sensor-channel x time] grid
        # and build their own token/positional embedding, so they bypass input_fc /
        # Time2Vec entirely. Masking of non-wear timesteps below still applies (it zeros
        # whole timesteps).
        if self.backbone not in ('vit', 'vit_plain'):
            # Time2Vec fed as input (Kazemi et al. 2019, x'_j = [x_j ; t2v(tau)]):
            # concat t2v(tau) to the sensor channels before input_fc.
            if self.t2v is not None:
                t2v = self.t2v(x.size(1), x.device, x.dtype)        # T x time2vec_dim
                x = torch.cat([x, t2v.unsqueeze(0).expand(x.size(0), -1, -1)], dim=-1)
            x = self.input_fc(x)  # B x T x Ch  (sensors [+ Time2Vec])
            # clock covariates (time-of-day / day-of-week) as an additive temporal
            # encoding -- kept out of input_fc and the seasonal decomposition
            if self.time_fc is not None:
                x = x + self.time_fc(x_time)

        # generate & apply mask. mask=None (the default) resolves to the CONFIGURED
        # augmentation while training and to no masking at eval, so masking is opt-in
        # via mask_mode instead of being silently unreachable -- see SUPPORTED_MASK_MODES.
        # With the default mask_mode='none' this is a no-op in both modes, i.e. bit-for-bit
        # the old mask='all_true' behaviour. Callers may still force a specific mask.
        if mask is None:
            mask = self.mask_mode if self.training else 'none'

        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1), p=self.mask_prob).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask in ('none', 'all_true'):   # 'all_true' kept as the upstream-CoST alias
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False

        mask &= nan_mask
        x[~mask] = 0

        # Factorized calendar PE on the TCN path -- added AFTER masking, deliberately. In
        # WavesFM (Cao et al. 2026, Sec. 3.2) a missing bin is replaced by a [MISSING] token
        # and Eq. 2 is applied on top, so the model always knows which calendar slot a gap
        # sits in; that is the point of anchoring to the clock rather than to the index.
        # Adding it before `x[~mask] = 0` zeroed the anchor at exactly the non-wear bins,
        # and since the Transformer receives it inside the backbone (i.e. after masking) the
        # two backbones were not running the same encoding -- which would have confounded
        # any tcn/factorized vs transformer/factorized comparison.
        if self.cal_pe is not None and self.backbone != 'transformer':
            x = x + self.cal_pe(x_time)

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        if self.cal_pe is not None and self.backbone == 'transformer':
            x = self.feature_extractor(x, self.cal_pe(x_time))  # B x Co x T
        else:
            x = self.feature_extractor(x)  # B x Co x T

        if tcn_output:
            return x.transpose(1, 2)

        # PLAIN mode: the backbone output IS the (single) representation of dim
        # output_dims -- no trend/seasonal split. Second element None signals
        # "no seasonal branch" to the loss and to encode().
        if not self.disentangle:
            return self.repr_dropout(x.transpose(1, 2)), None

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
