"""Disentangling encoder: a sequence backbone, then trend and seasonal heads.

    x (B, T, C) -> input projection (B, T, H) [+ temporal encoding]
      -> backbone (models.backbones: TCN, Transformer or Mamba) -> z (B, T, D)
    trend    V^T (B, T, D/2): mean of causal convolution experts, kernels 1, 2, 4, ..., T/8
    seasonal V^S (B, T, D/2): banded Fourier layers, one band per daily harmonic

Derived from salesforce/CoST (BSD-3), which vendors the dilated-convolution backbone of
TS2Vec (MIT). BandedFourierLayer and the heads are theirs, modified; see NOTICE.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import fft, nn

from models.backbones import TemporalEncoding, build_backbone
from models.dilated_conv import depth_for_window, receptive_field

__all__ = ["receptive_field", "depth_for_window", "rhythm_bands", "BandedFourierLayer",
           "CoSTEncoder", "DecomposedEncoder", "daily_average_kernel", "decompose_daily"]


def rhythm_bands(seq_len: int, bins_per_day: int, max_harmonics: int = 4):
    """rFFT bin ranges, one per resolvable daily harmonic.

    The 24 h cycle sits at bin D = seq_len // bins_per_day and harmonic k at k*D. Band edges
    are the midpoints between neighbouring harmonics. The first band starts at bin 1, so bin 0
    (the window mean) belongs to the trend branch. Harmonics at or above Nyquist are dropped.

        HRD    T=672, 96/day -> harmonics 7,14,21,28 -> [(1,10),(10,17),(17,24),(24,31)]
               max_harmonics=12 -> 12 bands, (1,10) ... (80,87)
        GLOBEM T=112,  4/day -> harmonics 28,56      -> [(1,42),(42,57)]

    Returns one full-spectrum band when fewer than two harmonics are resolvable.
    """
    total = seq_len // 2 + 1
    D = max(1, seq_len // int(bins_per_day))
    centres = [k * D for k in range(1, max_harmonics + 1) if k * D < total]
    if D < 2 or len(centres) < 2:
        return [(0, total)]

    edges = [(1, (centres[0] + centres[1]) // 2)]
    for i in range(1, len(centres)):
        lo = (centres[i - 1] + centres[i]) // 2
        hi = ((centres[i] + centres[i + 1]) // 2 if i + 1 < len(centres)
              else centres[i] + (centres[i] - centres[i - 1]) // 2)
        edges.append((lo, min(hi, total)))
    return edges


class BandedFourierLayer(nn.Module):
    """A learned complex linear map applied to one contiguous slice of the rFFT spectrum.

    Everything outside the band is zeroed before the inverse transform, so each band's
    output is a signal containing only its own periods.
    """

    def __init__(self, in_channels, out_channels, bounds, length):
        super().__init__()
        self.total_freqs = length // 2 + 1
        self.start, self.end = int(bounds[0]), min(int(bounds[1]), self.total_freqs)
        if not 0 <= self.start < self.end <= self.total_freqs:
            raise ValueError(f"band {bounds} outside spectrum of {self.total_freqs} bins")
        self.out_channels = out_channels
        n = self.end - self.start
        self.weight = nn.Parameter(torch.empty((n, in_channels, out_channels), dtype=torch.cfloat))
        self.bias = nn.Parameter(torch.empty((n, out_channels), dtype=torch.cfloat))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):                                   # B x T x C_in
        b, t, _ = x.shape
        with torch.autocast(device_type="cuda", enabled=False):
            xf = fft.rfft(x.float(), dim=1)
            out = torch.zeros(b, t // 2 + 1, self.out_channels,
                              device=x.device, dtype=torch.cfloat)
            out[:, self.start:self.end] = (
                torch.einsum("bti,tio->bto", xf[:, self.start:self.end], self.weight) + self.bias)
            return fft.irfft(out, n=t, dim=1)


def daily_average_kernel(bins_per_day):
    """Centred 24-h moving average: [0.5, 1, ..., 1, 0.5] / N over N+1 taps.

    Odd length and symmetric, so the filter is ZERO PHASE -- a phase-shifted trend would
    corrupt the phase of its complement, which is the quantity the decomposition exists to
    protect. Its transfer function sin(pi f N)/(N sin(pi f)) has exact zeros at f = k/N, i.e.
    at 1/day and every daily harmonic, so the trend view cannot carry the daily rhythm.
    """
    n = int(bins_per_day)
    if n < 2:
        raise ValueError(f"a 24-h average needs at least 2 bins per day, got {n}")
    w = torch.ones(n + 1)
    w[0] = w[-1] = 0.5
    return w / n


def decompose_daily(x, kernel):
    """x -> (MA24h(x), x - MA24h(x)), circularly padded and preserving missingness.

    Windows are whole days starting at midnight, so circular padding keeps the length and
    every rhythm's phase exactly. Missing bins are filtered as 0 and restored as NaN in both
    views, so each encoder sees the same missingness it would have seen on the raw input.
    """
    finite = torch.isfinite(x)
    filled = torch.where(finite, x, torch.zeros_like(x))
    b, t, c = filled.shape
    pad = (kernel.numel() - 1) // 2
    flat = filled.permute(0, 2, 1).reshape(b * c, 1, t)
    trend = F.conv1d(F.pad(flat, (pad, pad), mode="circular"),
                     kernel.to(flat.dtype).view(1, 1, -1)).reshape(b, c, t).permute(0, 2, 1)
    missing = torch.full_like(x, float("nan"))
    return (torch.where(finite, trend, missing),
            torch.where(finite, filled - trend, missing))


class DecomposedEncoder(nn.Module):
    """Two independent encoders over a fixed 24-h decomposition of the INPUT.

    The measured failure this addresses: with one shared backbone whose receptive field
    (1021) exceeds the window (672), every timestep already encodes the whole window, so the
    time-averaged trend readout carries rhythm regardless of what the branch is called. At
    random initialisation the shared model's trend block already predicts 24-h amplitude at
    R^2 0.827 and acrophase at 0.642 -- leakage is geometric, not learned, so no objective can
    remove it. Splitting the INPUT removes the access: the trend encoder never sees the daily
    rhythm, because the filter has exact zeros there.

    Each sub-encoder takes half the width and builds only its own head, so the pair costs
    about what the single shared encoder did. Nothing is shared after the split.
    """

    def __init__(self, **enc):
        super().__init__()
        width = enc.pop("output_dims")
        if width % 2:
            raise ValueError(f"decomposed output_dims must be even, got {width}")
        self.register_buffer("daily_kernel", daily_average_kernel(enc["bins_per_day"]))
        self.trend_net = CoSTEncoder(output_dims=width // 2, branch="trend", **enc)
        self.seasonal_net = CoSTEncoder(output_dims=width // 2, branch="seasonal", **enc)
        self.seq_len, self.bins_per_day = enc["seq_len"], enc["bins_per_day"]
        self.harmonics = self.seasonal_net.harmonics
        self.band_readout = self.seasonal_net.band_readout
        self.trend_dims = self.trend_net.trend_dims
        self.seasonal_dims = self.seasonal_net.seasonal_dims
        self.depth = self.trend_net.depth
        self.receptive_field = self.trend_net.receptive_field

    # Read back by the readout and the recorded config; properties, so the submodules are not
    # registered twice and the state dict keeps one name per parameter.
    @property
    def bands(self):
        return self.seasonal_net.bands

    @property
    def sfd(self):
        return self.seasonal_net.sfd

    @property
    def kernels(self):
        return self.trend_net.kernels

    def forward(self, x, tcn_output=False, mask=None):
        x_trend, x_seasonal = decompose_daily(x, self.daily_kernel)
        if tcn_output:
            return self.seasonal_net(x_seasonal, tcn_output=True, mask=mask)
        trend, _ = self.trend_net(x_trend, mask=mask)
        _, season = self.seasonal_net(x_seasonal, mask=mask)
        return trend, season


class CoSTEncoder(nn.Module):
    """Sensor window -> (trend V^T, seasonal V^S), each B x T x width.

    The backbone is chosen by name (models.backbones.BACKBONES). `tcn_depth` is the TCN's
    number of dilation levels (None = smallest receptive field covering the window);
    `n_layers` is the Transformer/Mamba layer count, `n_heads` the Transformer's attention
    heads, `bidirectional` whether Mamba also scans backwards.
    """

    def __init__(self, input_dims, output_dims, seq_len, bins_per_day, *,
                 hidden_dims=64, n_time_features=0, seasonal_bands="harmonics",
                 mask_mode="none", mask_prob=0.5, max_harmonics=4, trend_kernel_cap=None,
                 seasonal_frac=0.5, band_readout=False, backbone="tcn",
                 temporal_encoding="none", tcn_depth=None, n_layers=4, n_heads=4,
                 bidirectional=True, branch="both"):
        super().__init__()
        if seasonal_bands not in ("single", "harmonics"):
            raise ValueError(f"unsupported seasonal bands: {seasonal_bands}")
        if branch not in ("both", "trend", "seasonal"):
            raise ValueError(f"branch must be 'both', 'trend' or 'seasonal', got {branch!r}")
        self.branch = branch
        self.backbone_name = backbone
        self.seq_len, self.bins_per_day = seq_len, bins_per_day
        self.harmonics = int(max_harmonics)          # read back by cost.spectral_readout
        self.band_readout = bool(band_readout)       # encode-time only; see cost.band_keep
        self.n_time_features = n_time_features
        self.n_sensor_dims = input_dims - n_time_features
        self.mask_mode, self.mask_prob = mask_mode, mask_prob

        self.input_fc = nn.Linear(self.n_sensor_dims, hidden_dims)
        self.time_fc = nn.Linear(n_time_features, hidden_dims) if n_time_features else None
        self.temporal_encoding = TemporalEncoding(temporal_encoding, hidden_dims, bins_per_day)
        self.feature_extractor = build_backbone(
            backbone, width=hidden_dims, output_dims=output_dims, seq_len=seq_len,
            tcn_depth=tcn_depth, n_layers=n_layers, n_heads=n_heads, bidirectional=bidirectional)
        self.depth = getattr(self.feature_extractor, "depth", None)
        self.receptive_field = getattr(self.feature_extractor, "receptive_field", None)
        self.repr_dropout = nn.Dropout(p=0.1)

        # A single-branch encoder spends its whole width on that branch and builds no head for
        # the other, so DecomposedEncoder's two encoders cost what one shared encoder does.
        if branch == "trend":
            self.trend_dims, self.seasonal_dims = output_dims, 0
        elif branch == "seasonal":
            self.trend_dims, self.seasonal_dims = 0, output_dims
        else:
            self.seasonal_dims = int(round(output_dims * seasonal_frac))
            self.trend_dims = output_dims - self.seasonal_dims

        # Trend: causal convolution experts at powers-of-two kernels up to seq_len // 8
        # (upstream goes to seq_len // 2). The backbone already covers the window, so longer
        # kernels add parameters but no context.
        cap = trend_kernel_cap if trend_kernel_cap is not None else max(1, seq_len // 8)
        self.kernels = [2 ** i for i in range(int(math.floor(math.log2(max(cap, 1)))) + 1)]
        self.tfd = nn.ModuleList(
            [nn.Conv1d(output_dims, self.trend_dims, k, padding=k - 1) for k in self.kernels]
            if self.trend_dims else [])

        # Seasonal: one banded Fourier layer per band, widths splitting seasonal_dims.
        # seasonal_bands="single" is the upstream layer: one band over the whole spectrum.
        self.bands = (rhythm_bands(seq_len, bins_per_day, max_harmonics)
                      if seasonal_bands == "harmonics" else [(0, seq_len // 2 + 1)])
        nb = len(self.bands)
        widths = [self.seasonal_dims // nb] * nb
        widths[-1] += self.seasonal_dims - sum(widths)
        self.sfd = nn.ModuleList(
            [BandedFourierLayer(output_dims, w, b, seq_len) for w, b in zip(widths, self.bands)]
            if self.seasonal_dims else [])

    def forward(self, x, tcn_output=False, mask=None):      # x: B x T x input_dims
        x_time = None
        if self.n_time_features:
            x_time, x = x[..., self.n_sensor_dims:], x[..., :self.n_sensor_dims]

        nan_mask = ~x.isnan().any(dim=-1)
        x = x.clone()
        x[~nan_mask] = 0
        x = self.input_fc(x)
        if self.time_fc is not None:
            x = x + self.time_fc(x_time)
        x = self.temporal_encoding(x)

        # Timestep masking: opt-in via mask_mode, off at eval unless a caller forces it.
        mode = (self.mask_mode if self.training else "none") if mask is None else mask
        if mode == "binomial":
            m = torch.rand(x.size(0), x.size(1), device=x.device) < self.mask_prob
        elif mode in ("none", "all_true"):
            m = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        else:
            raise ValueError(f"unknown mask mode: {mode!r}")
        m = m & nan_mask
        x[~m] = 0

        x = self.feature_extractor(x).transpose(1, 2)       # B x D x T
        if tcn_output:
            return x.transpose(1, 2)

        if self.tfd:
            trend = []
            for k, mod in zip(self.kernels, self.tfd):
                out = mod(x)
                if k != 1:                                  # left-trim: causal
                    out = out[..., :x.size(-1)]
                trend.append(out.transpose(1, 2))
            trend = torch.stack(trend, dim=0).mean(dim=0)   # B x T x trend_dims
        else:
            trend = x.new_zeros(x.size(0), x.size(2), 0)

        x = x.transpose(1, 2)                               # B x T x D
        season = (torch.cat([mod(x) for mod in self.sfd], dim=-1) if self.sfd
                  else x.new_zeros(x.size(0), x.size(1), 0))
        return trend, self.repr_dropout(season)
