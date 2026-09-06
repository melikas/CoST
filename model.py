"""TCN encoder for disentangled seasonal/trend representations of wearable time series.

TCN only. The Transformer backbone, the eight attention positional encodings and Time2Vec
are part 4 of the research plan and are deliberately absent; nothing here branches on a
backbone choice.

Two corrections against the previous encoder, both measured rather than assumed:

  1. DEPTH IS DERIVED FROM THE WINDOW, not fixed at 10. `DilatedConvEncoder` builds
     `depth + 1` blocks with dilations 2^0..2^depth, so the receptive field is
     RF = 1 + 4*(2^(depth+1) - 1). At the shipped depth=10 that is 8189 steps: 12x the
     HRD window (T=672) and 73x the GLOBEM window (T=112). On GLOBEM the last four
     blocks convolve almost entirely over zero padding -- they add parameters and
     destroy signal-to-noise while adding no context. `depth_for_window` returns the
     smallest depth whose RF covers the window: 7 for HRD (RF=1021, 1.52x) and 4 for
     GLOBEM (RF=125, 1.12x). Both verified empirically by gradient tracing.

  2. THE SEASONAL BANDS NO LONGER SILENTLY DEGENERATE. The previous rule required all
     four circadian harmonics to fit below Nyquist and fell back to a single
     full-spectrum band otherwise. On GLOBEM (T=112, 4 bins/day) the 24 h cycle sits at
     rFFT bin 28 of 57, so 4*D = 112 >= 57 and every GLOBEM run labelled
     `seasonal_bands=harmonics` was in fact running the unbanded upstream layer.
     `rhythm_bands` instead keeps as many harmonics as the sampling rate can actually
     resolve. HRD is unchanged bit-for-bit (4 bands, identical edges), which is what
     keeps comparisons against run 2224103 valid; GLOBEM gains 2 real bands.

Derived from salesforce/CoST (BSD-3), which vendors the dilated-convolution backbone of
TS2Vec (MIT). SamePadConv/ConvBlock/DilatedConvEncoder and BandedFourierLayer are theirs,
modified; see NOTICE.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import fft, nn

__all__ = ["receptive_field", "depth_for_window", "rhythm_bands", "CoSTEncoder"]


# --------------------------------------------------------------------------------------
# Geometry: the two corrections above, as pure functions so they can be tested and quoted
# --------------------------------------------------------------------------------------
def receptive_field(depth: int, kernel: int = 3) -> int:
    """Receptive field of the stack `DilatedConvEncoder` builds for `depth`.

    It builds `depth + 1` ConvBlocks (channels = [hidden]*depth + [output]), each holding
    two convolutions at dilation 2^i, so RF = 1 + 2*(k-1)*(2^(depth+1) - 1).
    """
    return 1 + 2 * (kernel - 1) * (2 ** (depth + 1) - 1)


def depth_for_window(seq_len: int, kernel: int = 3) -> int:
    """Smallest `depth` whose receptive field covers a window of `seq_len` steps.

    Covering the window once is the target. Going deeper does not add context -- there is
    no more window to see -- it only adds parameters and pads with zeros.
    """
    depth = 0
    while receptive_field(depth, kernel) < seq_len:
        depth += 1
    return depth


def rhythm_bands(seq_len: int, bins_per_day: int, max_harmonics: int = 4):
    """rFFT bin ranges bracketing the rhythmic periods this window can resolve.

    A period of P samples lands on rFFT bin `seq_len / P`. The circadian cycle is
    `bins_per_day` samples, so it sits at D = seq_len // bins_per_day, and its k-th
    harmonic at k*D. Bands span the midpoints between neighbouring harmonics, so every
    period gets its own weights and nothing is double-counted. The first band starts at
    bin 1, which folds the infradian content (weekly at bin seq_len/(7*bins_per_day))
    in with the circadian fundamental, and excludes bin 0 -- the DC term, which is the
    window mean and belongs to the trend branch.

    Harmonics at or above Nyquist are dropped rather than triggering a fallback. That is
    the whole fix: the old rule demanded all four and gave up otherwise.

        HRD    T=672, 96/day -> D=7,  harmonics 7,14,21,28 of 337 -> [(1,10),(10,17),(17,24),(24,31)]
        GLOBEM T=112,  4/day -> D=28, harmonics 28,56      of  57 -> [(1,42),(42,57)]

    Returns a single full-spectrum band only when fewer than two harmonics are resolvable,
    i.e. when the window genuinely cannot support banding.
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


# --------------------------------------------------------------------------------------
# Backbone (upstream TS2Vec, unmodified)
# --------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------
# Seasonal head
# --------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------
# Encoder
# --------------------------------------------------------------------------------------
class CoSTEncoder(nn.Module):
    """Sensor window -> (trend V^T, seasonal V^S).

    `disentangle=False` is the plain-SSL control: the backbone output is the single
    representation and no head is built, so the control differs from the model in exactly
    one thing.
    """

    def __init__(self, input_dims, output_dims, seq_len, bins_per_day, *,
                 hidden_dims=64, depth=None, kernels=None, n_time_features=0,
                 seasonal_bands="harmonics", disentangle=True, mask_mode="none",
                 mask_prob=0.5, trend_causal=True, max_harmonics=4,
                 trend_kernel_cap=None, seasonal_frac=0.5):
        super().__init__()
        if depth is None:
            depth = depth_for_window(seq_len)

        self.seq_len, self.bins_per_day = seq_len, bins_per_day
        self.n_time_features = n_time_features
        self.n_sensor_dims = input_dims - n_time_features
        self.disentangle, self.mask_mode, self.mask_prob = disentangle, mask_mode, mask_prob
        self.trend_causal = trend_causal
        self.depth = depth
        self.receptive_field = receptive_field(depth)

        self.input_fc = nn.Linear(self.n_sensor_dims, hidden_dims)
        self.time_fc = nn.Linear(n_time_features, hidden_dims) if n_time_features else None
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims, [hidden_dims] * depth + [output_dims], kernel_size=3)
        self.repr_dropout = nn.Dropout(p=0.1)

        if not disentangle:
            self.component_dims = output_dims
            self.kernels, self.tfd, self.sfd, self.bands = [], None, None, []
            return

        # Width split between V^T and V^S. The concatenation is always output_dims wide, so
        # the plain control keeps the same dimensionality whatever the split.
        #
        # seasonal_frac=0.5 is the upstream half-and-half. Raising it moves width to the
        # block that was measured to carry the RQ2 signal (V^S phase, C=0.8802, against
        # V^T's 0.6054). It is left at 0.5 by default ON PURPOSE: with one shot, the
        # pre-registered design already spends its degrees of freedom on phase_readout x
        # term-weights, and a width change is a third bet whose effect could not then be
        # attributed. The flag exists so it can be a fifth arm, not a silent default.
        self.seasonal_dims = int(round(output_dims * seasonal_frac))
        self.trend_dims = output_dims - self.seasonal_dims
        self.component_dims = self.seasonal_dims          # kept for callers reading V^S width

        # Trend: a mixture of causal AR experts at powers-of-two kernel sizes.
        #
        # WHY THEY ARE CAPPED. Upstream runs kernels up to seq_len//2, i.e. [1..256] on the
        # HRD window. That costs output_dims * trend_dims * sum(kernels) = 320*160*511 =
        # 26.2M parameters -- 96.5% of the entire encoder -- for the block that carries the
        # LEAST downstream signal. It is also redundant: `depth` is now sized so the
        # backbone's receptive field already covers the whole window, so every position the
        # trend head reads has already integrated all of it. A 256-wide expert stacked on
        # top of that re-reads context the backbone has supplied, at quadratic cost.
        #
        # The cap keeps multi-scale coverage and drops the redundant tail. seq_len//8 lands
        # near one day on HRD (84 bins x 15 min = 21 h) and near half a week on GLOBEM
        # (14 bins x 6 h = 3.5 d) -- in both cohorts the longest scale the trend still owns
        # before the seasonal branch takes over. HRD: [1..64], sum 127, a 4.0x cut.
        cap = trend_kernel_cap if trend_kernel_cap is not None else max(1, seq_len // 8)
        self.kernels = kernels or [2 ** i for i in
                                   range(int(math.floor(math.log2(max(cap, 1)))) + 1)]
        self.tfd = nn.ModuleList(
            [nn.Conv1d(output_dims, self.trend_dims, k, padding=k - 1) for k in self.kernels])

        # Seasonal: one banded Fourier layer per resolvable rhythm, widths splitting
        # seasonal_dims so V^S keeps its width whatever the band count.
        self.bands = (rhythm_bands(seq_len, bins_per_day, max_harmonics)
                      if seasonal_bands == "harmonics" else [(0, seq_len // 2 + 1)])
        nb = len(self.bands)
        widths = [self.seasonal_dims // nb] * nb
        widths[-1] += self.seasonal_dims - sum(widths)
        self.sfd = nn.ModuleList(
            [BandedFourierLayer(output_dims, w, b, seq_len) for w, b in zip(widths, self.bands)])

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

        # Masking is opt-in via mask_mode and off at eval unless a caller forces it.
        mode = (self.mask_mode if self.training else "none") if mask is None else mask
        if mode == "binomial":
            m = torch.rand(x.size(0), x.size(1), device=x.device) < self.mask_prob
        elif mode in ("none", "all_true"):
            m = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        else:
            raise ValueError(f"unknown mask mode: {mode!r}")
        m = m & nan_mask
        x[~m] = 0

        x = self.feature_extractor(x.transpose(1, 2))       # B x C x T
        if tcn_output:
            return x.transpose(1, 2)
        if not self.disentangle:
            return self.repr_dropout(x.transpose(1, 2)), None

        trend = []
        for k, mod in zip(self.kernels, self.tfd):
            out = mod(x)
            if k != 1:                                      # left-trim to keep it causal
                lo = 0 if self.trend_causal else (k - 1) // 2
                out = out[..., lo:lo + x.size(-1)]
            trend.append(out.transpose(1, 2))
        trend = torch.stack(trend, dim=0).mean(dim=0)       # B x T x component_dims

        x = x.transpose(1, 2)                               # B x T x C
        season = torch.cat([mod(x) for mod in self.sfd], dim=-1)
        return trend, self.repr_dropout(season)
