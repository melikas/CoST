import math
from typing import List

import torch
from torch import nn
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
    TUPEPosition,
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


def seasonal_band_edges(length, bins_per_day):
    """rFFT bin ranges bracketing the weekly fundamental and the first circadian harmonics.

    D = length // bins_per_day is the 24 h bin (7 for a 672-step window at 96 bins/day), so the
    harmonics sit at D, 2D, 3D, 4D. Each band spans the midpoints between neighbouring
    harmonics, so every meaningful period gets its own weights and nothing is double-counted.
    Falls back to a single full-spectrum band when the window is too short to resolve them,
    which reproduces the original layer exactly.
    """
    total = (length // 2) + 1
    D = max(1, length // int(bins_per_day))
    if 4 * D >= total or D < 2:
        return [(0, total)]
    edges, centres = [], [D, 2 * D, 3 * D, 4 * D]
    edges.append((1, (centres[0] + centres[1]) // 2))          # weekly fundamental + 24 h
    for i in range(1, len(centres)):
        lo = (centres[i - 1] + centres[i]) // 2
        hi = ((centres[i] + centres[i + 1]) // 2 if i + 1 < len(centres)
              else centres[i] + (centres[i] - centres[i - 1]) // 2)
        edges.append((lo, min(hi, total)))
    return edges


class BandedFourierLayer(nn.Module):
    def __init__(self, in_channels, out_channels, band, num_bands, length=201, bounds=None):
        super().__init__()

        self.length = length
        self.total_freqs = (self.length // 2) + 1

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.band = band  # zero indexed
        self.num_bands = num_bands

        if bounds is None:
            # Equal split of the whole spectrum -- the original behaviour.
            self.num_freqs = self.total_freqs // self.num_bands + (self.total_freqs % self.num_bands if self.band == self.num_bands - 1 else 0)
            self.start = self.band * (self.total_freqs // self.num_bands)
            self.end = self.start + self.num_freqs
        else:
            # Explicit edges, so a band can be anchored on a physiological period rather than
            # on an arbitrary equal division of the spectrum.
            self.start, self.end = int(bounds[0]), min(int(bounds[1]), self.total_freqs)
            assert 0 <= self.start < self.end <= self.total_freqs, (bounds, self.total_freqs)
            self.num_freqs = self.end - self.start


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
        # TUPE shares one positional-correlation term across ALL layers (Ke et al. 2021).
        self.tupe = TUPEPosition(output_dims, n_heads, max_len) if pe == 'tupe' else None
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
        pos_bias = self.tupe(x.size(1), x.device) if self.tupe is not None else None
        for layer in self.layers:
            x = layer(x, pos_bias)
        x = self.norm(x)
        return x.transpose(1, 2)             # B x output_dims x T


class CoSTEncoder(nn.Module):
    def __init__(self, input_dims, output_dims,
                 kernels: List[int],
                 length: int,
                 hidden_dims=64, depth=10,
                 mask_mode='none', backbone='tcn', pe='sinusoidal',
                 n_time_features=0, time2vec_dim=65, disentangle=True,
                 bins_per_day=96, mask_prob=0.5, seasonal_bands='harmonics'):
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
                f"backbone must be one of 'tcn', 'transformer', got: {backbone}"
            )

        self.repr_dropout = nn.Dropout(p=0.1)

        self.kernels = kernels

        # Trend (TFD) and Seasonal (SFD) disentangling heads exist ONLY when
        # disentangling. In plain mode the backbone output is the representation.
        if self.disentangle:
            self.tfd = nn.ModuleList(
                [nn.Conv1d(output_dims, component_dims, k, padding=k-1) for k in kernels]
            )
            # Bands anchored on the circadian harmonics rather than one band over the whole
            # spectrum. At bins_per_day=96 and length=672, D=7 is the 24 h bin, so the bands
            # below bracket the weekly fundamental, 24 h, 12 h, and 8/6 h. Bins above 4D are
            # deliberately excluded: they are sub-6h content, which this project's hypothesis
            # (x = trend + seasonal + noise) treats as noise, so it belongs in the residual and
            # not in the seasonal component. `seasonal_bands` records the layout, since each
            # band now owns a fixed slice of V^(S).
            # 'harmonics' (default) anchors one band per circadian harmonic; 'single' is the
            # ONE full-spectrum band this layer had before banding existed. The choice is
            # explicit because it changes the architecture, and every run in results_hrd/ from
            # before the banding commit is 'single' -- a sweep meant to reproduce one of those
            # must be able to ask for it rather than silently training a different model.
            assert seasonal_bands in ("harmonics", "single"), seasonal_bands
            self.seasonal_bands = (seasonal_band_edges(length, bins_per_day)
                                   if seasonal_bands == "harmonics"
                                   else [(0, (length // 2) + 1)])
            nb = len(self.seasonal_bands)
            # Split component_dims across the bands so V^(S) keeps its contracted width; the
            # last band absorbs the remainder.
            widths = [component_dims // nb] * nb
            widths[-1] += component_dims - sum(widths)
            self.seasonal_widths = widths
            self.sfd = nn.ModuleList(
                [BandedFourierLayer(output_dims, w, b, nb, length=length, bounds=bd)
                 for b, (bd, w) in enumerate(zip(self.seasonal_bands, widths))]
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

        # Concatenate the per-band outputs: with one band this is the original single output,
        # with several each band contributes its own slice of V^(S).
        season = [mod(x) for mod in self.sfd]
        season = season[0] if len(season) == 1 else torch.cat(season, dim=-1)

        return trend, self.repr_dropout(season)
