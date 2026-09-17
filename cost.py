"""CoST pretraining and the frozen spectral readout.

MoCo on the trend branch, within-batch instance contrast of amplitude and phase on the
seasonal branch, as upstream. Derived from salesforce/CoST (BSD-3), which vendors TS2Vec
(MIT); see NOTICE. Differences from upstream: harmonic bands (models.encoder.rhythm_bands), a
smoothing augmentation in place of scaling, an amplitude-weighted circular phase contrast
(models.losses.seasonal_loss), the optional level-equivariance (w_eq) and anti-collapse (w_ac)
terms, and a readout of amplitude and phase at the daily harmonics.
"""
from __future__ import annotations

import math
import random
import hashlib
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import fft, nn
from torch.utils.data import DataLoader, Dataset

from models import losses as O
from models.encoder import CoSTEncoder

__all__ = ["PretrainDataset", "CoSTModel", "DSSL", "WEIGHTS", "REFERENCE_SHARED", "exact_numerics",
           "tf32_convolutions", "spectral_freqs", "spectral_readout", "band_keep",
           "smooth_bins_for"]


def exact_numerics():
    """Deterministic kernels and full float32 on GPU. By default an A100 runs convolutions in
    TF32 (about 1e-3 relative precision) with a kernel chosen by batch shape, so a window's
    encoding depended on its batch (up to 0.37% in Narval job 3068780) and a reloaded encoder
    did not reproduce it. Training switches convolutions back to TF32 (`tf32_convolutions`);
    every encoding is made in full float32. No effect on CPU."""
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@contextmanager
def tf32_convolutions():
    """TF32 convolutions while training: PyTorch's default, which the reported runs (train.py
    at 0a88999) never changed. Full float32 made the HRD CoST reference 8.6 s/update on a
    3g.20gb slice (Narval job 3072689). Training batches always have one shape, so runs stay
    deterministic."""
    old = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = old


# --------------------------------------------------------------------------------------
# Augmentations
# --------------------------------------------------------------------------------------
def smooth_bins_for(minutes, bins_per_day):
    """Widest smoothing box, in bins, for a width given in minutes: 5 bins on HRD at 75 min.
    Below 3 bins no odd box fits and smoothing is off (GLOBEM's 6-h bins)."""
    return int(minutes * bins_per_day // 1440)


class PretrainDataset(Dataset):
    """Two independently augmented views of the same window, and their level offset.

    `multiplier` revisits each window that many times per epoch with fresh augmentations.
    """

    def __init__(self, data, jitter_sigma=0.1, shift_sigma=0.5, p=0.5, multiplier=10,
                 smooth_bins=0, scale_sigma=0.0):
        super().__init__()
        self.data = data
        self.p, self.multiplier = p, multiplier
        self.jitter_sigma, self.shift_sigma = jitter_sigma, shift_sigma
        self.smooth_bins = int(smooth_bins)
        self.scale_sigma = float(scale_sigma)
        self.N, self.T, self.D = data.shape

    def __len__(self):
        return self.N * self.multiplier

    def __getitem__(self, item):
        """(view 1, view 2, per-channel offset d1 - d2). Only the equivariance term reads the
        offset, and tracking it draws no extra random numbers."""
        x = self.data[item % self.N]
        v1, d1 = self.transform(x)
        v2, d2 = self.transform(x)
        return v1, v2, d1 - d2

    def transform(self, x):
        if self.scale_sigma and random.random() <= self.p:
            x = x * (1 + torch.randn(x.size(-1)) * self.scale_sigma)
        x, d = self.shift(self.smooth(x))
        return self.jitter(x), d

    def smooth(self, x):
        """Circular box filter of random odd width up to `smooth_bins`. Circular and odd, so
        the window keeps its length and every rhythm keeps its phase exactly."""
        if self.smooth_bins < 3 or random.random() > self.p:
            return x
        w = random.randrange(3, (self.smooth_bins // 2) * 2 + 2, 2)
        xt = x.transpose(0, 1).unsqueeze(0)
        xp = F.pad(xt, ((w - 1) // 2, (w - 1) // 2), mode="circular")
        return F.avg_pool1d(xp, kernel_size=w, stride=1).squeeze(0).transpose(0, 1)

    def jitter(self, x):
        if random.random() > self.p:
            return x
        return x + torch.randn(x.shape) * self.jitter_sigma

    def shift(self, x):
        """Constant per-channel offset: changes only the f=0 bin (the MESOR). Returns the
        offset too, zero when no shift is applied."""
        if random.random() > self.p:
            return x, torch.zeros(x.size(-1))
        d = torch.randn(x.size(-1)) * self.shift_sigma
        return x + d, d


# --------------------------------------------------------------------------------------
# Readout
# --------------------------------------------------------------------------------------
def spectral_freqs(seq_len, bins_per_day, harmonics=4):
    """Resolved weekly and daily harmonics, excluding DC and Nyquist's fixed real phase.

    HRD: [1,7,14,21,28]. GLOBEM: [4,28]; bin 1 there means 28 days, not one week.
    This shared wearable readout is OUR ADAPTER, not the original forecasting readout.
    """
    if seq_len % int(bins_per_day):
        raise ValueError("spectral readout requires a whole number of days")
    days = seq_len // int(bins_per_day)
    bins = ([days // 7] if days % 7 == 0 else []) + [k * days for k in range(1, harmonics + 1)]
    return sorted({i for i in bins if 0 < i < seq_len / 2})


def band_keep(net):
    """Flat (bin, dim) indices of the band-matched readout, or None to keep every column.

    Under `band_readout` each band's dims are read only at the harmonics inside that band.
    Intended for 12 harmonics, where the other columns are ~92% of the readout; at 4 it
    lowers RQ3. Ordered bin-major, as the readout flattens.
    """
    if not getattr(net, "band_readout", False):
        return None
    d = net.seasonal_dims
    widths = [layer.out_channels for layer in net.sfd]
    starts = np.cumsum([0] + widths[:-1])
    idx = [i * d + int(s) + j
           for i, f in enumerate(spectral_freqs(net.seq_len, net.bins_per_day, net.harmonics))
           for (lo, hi), s, w in zip(net.bands, starts, widths) if lo <= f < hi
           for j in range(w)]
    return torch.as_tensor(idx, dtype=torch.long)


def spectral_readout(z, bins_per_day, phase_readout, harmonics=4, keep=None, eps=1e-3,
                     readout_norm="none"):
    """Seasonal branch -> (amplitude, phase) at the readout bins (spectral_freqs).

    rFFT over time, then amplitude and either the raw angle ('angle') or its (cos, sin)
    ('circular'). `keep` restricts both blocks to the band-matched columns. This is a
    frozen-feature READOUT choice, made only here and in CoSTModel._log_readout_amp; the
    training loss (seasonal_loss) always L2-normalises independently of it, so `readout_norm`
    changes what a trained encoder's representation looks like, never how it was trained.

    `eps` stabilises atan2 near a true amplitude of 0, where it is ill-conditioned: a bin with
    Real, Imag both near 0 has an angle that swings wildly for a tiny change in either. Under
    "none" many bins genuinely have amplitude at this scale (unlike "timestep", whose
    normalisation happened to keep every bin away from true 0), so ordinary float32
    non-associativity -- encoding the same window alone vs. in a larger batch shifted a bin's
    seasonal output by 7.45e-08, negligible everywhere else -- swung that bin's angle by up to
    0.65 rad when amplitude was close to eps (measured with the previous eps=1e-6, itself
    already at the amplitude floor of the affected bins: HRD smoke run, job train_encoder's
    save/reload check). eps=1e-3 is ~10,000x the measured noise floor (~1e-7) and, added inside
    the sqrt/atan2, negligible against any real signal (>=0.01 in every measurement so far);
    a near-zero bin now reports a stable, uninformative angle of pi/4 instead of noise.

    DEFAULT "none": use the raw seasonal sequence. "timestep" L2-normalises each timestep over
    channels first, as upstream CoST's seasonal loss does. THAT DESTROYS AMPLITUDE. DSSL's
    harmonic bands start at bin 1, so the seasonal sequence has no constant component to anchor
    the norm and it saturates toward a square wave: measured on an untrained HRD encoder,
    scaling the true 24 h amplitude by 0.5 / 1.0 / 1.5 moved the 24 h amplitude feature only
    8.06 / 12.20 / 14.09 with "timestep", and RQ2's amplitude arm inverted (0.404 at a=0.5,
    chance 0.5); with "none" the same features move 1.34 / 2.67 / 3.99 -- proportional -- and
    the arm scores 0.663. This is not free: on the same trained weights, switching only the
    readout from "timestep" to "none" also collapsed HRD Steps' own-branch acrophase R2 from
    0.765 to 0.0 (job 3183155, arm a2_amplitude) -- an unresolved side effect of this choice,
    unrelated to training.
    """
    f = spectral_freqs(z.size(1), bins_per_day, harmonics)
    if readout_norm not in ("timestep", "none"):
        raise ValueError(f"readout_norm must be 'timestep' or 'none', got {readout_norm!r}")
    seasonal = F.normalize(z.float(), dim=-1) if readout_norm == "timestep" else z.float()
    Z = fft.rfft(seasonal, dim=1)[:, f]
    amp = torch.sqrt((Z.real + eps).pow(2) + (Z.imag + eps).pow(2))
    ang = torch.atan2(Z.imag, Z.real + eps)
    pha = (torch.cos(ang), torch.sin(ang)) if phase_readout == "circular" else (ang,)

    def flat(p):
        p = p.reshape(p.size(0), -1)
        return p if keep is None else p[:, keep.to(p.device)]

    return flat(amp), torch.cat([flat(p) for p in pha], dim=-1)


# --------------------------------------------------------------------------------------
# Objective
# --------------------------------------------------------------------------------------
class CoSTModel(nn.Module):
    """MoCo on the trend branch, within-batch instance discrimination on the seasonal one."""

    def __init__(self, encoder_q, encoder_k, *, dim=128, alpha=0.005, K=4096, m=0.999,
                 T=0.07, phase_mode="circular_amp", readout_norm="none",
                 weights: O.TermWeights = O.PAPER, trend_views="same",
                 w_eq=0.0, eq_dims=0, w_ac=0.0, ac_gamma=0.7114, ac_queue=512, device="cuda"):
        super().__init__()
        if trend_views not in ("same", "disjoint_days"):
            raise ValueError(f"trend_views must be 'same' or 'disjoint_days', got {trend_views!r}")
        self.alpha, self.K, self.m, self.T = alpha, K, m, T
        self.phase_mode, self.readout_norm = phase_mode, readout_norm
        self.weights, self.device = weights, device
        self.trend_views = trend_views

        self.encoder_q, self.encoder_k = encoder_q, encoder_k
        self.head_q = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.head_k = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        for a, b in ((self.encoder_q, self.encoder_k), (self.head_q, self.head_k)):
            for pq, pk in zip(a.parameters(), b.parameters()):
                pk.data.copy_(pq.data)
                pk.requires_grad = False

        self.register_buffer("queue", F.normalize(torch.randn(dim, K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        if trend_views == "disjoint_days":
            # Which training window each key came from (-1: unknown), so a query's own week
            # is never its negative. Only in this mode, so "same" checkpoints are unchanged.
            self.register_buffer("queue_ids", torch.full((K,), -1, dtype=torch.long))
        self.last_top1 = float("nan")

        # Level equivariance. Built after the queue, so w_eq=0 draws no extra random numbers.
        # Bias-free: it maps a difference of readouts to a difference of offsets.
        self.w_eq = w_eq
        self.head_eq = nn.Linear(dim, eq_dims, bias=False) if w_eq > 0 else None
        # Anti-collapse on the readout's log amplitudes. `_ac_mem` holds detached rows of
        # recent batches: one batch cannot estimate a full-rank covariance of 160 channels.
        self.w_ac, self.ac_gamma, self.ac_queue = w_ac, ac_gamma, int(ac_queue)
        self._ac_mem = None

    def _log_readout_amp(self, z):
        """Log amplitude at every readout bin, B x bins x channels."""
        e = self.encoder_q
        amp, _ = spectral_readout(z, e.bins_per_day, "angle", e.harmonics, readout_norm=self.readout_norm)
        return amp.view(amp.size(0), -1, e.seasonal_dims).log()

    @torch.no_grad()
    def _momentum_update(self):
        for a, b in ((self.encoder_q, self.encoder_k), (self.head_q, self.head_k)):
            for pq, pk in zip(a.parameters(), b.parameters()):
                pk.data.mul_(self.m).add_(pq.data, alpha=1.0 - self.m)

    @torch.no_grad()
    def _enqueue(self, keys, ids=None):
        b = keys.shape[0]
        if self.K % b != 0:
            raise ValueError(f"queue size {self.K} must be divisible by batch size {b}")
        ptr = int(self.queue_ptr)
        self.queue[:, ptr:ptr + b] = keys.T
        if hasattr(self, "queue_ids"):
            self.queue_ids[ptr:ptr + b] = -1 if ids is None else ids
        self.queue_ptr[0] = (ptr + b) % self.K

    def _disjoint_days(self, x_q, x_k):
        """Hide all but a random half of the days in each view, and a different half in each.

        The two views of a week then share no timestep: the pair can only be matched through
        what persists across days, not by copying the input (raw pixels alone retrieve the
        full-window pair at top-1 0.891 among 3,803 HRD weeks). Hidden bins are NaN, which the
        encoder already zeroes. Returns the views and their visible-bin masks.
        """
        bpd = self.encoder_q.bins_per_day
        b, t, _ = x_q.shape
        days = t // bpd
        if days < 2:
            raise ValueError("disjoint_days needs at least two whole days per window")
        half = days // 2
        order = torch.argsort(torch.rand(b, days, device=x_q.device), dim=1)
        visible = []
        for chosen in (order[:, :half], order[:, half:2 * half]):
            day = torch.zeros(b, days, dtype=torch.bool, device=x_q.device).scatter_(1, chosen, True)
            per_bin = day.repeat_interleave(bpd, dim=1)
            visible.append(F.pad(per_bin, (0, t - days * bpd), value=False))
        views = [x.masked_fill(~v.unsqueeze(-1), float("nan")) for x, v in zip((x_q, x_k), visible)]
        return views, visible

    @staticmethod
    def _visible_mean(z, visible):
        w = visible.unsqueeze(-1).to(z.dtype)
        return (z * w).sum(dim=1) / w.sum(dim=1)

    def forward(self, x_q, x_k, update=True, return_parts=False, delta=None, ids=None):
        idx = np.random.randint(0, x_q.shape[1])            # the timestep the trend term contrasts
        q_t, q_s = self.encoder_q(x_q)

        q_level = q_t.mean(dim=1) if self.head_eq is not None else None   # the trend readout
        if self.trend_views == "same":
            q_t = F.normalize(self.head_q(q_t[:, idx]), dim=-1)
            with torch.no_grad():
                if update:
                    self._momentum_update()
                k_t, _ = self.encoder_k(x_k)
                k_t = F.normalize(self.head_k(k_t[:, idx]), dim=-1)
            trend, self.last_top1 = O.moco_ce_loss(q_t, k_t, self.queue.clone().detach(), self.T)
            if update:
                self._enqueue(k_t)
        else:
            # Same week, different days: the query is days A of view 1, the key days B of
            # view 2, each read out as the mean trend over its visible bins -- the quantity the
            # frozen trend block reports. One extra encoder pass, for the query.
            (x_a, x_b), (vis_a, vis_b) = self._disjoint_days(x_q, x_k)
            q_a, _ = self.encoder_q(x_a)
            q_h = F.normalize(self.head_q(self._visible_mean(q_a, vis_a)), dim=-1)
            with torch.no_grad():
                if update:
                    self._momentum_update()
                k_b, _ = self.encoder_k(x_b)
                k_h = F.normalize(self.head_k(self._visible_mean(k_b, vis_b)), dim=-1)
            own = None if ids is None else ids.unsqueeze(1) == self.queue_ids.unsqueeze(0)
            trend, self.last_top1 = O.moco_ce_loss(q_h, k_h, self.queue.clone().detach(),
                                                   self.T, neg_mask=own)
            if update:
                self._enqueue(k_h, ids)

        # Seasonal term as upstream: no queue and symmetric, so the key view also goes through
        # encoder_q with gradients (a third encoder pass).
        k_tq, k_s = self.encoder_q(x_k)
        amp, pha = O.seasonal_loss(q_s, k_s, self.weights, self.phase_mode)
        total = O.total_loss(trend, amp, pha, self.weights, self.alpha)
        if self.head_eq is not None:
            if delta is None:
                raise ValueError("w_eq > 0 but the batch carries no level offsets")
            total = total + self.w_eq * O.equivariance_loss(
                self.head_eq(q_level - k_tq.mean(dim=1)), delta)
        if self.w_ac > 0:
            # The queue holds training-mode rows (dropout on), so validation uses its own rows.
            cur = torch.cat([self._log_readout_amp(q_s), self._log_readout_amp(k_s)], dim=0)
            total = total + self.w_ac * O.anticollapse_loss(
                cur, self._ac_mem if self.training else None, self.ac_gamma)
            if update:
                mem = cur.detach() if self._ac_mem is None else torch.cat([cur.detach(), self._ac_mem])
                self._ac_mem = mem[:self.ac_queue]
        return (total, trend, amp + pha) if return_parts else total


def adjust_learning_rate(optimizer, lr, step, total):
    """Half-cycle cosine decay, as upstream."""
    cur = lr * 0.5 * (1.0 + math.cos(math.pi * step / max(total, 1)))
    for g in optimizer.param_groups:
        g["lr"] = cur
    return cur


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
WEIGHTS = {"paper": O.PAPER, "contracted": O.CONTRACTED}
# The only settings the CoST reference adapter takes from the experiment configuration: the
# shared readout, geometry and budget. Everything else is fixed to upstream CoST below.
REFERENCE_SHARED = ("output_dims", "hidden_dims", "tcn_depth", "harmonics", "seasonal_frac",
                    "phase_readout", "readout_norm", "alpha", "moco_k", "lr", "batch_size",
                    "mask_mode")


class DSSL:
    """Fit the encoder, then read frozen representations out of it.

    The defaults are the paper's DSSL: four harmonic bands, causal trend experts up to T/8,
    contracted seasonal weights, amplitude-weighted circular phase contrast, smoothing up to
    75 min, angle readout -- except level equivariance, now off (w_eq = 0): on HRD, w_eq
    1 / 0.05 / 0 gave RQ1 gain -0.0343 / -0.0136 / +0.0096 over 5 folds, ordered so in each. method="cost_reference"
    is upstream CoST behind the same readout: one full-spectrum band, trend experts up to
    T/2, raw-phase contrast, weights 1 / 0.5 / 0.5, no equivariance, scaling, jitter and
    shift augmentation at 0.5; callers pass it only REFERENCE_SHARED.
    """

    def __init__(self, input_dims, seq_len, bins_per_day, *, method="dssl",
                 output_dims=320, hidden_dims=64, n_time_features=0,
                 backbone="tcn", temporal_encoding="none", tcn_depth=None, n_layers=4,
                 n_heads=4, bidirectional=True, seasonal_bands="harmonics", harmonics=4,
                 trend_kernel_cap=None, seasonal_frac=0.5, band_readout=False, mask_mode="none",
                 phase_readout="angle", readout_norm="none", phase_mode="circular_amp",
                 weights="contracted", trend_views="same",
                 alpha=0.005, w_eq=0.0, w_ac=0.0, ac_gamma=0.7114, ac_queue=512, moco_k=4096,
                 jitter_sigma=0.1, shift_sigma=0.5, smooth_minutes=75.0, lr=5e-4, batch_size=64,
                 device="cuda", model_seed=None):
        if method not in ("dssl", "cost_reference"):
            raise ValueError("method must be dssl or cost_reference")
        self.method = method
        self.scale_sigma = 0.0
        if method == "cost_reference":
            if n_time_features:
                raise ValueError("the CoST reference adapter is sensor-only")
            backbone, temporal_encoding, seasonal_bands, band_readout = "tcn", "none", "single", False
            trend_kernel_cap, phase_mode, weights = max(1, seq_len // 2), "raw", "paper"
            trend_views = "same"
            w_eq, w_ac, jitter_sigma, shift_sigma, smooth_minutes = 0.0, 0.0, 0.5, 0.5, 0.0
            self.scale_sigma = 0.5
        if trend_kernel_cap is None:
            trend_kernel_cap = max(1, seq_len // 8)
        if isinstance(weights, str):
            if weights not in WEIGHTS:
                raise ValueError(f"weights must be one of {sorted(WEIGHTS)}, got {weights!r}")
            weights_name, weights = weights, WEIGHTS[weights]
        else:
            weights_name = next((k for k, v in WEIGHTS.items() if v == weights), "custom")
        if phase_readout not in ("angle", "circular"):
            raise ValueError(f"phase_readout must be 'angle' or 'circular', got {phase_readout!r}")
        if model_seed is not None:
            torch.manual_seed(model_seed)
            np.random.seed(model_seed % (2 ** 31))
            random.seed(model_seed)

        self.device = device
        self.seq_len, self.bins_per_day = seq_len, bins_per_day
        self.phase_readout, self.readout_norm = phase_readout, readout_norm
        self.batch_size, self.lr = batch_size, lr
        self.jitter_sigma, self.shift_sigma = jitter_sigma, shift_sigma
        self.smooth_bins = smooth_bins_for(smooth_minutes, bins_per_day)

        enc = dict(input_dims=input_dims, output_dims=output_dims, seq_len=seq_len,
                   bins_per_day=bins_per_day, hidden_dims=hidden_dims,
                   n_time_features=n_time_features, seasonal_bands=seasonal_bands,
                   max_harmonics=harmonics, mask_mode=mask_mode,
                   trend_kernel_cap=trend_kernel_cap, seasonal_frac=seasonal_frac,
                   band_readout=band_readout, backbone=backbone, temporal_encoding=temporal_encoding,
                   tcn_depth=tcn_depth, n_layers=n_layers, n_heads=n_heads,
                   bidirectional=bidirectional)
        self.net = CoSTEncoder(**enc).to(device)
        self._keep = band_keep(self.net)
        self.component_dims = self.net.seasonal_dims

        encoder_k = CoSTEncoder(**enc).to(device)
        self.cost = CoSTModel(
            self.net, encoder_k, dim=self.net.trend_dims, alpha=alpha, K=moco_k,
            phase_mode=phase_mode, readout_norm=readout_norm, weights=weights,
            trend_views=trend_views, w_eq=w_eq, eq_dims=input_dims, w_ac=w_ac,
            ac_gamma=ac_gamma, ac_queue=ac_queue, device=device).to(device)
        self.n_iters = 0
        self._optimizer = None
        self._permutation = None
        self._cursor = 0
        self._training_hash = None
        self._horizon = None
        self.history = {"iters": [], "train": [], "val": [], "top1": []}
        self.config = dict(method=method, input_dims=input_dims, seq_len=seq_len,
                           bins_per_day=bins_per_day, output_dims=output_dims,
                           hidden_dims=hidden_dims, n_time_features=n_time_features,
                           backbone=backbone, temporal_encoding=temporal_encoding,
                           tcn_depth=self.net.depth, receptive_field=self.net.receptive_field,
                           n_layers=n_layers, n_heads=n_heads, bidirectional=bidirectional,
                           seasonal_bands=seasonal_bands, bands=[list(b) for b in self.net.bands],
                           harmonics=harmonics, trend_kernel_cap=trend_kernel_cap,
                           trend_kernels=self.net.kernels, seasonal_frac=seasonal_frac,
                           band_readout=band_readout, mask_mode=mask_mode,
                           phase_readout=phase_readout, readout_norm=readout_norm,
                           phase_mode=phase_mode,
                           weights=weights_name, trend_views=trend_views,
                           alpha=alpha, w_eq=w_eq, w_ac=w_ac,
                           moco_k=moco_k, jitter_sigma=jitter_sigma, shift_sigma=shift_sigma,
                           scale_sigma=self.scale_sigma, smooth_minutes=smooth_minutes,
                           smooth_bins=self.smooth_bins, lr=lr, batch_size=batch_size,
                           model_seed=model_seed)

    # -- training ---------------------------------------------------------------------
    @tf32_convolutions()
    def fit(self, train_data, n_iters=1000, val_data=None, log_every=100, verbose=True,
            checkpoint_path=None, checkpoint_every=200, stop_after=None):
        """Pretrain. `train_data` is (N, T, D) float32; no labels are used."""
        array = np.ascontiguousarray(train_data, dtype=np.float32)
        if array.ndim != 3 or not np.isfinite(array).all():
            raise ValueError("SSL requires finite (windows,time,features) training input")
        digest = hashlib.sha256(array.tobytes()).hexdigest()
        if self._training_hash not in (None, digest) or self._horizon not in (None, n_iters):
            raise ValueError("resume requires identical training data/order and planned iteration horizon")
        self._training_hash, self._horizon = digest, n_iters
        ds = PretrainDataset(torch.from_numpy(array), jitter_sigma=self.jitter_sigma,
                             shift_sigma=self.shift_sigma, smooth_bins=self.smooth_bins,
                             scale_sigma=self.scale_sigma)
        if len(ds) < self.batch_size:
            raise ValueError(f"{len(ds)} windows is fewer than one batch "
                             f"({self.batch_size}); lower --batch-size or widen the fold")

        val_loader = None
        if val_data is not None and len(val_data) >= self.batch_size:
            val_loader = self._loader(val_data, shuffle=False, multiplier=1)

        params = [p for p in self.cost.parameters() if p.requires_grad]
        if self._optimizer is None:
            self._optimizer = torch.optim.SGD(params, lr=self.lr, momentum=0.9, weight_decay=1e-4)
        opt, hist = self._optimizer, self.history
        self.cost.train()
        finish = n_iters if stop_after is None else min(n_iters, self.n_iters + stop_after)
        while self.n_iters < finish:
            if self._permutation is None or self._cursor + self.batch_size > len(ds):
                self._permutation, self._cursor = torch.randperm(len(ds)), 0
            idx = self._permutation[self._cursor:self._cursor + self.batch_size]
            self._cursor += self.batch_size
            batch = [torch.stack(v) for v in zip(*(ds[int(i)] for i in idx))]
            adjust_learning_rate(opt, self.lr, self.n_iters, n_iters)
            loss = self._loss(batch, ids=(idx % ds.N).to(self.device))
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite SSL loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
                raise FloatingPointError("nonfinite SSL gradient")
            opt.step()
            self.n_iters += 1
            if self.n_iters % log_every == 0 or self.n_iters == n_iters:
                v = self._validation_loss(val_loader) if val_loader else None
                hist["iters"].append(self.n_iters)
                hist["train"].append(float(loss.item()))
                hist["val"].append(v)
                hist["top1"].append(self.cost.last_top1)
                if verbose:
                    print(f"    iter {self.n_iters:>6}  loss {loss.item():.4f}  "
                          f"val {v}  top1 {self.cost.last_top1:.4f}", flush=True)
            if checkpoint_path and (self.n_iters % checkpoint_every == 0 or self.n_iters == finish):
                self.save_training(checkpoint_path)
        self.cost.eval()
        return hist

    def _loader(self, data, *, shuffle, multiplier=None):
        kw = {} if multiplier is None else {"multiplier": multiplier}
        ds = PretrainDataset(torch.as_tensor(data, dtype=torch.float),
                             jitter_sigma=self.jitter_sigma, shift_sigma=self.shift_sigma,
                             smooth_bins=self.smooth_bins, scale_sigma=self.scale_sigma, **kw)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, drop_last=True)

    def _loss(self, batch, update=True, ids=None):
        """`ids`: each sample's training-window index; only trend_views="disjoint_days" reads
        it, to keep a week's own earlier keys out of its negatives."""
        x_q, x_k = batch[0].to(self.device), batch[1].to(self.device)
        return self.cost(x_q, x_k, update=update, delta=batch[2].to(self.device), ids=ids)

    @torch.no_grad()
    def _validation_loss(self, loader):
        """Held-out pretext loss; `update=False` so monitoring never mutates MoCo state."""
        self.cost.eval()
        tot = n = 0
        for batch in loader:
            tot += float(self._loss(batch, update=False))
            n += 1
        self.cost.train()
        return tot / max(n, 1)

    # -- readout ----------------------------------------------------------------------
    def _spectral(self, z):
        return spectral_readout(z, self.bins_per_day, self.phase_readout, self.net.harmonics,
                                self._keep, readout_norm=self.readout_norm)

    def blocks(self):
        """Column range (start, stop) of each branch in `encode` output: the trend readout,
        then the seasonal amplitude block, then the phase block (twice as wide if circular)."""
        t = self.net.trend_dims
        a = (len(self._keep) if self._keep is not None else
             len(spectral_freqs(self.seq_len, self.bins_per_day, self.net.harmonics)) * self.net.seasonal_dims)
        p = 2 * a if self.phase_readout == "circular" else a
        return {"trend": (0, t), "amplitude": (t, t + a), "phase": (t + a, t + a + p)}

    def pair_block(self):
        """(start, width) of the (cos, sin) phase columns in `encode` output, or None when the
        readout emits raw angles. Probes scale this block isotropically."""
        if self.phase_readout != "circular":
            return None
        start, stop = self.blocks()["phase"]
        return start, (stop - start) // 2

    @torch.no_grad()
    def encode(self, data, batch_size=256, pool="mean", parts=False):
        """Frozen representation [trend | amp | phase], one row per window. The trend branch
        is pooled over time (`pool`); the seasonal branch is read in the frequency domain,
        because its time mean is exactly zero. With `parts`, also returns each block."""
        self.net.eval()
        X = torch.as_tensor(data, dtype=torch.float)
        outs = {"trend": [], "amp": [], "phase": []}
        for i in range(0, len(X), batch_size):
            t, s = self.net(X[i:i + batch_size].to(self.device))
            outs["trend"].append(self._pool(t, pool).cpu())
            a, p = self._spectral(s)
            outs["amp"].append(a.cpu())
            outs["phase"].append(p.cpu())
        cat = {k: torch.cat(v).numpy() for k, v in outs.items()}
        full = np.concatenate([cat["trend"], cat["amp"], cat["phase"]], axis=-1)
        return {"full": full, **cat} if parts else full

    @staticmethod
    def _pool(z, how):
        if how == "mean":
            return z.mean(dim=1)
        if how == "last":
            return z[:, -1]
        if how == "max":
            return z.max(dim=1).values
        raise ValueError(f"unknown pool: {how!r}")

    # -- persistence ------------------------------------------------------------------
    def save(self, path):
        # The weights are readout-agnostic; the readout at construction is provenance only.
        torch.save({"net": self.net.state_dict(), "n_iters": self.n_iters,
                    "phase_readout_at_construction": self.phase_readout,
                    "config": self.config}, path)

    def load(self, path):
        """Load weights. The caller's `phase_readout` is kept: the readout is chosen at
        encode time and never comes from the checkpoint."""
        ck = torch.load(path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(ck["net"])
        self.n_iters = ck.get("n_iters", 0)
        return self

    def save_training(self, path):
        """Atomic full-state checkpoint, including the exact shuffled sampling position."""
        if self._optimizer is None:
            raise ValueError("no training state to save")
        state = np.random.get_state()
        payload = {"config": self.config, "cost": self.cost.state_dict(),
                   "optimizer": self._optimizer.state_dict(), "n_iters": self.n_iters,
                   "history": self.history, "permutation": self._permutation, "cursor": self._cursor,
                   "training_hash": self._training_hash, "horizon": self._horizon,
                   "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                   "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                   "numpy_rng": (state[0], state[1].tolist(), state[2], state[3], state[4])}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".tmp")
        torch.save(payload, temp)
        temp.replace(path)

    def load_training(self, path):
        """Resume the same model/configuration/data and planned learning-rate horizon."""
        state = torch.load(path, map_location=self.device, weights_only=True)
        if state["config"] != self.config:
            raise ValueError("training checkpoint configuration differs from the requested model")
        self.cost.load_state_dict(state["cost"])
        self._optimizer = torch.optim.SGD([p for p in self.cost.parameters() if p.requires_grad],
                                           lr=self.lr, momentum=.9, weight_decay=1e-4)
        self._optimizer.load_state_dict(state["optimizer"])
        self.n_iters, self.history = state["n_iters"], state["history"]
        self._permutation, self._cursor = state["permutation"].cpu(), state["cursor"]
        self._training_hash, self._horizon = state["training_hash"], state["horizon"]
        random.setstate(state["python_rng"])
        ns = state["numpy_rng"]
        np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
        torch.set_rng_state(state["torch_rng"].cpu())
        if state["cuda_rng"]:
            if not torch.cuda.is_available():
                raise ValueError("CUDA training state requires CUDA to reproduce the run")
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
        return self
