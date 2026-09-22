"""CoST pretraining and the frozen spectral readout.

MoCo on the trend branch, within-batch instance contrast of amplitude and phase on the
seasonal branch, as upstream. Derived from salesforce/CoST (BSD-3), which vendors TS2Vec
(MIT); see NOTICE. Differences from upstream: harmonic bands (models.encoder.rhythm_bands), a
smoothing augmentation in place of scaling, an amplitude-weighted circular phase contrast
(models.losses.seasonal_loss), and a readout of amplitude and phase at the daily harmonics,
read from the raw seasonal sequence.
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
           "band_support",
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
    """Two independently augmented views of the same window.

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
        x = self.data[item % self.N]
        return self.transform(x), self.transform(x)

    def transform(self, x):
        if self.scale_sigma and random.random() <= self.p:
            x = x * (1 + torch.randn(x.size(-1)) * self.scale_sigma)
        return self.jitter(self.shift(self.smooth(x)))

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
        """Constant per-channel offset: changes only the f=0 bin (the MESOR)."""
        if random.random() > self.p:
            return x
        return x + torch.randn(x.size(-1)) * self.shift_sigma


class LabelledDataset(PretrainDataset):
    """One augmented view of a labelled window and its label, for the supervised control:
    the same augmentations as pretraining, so only the objective and the data differ."""

    def __init__(self, data, labels, **kwargs):
        super().__init__(data, **kwargs)
        self.labels = torch.as_tensor(labels, dtype=torch.float)

    def __getitem__(self, item):
        i = item % self.N
        return self.transform(self.data[i]), self.labels[i]


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


def band_support(net):
    """Which (readout bin, seasonal channel) pairs the band structure can actually support.

    A `BandedFourierLayer` zeroes every frequency outside its own band before the inverse
    transform, so in the frequency domain its output is exactly zero at any bin outside that
    band. The readout, however, reads EVERY seasonal channel at EVERY bin in `spectral_freqs`.
    A (bin, channel) pair is therefore "supported" only when the band owning that channel
    contains that bin; every other pair is structurally zero and carries no signal.

    Returns (mask, report) where `mask` is a (n_bins, seasonal_dims) boolean array and
    `report` is a JSON-serialisable summary recorded in `DSSL.config`, so that every run
    states on the record how much of its own readout is structurally empty.

    The mask is built only from the band edges and the channel allocation. It never looks at
    data, labels or any score.
    """
    freqs = spectral_freqs(net.seq_len, net.bins_per_day, net.harmonics)
    widths = [layer.out_channels for layer in net.sfd]
    starts = np.cumsum([0] + widths[:-1]).tolist()
    mask = np.zeros((len(freqs), net.seasonal_dims), dtype=bool)
    for i, f in enumerate(freqs):
        for (lo, hi), s, w in zip(net.bands, starts, widths):
            if lo <= f < hi:
                mask[i, s:s + w] = True
    # A band read at no readout bin can never contribute a supported coordinate. On GLOBEM
    # the second harmonic lands exactly on the Nyquist bin, which `spectral_freqs` excludes
    # because the rFFT coefficient there is real and its phase is degenerate; the band is
    # still built, so its channels reach the objective but never the readout.
    unread = [list(b) for b in net.bands if not any(b[0] <= f < b[1] for f in freqs)]
    report = dict(readout_bins=list(freqs), band_widths=widths,
                  supported=int(mask.sum()), total=int(mask.size),
                  unsupported=int(mask.size - mask.sum()),
                  unsupported_fraction=round(float(1 - mask.mean()), 4),
                  bands_never_read=unread)
    return mask, report


def band_keep(net):
    """Flat (bin, dim) indices of the band-matched readout, or None to keep every column.

    Under `band_readout` each band's dims are read only at the harmonics inside that band,
    i.e. exactly the supported pairs of `band_support`. Ordered bin-major, as the readout
    flattens, so column order is a deterministic function of the geometry alone.

    Off by default: the full readout is what every reported result used. Restricting to the
    supported pairs is a change of representation, not a bug fix applied in place, so it is
    kept as a separate configuration to be compared under the protocol rather than switched
    on silently. Prior evidence is mixed -- at 12 harmonics the unsupported columns are ~92%
    of the readout and removing them helps, while at the canonical 4 harmonics an earlier
    screen lowered RQ3.
    """
    if not getattr(net, "band_readout", False):
        return None
    mask, _ = band_support(net)
    return torch.as_tensor(np.flatnonzero(mask.reshape(-1)), dtype=torch.long)


def spectral_readout(z, bins_per_day, phase_readout, harmonics=4, keep=None, eps=1e-3):
    """Seasonal branch -> (amplitude, phase) at the readout bins (spectral_freqs).

    rFFT over time, then amplitude and either the raw angle ('angle') or its (cos, sin)
    ('circular'). `keep` restricts both blocks to the band-matched columns. This is a
    frozen-feature READOUT choice. The training loss (seasonal_loss) normalises independently of
    it, so this affects what a trained encoder's representation looks like, never how it trained.

    `eps` stabilises atan2 near a true amplitude of 0, where it is ill-conditioned: a bin with
    Real, Imag both near 0 has an angle that swings wildly for a tiny change in either. Reading
    the sequence raw leaves many bins genuinely at this scale (the rejected per-timestep
    normalisation happened to keep every bin away from true 0), so ordinary float32
    non-associativity -- encoding the same window alone vs. in a larger batch shifted a bin's
    seasonal output by 7.45e-08, negligible everywhere else -- swung that bin's angle by up to
    0.65 rad when amplitude was close to eps (measured with the previous eps=1e-6, itself
    already at the amplitude floor of the affected bins: HRD smoke run, job train_encoder's
    save/reload check). eps=1e-3 is ~10,000x the measured noise floor (~1e-7) and, added inside
    the sqrt/atan2, negligible against any real signal (>=0.01 in every measurement so far);
    a near-zero bin now reports a stable, uninformative angle of pi/4 instead of noise.

    The seasonal sequence is read RAW. Upstream CoST L2-normalises each timestep across channels
    first, and that destroys amplitude: DSSL's harmonic bands start at bin 1, so the sequence has
    no constant component to anchor the norm and it saturates toward a square wave. Measured on an
    untrained HRD encoder, scaling a window's true 24 h amplitude by 0.5 / 1.0 / 1.5 moved the
    amplitude feature to 8.06 / 12.20 / 14.09 under normalisation (compressive) versus
    1.34 / 2.67 / 3.99 without it (proportional). At protocol grade the raw readout moved RQ2's
    strength arm from 0.438, below the 0.5 chance level, to 0.713. Two alternatives (normalised,
    and a split taking amplitude raw and phase normalised) were evaluated at protocol grade and
    rejected; see FAILED_EXPERIMENTS.md section 6.
    """
    f = spectral_freqs(z.size(1), bins_per_day, harmonics)
    Z = fft.rfft(z.float(), dim=1)[:, f]
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
                 T=0.07, phase_mode="circular_amp", weights: O.TermWeights = O.PAPER,
                 device="cuda"):
        super().__init__()
        self.alpha, self.K, self.m, self.T = alpha, K, m, T
        self.phase_mode, self.weights, self.device = phase_mode, weights, device

        self.encoder_q, self.encoder_k = encoder_q, encoder_k
        self.head_q = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.head_k = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        for a, b in ((self.encoder_q, self.encoder_k), (self.head_q, self.head_k)):
            for pq, pk in zip(a.parameters(), b.parameters()):
                pk.data.copy_(pq.data)
                pk.requires_grad = False

        self.register_buffer("queue", F.normalize(torch.randn(dim, K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.last_top1 = float("nan")


    @torch.no_grad()
    def _momentum_update(self):
        for a, b in ((self.encoder_q, self.encoder_k), (self.head_q, self.head_k)):
            for pq, pk in zip(a.parameters(), b.parameters()):
                pk.data.mul_(self.m).add_(pq.data, alpha=1.0 - self.m)

    @torch.no_grad()
    def _enqueue(self, keys):
        b = keys.shape[0]
        if self.K % b != 0:
            raise ValueError(f"queue size {self.K} must be divisible by batch size {b}")
        ptr = int(self.queue_ptr)
        self.queue[:, ptr:ptr + b] = keys.T
        self.queue_ptr[0] = (ptr + b) % self.K

    def forward(self, x_q, x_k, update=True, return_parts=False):
        idx = np.random.randint(0, x_q.shape[1])            # the timestep the trend term contrasts
        q_t, q_s = self.encoder_q(x_q)
        q_t = F.normalize(self.head_q(q_t[:, idx]), dim=-1)
        with torch.no_grad():
            if update:
                self._momentum_update()
            k_t, _ = self.encoder_k(x_k)
            k_t = F.normalize(self.head_k(k_t[:, idx]), dim=-1)
        trend, self.last_top1 = O.moco_ce_loss(q_t, k_t, self.queue.clone().detach(), self.T)
        if update:
            self._enqueue(k_t)

        # Seasonal term as upstream: no queue and symmetric, so the key view also goes through
        # encoder_q with gradients (a third encoder pass).
        _, k_s = self.encoder_q(x_k)
        amp, pha = O.seasonal_loss(q_s, k_s, self.weights, self.phase_mode)
        total = O.total_loss(trend, amp, pha, self.weights, self.alpha)
        return (total, trend, amp + pha) if return_parts else total


class SupervisedModel(nn.Module):
    """Supervised control: the DSSL encoder and readout, trained end to end on window labels.

    The head reads the representation that `DSSL.encode` returns (trend mean, amplitude and
    phase at the readout bins), with amplitude on a log scale and phase as (cos, sin), through a
    non-affine batch normalisation and one linear layer. The loss is class-balanced binary
    cross-entropy: `pos_weight` = negatives / positives of the training windows."""

    def __init__(self, encoder, dim, bins_per_day, harmonics, keep, readout="spectral"):
        super().__init__()
        self.encoder, self.bins_per_day, self.harmonics, self.keep = encoder, bins_per_day, harmonics, keep
        self.readout = readout
        self.norm = nn.BatchNorm1d(dim, affine=False)
        self.head = nn.Linear(dim, 1)
        self.register_buffer("pos_weight", torch.ones(1))
        self.last_top1 = float("nan")

    def logits(self, x):
        t, s = self.encoder(x)
        if self.readout == "pooled":        # the representation DSSL.encode returns
            f = torch.cat([t.max(dim=1).values, s.max(dim=1).values], dim=-1)
            return self.head(self.norm(f)).squeeze(-1)
        amp, ang = spectral_readout(s, self.bins_per_day, "angle", self.harmonics, self.keep)
        f = torch.cat([t.mean(dim=1), torch.log(amp), torch.cos(ang), torch.sin(ang)], dim=-1)
        return self.head(self.norm(f)).squeeze(-1)

    def forward(self, x, y, update=True):
        z = self.logits(x)
        with torch.no_grad():
            self.last_top1 = float(((z > 0).float() == y).float().mean())   # training accuracy
        return F.binary_cross_entropy_with_logits(z, y, pos_weight=self.pos_weight)


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
                    "phase_readout", "readout", "alpha", "moco_k", "lr", "batch_size", "mask_mode")


class DSSL:
    """Fit the encoder, then read frozen representations out of it.

    The defaults are the canonical DSSL: four harmonic bands, causal trend experts up to T/8,
    contracted seasonal weights, amplitude-weighted circular phase contrast, smoothing up to
    75 min, and an angle readout of the raw seasonal sequence. Options tested and rejected
    (level equivariance, alternative readouts, disjoint-day trend pairing, pre-encoder
    decomposition) are recorded in FAILED_EXPERIMENTS.md and are no longer configurable. method="cost_reference"
    is upstream CoST behind the same readout: one full-spectrum band, trend experts up to
    T/2, raw-phase contrast, weights 1 / 0.5 / 0.5, no equivariance, scaling, jitter and
    shift augmentation at 0.5; callers pass it only REFERENCE_SHARED.
    """

    def __init__(self, input_dims, seq_len, bins_per_day, *, method="dssl",
                 output_dims=320, hidden_dims=64, n_time_features=0,
                 backbone="tcn", temporal_encoding="none", tcn_depth=None, n_layers=4,
                 n_heads=4, bidirectional=True, seasonal_bands="harmonics", harmonics=4,
                 trend_kernel_cap=None, seasonal_frac=0.5, band_readout=False, mask_mode="none",
                 phase_readout="angle", phase_mode="circular_amp",
                 weights="contracted", alpha=0.005, moco_k=4096,
                 jitter_sigma=0.1, shift_sigma=0.5, smooth_minutes=75.0, lr=5e-4, batch_size=64,
                 device="cuda", model_seed=None, objective="contrastive", readout="spectral",
                 scale_sigma=0.0):
        if method not in ("dssl", "cost_reference"):
            raise ValueError("method must be dssl or cost_reference")
        if objective not in ("contrastive", "supervised") or (objective == "supervised" and method != "dssl"):
            raise ValueError("objective must be contrastive, or supervised with method='dssl'")
        if readout not in ("spectral", "pooled"):
            raise ValueError(f"readout must be 'spectral' or 'pooled', got {readout!r}")
        self.method, self.objective, self.readout = method, objective, readout
        self.scale_sigma = float(scale_sigma)
        if method == "cost_reference":
            if n_time_features:
                raise ValueError("the CoST reference adapter is sensor-only")
            # Upstream CoST (salesforce/CoST train.py and cost.py): one full-spectrum band, trend
            # kernels 1..128, raw-phase contrast, loss 1 * trend + alpha * (amp + phase) / 2, and
            # jitter, scaling and shift augmentation, each sigma 0.5 with p 0.5.
            backbone, temporal_encoding, seasonal_bands, band_readout = "tcn", "none", "single", False
            trend_kernel_cap, phase_mode, weights = 128, "raw", "paper"
            jitter_sigma, shift_sigma, smooth_minutes = 0.5, 0.5, 0.0
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
        self.phase_readout = phase_readout
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

        if objective == "supervised":
            blocks = self.blocks()
            dim = (output_dims if readout == "pooled" else
                   blocks["trend"][1] + 3 * (blocks["amplitude"][1] - blocks["amplitude"][0]))
            self.cost = SupervisedModel(self.net, dim, bins_per_day, self.net.harmonics,
                                        self._keep, readout).to(device)
        else:
            encoder_k = CoSTEncoder(**enc).to(device)
            self.cost = CoSTModel(
                self.net, encoder_k, dim=self.net.trend_dims, alpha=alpha, K=moco_k,
                phase_mode=phase_mode, weights=weights, device=device).to(device)
        self.n_iters = 0
        self._optimizer = None
        self._permutation = None
        self._cursor = 0
        self._training_hash = None
        self._horizon = None
        self.history = {"iters": [], "train": [], "val": [], "top1": []}
        self.config = dict(method=method, objective=objective, input_dims=input_dims, seq_len=seq_len,
                           bins_per_day=bins_per_day, output_dims=output_dims,
                           hidden_dims=hidden_dims, n_time_features=n_time_features,
                           backbone=backbone, temporal_encoding=temporal_encoding,
                           tcn_depth=self.net.depth, receptive_field=self.net.receptive_field,
                           n_layers=n_layers, n_heads=n_heads, bidirectional=bidirectional,
                           seasonal_bands=seasonal_bands, bands=[list(b) for b in self.net.bands],
                           harmonics=harmonics, trend_kernel_cap=trend_kernel_cap,
                           trend_kernels=self.net.kernels, seasonal_frac=seasonal_frac,
                           band_readout=band_readout, mask_mode=mask_mode,
                           readout=readout, phase_readout=phase_readout, phase_mode=phase_mode,
                           weights=weights_name, alpha=alpha,
                           moco_k=moco_k, jitter_sigma=jitter_sigma, shift_sigma=shift_sigma,
                           scale_sigma=self.scale_sigma, smooth_minutes=smooth_minutes,
                           smooth_bins=self.smooth_bins, lr=lr, batch_size=batch_size,
                           model_seed=model_seed)

    # -- training ---------------------------------------------------------------------
    @tf32_convolutions()
    def fit(self, train_data, n_iters=1000, val_data=None, log_every=100, verbose=True,
            checkpoint_path=None, checkpoint_every=200, stop_after=None, labels=None):
        """Train. `train_data` is (N, T, D) float32. The contrastive objective uses no labels;
        the supervised control requires one 0/1 `labels` entry per window."""
        array = np.ascontiguousarray(train_data, dtype=np.float32)
        if array.ndim != 3 or not np.isfinite(array).all():
            raise ValueError("SSL requires finite (windows,time,features) training input")
        supervised = self.objective == "supervised"
        if supervised:
            labels = np.asarray(labels, dtype=np.float32)
            if labels.shape != (len(array),) or set(np.unique(labels)) != {0.0, 1.0}:
                raise ValueError("supervised training needs one 0/1 label per window, both classes")
            val_data = None
        elif labels is not None:
            raise ValueError("the contrastive objective takes no labels")
        digest = hashlib.sha256(array.tobytes() + (labels.tobytes() if supervised else b"")).hexdigest()
        if self._training_hash not in (None, digest) or self._horizon not in (None, n_iters):
            raise ValueError("resume requires identical training data/order and planned iteration horizon")
        self._training_hash, self._horizon = digest, n_iters
        aug = dict(jitter_sigma=self.jitter_sigma, shift_sigma=self.shift_sigma,
                   smooth_bins=self.smooth_bins, scale_sigma=self.scale_sigma)
        if supervised:
            ds = LabelledDataset(torch.from_numpy(array), labels, **aug)
            self.cost.pos_weight.fill_(float((labels == 0).sum() / (labels == 1).sum()))
        else:
            ds = PretrainDataset(torch.from_numpy(array), **aug)
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
            loss = self._loss(batch)
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

    def _loss(self, batch, update=True):
        x_q, x_k = batch[0].to(self.device), batch[1].to(self.device)
        return self.cost(x_q, x_k, update=update)

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
                                self._keep)

    def blocks(self):
        """Column range (start, stop) of each branch in `encode` output. Spectral readout: the
        trend readout, then the seasonal amplitude block, then the phase block (twice as wide if
        circular). Pooled readout: the trend branch, then the seasonal branch."""
        t = self.net.trend_dims
        if self.readout == "pooled":
            return {"trend": (0, t), "seasonal": (t, t + self.net.seasonal_dims)}
        a = (len(self._keep) if self._keep is not None else
             len(spectral_freqs(self.seq_len, self.bins_per_day, self.net.harmonics)) * self.net.seasonal_dims)
        p = 2 * a if self.phase_readout == "circular" else a
        return {"trend": (0, t), "amplitude": (t, t + a), "phase": (t + a, t + a + p)}

    def pair_block(self):
        """(start, width) of the (cos, sin) phase columns in `encode` output, or None when the
        readout emits raw angles. Probes scale this block isotropically."""
        if self.readout == "pooled" or self.phase_readout != "circular":
            return None
        start, stop = self.blocks()["phase"]
        return start, (stop - start) // 2

    @torch.no_grad()
    def predict_proba(self, data, batch_size=256):
        """Supervised control only: the head's probability for each window."""
        if self.objective != "supervised":
            raise ValueError("predict_proba needs the supervised objective")
        self.cost.eval()
        X = torch.as_tensor(data, dtype=torch.float)
        out = [torch.sigmoid(self.cost.logits(X[i:i + batch_size].to(self.device))).cpu()
               for i in range(0, len(X), batch_size)]
        return torch.cat(out).numpy()

    @torch.no_grad()
    def encode(self, data, batch_size=256, pool="mean", parts=False):
        """Frozen representation, one row per window. Spectral readout: [trend | amp | phase],
        the trend branch pooled over time (`pool`) and the seasonal branch read in the frequency
        domain, because its time mean is exactly zero. Pooled readout: [V^T | V^S] max-pooled over
        time, CoST's full-series representation (upstream `_eval_with_pooling`, max_pool1d).
        With `parts`, also returns each block."""
        self.net.eval()
        X = torch.as_tensor(data, dtype=torch.float)
        if self.readout == "pooled":
            outs = {"trend": [], "seasonal": []}
            for i in range(0, len(X), batch_size):
                t, s = self.net(X[i:i + batch_size].to(self.device))
                outs["trend"].append(t.max(dim=1).values.cpu())
                outs["seasonal"].append(s.max(dim=1).values.cpu())
            cat = {k: torch.cat(v).numpy() for k, v in outs.items()}
            full = np.concatenate([cat["trend"], cat["seasonal"]], axis=-1)
            return {"full": full, **cat} if parts else full
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
