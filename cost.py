"""MoCo pretraining for the disentangled seasonal/trend encoder, and the frozen readout.

Derived from salesforce/CoST (BSD-3), which vendors TS2Vec (MIT). See NOTICE.

WHAT WAS PORTED, AND WHAT WAS NOT. The 4-arm design varies `phase_readout` and the objective
weights and holds everything else fixed, so every alternative that all four arms would leave
switched off was dropped rather than carried. Each of these was ruled out by a measurement
already in archive_logs.txt, not by taste:

  positive='participant'   downstream ceiling 0.6658, BELOW the 0.7198 an untrained random
                           projection of the raw window already reaches. It is the only HARD
                           pairing implemented and its ceiling is still dominated.
  positive='day-disjoint'  ceiling 0.6574. Dominated by both of the above.
  decomposition views      the trend-sharing pair is solved at initialisation (top-1 0.980);
                           `decomp_aug` was False in both reference runs.
  subject negatives        the study concluded participant identity is NOT the shortcut, so
                           the mode has served its purpose; `negatives='global'` in both runs.
  V^N / noise branch       `noise_weight=0.0` in both reference runs. Dropped, not ported.
  GradNorm balancing       `loss_balance='fixed'` in both reference runs.
  n_exact_tail             existed to keep integer calendar channels bit-exact under
                           augmentation, and calendar PE is part 4.

The three augmentations survive because they were measured to matter. `scale` stays removed:
amplitude is the discriminative circadian feature, so contrasting scaled views would train
the model to ignore exactly the signal.
"""
from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import fft, nn
from torch.utils.data import DataLoader, Dataset

import objective as O
from model import CoSTEncoder

__all__ = ["PretrainDataset", "CoSTModel", "CoST", "WindowClassifier", "finetune",
           "predict_windows", "spectral_readout", "readout_width",
           "smooth_bins_for"]


# --------------------------------------------------------------------------------------
def smooth_bins_for(minutes, bins_per_day):
    """Widest smoothing box, in bins, for a width given in MINUTES.

    The augmentation is defined physically -- detail finer than `minutes` is declared noise --
    so its width in bins must follow the cohort's resolution. It used to be given in bins,
    where the same 5 meant 1.25 h on HRD (15-min bins) but 30 h on GLOBEM (6-h bins).
    Measured on a pure 24 h cosine through `PretrainDataset.smooth`: GLOBEM kept 33% of the
    daily amplitude at width 3 and 20%, phase-INVERTED, at width 5, against >= 99.6% and no
    phase change on HRD. Every GLOBEM contrastive run at train.py's old default (5 bins) was
    therefore trained to treat the daily rhythm itself as noise. Below 3 bins no odd box
    wider than one bin fits, and the augmentation switches off -- which is exactly GLOBEM's
    case at any sub-day width.
    """
    return int(minutes * bins_per_day // 1440)


class PretrainDataset(Dataset):
    """Two independently augmented views of the same window.

    `multiplier` re-visits each window that many times per epoch, so a small cohort still
    yields a full-length epoch of distinct augmentations.
    """

    def __init__(self, data, jitter_sigma=0.1, shift_sigma=0.5, p=0.5, multiplier=10,
                 smooth_bins=0, labels=None, groups=None):
        super().__init__()
        self.data = data
        self.p, self.multiplier = p, multiplier
        self.jitter_sigma, self.shift_sigma = jitter_sigma, shift_sigma
        self.smooth_bins = int(smooth_bins)
        self.N, self.T, self.D = data.shape
        # Per-window label and participant index, read only by the supervised term.
        # Unlabelled windows carry -1. Without labels every window is unlabelled and its own
        # group, so the supervised term sees nothing and the views are drawn exactly as before.
        self.labels = (torch.full((self.N,), -1, dtype=torch.long) if labels is None
                       else torch.as_tensor(labels, dtype=torch.long))
        self.groups = (torch.arange(self.N) if groups is None
                       else torch.as_tensor(groups, dtype=torch.long))

    def __len__(self):
        return self.N * self.multiplier

    def __getitem__(self, item):
        i = item % self.N
        x = self.data[i]
        return self.transform(x), self.transform(x), self.labels[i], self.groups[i]

    def transform(self, x):
        return self.jitter(self.shift(self.smooth(x)))

    def smooth(self, x):
        """Circular box filter of random odd width up to `smooth_bins` -- what declares
        detail finer than that width to be noise.

        `smooth_bins` must come from `smooth_bins_for`, never be set in bins directly: the
        same bin count is 1.25 h on HRD and 30 h on GLOBEM, where it erased the daily rhythm.

        In a contrastive objective the augmentation IS the definition of noise: whatever it
        destroys, the representation learns to ignore. Each candidate therefore carries a
        ceiling, the predictive content of what survives it, measured on HRD over 24 seeds
        through an identical projection and probe:

            sub-hour smoothing  0.6926   the only one ABOVE the raw window's 0.6884
            jitter              0.6884   removes nothing, so it defines no task at all
            per-channel offset  0.6835
            per-channel gain    0.6492
            day permutation     0.6303
            time roll           0.6273

        The filter is circular, so the window keeps its length and every rhythm keeps its
        phase exactly. Widths are ODD only: an even box has no centre bin, so its padding is
        asymmetric and it moves every phase by half a bin -- 0.0327 rad on the 24 h component
        at 96 bins/day, exactly half of 2*pi/96. That would make this a time shift wearing a
        smoother's clothes, and the ceiling for time shifts is 0.6273.
        """
        if self.smooth_bins < 3 or random.random() > self.p:
            return x
        w = random.randrange(3, (self.smooth_bins // 2) * 2 + 2, 2)
        xt = x.transpose(0, 1).unsqueeze(0)
        xp = F.pad(xt, ((w - 1) // 2, (w - 1) // 2), mode="circular")
        return F.avg_pool1d(xp, kernel_size=w, stride=1).squeeze(0).transpose(0, 1)

    def jitter(self, x):
        """Sensor-noise level (sigma ~0.1), not the 0.5 upstream uses on other domains."""
        if random.random() > self.p:
            return x
        return x + torch.randn(x.shape) * self.jitter_sigma

    def shift(self, x):
        """A constant per-channel offset moves only the MESOR -- the f=0 bin -- and leaves
        the amplitude and phase of every rhythm untouched. Safe, non-trivial, and it guards
        against contrastive collapse."""
        if random.random() > self.p:
            return x
        return x + torch.randn(x.size(-1)) * self.shift_sigma


# --------------------------------------------------------------------------------------
def spectral_readout(z, bins_per_day, phase_readout, eps=1e-6):
    """Frequency-domain readout of the seasonal branch -- (amplitude, phase) blocks.

    Shared by the frozen path and the fine-tuning head so both read the representation the
    SAME way. Every operation is differentiable, which is what lets a classification head
    train through it: rfft, and a sqrt whose argument is kept strictly positive by `eps` so
    its gradient stays finite at the origin.
    """
    T = z.size(1)
    D = max(1, T // int(bins_per_day))
    f = [i for i in (1, D, 2 * D, 3 * D, 4 * D) if 0 < i <= T // 2]
    Z = fft.rfft(F.normalize(z.float(), dim=-1), dim=1)[:, f]
    amp = torch.sqrt((Z.real + eps).pow(2) + (Z.imag + eps).pow(2))
    ang = torch.atan2(Z.imag, Z.real + eps)
    pha = (torch.cos(ang), torch.sin(ang)) if phase_readout == "circular" else (ang,)
    flat = lambda p: p.reshape(p.size(0), -1)
    return flat(amp), torch.cat([flat(p) for p in pha], dim=-1)


def readout_width(seq_len, bins_per_day, phase_readout, trend_dims, seasonal_dims,
                  residual_dims=0):
    """Width of [trend | amp | phase | resid] -- what the classification head receives.

    `residual_dims` defaults to 0, so every arm without the V^N branch keeps exactly the
    width it had before that branch existed.
    """
    D = max(1, seq_len // int(bins_per_day))
    nf = len([i for i in (1, D, 2 * D, 3 * D, 4 * D) if 0 < i <= seq_len // 2])
    return (trend_dims + nf * seasonal_dims * (1 + (2 if phase_readout == "circular" else 1))
            + residual_dims)


class CoSTModel(nn.Module):
    """MoCo on the trend branch, within-batch instance discrimination on the seasonal one."""

    def __init__(self, encoder_q, encoder_k, *, dim=128, alpha=0.005, K=4096, m=0.999,
                 T=0.07, disentangle=True, phase_mode="circular_amp", trend_pool="random",
                 weights: O.TermWeights = O.PAPER, w_supcon=0.0, supcon_temp=0.1,
                 supcon_dims=0, bins_per_day=None, device="cuda"):
        super().__init__()
        if trend_pool not in ("random", "mean"):
            raise ValueError(f"trend_pool must be 'random' or 'mean', got {trend_pool!r}")
        self.alpha, self.K, self.m, self.T = alpha, K, m, T
        self.disentangle, self.phase_mode = disentangle, phase_mode
        self.trend_pool, self.weights, self.device = trend_pool, weights, device

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

        # Supervised-contrastive term. Its projection head reads the SAME vector the probes
        # and the fine-tuning head read -- [trend mean | seasonal amp | seasonal (cos, sin)] --
        # so the label signal is pushed into the representation that is actually scored, not
        # into a pooled vector nothing downstream uses. The circular form is fixed here
        # whatever the encode-time readout, so one encoder still serves both readouts.
        # Built AFTER the queue: at w_supcon=0 it does not exist and draws no random numbers,
        # which keeps the pure self-supervised run bit-identical to the pre-SupCon code.
        self.w_supcon, self.supcon_temp, self.bins_per_day = w_supcon, supcon_temp, bins_per_day
        self.head_sup = (nn.Sequential(nn.Linear(supcon_dims, 256), nn.ReLU(),
                                       nn.Linear(256, 128)) if w_supcon > 0 else None)

    def _sup_features(self, trend, season):
        amp, pha = spectral_readout(season, self.bins_per_day, "circular")
        return torch.cat([trend.mean(dim=1), amp, pha], dim=-1)

    # -- MoCo state -------------------------------------------------------------------
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

    def _trend_view(self, z, idx):
        """The vector the trend term contrasts: one timestep (upstream) or the mean.

        'random' is upstream, but the head it goes through is discarded at inference and
        encode() mean-pools the whole sequence, so that objective never constrains the
        vector the probes read. 'mean' contrasts what is actually read.
        """
        return z.mean(1) if self.trend_pool == "mean" else z[:, idx]

    # -- objective --------------------------------------------------------------------
    def forward(self, x_q, x_k, labels=None, groups=None, update=True, return_parts=False):
        idx = np.random.randint(0, x_q.shape[1])
        q_t, q_s, q_n = self.encoder_q(x_q)
        q_tr = q_t                                   # un-projected trend, for the SupCon term

        if not self.disentangle:
            # Plain-SSL control: one representation, one MoCo, no seasonal term at all.
            q = F.normalize(self.head_q(self._trend_view(q_t, idx)), dim=-1)
            with torch.no_grad():
                if update:
                    self._momentum_update()
                k_t, _, _ = self.encoder_k(x_k)
                k = F.normalize(self.head_k(self._trend_view(k_t, idx)), dim=-1)
            loss, self.last_top1 = O.moco_ce_loss(q, k, self.queue.clone().detach(), self.T)
            if update:
                self._enqueue(k)
            z = loss.new_zeros(())
            return (loss, z, z) if return_parts else loss

        q_t = F.normalize(self.head_q(self._trend_view(q_t, idx)), dim=-1)
        with torch.no_grad():
            if update:
                self._momentum_update()
            k_t, _, _ = self.encoder_k(x_k)
            k_t = F.normalize(self.head_k(self._trend_view(k_t, idx)), dim=-1)
        trend, self.last_top1 = O.moco_ce_loss(q_t, k_t, self.queue.clone().detach(), self.T)
        if update:
            self._enqueue(k_t)

        # The seasonal branch is deliberately NOT MoCo, matching upstream: it has no queue,
        # so there are no stale keys for a momentum encoder to keep consistent, and the loss
        # is symmetric -- detaching one side behind an EMA copy would zero half its gradient
        # path. The key therefore comes from encoder_q, with gradients, at the cost of a
        # third encoder pass. Do not "fix" this to encoder_k.
        k_tr, k_s, _ = self.encoder_q(x_k)
        amp, pha = O.seasonal_loss(q_s, k_s, self.weights, self.phase_mode)
        total = O.total_loss(trend, amp, pha, self.weights, self.alpha)
        if self.head_sup is not None:
            # Both views of every window, read out exactly as the probes read them. Reuses
            # the two encoder_q passes above, so the supervised term costs no extra forward.
            if labels is None:
                raise ValueError("w_supcon > 0 but the batch carries no labels")
            z = self.head_sup(torch.cat([self._sup_features(q_tr, q_s),
                                         self._sup_features(k_tr, k_s)]))
            total = total + self.w_supcon * O.supcon_loss(
                z, torch.cat([labels, labels]), torch.cat([groups, groups]), self.supcon_temp)
        return (total, trend, amp + pha) if return_parts else total


# --------------------------------------------------------------------------------------
def adjust_learning_rate(optimizer, lr, step, total):
    """Half-cycle cosine decay, as upstream."""
    cur = lr * 0.5 * (1.0 + math.cos(math.pi * step / max(total, 1)))
    for g in optimizer.param_groups:
        g["lr"] = cur
    return cur


class CoST:
    """Fit the encoder, then read frozen representations out of it."""

    def __init__(self, input_dims, seq_len, bins_per_day, *, output_dims=320, hidden_dims=64,
                 depth=None, n_time_features=0, seasonal_bands="harmonics", disentangle=True,
                 mask_mode="none", trend_kernel_cap=None, seasonal_frac=0.5,
                 residual_dims=0, w_supcon=0.0, supcon_temp=0.1,
                 phase_readout="circular", phase_mode="circular_amp", trend_pool="random",
                 weights: O.TermWeights = O.PAPER, alpha=0.005, moco_k=4096,
                 jitter_sigma=0.1, shift_sigma=0.5, smooth_minutes=75.0,
                 lr=5e-4, batch_size=64, max_train_length=None, device="cuda",
                 model_seed=None):
        if phase_readout not in ("angle", "circular"):
            raise ValueError(f"phase_readout must be 'angle' or 'circular', got {phase_readout!r}")
        if w_supcon and not disentangle:
            raise ValueError("w_supcon needs the disentangled encoder: the supervised term "
                             "reads the trend/seasonal readout, which --plain does not have")
        if model_seed is not None:
            torch.manual_seed(model_seed)
            np.random.seed(model_seed % (2 ** 31))
            random.seed(model_seed)

        self.device = device
        self.seq_len, self.bins_per_day = seq_len, bins_per_day
        self.phase_readout = phase_readout
        self.batch_size, self.lr = batch_size, lr
        self.max_train_length = max_train_length
        self.jitter_sigma, self.shift_sigma = jitter_sigma, shift_sigma
        self.smooth_bins = smooth_bins_for(smooth_minutes, bins_per_day)
        self.disentangle, self.w_supcon = disentangle, w_supcon

        enc = dict(input_dims=input_dims, output_dims=output_dims, seq_len=seq_len,
                   bins_per_day=bins_per_day, hidden_dims=hidden_dims, depth=depth,
                   n_time_features=n_time_features, seasonal_bands=seasonal_bands,
                   disentangle=disentangle, mask_mode=mask_mode,
                   trend_kernel_cap=trend_kernel_cap, seasonal_frac=seasonal_frac,
                   residual_dims=residual_dims)
        self.net = CoSTEncoder(**enc).to(device)
        self.component_dims = self.net.seasonal_dims if disentangle else output_dims

        encoder_k = CoSTEncoder(**enc).to(device)
        self.cost = CoSTModel(
            self.net, encoder_k, dim=(self.net.trend_dims if disentangle else output_dims),
            alpha=alpha, K=moco_k, disentangle=disentangle, phase_mode=phase_mode,
            trend_pool=trend_pool, weights=weights, w_supcon=w_supcon,
            supcon_temp=supcon_temp, bins_per_day=bins_per_day,
            supcon_dims=(readout_width(seq_len, bins_per_day, "circular",
                                       self.net.trend_dims, self.net.seasonal_dims)
                         if disentangle else 0),
            device=device).to(device)
        self.n_iters = 0

    # -- training ---------------------------------------------------------------------
    def fit(self, train_data, n_iters=1000, val_data=None, log_every=100, verbose=True,
            train_labels=None, train_groups=None, val_labels=None, val_groups=None):
        """Pretrain. `train_data` is (N, T, D) float32.

        Labels are read ONLY when w_supcon > 0, and must then be the fold's training-
        participant labels (-1 for unlabelled windows) -- train.run_fold builds them and
        asserts no held-out participant is present. At w_supcon = 0 they are ignored.
        """
        if self.w_supcon and train_labels is None:
            raise ValueError("w_supcon > 0 needs train_labels; without them the supervised "
                             "term would silently see an all-unlabelled batch")
        loader = self._loader(train_data, train_labels, train_groups, shuffle=True)
        if len(loader.dataset) < self.batch_size:
            raise ValueError(f"{len(loader.dataset)} windows is fewer than one batch "
                             f"({self.batch_size}); lower --batch-size or widen the fold")

        val_loader = None
        if val_data is not None and len(val_data) >= self.batch_size:
            val_loader = self._loader(val_data, val_labels, val_groups, shuffle=False,
                                      multiplier=1)

        params = [p for p in self.cost.parameters() if p.requires_grad]
        opt = torch.optim.SGD(params, lr=self.lr, momentum=0.9, weight_decay=1e-4)

        hist = {"iters": [], "train": [], "val": [], "top1": []}
        self.cost.train()
        done = False
        while not done:
            for batch in loader:
                if self.n_iters >= n_iters:
                    done = True
                    break
                adjust_learning_rate(opt, self.lr, self.n_iters, n_iters)
                loss = self._loss(batch)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                self.n_iters += 1

                if self.n_iters % log_every == 0 or self.n_iters == n_iters:
                    v = self._validation_loss(val_loader) if val_loader else float("nan")
                    hist["iters"].append(self.n_iters)
                    hist["train"].append(float(loss.item()))
                    hist["val"].append(v)
                    hist["top1"].append(self.cost.last_top1)
                    if verbose:
                        print(f"    iter {self.n_iters:>6}  loss {loss.item():.4f}  "
                              f"val {v:.4f}  top1 {self.cost.last_top1:.4f}", flush=True)
        self.cost.eval()
        return hist

    def _loader(self, data, labels=None, groups=None, *, shuffle, multiplier=None):
        """Two augmented views per window, plus its label and participant for the SupCon term."""
        kw = {} if multiplier is None else {"multiplier": multiplier}
        ds = PretrainDataset(torch.as_tensor(data, dtype=torch.float),
                             jitter_sigma=self.jitter_sigma, shift_sigma=self.shift_sigma,
                             smooth_bins=self.smooth_bins, labels=labels, groups=groups, **kw)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, drop_last=True)

    def _loss(self, batch, update=True):
        x_q, x_k = batch[0].to(self.device), batch[1].to(self.device)
        labels, groups = batch[2].to(self.device), batch[3].to(self.device)
        if self.max_train_length and x_q.size(1) > self.max_train_length:
            off = np.random.randint(x_q.size(1) - self.max_train_length + 1)
            sl = slice(off, off + self.max_train_length)
            x_q, x_k = x_q[:, sl], x_k[:, sl]
        return self.cost(x_q, x_k, labels=labels, groups=groups, update=update)

    @torch.no_grad()
    def _validation_loss(self, loader):
        """Held-out pretext loss. `update=False` so monitoring never mutates MoCo state."""
        self.cost.eval()
        tot = n = 0
        for batch in loader:
            tot += float(self._loss(batch, update=False))
            n += 1
        self.cost.train()
        return tot / max(n, 1)

    # -- readout ----------------------------------------------------------------------
    def _spectral(self, z):
        """Frequency-domain readout of the seasonal branch: [amp | phase] or [amp | cos | sin].

        Time-domain pooling destroys this branch by construction. The seasonal output is an
        irFFT, so its mean over the window is EXACTLY the f=0 coefficient -- every oscillation
        integrates to zero -- and 'last' is one arbitrary phase point on the edge. Reading
        amplitude and phase at the chronobiological harmonics instead keeps the content the
        branch exists to carry.

        `phase_readout` is the experiment's first axis. 'angle' emits the raw atan2 angle,
        which is what run 2224103 used; 'circular' emits (cos, sin), which is correct across
        the branch cut -- 23.5 h and 0.5 h are one hour apart but average to 12.0 as raw
        angles. probe.IsotropicPairScaler is what keeps the (cos, sin) pair from being
        sheared by per-column standardisation downstream.
        """
        return spectral_readout(z, self.bins_per_day, self.phase_readout)

    @torch.no_grad()
    def encode(self, data, batch_size=256, pool="mean", parts=False):
        """Frozen representation, one vector per window.

        With `parts=True` returns the blocks separately -- trend, seasonal amplitude,
        seasonal phase -- which is what RQ2 needs to read the phase block on its own.
        """
        self.net.eval()
        X = torch.as_tensor(data, dtype=torch.float)
        outs = {"trend": [], "amp": [], "phase": [], "resid": [], "plain": []}
        for i in range(0, len(X), batch_size):
            t, s, r = self.net(X[i:i + batch_size].to(self.device))
            if s is None:
                outs["plain"].append(self._pool(t, pool).cpu())
                continue
            outs["trend"].append(self._pool(t, pool).cpu())
            a, p = self._spectral(s)
            outs["amp"].append(a.cpu())
            outs["phase"].append(p.cpu())
            if r is not None:
                # Time-domain like the trend branch, so it is pooled the same way. The
                # residual is emitted as its own block because the archive scores it alone
                # (0.7117) far above trend and seasonal together (0.6228).
                outs["resid"].append(self._pool(r, pool).cpu())
        cat = {k: torch.cat(v).numpy() for k, v in outs.items() if v}
        if "plain" in cat:
            return {"full": cat["plain"]} if parts else cat["plain"]
        blocks = ["trend", "amp", "phase"] + (["resid"] if "resid" in cat else [])
        full = np.concatenate([cat[k] for k in blocks], axis=-1)
        if not parts:
            return full
        return {"full": full, **{k: cat[k] for k in blocks}}

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
        # `phase_readout_at_construction` is PROVENANCE ONLY and `load` must never apply it.
        # These weights are readout-agnostic -- the readout is chosen at encode time and
        # never enters training -- so a checkpoint cannot own one. It is recorded under a
        # name that cannot be mistaken for a setting.
        torch.save({"net": self.net.state_dict(), "n_iters": self.n_iters,
                    "w_supcon": self.w_supcon,
                    "residual_dims": self.net.residual_dims,
                    "phase_readout_at_construction": self.phase_readout}, path)

    def load(self, path):
        """Load weights. The caller's `phase_readout` is preserved, deliberately.

        An earlier version restored `phase_readout` from the checkpoint, which silently
        overrode the readout the caller had asked for: every encoder is written under the
        readout it happened to be constructed with, so asking for 'circular' and loading a
        checkpoint built as 'angle' returned angle output under a circular label. That is
        the same class of failure that made runs 2224103 and 2412728 disagree, and it would
        have made the readout axis of this experiment measure nothing.
        """
        ck = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ck["net"])
        self.n_iters = ck.get("n_iters", 0)
        return self


# --------------------------------------------------------------------------------------
# Step C: end-to-end fine-tuning
# --------------------------------------------------------------------------------------
class WindowClassifier(nn.Module):
    """Encoder + the frozen path's own readout + a linear head, trained end to end.

    The head sits on `spectral_readout`, NOT on a time-mean of the representation. Mean
    pooling would destroy the seasonal branch by construction -- its output is an irFFT, so
    its mean over the window is exactly the f=0 coefficient and every oscillation integrates
    to zero. Reading it the same way the frozen probe does is also what makes the two
    directly comparable: the only thing that changes between them is whether the backbone
    receives gradient.
    """

    def __init__(self, encoder, seq_len, bins_per_day, phase_readout, dropout=0.5):
        super().__init__()
        self.encoder = encoder
        self.bins_per_day, self.phase_readout = bins_per_day, phase_readout
        w = readout_width(seq_len, bins_per_day, phase_readout,
                          encoder.trend_dims, encoder.seasonal_dims, encoder.residual_dims)
        self.norm = nn.LayerNorm(w)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(w, 1)

    def forward(self, x):
        t, s, r = self.encoder(x)
        if s is None:
            feat = t.mean(dim=1)
        else:
            amp, pha = spectral_readout(s, self.bins_per_day, self.phase_readout)
            feat = torch.cat([t.mean(dim=1), amp, pha], dim=-1)
            if r is not None:
                feat = torch.cat([feat, r.mean(dim=1)], dim=-1)
        return self.head(self.drop(self.norm(feat))).squeeze(-1)


def finetune(clf, Xtr, ytr, Xva, yva, *, device="cuda", epochs=40, batch_size=64,
             lr=1e-4, weight_decay=1e-4, patience=8, verbose=False):
    """Fine-tune end to end, early-stopping on a PARTICIPANT-DISJOINT validation split.

    The validation windows must come from participants held out of the training set and
    absent from the test fold. Splitting windows at random instead would put the same person
    on both sides, and early stopping would then select the epoch that best memorised those
    people -- the exact leak the whole protocol exists to prevent, arriving through the back
    door of model selection.

    `pos_weight` handles the class imbalance so the loss does not simply learn the majority.
    The best validation AUROC is restored at the end, so what is scored is the selected
    model rather than whatever the last epoch happened to produce.
    """
    from sklearn.metrics import roc_auc_score
    clf = clf.to(device)
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float)
    ytr_t = torch.as_tensor(ytr, dtype=torch.float)
    Xva_t = torch.as_tensor(Xva, dtype=torch.float).to(device)
    npos, nneg = float((ytr_t == 1).sum()), float((ytr_t == 0).sum())
    pos_weight = torch.tensor([nneg / max(npos, 1.0)], device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=weight_decay)

    best, best_state, bad = -np.inf, None, 0
    n = len(Xtr_t)
    for ep in range(epochs):
        clf.train()
        perm = torch.randperm(n)
        for i in range(0, n - batch_size + 1, batch_size):
            b = perm[i:i + batch_size]
            loss = lossf(clf(Xtr_t[b].to(device)), ytr_t[b].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        clf.eval()
        with torch.no_grad():
            pv = torch.cat([clf(Xva_t[i:i + 256]) for i in range(0, len(Xva_t), 256)]).cpu().numpy()
        auc = (roc_auc_score(yva, pv) if len(np.unique(yva)) > 1 else float("nan"))
        if np.isfinite(auc) and auc > best:
            best, bad = auc, 0
            best_state = {k: v.detach().clone() for k, v in clf.state_dict().items()}
        else:
            bad += 1
        if verbose:
            print(f"      ep {ep:3d} val AUROC {auc:.4f}{'  *' if bad == 0 else ''}", flush=True)
        if bad >= patience:
            break
    if best_state is not None:
        clf.load_state_dict(best_state)
    clf.eval()
    return clf, float(best), ep + 1


@torch.no_grad()
def predict_windows(clf, X, device="cuda", batch_size=256):
    """Sigmoid scores, one per window."""
    clf.eval()
    X = torch.as_tensor(X, dtype=torch.float)
    out = [torch.sigmoid(clf(X[i:i + batch_size].to(device))).cpu()
           for i in range(0, len(X), batch_size)]
    return torch.cat(out).numpy()
