import copy, math, random
from typing import Union, Callable, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from torch.utils.data import TensorDataset, DataLoader, Dataset

import numpy as np
from einops import rearrange

from models.encoder import CoSTEncoder
from models.positional_encoding import CALENDAR_PES
from utils import split_with_nan, centerize_vary_length_series, torch_pad_nan


class PretrainDataset(Dataset):

    def __init__(self,
                 data,
                 jitter_sigma=0.1,
                 shift_sigma=0.5,
                 p=0.5,
                 multiplier=10,
                 n_exact_tail=0,
                 pids=None,
                 positive="window",
                 decomp=None,
                 n_sensors=0,
                 bins_per_day=96,
                 smooth_bins=0):
        super().__init__()
        self.data = data
        self.p = p
        self.jitter_sigma = jitter_sigma
        self.shift_sigma = shift_sigma
        self.multiplier = multiplier
        # Trailing channels that must survive augmentation BIT-EXACT. With --pe factorized the
        # last two channels are integer calendar indices (time-of-day, day-of-week) read
        # straight into nn.Embedding: noise there both destroys the calendar semantics the
        # encoding exists for and pushes indices out of range (a CUDA device-side assert).
        self.n_exact_tail = int(n_exact_tail)
        self.N, self.T, self.D = data.shape # num_ts, time, dim

        # 'window'      -- both views are the SAME window. Measured on an untrained encoder
        #                  with a queue of real keys, top-1 retrieval is 1.000 against a chance
        #                  of 1/(K+1): the pair is a near-identity transform, so the task is
        #                  already solved at initialisation and the gradient teaches nothing.
        # 'participant' -- the second view is a DIFFERENT window of the same participant. Same
        #                  measurement: top-1 0.150, i.e. chance. The pair shares the person's
        #                  circadian amplitude and phase and nothing else, so matching it
        #                  requires encoding the rhythm.
        # 'day-disjoint' -- the two views are the SAME window rebuilt out of DISJOINT halves of
        #                  its own days, drawn with replacement, moved on day boundaries so the
        #                  time of day is preserved exactly. Measured on real windows, this is
        #                  what the three pairings leave the views sharing:
        #
        #                      pairing                   daily harmonics   rest of spectrum
        #                      window (jitter+shift)          0.707             0.756
        #                      day-disjoint alone             0.757             0.383
        #                      day-disjoint + jitter+shift    0.555             0.287
        #
        #                  The problem with 'window' is not that it is hard or easy: it is that
        #                  the two views share the WHOLE spectrum equally, so any feature
        #                  solves the task and the objective never says which one to learn.
        #                  Under a day-disjoint split the only content reliably shared is the
        #                  content at multiples of the daily frequency -- the circadian cycle
        #                  and its harmonics -- because permuting whole days leaves exactly
        #                  those bins standing. So the rhythm becomes the only route to a
        #                  solution, and there is room to move: a rhythm-only reader retrieves
        #                  at 0.196 where the untrained network gets 0.019 (chance 0.001).
        #
        #                  It survives the thin pools a 7-day HRD window gives (halves of 3 and
        #                  4 days): daily coherence 0.606 against 0.022 off-harmonic.
        assert positive in ("window", "participant", "day-disjoint"), positive
        self.positive = positive
        self.bins_per_day = int(bins_per_day)
        # Widest box filter the smoothing augmentation may draw. 0 disables it, which is the
        # default, so a run that does not ask for it is bit-identical to before.
        self.smooth_bins = int(smooth_bins)
        if positive == "day-disjoint" and self.T // self.bins_per_day < 2:
            raise ValueError(f"positive='day-disjoint' needs >=2 days per window, "
                             f"got T={self.T} at {self.bins_per_day} bins/day")
        # Participant index per window, as a plain int code. Needed whenever the MoCo queue
        # has to know WHOSE key each slot holds -- see CoSTModel.queue_pid and `negatives`.
        # It is built for every run, not just positive='participant', because the negative
        # sampler is an independent choice from the positive sampler. -1 means "unknown",
        # which the subject-conditional sampler treats as not-matching anything.
        self.pid_idx = np.full(self.N, -1, dtype=np.int64)
        if pids is not None:
            _p = np.asarray(pids)
            if len(_p) != self.N:
                raise ValueError(f"{len(_p)} pids for {self.N} windows")
            self.pid_idx = np.unique(_p, return_inverse=True)[1].astype(np.int64)
        self.peers = None
        if positive == "participant":
            if pids is None:
                raise ValueError("positive='participant' needs the pretrain windows' pids")
            pids = np.asarray(pids)
            if len(pids) != self.N:
                raise ValueError(f"{len(pids)} pids for {self.N} windows")
            by_pid = {}
            for i, q in enumerate(pids):
                by_pid.setdefault(q, []).append(i)
            # A participant contributing a single window has no peer; that window falls back to
            # the old behaviour rather than being dropped, so the pretraining set is unchanged.
            self.peers = [np.array([j for j in by_pid[q] if j != i], dtype=np.int64)
                          for i, q in enumerate(pids)]
            self.n_paired = sum(1 for pr in self.peers if len(pr))

        # Decomposition-consistent views. `decomp` is (tau, sigma, resid), each (N, T, n_sensors),
        # from the closed-form harmonic fit. Only the sensor channels are recomposed; any
        # trailing clock channels are copied from the original window untouched, because they
        # are deterministic functions of the timestamp and have no trend/seasonal/noise split.
        self.decomp = None
        self.n_sensors = int(n_sensors) or self.D
        if decomp is not None:
            tau, sig, res = (torch.as_tensor(a, dtype=torch.float) for a in decomp)
            assert tau.shape == sig.shape == res.shape, "tau/sigma/resid shapes differ"
            assert tau.shape[0] == self.N and tau.shape[1] == self.T, tau.shape
            self.decomp = (tau, sig, res)

    def _noise(self, i):
        """A fresh noise realisation: the window's OWN residual, circularly time-shifted.

        A roll preserves the residual's autocorrelation and per-channel scale exactly, so the
        two views differ by a realisation of the same noise process rather than by white
        Gaussian, which is what the hypothesis actually assumes.
        """
        return torch.roll(self.decomp[2][i], random.randrange(self.T), dims=0)

    def _compose(self, tau_i, sig_i, i):
        """One view: recomposed sensor channels, plus this window's clock channels verbatim."""
        x = tau_i + sig_i + self._noise(i)
        if self.n_sensors < self.D:
            x = torch.cat([x, self.data[i][:, self.n_sensors:]], dim=-1)
        return x

    def _peer(self, i):
        """Another window of the same participant, or `i` when the person has only one."""
        pr = None if self.peers is None else self.peers[i]
        return int(pr[random.randrange(len(pr))]) if pr is not None and len(pr) else i

    def _day_views(self, i):
        """Two views of window `i`, each rebuilt from one half of its own days.

        The days are split into two disjoint halves, and each view draws `D` days WITH
        REPLACEMENT from its own half, so the views share no raw day. Days move whole and on
        day boundaries, which is what preserves the time of day -- and therefore the circadian
        phase -- exactly.

        The DC offset is applied here with certainty rather than at the usual p=0.5. Resampling
        days preserves the window's mean level EXACTLY, and level is close to a participant
        fingerprint: without the offset, the pair is matched on level alone and no rhythm is
        learned. Measured -- a mean-only reader retrieves at 0.241 under a bare day-disjoint
        split and at 0.016 once the offset is applied. That makes the offset part of the
        pairing, not an optional augmentation, so it cannot be left to a coin flip.

        Trailing clock channels are rebuilt from the same day order as the sensor channels, so
        a view stays a consistent window rather than sensor data under someone else's calendar.
        """
        B = self.bins_per_day
        D = self.T // B
        order = torch.randperm(D)
        x = self.data[i]
        head = x[:D * B].reshape(D, B, -1)
        tail = x[D * B:]                                  # bins that do not fill a whole day
        out = []
        for half in (order[:D // 2], order[D // 2:]):
            idx = half[torch.randint(0, len(half), (D,))]
            v = head[idx].reshape(D * B, -1)
            if len(tail):
                v = torch.cat([v, tail], dim=0)
            if self.n_exact_tail:
                k = v.size(-1) - self.n_exact_tail
                v = torch.cat([self.jitter(self._offset(v[..., :k])), v[..., k:]], dim=-1)
            else:
                v = self.jitter(self._offset(v))
            out.append(v)
        return out[0], out[1]

    def _offset(self, x):
        """`shift` with no coin flip -- see _day_views for why this one is not optional."""
        return x + (torch.randn(x.size(-1)) * self.shift_sigma)

    def _decomp_views(self, i):
        """Views for the SEASONAL branch: share this window's sigma, swap tau, resample noise.

        Restricted to the seasonal branch by measurement. On an untrained encoder, a positive
        pair that shares only one component retrieves at:

            trend (tau)        top-1 0.980      <- a fingerprint any random encoder reads
            seasonal (sigma)   top-1 0.193
            noise (residual)   top-1 0.147      (chance 0.125)

        `harmonic_reference` defines trend as a degree-3 polynomial -- smooth, low-dimensional
        and 36% of the variance -- so ANY pair sharing it is solved without learning. Sharing
        sigma instead leaves the task at chance, which is what makes it learnable. The trend
        branch therefore keeps the existing pair; contrasting trend cannot be repaired this way.
        """
        tau, sig, _ = self.decomp
        c, d = random.randrange(self.N), random.randrange(self.N)
        return self._compose(tau[c], sig[i], i), self._compose(tau[d], sig[i], i)

    def __getitem__(self, item):
        i = item % self.N
        j = i
        if self.peers is not None and len(self.peers[i]):
            j = int(self.peers[i][random.randrange(len(self.peers[i]))])
        # Same pair for both branches -- the historical behaviour, returned in the same
        # 4-view shape so the training loop has one code path.
        if self.positive == "day-disjoint":
            q, k = self._day_views(i)
        else:
            q, k = self.transform(self.data[i]), self.transform(self.data[j])
        # The QUERY's participant, `i` not `j`: the negative sampler asks "whose window is
        # being matched", and the query is what the loss scores against the queue.
        pid = int(self.pid_idx[i])
        if self.decomp is None:
            return q, k, q, k, pid
        return (q, k) + self._decomp_views(i) + (pid,)

    def __len__(self):
        return self.data.size(0) * self.multiplier

    def transform(self, x):
        # Rhythm-preserving views. `scale` (random per-channel amplitude scaling) is
        # REMOVED: amplitude is the discriminative circadian feature (blunted in
        # depression), so contrasting scaled views would force amplitude-invariance and
        # erase the signal. `jitter` is reduced to realistic sensor-noise level
        # (sigma ~0.1, not 0.5). `shift` is kept because a constant per-channel offset
        # only moves the MESOR (the 0-frequency/DC bin) and leaves the amplitude and
        # phase of every rhythm untouched -- a safe non-trivial augmentation that still
        # guards against contrastive collapse.
        if self.n_exact_tail:
            k = x.size(-1) - self.n_exact_tail
            return torch.cat([self.jitter(self.shift(self.smooth(x[..., :k]))),
                              x[..., k:]], dim=-1)
        return self.jitter(self.shift(self.smooth(x)))

    def smooth(self, x):
        """A circular box filter of a random sub-hour width -- what declares sub-hour detail
        to be noise.

        In a contrastive objective the augmentation IS the definition of noise: whatever it
        destroys, the representation is trained to ignore. So each candidate carries a ceiling
        -- the predictive content of what survives it -- and those were measured on HRD over
        24 seeds, every one through an identical random projection and probe:

            sub-hour smoothing   0.6926     the only one ABOVE the raw window's 0.6884
            jitter               0.6884     removes nothing, so it defines no task at all
            per-channel offset   0.6835
            per-channel gain     0.6492
            day permutation      0.6303
            time roll            0.6273
            low-pass to tau+sig  0.6228     the decomposition itself, the costliest of all

        Only this one both defines a real invariance and raises the ceiling. Sub-hour variation
        is therefore noise by the operational definition, and everything coarser is signal.

        The filter is circular, so the window keeps its length and every rhythm keeps its phase
        exactly -- a same-length pad is what stops this from becoming a time-crop, whose
        ceiling is 0.6444.

        Widths are ODD only. An even box filter has no centre bin, so its pad is asymmetric
        and it moves every phase by half a bin -- measured at 0.0327 rad on the 24 h component
        at 96 bins/day, which is exactly half of 2*pi/96. That would make this a time shift
        wearing a smoother's clothes, and the ceiling for time shifts is 0.6273.
        """
        if self.smooth_bins < 3 or random.random() > self.p:
            return x
        w = random.randrange(3, (int(self.smooth_bins) // 2) * 2 + 2, 2)
        xt = x.transpose(0, 1).unsqueeze(0)                       # (1, C, T)
        xp = F.pad(xt, ((w - 1) // 2, (w - 1) // 2), mode="circular")
        return F.avg_pool1d(xp, kernel_size=w, stride=1).squeeze(0).transpose(0, 1)

    def jitter(self, x):
        if random.random() > self.p:
            return x
        return x + (torch.randn(x.shape) * self.jitter_sigma)

    def shift(self, x):
        if random.random() > self.p:
            return x
        return x + (torch.randn(x.size(-1)) * self.shift_sigma)


# How the seasonal (SFD) contrastive term compares FFT phases. See CoSTModel.circular_phase.
#   'raw'          -- upstream CoST: the raw atan2 angle, whose dot product is not a
#                     similarity between angles at all. Keep only to reproduce archived runs.
#   'circular'     -- [sin, cos], so the score is a function of the angular gap alone.
#   'circular_amp' -- 'circular' additionally weighted by each channel's own amplitude, so
#                     channels whose phase is undefined noise stop counting as much as real
#                     rhythms. A strict generalisation of 'circular' (identical when the
#                     amplitudes are equal).
PHASE_MODES = ("raw", "circular", "circular_amp")



class CoSTModel(nn.Module):
    def __init__(self,
                 encoder_q: nn.Module, encoder_k: nn.Module,
                 kernels: List[int],
                 device: Optional[str] = 'cuda',
                 dim: Optional[int] = 128,
                 alpha: Optional[float] = 0.05,
                 K: Optional[int] = 65536,
                 m: Optional[float] = 0.999,
                 T: Optional[float] = 0.07,
                 disentangle: bool = True,
                 phase_mode: str = "circular",
                 trend_pool: str = "random",
                 negatives: str = "global",
                 n_negatives: int = 0,
                 noise_weight: float = 0.0,
                 noise_mask_frac: float = 0.3,
                 noise_span: int = 8):
        super().__init__()
        # Weight of the V^N term. 0 turns the branch's objective off completely, so a run
        # written before this key existed trains exactly the model it trained.
        self.noise_weight = float(noise_weight)
        self.noise_mask_frac = float(noise_mask_frac)
        self.noise_span = int(noise_span)

        # 'random' is upstream CoST: the trend term contrasts ONE random timestep, pushed
        # through head_q. But head_q is discarded at inference and encode() mean-pools the
        # whole sequence, so the objective never constrains the vector the probes read.
        # Measured on run 1239199, two DIFFERENT participants sit at cos 0.9956 in that
        # mean-pooled vector while two views of the SAME window sit at 0.9652 -- the
        # augmentation moves it further than identity does. 'mean' contrasts what is
        # actually read. A/B it; do not switch the default without that measurement.
        assert trend_pool in ("random", "mean"), trend_pool
        self.trend_pool = trend_pool
        # The experimental variable of the subject-conditional-negatives study, plus the
        # fixed negative count that makes the two modes comparable. See select_negatives.
        assert negatives in ("global", "subject"), negatives
        self.negatives = negatives
        self.n_negatives = int(n_negatives)
        self.neg_short = 0           # queries whose participant had < n_negatives slots
        self.neg_calls = 0           # queries seen, so the shortfall RATE is reportable
        self.K = K
        self.m = m
        self.T = T
        self.device = device
        # How the seasonal PHASE is compared -- see circular_phase and PHASE_MODES.
        if phase_mode not in PHASE_MODES:
            raise ValueError(f"phase_mode must be one of {PHASE_MODES}, got: {phase_mode}")
        self.phase_mode = phase_mode

        self.kernels = kernels

        self.alpha = alpha

        self.encoder_q = encoder_q
        self.encoder_k = encoder_k

        # create the encoders
        self.head_q = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.head_k = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.register_buffer('queue', F.normalize(torch.randn(dim, K), dim=0))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        # Whose key sits in each queue slot. -1 = never written, which select_negatives
        # excludes so the initial random vectors are never contrasted against.
        self.register_buffer('queue_pid', torch.full((K,), -1, dtype=torch.long))

        # GradNorm: learnable weights for the two task losses [trend, seasonal]. Used only
        # when loss balancing = 'gradnorm'; ignored (kept =1) in fixed-alpha mode.
        self.loss_w = nn.Parameter(torch.ones(2))
        self.register_buffer('loss_L0', torch.zeros(2))     # initial losses (set on first step)
        self.loss_L0_set = False

        self.disentangle = disentangle          # False = plain SSL: single rep, no TFD/SFD at all


    def shared_param(self):
        """Last trainable parameter of the SHARED backbone (before the TFD/SFD split);
        GradNorm measures each task's gradient magnitude with respect to this tensor."""
        ps = [p for p in self.encoder_q.feature_extractor.parameters() if p.requires_grad]
        return ps[-1]


    def select_negatives(self, pid):
        """Indices of the queue slots this batch contrasts against -- (N, n_negatives).

        THE experimental variable. `n_negatives` is drawn either way, so the two modes differ
        in the COMPOSITION of the denominator and in nothing else; sampling a fixed count is
        what makes them comparable, since InfoNCE improves with more negatives and a
        subject-conditional queue is inevitably the smaller pool.

          'global'  -- uniform over the whole queue. The shipped behaviour, and degenerate:
                       the negatives are almost all OTHER participants, so a query is matched
                       to its own augmented view by participant identity alone. Measured on an
                       untrained encoder, top-1 retrieval is 1.000 against a chance of
                       1/(K+1) -- the task is solved at initialisation and teaches nothing.
          'subject' -- uniform over the queue slots holding keys from the SAME participant.
                       Identity is then constant across the denominator and cannot separate
                       anything, so the only thing left to encode is how this window differs
                       from that person's OTHER windows, i.e. their week-to-week rhythm state.

        Slots that have never been written (queue_pid < 0) are excluded in BOTH modes, so the
        comparison is not contaminated by the random vectors the queue is initialised with.

        A query whose participant holds fewer than `n_negatives` slots is sampled WITH
        replacement rather than topped up from other participants. Reverting to a global draw
        would silently turn that query back into the control condition, which is the one
        failure this experiment cannot tolerate -- with K/n_pretrain_pids only ~35 at the
        shipped K=4096 against n_negatives=32 it would have fired constantly. Duplicated
        negatives merely reweight the denominator; foreign negatives would change what is
        being tested. `neg_short` counts it so the rate is reportable either way.
        """
        # n_negatives <= 0 means "the whole queue", which is the SHIPPED behaviour and the
        # default. Returning None here keeps that path bit-identical to the original code
        # rather than approximating it with a large sample -- a default that quietly changed
        # what every future run contrasts against would invalidate comparisons with every run
        # already in results_hrd/.
        if self.n_negatives <= 0 and self.negatives == "global":
            return None
        n_neg, dev = self.n_negatives, self.queue.device
        valid = (self.queue_pid >= 0)
        pool = valid.nonzero(as_tuple=True)[0]
        if pool.numel() < n_neg:                      # queue not warm yet: use everything
            pool = torch.arange(self.K, device=dev)
        n = pid.shape[0] if pid is not None else 1
        glob = pool[torch.randint(pool.numel(), (n, n_neg), device=dev)]
        if self.negatives == "global" or pid is None:
            self.neg_calls += n
            return glob
        idx = glob.clone()
        for r in range(n):
            same = (valid & (self.queue_pid == pid[r])).nonzero(as_tuple=True)[0]
            if same.numel() >= n_neg:
                idx[r] = same[torch.randperm(same.numel(), device=dev)[:n_neg]]
            elif same.numel() > 0:
                idx[r] = same[torch.randint(same.numel(), (n_neg,), device=dev)]
                self.neg_short += 1
            else:
                # This participant has NOTHING in the queue yet -- only reachable in the first
                # few iterations, before the queue has seen them at all.
                self.neg_short += 1
        self.neg_calls += n
        return idx

    def compute_loss(self, q, k, k_negs):
        # compute logits
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK  (k_negs is (N, n_negatives, C) once the sampler has chosen)
        l_neg = (torch.einsum('nc,nkc->nk', [q, k_negs]) if k_negs.dim() == 3
                 else torch.einsum('nc,ck->nk', [q, k_negs]))

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # Top-1 retrieval: the fraction of queries whose POSITIVE outscores every negative.
        # This is the project's difficulty measure for the pretext task, and it is recorded
        # here rather than recomputed elsewhere so it can never drift from the loss it
        # describes. At 1.000 on an untrained encoder the task is already solved and the
        # gradient teaches nothing; chance is 1/(1 + n_negatives).
        with torch.no_grad():
            self.last_top1 = float((l_pos > l_neg.max(dim=1, keepdim=True).values)
                                   .float().mean())

        # labels: positive key indicators - first dim of each batch
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)

        return loss

    def convert_coeff(self, x, eps=1e-6):
        amp = torch.sqrt((x.real + eps).pow(2) + (x.imag + eps).pow(2))
        phase = torch.atan2(x.imag, x.real + eps)
        return amp, phase

    @staticmethod
    def circular_phase(phase, amp=None, eps=1e-12):
        """Unit-circle embedding of an angle: ``phi -> [sin phi ; cos phi]``, optionally
        weighted per channel by ``amp``.

        WHY AT ALL. instance_contrastive_loss scores pairs with a DOT PRODUCT over the channel
        axis. On raw atan2 output that is not a similarity between angles -- ``<phi_i, phi_j>``
        depends on where the angles sit, not how far apart they are: two IDENTICAL phases score
        0 at ``phi=0`` but ``pi^2`` at ``phi=pi``, and the pair ``(pi-eps, -pi+eps)`` -- the
        same angle either side of atan2's branch cut -- scores the most NEGATIVE value
        possible. With ``C = component_dims = 160`` channels ~64% of near-identical view pairs
        have at least one channel straddling the cut. After the embedding the dot product
        becomes ``sum_c cos(phi_i,c - phi_j,c)``: a function of the angular gap alone,
        2pi-periodic and monotone in it.

        WHY THE WEIGHT. Unweighted, every channel gets unit norm -- so a channel whose
        amplitude is ~0, where the phase is undefined noise, counts exactly as much as a strong
        rhythmic one. Passing ``amp`` weights each channel by its own amplitude, turning the
        score into

            sum_c  w_i,c * w_j,c * cos(phi_i,c - phi_j,c)

        i.e. an AMPLITUDE-WEIGHTED phase coherence (the weighting used by phase-locking
        statistics), so weak channels contribute little and strong rhythms dominate.

        THE WEIGHT IS RMS-NORMALISED, not L2-normalised, and that matters. The loss applies
        log_softmax straight to these dot products with no temperature, so the embedding's
        NORM sets the effective softmax temperature. Unweighted, ``||emb|| = sqrt(C)``. With
        ``||w||=1`` (plain L2) it would collapse to 1 -- a ~sqrt(C) = 12.6x logit shrink at
        C=160, flattening the softmax and starving the phase branch of gradient. Normalising
        so ``mean(w^2) = 1`` keeps ``||emb|| = sqrt(C)`` exactly, so this mode changes only the
        RELATIVE weighting of channels and nothing else. It is therefore a strict
        generalisation: with equal amplitudes ``w == 1`` and it reduces to the unweighted
        embedding bit for bit.

        ``amp`` IS DETACHED. The weight expresses "how much do I trust this channel's phase",
        not a free parameter for the phase term to optimise -- amplitude is already trained by
        its own contrastive term, and letting the phase loss push on it too would let the model
        lower the phase loss by shrinking amplitudes rather than by aligning phases.
        """
        s, c = phase.sin(), phase.cos()
        if amp is not None:
            # mean(w^2) = 1 over the channel axis -> preserves ||emb|| = sqrt(C); see above.
            w = amp.detach()
            w = w / w.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)
            s, c = s * w, c * w
        return torch.cat([s, c], dim=-1)

    def instance_contrastive_loss(self, z1, z2):
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # 2B x T x C
        z = z.transpose(0, 1)  # T x 2B x C
        sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B x (2B-1)
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]
        logits = -F.log_softmax(logits, dim=-1)

        i = torch.arange(B, device=z1.device)
        loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
        return loss

    def _masked_noise_loss(self, x):
        """MSE on the residual at timesteps the branch was not shown.

        Not contrastive, and that is the point. Every other objective here is invariance
        learning: it can only discard, and its ceiling is the predictive content of whatever
        its positive pair leaves invariant. Measured over every implemented pair on HRD, no
        such ceiling clears what an untrained baseline already reaches -- window 0.7151,
        participant 0.6658, day-disjoint 0.6574, against 0.7198 for a random projection of
        the raw window. That family is bounded on this data.

        A prediction task is not bounded that way. It cannot be satisfied by throwing
        anything away, and it cannot be solved at initialisation -- which the contrastive
        trend task can, at 26x chance.

        Contiguous SPANS, not scattered timesteps. A residual is high-frequency, so a
        scattered mask is filled by interpolating the neighbours on either side, and the
        branch would learn a smoother rather than the structure. A span long enough to cover
        several bins has no such shortcut.
        """
        import math
        B, T = x.size(0), x.size(1)
        n_span = max(1, int(round(self.noise_mask_frac * T / self.noise_span)))
        mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
        starts = torch.randint(0, max(1, T - self.noise_span), (B, n_span), device=x.device)
        for k in range(n_span):
            idx = starts[:, k].unsqueeze(1) + torch.arange(self.noise_span, device=x.device)
            mask.scatter_(1, idx.clamp(max=T - 1), True)
        pred, target = self.encoder_q.reconstruct_noise(x, mask)
        m = mask.unsqueeze(-1).expand_as(target)
        if not m.any():
            return None
        # Per CHANNEL over the whole batch, not per window. Normalising each window's own
        # standard deviation looked equivalent and is not: a near-binary channel -- sleep --
        # has windows whose residual is almost constant, the clamp at 1e-6 then divides by
        # about nothing, and the loss came out at 5.1e6 instead of order 1. The floor is
        # relative to the batch's own scale so it cannot be tuned into irrelevance by the
        # units the data happens to arrive in.
        sd = target.std(dim=(0, 1), keepdim=True)
        sd = sd.clamp_min(0.01 * sd.mean().clamp_min(1e-8))
        return F.mse_loss((pred / sd)[m], (target / sd)[m])

    def _trend_view(self, z, idx):
        """The vector the trend term contrasts: one timestep (upstream) or the mean."""
        return z.mean(1) if self.trend_pool == "mean" else z[:, idx]

    def forward(self, x_q, x_k, x_q_s=None, x_k_s=None, update=True, return_parts=False,
                pid=None):
        # `update=False` runs the loss WITHOUT mutating MoCo state (no momentum
        # update / no queue enqueue) -- used to monitor a held-out validation loss.
        # compute query features
        rand_idx = np.random.randint(0, x_q.shape[1])

        q_t, q_s = self.encoder_q(x_q)
        # The seasonal branch gets its OWN positive pair when the views are branch-specific.
        # Identical tensors mean the historical single-pair behaviour, and the extra encoder
        # pass is skipped.
        if x_q_s is not None and x_q_s is not x_q:
            _, q_s = self.encoder_q(x_q_s)

        # --- plain contrastive SSL (no disentangler): the encoder returns a SINGLE
        #     representation (q_s is None); ONE MoCo on it. NO seasonal FFT loss, and
        #     encode() returns that single vector -- so the whole trend/seasonal split
        #     is ABSENT (not just its objective). ---
        if not self.disentangle:
            rep_q = q_t if q_s is None else q_t + q_s
            q = F.normalize(self.head_q(self._trend_view(rep_q, rand_idx)), dim=-1)
            with torch.no_grad():
                if update:
                    self._momentum_update_key_encoder()
                k_t, k_s = self.encoder_k(x_k)
                rep_k = k_t if k_s is None else k_t + k_s
                k = F.normalize(self.head_k(self._trend_view(rep_k, rand_idx)), dim=-1)
            loss = self.compute_loss(q, k, self.queue.clone().detach())
            if update:
                self._dequeue_and_enqueue(k)
            z = loss.new_zeros(())
            return (loss, z) if return_parts else loss

        if q_t is not None:
            q_t = F.normalize(self.head_q(self._trend_view(q_t, rand_idx)), dim=-1)

        # compute key features
        with torch.no_grad():  # no gradient for keys
            if update:
                self._momentum_update_key_encoder()  # update key encoder
            k_t, k_s = self.encoder_k(x_k)
            if k_t is not None:
                k_t = F.normalize(self.head_k(self._trend_view(k_t, rand_idx)), dim=-1)

            # (N, n_negatives, C): the queue rows this batch actually contrasts against.
            neg_idx = self.select_negatives(pid)
        _q = self.queue.clone().detach()
        k_negs = _q if neg_idx is None else _q.T[neg_idx]
        trend_loss = self.compute_loss(q_t, k_t, k_negs)
        if update:
            self._dequeue_and_enqueue(k_t, pid)

        # NOTE -- the seasonal branch is deliberately NOT MoCo, and this asymmetry with the
        # trend branch above is intentional (it matches salesforce/CoST upstream exactly):
        #   * the seasonal KEY comes from encoder_q, not the EMA encoder_k;
        #   * this call is OUTSIDE the no_grad block, so both seasonal views carry gradients.
        # Why: SFD uses a within-batch instance-discrimination loss (TS2Vec style) with NO
        # queue. The momentum encoder exists in MoCo only to keep stale queued keys consistent
        # as the encoder drifts -- with no queue there is nothing to keep consistent. And
        # instance_contrastive_loss is symmetric in its two arguments, so detaching one side
        # behind an EMA copy would zero out half the objective's gradient path. Cost: a third
        # encoder forward per step (x_q->encoder_q, x_k->encoder_k, x_k->encoder_q).
        # Do NOT "fix" this to encoder_k -- it would diverge from the paper and upstream.
        q_s = F.normalize(q_s, dim=-1)
        _, k_s = self.encoder_q(x_k if x_k_s is None else x_k_s)
        k_s = F.normalize(k_s, dim=-1)

        with torch.autocast(device_type='cuda', enabled=False):
            q_s_freq = fft.rfft(q_s.float(), dim=1)
            k_s_freq = fft.rfft(k_s.float(), dim=1)
            q_s_amp, q_s_phase = self.convert_coeff(q_s_freq)
            k_s_amp, k_s_phase = self.convert_coeff(k_s_freq)
            # Contrast the phase on the unit circle, not as a raw angle -- see circular_phase.
            # Kept inside the fp32 block so sin/cos are not evaluated in bf16.
            if self.phase_mode != "raw":
                # 'circular_amp' additionally weights each channel by its own amplitude, so a
                # channel whose phase is undefined noise stops counting as much as a real
                # rhythm. Each view is weighted by ITS OWN amplitude.
                w_q = q_s_amp if self.phase_mode == "circular_amp" else None
                w_k = k_s_amp if self.phase_mode == "circular_amp" else None
                q_s_phase = self.circular_phase(q_s_phase, w_q)
                k_s_phase = self.circular_phase(k_s_phase, w_k)

        seasonal_loss = (self.instance_contrastive_loss(q_s_amp, k_s_amp) +
                         self.instance_contrastive_loss(q_s_phase, k_s_phase)) / 2

        # V^N: the third component the hypothesis names. Contrasted at the WINDOW level,
        # which is not a free choice -- the invariant subspace of a participant-level
        # positive pair carries 0.5856 against the residual's own 0.7117 (4/24, p=0.0015),
        # so that pairing is ruled out by measurement rather than by taste.
        #
        # The same in-batch instance contrast the seasonal branch uses, not a second MoCo
        # queue: positives are two views of one window and negatives are the rest of the
        # batch, which is exactly the pairing whose ceiling was measured at 0.7202 -- above
        # every arm in this project.
        noise_loss = None
        if self.noise_weight > 0 and getattr(self.encoder_q, "noise_branch", False):
            noise_loss = self._masked_noise_loss(x_q)

        # GradNorm balances trend vs seasonal.
        if return_parts:
            return trend_loss, seasonal_loss
        total = trend_loss + self.alpha * seasonal_loss
        return total if noise_loss is None else total + self.noise_weight * noise_loss

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update for key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)
        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, pid=None):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        # replace keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        # The owner of each slot moves with it. Without this the queue would say nothing
        # about whose key it holds and 'subject' negatives could not be selected at all.
        self.queue_pid[ptr:ptr + batch_size] = (
            torch.full((batch_size,), -1, dtype=torch.long, device=self.queue_pid.device)
            if pid is None else pid.to(self.queue_pid.device).long())

        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr


class CoST:
    def __init__(self,
                 input_dims: int,
                 n_time_features: int,
                 kernels: List[int],
                 alpha: bool,
                 max_train_length: int,
                 output_dims: int = 320,
                 hidden_dims: int = 64,
                 depth: int = 10,
                 backbone: str = 'tcn',
                 pe: str = 'sinusoidal',
                 time2vec_dim: int = 65,
                 loss_balance: str = "fixed",
                 bins_per_day: int = 96,
                 seasonal_bands: str = 'harmonics',
                 disentangle: bool = True,
                 jitter_sigma: float = 0.1,
                 shift_sigma: float = 0.5,
                 smooth_bins: int = 0,
                 phase_readout: str = "angle",
                 # V^N, the third component the hypothesis names and the model never had.
                 # noise_weight=0 (the default) leaves the branch unbuilt and the loss
                 # untouched, so every archived config trains the model it trained.
                 # noise_branch=True builds it WITHOUT training it, which is what the
                 # architecture-matched random-init control needs.
                 noise_weight: float = 0.0,
                 noise_branch: bool = False,
                 noise_depth: Optional[int] = None,
                 # Fraction of timesteps hidden, and how long each hidden run is. Spans, not
                 # scattered steps: a scattered mask on a high-frequency residual is filled
                 # by interpolating its neighbours, which teaches a smoother.
                 noise_mask_frac: float = 0.3,
                 noise_span: int = 8,
                 moco_k: int = 4096,
                 trend_pool: str = "random",
                 positive_pair: str = "window",
                 negatives: str = "global",
                 n_negatives: int = 0,
                 decomp_aug: bool = False,
                 n_sensors: int = 0,
                 mask_mode: str = 'none',
                 mask_prob: float = 0.5,
                 phase_mode: str = "circular",
                 device: 'str' ='cuda',
                 lr: float = 0.001,
                 batch_size: int = 16,
                 after_iter_callback: Union[Callable, None] = None,
                 after_epoch_callback: Union[Callable, None] = None):

        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self.loss_balance = loss_balance
        # augmentation strength (CoST default jitter_sigma=0.1). `mask_mode` selects the
        # training-time timestep-masking augmentation and defaults to 'none' -- upstream
        # CoST never applied one (its encoder's mask argument hard-defaulted to 'all_true'),
        # so 'none' reproduces published behaviour and 'binomial'/'continuous' are opt-in.
        # `mask_prob` is the binomial KEEP-probability and is read only under 'binomial'.
        self.jitter_sigma = jitter_sigma
        # `shift` adds a per-channel constant. Its comment calls that safe because it moves
        # only the MESOR, i.e. the 0-frequency bin -- true of the INPUT, but the backbone is
        # non-linear, so it is not true of the representation: measured on run 1239199,
        # shift alone drops the seasonal readout's self-similarity to 0.739 while jitter alone
        # leaves it at 0.931. Setting this to 0 removes the augmentation and asks whether the
        # level-invariance it imposes is what costs MESOR recovery against random-init.
        self.shift_sigma = shift_sigma
        # Widest sub-hour box filter the smoothing augmentation may draw; 0 disables it. See
        # PretrainDataset.smooth for the ceiling each augmentation family imposes.
        self.smooth_bins = int(smooth_bins)
        # How the seasonal readout emits phase. "angle" is the raw atan2 output and is
        # kept only so archived configs -- which have no such key and fall back to it --
        # rebuild the model they actually ran. New runs default to "circular".
        assert phase_readout in ("angle", "circular"), phase_readout
        self.phase_readout = phase_readout
        self.moco_k = moco_k
        self.trend_pool = trend_pool
        self.mask_mode = mask_mode
        self.mask_prob = mask_prob
        # The calendar PEs append 2 raw [tod, dow] channels read as a wall-clock index; they
        # must reach the encoder unaugmented (see PretrainDataset.n_exact_tail) -- jittering a
        # clock index would silently move the bin to a different time of day.
        self._n_exact_tail = 2 if pe in CALENDAR_PES else 0
        self.bins_per_day = int(bins_per_day)   # used by the seasonal spectral readout

        # GradNorm balances exactly two losses: _gradnorm_step unpacks
        # `L_t, L_s = cost(..., return_parts=True)`, so a third term would be built, read at
        # inference, and never trained -- silently, and the run would report a V^N model
        # whose branch had never seen a gradient. Balancing three losses is a design change
        # that has not been validated, so this refuses the combination instead of guessing.
        if noise_weight and loss_balance == "gradnorm":
            raise ValueError(
                "noise_weight>0 with loss_balance='gradnorm' would train V^T and V^S only "
                "and leave V^N untrained while still reading it. Use --loss-balance fixed.")

        if kernels is None:
            kernels = []

        self.net = CoSTEncoder(
            input_dims=input_dims, output_dims=output_dims,
            kernels=kernels,
            length=max_train_length,
            hidden_dims=hidden_dims, depth=depth,
            backbone=backbone, pe=pe,
            n_time_features=n_time_features,
            time2vec_dim=time2vec_dim,
            disentangle=disentangle,
            bins_per_day=bins_per_day,
            seasonal_bands=seasonal_bands,
            mask_mode=mask_mode,
            mask_prob=mask_prob,
            # A non-zero weight implies the branch; `noise_branch` alone builds it without
            # training it, which is what the architecture-matched random-init control needs.
            noise_branch=bool(noise_weight) or bool(noise_branch),
            noise_depth=noise_depth,
        ).to(self.device)

        # MoCo head/queue dim: two branches (CoST) -> per-branch component_dims;
        # plain single representation (--no-disentangle) -> the full output_dims.
        moco_dim = self.net.component_dims if disentangle else self.net.output_dims

        # Queue length. HELD AT THE UPSTREAM CoST VALUE, 4096.
        #
        # A K=1024 change was applied and then reverted unrun. The argument for it is real --
        # this cohort pretrains on ~3,000 NON-overlapping windows, so K=4096 > N and the queue
        # holds stale copies of the query's own window as false negatives, which MoCo's K << N
        # assumption exists to avoid. But it was never measured, and all 17 healthy runs used
        # 4096 (pretrain loss 0.079-0.537, clean 7-cycle seasonal structure), so it is not the
        # thing that broke run 19937323 -- MASK_MODE=binomial was (see scripts/run.sh).
        #
        # Changing K also moves the loss scale: chance is ln(K+1), so 8.32 nats at 4096 versus
        # 6.93 at 1024. Altering it in the same run that reverts the mask would make the new
        # loss incomparable to the historical baseline, which is the one measurement that
        # verifies the revert worked. Re-baseline first at 4096, then A/B K=1024 on one seed.
        self.positive_pair = positive_pair
        # Seasonal-branch views recomposed from each window's own decomposition; see
        # PretrainDataset._decomp_views for why the trend branch is excluded.
        self.decomp_aug = bool(decomp_aug)
        self.n_sensors = int(n_sensors)
        self.cost = CoSTModel(
            self.net,
            copy.deepcopy(self.net),
            kernels=kernels,
            dim=moco_dim,
            alpha=alpha,
            K=moco_k,
            disentangle=disentangle,
            phase_mode=phase_mode,
            trend_pool=trend_pool,
            negatives=negatives,
            n_negatives=n_negatives,
            noise_weight=noise_weight,
            noise_mask_frac=noise_mask_frac,
            noise_span=noise_span,
            device=self.device,
        ).to(self.device)

        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        
        self.n_epochs = 0
        self.n_iters = 0
        self.loss_w_log = []

    def _gradnorm_step(self, x_q, x_k, optimizer, gn_optimizer, gamma=1.5):
        """One optimisation step with GradNorm balancing of the trend vs seasonal losses.
        Adaptively reweights the two so they train at comparable rates -- fixing the seasonal
        branch being starved by a tiny fixed alpha. Runs in fp32 (GradNorm builds a stable
        second-order graph). Returns the scalar total loss."""
        cost = self.cost
        L_t, L_s = cost(x_q, x_k, return_parts=True)            # trend, seasonal (unweighted)
        losses = torch.stack([L_t, L_s])
        if not cost.loss_L0_set:
            cost.loss_L0.copy_(losses.detach()); cost.loss_L0_set = True
        w = cost.loss_w
        weighted = w * losses
        total = weighted.sum()

        optimizer.zero_grad(); gn_optimizer.zero_grad()
        total.backward(retain_graph=True)                        # main model gradients

        W = cost.shared_param()                                  # shared backbone tensor
        G = torch.stack([torch.autograd.grad(weighted[i], W, retain_graph=True,
                                             create_graph=True)[0].norm() for i in range(2)])
        G_bar = G.mean().detach()
        ratio = losses.detach() / (cost.loss_L0 + 1e-8)          # inverse training rate
        r = ratio / ratio.mean()
        target = (G_bar * r.pow(gamma)).detach()
        L_grad = (G - target).abs().sum()
        w.grad = torch.autograd.grad(L_grad, w)[0]               # overwrite: GradNorm grad only

        optimizer.step()                                         # model params (w is excluded)
        gn_optimizer.step()                                      # the two task weights
        with torch.no_grad():                                    # keep sum(w)=2, positive
            cost.loss_w.data.clamp_(min=1e-3)
            cost.loss_w.data.mul_(2.0 / cost.loss_w.data.sum())
        self.loss_w_log.append(cost.loss_w.detach().cpu().numpy().copy())
        return float(total.item())

    def fit(self, train_data, n_epochs=None, n_iters=None, valid_data=None,
            verbose=False, pids=None, valid_pids=None):
        assert train_data.ndim == 3

        if n_iters is None and n_epochs is None:
            n_iters = 200 if train_data.size <= 100000 else 600

        # `pids` must follow every reshape and every dropped row, or a window would be paired
        # with a peer belonging to somebody else -- silently, and in the direction that makes
        # the objective look easier.
        pids = None if pids is None else np.asarray(pids)
        if pids is not None and len(pids) != len(train_data):
            raise ValueError(f"{len(pids)} pids for {len(train_data)} pretrain windows")

        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)
                if pids is not None:
                    pids = np.tile(pids, sections)      # split_with_nan stacks the sections

        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)

        keep = ~np.isnan(train_data).all(axis=2).all(axis=1)
        train_data = train_data[keep]
        if pids is not None:
            pids = pids[keep]

        # `shift` is a per-channel DC offset (moves only the MESOR / 0-frequency bin), so it
        # preserves every rhythm's phase and amplitude, and is useful (jitter alone is weak).
        multiplier = 1 if train_data.shape[0] >= self.batch_size else math.ceil(self.batch_size / train_data.shape[0])
        # Closed-form trend/seasonal split, computed ONCE -- the same decomposition RQ1
        # scores against, so augmentation and evaluation share one definition.
        decomp = None
        if self.decomp_aug:
            from tasks.decomposition import harmonic_reference
            ns = self.n_sensors or train_data.shape[-1]
            xs = np.nan_to_num(train_data[..., :ns], nan=0.0)
            _tau, _sig = harmonic_reference(xs, self.bins_per_day)
            decomp = (_tau, _sig, xs - _tau - _sig)

        train_dataset = PretrainDataset(torch.from_numpy(train_data).to(torch.float), jitter_sigma=self.jitter_sigma, shift_sigma=self.shift_sigma, multiplier=multiplier, n_exact_tail=self._n_exact_tail,
                                        pids=pids, positive=self.positive_pair,
                                        decomp=decomp, n_sensors=self.n_sensors,
                                        bins_per_day=self.bins_per_day,
                                        smooth_bins=self.smooth_bins)
        if self.positive_pair == "participant" and verbose:
            print(f"[pretrain] positive pair = another window of the same participant "
                  f"({train_dataset.n_paired}/{len(train_data)} windows have a peer)")
        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True, drop_last=True)

        # optional held-out set to monitor the SSL loss over training (no labels used)
        val_loader = None
        if valid_data is not None:
            vdec = None
            vkeep = ~np.isnan(valid_data).all(axis=2).all(axis=1)
            vd = valid_data[vkeep]
            vp = None if valid_pids is None else np.asarray(valid_pids)[vkeep]
            if self.decomp_aug and len(vd):
                from tasks.decomposition import harmonic_reference
                ns = self.n_sensors or vd.shape[-1]
                vxs = np.nan_to_num(vd[..., :ns], nan=0.0)
                _vt, _vs = harmonic_reference(vxs, self.bins_per_day)
                vdec = (_vt, _vs, vxs - _vt - _vs)
            if len(vd) >= self.batch_size:
                # The held-out loss must measure the SAME task, so the validation views are
                # paired the same way; without pids it falls back to same-window pairing and
                # would report a different, easier objective.
                val_ds = PretrainDataset(torch.from_numpy(vd).to(torch.float), jitter_sigma=self.jitter_sigma, shift_sigma=self.shift_sigma, multiplier=1, n_exact_tail=self._n_exact_tail,
                                         pids=vp,
                                         positive=(self.positive_pair
                                                   if vp is not None or self.positive_pair != "participant"
                                                   else "window"),
                                         decomp=vdec, n_sensors=self.n_sensors,
                                         bins_per_day=self.bins_per_day,
                                         smooth_bins=self.smooth_bins)
                val_loader = DataLoader(val_ds, batch_size=min(self.batch_size, len(val_ds)),
                                        shuffle=False, drop_last=True)

        # The Transformer backbone trains unstably under SGD (large epoch-to-epoch
        # val-loss oscillation, so the final checkpoint is a lottery). AdamW with a
        # linear-warmup + cosine schedule tames it. The TCN is well-behaved under SGD,
        # so it keeps the original optimizer (and the original results stay comparable).
        is_transformer = getattr(self.net, "backbone", None) == "transformer"
        gradnorm = self.loss_balance == "gradnorm"
        # GradNorm's task weights (cost.loss_w) are trained by a SEPARATE optimizer, never by
        # the main one -- exclude them here.
        main_params = [p for n, p in self.cost.named_parameters()
                       if p.requires_grad and n != "loss_w"]
        if is_transformer:
            optimizer = torch.optim.AdamW(main_params, lr=self.lr, betas=(0.9, 0.999),
                                          weight_decay=1e-4)
        else:
            optimizer = torch.optim.SGD(main_params, lr=self.lr, momentum=0.9,
                                        weight_decay=1e-4)
        gn_optimizer = torch.optim.Adam([self.cost.loss_w], lr=0.025) if gradnorm else None
        self.loss_w_log = []

        loss_log = []
        val_loss_log = []
        iters_log = []
        best_val, best_state, best_iter = float("inf"), None, None   # best-checkpoint tracking

        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            
            # Epoch-granularity LR schedule (runs when the budget is given in epochs).
            # Set BEFORE the epoch's updates: self.n_epochs is the 0-based index of the
            # epoch about to run. Setting it after the epoch would leave the whole first
            # epoch at the full LR, skipping the warmup ramp entirely.
            if n_epochs is not None and n_iters is None:
                if is_transformer:
                    warmup_cosine_lr(optimizer, self.lr, self.n_epochs, n_epochs,
                                     max(1, int(WARMUP_FRAC * n_epochs)))
                else:
                    adjust_learning_rate(optimizer, self.lr, self.n_epochs, n_epochs)

            cum_loss = 0
            n_epoch_iters = 0

            interrupted = False
            for batch in train_loader:
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break

                # Iteration-granularity LR schedule. Set BEFORE optimizer.step(): self.n_iters
                # is the 0-based index of the update about to be taken. Scheduling after the
                # step would run update #1 at the full (un-warmed) LR -- the exact step the
                # Transformer ramp guards against -- and offset every later step by one.
                if n_iters is not None:
                    if is_transformer:
                        warmup_cosine_lr(optimizer, self.lr, self.n_iters, n_iters,
                                         max(1, int(WARMUP_FRAC * n_iters)))
                    else:
                        adjust_learning_rate(optimizer, self.lr, self.n_iters, n_iters)

                # The 5th item is the QUERY's participant index, used only by the
                # subject-conditional negative sampler; it is not a model input.
                x_q, x_k, x_q_s, x_k_s, pid_b = (t.to(self.device) for t in batch)
                if self.max_train_length is not None and x_q.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x_q.size(1) - self.max_train_length + 1)
                    x_q = x_q[:, window_offset : window_offset + self.max_train_length]
                    x_k = x_k[:, window_offset : window_offset + self.max_train_length]

                if gradnorm:
                    loss_val = self._gradnorm_step(x_q, x_k, optimizer, gn_optimizer)
                else:
                    optimizer.zero_grad()
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        loss = self.cost(x_q, x_k, x_q_s, x_k_s, pid=pid_b)
                    loss.backward()
                    optimizer.step()
                    loss_val = loss.item()

                cum_loss += loss_val
                n_epoch_iters += 1

                self.n_iters += 1

                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, loss_val)

            if interrupted:
                break
            
            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            iters_log.append(self.n_iters)
            if val_loader is not None:
                vl = self._validation_loss(val_loader)
                val_loss_log.append(vl)
                if vl < best_val:                       # keep the lowest-val-loss checkpoint
                    best_val = vl
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in self.net.state_dict().items()}
                    best_iter = self.n_iters
            if verbose:
                msg = f"Epoch #{self.n_epochs}: loss={cum_loss}"
                if val_loss_log:
                    msg += f"  val_loss={val_loss_log[-1]}"
                    if best_iter == self.n_iters:
                        msg += "  [best]"
                print(msg)
            self.n_epochs += 1

            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cum_loss)

        # Best-checkpoint (early stopping on the held-out SSL loss): the FINAL iterate can
        # be a poor point when the val loss oscillates (esp. absolute-PE transformers), so
        # restore the weights from the epoch with the LOWEST held-out loss. Only self.net
        # (the query encoder, used by encode() downstream) needs restoring; no-op without a
        # validation set. best_state is kept on CPU to avoid holding a second GPU copy.
        # self.net IS cost.encoder_q (same object, see __init__), so the query encoder is
        # restored inside CoSTModel too; every downstream consumer (encode/save/load here,
        # model.net.* in hrd_rhythm) reads only self.net, so the rest of the MoCo state is
        # dead once fit() returns. CAVEAT for any future resume: encoder_k, head_q/head_k,
        # the queue and the GradNorm weights are NOT rolled back, so calling fit() a second
        # time would pair a best-iterate query encoder with a final-iterate momentum encoder
        # and a queue of keys from that other trajectory point. Both entry points call fit()
        # exactly once; restore those too before adding a resume path.
        if best_state is not None:
            self.net.load_state_dict(best_state)
            if verbose:
                print(f"[best-checkpoint] restored weights from iter {best_iter} "
                      f"(val_loss={best_val:.4f})")
        self.best_val_loss = best_val if best_state is not None else None
        self.best_iter = best_iter

        self.loss_log = loss_log
        self.val_loss_log = val_loss_log        # per-epoch held-out SSL loss (or empty)
        self.iters_log = iters_log              # cumulative iterations at each logged epoch
        return loss_log

    def _validation_loss(self, val_loader):
        """Mean CoST loss on the held-out set, WITHOUT touching MoCo state or dropout.
        Under GradNorm the loss is weighted by the CURRENT task weights so the best-checkpoint
        selection is consistent with the objective actually being optimised."""
        was_training = self.cost.training
        self.cost.eval()
        gradnorm = self.loss_balance == "gradnorm"
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                # The 5th item is the QUERY's participant index, used only by the
                # subject-conditional negative sampler; it is not a model input.
                x_q, x_k, x_q_s, x_k_s, pid_b = (t.to(self.device) for t in batch)
                if self.max_train_length is not None and x_q.size(1) > self.max_train_length:
                    off = np.random.randint(x_q.size(1) - self.max_train_length + 1)
                    x_q = x_q[:, off: off + self.max_train_length]
                    x_k = x_k[:, off: off + self.max_train_length]
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    if gradnorm:
                        lt, ls = self.cost(x_q, x_k, x_q_s, x_k_s, update=False,
                                           return_parts=True, pid=pid_b)
                        loss = (self.cost.loss_w.detach() * torch.stack([lt, ls])).sum()
                    else:
                        loss = self.cost(x_q, x_k, x_q_s, x_k_s, update=False, pid=pid_b)
                total += loss.item()
                n += 1
        if was_training:
            self.cost.train()
        return total / max(n, 1)

    def _seasonal_spectral(self, z, mode):
        """Frequency-domain readout for the SEASONAL branch.

        Time-domain pooling destroys this branch by construction: the seasonal output is an
        irFFT, so its mean over a whole window is EXACTLY the f=0 (DC) coefficient -- every
        oscillation integrates to zero -- and 'last' is one arbitrary phase point on the edge.
        Either way the circadian/circaseptan content the branch exists to carry never reaches
        the probe, which is why `Season V^(S)` sits at chance while `Seasonal phase` does not.

        Instead the sequence is L2-normalised and rFFT'd (exactly as in the CoST seasonal loss,
        cost.py: convert_coeff(rfft(normalize(season)))), and amplitude/phase are read at the
        chronobiological harmonics only: 1 cycle per window (circaseptan), 1 cycle per day
        (circadian) and its 2nd-4th harmonics (12/8/6 h at a 7-day window). That keeps the
        readout compact -- |f|*d rather than the full (T/2+1)*d spectrum -- and interpretable.

        'spec_band' keeps every bin down to a two-hour period instead of five lines. Reading
        five harmonics is compact but it is not free, and the cost was measured on HRD over 24
        seeds by applying each restriction to the RAW signal and probing through an identical
        random projection:

            the full window                     0.7123
            amp+phase, bins 1..T/8              0.7118   -0.0005, 13/24
            amp+phase, bins 1..4D               0.6838   -0.0285,  9/24
            amp+phase at five harmonics         0.6622   -0.0502,  6/24
            amp+phase, every bin                0.6411   -0.0712,  4/24

        So the five-line truncation throws away 0.05 of achievable AUC, and everything down to
        a two-hour period recovers it. Keeping the WHOLE spectrum is worse than both: bins
        above that only add dimensions the penalised probe has to pay for, so there is an
        optimum rather than a monotone gain.

        'spec_band' needs a Fourier layer that actually produces those bins. The harmonic band
        layout stops at 4D (bin 31 at a 7-day window), well short of T/8 = 84, so pair it with
        --seasonal-bands single or the readout will be asking for frequencies the layer never
        writes.
        """
        T, eps = z.size(1), 1e-6
        D = max(1, T // int(self.bins_per_day))          # days spanned by the window
        if mode == "spec_band":
            f = list(range(1, max(2, T // 8)))
        else:
            f = [i for i in (1, D, 2 * D, 3 * D, 4 * D) if 0 < i <= T // 2]
        Z = fft.rfft(F.normalize(z.float(), dim=-1), dim=1)[:, f]        # (b, |f|, d) complex
        amp = torch.sqrt((Z.real + eps).pow(2) + (Z.imag + eps).pow(2))
        ang = torch.atan2(Z.imag, Z.real + eps)
        # Phase is an angle on a circle, and every consumer downstream treats readout
        # columns as ordinary numbers: persubject_rows takes a participant arithmetic
        # mean and standard deviation over their windows, RQ2 takes a Euclidean
        # distance, the probes fit a linear model. On raw angles all four are wrong
        # across the branch cut -- 23.5 h and 0.5 h are one hour apart and average to
        # 12.0, the opposite time of day -- and a depression-related phase delay is
        # exactly the thing that pushes estimates over midnight.
        #
        # Emitting (cos, sin) makes all of them correct by construction: the pair
        # lives in R^2, its arithmetic mean is the resultant vector whose direction is
        # the circular mean, and the Euclidean distance between two pairs is a
        # monotone function of the angle between them.
        pha = ((torch.cos(ang), torch.sin(ang)) if self.phase_readout == "circular"
               else (ang,))
        parts = {"spec_amp": (amp,), "spec_phase": pha,
                 "spec": (amp,) + pha, "spec_band": (amp,) + pha}[mode]
        out = torch.cat([p.reshape(p.size(0), -1) for p in parts], dim=-1)
        if mode != "spec_band":
            return out
        # Compressed to exactly the width 'spec' produces. The band has 83 bins at a 7-day
        # window against five, so on a 160-dim latent it would hand the probe 26,560 columns --
        # 53,440 after the per-participant mean and sd, for 78 training participants. The
        # ceiling that motivated this mode was measured on a random projection to 1760 dims,
        # so it is a claim about information, not about that width being usable.
        #
        # The matrix is fixed and seeded from nothing but the shape, so the trained encoder,
        # its random-init control and every ablation share one readout. A per-model matrix
        # would make the readout part of what is being compared.
        # `len(parts)` rather than a literal 2, so the target width follows the phase layout:
        # 'circular' emits (amp, cos, sin) where 'angle' emits (amp, phase).
        want = (len(parts)
                * len([i for i in (1, D, 2 * D, 3 * D, 4 * D) if 0 < i <= T // 2])
                * z.size(-1))
        key = (out.size(-1), want)
        if getattr(self, "_band_proj_key", None) != key:
            g = torch.Generator().manual_seed(20260901)
            # 1/sqrt(OUT), not 1/sqrt(in): E||Wx||^2 = out * var * ||x||^2, so only the output
            # width makes the embedding norm-preserving. Scaling by the input width shrinks
            # every distance by sqrt(out/in) -- 0.247 here -- which a penalised probe would
            # then have to undo through its C grid.
            self._band_proj = torch.randn(out.size(-1), want, generator=g) / math.sqrt(want)
            self._band_proj_key = key
        return out @ self._band_proj.to(out.device, out.dtype)

    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None, pool="mean",
                           season_pool=None):
        """Collapse the per-timestep trend/seasonal sequences into ONE vector per window.
        `pool` chooses how the whole 7-day window is summarised:
          'last'    -> last timestep only (original CoST forecasting readout);
          'mean'    -> average over all timesteps (whole-window summary, default);
          'max'     -> max over all timesteps;
          'meanmax' -> concat of mean and max (doubles the dimension).

        `season_pool` overrides the readout of the SEASONAL half only; None (default) keeps it
        identical to `pool`, so every existing result is reproduced bit for bit. Set it to
        'spec_amp' / 'spec_phase' / 'spec' to read that branch in the frequency domain instead
        (see _seasonal_spectral: time-domain pooling provably discards the rhythm).

        `mask` is forwarded to the encoder (it used to be accepted and silently dropped).
        None -- the default, and what encode() passes unless told otherwise -- means no
        masking here, since encode() puts the net in eval mode.
        """
        out_t, out_s = self.net(x.to(self.device, non_blocking=True), mask=mask)  # (b, t, d); out_s None in plain
        def collapse(z):
            if pool == "last":
                return z[:, -1]
            if pool == "mean":
                return z.mean(dim=1)
            if pool == "max":
                return z.max(dim=1).values
            if pool == "meanmax":
                return torch.cat([z.mean(dim=1), z.max(dim=1).values], dim=-1)
            if pool.startswith("seg"):
                # 'segN' -- the mean within each of N equal time segments, concatenated.
                # 'seg1' IS 'mean', so this is a strict generalisation of the default and
                # nothing that ran before behaves differently.
                #
                # Measured on GLOBEM LODO, 4 folds x 24 seeds, paired per variant against
                # the shipped readout ('mean' + a spectral seasonal half):
                #
                #   pool=seg2, season_pool=same   +0.0183 +0.0280 +0.0236 +0.0150
                #                                  20/24   20/24   18/24   19/24
                #                                  p=.0015 p=.0015 p=.0227 p=.0066
                #
                # significant in all four folds, and for the untrained control too -- so it
                # is a better readout, not evidence of anything learned. Note it is also
                # NARROWER: 4 x component_dims against the shipped 7 x, because GLOBEM's
                # 112-step window resolves only three of the harmonic bands.
                n = int(pool[3:])
                if n < 1 or n > z.size(1):
                    raise ValueError(f"pool '{pool}' wants {n} segments of a "
                                     f"{z.size(1)}-step window")
                w = z.size(1) // n
                return z[:, :n * w].reshape(z.size(0), n, w, z.size(2)).mean(dim=2)                                    .reshape(z.size(0), -1)
            raise ValueError(f"unknown pool '{pool}' (use last/mean/max/meanmax/segN)")
        # plain: single representation (out_s is None). disentangled: [trend ; seasonal].
        if out_s is None:
            out = collapse(out_t)
        else:
            season = (collapse(out_s) if season_pool is None
                      else self._seasonal_spectral(out_s, season_pool))
            out = torch.cat([collapse(out_t), season], dim=-1)
        # V^N joins the readout when the branch exists, pooled the same way as the trend
        # half. A branch trained but not read would be a cost with no effect, which is a
        # failure mode this project has already paid for once.
        v_n = self.net.encode_noise(x.to(self.device, non_blocking=True))
        if v_n is not None:
            out = torch.cat([out, collapse(v_n)], dim=-1)
        return rearrange(out.cpu(), 'b d -> b () d')
    
    def encode(self, data, mode, mask=None, encoding_window=None, casual=False, sliding_length=None, sliding_padding=0, batch_size=None, pool="mean", season_pool=None):
        if mode == 'forecasting':
            encoding_window = None
            slicing = None
        else:
            raise NotImplementedError(f"mode {mode} has not been implemented")

        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        org_training = self.net.training
        self.net.eval()
        
        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)
        
        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                if sliding_length is not None:
                    reprs = []
                    if n_samples < batch_size:
                        calc_buffer = []
                        calc_buffer_l = 0
                    for i in range(0, ts_l, sliding_length):
                        l = i - sliding_padding
                        r = i + sliding_length + (sliding_padding if not casual else 0)
                        x_sliding = torch_pad_nan(
                            x[:, max(l, 0) : min(r, ts_l)],
                            left=-l if l<0 else 0,
                            right=r-ts_l if r>ts_l else 0,
                            dim=1
                        )
                        if n_samples < batch_size:
                            if calc_buffer_l + n_samples > batch_size:
                                out = self._eval_with_pooling(
                                    torch.cat(calc_buffer, dim=0),
                                    mask,
                                    slicing=slicing,
                                    encoding_window=encoding_window
                                )
                                reprs += torch.split(out, n_samples)
                                calc_buffer = []
                                calc_buffer_l = 0
                            calc_buffer.append(x_sliding)
                            calc_buffer_l += n_samples
                        else:
                            out = self._eval_with_pooling(
                                x_sliding,
                                mask,
                                slicing=slicing,
                                encoding_window=encoding_window
                            )
                            reprs.append(out)

                    if n_samples < batch_size:
                        if calc_buffer_l > 0:
                            out = self._eval_with_pooling(
                                torch.cat(calc_buffer, dim=0),
                                mask,
                                slicing=slicing,
                                encoding_window=encoding_window
                            )
                            reprs += torch.split(out, n_samples)
                            calc_buffer = []
                            calc_buffer_l = 0
                    
                    out = torch.cat(reprs, dim=1)
                    if encoding_window == 'full_series':
                        out = F.max_pool1d(
                            out.transpose(1, 2).contiguous(),
                            kernel_size = out.size(1),
                        ).squeeze(1)
                else:
                    out = self._eval_with_pooling(x, mask, encoding_window=encoding_window, pool=pool,
                                                  season_pool=season_pool)
                    if encoding_window == 'full_series':
                        out = out.squeeze(1)
                        
                output.append(out)
                
            output = torch.cat(output, dim=0)

        self.net.train(org_training)
        return output.numpy()
    
    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.net.state_dict(), fn)
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device)
        # Checkpoints written BEFORE the seasonal layer was banded hold a single
        # full-spectrum `sfd.0`; the current encoder builds one BandedFourierLayer per
        # circadian harmonic. Loading such a file otherwise dies on a shape mismatch, which
        # silently puts every pre-banding run -- most of results_hrd/ -- out of reach of any
        # re-evaluation. Rebuilding `sfd` to the checkpoint's own layout restores exactly the
        # architecture that produced it, so the archived encoder is reproduced rather than
        # approximated. Nothing about the current default changes.
        sfd = getattr(self.net, "sfd", None)
        n_ckpt = len({k.split(".")[1] for k in state_dict if k.startswith("sfd.")})
        if sfd is not None and n_ckpt and n_ckpt != len(sfd):
            from models.encoder import BandedFourierLayer
            w = state_dict["sfd.0.weight"]                 # (num_freqs, in_ch, out_ch)
            length = 2 * (self.net.sfd[0].total_freqs - 1)
            self.net.sfd = nn.ModuleList(
                [BandedFourierLayer(int(w.shape[1]), int(w.shape[2]), b, n_ckpt, length=length)
                 for b in range(n_ckpt)]).to(self.device)
            self.net.seasonal_bands = [(m.start, m.end) for m in self.net.sfd]
            self.net.seasonal_widths = [m.out_channels for m in self.net.sfd]
            print(f"[load] checkpoint predates seasonal banding: rebuilt sfd with {n_ckpt} "
                  f"band(s) of width {int(w.shape[2])} to match it")
        self.net.load_state_dict(state_dict)


def adjust_learning_rate(optimizer, lr, epoch, epochs):
    """Decay the learning rate based on schedule"""
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


WARMUP_FRAC = 0.1   # fraction of total steps spent on linear LR warmup (Transformer)


def warmup_cosine_lr(optimizer, lr, step, total, warmup):
    """Linear warmup for `warmup` steps, then cosine decay to 0 over the remaining steps.

    `step` is the 0-based index of the update ABOUT TO BE TAKEN and `total` the planned
    total, so the caller must apply this before optimizer.step() -- applying it afterwards
    leaves the first update at the full LR, i.e. skips the one step the ramp exists for.
    Step 0 gets lr/warmup, step `warmup-1` reaches the peak `lr`, and the final step decays
    to ~0. Used for the Transformer backbone, whose SSL training is unstable under a bare
    cosine schedule; the warmup ramp removes the early-step blow-up."""
    if step < warmup:
        cur = lr * (step + 1) / max(1, warmup)
    else:
        progress = (step - warmup) / max(1, total - warmup)
        cur = lr * 0.5 * (1. + math.cos(math.pi * min(progress, 1.0)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur
