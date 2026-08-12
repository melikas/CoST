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
                 n_exact_tail=0):
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

    def __getitem__(self, item):
        ts = self.data[item % self.N]
        return self.transform(ts), self.transform(ts)

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
            return torch.cat([self.jitter(self.shift(x[..., :k])), x[..., k:]], dim=-1)
        return self.jitter(self.shift(x))

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
                 phase_mode: str = "circular"):
        super().__init__()

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


    def compute_loss(self, q, k, k_negs):
        # compute logits
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, k_negs])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

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
        B, T = z1.size(0), z1.size(1)
        z = torch.cat([z1, z2], dim=0)  # 2B x T x C
        z = z.transpose(0, 1)  # T x 2B x C
        sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B x (2B-1)
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]
        logits = -F.log_softmax(logits, dim=-1)

        i = torch.arange(B, device=z1.device)
        loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
        return loss

    def forward(self, x_q, x_k, update=True, return_parts=False):
        # `update=False` runs the loss WITHOUT mutating MoCo state (no momentum
        # update / no queue enqueue) -- used to monitor a held-out validation loss.
        # compute query features
        rand_idx = np.random.randint(0, x_q.shape[1])

        q_t, q_s = self.encoder_q(x_q)

        # --- plain contrastive SSL (no disentangler): the encoder returns a SINGLE
        #     representation (q_s is None); ONE MoCo on it. NO seasonal FFT loss, and
        #     encode() returns that single vector -- so the whole trend/seasonal split
        #     is ABSENT (not just its objective). ---
        if not self.disentangle:
            rep_q = q_t if q_s is None else q_t + q_s
            q = F.normalize(self.head_q(rep_q[:, rand_idx]), dim=-1)
            with torch.no_grad():
                if update:
                    self._momentum_update_key_encoder()
                k_t, k_s = self.encoder_k(x_k)
                rep_k = k_t if k_s is None else k_t + k_s
                k = F.normalize(self.head_k(rep_k[:, rand_idx]), dim=-1)
            loss = self.compute_loss(q, k, self.queue.clone().detach())
            if update:
                self._dequeue_and_enqueue(k)
            z = loss.new_zeros(())
            return (loss, z) if return_parts else loss

        if q_t is not None:
            q_t = F.normalize(self.head_q(q_t[:, rand_idx]), dim=-1)

        # compute key features
        with torch.no_grad():  # no gradient for keys
            if update:
                self._momentum_update_key_encoder()  # update key encoder
            k_t, k_s = self.encoder_k(x_k)
            if k_t is not None:
                k_t = F.normalize(self.head_k(k_t[:, rand_idx]), dim=-1)

        trend_loss = self.compute_loss(q_t, k_t, self.queue.clone().detach())
        if update:
            self._dequeue_and_enqueue(k_t)

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
        _, k_s = self.encoder_q(x_k)
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

        # GradNorm balances trend vs seasonal.
        if return_parts:
            return trend_loss, seasonal_loss
        return trend_loss + self.alpha * seasonal_loss

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
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        # replace keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T

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
                 time2vec_dim: int = 16,
                 loss_balance: str = "fixed",
                 bins_per_day: int = 96,
                 disentangle: bool = True,
                 jitter_sigma: float = 0.1,
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
        self.mask_mode = mask_mode
        self.mask_prob = mask_prob
        # The calendar PEs append 2 raw [tod, dow] channels read as a wall-clock index; they
        # must reach the encoder unaugmented (see PretrainDataset.n_exact_tail) -- jittering a
        # clock index would silently move the bin to a different time of day.
        self._n_exact_tail = 2 if pe in CALENDAR_PES else 0
        self.bins_per_day = int(bins_per_day)   # used by the seasonal spectral readout

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
            mask_mode=mask_mode,
            mask_prob=mask_prob,
        ).to(self.device)

        # MoCo head/queue dim: two branches (CoST) -> per-branch component_dims;
        # plain single representation (--no-disentangle) -> the full output_dims.
        moco_dim = self.net.component_dims if disentangle else self.net.output_dims

        self.cost = CoSTModel(
            self.net,
            copy.deepcopy(self.net),
            kernels=kernels,
            dim=moco_dim,
            alpha=alpha,
            K=4096,
            disentangle=disentangle,
            phase_mode=phase_mode,
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

    def fit(self, train_data, n_epochs=None, n_iters=None, valid_data=None, verbose=False):
        assert train_data.ndim == 3

        if n_iters is None and n_epochs is None:
            n_iters = 200 if train_data.size <= 100000 else 600

        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)

        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)

        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]

        # `shift` is a per-channel DC offset (moves only the MESOR / 0-frequency bin), so it
        # preserves every rhythm's phase and amplitude, and is useful (jitter alone is weak).
        multiplier = 1 if train_data.shape[0] >= self.batch_size else math.ceil(self.batch_size / train_data.shape[0])
        train_dataset = PretrainDataset(torch.from_numpy(train_data).to(torch.float), jitter_sigma=self.jitter_sigma, shift_sigma=0.5, multiplier=multiplier, n_exact_tail=self._n_exact_tail)
        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True, drop_last=True)

        # optional held-out set to monitor the SSL loss over training (no labels used)
        val_loader = None
        if valid_data is not None:
            vd = valid_data[~np.isnan(valid_data).all(axis=2).all(axis=1)]
            if len(vd) >= self.batch_size:
                val_ds = PretrainDataset(torch.from_numpy(vd).to(torch.float), jitter_sigma=self.jitter_sigma, shift_sigma=0.5, multiplier=1, n_exact_tail=self._n_exact_tail)
                val_loader = DataLoader(val_ds, batch_size=min(self.batch_size, len(val_ds)),
                                        shuffle=False, drop_last=True)

        # The Transformer backbone trains unstably under SGD (large epoch-to-epoch
        # val-loss oscillation, so the final checkpoint is a lottery). AdamW with a
        # linear-warmup + cosine schedule tames it. The TCN is well-behaved under SGD,
        # so it keeps the original optimizer (and the original results stay comparable).
        is_transformer = getattr(self.net, "backbone", None) in ("transformer", "vit")
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

                x_q, x_k = map(lambda x: x.to(self.device), batch)
                if self.max_train_length is not None and x_q.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x_q.size(1) - self.max_train_length + 1)
                    x_q = x_q[:, window_offset : window_offset + self.max_train_length]
                    x_k = x_k[:, window_offset : window_offset + self.max_train_length]

                if gradnorm:
                    loss_val = self._gradnorm_step(x_q, x_k, optimizer, gn_optimizer)
                else:
                    optimizer.zero_grad()
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        loss = self.cost(x_q, x_k)
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
                x_q, x_k = map(lambda x: x.to(self.device), batch)
                if self.max_train_length is not None and x_q.size(1) > self.max_train_length:
                    off = np.random.randint(x_q.size(1) - self.max_train_length + 1)
                    x_q = x_q[:, off: off + self.max_train_length]
                    x_k = x_k[:, off: off + self.max_train_length]
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    if gradnorm:
                        lt, ls = self.cost(x_q, x_k, update=False, return_parts=True)
                        loss = (self.cost.loss_w.detach() * torch.stack([lt, ls])).sum()
                    else:
                        loss = self.cost(x_q, x_k, update=False)
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
        """
        T, eps = z.size(1), 1e-6
        D = max(1, T // int(self.bins_per_day))          # days spanned by the window
        f = [i for i in (1, D, 2 * D, 3 * D, 4 * D) if 0 < i <= T // 2]
        Z = fft.rfft(F.normalize(z.float(), dim=-1), dim=1)[:, f]        # (b, |f|, d) complex
        amp = torch.sqrt((Z.real + eps).pow(2) + (Z.imag + eps).pow(2))
        pha = torch.atan2(Z.imag, Z.real + eps)
        parts = {"spec_amp": (amp,), "spec_phase": (pha,), "spec": (amp, pha)}[mode]
        return torch.cat([p.reshape(p.size(0), -1) for p in parts], dim=-1)

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
            raise ValueError(f"unknown pool '{pool}' (use last/mean/max/meanmax)")
        # plain: single representation (out_s is None). disentangled: [trend ; seasonal].
        if out_s is None:
            out = collapse(out_t)
        else:
            season = (collapse(out_s) if season_pool is None
                      else self._seasonal_spectral(out_s, season_pool))
            out = torch.cat([collapse(out_t), season], dim=-1)
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
