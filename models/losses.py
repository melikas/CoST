"""The SSL objective: a MoCo trend term and a within-batch seasonal term, with the
seasonal term split into amplitude and phase and each term separately weighted.

    L  =  w_trend * L_trend  +  alpha * ( w_amp * L_amp  +  w_phase * L_phase )  [+ w_eq * L_eq]
                                                                              [+ w_ac * L_ac]

Upstream fixes w_trend = 1 and w_amp = w_phase = 1/2. Making the three weights explicit
is the only structural change here, and it is what lets the objective be contracted onto
the phase term.

WHY CONTRACT ONTO PHASE. RQ2 measures how well each block of the representation supports
a personal rhythm baseline, as a concordance C (0.5 = chance). Read block by block on run
2224103, 24 seeds:

    V^S phase   C = 0.8802      <- carries essentially all of it; ALONE it beats the
    V^S amp     C = 0.5661         full representation (0.8743), so the other two blocks
    V^T trend   C = 0.6054         are, on this task, dilution

The three blocks are trained by three terms of one sum, and the sum weights them 1 : 1/2 :
1/2 in exactly the wrong order. `CONTRACTED` re-allocates the weights in proportion to the
signal each block was measured to carry, using C - 0.5 (concordance above chance) as the
signal:

    phase 0.3802,  trend 0.1054,  amp 0.0661     ->  normalised to phase = 1:
    w_phase = 1.000,  w_trend = 0.277,  w_amp = 0.174

The seasonal pair is then renormalised to sum to 1, so `alpha` keeps its old meaning as
the seasonal-vs-trend scale and the contraction is a pure re-allocation rather than a
change in total seasonal magnitude.

    w_amp = 0.174 / 1.174 = 0.148        w_phase = 1.000 / 1.174 = 0.852

THIS IS A BET, NOT A CORRECTION, AND IT IS THE ONLY ONE HERE. The depth and band changes
in models/encoder.py fix defects that are wrong under any hypothesis. This re-weighting is inferred
from RQ2 concordances that were themselves measured under one configuration, so `PAPER`
reproduces run 2224103's objective bit-for-bit and must be run as the paired control.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import fft


@dataclass(frozen=True)
class TermWeights:
    """Relative weights of the three objective terms. See the module docstring."""
    trend: float = 1.0
    amp: float = 0.5
    phase: float = 0.5

    def normalised(self) -> "TermWeights":
        """Seasonal pair rescaled to sum to 1, so `alpha` means the same across presets."""
        s = self.amp + self.phase
        if s <= 0:
            raise ValueError("seasonal weights must not both be zero")
        return TermWeights(self.trend, self.amp / s, self.phase / s)


#: Run 2224103's objective, bit-for-bit: trend + alpha * (L_amp + L_phase) / 2.
PAPER = TermWeights(trend=1.0, amp=0.5, phase=0.5)


#: Contracted onto phase, weights derived from measured RQ2 concordance (see docstring).
CONTRACTED = TermWeights(trend=0.277, amp=0.174, phase=1.0).normalised()


def convert_coeff(x, eps: float = 1e-6):
    """Complex rFFT coefficients -> (amplitude, phase)."""
    amp = torch.sqrt((x.real + eps).pow(2) + (x.imag + eps).pow(2))
    phase = torch.atan2(x.imag, x.real + eps)
    return amp, phase


def circular_phase(phase, amp=None, eps: float = 1e-12):
    """Unit-circle embedding of an angle, phi -> [sin phi ; cos phi], optionally
    amplitude-weighted per channel.

    WHY. `instance_contrastive_loss` scores pairs with a dot product over channels. On raw
    atan2 output that is not a similarity between angles: two IDENTICAL phases score 0 at
    phi=0 but pi^2 at phi=pi, and (pi-eps, -pi+eps) -- the same angle either side of the
    branch cut -- scores the most negative value possible. After the embedding the dot
    product is sum_c cos(phi_i,c - phi_j,c): a function of the angular gap alone.

    WHY WEIGHT. Unweighted, a channel whose amplitude is ~0 -- where the phase is undefined
    noise -- counts as much as a strong rhythm. Weighting by amplitude turns the score into
    sum_c w_i,c * w_j,c * cos(dphi), an amplitude-weighted phase coherence.

    WHY RMS AND NOT L2. The loss applies log_softmax directly to these dot products with no
    temperature, so the embedding's norm sets the effective softmax temperature. Unweighted,
    ||emb|| = sqrt(C); under L2 weights it would collapse to 1, a sqrt(C)-fold logit shrink
    that starves the phase branch of gradient. Normalising so mean(w^2) = 1 keeps
    ||emb|| = sqrt(C) exactly, so this changes only the RELATIVE weighting of channels --
    with equal amplitudes w == 1 and it reduces to the unweighted embedding bit for bit.

    `amp` is detached: it says how much to trust a channel's phase, and letting the phase
    term push on it would let the model lower its loss by shrinking amplitudes instead of
    aligning phases.
    """
    s, c = phase.sin(), phase.cos()
    if amp is not None:
        w = amp.detach()
        w = w / w.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)
        s, c = s * w, c * w
    return torch.cat([s, c], dim=-1)


def instance_contrastive_loss(z1, z2):
    """Within-batch instance discrimination (TS2Vec): positives are the two views of one
    window at one timestep, negatives are every other window in the batch."""
    b = z1.size(0)
    z = torch.cat([z1, z2], dim=0).transpose(0, 1)          # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))                # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits = logits + torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    i = torch.arange(b, device=z1.device)
    return (logits[:, i, b + i - 1].mean() + logits[:, b + i, i].mean()) / 2


def moco_ce_loss(q, k, k_negs, temperature: float = 0.07, neg_mask=None):
    """MoCo InfoNCE for the trend branch, plus the top-1 retrieval rate.

    `top1` is the project's pretext-difficulty measure and is returned rather than logged
    so it can never drift from the loss it describes. Chance is 1/(1 + n_negatives); at
    top1 ~ 1.0 the task is already solved at initialisation and its gradient teaches
    nothing, which is the measured state of the shipped window-pairing (0.8223, 6737x
    chance) and the reason the trend branch has never separated from its own control.

    `neg_mask` (N x K, True = not a negative) drops queue entries that are the query's own
    instance: a queue larger than the training set always holds some.
    """
    l_pos = torch.einsum("nc,nc->n", q, k).unsqueeze(-1)                    # N x 1
    l_neg = (torch.einsum("nc,nkc->nk", q, k_negs) if k_negs.dim() == 3
             else torch.einsum("nc,ck->nk", q, k_negs))                     # N x K
    if neg_mask is not None:
        l_neg = l_neg.masked_fill(neg_mask, float("-inf"))
    logits = torch.cat([l_pos, l_neg], dim=1) / temperature
    with torch.no_grad():
        top1 = float((l_pos > l_neg.max(dim=1, keepdim=True).values).float().mean())
    # Cross-entropy with every positive at column 0, written without NLLLoss: CUDA NLLLoss
    # raises under torch.use_deterministic_algorithms(True), which the runner enables.
    return -F.log_softmax(logits, dim=1)[:, 0].mean(), top1


def seasonal_loss(q_s, k_s, weights: TermWeights, phase_mode: str = "circular_amp"):
    """Frequency-domain seasonal term, returned as its two weighted halves.

    Both views carry gradients and the key comes from the query encoder, not an EMA copy:
    this branch has no queue, so there are no stale keys to keep consistent, and the loss
    is symmetric in its arguments -- detaching one side would zero half the gradient path.
    That asymmetry with the trend branch is intentional and matches upstream.
    """
    if phase_mode not in ("raw", "circular", "circular_amp"):
        raise ValueError(f"unknown phase_mode: {phase_mode!r}")
    q_s = F.normalize(q_s, dim=-1)
    k_s = F.normalize(k_s, dim=-1)
    with torch.autocast(device_type="cuda", enabled=False):
        q_amp, q_pha = convert_coeff(fft.rfft(q_s.float(), dim=1))
        k_amp, k_pha = convert_coeff(fft.rfft(k_s.float(), dim=1))
        if phase_mode != "raw":
            w = phase_mode == "circular_amp"
            q_pha = circular_phase(q_pha, q_amp if w else None)
            k_pha = circular_phase(k_pha, k_amp if w else None)
    return (weights.amp * instance_contrastive_loss(q_amp, k_amp),
            weights.phase * instance_contrastive_loss(q_pha, k_pha))


def equivariance_loss(pred, delta):
    """Level equivariance for the trend branch (Dangovski et al., ICLR 2022).

    `shift` adds a per-channel offset -- the MESOR, the f=0 bin -- to each view, and the trend
    branch is read out as its time-mean, i.e. that same content. Contrasting the two views
    therefore trains the trend branch to DISCARD the quantity it is read out as: an
    augmentation defines what the representation throws away (Xiao et al., ICLR 2021). This
    term makes the branch predict the offset between the views instead. `pred` is a bias-free
    linear map of the difference of the two views' time-mean trend, and `delta` is d1 - d2,
    so at the optimum level is a linear direction of the trend readout.
    """
    return F.mse_loss(pred, delta)


def anticollapse_loss(cur, mem, gamma: float, mu: float = 25.0, nu: float = 1.0,
                      eps: float = 1e-4):
    """VICReg variance and covariance terms (Bardes, Ponce & LeCun, ICLR 2022) on the seasonal
    readout's LOG amplitudes, one frequency bin at a time.

    `cur` is B x F x C with gradient, `mem` M x F x C detached rows from earlier batches (or
    None). Statistics are taken over both, because one batch of 64 cannot estimate a full-rank
    covariance of 160 channels; only `cur` carries gradient. Per bin: every channel's standard
    deviation across windows is held above `gamma` (the variance floor), and the channels are
    decorrelated (the covariance term) -- which is what the readout's effective rank measures.
    The loss depends on |Z| only, so its gradient with respect to every phase is exactly zero.
    """
    x = cur if mem is None or len(mem) == 0 else torch.cat([cur, mem], dim=0)
    x = x - x.mean(dim=0, keepdim=True)
    n, _, c = x.shape
    var = F.relu(gamma - torch.sqrt(x.var(dim=0) + eps)).mean()
    cov = torch.einsum("nfc,nfd->fcd", x, x) / (n - 1)
    off = cov - torch.diag_embed(torch.diagonal(cov, dim1=1, dim2=2))
    return mu * var + nu * off.pow(2).sum(dim=(1, 2)).div(c).mean()


def total_loss(trend_term, amp_term, phase_term, weights: TermWeights, alpha: float):
    """L = w_trend * L_trend + alpha * (w_amp * L_amp + w_phase * L_phase).

    `amp_term` and `phase_term` arrive already weighted, from `seasonal_loss`.
    """
    return weights.trend * trend_term + alpha * (amp_term + phase_term)
