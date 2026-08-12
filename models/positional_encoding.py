"""Positional-encoding library for the CoST Transformer backbone.

This module collects the positional-encoding (PE) variants that can be swapped
into :class:`models.encoder.TransformerFeatureExtractor`, plus the ``Time2Vec``
time-embedding that is also usable with the TCN backbone.

Two families are supported:

* **Absolute** PEs are added to the token embeddings before the attention
  stack (``sinusoidal``, ``learnable``, ``tape``).
* **Attention** PEs inject position information inside every self-attention
  layer (``rpe``, ``erpe``, ``tupe``, ``convspe``, ``tpe``).
* **Time2Vec** (``time2vec``) is handled differently, faithful to Kazemi et al.
  2019: it is fed as an INPUT feature -- ``t2v(tau)`` is concatenated to the
  channels before ``input_fc`` (``x'_j = [x_j ; t2v(tau)]``), NOT added as a
  positional encoding. The concatenation happens in :class:`models.encoder.CoSTEncoder`.

The single entry points used by the encoder are :func:`add_absolute_pe`
(for the absolute family) and :class:`PETransformerEncoderLayer` (which wraps
:class:`PESelfAttention` and handles the attention family transparently).
"""
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ABSOLUTE_PES = ("sinusoidal", "learnable", "tape", "time2vec", "factorized", "circular")
# PEs anchored to the REAL wall clock: they consume the 2 raw [tod, dow] index channels that
# data_preprocessing appends under `calendar_index`. Every site that has to know about those
# channels imports this tuple, so adding a calendar PE never means editing five literals.
CALENDAR_PES = ("factorized", "circular")
ATTENTION_PES = ("rpe", "erpe", "tupe", "convspe", "tpe")
SUPPORTED_PES = ABSOLUTE_PES + ATTENTION_PES


class FactorizedCalendarPE(nn.Module):
    """Factorized calendar PE -- WavesFM (Cao et al. 2026), Eq. 2:
    ``v_t <- v_t + P_dow[dow(t)] + P_tod[tod(t)]``.

    Two learnable lookup tables indexed by each bin's REAL wall-clock, so bins from the
    same time of day / day of week share an encoding wherever they sit in the window.
    An index-anchored PE cannot do this unless every window starts at the same hour --
    which free-living recordings do not (``align_midnight=False`` here).

    Table sizes follow the bin width: ``bins_per_day`` = 96 at 15 min (the paper's 288 at
    5 min) and 7 days. The indices arrive as the trailing 2 raw channels built by
    ``data_preprocessing._calendar_index_features`` (never standardised).
    """

    def __init__(self, bins_per_day, dim):
        super().__init__()
        self.p_tod = nn.Embedding(bins_per_day, dim)
        self.p_dow = nn.Embedding(7, dim)

    def forward(self, idx):                    # idx: B x T x 2 = [tod, dow] -> B x T x dim
        # round-then-clamp, not a bare .long(): an out-of-range index is a CUDA device-side
        # assert that kills the job with no usable traceback, so never let one reach the table.
        i = idx.round().long()
        return (self.p_tod(i[..., 0].clamp(0, self.p_tod.num_embeddings - 1))
                + self.p_dow(i[..., 1].clamp(0, self.p_dow.num_embeddings - 1)))

    def tables(self):
        """``(P_tod, P_dow)`` -- the two additive calendar codes (WavesFM Fig. 13 geometry)."""
        return self.p_tod.weight, self.p_dow.weight


class CircularCalendarPE(nn.Module):
    """Circular calendar PE: ``v_t <- v_t + W [sin,cos](2*pi*tod/bpd, 2*pi*dow/7)``.

    The hand-designed counterpart of :class:`FactorizedCalendarPE` -- same wall-clock anchor,
    same additive site and width, so pairing the two isolates the ENCODER from the reference
    frame: `circular` vs `factorized` answers "does a smooth fixed circular basis suffice, or
    is a learnable per-bin table doing the work?", while both against an index PE answers
    "does wall-clock anchoring help at all?".

    Two consequences of the basis, both intended:
      * ~``4*dim`` parameters against the tables' ``(bpd + 7) * dim``;
      * adjacent bins are forced to be similar and midnight wraps continuously, whereas the
        lookup tables are free to make neighbouring bins arbitrarily different.

    Periodicity also makes it total: an out-of-range index wraps instead of indexing off the
    end of a table, so no round-then-clamp guard is needed here.
    """

    def __init__(self, bins_per_day, dim):
        super().__init__()
        # non-persistent: derived from config, must not enter (or be required by) a state_dict
        self.register_buffer("_period", torch.tensor([float(bins_per_day), 7.0]),
                             persistent=False)
        self.proj = nn.Linear(4, dim)

    def forward(self, idx):                    # idx: B x T x 2 = [tod, dow] -> B x T x dim
        ang = idx * (2 * math.pi / self._period.to(idx.dtype))
        return self.proj(torch.cat([ang.sin(), ang.cos()], dim=-1))

    @torch.no_grad()
    def tables(self):
        """``(P_tod, P_dow)`` in the same form :class:`FactorizedCalendarPE` reports.

        ``proj`` is linear, so the encoding splits additively exactly as the two lookup
        tables do. Each half is read by sweeping one field with the other pinned at 0 and
        subtracting the shared origin, which removes the bias that would otherwise be
        counted twice. Lets one geometry figure serve both calendar PEs.
        """
        w = self.proj.weight
        origin = self(w.new_zeros(1, 1, 2))

        def sweep(n, col):
            g = w.new_zeros(1, n, 2)
            g[0, :, col] = torch.arange(n, device=w.device, dtype=w.dtype)
            return (self(g) - origin)[0]

        return sweep(int(self._period[0]), 0), sweep(int(self._period[1]), 1)


# ---------------------------------------------------------------------------
# Absolute encodings (added to the embeddings)
# ---------------------------------------------------------------------------
def sinusoidal_pe(T, d, device, dtype, scale=1.0):
    """Vaswani et al. 2017 (Sec 3.3.1): PE(p,2i)=sin(p/10000^{2i/d}); cos for 2i+1."""
    pe = torch.zeros(T, d, device=device, dtype=dtype)
    pos = torch.arange(T, device=device, dtype=dtype).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=dtype)
                    * (-math.log(10000.0) / d)) * scale
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe


def tape_pe(T, d, device, dtype):
    """tAPE: length-aware sinusoidal PE (Foumani et al. 2024)."""
    return sinusoidal_pe(T, d, device, dtype, scale=d / max(T, 1))


class Time2VecPE(nn.Module):
    """Time2Vec (Kazemi et al. 2019, arXiv:1907.05321), Eq. (1).

    For a scalar notion of time ``tau`` (here ``tau = 0..T-1``, time-from-start, matching
    the paper's Appendix A convention of shifting each sequence to start at 0):
    ``t2v(tau)[0] = w_0*tau + b_0`` (linear term) and
    ``t2v(tau)[i] = sin(w_i*tau + b_i)`` for ``i >= 1``, with ``w`` (frequencies)
    and ``b`` (phase-shifts) learnable. ``d_model`` is the vector size ``k+1``
    (1 linear + ``k`` sines). Eq. (1) is applied verbatim, with no normalisation of the
    output; the initialisation (which the paper leaves free) carries the scale instead --
    see :meth:`_reset_parameters`.

    NOTE on the temporal reference frame: ``tau`` is the WITHIN-WINDOW index, so with
    ``align_midnight=False`` (free-living windows start at an arbitrary hour) a given
    ``tau`` maps to a different wall-clock time in every window. This variant can
    therefore encode ELAPSED-time structure but not circadian PHASE; that is the
    ``factorized`` calendar PE's job. Compare the two only on that understanding.

    Faithful to the paper, this is fed as an INPUT feature: the encoder
    concatenates the ``T x d_model`` output to the channels before ``input_fc``
    (``x'_j = [x_j ; t2v(tau)]``); it is NOT added as a positional encoding.

    Also per the paper, Time2Vec REPLACES the notion of time -- it is the sole time
    representation and is never combined with another time encoding. So a pe=time2vec run
    is kept sensor-only (no --with-clock-features); mixing the two would confound the
    "does Time2Vec help?" question (enforced in scripts/run.sh).
    """

    def __init__(self, d_model, max_len):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.w = nn.Parameter(torch.empty(d_model))
        self.b = nn.Parameter(torch.empty(d_model))
        self._reset_parameters()

    def _reset_parameters(self):
        """Scale is handled ENTIRELY here, so that :meth:`forward` can be Eq. (1) verbatim.

        The paper does not prescribe an initialisation -- Sec. 5.4 only establishes that the
        frequencies and phase-shifts must be LEARNED rather than fixed, which they are. The
        reference ``randn`` init is what makes a raw Eq. (1) unusable at this sequence
        length: it puts the linear term at ``w_0*tau`` up to ~T (three orders above the
        z-scored sensor channels it is concatenated with) and leaves most sines below one
        cycle per window, i.e. batch-shared near-constants. Both dominate ``input_fc`` and
        collapse the contrastive objective. Fixing that at init -- rather than normalising
        the output -- keeps ``t2v`` a pure function of ``tau``.
        """
        with torch.no_grad():
            k = self.d_model - 1
            T = max(self.max_len, 2)
            # i = 0: omega_0 = 1/(T-1), phi_0 = -0.5 -> the ramp spans exactly [-0.5, +0.5]
            # over one window. Both parameters stay free to move from there.
            self.w[0] = 1.0 / (T - 1)
            self.b[0] = -0.5
            if k > 0:
                # i >= 1: frequencies geometrically spaced from ONE cycle per window
                # (2*pi/T) up to a quarter-Nyquist ceiling of pi/2 (a 4-bin period). Below
                # 2*pi/T a sine is near-constant across a window; AT the Nyquist limit pi it
                # is degenerate, since sin(pi*n) == 0 for every integer sample n, so the
                # ceiling must stay strictly under it. The circadian component sits well
                # inside the range at omega = 2*pi/bins_per_day.
                self.w[1:] = torch.logspace(math.log10(2 * math.pi / T),
                                            math.log10(math.pi / 2), k)
                self.b[1:] = torch.zeros(k)

    def forward(self, T, device, dtype):
        # Eq. (1) verbatim, on raw tau: element 0 is the linear term w_0*tau + b_0, elements
        # 1..k are sin(w_i*tau + b_i). No post-hoc normalisation of any kind -- that is what
        # keeps t2v a pure function of tau (independent of the window's own statistics and of
        # T), which in turn preserves the paper's rescaling invariance (Prop. 1) and its
        # extrapolation behaviour (Sec. 5.2, 5.4). It also keeps omega_i in physical units
        # (rad per bin), so a learned 24 h component is readable as 2*pi/omega.
        pos = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)
        v = pos * self.w + self.b
        return torch.cat([v[:, :1], torch.sin(v[:, 1:])], dim=1).to(dtype)


# ---------------------------------------------------------------------------
# Self-attention with swappable positional encoding
# ---------------------------------------------------------------------------
class PESelfAttention(nn.Module):
    """Multi-head self-attention with a swappable positional-encoding mechanism.

    For absolute PEs (and an unrecognised name) this is vanilla scaled-dot-product
    attention; the position signal is added to the embeddings upstream. For the
    attention family the position term is injected directly into the scores.
    """

    def __init__(self, method, d_model, n_heads, max_len,
                 dropout=0.1, conv_kernel=15, spe_realizations=16):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.method = method
        self.h = n_heads
        self.dh = d_model // n_heads
        self.d_model = d_model
        self.max_len = max_len
        self.scale = 1.0 / math.sqrt(self.dh)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        if method == "rpe":
            self.rel_k = nn.Parameter(torch.empty(2 * max_len - 1, self.dh))
            nn.init.normal_(self.rel_k, std=0.02)
        elif method == "erpe":
            self.rel_bias = nn.Parameter(torch.zeros(self.h, 2 * max_len - 1))
        elif method == "tupe":
            self.register_buffer(
                "tupe_pos",
                sinusoidal_pe(max_len, d_model, torch.device("cpu"), torch.float32),
                persistent=False,
            )
            self.pq = nn.Linear(d_model, d_model, bias=False)
            self.pk = nn.Linear(d_model, d_model, bias=False)
        elif method == "convspe":
            self.R = spe_realizations
            pad = conv_kernel // 2
            self.spe_q = nn.Conv1d(self.h, self.h, conv_kernel, padding=pad, groups=self.h, bias=False)
            self.spe_k = nn.Conv1d(self.h, self.h, conv_kernel, padding=pad, groups=self.h, bias=False)
        elif method == "tpe":
            # T-PE semantic component, S(i,j) = exp(-||x_i - x_j||^2 / 2 sigma^2)
            # (Zhang et al. 2024; Irani & Metsis 2025, Sec. 3.9). sigma_0 = sqrt(d_model),
            # NOT 1: the scores are computed on LayerNorm'd tokens, so ||x||^2 ~ d_model and
            # ||x_i - x_j||^2 ~ 2*d_model for a typical pair. With sigma_0 = 1 that gives
            # S ~ exp(-d_model) ~ 0 (S is the identity matrix) AND a vanishing gradient
            # dS/dsigma = S*dist^2/sigma^3, so the semantic term could never train its way
            # out. sigma_0 = sqrt(d_model) puts 2 sigma^2 = 2 d_model, i.e. S ~ exp(-1).
            self.log_sigma = nn.Parameter(torch.full((1,), 0.5 * math.log(d_model)))
        # Cache for the deterministic eval-time ConvSPE position term (see _convspe_pos).
        # Deliberately a plain attribute, not a buffer: it is DERIVED from spe_q/spe_k, so it
        # must never enter a state_dict nor outlive a weight change. Invalidated on train()
        # and after load_state_dict.
        self._spe_eval_pos = None
        if hasattr(self, "register_load_state_dict_post_hook"):
            self.register_load_state_dict_post_hook(
                lambda module, incompatible_keys: setattr(module, "_spe_eval_pos", None))

    def train(self, mode=True):
        # Entering train mode means the weights are about to move, so the cached eval kernel
        # goes stale. Only cleared for mode=True: eval() -> eval() must reuse the cache, and
        # every train->eval transition is preceded by a train(True) that already cleared it.
        if mode:
            self._spe_eval_pos = None
        return super().train(mode)

    def _convspe_pos(self, T, device, dtype):
        """ConvSPE position term (Liutkus et al. 2021, "Relative Positional Encoding for
        Transformers with Linear Complexity").

        TRAINING keeps the paper's Monte-Carlo estimator: R fresh noise realizations filtered
        by the two depthwise convs and cross-correlated. Resampling every step IS the method
        (SPE is a stochastic PE) and is left untouched.

        EVAL uses the EXACT expectation of that same estimator, so evaluation is deterministic
        -- repeated encode() calls agree bit for bit. Writing the convs as linear maps
        ``qpe_r = Cq z_r``, ``kpe_r = Ck z_r`` with ``z ~ N(0, I)``:

            E[ (1/R) sum_r qpe_r(t) kpe_r(s) ] = (Cq Ck^T)[t, s]

        i.e. the ``R -> inf`` limit of the training estimator. That makes eval both
        deterministic AND unbiased, unlike freezing one sampled realization, which would bake
        that draw's Monte-Carlo error in permanently -- at the default R=16 that error is
        ~0.35x the size of the position term itself. Feeding the identity through the same
        convs recovers Cq/Ck exactly, including at the zero-padded edges where the kernel is
        NOT Toeplitz (in the interior it reduces to the filters' cross-correlation, which is
        what position_matrix() reports).

        Depends only on the conv weights and T -- never on the input -- so it is cached.
        """
        if self.training:
            z = torch.randn(self.R, self.h, T, device=device, dtype=dtype)
            return torch.einsum("rht,rhs->hts", self.spe_q(z), self.spe_k(z)) / self.R

        cached = self._spe_eval_pos
        if (cached is not None and cached.shape[-1] == T
                and cached.device == device and cached.dtype == dtype):
            return cached
        # realization n is the unit impulse at position n, so the conv output IS column n of
        # the linear map -- (T realizations, h, T) in, Cq/Ck out.
        eye = torch.eye(T, device=device, dtype=dtype)[:, None, :].expand(T, self.h, T).contiguous()
        pos = torch.einsum("nht,nhs->hts", self.spe_q(eye), self.spe_k(eye))
        if not torch.is_grad_enabled():   # never cache a tensor that carries an autograd graph
            self._spe_eval_pos = pos
        return pos

    def _rel_index(self, T, device):
        idx = torch.arange(T, device=device)
        rel = idx[:, None] - idx[None, :] + (self.max_len - 1)
        return rel.clamp_(0, 2 * self.max_len - 2)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.k(x).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.v(x).view(B, T, self.h, self.dh).transpose(1, 2)
        if self.method == "rpe":
            rel = self.rel_k[self._rel_index(T, x.device)]
            rel_scores = torch.einsum("bhid,ijd->bhij", q, rel)
            scores = (torch.matmul(q, k.transpose(-2, -1)) + rel_scores) * self.scale
            out = torch.matmul(self.drop(scores.softmax(-1)), v)
        elif self.method == "erpe":
            # Bias added AFTER the softmax -- intentional, and the defining feature of eRPE
            # (Foumani et al. 2024, eq. 14: alpha_i = sum_j (A_ij + w_{i-j}) x_j). It matches
            # the reference impl (ConvTran Attention_Rel_Scl: softmax, THEN += relative_bias).
            # Consequence: rows no longer sum to 1 and entries may go negative, so `attn` is
            # NOT a probability distribution. That is the mechanism, not a defect -- the
            # output splits into sum_j A_ij v_j + sum_j w_{i-j} v_j, i.e. softmax attention
            # PLUS a content-independent depthwise convolution over v with a learned Toeplitz
            # kernel w. Pre-softmax placement would squash w through the normaliser and let
            # content dilute it; the authors' stated aim is to keep the position signal
            # "sharper". w is zero-init, so this starts as exactly vanilla attention.
            # NO attention dropout here -- deliberately, to match the reference. ConvTran's
            # Attention_Rel_Scl applies dropout only in its output projection (`to_out`),
            # never to `attn + bias`. Dropping elements of that sum is not a neutral
            # regulariser: post-softmax the matrix is NOT normalised, so zeroing entries
            # rescales the learned Toeplitz bias `w` asymmetrically against the attention
            # term `A` (and, because dropout rescales survivors by 1/(1-p), inflates the
            # surviving bias values). The bias is exactly the position signal eRPE exists to
            # keep "sharper", so perturbing it here works against the mechanism.
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn = scores.softmax(-1)
            bias = self.rel_bias[:, self._rel_index(T, x.device)]
            attn = attn + bias.unsqueeze(0)
            out = torch.matmul(attn, v)
        elif self.method == "tupe":
            content = torch.matmul(q, k.transpose(-2, -1))
            pos = self.tupe_pos[:T].to(device=x.device, dtype=x.dtype)
            pq = self.pq(pos).view(T, self.h, self.dh).permute(1, 0, 2)
            pk = self.pk(pos).view(T, self.h, self.dh).permute(1, 0, 2)
            pos_scores = torch.matmul(pq, pk.transpose(-2, -1)).unsqueeze(0)
            scores = (content + pos_scores) / math.sqrt(2 * self.dh)
            out = torch.matmul(self.drop(scores.softmax(-1)), v)
        elif self.method == "convspe":
            content = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            # MC estimate while training, its exact expectation at eval -- see _convspe_pos.
            pos = self._convspe_pos(T, x.device, x.dtype)
            out = torch.matmul(self.drop((content + pos.unsqueeze(0)).softmax(-1)), v)
        elif self.method == "tpe":
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            sigma = self.log_sigma.exp().clamp_min(1e-4)
            dist2 = torch.cdist(x, x) ** 2
            S = torch.exp(-dist2 / (2 * sigma * sigma)).unsqueeze(1)
            out = torch.matmul(self.drop((scores + S).softmax(-1)), v)
        else:  # vanilla attention (absolute PEs handle position upstream)
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            out = torch.matmul(self.drop(scores.softmax(-1)), v)
        out = out.transpose(1, 2).reshape(B, T, self.d_model)
        return self.proj(out)


class PETransformerEncoderLayer(nn.Module):
    """Pre-norm Transformer block: PE-aware self-attention + feed-forward."""

    def __init__(self, method, d_model, n_heads, max_len, dropout=0.1, ff_mult=4):
        super().__init__()
        self.method = method
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = PESelfAttention(method, d_model, n_heads, max_len, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        if self.method == "tpe":
            # T-PE geometric component. The sinusoidal code is re-injected at EVERY block --
            # "applies enhanced sinusoidal positional encoding across multiple layers of the
            # transformer to maintain positional information throughout the network"
            # (Zhang et al. 2024; Irani & Metsis 2025, Sec. 3.9). Adding it once at the stack
            # input instead (which `add_absolute_pe` used to do for 'tpe') collapses T-PE's
            # geometric half onto plain `sinusoidal`, leaving only the semantic bias to
            # distinguish the two. Injected here rather than upstream so it is added exactly
            # once per block, including the first.
            x = x + sinusoidal_pe(x.size(1), x.size(-1), x.device, x.dtype).unsqueeze(0)
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Helpers used by the encoder
# ---------------------------------------------------------------------------
@torch.no_grad()
def position_matrix(net, T):
    """Data-free ``T x T`` position-interaction matrix for whatever PE ``net`` uses.

    The counterpart of Figure 13 in WavesFM (Cao et al., 2026), which compares a model's
    learned time embeddings against each other. We have no single such table, so each
    family contributes the object that plays the same role:

      * absolute (sinusoidal / tape / learnable / time2vec / tpe) -> ``cos(p_i, p_j)``.
        For ``tpe`` this is only the GEOMETRIC half; its semantic term
        ``S(i,j)=exp(-||x_i-x_j||^2/2 sigma^2)`` is data-dependent and so has no data-free
        table -- probe it with a forward pass, not here;
      * ``tupe`` -> its position-position attention term, the only attention PE whose
        position code is not purely a function of the offset;
      * ``rpe`` / ``erpe`` / ``convspe`` -> these ARE functions of ``i - j`` by
        construction, so one profile ``g(delta)`` is expanded to a Toeplitz matrix.

    Returns None when the variant carries no position code (e.g. the TCN with pe='none').

    NOTE the attention tables are allocated at ``max_len`` (2048) while a window is 672
    bins, so only the rows actually reached during training are read; ConvSPE is exactly
    zero beyond its kernel half-width, which is a real property of that encoding and not
    a truncation artefact.
    """
    pe = getattr(net, "pe", "none")
    cpu, f32 = torch.device("cpu"), torch.float32
    cos = lambda P: (lambda Q: (Q @ Q.T).numpy())(F.normalize(P.detach().cpu().float(), dim=-1))

    if pe in ("sinusoidal", "tpe"):
        return cos(sinusoidal_pe(T, net.output_dims, cpu, f32))
    if pe == "tape":
        return cos(tape_pe(T, net.output_dims, cpu, f32))
    if pe == "time2vec":
        # Build on the module's OWN device: t2v multiplies its learnable w/b by the time
        # vector, so asking for a cpu vector while the weights sit on cuda raises
        # "expected all tensors to be on the same device". `cos` moves the result to cpu.
        return cos(net.t2v(T, next(net.t2v.parameters()).device, f32))

    fe = getattr(net, "feature_extractor", None)
    if pe == "learnable" and getattr(fe, "lpe", None) is not None:
        return cos(fe.lpe.weight[:T])
    if not getattr(fe, "layers", None):
        return None
    attn = fe.layers[0].attn
    delta = np.arange(T)[:, None] - np.arange(T)[None, :]

    if pe == "tupe":
        p = attn.tupe_pos[:T].to(attn.pq.weight.device, attn.pq.weight.dtype)
        q = attn.pq(p).view(T, attn.h, attn.dh)
        k = attn.pk(p).view(T, attn.h, attn.dh)
        return (torch.einsum("ihd,jhd->ij", q, k).float().cpu().numpy()
                / (attn.h * math.sqrt(attn.dh)))
    if pe == "rpe":                       # how the offset-delta code compares to offset 0
        r = F.normalize(attn.rel_k.detach().cpu().float(), dim=-1)
        g, centre = (r @ r[attn.max_len - 1]).numpy(), attn.max_len - 1
    elif pe == "erpe":                    # the learned post-softmax bias itself
        g, centre = attn.rel_bias.detach().cpu().float().mean(0).numpy(), attn.max_len - 1
    elif pe == "convspe":                 # induced kernel = cross-correlation of the filters
        wq = attn.spe_q.weight.detach().cpu().float().squeeze(1).numpy()
        wk = attn.spe_k.weight.detach().cpu().float().squeeze(1).numpy()
        g = np.mean([np.correlate(b, a, "full") for a, b in zip(wq, wk)], axis=0)
        centre = wq.shape[1] - 1
    else:
        return None
    idx = delta + centre
    return np.where((idx >= 0) & (idx < len(g)), g[np.clip(idx, 0, len(g) - 1)], 0.0)


def add_absolute_pe(x, pe, d_model, learnable_pe=None):
    """Add the absolute position signal for ``pe`` to ``x`` (B, T, d_model).

    Attention-family PEs add nothing here (they act inside attention). ``tpe`` also adds
    nothing here: its geometric (sinusoidal) component is re-injected inside EVERY
    :class:`PETransformerEncoderLayer`, which is what distinguishes T-PE from plain
    ``sinusoidal`` -- doing it here as well would double it at the first block.
    ``time2vec`` adds nothing here either: it is fed as an input feature upstream
    (concatenated before ``input_fc``, per Kazemi et al. 2019).
    """
    T = x.size(1)
    if pe == "sinusoidal":
        return x + sinusoidal_pe(T, d_model, x.device, x.dtype).unsqueeze(0)
    if pe == "tape":
        return x + tape_pe(T, d_model, x.device, x.dtype).unsqueeze(0)
    if pe == "learnable":
        pos = torch.arange(T, device=x.device)
        return x + learnable_pe(pos).to(x.dtype).unsqueeze(0)
    return x
