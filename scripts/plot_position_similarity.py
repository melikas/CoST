#!/usr/bin/env python
"""Idea B -- position-position similarity heatmaps for every positional encoding.

For each encoding we build an ``L x L`` matrix ``S_{ij}`` and render it as a
heatmap so the reader *sees* the temporal prior the encoding carries
(taxonomy doc, sec. 12, "Idea B"; mirrors Fig. 1 of Li et al. 2021 and Fig. 2
of the survey):

  * sinusoidal / tAPE          -> smooth decay off the diagonal
  * learnable PE               -> diffuse / noisy (memorised per index)
  * RPE / eRPE / ConvSPE       -> banded (Toeplitz), translation-invariant
  * TUPE                       -> untied absolute position-position score
  * Time2Vec / LFF (M-dim)     -> repeating diagonal bands at the (learned)
                                  circadian period -- the visual signature of
                                  rhythmicity

Faithfulness note (this is the whole point of the script)
---------------------------------------------------------
``S_{ij} = P_i . P_j`` is only *directly* defined for encodings that give every
position a code vector ``P_i`` (the absolute / input families). The
attention-family PEs never form a per-position ``P_i``; they inject position as
a bias on the ``Q K^T`` score. So for each family we compute the position-
position matrix in the way that faithfully reflects **how that exact module in
this repo injects position** -- never a hand-wavy stand-in:

  vector codes (sinusoidal, tape, learnable, time2vec, lff)
      S = P P^T with P the ACTUAL code from the module
      (``sinusoidal_pe`` / ``tape_pe`` / ``nn.Embedding`` / ``Time2VecPE`` /
       ``LearnableFourierMultiDim``).
  rpe   S_{ij} = a_{i-j} . a_0   (autocorrelation of the learned relative-key
        embeddings ``PESelfAttention.rel_k`` across lags; Toeplitz by
        construction -- the content-free part of the Shaw-style term q.a_{i-j}).
  erpe  S_{ij} = mean_h rel_bias[h, i-j]   (exactly the post-softmax additive
        bias ``PESelfAttention.rel_bias``; Toeplitz).
  convspe S_{ij} = mean_h E_z[ (spe_q z)_i . (spe_k z)_j ] / R   (the ConvSPE
        stochastic positional kernel, estimated with R realisations -- exactly
        the ``pos`` term the module adds to the scores).
  tupe  S_{ij} = mean_h (p^Q_i . p^K_j) / sqrt(2 d_h)   (the untied positional
        score ``pq(pos) . pk(pos)`` the module adds; ``pos`` is its sinusoidal
        buffer).
  tpe   S = P P^T for the SINUSOIDAL anchor T-PE adds upstream; its in-attention
        Gaussian ``exp(-||x_i-x_j||^2/2 sigma^2)`` is content-dependent and
        cannot be drawn without data (noted in the panel title).

All matrices are computed under the model's real hyper-parameters
(``--length`` = window-hours*60/bin-minutes, ``--bins-per-day`` = 24*60/bin,
``--repr-dims``, ``--time2vec-dim``) at INITIALISATION, seeded for
reproducibility. train_hrd.py does not checkpoint the model, so init is what is
available by default; pass ``--ckpt path/to/net_state_dict.pt`` (from
``cost.CoST.save``) to substitute the *trained* parameters of whichever PE that
run used, or ``--illustrative`` to set the learnable-frequency encodings
(Time2Vec / LFF / eRPE) to the circadian spectrum so the "learned" bands are
visible without training (clearly labelled in the panel titles).

Examples
--------
    # faithful init-time priors at the HRD default geometry (168 h window, 15 min bins)
    python scripts/plot_position_similarity.py

    # teaching figure: learnable-frequency PEs pinned to 24 h / 12 h / 168 h
    python scripts/plot_position_similarity.py --illustrative --mark-period

    # trained Time2Vec kernel from a saved checkpoint
    python scripts/plot_position_similarity.py --pes time2vec --ckpt run/net.pt
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Blue (+) / orange (-) diverging colormap for the encoding-matrix heatmaps.
BLUE_ORANGE = LinearSegmentedColormap.from_list(
    "blue_orange",
    ["#7f2704", "#e6550d", "#fdae6b", "#ffffff", "#9ecae1", "#4292c6", "#08306b"],
)

# --- make the repo importable so we use the SAME modules the model trains with -
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.positional_encoding import (          # noqa: E402
    sinusoidal_pe,
    tape_pe,
    Time2VecPE,
    PESelfAttention,
)
from models.encoder import LearnableFourierMultiDim  # noqa: E402


# One-line "how the matrix is derived" tag per PE, shown under each panel title.
FAMILY = {
    "sinusoidal": "additive absolute  |  S=P.P^T (fixed sinusoid)",
    "tape":       "additive absolute  |  S=P.P^T (length-aware sinusoid)",
    "learnable":  "additive absolute  |  S=P.P^T (nn.Embedding)",
    "time2vec":   "input-concat clock |  S=P.P^T, P=t2v(tau)",
    "lff":        "additive intrinsic |  S=P.P^T, P=LFF[within-day, day]",
    "rpe":        "attention bias      |  S_ij=a_(i-j).a_0",
    "erpe":       "attention bias      |  S_ij=bias[i-j] (post-softmax)",
    "convspe":    "attention bias      |  S=E[q^pe_i.k^pe_j] (conv kernel)",
    "tupe":       "attention bias      |  S=p^Q_i.p^K_j (untied)",
    "tpe":        "hybrid              |  S=P.P^T (sinusoid anchor; +content Gauss.)",
}
DEFAULT_PES = ["sinusoidal", "tape", "learnable", "time2vec", "lff",
               "rpe", "erpe", "convspe", "tupe", "tpe"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def gram(P: torch.Tensor) -> torch.Tensor:
    """S_ij = P_i . P_j  for a code matrix P of shape (L, d)."""
    return P @ P.t()


def time_positions_2d(L: int, bins_per_day: int) -> torch.Tensor:
    """Metric 2-D time position [within-day phase, day-of-week], each in [0,1).

    Exact copy of ``ViT2DFeatureExtractor._time_positions`` (models/encoder.py)
    so the LFF panel sees precisely the positions the model feeds it.
    """
    t = torch.arange(L, dtype=torch.float32)
    bpd = float(bins_per_day)
    within_day = torch.remainder(t, bpd) / bpd
    n_days = max(1.0, L / bpd)
    day = torch.div(t, bpd, rounding_mode="floor") / n_days
    return torch.stack([within_day, day], dim=-1)          # (L, 2)


def toeplitz_from_lags(g: torch.Tensor, L: int) -> torch.Tensor:
    """Build S_ij = g(i-j) from a lag function g indexed by delta+ (L-1)."""
    idx = torch.arange(L)
    lag = idx[:, None] - idx[None, :] + (L - 1)            # in [0, 2L-2]
    return g[lag]


def find_param(state: dict, suffix: str, layer: int = 0):
    """Return the checkpoint tensor whose key ends with ``suffix`` (preferring
    ``.layers.{layer}.`` for the per-layer attention params), else None."""
    cands = [(k, v) for k, v in state.items() if k.endswith(suffix)]
    if not cands:
        return None
    for k, v in cands:
        if f".layers.{layer}." in k:
            return v
    return cands[0][1]


def _load_state(path):
    st = torch.load(path, map_location="cpu")
    return st.get("state_dict", st) if isinstance(st, dict) else st


def state_for_pe(pe: str, args, global_state):
    """Resolve the trained weights for one PE. With ``--ckpt-dir`` each PE gets
    its OWN run's net.pt (folder ``*_{pe}_seed*/net.pt``); with ``--ckpt`` the one
    given state is used; otherwise None (init)."""
    ckpt_dir = getattr(args, "ckpt_dir", None)
    if ckpt_dir:
        hits = sorted(Path(ckpt_dir).glob(f"**/*_{pe}_seed*/net.pt"))
        if hits:
            print(f"  [ckpt] {pe} <- {hits[0]}", flush=True)
            return _load_state(str(hits[0]))
        print(f"  [ckpt] {pe}: no net.pt found under {ckpt_dir} (using init)", flush=True)
        return None
    return global_state


def infer_d(state, default_d):
    """Model width d from a checkpoint (backbones differ: TCN=320, Transformer=240),
    so a trained state loads into a matching-size module. Falls back to default."""
    if state is not None:
        w = find_param(state, "input_proj.weight")     # (output_dims, hidden_dims)
        if w is not None:
            return int(w.shape[0])
        n = find_param(state, "feature_extractor.norm.weight")
        if n is not None:
            return int(n.shape[0])
    return default_d


def infer_t2v_dim(state, default):
    """Time2Vec vector size (k+1) from a checkpoint (t2v.w), else default."""
    if state is not None:
        w = find_param(state, "t2v.w")
        if w is not None:
            return int(w.shape[0])
    return default


# ---------------------------------------------------------------------------
# per-position code builders, shared by build_matrix (S = P P^T) and
# build_encoding_map (P itself) so the checkpoint-loading and --illustrative
# rules live in exactly one place per encoding.
# ---------------------------------------------------------------------------
def _lpe_code(cfg, state, d, n) -> torch.Tensor:
    """Learnable PE code for the first ``n`` positions (trained table if available)."""
    w = find_param(state, "lpe.weight") if state is not None else None
    ml = int(w.shape[0]) if w is not None else cfg.max_len   # model uses max_len, not L
    emb = torch.nn.Embedding(ml, d)                          # default N(0,1) init
    if w is not None:
        emb.weight.copy_(w)
    return emb(torch.arange(n))


def _t2v_code(cfg, state, dim, n) -> torch.Tensor:
    """Time2Vec code t2v(tau) for the first ``n`` steps.

    ``--illustrative`` pins the sine periods to the circadian set (24 h / 12 h /
    168 h / ...) so the rhythmic bands are visible without training; channel 0
    stays the Time2Vec linear term (Kazemi et al. Eq. 1).
    """
    t2v = Time2VecPE(dim, cfg.max_len)
    if state is not None:
        w, b = find_param(state, "t2v.w"), find_param(state, "t2v.b")
        if w is not None and b is not None:
            t2v.w.copy_(w); t2v.b.copy_(b)
    if cfg.illustrative:
        periods = _circadian_periods_steps(cfg.bins_per_day)
        w, b = t2v.w.clone(), t2v.b.clone().zero_()
        w[0] = 1.0 / max(cfg.max_len, 1)
        for i in range(1, dim):
            w[i] = 2 * math.pi / periods[(i - 1) % len(periods)]
        t2v.w.copy_(w); t2v.b.copy_(b)
    return t2v(n, torch.device("cpu"), torch.float32)


def _lff_module(cfg, state, d) -> LearnableFourierMultiDim:
    """Learnable Fourier Features module, with the trained Wr/MLP if available."""
    module = LearnableFourierMultiDim(d, n_dims=2, n_freqs=cfg.lf_freqs, gamma=cfg.lf_gamma)
    if state is not None:
        wr = find_param(state, "time_pe.Wr")
        if wr is not None:
            module.Wr.copy_(wr)
            for i in (0, 2):                                 # mlp.0 / mlp.2 Linear
                wk = find_param(state, f"time_pe.mlp.{i}.weight")
                bk = find_param(state, f"time_pe.mlp.{i}.bias")
                if wk is not None:
                    module.mlp[i].weight.copy_(wk)
                    module.mlp[i].bias.copy_(bk)
    return module


# ---------------------------------------------------------------------------
# per-PE matrix builders (each returns an (L, L) numpy array)
# ---------------------------------------------------------------------------
@torch.no_grad()
def build_matrix(pe: str, cfg, state: dict | None) -> np.ndarray:
    L = cfg.length
    d = infer_d(state, cfg.repr_dims)                  # match the checkpoint's backbone width
    t2v_dim = infer_t2v_dim(state, cfg.time2vec_dim)
    cpu = torch.device("cpu")
    f32 = torch.float32

    # ---- absolute vector codes -------------------------------------------
    if pe == "sinusoidal":
        return gram(sinusoidal_pe(L, d, cpu, f32)).numpy()

    if pe == "tape":
        return gram(tape_pe(L, d, cpu, f32)).numpy()

    if pe == "learnable":
        return gram(_lpe_code(cfg, state, d, L)).numpy()

    # ---- metric clock: Time2Vec ------------------------------------------
    if pe == "time2vec":
        return gram(_t2v_code(cfg, state, t2v_dim, L)).numpy()

    # ---- metric multi-dim Learnable Fourier Features ----------------------
    if pe == "lff":
        module = _lff_module(cfg, state, d)
        pos = time_positions_2d(L, cfg.bins_per_day)       # (L, 2)
        if cfg.illustrative:
            # Show the paper's shift-invariant kernel r_i.r_j directly with Wr
            # rows aligned to daily / semidiurnal / weekly axes (labelled illustrative).
            n_days = max(1.0, L / cfg.bins_per_day)
            rows = [[2 * math.pi, 0.0], [4 * math.pi, 0.0], [6 * math.pi, 0.0],
                    [0.0, 2 * math.pi * n_days], [0.0, 2 * math.pi * n_days / 7]]
            Wr = torch.tensor(rows, dtype=torch.float32)
            proj = pos @ Wr.t()
            r = torch.cat([proj.cos(), proj.sin()], dim=-1) / math.sqrt(Wr.shape[0])
            return gram(r).numpy()                         # raw Fourier kernel
        P = module(pos)                                    # (L, d) injected code
        return gram(P).numpy()

    # ---- attention-family: need a real PESelfAttention module -------------
    attn = PESelfAttention(pe if pe != "tpe" else "tpe", d, cfg.n_heads, cfg.max_len)

    if pe == "rpe":
        a = attn.rel_k                                     # (2*max_len-1, d_h)
        if state is not None:
            w = find_param(state, "attn.rel_k", cfg.layer)
            if w is not None:
                a = w
        off0 = (a.shape[0] - 1) // 2                       # zero lag (max_len may be 2048)
        a0 = a[off0]                                       # zero-lag embedding
        g_full = (a * a0).sum(-1)                          # a_delta . a_0 over all lags
        lags = torch.arange(-(L - 1), L) + off0
        g = g_full[lags]                                   # (2L-1,) indexed delta+(L-1)
        return toeplitz_from_lags(g, L).numpy()

    if pe == "erpe":
        bias = attn.rel_bias                               # (h, 2*max_len-1), zeros at init
        if state is not None:
            w = find_param(state, "attn.rel_bias", cfg.layer)
            if w is not None:
                bias = w
        if cfg.illustrative and float(bias.abs().max()) < 1e-8:
            bias = _illustrative_erpe_bias(cfg)            # decaying + circadian Toeplitz
        off0 = (bias.shape[1] - 1) // 2
        lags = torch.arange(-(L - 1), L) + off0
        g = bias[:, lags].mean(0)                          # mean over heads -> (2L-1,)
        return toeplitz_from_lags(g, L).numpy()

    if pe == "convspe":
        if state is not None:
            for nm in ("spe_q.weight", "spe_k.weight"):
                w = find_param(state, f"attn.{nm}", cfg.layer)
                if w is not None:
                    dict(attn.named_parameters())[
                        f"spe_{'q' if 'q' in nm else 'k'}.weight"].copy_(w)
        torch.manual_seed(cfg.seed)                        # reproducible realisations
        R = cfg.spe_realizations
        z = torch.randn(R, cfg.n_heads, L)
        qpe = attn.spe_q(z)                                # (R, h, L)
        kpe = attn.spe_k(z)
        pos = torch.einsum("rht,rhs->hts", qpe, kpe) / R   # (h, L, L) kernel estimate
        return pos.mean(0).numpy()

    if pe == "tupe":
        posv = attn.tupe_pos[:L]                           # (L, d) sinusoidal buffer
        if state is not None:
            for nm in ("pq.weight", "pk.weight"):
                w = find_param(state, f"attn.{nm}", cfg.layer)
                if w is not None:
                    dict(attn.named_parameters())[nm].copy_(w)
        pq = attn.pq(posv).view(L, cfg.n_heads, attn.dh).permute(1, 0, 2)
        pk = attn.pk(posv).view(L, cfg.n_heads, attn.dh).permute(1, 0, 2)
        scores = torch.matmul(pq, pk.transpose(-2, -1))    # (h, L, L)
        return (scores.mean(0) / math.sqrt(2 * attn.dh)).numpy()

    if pe == "tpe":
        # Only the sinusoidal anchor is position-only; the Gaussian is content-based.
        return gram(sinusoidal_pe(L, d, cpu, f32)).numpy()

    raise ValueError(f"unknown pe: {pe}")


def _circadian_periods_steps(bins_per_day: int):
    """Periods (in timesteps) at 24 h, 12 h, 168 h, 8 h, 6 h given bins/day."""
    bpd = float(bins_per_day)
    return [bpd, bpd / 2, 7 * bpd, bpd / 3, bpd / 4, bpd / 6]


def _illustrative_erpe_bias(cfg) -> torch.Tensor:
    """A representative *learned-looking* eRPE Toeplitz bias (labelled illustrative):
    local sharpening exp(-|d|/tau) plus a circadian band cos(2 pi d / bpd)."""
    lags = torch.arange(-(cfg.max_len - 1), cfg.max_len).float()
    tau = cfg.bins_per_day / 2.0
    local = torch.exp(-lags.abs() / tau)
    band = 0.4 * torch.cos(2 * math.pi * lags / cfg.bins_per_day) * \
        torch.exp(-lags.abs() / (4 * cfg.bins_per_day))
    g = local + band
    return g.unsqueeze(0).repeat(cfg.n_heads, 1)           # (h, 2*max_len-1)


# ---------------------------------------------------------------------------
# Encoding-matrix heatmaps: the raw code P[position, dim] each PE adds to the
# tokens, plotted like the classic sinusoidal-PE heatmap (position on y,
# embedding dimension on x). Only the additive/vector family has a genuine
# per-position code; relative/attention PEs have none, so their nearest real
# object (relative-offset embedding / bias / conv filter) is shown with a
# relabelled y-axis and a note. Returns (M, ylabel, xlabel, note, is_offset).
# ---------------------------------------------------------------------------
@torch.no_grad()
def build_encoding_map(pe: str, cfg, state: dict | None):
    P = min(cfg.enc_positions, cfg.length)
    d = infer_d(state, cfg.repr_dims)                  # match the checkpoint's backbone width
    t2v_dim = infer_t2v_dim(state, cfg.time2vec_dim)
    cpu, f32 = torch.device("cpu"), torch.float32
    ypos, xdim = "Position (timestep)", "Embedding dimension"

    if pe == "sinusoidal":
        return sinusoidal_pe(P, d, cpu, f32).numpy(), ypos, xdim, "P[pos,dim] = fixed sinusoid", False
    if pe == "tape":
        return tape_pe(P, d, cpu, f32).numpy(), ypos, xdim, "length-aware sinusoid (scale d/L)", False
    if pe == "learnable":
        return (_lpe_code(cfg, state, d, P).numpy(), ypos, xdim,
                "nn.Embedding (random at init)", False)
    if pe == "time2vec":
        return (_t2v_code(cfg, state, t2v_dim, P).numpy(), ypos,
                "Time2Vec dim (0 = linear, 1+ = sines)", "t2v(tau), concatenated to input", False)
    if pe == "lff":
        pos = time_positions_2d(P, cfg.bins_per_day)
        return (_lff_module(cfg, state, d)(pos).numpy(), ypos, xdim,
                "LFF[within-day, day-of-week] code", False)
    if pe == "tpe":
        return sinusoidal_pe(P, d, cpu, f32).numpy(), ypos, xdim, "sinusoidal anchor (+ content Gaussian, not shown)", False

    attn = PESelfAttention(pe if pe != "tpe" else "tpe", d, cfg.n_heads, cfg.max_len)

    if pe == "tupe":
        posv = attn.tupe_pos[:P]                          # sinusoidal buffer (regenerated, deterministic)
        if state is not None:
            w = find_param(state, "attn.pq.weight", cfg.layer)
            if w is not None:
                attn.pq.weight.copy_(w)
        return attn.pq(posv).numpy(), ypos, xdim, "untied positional query pq(sinusoid)", False
    if pe == "rpe":
        a = attn.rel_k                                   # (2*max_len-1, d_h)
        if state is not None:
            w = find_param(state, "attn.rel_k", cfg.layer)
            if w is not None:
                a = w
        off0 = (a.shape[0] - 1) // 2                      # centre = zero lag (max_len can be 2048)
        W = min(P, off0)
        rows = a[off0 - W: off0 + W + 1]                  # (2W+1, d_h)
        return (rows.numpy(), "Relative offset (i - j)", "Head dimension",
                "relative-key embedding a_delta (no per-position code)", True)
    if pe == "erpe":
        bias = attn.rel_bias                             # (h, 2*max_len-1)
        note = "relative bias b[delta] per head (post-softmax)"
        if state is not None:
            w = find_param(state, "attn.rel_bias", cfg.layer)
            if w is not None:
                bias = w
        if float(bias.abs().max()) < 1e-8:               # zeros at init -> show learned shape
            bias = _illustrative_erpe_bias(cfg)
            note = "relative bias b[delta] (illustrative learned shape; 0 at init)"
        off0 = (bias.shape[1] - 1) // 2
        W = min(P, off0)
        rows = bias[:, off0 - W: off0 + W + 1].t()       # (2W+1, h)
        return (rows.numpy(), "Relative offset (i - j)", "Head", note, True)
    if pe == "convspe":
        w = find_param(state, "attn.spe_q.weight", cfg.layer) if state is not None else None
        f = (w if w is not None else attn.spe_q.weight).squeeze(1).t()   # (kernel, h)
        return (f.numpy(), "Conv tap", "Head",
                "learned conv filter spe_q (no per-position code)", False)
    raise ValueError(f"unknown pe: {pe}")


# Complete encoding grid: EVERY method, each on its NATURAL axes. Absolute PEs
# have a per-position code P[position, dim]; relative PEs (RPE/eRPE/ConvSPE) have
# none, so their actual learnable object is shown (relative-offset embedding /
# bias / conv filter). T-PE's additive part == sinusoidal, so it is omitted.
ENCMAP_METHODS = ["sinusoidal", "tape", "learnable", "time2vec", "tupe",
                  "rpe", "erpe", "convspe"]
# Display name of every encoding, used by all four figure modes.
TITLES = {"sinusoidal": "Sinusoidal", "tape": "tAPE", "learnable": "Learnable",
          "time2vec": "Time2Vec", "lff": "LFF", "tupe": "TUPE", "tpe": "T-PE",
          "rpe": "RPE", "erpe": "eRPE", "convspe": "ConvSPE"}


def run_encmap(args, out: Path, global_state):
    pes = [p for p in ENCMAP_METHODS if p in args.pes]
    n = len(pes)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.8 * nrows),
                             squeeze=False, constrained_layout=True)
    L = args.length
    ypos, ylab = _pos_ticks(L, args.bins_per_day)          # position axis (for the APE panels)
    trained = bool(args.ckpt or getattr(args, "ckpt_dir", None))
    for ax, pe in zip(axes.ravel(), pes):
        print(f"[encmap] {pe}", flush=True)
        st = state_for_pe(pe, args, global_state)          # deterministic -> exact; learned -> real weights
        M, ylabel, xlabel, note, is_off = build_encoding_map(pe, args, st)
        rows, cols = M.shape
        v = float(np.percentile(np.abs(M), 99.5)) or 1.0
        extent = [0, cols, rows // 2, -(rows // 2)] if is_off else [0, cols, L, 0]
        im = ax.imshow(M, cmap=BLUE_ORANGE, vmin=-v, vmax=v, aspect="auto",
                       origin="upper", extent=extent, interpolation="nearest")
        ax.set_title(TITLES[pe], fontsize=14, pad=13)
        ax.text(0.5, 1.005, note, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=6, color="0.45")                  # what this panel actually is
        if is_off:                                         # relative: offset / tap on y
            ax.axhline(0, color="0.2", lw=0.6, ls=":")
            ax.set_ylabel(ylabel, fontsize=8)
        else:                                              # absolute: position (timestep) on y
            ax.set_yticks(ypos); ax.set_yticklabels(ylab, fontsize=6.5)
            ax.set_ylabel("Position (timestep;  Nd = day N)", fontsize=8)
        ax.set_xlabel(f"{xlabel}: {cols}", fontsize=8)
        ax.tick_params(labelsize=6.5, length=2)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("value (true)", fontsize=6.5)
        cb.ax.tick_params(labelsize=6)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Encoding matrix of every positional encoding  (absolute: position × dim; "
                 "relative: offset/tap — no per-position code)", fontsize=13, fontweight="bold")
    p = out / ("encoding_maps" + ("_illustrative" if args.illustrative else "")
               + ("_trained" if trained else "") + ".png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {p}")


# Rhythmicity visualization (Zhou et al. 2024, Fig. 6 adapted to the TIME axis):
# for a queried time position, cos(P[ref], P[t]) to every other timestep.
#   left  : reshaped to [day-of-week x hour-of-day] -> a VERTICAL band at the ref
#           hour across all days == the encoding recognises the 24 h rhythm.
#   right : the same over the whole week, vs the fixed sine-cosine baseline, with a
#           circadian score = corr( curve(t), curve(t - 1 day) ) (daily self-similarity).
RHYTHM_METHODS = ["sinusoidal", "learnable", "time2vec", "tupe"]


def _cos_to_ref(P, ref):
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    return Pn @ Pn[ref]                                    # (L,) cosine of every position to ref


def run_rhythmviz(args, out: Path, global_state):
    L, bpd = args.length, args.bins_per_day
    n_days = max(1, L // bpd)
    Lf = n_days * bpd                                      # whole days only, for the [day x hour] reshape
    ref = (n_days // 2) * bpd + bpd // 2                   # a mid-week noon reference
    steps_to_h = args.bin_minutes / 60.0
    t_h = np.arange(Lf) * steps_to_h

    # fixed sine-cosine baseline (Vaswani), like the paper's orange reference curve
    base = build_encoding_map("sinusoidal", args, state_for_pe("sinusoidal", args, global_state))[0]
    cbase = _cos_to_ref(base, ref)[:Lf]

    pes = [p for p in RHYTHM_METHODS if p in args.pes]
    trained = bool(args.ckpt or getattr(args, "ckpt_dir", None))
    nrows = len(pes)
    fig, axes = plt.subplots(nrows, 2, figsize=(13, 3.0 * nrows), squeeze=False,
                             gridspec_kw={"width_ratios": [1.0, 1.9]}, constrained_layout=True)
    for r, pe in enumerate(pes):
        print(f"[rhythmviz] {pe}", flush=True)
        st = state_for_pe(pe, args, global_state)
        P, _, _, _, _ = build_encoding_map(pe, args, st)
        c = _cos_to_ref(P, ref)[:Lf]
        # circadian score: daily self-similarity (correlation with a 1-day-shifted copy)
        a, b = c[:Lf - bpd], c[bpd:]
        score = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9 else 0.0

        # --- left: [day-of-week x hour-of-day] map ---
        axl = axes[r][0]
        M2d = c.reshape(n_days, bpd)
        im = axl.imshow(M2d, cmap="magma", aspect="auto", extent=[0, 24, n_days, 0],
                        vmin=float(c.min()), vmax=float(c.max()), interpolation="nearest")
        axl.plot((bpd // 2) / bpd * 24, n_days // 2 + 0.5, "o", mfc="none",
                 mec="cyan", ms=12, mew=2)             # the queried time (ref)
        axl.set_title(TITLES[pe], fontsize=13)
        axl.set_xlabel("hour of day", fontsize=8)
        axl.set_ylabel("day of week", fontsize=8)
        axl.set_xticks([0, 6, 12, 18, 24]); axl.set_yticks(range(n_days))
        axl.tick_params(labelsize=7)
        fig.colorbar(im, ax=axl, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)

        # --- right: over the week vs sine-cosine baseline ---
        axr = axes[r][1]
        axr.plot(t_h, cbase, color="#e8a13a", lw=1.0, alpha=0.9, label="sine-cosine (fixed)")
        axr.plot(t_h, c, color="#1f5fb0", lw=1.2, label=TITLES[pe])
        for day in range(1, n_days):
            axr.axvline(day * 24, color="0.6", ls="--", lw=0.6)   # day boundaries (24 h)
        axr.set_xlim(0, Lf * steps_to_h)
        axr.set_xlabel("time over the week (hours;  dashed = day boundary)", fontsize=8)
        axr.set_ylabel("cosine similarity to queried time", fontsize=8)
        axr.tick_params(labelsize=7)
        axr.legend(fontsize=7, loc="upper right", ncol=2, framealpha=0.9)
        axr.text(0.02, 0.06, f"circadian score (24 h self-similarity): {score:+.2f}",
                 transform=axr.transAxes, fontsize=9, fontweight="bold", color="0.15")

    mode = ("trained (real weights)" if trained else "initialisation")
    fig.suptitle(
        f"Rhythmicity signature of each position encoding  (queried time = day {n_days // 2}, noon)   "
        f"|   {mode}", fontsize=13, fontweight="bold")
    p = out / ("rhythm_signature" + ("_trained" if trained else "") + ".png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {p}")


# Token signal BEFORE vs AFTER embedding, for a depressed + a non-depressed test
# window, across every technique -- content-dependent, so it reads the arrays that
# train_hrd.py wrote (signal_embedding.npz per variant). Needs --ckpt-dir (a run
# folder) + trained model; not computable from position encodings alone.
_SIGVIZ_ORDER = ["sinusoidal", "tape", "learnable", "time2vec", "tupe",
                 "rpe", "erpe", "convspe", "tpe", "none"]


def run_signalviz(args, out: Path):
    import re
    ckpt_dir = getattr(args, "ckpt_dir", None)
    if not ckpt_dir:
        print("[signalviz] needs --ckpt-dir (reads signal_embedding.npz written by train_hrd); skipping")
        return
    files = sorted(Path(ckpt_dir).glob("**/signal_embedding.npz"))
    if not files:
        print(f"[signalviz] no signal_embedding.npz under {ckpt_dir} (train first); skipping")
        return
    entries = {}                                           # pe -> npz path (first seed wins)
    for f in files:
        base = re.sub(r"_seed.*$", "", f.parent.name)
        pe = base.split("_", 1)[1] if "_" in base else base
        entries.setdefault(pe, f)
    pes = [p for p in _SIGVIZ_ORDER if p in entries] + [p for p in entries if p not in _SIGVIZ_ORDER]

    d0 = np.load(files[0], allow_pickle=True)              # raw signal is identical across variants
    sig_dep, sig_non = d0["sig_dep"], d0["sig_non"]
    pid_dep, pid_non = str(d0["pid_dep"]), str(d0["pid_non"])
    n_sensors = sig_dep.shape[1]
    T = sig_dep.shape[0]
    ypos, ylab = _pos_ticks(T, args.bins_per_day)

    nrows = 1 + len(pes)
    fig, axes = plt.subplots(nrows, 2, figsize=(9.5, 2.6 * nrows), squeeze=False,
                             constrained_layout=True)

    def panel(ax, M, title, xlabel):
        v = float(np.percentile(np.abs(M), 99.5)) or 1.0
        im = ax.imshow(M, cmap=BLUE_ORANGE, vmin=-v, vmax=v, aspect="auto", origin="upper",
                       extent=[0, M.shape[1], T, 0], interpolation="nearest")
        ax.set_title(title, fontsize=9.5)
        ax.set_yticks(ypos); ax.set_yticklabels(ylab, fontsize=6)
        ax.set_xlabel(xlabel, fontsize=7); ax.tick_params(labelsize=6, length=2)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=5)

    panel(axes[0][0], sig_dep, f"DEPRESSED  (pid {pid_dep})\nraw signal (before embedding)",
          f"sensor channel: {n_sensors}")
    panel(axes[0][1], sig_non, f"NON-depressed  (pid {pid_non})\nraw signal (before embedding)",
          f"sensor channel: {n_sensors}")
    for r, pe in enumerate(pes, start=1):
        d = np.load(entries[pe], allow_pickle=True)
        panel(axes[r][0], d["emb_dep"], f"{pe.upper()}  -  encoded (after embedding)",
              f"embedding dim: {d['emb_dep'].shape[1]}")
        panel(axes[r][1], d["emb_non"], f"{pe.upper()}  -  encoded (after embedding)",
              f"embedding dim: {d['emb_non'].shape[1]}")

    fig.suptitle(
        "Token signal BEFORE vs AFTER embedding   |   depressed vs non-depressed "
        "(one held-out test window each)   |   y = time (day markers)",
        fontsize=12, fontweight="bold")
    p = out / "signal_embedding.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {p}")


# ---------------------------------------------------------------------------
# Position-wise COSINE-similarity matrices (Wang & Chen 2020, Fig. 10 style).
# cos(P_i,P_j) = <P_i,P_j> / (||P_i|| ||P_j||) for the ABSOLUTE per-position
# embedding P of each APE method (relative/bias PEs are NOT APE and are excluded,
# exactly as in the reference figure). Diagonal is always 1; sequential magma
# colormap; no titles/notes above beyond the method name -- kept clean.
# ALL positional encodings on ONE comparable position×position grid. APE methods
# show cosine(P_i,P_j) (per-position code exists); relative/hybrid methods have no
# per-position code, so they show the position-position term they inject into the
# attention scores (S_ij). Both live on the SAME [query x key position] axes, so
# every method is directly comparable in a single figure.
COSINE_ALL = ["sinusoidal", "tape", "learnable", "time2vec", "tupe",
              "tpe", "rpe", "erpe", "convspe"]
COSINE_APE_SET = {"sinusoidal", "tape", "learnable", "time2vec", "tupe", "tpe"}


def run_cosine(args, out: Path, global_state):
    pes = [p for p in COSINE_ALL if p in args.pes]
    n = len(pes)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 4.0 * nrows),
                             squeeze=False, constrained_layout=True)
    L = args.length
    step = 100 if L > 300 else 50                          # plain integer ticks (Wang & Chen Fig. 10 style)
    ticks = list(range(0, L, step))
    for ax, pe in zip(axes.ravel(), pes):
        print(f"[cosine] {pe}", flush=True)
        st = state_for_pe(pe, args, global_state)         # per-PE trained weights
        if pe in COSINE_APE_SET:                          # absolute: cosine of the position code
            P, _, _, _, _ = build_encoding_map(pe, args, st)
            Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
            C = Pn @ Pn.T
            vmin, vmax = float(C.min()), 1.0
        else:                                             # relative: position-position attention bias
            S = build_matrix(pe, args, st)
            C = S / (np.abs(S).max() + 1e-12)
            vmin, vmax = float(C.min()), float(C.max())
        im = ax.imshow(C, cmap="magma", vmin=vmin, vmax=vmax,
                       origin="upper", aspect="equal", interpolation="nearest")
        ax.set_title(TITLES[pe], fontsize=15)      # method name only (like Fig. 10)
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.tick_params(labelsize=8, length=2)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", location="bottom",
                          fraction=0.05, pad=0.10)         # horizontal colorbar BELOW, as in Fig. 10
        cb.ax.tick_params(labelsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    trained = bool(args.ckpt or getattr(args, "ckpt_dir", None))
    p = out / ("cosine_similarity" + ("_illustrative" if args.illustrative else "")
               + ("_trained" if trained else "") + ".png")
    fig.savefig(p, dpi=220, bbox_inches="tight")           # higher resolution for fine off-diagonal detail
    plt.close(fig)
    print(f"[saved] {p}")


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def _pos_ticks(L: int, bpd: int):
    """Tick positions (timesteps) and labels, marking day boundaries."""
    if bpd > 0 and L // bpd <= 12:
        pos = list(range(0, L, bpd)) + [L - 1]
    else:
        step = max(1, round(L / 8))
        pos = list(range(0, L, step)) + [L - 1]
    pos = sorted(set(min(p, L - 1) for p in pos))
    labels = [(f"{p}\n{p // bpd}d" if bpd > 0 and p % bpd == 0 else f"{p}") for p in pos]
    return pos, labels


def draw_panel(fig, ax, S: np.ndarray, title: str, subtitle: str, cfg):
    """One panel: real position axes + a per-panel colorbar in TRUE S units
    (symmetric diverging scale centred at 0), matching Fig. 1 of Li et al."""
    L = S.shape[0]
    v = float(np.abs(S).max())
    im = ax.imshow(S, cmap="RdBu_r", vmin=-v, vmax=v,
                   origin="upper", interpolation="nearest", aspect="equal")
    ax.set_title(title, fontsize=10.5, pad=16, fontweight="bold")
    ax.text(0.5, 1.015, subtitle, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=6.2, color="0.4")
    xpos, xlab = _pos_ticks(L, cfg.bins_per_day)
    ax.set_xticks(xpos); ax.set_xticklabels(xlab, fontsize=5.5)
    ax.set_yticks(xpos)
    ax.set_yticklabels([l.split("\n")[0] for l in xlab], fontsize=5.5)
    ax.set_xlabel("key position j  (timestep;  Nd = day N)", fontsize=6.5)
    ax.set_ylabel("query position i", fontsize=6.5)
    ax.tick_params(length=2)
    if v < 1e-8:
        ax.text(0.5, 0.5, "= 0 at init", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="0.4")
    if cfg.mark_period:
        for c in range(cfg.bins_per_day, L, cfg.bins_per_day):
            for off in (c, -c):                            # lines i - j = off (lag = off)
                ax.plot([0, L - 1], [off, L - 1 + off], lw=0.4, ls=":",
                        color="0.15", alpha=0.5)
        ax.set_xlim(-0.5, L - 0.5); ax.set_ylim(L - 0.5, -0.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=5.5)
    cb.set_label("S_ij = P_i·P_j  (true units)", fontsize=6)
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pes", nargs="+", default=DEFAULT_PES,
                    help=f"subset of {DEFAULT_PES}")
    # geometry -- defaults reproduce the HRD run (168 h window, 15 min bins)
    ap.add_argument("--window-hours", type=int, default=168)
    ap.add_argument("--bin-minutes", type=int, default=15)
    ap.add_argument("--length", type=int, default=None,
                    help="override L (default = window-hours*60/bin-minutes)")
    ap.add_argument("--repr-dims", type=int, default=320)
    ap.add_argument("--time2vec-dim", type=int, default=16)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--lf-freqs", type=int, default=None)
    ap.add_argument("--lf-gamma", type=float, default=1.0)
    ap.add_argument("--layer", type=int, default=0,
                    help="which transformer layer's attn params to read from --ckpt")
    ap.add_argument("--spe-realizations", type=int, default=512,
                    help="R for the ConvSPE kernel estimate (higher = smoother)")
    ap.add_argument("--illustrative", action="store_true",
                    help="pin Time2Vec/LFF/eRPE frequencies to the circadian spectrum")
    ap.add_argument("--mark-period", action="store_true",
                    help="overlay dotted lines every bins-per-day lags")
    ap.add_argument("--ckpt", default=None,
                    help="path to a CoST net state_dict (cost.CoST.save) with trained PE params")
    ap.add_argument("--ckpt-dir", default=None,
                    help="results dir (e.g. results_hrd/<jobid>): each PE loads its OWN "
                         "run's net.pt from */*_{pe}_seed*/net.pt -> full REAL-weight figure")
    ap.add_argument("--ncols", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(ROOT / "pics" / "pe_similarity"))
    ap.add_argument("--per-pe", action="store_true", help="also save one PNG per PE")
    ap.add_argument("--save-npy", action="store_true", help="also dump each raw S as .npy")
    # Encoding-matrix heatmaps P[position, dim] (classic sinusoidal-PE view)
    ap.add_argument("--encmap", action="store_true",
                    help="render the P[position, dim] encoding heatmaps (blue/orange)")
    ap.add_argument("--enc-positions", type=int, default=None,
                    help="timesteps (rows) in the encoding heatmaps (default = full model length L)")
    ap.add_argument("--cosine", action="store_true",
                    help="render position-wise COSINE similarity (Wang & Chen 2020, Fig. 10 style)")
    ap.add_argument("--rhythmviz", action="store_true",
                    help="rhythmicity signature (Zhou et al. Fig. 6 style: queried-time similarity, day x hour)")
    ap.add_argument("--signalviz", action="store_true",
                    help="token signal before vs after embedding, depressed vs non (reads train_hrd npz; needs --ckpt-dir)")
    args = ap.parse_args()

    args.length = args.length or (args.window_hours * 60 // args.bin_minutes)
    args.bins_per_day = 24 * 60 // args.bin_minutes
    args.max_len = args.length                             # match training (max_train_length=seq_len)
    args.enc_positions = args.enc_positions or args.length  # exact model window by default

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    state = None
    if args.ckpt:
        state = torch.load(args.ckpt, map_location="cpu")
        state = state.get("state_dict", state) if isinstance(state, dict) else state

    if args.encmap:
        run_encmap(args, out, state)
        return

    if args.cosine:
        run_cosine(args, out, state)
        return

    if args.rhythmviz:
        run_rhythmviz(args, out, state)
        return

    if args.signalviz:
        run_signalviz(args, out)
        return

    mats = {}
    for pe in args.pes:
        if pe not in FAMILY:
            raise SystemExit(f"unknown pe '{pe}'; choose from {list(FAMILY)}")
        print(f"[build] {pe:<11s} L={args.length} d={args.repr_dims} ...", flush=True)
        S = build_matrix(pe, args, state)
        mats[pe] = S
        if args.save_npy:
            np.save(out / f"S_{pe}.npy", S)

    # --- combined grid (each panel: real position axes + its own colorbar) --
    n = len(args.pes)
    ncols = min(args.ncols, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.7 * ncols, 3.9 * nrows),
                             squeeze=False, constrained_layout=True)
    for ax, pe in zip(axes.ravel(), args.pes):
        draw_panel(fig, ax, mats[pe], pe.upper(), FAMILY[pe], args)
    for ax in axes.ravel()[n:]:
        ax.axis("off")

    mode = ("illustrative (circadian-pinned)" if args.illustrative
            else "trained (--ckpt)" if state is not None else "initialisation prior")
    fig.suptitle(
        f"Position-position similarity  S_ij = P_i·P_j   |   "
        f"L={args.length} steps, {args.bin_minutes} min bins, {args.bins_per_day}/day "
        f"(1 day = {args.bins_per_day} steps)   |   {mode}",
        fontsize=13, fontweight="bold")
    grid_path = out / ("position_similarity_grid"
                       + ("_illustrative" if args.illustrative else "")
                       + ("_trained" if state is not None else "") + ".png")
    fig.savefig(grid_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {grid_path}")

    # --- optional per-PE panels ------------------------------------------
    if args.per_pe:
        for pe in args.pes:
            f, a = plt.subplots(figsize=(5.0, 4.6))
            draw_panel(f, a, mats[pe], pe.upper(), FAMILY[pe], args)
            f.tight_layout()
            p = out / f"S_{pe}.png"
            f.savefig(p, dpi=160, bbox_inches="tight")
            plt.close(f)
            print(f"[saved] {p}")


if __name__ == "__main__":
    main()
