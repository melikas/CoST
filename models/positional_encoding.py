"""Positional-encoding library for the CoST Transformer backbone.

This module collects the positional-encoding (PE) variants that can be swapped
into :class:`models.encoder.TransformerFeatureExtractor`, plus the ``Time2Vec``
time-embedding that is also usable with the TCN backbone.

Two families are supported:

* **Absolute** PEs are added to the token embeddings before the attention
  stack (``sinusoidal``, ``learnable``, ``tape``, ``time2vec``).
* **Attention** PEs inject position information inside every self-attention
  layer (``rpe``, ``erpe``, ``tupe``, ``convspe``, ``tpe``).

The single entry points used by the encoder are :func:`add_absolute_pe`
(for the absolute family) and :class:`PETransformerEncoderLayer` (which wraps
:class:`PESelfAttention` and handles the attention family transparently).
"""
import math

import torch
from torch import nn

ABSOLUTE_PES = ("sinusoidal", "learnable", "tape", "time2vec")
ATTENTION_PES = ("rpe", "erpe", "tupe", "convspe", "tpe")
SUPPORTED_PES = ABSOLUTE_PES + ATTENTION_PES


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
    """Time2Vec (Kazemi et al. 2019, arXiv:1907.05321) as a learnable absolute PE.

    Channel 0 is a linear time term; channels 1.. are periodic ``sin`` features.
    Usable with either backbone: pass the model/hidden width as ``d_model``.
    """

    def __init__(self, d_model, max_len):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.w = nn.Parameter(torch.empty(d_model))
        self.b = nn.Parameter(torch.empty(d_model))
        self._reset_parameters()

    def _reset_parameters(self):
        with torch.no_grad():
            d = self.d_model
            div = torch.exp(torch.arange(d, dtype=torch.float32)
                            * (-math.log(10000.0) / max(d, 1)))
            self.w.copy_(div)
            self.w[0] = 1.0 / max(self.max_len, 1)
            self.b.zero_()

    def forward(self, T, device, dtype):
        pos = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)
        v = pos * self.w + self.b
        out = v.clone()
        out[:, 1:] = torch.sin(v[:, 1:])
        return out.to(dtype)


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
            self.log_sigma = nn.Parameter(torch.zeros(1))

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
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn = scores.softmax(-1)
            bias = self.rel_bias[:, self._rel_index(T, x.device)]
            attn = self.drop(attn + bias.unsqueeze(0))
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
            z = torch.randn(self.R, self.h, T, device=x.device, dtype=x.dtype)
            qpe = self.spe_q(z)
            kpe = self.spe_k(z)
            pos = torch.einsum("rht,rhs->hts", qpe, kpe) / self.R
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
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Helpers used by the encoder
# ---------------------------------------------------------------------------
def add_absolute_pe(x, pe, d_model, learnable_pe=None, time2vec_pe=None):
    """Add the absolute position signal for ``pe`` to ``x`` (B, T, d_model).

    Attention-family PEs add nothing here (they act inside attention); ``tpe``
    is given a sinusoidal anchor in addition to its in-attention Gaussian bias.
    """
    T = x.size(1)
    if pe == "sinusoidal" or pe == "tpe":
        return x + sinusoidal_pe(T, d_model, x.device, x.dtype).unsqueeze(0)
    if pe == "tape":
        return x + tape_pe(T, d_model, x.device, x.dtype).unsqueeze(0)
    if pe == "learnable":
        pos = torch.arange(T, device=x.device)
        return x + learnable_pe(pos).to(x.dtype).unsqueeze(0)
    if pe == "time2vec":
        return x + time2vec_pe(T, x.device, x.dtype).unsqueeze(0)
    return x
