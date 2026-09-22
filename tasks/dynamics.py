"""Time-resolved summaries of a trained encoder's trend (V^T) and seasonal (V^S) sequences.

The pooled representation keeps one vector per window. The figures of rhythm shape, time scale
and group differences need the sequences themselves, which only the saved encoder can produce.
`export_dynamics` encodes every window once and keeps small per-participant and per-window
summaries. Held-out participants are reported; training participants of the same fold fit every
normalisation, so nothing about a test participant, and no label, shapes the scale of its numbers.

Stored (all float32):
  profile_<branch>   (n_test, b, k)  24 h profile of the first k principal components of the branch,
                                     in units of the training participants' SD of each component.
                                     Components are fitted on training participants' daily
                                     profiles; each sign is fixed so the training-mean profile
                                     correlates positively with the population step profile.
  spectrum_<branch>  (n_test, F)     power per rFFT bin summed over channels, divided by the
                                     training participants' mean total power (relative power).
  acf_<branch>       (n_test, L)     autocorrelation at lags 0..L-1 bins, pooled over channels.
  window_*           per test window: 24 h / 12 h relative power of V^S, V^S peak hour (from the 24 h
                                     coefficient of seasonal PC1), day-to-day correlation of V^S and
                                     V^T, trend PC1 level (training-SD units) and within-window slope.
  rq2_*              per test participant with four contiguous reference weeks: pooled
                                     representations of those weeks, of the next week, and of that
                                     week shifted (0.5-4 h) or with its 24 h component scaled (1 +- a).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import fft

from tasks.personalized import AMPLITUDE_FRACTIONS, PHASE_HOURS
from tasks.rhythm import amplitude_scale, phase_shift

K = 3                                 # principal components kept per branch


def _branches(model, x):
    """(trend, seasonal) sequences, each (B, T, C), in evaluation mode."""
    t, s = model.net(torch.as_tensor(x, dtype=torch.float).to(model.device))
    return t.float().cpu(), s.float().cpu()


def _acf(seq, lags):
    """Autocorrelation pooled over channels: sum_c sum_t x_t x_{t+L} / sum_c sum_t x_t^2."""
    z = seq - seq.mean(dim=1, keepdim=True)
    n = z.shape[1]
    f = fft.rfft(z, n=2 * n, dim=1)
    r = fft.irfft(f.abs() ** 2, n=2 * n, dim=1)[:, :lags].sum(dim=-1)
    return (r / r[:, :1].clamp_min(1e-12)).numpy()


def _day_corr(seq, b):
    """Mean correlation between consecutive days of the (days x bins*channels) matrix."""
    B, T, C = seq.shape
    d = seq.reshape(B, T // b, b * C)
    d = d - d.mean(dim=-1, keepdim=True)
    d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return (d[:, 1:] * d[:, :-1]).sum(dim=-1).mean(dim=1).numpy()


def _pca(rows, k):
    mean = rows.mean(0)
    _, sv, vt = np.linalg.svd(rows - mean, full_matrices=False)
    return mean, vt[:k].T, (sv[:k] ** 2 / (sv ** 2).sum())


def _reference_weeks(window_ids, idx, weeks=4):
    """Index of the latest window with `weeks` contiguous preceding weeks, and those weeks."""
    days = np.array([np.datetime64(str(w).rsplit("_", 1)[1][:10]) for w in window_ids[idx]])
    order = np.argsort(days)
    idx, days = idx[order], days[order]
    for j in range(len(idx) - 1, weeks - 1, -1):
        if np.all(np.diff(days[j - weeks:j + 1]) == np.timedelta64(7, "D")):
            return idx[j], idx[j - weeks:j]
    return None, None


@torch.no_grad()
def export_dynamics(model, X, pids, window_ids, train_ids, test_ids, bins_per_day, bin_minutes,
                    align_channel, batch_size=32):
    model.net.eval()
    N, T, _ = X.shape
    b, days = int(bins_per_day), T // int(bins_per_day)
    lags = T - 2 * b                          # at least two days of pairs at the longest lag
    pids = np.asarray(pids).astype(str)
    people = np.unique(pids)
    row = {p: i for i, p in enumerate(people)}
    ct, cs = model.net.trend_dims, model.net.seasonal_dims
    prof = {"trend": np.zeros((len(people), b, ct)), "seasonal": np.zeros((len(people), b, cs))}
    spec = {"trend": np.zeros((len(people), T // 2 + 1)), "seasonal": np.zeros((len(people), T // 2 + 1))}
    acf = {"trend": np.zeros((len(people), lags)), "seasonal": np.zeros((len(people), lags))}
    # Windows per participant. Counting with `count[rows] += 1` would be wrong: NumPy fancy
    # indexing applies a repeated index once, and a batch usually holds several weeks of one person.
    count = np.array([np.sum(pids == p) for p in people], dtype=float)
    w = dict(p24=np.zeros(N), p12=np.full(N, np.nan), day_corr_seasonal=np.zeros(N),
             day_corr_trend=np.zeros(N))
    trend_mean = np.zeros((N, ct), np.float32)
    c24 = np.zeros((N, cs), np.complex64)
    h12 = 2 * days if 2 * days < T / 2 else None      # 12 h lies strictly below Nyquist on HRD only
    for i in range(0, N, batch_size):
        t, s = _branches(model, X[i:i + batch_size])
        rows = [row[p] for p in pids[i:i + batch_size]]
        for name, seq in (("trend", t), ("seasonal", s)):
            daily = seq.reshape(len(seq), days, b, -1).mean(dim=1).numpy()
            power = (fft.rfft(seq, dim=1).abs() ** 2).sum(dim=-1).numpy()
            a = _acf(seq, lags)
            for r, dp, pw, ac in zip(rows, daily, power, a):
                prof[name][r] += dp
                spec[name][r] += pw
                acf[name][r] += ac
        S = fft.rfft(s, dim=1)
        w["p24"][i:i + len(s)] = (S[:, days].abs() ** 2).sum(dim=-1).numpy()
        if h12 is not None:
            w["p12"][i:i + len(s)] = (S[:, h12].abs() ** 2).sum(dim=-1).numpy()
        w["day_corr_seasonal"][i:i + len(s)] = _day_corr(s, b)
        w["day_corr_trend"][i:i + len(s)] = _day_corr(t, b)
        trend_mean[i:i + len(s)] = t.mean(dim=1).numpy()
        c24[i:i + len(s)] = S[:, days].numpy()
    for d in (prof, spec, acf):
        for name in d:
            d[name] /= count[:, None, None] if d[name].ndim == 3 else count[:, None]

    train = np.isin(people, list(map(str, train_ids)))
    test = np.isin(people, list(map(str, test_ids)))
    out = dict(participants=people[test], lags_hours=np.arange(lags) * bin_minutes / 60.0,
               period_hours=np.r_[np.inf, T * bin_minutes / 60.0 / np.arange(1, T // 2 + 1)],
               bins_per_day=b, days=days)
    # Population step profile of training participants, the sign reference for every component.
    tr_windows = np.isin(pids, people[train])
    ref = X[tr_windows][:, :, align_channel].reshape(-1, days, b).mean(axis=(0, 1))
    loadings = {}
    for name in ("trend", "seasonal"):
        rows = prof[name][train].reshape(-1, prof[name].shape[-1])
        mean, V, ratio = _pca(rows, K)
        scores = (prof[name] - mean) @ V                               # (people, b, K)
        for k in range(K):
            if np.corrcoef(scores[train, :, k].mean(0), ref)[0, 1] < 0:
                V[:, k], scores[:, :, k] = -V[:, k], -scores[:, :, k]
        sd = scores[train].reshape(-1, K).std(0)
        out[f"profile_{name}"] = (scores[test] / sd).astype(np.float32)
        out[f"explained_{name}"] = ratio.astype(np.float32)
        total = spec[name][train].sum(1).mean()
        out[f"spectrum_{name}"] = (spec[name][test] / total).astype(np.float32)
        out[f"acf_{name}"] = acf[name][test].astype(np.float32)
        loadings[name] = V[:, 0]
    # Per test window.
    tw = np.flatnonzero(np.isin(pids, people[test]))
    p24_scale = w["p24"][tr_windows].mean()
    out["window_participant"] = pids[tw]
    out["window_id"] = np.asarray(window_ids)[tw].astype(str)
    out["window_p24"] = (w["p24"][tw] / p24_scale).astype(np.float32)
    out["window_p12"] = (w["p12"][tw] / (np.nanmean(w["p12"][tr_windows]) if h12 else 1)).astype(np.float32)
    out["window_day_corr_seasonal"] = w["day_corr_seasonal"][tw].astype(np.float32)
    out["window_day_corr_trend"] = w["day_corr_trend"][tw].astype(np.float32)
    # V^S peak hour: angle of the 24 h coefficient of seasonal PC1. A delay by h hours rotates the
    # rFFT coefficient by -2 pi h / 24, so the peak hour is -angle * 24 / (2 pi), modulo 24.
    z = c24[tw] @ loadings["seasonal"]
    out["window_peak_hour"] = ((-np.angle(z) * 12 / np.pi) % 24).astype(np.float32)
    tm_mean, tm_V, _ = _pca(trend_mean[tr_windows], 1)
    if np.corrcoef(tm_V[:, 0], loadings["trend"])[0, 1] < 0:
        tm_V = -tm_V
    level = (trend_mean - tm_mean) @ tm_V[:, 0]
    out["window_trend_level"] = (level[tw] / level[tr_windows].std()).astype(np.float32)
    slopes = []
    for i in range(0, len(tw), batch_size):
        t, _ = _branches(model, X[tw[i:i + batch_size]])
        proj = (t.numpy() - tm_mean) @ tm_V[:, 0] / level[tr_windows].std()   # (B, T)
        hours = np.arange(T) * bin_minutes / 60.0
        slopes.append(np.polyfit(hours / 24.0, proj.T, 1)[0])                 # per day
    out["window_trend_slope"] = np.concatenate(slopes).astype(np.float32)
    # RQ2: pooled representations of a reference month, the next week and its perturbations.
    rq2 = dict(participant=[], ref=[], current=[], shifted=[], scaled_up=[], scaled_down=[])
    # RQ2 needs seven-day windows and shifts that fall on the sampling grid (HRD only, as in RQ2).
    rq2_ok = days == 7 and all(float(h * 60 / bin_minutes).is_integer() for h in PHASE_HOURS)
    for p in people[test] if rq2_ok else []:
        cur, refs = _reference_weeks(np.asarray(window_ids), np.flatnonzero(pids == p))
        if cur is None:
            continue
        x = X[cur][None].astype(np.float32)
        rq2["participant"].append(p)
        rq2["ref"].append(model.encode(X[refs], batch_size=8))
        rq2["current"].append(model.encode(x, batch_size=8)[0])
        rq2["shifted"].append(model.encode(np.concatenate(
            [phase_shift(x, h, x.shape[-1], bin_minutes) for h in PHASE_HOURS]), batch_size=8))
        rq2["scaled_up"].append(model.encode(np.concatenate(
            [amplitude_scale(x, 1 + a, x.shape[-1], b) for a in AMPLITUDE_FRACTIONS]), batch_size=8))
        rq2["scaled_down"].append(model.encode(np.concatenate(
            [amplitude_scale(x, 1 - a, x.shape[-1], b) for a in AMPLITUDE_FRACTIONS]), batch_size=8))
    for k, v in rq2.items():
        out[f"rq2_{k}"] = np.asarray(v, dtype=str if k == "participant" else np.float32)
    out["rq2_shift_hours"] = np.asarray(PHASE_HOURS, np.float32)
    out["rq2_alpha"] = np.asarray(AMPLITUDE_FRACTIONS, np.float32)
    return out
