"""The paper's Cosinor model -- the project's ONLY cosinor, used both as the classical
baseline and as the source of the rhythm markers the latent space is validated against.

Method, from the reference implementation

    HAI-lab-UVA/Human-Rhythms-Dataset -> rhythms_consinor.py
    (Yan et al. 2022, "A Computational Framework for Modeling Biobehavioral Rhythms",
     ACM TIST 13(3), Article 47)

which builds on **CosinorPy** (``cosinor.fit_me``): per series, a Fourier periodogram selects
the significant periods, and a single-component cosinor is fitted at each, giving 12 rhythm
parameters per period. Only that path is kept here -- the reference's ``convert_to_continuous``
and ``reduce_periods`` are never reached from it (the latter's validation rejects its own
documented defaults), and ``periodogram`` is reduced to its ``period_type='per'`` branch with
the argument validation dropped, since the single caller passes fixed, valid arguments. The
numerics are unchanged; see the identity test in the commit that introduced this file.

Two properties make the wrapper the reference rather than one variant among several:
  * the time axis is anchored to MIDNIGHT, so Acrophase / Orthophase / Bathyphase mean clock
    time and are comparable across participants;
  * parameters are aggregated to the SUBJECT, the unit cosinor describes.

NumPy-2.0 note: CosinorPy (<=3.1) predates NumPy 2.0 and still references removed aliases
(``np.round_`` etc.). They are restored below BEFORE importing CosinorPy; a compatibility
shim, not a change to the cosinor math.
"""
import os
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

for _n, _v in {"round_": np.round, "float_": np.float64, "NaN": np.nan, "NAN": np.nan,
               "Inf": np.inf, "alltrue": np.all, "sometrue": np.any,
               "product": np.prod, "cumproduct": np.cumprod}.items():
    if not hasattr(np, _n):                      # NumPy-2.0 shim, must precede CosinorPy
        setattr(np, _n, _v)

import scipy.signal as signal          # noqa: E402  (after the shim, per the note above)
from joblib import Parallel, delayed   # noqa: E402  (ships with scikit-learn)
from CosinorPy import cosinor          # noqa: E402  the paper's exact cosinor engine

COSINOR_PARAM_COLS = [
    "Period", "MESOR", "Amplitude", "Magnitude", "Acrophase", "Orthophase", "Bathyphase",
    "P-Value", "Signal to Noise Ratio", "Residual Sum of Squares",
    "Standard Error of Residuals", "Margin of Error",
]
N_PARAMS = len(COSINOR_PARAM_COLS)               # 12 rhythm parameters per (channel, period)
PHASE_COLS = (4, 5, 6)                           # Acrophase / Orthophase / Bathyphase


def periodogram(y, significance_level=0.05):
    """Significant periods of one series, strongest first (Yan et al., 'per' branch).

    The series is sampled on a regular unit grid, so the reference's unique/median
    de-duplication is the identity and its sampling frequency is 1. A period is significant
    when its power exceeds Fisher's threshold ``(1-(alpha/N)^(1/(N-1))) * sum(Pxx)``.
    Returns periods in SAMPLES."""
    f, Pxx = signal.periodogram(y, 1.0)
    f, Pxx = f[1:], Pxx[1:]                                   # drop DC
    keep = Pxx >= (1 - (significance_level / len(y)) ** (1 / (len(y) - 1))) * Pxx.sum()
    per = 1 / f[keep]
    return per[np.argsort(-Pxx[keep])]                        # strongest period first


def calculate_cosinor(y, periods, t):
    """Single-component cosinor at each period -> (len(periods), 12) in COSINOR_PARAM_COLS
    order. `t` is the CLOCK-anchored sample index, so every phase parameter is wall-clock."""
    time, values = pd.Series(t), pd.Series(np.asarray(y, dtype=float))
    out = np.full((len(periods), N_PARAMS), np.nan)
    for i, p in enumerate(periods):
        stats, params = cosinor.fit_me(time, values, n_components=1, period=p, plot=False)[1:3]
        peak = params["peaks"][0] if len(params.get("peaks", [])) else np.nan
        try:
            magnitude = params["heights"][0] - params["heights2"][0]
        except Exception:
            magnitude = params["amplitude"] * 2
        out[i] = [p, params["mesor"], params["amplitude"], magnitude, peak, peak,
                  params["troughs"][0] if len(params.get("troughs", [])) else np.nan,
                  stats["p"], stats["SNR"], stats["RSS"], stats["resid_SE"], stats["ME"]]
    return out


# --------------------------------------------------------------------------------------
# Windowed sensor tensor -> subject-level rhythm-parameter matrix
# --------------------------------------------------------------------------------------
def _start_bins(window_ids, bin_minutes, n):
    """Bin-of-day of each window's FIRST sample, from the id ``f"{pid}_{start.isoformat()}"``.
    Without it the time axis has no calendar origin and every phase parameter is relative to
    an arbitrary per-participant hour -- constant within a person, so it silently breaks every
    BETWEEN-person comparison."""
    if window_ids is None:
        print("[paper_cosinor] WARNING: no window_ids -> phase parameters are NOT "
              "clock-anchored and not comparable across participants", flush=True)
        return np.zeros(n, dtype=int)
    return np.array([(lambda d: (d.hour * 60 + d.minute) // bin_minutes)(
        datetime.fromisoformat(str(w).rsplit("_", 1)[1])) for w in window_ids], dtype=int)


def _channel_features(y, t, fallback_period, top_k, sig_level):
    """(top_k * 12,) for ONE channel: top-k significant periods, cosinor fitted at each.
    Falls back to the circadian period when nothing is significant; unfilled slots stay 0."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            per = periodogram(y, sig_level)[:top_k]
        except Exception:
            per = np.array([])
        if not len(per):
            per = np.array([fallback_period])
        try:
            vals = calculate_cosinor(y, per, t)
        except Exception:
            vals = np.zeros((0, N_PARAMS))
    out = np.zeros(top_k * N_PARAMS, dtype=np.float32)
    k = min(top_k, len(vals))
    out[:k * N_PARAMS] = np.nan_to_num(vals[:k], nan=0.0, posinf=0.0, neginf=0.0).ravel()
    return out


def _window_row(win, t, fallback_period, top_k, sig_level):
    """All channels of ONE window, concatenated -- the unit of parallel work."""
    return np.concatenate([_channel_features(win[:, c], t, fallback_period, top_k, sig_level)
                           for c in range(win.shape[1])]).astype(np.float32)


def _aggregate_to_subject(feats, pids, n_channels, top_k):
    """Collapse the per-window fits to ONE vector per participant (population-mean cosinor,
    Bingham et al. 1982) and write it back over that subject's rows, so the caller's row
    indexing and all downstream masks are unchanged. Phase columns are times inside
    [0, Period), so they are averaged as ANGLES -- an arithmetic mean is wrong across the wrap."""
    feats, pids = np.asarray(feats), np.asarray(pids)
    out = feats.copy()
    for p in np.unique(pids):
        m = np.where(pids == p)[0]
        blk = feats[m].astype(np.float64)
        agg = blk.mean(0)
        for j in range(n_channels * top_k):            # one 12-param block per (channel, period)
            b, per = j * N_PARAMS, blk[:, j * N_PARAMS]
            ok = per > 0
            if ok.any():
                for q in PHASE_COLS:
                    th = np.angle(np.exp(2j * np.pi * blk[ok, b + q] / per[ok]).mean())
                    agg[b + q] = (th % (2 * np.pi)) / (2 * np.pi) * agg[b]
        out[m] = agg.astype(np.float32)
    return out


def _load_cache(path, dim):
    if path and Path(path).exists():
        try:
            z = np.load(path, allow_pickle=False)
            if z["feats"].shape[1] == dim:
                return dict(zip(z["ids"].tolist(), z["feats"]))
        except Exception:
            pass
    return {}


def _save_cache(path, cache, dim):
    if not path or not cache:
        return
    try:                                    # atomic: a killed job must not leave a torn cache
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(Path(path).parent), suffix=".npz")
        os.close(fd)
        np.savez(tmp, ids=np.array(list(cache), dtype=object).astype(str),
                 feats=np.stack(list(cache.values())).astype(np.float32).reshape(-1, dim))
        os.replace(tmp, path)
    except Exception as e:
        print(f"[paper_cosinor] cache not saved ({type(e).__name__}: {e})", flush=True)


def paper_cosinor_features(Xs, bin_minutes, need_mask=None, top_k=2, sig_level=0.05,
                           cache_path=None, window_ids=None, pids=None, verbose=True):
    """Rhythm-parameter matrix (N, C * top_k * 12), one vector per SUBJECT broadcast over
    that subject's windows.

    Xs        : (N, T, C) SENSOR-only windows (already gap-interpolated upstream).
    need_mask : (N,) bool; only these rows are fitted. Unset -> all. Unfitted rows stay zero.
    top_k     : dominant periods per channel (2 mirrors the paper's top-two tables).
    pids      : (N,) participant id; required for the subject-level aggregation.
    cache_path: optional .npz. A fit depends on the window content AND its clock offset, so
                both are in the key -- a stale pre-anchoring cache can never hit.
    """
    Xs = np.asarray(Xs)
    N, T, C = Xs.shape
    dim = C * top_k * N_PARAMS
    feats = np.zeros((N, dim), dtype=np.float32)
    need = np.where(np.ones(N, bool) if need_mask is None else need_mask)[0]
    fallback = 24.0 * 60.0 / bin_minutes                        # circadian period, in bins
    start = _start_bins(window_ids, bin_minutes, N)
    base = (np.asarray(window_ids).astype(str) if window_ids is not None
            else np.arange(N).astype(str))
    ids = np.array([f"{b}@{s}" for b, s in zip(base, start)])

    cache = _load_cache(cache_path, dim)
    todo = [n for n in need if ids[n] not in cache]
    for n in need:
        if ids[n] in cache:
            feats[n] = cache[ids[n]]

    # One CosinorPy fit per (window, channel) is the dominant cost of the whole rhythm
    # analysis, so spread it over the cores the job already reserves. Processes, not threads:
    # the work is pure Python/NumPy inside CosinorPy, so the GIL would serialise a pool.
    if todo:
        n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK") or 0) or (os.cpu_count() or 1)
        n_jobs = max(1, min(n_jobs, len(todo)))
        if verbose:
            print(f"[paper_cosinor] fitting {len(todo)}/{len(need)} windows on {n_jobs} "
                  f"core(s) ({len(need) - len(todo)} cached) ...", flush=True)
        rows = Parallel(n_jobs=n_jobs)(
            delayed(_window_row)(Xs[n], start[n] + np.arange(T), fallback, top_k, sig_level)
            for n in todo)
        for n, row in zip(todo, rows):
            feats[n] = cache[ids[n]] = row

    _save_cache(cache_path, cache, dim)
    if pids is not None:
        feats = _aggregate_to_subject(feats, pids, C, top_k)
    if verbose:
        unit = f"{len(np.unique(pids))} subjects" if pids is not None else "windows (NOT aggregated)"
        print(f"[paper_cosinor] {len(need)} windows x {dim} dims ({C} ch x {top_k} periods "
              f"x {N_PARAMS} params) -> {unit} | {len(todo)} fitted, "
              f"{len(need) - len(todo)} cached", flush=True)
    return feats
