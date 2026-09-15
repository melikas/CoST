"""Yan et al. periodogram-selected cosinor: the HRD reference-paper rhythm model (RQ3 baseline).

Clone of HAI-lab-UVA/Human-Rhythms-Dataset ``rhythms_cosinor.py`` (Yan et al. 2022, ACM TIST
13(3):47; the HRD paper, Zhao et al. 2025, IMWUT, doi:10.1145/3770698) on CosinorPy
``cosinor.fit_me``: per window and channel, a Fourier periodogram selects the significant
periods and a single-component cosinor is fitted at each, 12 parameters per period. The time
axis starts at the window's clock time, so phase parameters are wall-clock. Window fits are
averaged to one vector per participant, phases as angles. Physical units; labels are never read.
Constant or failed fits are unavailable (NaN), never zero; a missing CosinorPy/seaborn fails.
"""
from datetime import datetime
import os
import warnings

import numpy as np
import pandas as pd
import scipy.signal as signal
from joblib import Parallel, delayed

# CosinorPy 3.1 predates NumPy 2.0; restore the removed aliases it uses before importing it.
for _n, _v in {"round_": np.round, "float_": np.float64, "NaN": np.nan, "NAN": np.nan,
               "Inf": np.inf, "alltrue": np.all, "sometrue": np.any,
               "product": np.prod, "cumproduct": np.cumprod}.items():
    if not hasattr(np, _n):
        setattr(np, _n, _v)

PARAMS = ["Period", "MESOR", "Amplitude", "Magnitude", "Acrophase", "Orthophase", "Bathyphase",
          "P-Value", "Signal to Noise Ratio", "Residual Sum of Squares",
          "Standard Error of Residuals", "Margin of Error"]
PHASES = (4, 5, 6)
TOP_K = 2                       # dominant periods per channel, as in the paper's tables


def periodogram(y, significance_level=0.05):
    """Significant periods in samples, strongest first (Fisher threshold, 'per' branch)."""
    f, pxx = signal.periodogram(y, 1.0)
    f, pxx = f[1:], pxx[1:]
    keep = pxx >= (1 - (significance_level / len(y)) ** (1 / (len(y) - 1))) * pxx.sum()
    return (1 / f[keep])[np.argsort(-pxx[keep])]


def _channel(y, t, fallback):
    from CosinorPy import cosinor
    out = np.full((TOP_K, len(PARAMS)), np.nan)
    if np.ptp(y) <= 1e-12 * max(1.0, np.abs(y).max()):
        return out.ravel()
    periods = periodogram(y)[:TOP_K]
    for i, p in enumerate(periods if len(periods) else [fallback]):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stats, params = cosinor.fit_me(pd.Series(t), pd.Series(y), n_components=1,
                                               period=p, plot=False)[1:3]
            peaks, troughs = params.get("peaks", []), params.get("troughs", [])
            try:
                magnitude = params["heights"][0] - params["heights2"][0]
            except (KeyError, IndexError, TypeError):
                magnitude = params["amplitude"] * 2
            out[i] = [p, params["mesor"], params["amplitude"], magnitude,
                      peaks[0] if len(peaks) else np.nan, peaks[0] if len(peaks) else np.nan,
                      troughs[0] if len(troughs) else np.nan, stats["p"], stats["SNR"],
                      stats["RSS"], stats["resid_SE"], stats["ME"]]
        except (ValueError, np.linalg.LinAlgError, FloatingPointError, KeyError, IndexError):
            continue
    out[~np.isfinite(out)] = np.nan
    return out.ravel()


def _window(win, t, fallback):
    return np.concatenate([_channel(win[:, c].astype(float), t, fallback)
                           for c in range(win.shape[1])])


def yan_cosinor_features(raw_X, window_ids, pids, bin_minutes, keep):
    """(windows, channels*TOP_K*12): each participant's aggregate on all of its `keep` windows.

    Rows outside `keep` stay NaN. Phase columns are times within the period, so they are
    averaged on the circle and mapped back with the participant's mean period.
    """
    from CosinorPy import cosinor  # noqa: F401  (fail before fitting if it or seaborn is absent)
    raw_X, pids = np.asarray(raw_X), np.asarray(pids)
    n, length, channels = raw_X.shape
    start = np.array([(d.hour * 60 + d.minute) // bin_minutes for d in
                      (datetime.fromisoformat(str(w).rsplit("_", 1)[1]) for w in window_ids)])
    rows = np.flatnonzero(keep)
    workers = max(1, min(int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1), len(rows)))
    window = np.full((n, channels * TOP_K * len(PARAMS)), np.nan)
    if len(rows):
        window[rows] = Parallel(n_jobs=workers)(
            delayed(_window)(raw_X[i], start[i] + np.arange(length), 1440 / bin_minutes) for i in rows)
    out = np.full_like(window, np.nan)
    for p in np.unique(pids[rows]):
        m = rows[pids[rows] == p]
        block = window[m]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            agg = np.nanmean(block, axis=0)
        for b in range(0, block.shape[1], len(PARAMS)):
            for q in PHASES:
                ok = np.isfinite(block[:, b]) & np.isfinite(block[:, b + q]) & (block[:, b] > 0)
                if ok.any():
                    angle = np.angle(np.exp(2j * np.pi * block[ok, b + q] / block[ok, b]).mean())
                    agg[b + q] = (angle % (2 * np.pi)) / (2 * np.pi) * agg[b]
        out[m] = agg
    return out
