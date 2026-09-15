"""Rhythm tools shared by RQ2 and RQ3: 24 h cosinor coefficients, window start days,
personal baselines and deviation scores, phase and amplitude perturbations, and the
stratified concordance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def cosinor_z(Xs, bpd):
    """Per-window, per-channel 24 h cosinor coefficient as one complex number z = a + ib.

    Amplitude is |z| and acrophase is arg z, so a phase shift is exactly a rotation of z.
    Never pooled across channels: sleep runs roughly antiphase to activity and heart rate,
    so a pooled circular mean lands between two different constructs.
    """
    t = 2 * np.pi * np.arange(Xs.shape[1]) / bpd
    Z = Xs - Xs.mean(1)[:, None]
    a = 2 * (Z * np.cos(t)[None, :, None]).mean(1)
    b = 2 * (Z * np.sin(t)[None, :, None]).mean(1)
    return a + 1j * b


def individual_markers(raw_X, observed, bins_per_day, sensor_cols, min_coverage=0.7):
    """Original manuscript targets in physical feature units, one window x channel.

    Cosinor, IS and IV use completed regular-grid windows; coverage is measured BEFORE
    filling. These are explicitly completed-signal estimands, not observations of missing
    rhythms. IS is variance of the average daily profile / total variance. IV is mean
    squared successive difference / total variance, at the recorded sampling interval.
    Activity RA=(M10-L5)/(M10+L5) uses circular 10-hour/5-hour profile means. It is undefined
    on a grid that cannot express those durations (e.g. GLOBEM's six-hour segments).

    See https://ghammad.github.io/pyActigraphy/_modules/pyActigraphy/metrics/metrics.html
    for standard NPCRA definitions. No actigraphy threshold/binarization is imposed here.
    """
    x = np.asarray(raw_X, dtype=float)
    mask = np.asarray(observed, dtype=bool)
    bpd = int(bins_per_day)
    if x.ndim != 3 or mask.shape != x.shape or x.shape[-1] != len(sensor_cols):
        raise ValueError("physical windows, observation masks and sensor names must align")
    if bpd < 3 or x.shape[1] % bpd or x.shape[1] < 2 * bpd:
        raise ValueError("rhythm markers require at least two complete days on a fixed grid")
    if not np.isfinite(x).all() or not 0 <= min_coverage <= 1:
        raise ValueError("completed windows must be finite and coverage must be in [0,1]")
    coverage = mask.mean(axis=1)
    eligible = coverage >= min_coverage
    mean = x.mean(axis=1)
    variance = ((x - mean[:, None]) ** 2).mean(axis=1)
    varying = variance > np.finfo(float).eps * np.maximum(1., mean ** 2)
    profile = x.reshape(len(x), -1, bpd, x.shape[-1]).mean(axis=1)
    z = cosinor_z(x, bpd)
    amp, phase = np.abs(z), np.angle(z)
    identifiable_phase = amp > 1e-6 * np.maximum(1., np.sqrt(variance))
    values = {"MESOR": mean.copy(), "amplitude": amp,
              "phase_cos": np.where(identifiable_phase, np.cos(phase), np.nan),
              "phase_sin": np.where(identifiable_phase, np.sin(phase), np.nan)}
    denominator = np.where(varying, variance, np.nan)
    values["IS"] = ((profile - mean[:, None]) ** 2).mean(axis=1) / denominator
    values["IV"] = (np.diff(x, axis=1) ** 2).mean(axis=1) / denominator
    ra = np.full_like(mean, np.nan)
    durations_resolved = (10 * bpd) % 24 == 0 and (5 * bpd) % 24 == 0
    if durations_resolved:
        k10, k5 = 10 * bpd // 24, 5 * bpd // 24
        m10 = (sum(np.roll(profile, -j, axis=1) for j in range(k10)) / k10).max(axis=1)
        l5 = (sum(np.roll(profile, -j, axis=1) for j in range(k5)) / k5).min(axis=1)
        for c, name in enumerate(sensor_cols):
            if name in ("Steps", "f_steps:fitbit_steps_intraday_rapids_sumsteps"):
                valid = (x[:, :, c].min(axis=1) >= -1e-6) & (m10[:, c] + l5[:, c] > 1e-8)
                np.divide(m10[:, c] - l5[:, c], m10[:, c] + l5[:, c],
                          out=ra[:, c], where=valid)
    values["RA"] = ra
    for value in values.values():
        value[~eligible] = np.nan
    return {"values": values, "coverage": coverage,
            "definition": {"min_observed_coverage": min_coverage,
                           "input": "completed_physical_unit_windows",
                           "IS_IV_sampling_minutes": 1440 / bpd,
                           "RA_grid_supported": durations_resolved,
                           "RA_channels": "step counts only; no binarization",
                           "phase": "time-of-maximum angle relative to window start",
                           "phase_numerical_floor": "1e-6 * max(1, within-window SD)"}}


def window_start_days(window_ids):
    """Elapsed days of each window's start, from the "pid_<isotime>" window id."""
    t = pd.to_datetime([str(w).rsplit("_", 1)[1] for w in window_ids])
    return (t - t.min()).total_seconds().to_numpy() / 86400.0


def personal_baseline(V, pids, R, tdays=None, max_span=None):
    """(mu, sd, ok): each window's reference is its R PRECEDING windows of the same person.

    Windows without a full reference are never scored. `max_span` additionally requires the
    reference to be contiguous in time -- the reference is selected by position, and a
    dropped window silently stretches it (8.6% of scored windows otherwise carry a reference
    spanning more than the nominal R-1 weeks, up to 77 days).
    """
    V, pids = np.asarray(V, dtype=float), np.asarray(pids)
    if R < 1 or len(V) != len(pids):
        raise ValueError("personal reference needs positive R and aligned rows")
    if max_span is not None and (tdays is None or R < 2 or max_span <= 0):
        raise ValueError("a contiguous reference needs timestamps, R>=2 and positive span")
    if tdays is not None:
        tdays = np.asarray(tdays, dtype=float)
        if len(tdays) != len(V) or not np.isfinite(tdays).all():
            raise ValueError("reference timestamps must be finite and aligned")
    mu, sd, ok = np.zeros_like(V), np.ones_like(V), np.zeros(len(V), bool)
    for p in np.unique(pids):
        idx = np.flatnonzero(pids == p)
        if tdays is not None and np.any(np.diff(tdays[idx]) <= 0):
            raise ValueError("windows must be sorted chronologically with unique starts per identifier")
        for j in range(R, len(idx)):
            # Include the current start, not just the span of the reference starts.
            if max_span is not None and not np.allclose(
                    np.diff(tdays[idx[j - R:j + 1]]), max_span / (R - 1), rtol=0, atol=1e-8):
                continue
            ref = V[idx[j - R:j]]
            mu[idx[j]], sd[idx[j]], ok[idx[j]] = ref.mean(0), ref.std(0) + 1e-6, True
    return mu, sd, ok


def dscore(V, mu, sd):
    """Standardised Euclidean distance to the personal baseline: each dimension divided by
    its own within-person SD, then a plain Euclidean norm. Not Mahalanobis -- there is no
    inverse covariance, so correlated dimensions are counted once each."""
    return np.sqrt((((V - mu) / sd) ** 2).mean(1))


def raw_deviation(Zw, zbar):
    """The same functional form as `dscore`, in raw 24 h cosinor space. dd and dg are one
    object measured in two spaces, which is what makes their comparison interpretable."""
    return np.sqrt((np.abs(Zw - zbar) ** 2).mean(1))


def phase_shift(X, hours, n_sensors, bin_minutes):
    """Circular shift of the sensor channels only -- exactly a rotation of z. Clock channels
    are left alone; shifting them would move the reference frame with the signal."""
    Xp = X.copy()
    bins = hours * 60 / bin_minutes
    if not np.isfinite(bins) or not np.isclose(bins, round(bins), rtol=0, atol=1e-8):
        raise ValueError("requested phase shift is not exactly representable on the sampling grid")
    k = int(round(bins))
    Xp[:, :, :n_sensors] = np.roll(X[:, :, :n_sensors], k, axis=1)
    return Xp


def amplitude_scale(X, s, n_sensors, bpd):
    """Scale ONLY the 24 h cosinor component of every sensor channel by `s`.

    x' = x + (s - 1)(a cos wt + b sin wt), where a + ib = cosinor_z(x). Over a window of whole
    days that component is orthogonal to the mean and to every other harmonic, so x' has
    cosinor coefficient exactly s * z. The rhythm's strength changes; its level and timing
    do not. Clock channels are left alone, as in phase_shift.
    """
    Xs = X[:, :, :n_sensors]
    t = 2 * np.pi * np.arange(X.shape[1]) / bpd
    z = cosinor_z(Xs, bpd)
    comp = (z.real[:, None, :] * np.cos(t)[None, :, None]
            + z.imag[:, None, :] * np.sin(t)[None, :, None])
    Xp = X.copy()
    Xp[:, :, :n_sensors] = Xs + (s - 1.0) * comp
    return Xp


def resolve_phase_levels(levels, bin_minutes):
    """Validate the declared shifts; never replace them with a different experiment."""
    if bin_minutes <= 0:
        raise ValueError("bin_minutes must be positive")
    keep = sorted(set(float(lv) for lv in levels))
    for lv in keep:
        k = lv * 60 / bin_minutes
        if not np.isfinite(k) or k <= 0 or not np.isclose(k, round(k), rtol=0, atol=1e-8):
            raise ValueError(f"phase level {lv} h is not expressible at {bin_minutes} min/bin")
    return keep


def stratum_pairs(dd, dg, keys):
    """Per-stratum (U, n_pairs). U counts concordant (dg>0, dg<0) pairs, ties at 1/2, from
    ranks rather than an O(n^2) loop. A stratum with only one sign contributes nothing."""
    out = {}
    for s in np.unique(keys):
        m = (keys == s) & np.isfinite(dd) & np.isfinite(dg)
        pos, neg = dd[m & (dg > 0)], dd[m & (dg < 0)]
        if len(pos) == 0 or len(neg) == 0:
            continue
        r = rankdata(np.r_[pos, neg])
        out[s] = (float(r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2),
                  float(len(pos) * len(neg)))
    return out


def concordance(per_stratum):
    u = sum(v[0] for v in per_stratum.values())
    n = sum(v[1] for v in per_stratum.values())
    return (u / n) if n else float("nan")
