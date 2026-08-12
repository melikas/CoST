"""Between-person rhythm analysis -- GLOBEM only.

Asks whether the representation separates PEOPLE by their rhythm, which needs the many
short windows per participant that GLOBEM's 28-day / 7-day-stride design produces. HRD's
layout cannot support it, so it lives here rather than in tasks/.

Everything else it needs (harmonic_reference, extract_components, the probes) comes from
tasks.decomposition -- the HRD version is the master copy.
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tasks.decomposition import _probe_r2


def _short_label(name, max_len=16):
    """Compact a long sensor-column name for the x-axis only (the CSV/MD keep the full name).
    Uses the curated GLOBEM map when available; otherwise keeps the sensor group before ':'
    (minus the ``f_`` prefix) + the most specific trailing token, hard-capped. Names without
    ':' (e.g. HRD channels) are left as-is apart from the cap -- a no-op for short HRD labels."""
    s = str(name)
    if s in FEATURE_SHORT:
        return FEATURE_SHORT[s]
    if ":" in s:
        group, rest = s.split(":", 1)
        group = group[2:] if group.startswith("f_") else group
        s = f"{group}:{rest.split('_')[-1]}"
    return s if len(s) <= max_len else s[:max_len - 1] + "…"

def _person_descriptors(X, pids, n_sensors, per_day):
    """Per-participant circadian descriptors, one row per participant, per channel.

    All five are scale-free or scale-relative, which matters because the pipeline's X is
    already z-scored WITHIN each participant: absolute level and spread are gone, what
    survives is the SHAPE of that person's daily cycle.

      amp    24 h amplitude of their mean daily profile (FFT fundamental)
      phase  acrophase of that fundamental, carried as (cos, sin) so it stays circular
      IS     interdaily stability -- how reproducible the profile is across days
      IV     intradaily variability -- fragmentation
      RA     relative amplitude (peak-trough)/(peak+trough). Classical actigraphy RA assumes
             NON-NEGATIVE activity, where the trough sits near zero. It is therefore NaN (and
             reported as n/a) whenever the profile dips below zero -- which is always the case
             on the default per-participant z-scored input, where amp already plays its role.
             Run with --no-zscore to obtain a real RA.
    """
    out = {k: [] for k in ("amp", "phase", "IS", "IV", "RA")}
    n_days = X.shape[1] // per_day
    for p in np.unique(pids):
        d = X[pids == p, :n_days * per_day, :n_sensors].reshape(-1, per_day, n_sensors)
        prof = d.mean(axis=0)                                    # (per_day, C) daily profile
        Z = np.fft.rfft(prof, axis=0)[1]                         # 24 h fundamental per channel
        flat = d.reshape(-1, n_sensors)
        tot = flat.var(axis=0)
        safe = np.where(tot > 0, tot, 1.0)
        diff = np.diff(flat, axis=0)
        hi, lo = prof.max(axis=0), prof.min(axis=0)
        den = hi + lo
        out["amp"].append(2.0 * np.abs(Z) / per_day)
        out["phase"].append(np.concatenate([np.cos(np.angle(Z)), np.sin(np.angle(Z))]))
        out["IS"].append(prof.var(axis=0) / safe)
        out["IV"].append((diff ** 2).mean(axis=0) / safe)
        ok_ra = (lo >= 0) & (den > 0)                         # RA needs non-negative activity
        out["RA"].append(np.where(ok_ra, (hi - lo) / np.where(den > 0, den, 1.0), np.nan))
    return {k: np.asarray(v) for k, v in out.items()}

def between_person_rhythm(X, VF, pids, train_mask, test_mask, n_sensors, period_bins, alpha):
    """Held-out R2 of representation -> each PARTICIPANT's own circadian descriptors.

    `rec_full_rhythm` asks whether the representation reconstructs the 24 h waveform of a
    window. That waveform is shared by everyone -- "it is 3 a.m." is the same fact for every
    participant -- so a representation can score high on it while discarding every individual
    deviation from the population rhythm. Depression is hypothesised to live in exactly that
    deviation, which makes the two quantities orthogonal by construction. This measures the
    deviation: a ridge fit on TRAIN participants predicts held-out participants' descriptors
    from their mean representation. Returned per descriptor so the five can be compared.
    """
    per_day = int(round(period_bins))
    keys = ("amp", "phase", "IS", "IV", "RA")
    if per_day < 2 or X.shape[1] < per_day:
        return {f"rec_person_{k}": float("nan") for k in keys}
    uniq = np.unique(pids)
    desc = _person_descriptors(X, pids, n_sensors, per_day)
    feat = np.stack([VF[pids == p].mean(axis=(0, 1)) for p in uniq])
    tr = np.isin(uniq, np.unique(pids[train_mask]))
    te = np.isin(uniq, np.unique(pids[test_mask]))
    res = {}
    for k in keys:
        y = desc[k]
        col = np.isfinite(y).all(axis=0)                         # drop degenerate channels
        if tr.sum() < 10 or te.sum() < 5 or not col.any():
            res[f"rec_person_{k}"] = float("nan")
            continue
        yk = y[:, col]
        pred = Ridge(alpha=alpha).fit(feat[tr], yk[tr]).predict(feat[te])
        true = yk[te]
        ss_tot = ((true - true.mean(axis=0)) ** 2).sum(axis=0)
        ok = ss_tot > 0
        if not ok.any():
            res[f"rec_person_{k}"] = float("nan")
            continue
        r2 = 1 - ((true - pred) ** 2).sum(axis=0)[ok] / ss_tot[ok]
        res[f"rec_person_{k}"] = float((ss_tot[ok] / ss_tot[ok].sum()) @ r2)
    return res
