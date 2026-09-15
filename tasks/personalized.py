"""RQ2: unlabeled personal baselines and within-person rhythmic deviation detection."""
from __future__ import annotations

import numpy as np

from tasks.rhythm import (amplitude_scale, cosinor_z, dscore, personal_baseline,
                          phase_shift, raw_deviation, resolve_phase_levels,
                          window_start_days)

BASELINE_WEEKS = 4
PHASE_HOURS = (0.5, 1.0, 2.0, 3.0, 4.0)
AMPLITUDE_FRACTIONS = (0.05, 0.1, 0.2, 0.3, 0.5)


def _reference(model, X, physical_X, pids, window_ids, test_ids, bins_per_day):
    """Freeze four preceding contiguous weeks in representation and raw rhythm space."""
    tdays = window_start_days(window_ids)
    representations = model.encode(X, parts=False, batch_size=16)
    # personal_baseline checks all four reference-to-current gaps are exactly seven days.
    mu, sd, eligible = personal_baseline(
        representations, pids, BASELINE_WEEKS, tdays,
        max_span=7.0 * (BASELINE_WEEKS - 1))
    z = cosinor_z(physical_X, bins_per_day)
    zbar = np.zeros_like(z)
    for pid in np.unique(pids):
        idx = np.flatnonzero(pids == pid)
        for j in range(BASELINE_WEEKS, len(idx)):
            zbar[idx[j]] = z[idx[j-BASELINE_WEEKS:j]].mean(0)
    eligible &= np.isin(pids, test_ids)
    return dict(mu=mu, sd=sd, d0=dscore(representations, mu, sd), z=z,
                zbar=zbar, g0=raw_deviation(z, zbar), eligible=eligible)


def personalized_records(model, method, X, physical_X, pids, window_ids, test_ids,
                         bins_per_day, bin_minutes):
    """Return per-window controlled-deviation evidence; endpoint labels are never accepted.

    Timing rows compare a shifted week with its unmodified version against one frozen
    personal baseline. Strength rows compare equally sized attenuation/amplification of
    the same week's 24-hour component. `representation_delta` and `raw_delta` therefore
    have matching signs when the representation detects the known rhythmic deviation.
    """
    if X.shape[1] / bins_per_day != 7:
        return [], {"status": "not_applicable",
                    "reason": "requires non-overlapping seven-day windows and four preceding weeks"}
    levels = resolve_phase_levels(PHASE_HOURS, bin_minutes)
    keep = np.isin(pids, test_ids)
    X, physical_X, pids, window_ids = X[keep], physical_X[keep], pids[keep], np.asarray(window_ids)[keep]
    ref = _reference(model, X, physical_X, pids, window_ids, test_ids, bins_per_day)
    rows = []
    for hours in levels:
        changed = phase_shift(X, hours, X.shape[-1], bin_minutes)
        physical_changed = phase_shift(physical_X, hours, physical_X.shape[-1], bin_minutes)
        d = dscore(model.encode(changed, parts=False, batch_size=16), ref["mu"], ref["sd"])
        g = raw_deviation(cosinor_z(physical_changed, bins_per_day), ref["zbar"])
        for i in np.flatnonzero(ref["eligible"] & np.isfinite(d) & np.isfinite(g)):
            rows.append(dict(method=method, participant=str(pids[i]), window_id=str(window_ids[i]),
                             perturbation="phase", level=float(hours),
                             representation_delta=float(d[i]-ref["d0"][i]),
                             raw_delta=float(g[i]-ref["g0"][i])))
    for fraction in AMPLITUDE_FRACTIONS:
        low = amplitude_scale(X, 1.0-fraction, X.shape[-1], bins_per_day)
        high = amplitude_scale(X, 1.0+fraction, X.shape[-1], bins_per_day)
        physical_low = amplitude_scale(physical_X, 1.0-fraction, physical_X.shape[-1], bins_per_day)
        physical_high = amplitude_scale(physical_X, 1.0+fraction, physical_X.shape[-1], bins_per_day)
        d_low = dscore(model.encode(low, parts=False, batch_size=16), ref["mu"], ref["sd"])
        d_high = dscore(model.encode(high, parts=False, batch_size=16), ref["mu"], ref["sd"])
        g_low = raw_deviation(cosinor_z(physical_low, bins_per_day), ref["zbar"])
        g_high = raw_deviation(cosinor_z(physical_high, bins_per_day), ref["zbar"])
        for i in np.flatnonzero(ref["eligible"] & np.isfinite(d_low) & np.isfinite(d_high)):
            rows.append(dict(method=method, participant=str(pids[i]), window_id=str(window_ids[i]),
                             perturbation="amplitude", level=float(fraction),
                             representation_delta=float(d_high[i]-d_low[i]),
                             raw_delta=float(g_high[i]-g_low[i])))
    return rows, {"status": "complete", "baseline_weeks": BASELINE_WEEKS,
                  "phase_hours": list(levels), "amplitude_fractions": list(AMPLITUDE_FRACTIONS),
                  "unit": "held-out participant-week", "uses_endpoint_labels": False,
                  "baseline": "four preceding contiguous non-overlapping weeks"}
