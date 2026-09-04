"""x = trend + seasonal + residual, the project's own hypothesis, computed explicitly.

The decomposition lives here rather than inside one experiment because three things need
the same one and must not drift: the dispersion ceiling that first used it, the noise-branch
ceiling that chose the positive pair for V^N, and the V^N branch itself.
"""
import numpy as np


def decompose(X, bins_per_day, n_sensors, harmonics=4, poly_degree=3):
    """x = trend + seasonal + residual, the project's own hypothesis, computed explicitly.

    ONE least-squares fit against [polynomial in t | daily harmonics]. The polynomial is the
    trend (the slow drift across the week) and the harmonics are the seasonal part (one fixed
    daily template, which is what cosinor fits). The residual is therefore each day's
    departure from that person's own average day -- the quantity the clinical account calls
    irregularity of routine.

    A moving average would be the obvious way to get the trend and it is the wrong one here.
    A 7-day window is not periodic, so the filter needs padding, and every padding scheme
    biases the first and last day: measured on a planted rhythm-plus-drift signal, circular
    padding left 14%% of the rhythm in the residual, edge replication 7%%, odd reflection 5%%
    -- and that edge error then propagates through the global harmonic fit into the interior.
    A design-matrix fit has no edges at all and recovers the same signal exactly (residual rms
    0.00000 on that test).
    """
    S = np.nan_to_num(np.asarray(X[:, :, :n_sensors], dtype=float), nan=0.0)
    n, T, C = S.shape
    w = int(bins_per_day)
    t = np.arange(T)
    u = (t - t.mean()) / (t.std() + 1e-12)
    cols = [u ** k for k in range(poly_degree + 1)]
    n_trend = len(cols)
    for k in range(1, harmonics + 1):
        cols += [np.cos(2 * np.pi * k * t / w), np.sin(2 * np.pi * k * t / w)]
    D = np.stack(cols, axis=1)
    coef = np.linalg.lstsq(D, S.transpose(1, 0, 2).reshape(T, -1), rcond=None)[0]
    unflat = lambda A: A.reshape(T, n, C).transpose(1, 0, 2)
    trend = unflat(D[:, :n_trend] @ coef[:n_trend])
    seasonal = unflat(D[:, n_trend:] @ coef[n_trend:])
    return trend, seasonal, S - trend - seasonal
