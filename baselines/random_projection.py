"""The random-projection control: a Gaussian readout of the raw window, no encoder.

This is the control that produced the projects central negative result -- on HRD it
reaches AUROC 0.7198, above DSSL, above cosinor and above the supervised ceiling. See
archive_logs.txt section 10.2 for what that measurement rules out.

Extracted verbatim from analysis/random_init_audit.py, which was deleted with the rest
of the one-off probes; experiment_q3 imported only this function.
"""
import numpy as np

def raw_projection(X, n_sensors, width, seed, bands=None, length=None):
    """A Gaussian random projection of the raw window -- no encoder anywhere.

    With `bands`, the window is taken to the frequency domain first and only the listed rFFT
    bin ranges are kept before projecting. That is the only difference between arm 2 and arm 3:
    whether the random mixture is over the whole signal or over the circadian bands alone.
    """
    S = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)
    if S.ndim == 2:
        # An already-flat feature block, so there is no time axis to slice or band-limit.
        # This is the second implementation folded back in: analysis/projection_sanity.py
        # carried its own copy because this one assumed (N, T, C), and the two differed in
        # their scale -- 1/sqrt(d_in) here against 1/sqrt(d_out) there. That difference is
        # invisible to every consumer, because make_probe puts a StandardScaler first and a
        # global factor on all columns cannot survive it; measured, both give AUC 0.654321
        # on the same windows. Which is exactly why two versions could drift apart unnoticed.
        if bands is not None:
            raise ValueError("bands need a (N, T, C) window; got a flat matrix")
        F = S
    elif bands is None:
        F = S[:, :, :n_sensors].reshape(len(S), -1)
    else:
        S = S[:, :, :n_sensors]
        Z = np.fft.rfft(S, axis=1)
        keep = np.concatenate([np.arange(lo, min(hi, Z.shape[1])) for lo, hi in bands])
        z = Z[:, keep]
        F = np.concatenate([z.real.reshape(len(S), -1), z.imag.reshape(len(S), -1)], axis=1)
    F = (F - F.mean(0)) / (F.std(0) + 1e-8)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1.0 / np.sqrt(F.shape[1]), (F.shape[1], int(width)))
    return (F @ W).astype(np.float32)


