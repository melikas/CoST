"""Downstream probes, and the scaler that stops per-column standardisation from destroying
the geometry of a circular phase readout.

THE BUG THIS FIXES. The seasonal readout emits phase either as a raw angle
(`phase_readout='angle'`) or as a unit-circle pair (`phase_readout='circular'`, columns
[amp | cos | sin]). Every probe begins with `StandardScaler`, which centres and scales each
column INDEPENDENTLY:

    cos' = (cos - mu_c) / sigma_c        sin' = (sin - mu_s) / sigma_s

When acrophases are concentrated -- and they are, daily phase concentration R ~ 0.48-0.56 --
sigma_c != sigma_s, so the map is an anisotropic scaling. It takes the unit circle to an
ellipse, and angular distance is no longer proportional to Euclidean distance: the readout
that was constructed precisely so that ||(cos_i,sin_i) - (cos_j,sin_j)|| is monotone in the
angular gap loses that property before the classifier ever sees it.

That is the leading explanation for why run 2412728 (circular readout, C=0.8289) lost to run
2224103 (angle readout, C=0.8743) despite the circular readout being the better-motivated
one. Under `angle`, phases barely wrap at R~0.5, so the raw angle is near-linear and survives
per-column scaling; under `circular` the shear bites.

THE FIX. Scale each (cos, sin) pair ISOTROPICALLY: one shared scale for both columns, taken
as the RMS of their two standard deviations. Centring stays per-column, because subtracting
the pair's mean vector is a translation of the point cloud and translations preserve angles.
The composition is a similarity transform -- translation plus uniform scaling -- so the
circle stays a circle and angular distance stays proportional to Euclidean distance.

This makes the comparison between the two readouts fair. It does not prejudge it: `angle`
is untouched, and if `circular` still loses under isotropic scaling then the readout, not
the scaler, is what costs the signal.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

__all__ = ["IsotropicPairScaler", "phase_block_layout", "make_probe",
           "best_threshold", "balanced_accuracy"]


def balanced_accuracy(y, pred):
    """(sensitivity + specificity) / 2. Chance is exactly 0.5 at any prevalence."""
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    p, n = (y == 1), (y == 0)
    if not p.any() or not n.any():
        return float("nan")
    return float(((pred[p] == 1).mean() + (pred[n] == 0).mean()) / 2)


def best_threshold(y, prob):
    """Threshold maximising balanced accuracy, to be chosen on the TRAINING split ONLY.

    Balanced accuracy needs an operating point, and picking it on the test split would be
    fitting the metric to the answer -- the single easiest way to manufacture a benchmark
    win. Selected here on training predictions and then applied unchanged to the test set.

    Computed by sweeping every distinct score with cumulative sums rather than looping, so
    it stays O(n log n) and can run per fold per rung without being noticed.
    """
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=float)
    P, N = int((y == 1).sum()), int((y == 0).sum())
    if P == 0 or N == 0:
        return 0.5
    o = np.argsort(-prob, kind="mergesort")
    ys = y[o]
    tp = np.cumsum(ys == 1)                      # predicted positive = the top k scores
    fp = np.cumsum(ys == 0)
    bacc = (tp / P + (N - fp) / N) / 2.0
    k = int(np.argmax(bacc))
    ps = prob[o]
    # Sit strictly below the last included score so the comparison `prob >= thr` reproduces
    # exactly the top-k split that was scored.
    return float(ps[k] - 1e-12 if k + 1 >= len(ps) else (ps[k] + ps[k + 1]) / 2.0)


class IsotropicPairScaler(BaseEstimator, TransformerMixin):
    """StandardScaler, except that two designated column blocks are scaled together.

    Columns [0, pair_start) are standardised per column, as usual. The two blocks
    [pair_start, pair_start + width) and [pair_start + width, pair_start + 2*width) are the
    cos and sin halves of a phase readout: column i of the first pairs with column i of the
    second, and both get the same scale, so their plane is rescaled uniformly.

    With `pair_start=None` this is exactly StandardScaler, which is what the `angle` arm
    gets -- the two arms differ in the readout, not in the scaler's treatment of a block
    that only one of them has.
    """

    def __init__(self, pair_start=None, width=0):
        self.pair_start = pair_start
        self.width = width

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale < 1e-12] = 1.0
        if self.pair_start is not None and self.width > 0:
            a = int(self.pair_start)
            b = a + int(self.width)
            c = b + int(self.width)
            if c > X.shape[1]:
                raise ValueError(
                    f"phase pair block [{a}:{c}) exceeds {X.shape[1]} columns")
            # One scale per (cos_i, sin_i) pair: the RMS of the two column deviations.
            # Using the RMS rather than either column alone keeps the total variance of
            # the pair at 2, matching what per-column standardisation would have given,
            # so the phase block does not silently change weight relative to amplitude.
            shared = np.sqrt((scale[a:b] ** 2 + scale[b:c] ** 2) / 2.0)
            shared[shared < 1e-12] = 1.0
            scale[a:b] = shared
            scale[b:c] = shared
        self.scale_ = scale
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_


def spectral_freqs(seq_len, bins_per_day, harmonics=4):
    """The rFFT bins the seasonal readout reports: 1 cycle per window (circaseptan), 1 per
    day (circadian), and its harmonics up to `harmonics` cycles per day, dropping any above
    Nyquist. The one definition: the readout, its width and the pair-scaler layout read it.

        HRD    T=672, 96/day -> D=7  -> [1, 7, 14, 21, 28]        (5, harmonics=4)
                                     -> [1, 7, 14, ..., 84]       (13, harmonics=12)
        GLOBEM T=112,  4/day -> D=28 -> [1, 28, 56]               (3, any harmonics >= 2)
    """
    D = max(1, seq_len // int(bins_per_day))
    return [i for i in (1, *(k * D for k in range(1, harmonics + 1))) if 0 < i <= seq_len // 2]


def phase_block_layout(readout, seasonal_dims, seq_len, bins_per_day, n_leading=0,
                       harmonics=4, block=None):
    """Where the (cos, sin) blocks sit in a representation, for `IsotropicPairScaler`.

    Two offsets have to be right, and getting either wrong is worse than not scaling at all,
    because it couples columns that are not a pair and leaves the real pair sheared:

      * EACH BLOCK IS |f| * seasonal_dims WIDE, not seasonal_dims. The readout stacks blocks
        shaped (b, |f|, d) flattened to |f|*d, and HRD reports |f| = 5 bins at the default
        4 harmonics, so a block is 5 * 160 = 800 columns (13 * 160 = 2080 at 12).
      * THE TREND BLOCK COMES FIRST in the full representation. `CoST.encode` concatenates
        [trend | amp | cos | sin], so on HRD the cos block starts at 160 + 800 = 960. Pass
        `n_leading=trend_dims` when probing the full vector, and `n_leading=0` when probing
        the seasonal block on its own.

    `block` overrides the width for a readout that keeps fewer columns (the band-matched
    one); pass the encoded amplitude block's width.

    Returns (pair_start, width), or (None, 0) for the 'angle' readout, which has no pair.
    """
    if readout != "circular":
        return None, 0
    if block is None:
        block = len(spectral_freqs(seq_len, bins_per_day, harmonics)) * int(seasonal_dims)
    return int(n_leading) + int(block), int(block)


def make_probe(mode, C, seed, n_pca=0, pair_start=None, pair_width=0):
    """Probe factory. `pair_start`/`pair_width` switch the scaler to isotropic pair scaling;
    the default is plain per-column standardisation, i.e. the previous behaviour exactly.

    The scaler sits inside the pipeline so it is refit on each fold's training participants
    and never sees the held-out ones.
    """
    if mode == "forest":
        # A second probe family, selected on validation exactly like the penalty and applied
        # identically to every arm. It is here because the probe was measurably the binding
        # constraint: on the GLOBEM benchmark protocol the raw window scores 0.5279 through
        # a logistic probe and 0.5507 through a forest, against a published best of 0.547.
        # C selects the depth budget rather than a penalty, so one grid keeps its meaning
        # across both families. Trees are scale-invariant, so the scaler is irrelevant here.
        return RandomForestClassifier(
            n_estimators=400, random_state=seed, n_jobs=-1, class_weight="balanced",
            min_samples_leaf=max(1, int(round(1.0 / max(C, 1e-3)))))
    if mode != "supervised":
        raise ValueError(f"unknown probe mode: {mode!r}")

    steps = [IsotropicPairScaler(pair_start, pair_width) if pair_width
             else StandardScaler()]
    if n_pca and n_pca > 0:
        steps.append(PCA(n_components=int(n_pca), random_state=seed))
    steps.append(LogisticRegression(C=C, max_iter=3000,
                                    class_weight="balanced", random_state=seed))
    return make_pipeline(*steps)
