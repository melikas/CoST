"""Statistics: a paired participant bootstrap of AUROC differences."""
from __future__ import annotations

import numpy as np


def paired_auc_interval(y, scores_a, scores_b, seed=1, n_boot=2000):
    """Paired stratified participant bootstrap, conditional on the fitted predictions.

    Rows of scores are initialization/repeat, columns are UNIQUE participants. Resample
    the same participants for both methods and every repeat; average repeat AUROCs, not
    probabilities. This interval does not include retraining uncertainty or correct the
    effects of prior test-set model selection. No independent-repeat p-value is invented.
    """
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    a, b = np.atleast_2d(scores_a), np.atleast_2d(scores_b)
    if a.shape != b.shape or a.shape[1] != len(y):
        raise ValueError("paired score matrices must align on repeats and unique participants")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("bootstrap needs finite scores and binary labels")
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if min(len(pos), len(neg)) < 2 or n_boot < 100:
        raise ValueError("need >=2 participants per class and >=100 bootstrap replicates")
    def difference(idx):
        return float(np.mean([roc_auc_score(y[idx], av[idx]) - roc_auc_score(y[idx], bv[idx])
                              for av, bv in zip(a, b)]))
    rng = np.random.default_rng(seed)
    draws = [difference(np.r_[rng.choice(pos, len(pos), replace=True),
                               rng.choice(neg, len(neg), replace=True)]) for _ in range(n_boot)]
    return {"mean_auc_difference": difference(np.arange(len(y))),
            "ci95": np.quantile(draws, [.025, .975]).tolist(), "n_participants": len(y),
            "n_repeats": len(a), "bootstrap_seed": seed, "n_boot": n_boot,
            "uncertainty": "paired participant bootstrap conditional on fitted OOF predictions"}
