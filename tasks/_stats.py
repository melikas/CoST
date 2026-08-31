"""Statistics shared by the experiment scripts. NumPy and SciPy only -- deliberately.

This module exists so a step that reads nothing but saved JSON does not have to import
torch to get a t-test. `paired` lived in experiment_readout.py, which imports torch,
model_build and the task package at module scope; `rhythm_stability.py --aggregate`
reads only per-seed JSON files and could not run on a login node because of it.
"""
import numpy as np


def paired(a, b, n_splits):
    """Corrected resampled t-test over repeated-CV folds (Nadeau & Bengio, 2003).

    A NAIVE paired t over repeated k-fold folds is anti-conservative and must not be used
    here: the folds share training participants, so the fold differences are positively
    correlated and the ordinary variance estimate is too small -- p-values come out far below
    their true value. On the first run of this script one contrast moved from p=0.0036 naive
    to p=0.41 corrected, i.e. from "significant" to nothing.

    The correction inflates the variance of the mean by (1/n + n_test/n_train). For k-fold
    that ratio is 1/(k-1), so

        t = mean(d) / sqrt( (1/n + 1/(k-1)) * var(d) ),   df = n - 1.

    Both the naive and the corrected p are returned, so the size of the correction is visible
    rather than hidden.
    """
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    n = len(d)
    if n < 3 or d.std(ddof=1) == 0:
        return dict(diff=float(d.mean()) if n else np.nan, p=np.nan, p_naive=np.nan,
                    wins=0, n=int(n), dz=np.nan)
    var = d.var(ddof=1)
    t_corr = d.mean() / np.sqrt((1.0 / n + 1.0 / (n_splits - 1)) * var)
    p_corr = 2 * stats.t.sf(abs(t_corr), n - 1)
    return dict(diff=float(d.mean()), p=float(p_corr),
                p_naive=float(stats.ttest_rel(a[ok], b[ok]).pvalue),
                wins=int((d > 0).sum()), n=int(n), dz=float(d.mean() / np.sqrt(var)))
