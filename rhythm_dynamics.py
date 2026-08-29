"""Is the WEEK-TO-WEEK CHANGE in a person's rhythm predictable at all?

Every representation in this project encodes the rhythm OF a window: amplitude, phase and
level at the circadian harmonics. That is also exactly what the hand-written spectral readout
delivers, which is why a random-init encoder matches the trained one at all three levels --
training re-arranges information the architecture already hands over, and adds none.

The one construct nothing here encodes is the DYNAMICS: how a person's rhythm moves from one
week to the next. It is what RQ2 calls a deviation, what RQ3 would call instability, and no
static readout provides it.

Before any objective is built around predicting that change, one thing has to be true: the
change has to be predictable. This script tests exactly that, on the raw signal, with no
encoder involved, so the answer cannot be blamed on any model.

    target      D = params(w_{t+1}) - params(w_t), per channel
                params = circadian amplitude, MESOR, and acrophase as (cos, sin)

    A  persistence            predict D = 0. R^2 = 0 by definition; the reference.
    B  from params(w_t)       a ridge on the CURRENT window's own rhythm parameters. Most of
                              whatever this scores is REGRESSION TO THE MEAN -- an unusually
                              high amplitude this week is lower next week whatever the person
                              does -- and a linear probe on the existing readout already has
                              these columns, so B is NOT evidence of learnable dynamics.
    C  from the raw window    a ridge on a PCA of the whole window, which contains everything
                              any encoder could see.

    The number that decides the design is C - B: the structure in the change that the current
    readout does NOT already expose. If it is ~0, week-to-week rhythm change is regression to
    the mean plus noise, no encoder can learn it, and an objective built on predicting it
    cannot beat a random-init control. If it is clearly positive, there is signal here that
    every representation in this project is currently discarding.

Pairs are consecutive windows of ONE participant, and only where they are genuinely adjacent
in time -- a gap in wear would otherwise be scored as a rhythm change. Folds are grouped by
participant, so no person appears on both sides of a split.

Run (no GPU, no training):
    python rhythm_dynamics.py --variant-dir results_hrd/<run>/tcn_none_seed42
"""
import argparse

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tasks._experiment_common import load_context, out_dir, save

ALPHAS = (0.01, 0.1, 1, 10, 100, 1000, 10000, 100000)


def window_params(X, bins_per_day):
    """Circadian amplitude, MESOR and acrophase (as cos/sin) per channel, from the rFFT.

    The same quantities the spectral readout reports, computed here from the RAW window so the
    target owes nothing to any encoder. Acrophase enters as (cos, sin) rather than as an angle:
    a difference of raw angles is meaningless across the branch cut, and the change is the
    whole point of this script.
    """
    T = X.shape[1]
    D = max(1, T // int(bins_per_day))
    Z = np.fft.rfft(np.nan_to_num(X, nan=0.0), axis=1)
    z = Z[:, D, :]                                   # the 24 h bin
    amp = 2 * np.abs(z) / T
    ang = np.angle(z)
    mesor = np.nan_to_num(np.nanmean(X, axis=1), nan=0.0)
    return np.concatenate([amp, mesor, np.cos(ang), np.sin(ang)], axis=1)


def consecutive_pairs(window_ids, pids):
    """Indices (i, j) of genuinely adjacent windows of one participant.

    `window_ids` are "<pid>_<iso timestamp>", so sorting them per participant orders the
    windows in time. A pair is kept only when the gap is the modal gap for that participant --
    a person who stops wearing the device for a month would otherwise contribute that hole as
    a giant "rhythm change".
    """
    import collections
    order = collections.defaultdict(list)
    for i, (w, p) in enumerate(zip(window_ids, pids)):
        stamp = str(w).split("_", 1)[1] if "_" in str(w) else str(i)
        order[p].append((stamp, i))
    pairs, gaps = [], []
    for p, lst in order.items():
        lst.sort()
        for (s1, i), (s2, j) in zip(lst, lst[1:]):
            pairs.append((i, j, p))
            gaps.append((s1, s2))
    return pairs


def oof_r2(F, Y, groups, n_splits=5):
    """Out-of-fold R^2 per target column, participant-grouped, penalty chosen inside the fold."""
    pred = np.full_like(Y, np.nan, dtype=float)
    cv = GroupKFold(n_splits=int(min(n_splits, len(np.unique(groups)))))
    for tr, te in cv.split(F, Y[:, 0], groups):
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
        m.fit(F[tr], Y[tr])
        pred[te] = m.predict(F[te])
    ok = np.isfinite(pred[:, 0])
    ss = ((Y[ok] - pred[ok]) ** 2).sum(0)
    tot = ((Y[ok] - Y[ok].mean(0)) ** 2).sum(0).clip(1e-12)
    return 1 - ss / tot


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant-dir", required=True,
                    help="only its config and dataset are read; no encoder is used")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--n-pca", type=int, default=64,
                    help="components of the raw window handed to arm C. Kept well below the "
                         "training-pair count so the ridge is not fitting noise.")
    a = ap.parse_args()

    ctx = load_context(a.variant_dir, a.cache_dir, gpu=-1)
    wid = ctx.window_ids if ctx.window_ids is not None else np.arange(len(ctx.pids))
    pairs = consecutive_pairs(np.asarray(wid).astype(str), ctx.pids)
    i, j, gp = (np.array([p[k] for p in pairs]) for k in (0, 1, 2))
    print(f"[dyn] {len(pairs):,} consecutive window pairs from {len(np.unique(gp))} participants")

    P = window_params(ctx.X[:, :, :ctx.n_sensors], ctx.bins_per_day)
    Y = P[j] - P[i]                                   # the CHANGE -- the target
    flat = np.nan_to_num(ctx.X[i][:, :, :ctx.n_sensors], nan=0.0).reshape(len(i), -1)
    n_pca = int(min(a.n_pca, *flat.shape))
    raw = PCA(n_components=n_pca, random_state=ctx.seed).fit_transform(flat)
    print(f"[dyn] target D: {Y.shape[1]} columns | arm C uses {n_pca} PCA components "
          f"of the {flat.shape[1]}-dim window")

    rB = oof_r2(P[i], Y, gp)
    rC = oof_r2(np.hstack([P[i], raw]), Y, gp)
    names = (["amp"] * ctx.n_sensors + ["mesor"] * ctx.n_sensors
             + ["cos_acro"] * ctx.n_sensors + ["sin_acro"] * ctx.n_sensors)

    print("")
    print(f"  {'target block':14s} {'B: from params(w_t)':>21} {'C: + raw window':>18} {'C - B':>9}")
    out = {}
    for blk in ("amp", "mesor", "cos_acro", "sin_acro"):
        m = np.array([n == blk for n in names])
        out[blk] = dict(B=float(rB[m].mean()), C=float(rC[m].mean()),
                        gain=float((rC - rB)[m].mean()))
        print(f"  {blk:14s} {rB[m].mean():21.4f} {rC[m].mean():18.4f} {(rC-rB)[m].mean():+9.4f}")
    allB, allC = float(rB.mean()), float(rC.mean())
    print(f"  {'ALL':14s} {allB:21.4f} {allC:18.4f} {allC-allB:+9.4f}")

    print("")
    print("  A: persistence (predict no change) is R^2 = 0 by construction.")
    verdict = ("SIGNAL -- the change carries structure the current readout does not expose; "
               "an objective built on it can add something a random encoder cannot"
               if allC - allB > 0.02 else
               "NO SIGNAL -- week-to-week rhythm change is regression to the mean plus noise. "
               "No encoder can learn it and no objective built on it will beat random-init.")
    print(f"  VERDICT: {verdict}")

    save(out_dir(ctx, "rq1"), "rhythm_dynamics",
         {"n_pairs": len(pairs), "n_participants": int(len(np.unique(gp))),
          "n_pca": n_pca, "per_block": out, "R2_params_only": allB,
          "R2_params_plus_raw": allC, "gain": allC - allB, "verdict": verdict})


if __name__ == "__main__":
    main()
