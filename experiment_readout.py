"""What did PRETRAINING actually change, block by block?

The first version of this script asked whether the wrapped-angle phase readout was costing the
probe anything. It is not: cos/sin against raw atan2 came out at -0.0062 (p=0.86) on the
composed representation and -0.0189 (p=0.63) on the phase block alone, so that question is
closed and those cells are gone.

What the same run turned up instead was an asymmetry nobody had tested. Comparing the frozen
DSSL encoder against its own random-init control:

    with the phase block      DSSL 0.6896   random-init 0.6807
    phase deleted             DSSL 0.6440   random-init 0.6985

Pretraining appears to IMPROVE the phase block and DEGRADE the amplitude block, netting out to
nothing. Two independent measurements agree with that reading:

  * the alpha sweep (runs 1568399/400/401, never analysed): raising the weight of the seasonal
    loss makes DSSL monotonically worse against the same control -- -0.0086 at alpha=0.05,
    -0.0129 at 0.5, -0.0498 at 5.0 (p=0.013). More seasonal training, worse representation.

  * cost.py itself is asymmetric. The phase is mapped to the unit circle before the contrastive
    term (`circular_phase`, on the argument that a dot product on raw angles "is not a
    similarity between angles at all"). The amplitude is NOT normalised and gets no
    temperature, while `instance_contrastive_loss` scores pairs with a bare dot product:

        sim = torch.matmul(z, z.transpose(1, 2));  logits = -F.log_softmax(logits, -1)

    On |F| >= 0 -- a non-negative vector -- that dot product is dominated by magnitude rather
    than direction, and the MoCo trend branch's temperature (T=0.07) does not apply here.

So the hypothesis under test is: pretraining helps the phase block and hurts the amplitude
block, because the loss normalises one and not the other.

That is currently an observation about MEANS, from cells that were not designed to isolate
blocks (the old "amp only" cell still carried the trend half) and whose per-fold values were
never saved. This version isolates the three blocks, runs each through both models, and tests
the difference-in-differences that the hypothesis actually predicts.

  blocks      T   V^(T) trend only
              A   seasonal amplitude only, at the 5 chronobiological harmonics
              P   seasonal phase only, same harmonics
              TA  T + A         (what remains if the phase block is dropped)
              R1  T + A + P     (the readout RQ3's ladder probes)

  contrasts   DSSL - random-init, per block  -- what pretraining bought, or cost, in each
              (P effect) - (A effect)        -- the difference-in-differences the story needs
              R1 - TA, per model             -- what the phase block adds to each

Cells are paired over repeated-CV folds and compared with the corrected resampled t-test; see
`paired`. Per-fold arrays are saved this time, so any further contrast can be tested without
re-running.

Run (no GPU needed), once per available encoder:
    python experiment_readout.py --variant-dir results_hrd/<run>/tcn_none_seed42
"""
import argparse

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from model_build import encode_repr
from tasks._stats import paired
from tasks._eval_protocols import fit_persubject_probe, persubject_rows
from tasks._experiment_common import load_context, out_dir, random_init_model, save, write_csv


# ----------------------------------------------------------------------------- readout cells
def build_cells(model, X, cfg):
    """The three blocks of the RQ3 readout, isolated, plus the two compositions.

    `encode_repr` is the production readout RQ3's ladder probes; it returns
    [V^(T) | amp | phi] with the seasonal half read at the 5 chronobiological harmonics, so
    the blocks below are slices of the very vector the ladder scores -- nothing is recomputed.
    """
    with torch.no_grad():
        full = encode_repr(model, X, cfg)                       # (N, dS + 10*dS)
    dS = model.net.component_dims
    trend, seas = full[:, :dS], full[:, dS:]
    assert seas.shape[1] % 2 == 0, seas.shape
    nf = seas.shape[1] // 2                                     # [amp | phi], equal blocks
    amp, phi = seas[:, :nf], seas[:, nf:]
    return {
        "T  trend only":        trend,
        "A  amp only":          amp,
        "P  phase only":        phi,
        "TA trend+amp":         np.hstack([trend, amp]),
        "R1 trend+amp+phase":   np.hstack([trend, amp, phi]),
    }


# ----------------------------------------------------------------------------- evaluation
def cv_auc(feat, pids, y, pool_pids, pid_label, n_splits, n_repeats, seed):
    """Out-of-fold participant AUC, one value per (repeat, fold).

    The probe is refit inside every fold and its penalty selected on a participant-disjoint
    slice of that fold's TRAINING half, so no held-out participant touches either the fit or
    the model selection. Folds are stratified on the participant label.
    """
    pool = np.asarray(sorted(pool_pids))
    lab = np.array([pid_label[p] for p in pool])
    aucs = []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for tr_i, te_i in skf.split(pool, lab):
            tr_p, te_p = pool[tr_i], pool[te_i]
            # penalty-selection split, disjoint from the fitting participants
            rng = np.random.default_rng(seed + r)
            perm = rng.permutation(len(tr_p))
            n_val = max(2, int(round(0.2 * len(tr_p))))
            val_p, fit_p = tr_p[perm[:n_val]], tr_p[perm[n_val:]]
            if len(set(pid_label[p] for p in val_p)) < 2:        # need both classes to select
                val_p, fit_p = tr_p, tr_p
            fit_m, val_m, te_m = (np.isin(pids, fit_p), np.isin(pids, val_p),
                                  np.isin(pids, te_p))
            clf = fit_persubject_probe(feat, pids, y, fit_m, val_m, seed)
            Xs, ys, _ = persubject_rows(feat, pids, y, te_m)
            aucs.append(roc_auc_score(ys, clf.predict_proba(Xs)[:, 1])
                        if len(set(ys)) > 1 else np.nan)
    return np.array(aucs, float)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant-dir", required=True,
                    help="a variant directory that still holds encoder.pt (the sweep keeps it "
                         "only for the first seed)")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=10,
                    help="repeats of the k-fold; the pairing unit is (repeat, fold)")
    a = ap.parse_args()

    ctx = load_context(a.variant_dir, a.cache_dir, gpu=-1)       # CPU: no GPU is needed
    # The probe pool is the labelled cohort minus the run's own test participants -- the same
    # participants train_hrd's internal CV uses. The transductive caveat (the encoder was
    # pretrained on their unlabelled windows) is identical for every cell, so it cannot
    # explain a DIFFERENCE between blocks, which is what this script measures.
    pool = sorted({p for p in ctx.pid_label if ctx.train_mask[ctx.pids == p].any()
                   or ctx.val_mask[ctx.pids == p].any()})
    print(f"[readout] {ctx.tag} seed={ctx.seed} | {len(pool)} pool participants | "
          f"{a.repeats}x{a.folds}-fold = {a.repeats * a.folds} paired folds")

    rows, curves = [], {}
    for tag, model in (("DSSL", ctx.model), ("Random-init", random_init_model(ctx))):
        for name, feat in build_cells(model, ctx.X, ctx.cfg).items():
            auc = cv_auc(feat, ctx.pids, ctx.y, pool, ctx.pid_label,
                         a.folds, a.repeats, ctx.seed)
            curves[(tag, name)] = auc
            rows.append((tag, name, feat.shape[1], float(np.nanmean(auc)),
                         float(np.nanstd(auc, ddof=1))))
            print(f"  {tag:12s} {name:22s} dim={feat.shape[1]:5d}  "
                  f"AUC {np.nanmean(auc):.4f} +-{np.nanstd(auc, ddof=1):.4f}")

    D, R = "DSSL", "Random-init"
    blocks = ["T  trend only", "A  amp only", "P  phase only",
              "TA trend+amp", "R1 trend+amp+phase"]
    # per-fold "what pretraining bought" in each block -- the arrays every contrast is built on
    gain = {b: curves[(D, b)] - curves[(R, b)] for b in blocks}

    tests = {}
    for b in blocks:
        tests[f"pretraining effect on {b}"] = (curves[(D, b)], curves[(R, b)])
    # The difference-in-differences the hypothesis predicts: pretraining should help the phase
    # block MORE than the amplitude block. This is the contrast that decides the story, and it
    # is paired fold-by-fold like everything else.
    tests["DiD  (phase gain) - (amp gain)"] = (gain["P  phase only"], gain["A  amp only"])
    # What the phase block adds on top of trend+amp, per model, and whether that differs.
    tests["phase adds, DSSL        (R1-TA)"] = (curves[(D, "R1 trend+amp+phase")],
                                                curves[(D, "TA trend+amp")])
    tests["phase adds, random-init (R1-TA)"] = (curves[(R, "R1 trend+amp+phase")],
                                                curves[(R, "TA trend+amp")])
    tests["DiD  phase-adds, DSSL - random"] = (
        curves[(D, "R1 trend+amp+phase")] - curves[(D, "TA trend+amp")],
        curves[(R, "R1 trend+amp+phase")] - curves[(R, "TA trend+amp")])

    print("")
    print(f"  {'contrast':40s} {'diff':>8} {'wins':>8} {'p corr':>9} {'p naive':>9} {'dz':>7}")
    res = {}
    for label, (x, y) in tests.items():
        r = paired(x, y, a.folds)
        res[label] = r
        star = "" if not np.isfinite(r["p"]) else (" ***" if r["p"] < .001 else
                                                   " **" if r["p"] < .01 else
                                                   " *" if r["p"] < .05 else "")
        print(f"  {label:40s} {r['diff']:+8.4f} {r['wins']:4d}/{r['n']:<3d} "
              f"{r['p']:9.5f} {r['p_naive']:9.5f} {r['dz']:+7.2f}{star}")

    d = out_dir(ctx, "rq3")
    write_csv(d, "readout_cells", ["model", "cell", "dim", "auc_mean", "auc_sd"], rows)
    save(d, "readout", {
        "variant": ctx.tag, "seed": ctx.seed, "n_pool": len(pool),
        "folds": a.folds, "repeats": a.repeats,
        "cells": {f"{m} | {c}": float(np.nanmean(v)) for (m, c), v in curves.items()},
        # Per-fold values, so a contrast nobody thought of today can be tested tomorrow
        # without spending another run. Their absence is why this rerun was needed at all.
        "per_fold": {f"{m} | {c}": [None if not np.isfinite(x) else float(x) for x in v]
                     for (m, c), v in curves.items()},
        "contrasts": res})


if __name__ == "__main__":
    main()
