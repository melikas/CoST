"""Shared evaluation protocols: participant-level aggregation, thresholds, metrics.

Lifted verbatim from ``train_hrd.py`` so that every task scores its predictions the
same way without importing the training script. Upstream CoST keeps the same slot
for downstream protocols.
"""
import math

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score)


def participant_aggregate(pids, probs, labels):
    """Mean window probability per participant + the participant label."""
    uniq = np.unique(pids)
    pid_prob = np.array([probs[pids == p].mean() for p in uniq], dtype=np.float64)
    pid_lbl = np.array([int(labels[pids == p][0]) for p in uniq], dtype=int)
    return pid_prob, pid_lbl


def best_threshold(y_true, y_prob):
    """Decision threshold in [0.05, 0.95] that maximises BALANCED ACCURACY (equivalently
    Youden's J = sensitivity + specificity - 1, up to a constant). Unlike F1-max, a
    degenerate all-one-class predictor scores only 0.5 balanced accuracy, so the collapsed
    threshold is never chosen when a genuinely discriminative one exists -- this fixes the
    MCC=0 "predicts everything one class" failure. AUC is threshold-free and unaffected."""
    best_thr, best_ba = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 37):
        ba = balanced_accuracy_score(y_true, (y_prob >= thr).astype(int))
        if ba > best_ba:
            best_ba, best_thr = ba, float(thr)
    return best_thr


def participant_bootstrap_auc(y_true, y_prob, pids, n_boot=1000, seed=0, alpha=0.05,
                              stat=None):
    """Confidence interval for AUROC -- or any `stat(y_true, y_pred)` -- obtained by
    resampling PARTICIPANTS, not rows.

    Rows are not independent here: with sliding windows one person contributes ~150 windows
    that share six of their seven days, so a row-level bootstrap (or any formula assuming n
    independent observations) reports an interval far too narrow -- it answers "how stable is
    this number on these people", when the claim being made is "how well does this transfer to
    a NEW person". Drawing whole participants with replacement, and taking all of a drawn
    participant's rows together, matches that claim.

    Returns {'lo','hi','mean','n_participants','n_boot_valid'}; the bounds are NaN when too few
    draws yield a defined value (a draw can land on one class only, and a constant predictor
    has no defined rank correlation)."""
    stat = stat or roc_auc_score
    y_true, y_prob, pids = np.asarray(y_true), np.asarray(y_prob), np.asarray(pids)
    uniq = np.unique(pids)
    idx_by_pid = {p: np.where(pids == p)[0] for p in uniq}
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(int(n_boot)):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_pid[p] for p in drawn])
        try:                                         # degenerate draws are skipped, not fixed
            v = float(stat(y_true[idx], y_prob[idx]))
        except Exception:
            continue
        if np.isfinite(v):
            aucs.append(v)
    nan = float("nan")
    if len(aucs) < max(20, 0.1 * n_boot):
        return {"lo": nan, "hi": nan, "mean": nan,
                "n_participants": int(len(uniq)), "n_boot_valid": len(aucs)}
    lo, hi = np.percentile(aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"lo": float(lo), "hi": float(hi), "mean": float(np.mean(aucs)),
            "n_participants": int(len(uniq)), "n_boot_valid": len(aucs)}


def binary_metrics(y_true, y_prob, thr):
    y_true = np.asarray(y_true)
    y_pred = (y_prob >= thr).astype(int)
    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    # clinical rates: sensitivity = TP/(TP+FN) = fraction of depressed caught; specificity =
    # TN/(TN+FP) = fraction of non-depressed correctly cleared. Reported so the summary table
    # carries them for the SSL models too (the separability baselines already report them).
    tp = int(((y_pred == 1) & (y_true == 1)).sum()); fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum()); fp = int(((y_pred == 1) & (y_true == 0)).sum())
    return {
        "auc_roc": float(auroc),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # balanced accuracy + MCC are the honest threshold metrics on this (near-)balanced
        # test set: F1 saturates at the trivial 0.667 for an all-positive classifier, whereas
        # MCC is 0 there. Reported alongside F1 so the comparison stays consistent.
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "threshold": float(thr),
    }


def prevalence_transport(sens, spec, prevalence):
    """What the SAME classifier's threshold metrics become at a target base rate.

    Sensitivity and specificity are WITHIN-class rates, so they carry over unchanged from one
    class mix to another. Nothing else on a confusion matrix does. The test cohort here is
    balanced by construction (--test-per-class holds out 18 + 18), so every threshold metric
    in binary_metrics -- accuracy, F1, MCC, and the PPV they imply -- is quoted at an implied
    50% prevalence, which no deployment population has. Given sens/spec at a fixed threshold,
    this returns those metrics at `prevalence` exactly, by Bayes' rule: no refitting, no extra
    data, no simulation.

    AUROC and balanced accuracy are deliberately absent: both are already prevalence-invariant
    (AUROC ranks; balanced accuracy is just (sens+spec)/2), so they transfer as measured. It is
    PPV that collapses -- at 10% prevalence a 90%/90% classifier has PPV 0.5, not 0.9.
    """
    pi = float(prevalence)
    if not (0.0 < pi < 1.0) or not np.isfinite(sens) or not np.isfinite(spec):
        return {"prevalence": pi, "ppv": float("nan"), "npv": float("nan"),
                "f1": float("nan"), "accuracy": float("nan"), "mcc": float("nan")}
    # expected confusion matrix per unit population -- MCC is scale-free, so rates work
    tp, fn = pi * sens, pi * (1.0 - sens)
    tn, fp = (1.0 - pi) * spec, (1.0 - pi) * (1.0 - spec)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    f1 = (2 * ppv * sens / (ppv + sens)) if np.isfinite(ppv) and (ppv + sens) > 0 else float("nan")
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "prevalence": pi,
        "ppv": float(ppv),
        "npv": float(npv),
        "f1": float(f1),
        "accuracy": float(pi * sens + (1.0 - pi) * spec),
        "mcc": float((tp * tn - fp * fn) / den) if den > 0 else float("nan"),
    }


def prior_shift(prob, prevalence_from, prevalence_to):
    """Re-anchor scores measured at one base rate to another (Elkan 2001 / Saerens et al.
    2002 prior correction): multiply the odds by the prevalence-odds ratio. Used to quote
    calibration at deployment prevalence instead of the balanced test cohort's 50%."""
    p0, p1 = float(prevalence_from), float(prevalence_to)
    if not (0.0 < p0 < 1.0 and 0.0 < p1 < 1.0):
        return np.asarray(prob, dtype=float)
    prob = np.clip(np.asarray(prob, dtype=float), 1e-12, 1 - 1e-12)
    odds = prob / (1.0 - prob) * ((p1 / (1.0 - p1)) * ((1.0 - p0) / p0))
    return odds / (1.0 + odds)


def calibration_metrics(y_true, y_prob, n_bins=10):
    """Brier score + expected / maximum calibration error + the reliability bins.

    A tuned threshold says nothing about whether the SCORES mean anything: two models with
    identical AUROC can differ wildly in whether `prob=0.7` corresponds to a 70% chance.
    `bins` carries (count, mean predicted, observed rate) so a reliability curve can be drawn
    after the fact without re-running the model."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.size == 0:
        return {"brier": float("nan"), "ece": float("nan"), "mce": float("nan"), "bins": []}
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, int(n_bins) - 1)
    bins, ece, mce = [], 0.0, 0.0
    for b in range(int(n_bins)):
        m = idx == b
        if not m.any():
            continue
        conf, acc, w = float(y_prob[m].mean()), float(y_true[m].mean()), int(m.sum())
        bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]),
                     "n": w, "mean_pred": conf, "observed": acc})
        ece += (w / y_true.size) * abs(acc - conf)
        mce = max(mce, abs(acc - conf))
    return {"brier": float(np.mean((y_prob - y_true) ** 2)),
            "ece": float(ece), "mce": float(mce), "bins": bins}


def threshold_at_sensitivity(y_true, y_prob, target_sens):
    """Highest threshold whose sensitivity is still >= `target_sens`.

    Must be called on VALIDATION / out-of-fold predictions, never on test -- the point of a
    fixed operating point is that it was committed to without seeing the test cohort."""
    y_true, y_prob = np.asarray(y_true, dtype=int), np.asarray(y_prob, dtype=float)
    if y_true.sum() == 0:
        return 0.5
    best = 0.0
    for thr in np.unique(np.concatenate([y_prob, [0.0, 1.0]])):
        pred = (y_prob >= thr).astype(int)
        sens = pred[y_true == 1].mean()
        if sens >= target_sens:
            best = max(best, float(thr))
    return best


def operating_point_report(y_true, y_prob, thresholds, prevalences):
    """binary_metrics at each FIXED threshold, each transported to each target prevalence.

    Separates the two things the balanced cohort + tuned threshold conflate: the choice of
    operating point (here fixed in advance, not tuned to maximise anything on test) and the
    base rate the resulting rates are quoted at."""
    out = {}
    for name, thr in thresholds.items():
        m = binary_metrics(y_true, y_prob, thr)
        m["at_prevalence"] = {f"{p:g}": prevalence_transport(m["sensitivity"], m["specificity"], p)
                              for p in prevalences}
        out[name] = m
    return out
