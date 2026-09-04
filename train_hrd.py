"""Train CoST on the HRD wearable data for depression-endpoint classification.

Pipeline (leakage-safe, participant-level throughout):
  1. data_preprocessing.prepare_hrd_dataset -> windows X (N, T, C), labels y, pids
  2. hold out 30% of the CONSISTENT (baseline==endpoint) participants as the test set
  3. self-supervised pretraining of the CoST encoder on ALL non-test windows
  4. encode windows -> representations; fit a logistic-regression classifier on the
     remaining 70% consistent cohort (a participant-level val split picks the
     decision threshold); evaluate on the held-out test set
  5. report window- and participant-level AUC / F1 / accuracy

Run:  python train_hrd.py --sensor-csv datasets/HRD_RAW_MinuteLevel.csv --backbone transformer
"""
import argparse
from tasks.rq_paths import rq_path
import json
import math
import os
import time
from pathlib import Path

import re

import numpy as np
import torch

from baselines.supervised import supervised_baseline_row
from model_build import paper_kernels, random_init_repr
from cost import CoST
from models.positional_encoding import CALENDAR_PES
# tasks.energy is a pure library (its entry point is train_hrd_energy.py), so this import
# never cycles and the handcrafted rung is defined once for the EE probe, the RQ3 ladder and
# both separability tables.
from tasks.energy import handcrafted_features
from tasks._eval_protocols import (binary_metrics, best_threshold, clamp_pca,
                                   make_probe,
                                   persubject_rows,
                                   calibration_metrics, operating_point_report,
                                   participant_aggregate, participant_bootstrap_auc,
                                   prevalence_transport, prior_shift,
                                   threshold_at_sensitivity)
from data_processing.data_preprocessing import prepare_hrd_dataset
from utils import init_dl_program, stratified_pid_holdout


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _pool_arg(v):
    """`--pool` accepts the four fixed names plus segN for any positive N.

    argparse `choices` cannot express that, and dropping the check altogether would let a
    typo reach cost.py, which raises only once a readout is actually collapsed -- after the
    pretraining it was meant to guard.
    """
    import argparse
    if v in ("last", "mean", "max", "meanmax"):
        return v
    if re.fullmatch(r"seg[1-9][0-9]*", v):
        return v
    raise argparse.ArgumentTypeError(
        f"invalid pool {v!r}: use last, mean, max, meanmax, or segN (e.g. seg2)")


def balanced_pid_holdout(unique_pids, pid_label, n_per_class, seed):
    """Hold out EXACTLY `n_per_class` participants of each label (0 and 1) as the test
    set -- a class-balanced test cohort. The held-out participants appear in NEITHER the
    pretrain nor the fine-tune pool (the caller builds pretrain = all non-test windows).
    Returns (rest_pids, test_pids)."""
    rng = np.random.default_rng(seed)
    by_class = {0: [], 1: []}
    for p in sorted(unique_pids):
        c = pid_label.get(p)
        if c in (0, 1):
            by_class[c].append(p)
    held = set()
    for c in (0, 1):
        pool = by_class[c]
        if len(pool) < n_per_class:
            print(f"[split] WARNING: only {len(pool)} consistent participants with label {c} "
                  f"(< requested {n_per_class}); holding out all of them.")
        k = min(n_per_class, len(pool))
        if k:
            held.update(rng.choice(np.array(pool), size=k, replace=False).tolist())
    rest = set(sorted(unique_pids)) - held
    return rest, held



def _md_table(headers, rows):
    """Minimal markdown table; None/NaN render as an em dash."""
    def cell(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "--"
        return f"{v:.3f}" if isinstance(v, float) else str(v)
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(cell(v) for v in r) + " |")
    return "\n".join(out) + "\n"


def write_run_report(variant_dir, args, result, split_seed, model_seed):
    """Human-readable companion to metrics.json, written once per run.

    Everything here is already in metrics.json. The point of the file is that the CAVEATS
    travel with the numbers: the headline threshold metrics come from a class-balanced test
    cohort at a tuned threshold, so they are never shown alone -- always beside their
    prevalence-transported values and the fixed operating points. The provenance table
    likewise records the knobs that silently change what the numbers mean (the two seeds, the
    masking mode, the seasonal phase encoding), so a report can be read years later without
    guessing which code version produced it."""
    pid = result.get("participant_level", {}) or {}
    win = result.get("window_level", {}) or {}
    prevs = list(pid.get("at_prevalence", {}))
    L = []

    L.append("# DSSL on HRD -- run report\n")
    # getattr throughout: the two trees do not expose an identical flag set (e.g. no
    # --season-pool in the GLOBEM fork), and the report must never be what crashes a run.
    _sp = getattr(args, "season_pool", None)
    L.append(f"**backbone** `{args.backbone}` | **pe** `{args.pe}` | "
             f"**pool** `{getattr(args, 'pool', '?')}`"
             + (f" / season `{_sp}`" if _sp else "")
             + f" | **disentangle** "
               f"{'yes' if getattr(args, 'disentangle', True) else 'no (plain SSL)'}\n")

    L.append("\n## Reproducibility / provenance\n")
    rows = [
        ("`--split-seed`", split_seed,
         "which participants are test/val, CV folds, bootstrap resampling"),
        ("`--model-seed`", model_seed,
         "weight init, augmentation stream, loader order, probe estimators"),
        ("`--mask-mode`", f"`{getattr(args, 'mask_mode', 'none')}`",
         "training-time timestep masking (`none` = upstream CoST, no augmentation)"),
        ("`--phase-encoding`", f"`{getattr(args, 'phase_encoding', 'circular')}`",
         {"raw": "upstream CoST's raw atan2 angle -- the dot product is NOT a similarity "
                 "between angles (see circular_phase)",
          "circular": "[sin, cos], so phase similarity is a function of the angular gap alone",
          "circular_amp": "[sin, cos] weighted by each channel's amplitude -- weak channels, "
                          "whose phase is undefined noise, stop counting as much as real rhythms",
          }.get(getattr(args, "phase_encoding", "circular"), "seasonal phase comparison")),
    ]
    L.append(_md_table(["knob", "value", "what it controls"], rows))
    if split_seed == model_seed:
        L.append(f"\n> Both seeds are {split_seed}, so cohort variance and optimisation "
                 f"variance are **confounded** in this run: a seed-to-seed difference cannot "
                 f"be attributed to either. Cross `--split-seed` x `--model-seed` to separate "
                 f"them.\n")
    if args.pe == "convspe":
        L.append("\n> `convspe` draws random features while training (the SPE method) but is "
                 "evaluated at the exact expectation of that estimator, so `encode()` is "
                 "deterministic and repeated evaluations agree bit for bit.\n")

    L.append("\n## Cohort\n")
    tp = pid.get("test_prevalence", float("nan"))
    cp = pid.get("cohort_prevalence", float("nan"))
    L.append(f"- participants: {result.get('n_participants_total','?')} total = "
             f"{result.get('n_labeled_participants','?')} labelled "
             f"[{result.get('n_test_participants','?')} test + "
             f"{result.get('n_probe_participants','?')} probe] + "
             f"{result.get('n_unlabeled_participants','?')} unlabelled (pretrain-only)\n")
    L.append(f"- test cohort is **{tp:.0%} positive by construction** "
             f"(`--test-per-class {getattr(args, 'test_per_class', '?')}`); "
             f"observed cohort base rate is **{cp:.1%}**\n")

    L.append("\n## Headline (participant level)\n")
    L.append(_md_table(
        ["metric", "value", "transfers to another prevalence?"],
        [("AUROC", pid.get("auc_roc"), "yes -- ranking statistic"),
         ("balanced accuracy", pid.get("balanced_accuracy"), "yes -- (sens+spec)/2"),
         ("sensitivity", pid.get("sensitivity"), "yes -- within-class rate"),
         ("specificity", pid.get("specificity"), "yes -- within-class rate"),
         ("accuracy", pid.get("accuracy"), "**no** -- quoted at the cohort's 50%"),
         ("F1", pid.get("f1"), "**no** -- quoted at the cohort's 50%"),
         ("MCC", pid.get("mcc"), "**no** -- quoted at the cohort's 50%")]))
    ci = pid.get("auc_ci") or {}
    if ci:
        L.append(f"\nAUROC 95% CI (bootstrap over participants, the independent unit): "
                 f"[{ci.get('lo', float('nan')):.3f}, {ci.get('hi', float('nan')):.3f}] "
                 f"over {ci.get('n_participants','?')} participants.\n")
    L.append(f"\nDecision threshold {pid.get('threshold', float('nan')):.3f}, chosen to maximise "
             f"balanced accuracy on validation / out-of-fold predictions (never on test).\n")

    if prevs:
        L.append("\n## The same model at other base rates\n")
        L.append("Sensitivity and specificity are unchanged -- only the prevalence-dependent "
                 "metrics move. Computed exactly by Bayes' rule, no refitting.\n\n")
        L.append(_md_table(
            ["prevalence", "PPV", "NPV", "F1", "accuracy", "MCC"],
            [(f"{float(k):.1%}", v.get("ppv"), v.get("npv"), v.get("f1"),
              v.get("accuracy"), v.get("mcc"))
             for k, v in pid["at_prevalence"].items()]))

    ops = pid.get("operating_points") or {}
    if ops:
        p0 = prevs[0] if prevs else None
        L.append("\n## Fixed operating points\n")
        L.append("Thresholds committed to WITHOUT seeing the test cohort: `nominal_0.5` a "
                 "priori, the `sens*_on_val` points from validation / out-of-fold predictions. "
                 "`tuned_balanced_acc` is the headline threshold, shown for continuity.\n\n")
        hdr = ["operating point", "thr", "sens", "spec", "F1 @50%"]
        if p0 is not None:
            hdr += [f"PPV @{float(p0):.1%}", f"F1 @{float(p0):.1%}"]
        rows = []
        for name, m in ops.items():
            row = [f"`{name}`", m.get("threshold"), m.get("sensitivity"),
                   m.get("specificity"), m.get("f1")]
            if p0 is not None:
                t = m.get("at_prevalence", {}).get(p0, {})
                row += [t.get("ppv"), t.get("f1")]
            rows.append(tuple(row))
        L.append(_md_table(hdr, rows))

    cal = pid.get("calibration") or {}
    if cal:
        L.append("\n## Calibration\n")
        L.append("Whether the SCORES mean anything, not just how they rank. Lower is better.\n\n")
        rows = [("as measured (balanced cohort)", cal.get("brier"), cal.get("ece"), cal.get("mce"))]
        for k, v in (pid.get("calibration_at_prevalence") or {}).items():
            rows.append((f"prior-shifted to {float(k):.1%}", v.get("brier"),
                         v.get("ece"), v.get("mce")))
        L.append(_md_table(["scores", "Brier", "ECE", "MCE"], rows))

    if win:
        L.append("\n## Window level (secondary)\n")
        shared = [k for k in ("auc_roc", "balanced_accuracy", "sensitivity", "specificity",
                              "accuracy", "f1", "mcc") if k in win and k in pid]
        if shared and all(win[k] == pid[k] for k in shared):
            L.append("> **These are identical to the participant-level numbers above** -- in "
                     "this configuration the participant aggregation is a no-op, so the two "
                     "sections are the same measurement under two names, not two units.\n\n")
        L.append(_md_table(["metric", "window level", "participant level"],
                           [(k, win.get(k), pid.get(k)) for k in shared]))

    L.append("\n_Full numbers, including reliability bins and every operating point, are in "
             "`metrics.json`._\n")
    report = "".join(L)
    (Path(variant_dir) / "report.md").write_text(report, encoding="utf-8")
    return report



def probe_cv_within_pool(reprs, y, pids, pid_label, pool_pids, probe_sel,
                         n_folds, C, seed, probe_mode="supervised", n_pca=0,
                         fold_seed=None):
    """Participant-level k-fold CV WITHIN the probe pool -- the held-out test set is never
    touched. Each fold trains the logistic probe on the other folds' participants and
    predicts the held-out fold, so every pool participant gets one out-of-fold (OOF)
    prediction. The decision threshold is tuned on the *pooled OOF* predictions (far more
    stable than a single val split, and it uses all pool participants), and those OOF
    predictions also give an internal-CV estimate of probe quality. The RETURNED probe is
    refit on ALL pool participants -- more training data than a single train/val split.

    `seed` seeds the probe ESTIMATORS (PCA / LR); `fold_seed` seeds the fold ASSIGNMENT, i.e.
    which participants land in which fold. They are separate so a crossed --split-seed /
    --model-seed design keeps the fold partition on the split side. `fold_seed=None` keeps
    both on `seed`, the historical behaviour.

    Returns (clf_all, thr, cv_metrics)."""
    from sklearn.model_selection import StratifiedKFold
    fold_seed = seed if fold_seed is None else fold_seed
    pool_pids = np.array(sorted(pool_pids))
    labels = np.array([int(pid_label[p]) for p in pool_pids])
    # clamp folds to the smaller class so StratifiedKFold never fails on a tiny class
    min_class = int(min((labels == 0).sum(), (labels == 1).sum()))
    n_folds = max(2, min(int(n_folds), max(2, min_class)))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=fold_seed)
    oof_prob, oof_lbl = [], []
    for tr, va in skf.split(pool_pids, labels):
        tr_mask = np.isin(pids, pool_pids[tr]) & probe_sel
        va_mask = np.isin(pids, pool_pids[va]) & probe_sel
        # Clamp per fold: the smallest fold decides how many components are admissible.
        k = clamp_pca(n_pca, int(tr_mask.sum()), reprs.shape[1])
        clf = make_probe(probe_mode, C, seed, k); clf.fit(reprs[tr_mask], y[tr_mask])
        pp, pl = participant_aggregate(pids[va_mask],
                                       clf.predict_proba(reprs[va_mask])[:, 1], y[va_mask])
        oof_prob.append(pp); oof_lbl.append(pl)
    oof_prob, oof_lbl = np.concatenate(oof_prob), np.concatenate(oof_lbl)
    thr = best_threshold(oof_lbl, oof_prob)
    cv_metrics = binary_metrics(oof_lbl, oof_prob, thr)      # internal OOF-CV over the pool
    cv_metrics["n_folds"] = int(n_folds)
    cv_metrics["n_pool_participants"] = int(len(pool_pids))
    # Persisted so a FIXED operating point (e.g. "threshold at 80% sensitivity") can be chosen
    # from out-of-fold predictions rather than from the test cohort, and so that choice stays
    # auditable after the run. These never touch test -- the pool excludes it by construction.
    cv_metrics["oof_prob"] = [float(v) for v in oof_prob]
    cv_metrics["oof_label"] = [int(v) for v in oof_lbl]

    pool_mask = np.isin(pids, pool_pids) & probe_sel
    clf_all = make_probe(probe_mode, C, seed,
                          clamp_pca(n_pca, int(pool_mask.sum()), reprs.shape[1]))
    clf_all.fit(reprs[pool_mask], y[pool_mask])
    return clf_all, thr, cv_metrics



def save_loss_curves(iters, train_loss, val_loss, variant_dir, tag):
    """Save 'pretrain loss over iterations' and 'val loss over iterations' as two PNGs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    aligned = len(iters) == len(train_loss) and len(iters) > 0
    x = list(iters) if aligned else list(range(1, len(train_loss) + 1))
    xlabel = "iteration" if aligned else "epoch"

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, train_loss, marker="o", ms=3, lw=1.8, color="#0072B2")
    ax.set_xlabel(xlabel); ax.set_ylabel("contrastive loss")
    ax.set_title(f"Pretrain loss over {xlabel}s  -  {tag}"); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(rq_path(variant_dir, "pretrain_loss.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    if val_loss:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(x[:len(val_loss)], val_loss, marker="o", ms=3, lw=1.8, color="#D55E00")
        ax.set_xlabel(xlabel); ax.set_ylabel("held-out contrastive loss")
        ax.set_title(f"Validation loss over {xlabel}s  -  {tag}"); ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(rq_path(variant_dir, "val_loss.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)






def save_signal_embedding(model, X, pids, test_pids, pid_label, variant_dir,
                          sensor_cols, n_sensors):
    """Save ONE depressed + ONE non-depressed held-out TEST window: the raw token
    signal (before embedding) and the encoder's per-timestep representation (after
    embedding, ``model.net(x, tcn_output=True)``). Read back by
    ``plot_position_similarity.py --signalviz`` to compare, across PE techniques and
    both groups, what the embedding does to the signal. Deterministic (sorted pid)."""
    import numpy as _np
    dep = sorted(p for p in test_pids if pid_label.get(p) == 1)
    non = sorted(p for p in test_pids if pid_label.get(p) == 0)
    if not dep or not non:
        print("[signal-embed] no depressed/non-depressed test pid; skipping")
        return
    pid_d, pid_n = dep[0], non[0]
    idx_d = _np.where(pids == pid_d)[0][-1]                 # last (most recent) window of each
    idx_n = _np.where(pids == pid_n)[0][-1]
    org = model.net.training
    model.net.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X[[idx_d, idx_n]]).float().to(model.device)
        emb = model.net(xb, tcn_output=True).cpu().numpy()  # (2, T, output_dims)
    model.net.train(org)
    _np.savez_compressed(
        rq_path(variant_dir, "signal_embedding.npz"),
        sig_dep=X[idx_d, :, :n_sensors].astype("float32"),  # (T, n_sensors) before embedding
        sig_non=X[idx_n, :, :n_sensors].astype("float32"),
        emb_dep=emb[0].astype("float32"),                   # (T, output_dims) after embedding
        emb_non=emb[1].astype("float32"),
        sensor_cols=_np.array(list(sensor_cols), dtype=object),
        pid_dep=str(pid_d), pid_non=str(pid_n),
    )
    print(f"[signal-embed] saved dep={pid_d} / non={pid_n} -> {variant_dir/'signal_embedding.npz'}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="CoST on HRD: depression-endpoint classification")
    p.add_argument("--sensor-csv", required=True, help="Path to HRD_RAW_MinuteLevel.csv")
    p.add_argument("--label-col", default="depression_status_endpoint")

    # --- GLOBEM (--dataset globem). Ignored entirely on the default HRD path. ---------
    p.add_argument("--dataset", choices=["hrd", "globem"], default="hrd",
                   help="'hrd' = minute-level HRD_RAW_MinuteLevel.csv (default); 'globem' = "
                        "segment-level GLOBEM_REDUCED.csv (12 features, 4 segments/day).")
    p.add_argument("--window-days", type=int, default=28,
                   help="[globem] window length in days (x4 segments -> T). Default 28 -> T=112.")
    p.add_argument("--stride-days", type=int, default=7,
                   help="[globem] slide between consecutive windows, in days.")
    p.add_argument("--globem-anchor-weekday", type=int, default=0,
                   help="[globem] weekday every window starts on (0=Mon .. 6=Sun; -1 disables).")
    p.add_argument("--globem-label", choices=["weekly", "endpoint"], default="weekly",
                   help="[globem] 'weekly' = per-window time-varying LABEL_WEEKLY; "
                        "'endpoint' = one end-of-study label per participant.")
    p.add_argument("--cohort", choices=["consistent", "labeled"], default="consistent",
                   help="Participants forming the split: 'consistent' = only baseline==endpoint "
                        "(clean label, fewer people); 'labeled' = EVERY participant with an "
                        "endpoint label (more samples, but status-changers add label noise).")
    p.add_argument("--probe-unit", choices=["last", "all", "persubject"], default="last",
                   help="What counts as one probe sample. 'last' = the participant's most recent "
                        "window (default, no pseudo-replication but discards the other windows); "
                        "'all' = every window carries its participant's label (uses all data but "
                        "the effective n is still the participant count); 'persubject' = one row "
                        "per participant holding [mean|std] of that participant's window "
                        "embeddings (all data AND correct n). Pair 'persubject' with --probe-pca.")
    p.add_argument("--probe-pca", type=int, default=0,
                   help="PCA components between the scaler and the classifier; 0 disables it. "
                        "Fitted inside each CV fold on that fold's training participants only. "
                        "Automatically clamped to min(n_train-1, n_features). ~20 is a sensible "
                        "value for 'persubject', where the feature vector doubles in width.")
    p.add_argument("--probe-last-window", action="store_true",
                   help="DEPRECATED alias for '--probe-unit last', which is already the default; "
                        "kept only so older command lines and notes do not fail. If --probe-unit "
                        "is also given explicitly, --probe-unit wins and a note is printed. "
                        "Pretraining is unaffected either way (it always uses all windows).")
    p.add_argument("--probe-c", type=float, default=1.0,
                   help="Inverse L2 strength of the logistic probe. Use a small value (e.g. 0.1) "
                        "for --probe-unit last/persubject, where only ~1 sample per participant "
                        "is available.")
    p.add_argument("--probe-mode", choices=["supervised", "anomaly"], default="supervised",
                   help="'supervised' (default) = standard 2-class logistic-regression probe. "
                        "'anomaly' = semi-supervised alternative: fits a robust Gaussian on ONLY "
                        "the non-depressed training participants (what 'normal' rhythm looks "
                        "like) and scores everyone by Mahalanobis distance to it -- depression "
                        "as a deviation from the learned norm, rather than a discriminated class.")
    p.add_argument("--cv-folds", type=int, default=1,
                   help="Participant-level k-fold CV WITHIN the probe pool (test untouched): tunes "
                        "the threshold on pooled out-of-fold predictions and trains the final probe "
                        "on ALL probe participants. 1 = single train/val split (no CV).")
    p.add_argument("--window-hours", type=int, default=168)
    p.add_argument("--bin-minutes", type=int, default=15)
    p.add_argument("--max-missing", type=float, default=0.30,
                   help="Drop participants with more than this fraction of wear-channel missingness")
    p.add_argument("--max-window-missing", type=float, default=0.30,
                   help="Algorithm-1 (Yan et al. 2022) threshold: drop a window if ANY sensor "
                        "channel has more than this fraction of missing time-bins, or if a "
                        "channel's first/last bin is missing")
    p.add_argument("--no-zscore", action="store_true", help="Disable per-participant z-scoring")
    p.add_argument("--save-encoder", action="store_true",
                   help="Write encoder.pt (~235 MB per variant). Off by default; needed only "
                        "to redo a latent analysis without retraining.")
    p.add_argument("--with-clock-features", action="store_true",
                   help="Opt-in: append the clock/time channels (time-of-day & "
                        "day-of-week sin/cos + linear ramp) and inject them as a "
                        "temporal positional encoding. OFF by default -- time is "
                        "excluded from the model entirely (neither an input feature "
                        "nor a positional encoding).")
    p.add_argument("--test-per-class", type=int, default=18,
                   help="Hold out EXACTLY this many CONSISTENT participants of each class "
                        "(depressed / non-depressed) as a class-balanced test set; they are "
                        "excluded from both pretrain and fine-tune. Default 18 -> 36 test pids.")
    p.add_argument("--test-frac", type=float, default=0.35,
                   help="(deprecated) old fraction-based test holdout; ignored when "
                        "--test-per-class is used")
    p.add_argument("--target-prevalence", type=float, nargs="+", default=None,
                   help="Base rate(s) at which to ALSO report PPV/NPV/F1/accuracy/MCC. The "
                        "test cohort is balanced by construction, so the headline threshold "
                        "metrics are quoted at an implied 50%% prevalence and do not transfer; "
                        "sensitivity/specificity/AUROC do. Defaults to the observed cohort "
                        "prevalence. Pass e.g. 0.05 0.1 0.2 to bracket a deployment rate.")
    p.add_argument("--val-frac", type=float, default=0.25,
                   help="Fraction of the fine-tune cohort used to pick the decision threshold")
    p.add_argument("--pretrain-val-frac", type=float, default=0.10,
                   help="Fraction of pretrain windows held out to monitor the SSL validation loss")
    # CoST encoder / pretraining
    p.add_argument("--backbone", default="tcn",
                    choices=["tcn", "transformer"])
    p.add_argument(
        "--pe", default=None,
        choices=["sinusoidal", "learnable", "tape", "rpe", "erpe", "tupe",
                 "convspe", "tpe", "time2vec", "factorized", "circular", "none"],
        help="Positional encoding. Transformer accepts all 8 index-based PE methods plus "
             "time2vec and the two wall-clock ones (default: sinusoidal). TCN accepts "
             "'none' (baseline, default), 'time2vec', 'factorized' or 'circular' -- the "
             "input-side encodings, which is what lets a reference-frame contrast be read "
             "on both backbones.",
    )
    p.add_argument("--time2vec-dim", type=int, default=65,
                   help="Time2Vec vector size k+1 (1 linear + k learnable-frequency "
                        "sines), concatenated to the input when --pe time2vec "
                        "(Kazemi et al. 2019, fed as input). Default 65 = k=64 sines, the "
                        "paper's best (App. B: 64 outperforms 32 and 16 in most cases). Was "
                        "16, i.e. k=15 -- BELOW the smallest configuration the paper ever "
                        "tested -- and tcn:time2vec then failed to converge in run 19649817 "
                        "(pretrain loss fell only 48%%, to 3.50, where every other variant "
                        "reached ~0.10). That is the exact failure the paper attributes to "
                        "too few sines: Sec. 6 reports optimisation trouble only 'when using "
                        "only a few sine functions' and credits 'using many sine functions "
                        "which reduces the distance to the goal'.")
    p.add_argument("--repr-dims", type=int, default=320)
    p.add_argument("--hidden-dims", type=int, default=64)
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--pool", type=_pool_arg, default="mean",
                   metavar="{last,mean,max,meanmax,segN}",
                   help="How the frozen representation is collapsed over the 7-day window "
                        "before the linear probe: 'mean' (default) / 'max' summarise the WHOLE "
                        "window; 'last' = final timestep only (original CoST forecasting "
                        "readout); 'meanmax' = mean+max concatenated; 'segN' averages within "
                        "each of N equal time segments and concatenates, so 'seg1' IS 'mean'. "
                        "Pair segN with --season-pool same to apply it to both halves: that "
                        "beat the shipped readout in all four GLOBEM LODO folds "
                        "(seg2: +0.018 to +0.028, p<=0.023 paired per variant), for the "
                        "untrained control as well, so it is a better readout rather than "
                        "evidence of anything learned.")
    p.add_argument("--season-pool",
                   choices=["spec", "spec_band", "spec_amp", "spec_phase", "same"],
                   default="spec",
                   help="Readout of the SEASONAL half only. Default 'spec' reads amplitude "
                        "AND phase at the chronobiological harmonics (circaseptan, circadian, "
                        "12/8/6h). 'same' = use --pool for it too, which is the ABLATION: the "
                        "seasonal branch is an irFFT, so time-averaging it returns exactly the "
                        "f=0 (DC/MESOR) coefficient and every rhythm integrates to zero. "
                        "'spec_band' keeps every bin down to a two-hour period instead of "
                        "five lines: measured on HRD over 24 seeds by applying each "
                        "restriction to the RAW signal, the five-harmonic truncation costs "
                        "0.0502 AUC against the full window while bins 1..T/8 cost 0.0005. "
                        "Keeping the whole spectrum is worse than both (-0.0712), so there "
                        "is an optimum rather than a monotone gain. Pair it with "
                        "--seasonal-bands single: the harmonic layout stops at bin 31 and "
                        "the readout would be asking for bins the layer never writes.")
    p.add_argument("--kernels", type=int, nargs="+", default=None,
                   help="AR-expert kernel sizes (default: CoST powers-of-2 from window length)")
    p.add_argument("--alpha", type=float, default=0.0005,
                   help="Seasonal-loss weight (fixed mode only; ignored under --loss-balance gradnorm)")
    p.add_argument("--jitter-sigma", type=float, default=0.1,
                   help="Std of the additive-noise (jitter) augmentation. Lower preserves phase "
                        "estimation (CoST default 0.1; try 0.05 to strengthen seasonal phase).")
    p.add_argument("--phase-encoding", choices=["circular", "circular_amp", "raw"],
                   default="circular",
                   help="How the seasonal (SFD) loss compares FFT phases. 'circular' (default) "
                        "embeds each angle as [sin, cos] so the contrastive dot product becomes "
                        "cos(phi_i - phi_j) -- a true function of the angular gap. "
                        "'circular_amp' additionally weights each channel by its own amplitude "
                        "(amplitude-weighted phase coherence), so channels whose phase is "
                        "undefined noise stop counting as much as real rhythms; it is a strict "
                        "generalisation of 'circular' and preserves the logit scale. 'raw' is "
                        "upstream CoST's raw atan2 angle, where the +/-pi branch cut makes two "
                        "IDENTICAL phases look maximally dissimilar; keep it only to reproduce "
                        "archived runs.")
    # ---- three one-line ablations of the pretraining objective (RQ0 diagnostics) --------
    # Each isolates ONE candidate explanation for the trained encoder losing to a random-init
    # one on RQ1/RQ3 in run 1239199. Defaults reproduce that run exactly, so leaving them
    # alone changes nothing.
    p.add_argument("--shift-sigma", type=float, default=0.5,
                   help="A: strength of the per-channel constant-offset augmentation. 0 "
                        "removes it. The default forces invariance to a half-SD level shift, "
                        "which IS the MESOR -- and trained recovery of MESOR sits below "
                        "random-init in all four channels.")
    p.add_argument("--decomp-aug", action="store_true",
                   help="Build the contrastive views by RE-COMPOSING each window from its own "
                        "closed-form decomposition: trend + seasonal + resampled noise. The "
                        "trend branch gets a pair sharing tau with swapped sigma, the seasonal "
                        "branch a pair sharing sigma with swapped tau, so each branch's "
                        "positive shares only its own component. This is the project's "
                        "x = trend + seasonal + noise hypothesis made literal, and it targets "
                        "two measured defects: the positive pair was a near-identity transform "
                        "(top-1 1.000 at init vs chance 1/(K+1)) and both branches contrasted "
                        "the same pair (Full minus plain = +0.0007, p=0.979).")
    p.add_argument("--globem-split", choices=["random", "lodo"], default="random",
                   help="How GLOBEM holds out participants. 'random' (default) is this "
                        "project's own class-balanced holdout. 'lodo' is the published "
                        "benchmark's leave-one-dataset-out protocol: three of the four study "
                        "years train and the fourth tests, which is what makes our numbers "
                        "comparable to theirs. Only their CROSS-dataset results are a fair "
                        "target -- their single-dataset setup trains on the first 80%% of "
                        "EVERY user's data and tests on the last 20%%, so the same people are "
                        "on both sides and a personal baseline scores. Their reported "
                        "cross-dataset best is 0.547 balanced accuracy (Reorder, "
                        "leave-one-dataset-out) and 0.536 for the best depression-specific "
                        "algorithm (Chikersal et al.), against a 0.500 majority baseline.")
    p.add_argument("--lodo-fold", type=int, default=0, metavar="K",
                   help="Which study year is held out under --globem-split lodo (0-3).")
    p.add_argument("--drop-channels", nargs="*", default=None, metavar="NAME",
                   help="Sensor channels to remove after the windows are built. Measured on "
                        "HRD over 24 seeds, through an identical random projection and probe: "
                        "dropping Steps raises the achievable AUC from 0.6884 to 0.7123, and "
                        "with hourly bins to 0.7137. The channel is not merely uninformative "
                        "for this endpoint, it is in the way. Applied AFTER windowing so the "
                        "missingness filters and z-scoring still see it and the surviving "
                        "windows stay the ones every other run scored.")
    p.add_argument("--smooth-bins", type=int, default=0, metavar="W",
                   help="Widest box filter the smoothing augmentation may draw, in time bins; "
                        "0 (default) disables it. In a contrastive objective the augmentation "
                        "IS the definition of noise, so each candidate has a ceiling -- the "
                        "predictive content of what survives it. Measured on HRD: sub-hour "
                        "smoothing 0.6926, jitter 0.6884, per-channel offset 0.6835, gain "
                        "0.6492, day permutation 0.6303, time roll 0.6273, low-pass to "
                        "tau+sigma 0.6228. Sub-hour smoothing is the only family that both "
                        "defines a real invariance and raises the ceiling above the raw "
                        "window, so at 15 min bins W=4 declares sub-hour detail to be noise.")
    p.add_argument("--positive-pair",
                   choices=["window", "participant", "day-disjoint"],
                   default="window",
                   help="What the contrastive positive is. 'window' (default) uses two "
                        "augmented copies of the SAME window -- measured on an untrained "
                        "encoder with a queue of real keys, top-1 retrieval is 1.000 against a "
                        "chance of 1/(K+1), so the task is solved before training starts and "
                        "the frozen representation never beats Random-init. 'participant' "
                        "draws the second view from a DIFFERENT window of the same person: the "
                        "pair then shares only that person's circadian amplitude and phase "
                        "(top-1 0.150, i.e. chance), so matching it requires encoding the "
                        "rhythm. 'day-disjoint' rebuilds the SAME window twice out of "
                        "disjoint halves of its own days: whole days are moved on day "
                        "boundaries, so the time of day survives and the only content "
                        "the two views reliably share is the circadian cycle and its "
                        "harmonics. Measured on real windows, off-harmonic agreement "
                        "falls from 0.756 to 0.287 while the daily harmonics hold at "
                        "0.555, so rhythm becomes the only route to a solution -- which "
                        "is the point: 'window' shares the whole spectrum equally, so "
                        "any feature solves it and the objective never says which to "
                        "learn.")
    p.add_argument("--phase-readout", choices=["circular", "angle"], default="circular",
                   help="How the seasonal readout emits phase. Phase is an angle on the "
                        "24 h circle, and every consumer treats readout columns as ordinary "
                        "numbers: the per-participant mean and sd, RQ2's Euclidean deviation "
                        "score, the linear probes. On raw angles all of them fail across the "
                        "branch cut -- 23.5 h and 0.5 h are one hour apart and average to "
                        "12.0, the opposite time of day -- which is precisely where a "
                        "hypothesised phase delay would put the estimates. 'circular' emits "
                        "(cos, sin) instead, making all of them correct by construction. "
                        "'angle' is the old raw-atan2 behaviour, kept because configs "
                        "written before this flag existed fall back to it and so rebuild the "
                        "model they actually ran. Measured on HRD over 24 seeds the two "
                        "encodings differ by -0.0093 AUC at p=0.90: this is a correctness "
                        "fix, not a performance one.")
    p.add_argument("--seasonal-bands", choices=["harmonics", "single"], default="harmonics",
                   help="Layout of the seasonal (SFD) Fourier layer. 'harmonics' gives each "
                        "circadian harmonic its own band; 'single' is the one full-spectrum "
                        "band the layer had before banding existed, and is what every run in "
                        "results_hrd/ from before that commit used -- required to reproduce "
                        "one of them rather than silently training a different architecture.")
    p.add_argument("--negatives", choices=["global", "subject"], default="global",
                   help="WHOSE keys the trend InfoNCE denominator holds. 'global' is the "
                        "shipped behaviour and is degenerate: the negatives are almost all "
                        "OTHER participants, so a query is matched to its own augmented view "
                        "by participant identity alone -- top-1 retrieval on an UNTRAINED "
                        "encoder is 1.000 against a chance of 1/(K+1), i.e. the pretext task "
                        "is solved at initialisation and the gradient teaches nothing. "
                        "'subject' draws them from the SAME participant, so identity is "
                        "constant across the denominator and the only thing left to encode is "
                        "how this window differs from that person's other windows.")
    p.add_argument("--n-negatives", type=int, default=0,
                   help="Negatives drawn per query. 0 (the default) means the WHOLE queue, which is the shipped behaviour and keeps every run comparable with those already in results_hrd/. A positive value fixes the count in BOTH modes, which is what "
                        "makes them comparable: InfoNCE improves with more negatives and the "
                        "same-participant pool is inevitably smaller (K/n_pretrain_pids ~ 35 "
                        "here), so an unmatched count would confound composition with size.")
    p.add_argument("--trend-pool", choices=["random", "mean"], default="random",
                   help="B: what the trend MoCo term contrasts. 'random' (upstream CoST) uses "
                        "one random timestep through a projection head that inference "
                        "discards; 'mean' contrasts the mean-pooled vector the probes "
                        "actually read.")
    p.add_argument("--moco-k", type=int, default=4096,
                   help="C: MoCo queue length. At the default the queue (4096) is LONGER than "
                        "the pretraining set (3014 windows), so every anchor's own window sits "
                        "in it as a negative ~1.36 times.")
    p.add_argument("--mask-mode", choices=["none", "binomial", "continuous"], default="none",
                   help="Training-time timestep-masking augmentation. 'none' (default) matches "
                        "upstream CoST, which never applied one -- its encoder's mask argument "
                        "hard-defaulted to 'all_true', so no published CoST run was masked. "
                        "'binomial'/'continuous' are opt-in and change SSL results: the mask "
                        "stacks on real non-wear gaps and adds broadband noise to the rFFT "
                        "amplitude/phase the seasonal loss fits. A/B one seed before a sweep.")
    p.add_argument("--mask-keep-prob", type=float, default=0.5,
                   help="Keep-probability of the binomial training mask; only has an effect "
                        "with --mask-mode binomial. Higher = LESS masking, better temporal "
                        "alignment / phase (CoST's nominal value 0.5; try 0.75).")
    p.add_argument("--loss-balance", choices=["fixed", "gradnorm"], default="fixed",
                   help="How trend vs seasonal losses are weighted: 'fixed' = weight seasonal "
                        "by --alpha (original CoST); 'gradnorm' = adaptively balance the two so "
                        "the seasonal branch is not starved.")
    p.add_argument("--disentangle", action=argparse.BooleanOptionalAction, default=True,
                   help="--disentangle = CoST (trend MoCo + seasonal FFT). --no-disentangle = a plain "
                        "single-representation self-supervised encoder: NO trend/seasonal split, one MoCo "
                        "on the encoder output, and no disentanglement eval files (prediction only).")
    p.add_argument("--max-train-length", type=int, default=None,
                   help="Crop length for pretraining; defaults to the window length T "
                        "(the CoST Fourier layer is sized to this, so it must equal T)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--iters", type=int, default=None, help="Pretraining iterations")
    p.add_argument("--epochs", type=int, default=None, help="Pretraining epochs")
    # misc
    p.add_argument("--seed", type=int, default=42,
                   help="Shared fallback for --split-seed and --model-seed. Setting only this "
                        "reproduces the historical behaviour where one seed drives BOTH.")
    # Splitting the seed in two: with a single --seed, changing 'the seed' changes WHICH
    # participants are in the test set AND the weight init / augmentation / loader order at
    # once, so the usual "mean +/- sd over seeds" is a mixture of split variance and
    # optimisation variance and cannot be read as either. With ~36 test participants the
    # split term is expected to dominate. Cross the two to separate them: hold --split-seed
    # fixed and vary --model-seed for optimisation noise; hold --model-seed fixed and vary
    # --split-seed for cohort sensitivity; run the grid for the interaction.
    p.add_argument("--split-seed", type=int, default=None,
                   help="Seed for WHO goes where: test/val participant splits, the pretrain "
                        "val holdout, CV fold assignment and the participant bootstrap. "
                        "Defaults to --seed.")
    p.add_argument("--model-seed", type=int, default=None,
                   help="Seed for HOW the model trains: weight init, augmentation stream, "
                        "loader shuffling, probe estimators and the analysis embeddings. "
                        "Defaults to --seed.")
    p.add_argument("--gpu", type=int, default=0, help="GPU index, or a negative value to force CPU")
    p.add_argument("--max-threads", type=int, default=None)
    p.add_argument("--no-plain-ssl", action="store_true",
                   help="Skip the non-disentangled SSL twin. It is a SECOND full pretraining "
                        "of the same size as the main one, so it is roughly half this script's "
                        "GPU time. Drop it only when the comparison being made does not use it "
                        "-- the objective ablations compare against Random-init, not against "
                        "the plain twin -- and never for a run whose results are reported.")
    p.add_argument("--no-rhythm-viz", action="store_true",
                   help="Skip the HRD test-set rhythm analysis (t-SNE + amplitude/"
                        "phase profiles + separability table) saved per variant.")
    p.add_argument("--paper-cosinor-topk", type=int, default=2,
                   help="Dominant periods per channel for the paper-cosinor LR baseline "
                        "(exact CosinorPy clone). 2 mirrors the paper's top-two-period tables. "
                        "Feature dim = n_sensors * topk * 12.")
    # Optional second downstream, REUSING this run's frozen encoder (no extra pretraining):
    # probe the daily self-reported emotional_energy (1-5) on the SAME held-out test
    # participants. Leakage-free (test was already excluded from pretraining) and cheap.
    p.add_argument("--energy-probe", action="store_true",
                   help="After pretraining, also probe daily emotional_energy (mode-B sliding "
                        "windows) reusing the frozen encoder; writes to --energy-output-dir.")
    p.add_argument("--no-energy-supervised", action="store_true",
                   help="Drop the end-to-end supervised rung from the two BINARY energy tasks. "
                        "It is the ceiling rung of the design's ladder, but unlike every other "
                        "rung it TRAINS a network -- twice, once per binary task -- so it is "
                        "the only part of the energy probe with a real wall-clock cost.")
    p.add_argument("--energy-stride", type=int, default=3,
                   help="Keep every k-th labelled day for the energy probe (default 3). "
                        "Consecutive trailing windows share 6 of 7 days, so the inputs are "
                        "highly redundant; k=3 keeps a ~21:1 sample-to-feature ratio while "
                        "cutting the windowing time and the ~0.5 GiB array to a third. Use "
                        "1 for every labelled day (the previous behaviour).")
    p.add_argument("--energy-pool", choices=["last", "mean", "max", "meanmax"], default="last",
                   help="Window pooling for the energy probe (default 'last', closest to the "
                        "labelled day). Independent of the depression --pool.")
    p.add_argument("--energy-threshold", type=float, default=4.0,
                   help="emotional_energy >= this = 'high-energy day' (default 4 -> ~47%% positive).")
    p.add_argument("--energy-output-dir", default="./results_hrd_energy",
                   help="Where the emotional-energy report.md/metrics.json go (mirrors --run-id).")
    p.add_argument("--output-dir", default="./results_hrd")
    p.add_argument("--run-id", default=None,
                   help="Sub-folder grouping a sweep's runs under --output-dir "
                        "(default: $SLURM_JOB_ID, or 'local' off-cluster).")
    return p.parse_args()


def main():
    args = parse_args()
    # Resolve the per-backbone default PE: sinusoidal for the Transformer, none
    # (the position-aware convolutions need no PE) for the TCN.
    if args.pe is None:
        args.pe = "sinusoidal" if args.backbone == "transformer" else "none"
    # The calendar PEs are allowed on the TCN too: convolutions carry RELATIVE position but
    # have no access to absolute calendar phase, which is exactly what they add.
    if args.backbone == "tcn" and args.pe not in ("none", "time2vec") + CALENDAR_PES:
        raise SystemExit(
            f"--pe {args.pe} is not valid for the TCN backbone "
            "(use 'none', 'time2vec', 'factorized' or 'circular')."
        )
    if args.pe in CALENDAR_PES and args.with_clock_features:
        raise SystemExit(
            f"--pe {args.pe} replaces the clock covariates with its own calendar encoding; "
            "drop --with-clock-features."
        )
    if args.backbone == "transformer" and args.pe == "none":
        raise SystemExit("--pe none is not valid for the Transformer backbone.")
    # Deprecated alias: it means exactly --probe-unit last, which is already the default, so it
    # only ever matters when the two disagree. Say so at startup rather than silently picking
    # one, because the choice changes what a probe sample is and is not visible in the metrics.
    if args.probe_last_window and args.probe_unit != "last":
        print(f"[probe] NOTE: --probe-last-window is deprecated and is being ignored, because "
              f"--probe-unit {args.probe_unit} was given explicitly.")
    if args.probe_pca and args.probe_unit == "all":
        print("[probe] NOTE: --probe-pca with --probe-unit all fits the components on windows, "
              "not participants; it is intended for --probe-unit persubject.")

    # Resolve the two seeds; both fall back to --seed, so a run that passes neither behaves
    # exactly as before. split_seed decides WHO goes where, model_seed decides HOW it trains.
    split_seed = args.seed if args.split_seed is None else args.split_seed
    model_seed = args.seed if args.model_seed is None else args.model_seed
    if split_seed != model_seed:
        print(f"[seed] split_seed={split_seed} (cohort) | model_seed={model_seed} "
              f"(init/augmentation/probe) -- crossed design")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = "cpu" if args.gpu < 0 else args.gpu
    device = init_dl_program(dev, seed=model_seed, max_threads=args.max_threads)
    t_start = time.time()

    # 1. data -------------------------------------------------------------
    if args.dataset == "globem":
        from data_processing.globem_preprocessing import prepare_globem_dataset
        # 4 segments/day -> a segment is a 6 h "bin"; pinning bin_minutes here keeps every
        # downstream time-aware component (bins_per_day=4, rhythm/decomp labels) correct.
        args.bin_minutes = 24 * 60 // 4
        # Written BACK onto args, not kept local: metrics.json stores vars(args), and every
        # RQ script rebuilds the dataset from it. Leaving the HRD default in the config made
        # the config describe a run that never happened, and the rebuild then asked the GLOBEM
        # CSV for a column it does not have.
        if args.label_col == "depression_status_endpoint":
            args.label_col = "LABEL_ENDPOINT"
        label_col = args.label_col
        data = prepare_globem_dataset(
            args.sensor_csv,
            window_days=args.window_days,
            stride_days=args.stride_days,
            label_col=label_col,
            z_score=not args.no_zscore,
            clock_features=args.with_clock_features,
            weekly_labels=(args.globem_label == "weekly"),
            anchor_weekday=args.globem_anchor_weekday,
        )
    else:
      data = prepare_hrd_dataset(
          args.sensor_csv,
          window_hours=args.window_hours,
          bin_minutes=args.bin_minutes,
          label_col=args.label_col,
          max_missing=args.max_missing,
          max_window_missing=args.max_window_missing,
          z_score=not args.no_zscore,
          clock_features=args.with_clock_features,
          calendar_index=args.pe in CALENDAR_PES,
          # Keep the parsed CSV only when the energy probe will re-read it below.
          cache_raw=args.energy_probe,
      )
    if args.drop_channels:
        from data_processing.data_preprocessing import drop_sensor_channels
        data = drop_sensor_channels(data, args.drop_channels)
    X, y, pids = data["X"], data["y"], data["pids"]
    # cohort for the split: 'consistent' = only baseline==endpoint (clean label); 'labeled' =
    # every participant with an endpoint label (more samples; status-changers add label noise).
    cohort_pids = data["labeled_pids"] if args.cohort == "labeled" else data["consistent_pids"]
    pid_label = {p: int(y[pids == p][0]) for p in np.unique(pids)}
    # A labeled participant can survive the label table yet produce zero windows (all windows
    # dropped by the per-window wear-missing filter). Such a pid is in cohort_pids but NOT in
    # pid_label -> keep only cohort participants that actually have windows, else the probe's
    # pid_label[p] lookup KeyErrors on it.
    cohort_pids = {p for p in cohort_pids if p in pid_label}
    # latest (last-week) window of each participant: windows are appended chronologically, so a
    # pid's last index is its most recent window -- the one closest to the endpoint label.
    # The participant's most recent LABELLED window. Taking the most recent window outright is
    # the same thing on HRD, where every window carries the endpoint label -- but on GLOBEM's
    # weekly mode a person's last window often has no survey in its span (y = -1), and that row
    # then became the participant's single probe row, carrying a third class into
    # roc_auc_score. A participant with no labelled window gets no row at all.
    labelled = np.asarray(y) >= 0
    last_window_mask = np.zeros(len(pids), bool)
    for p in np.unique(pids):
        idx = np.where((pids == p) & labelled)[0]
        if len(idx):
            last_window_mask[idx[-1]] = True

    # 2. leakage-safe split ----------------------------------------------
    # Class-balanced test set: exactly args.test_per_class depressed + the same number
    # non-depressed, held out from BOTH pretrain and fine-tune (pretrain = all non-test).
    if args.dataset == "globem" and args.globem_split == "lodo":
        # The published benchmark's own protocol: three study years train, the fourth
        # tests. Only their CROSS-dataset numbers are a fair target -- their
        # single-dataset setup puts the first 80% of every user's data in training and
        # the last 20% in test, so the same people are on both sides.
        from utils import lodo_test_pids
        test_pids, held, years, mixed = lodo_test_pids(
            data.get("window_ids"), pids, args.lodo_fold)
        if mixed:
            raise RuntimeError(f"{len(mixed)} participants span more than one study "
                               f"year, so a year split is not participant-disjoint")
        test_pids = [p for p in test_pids if p in pid_label]
        rest_cons = [p for p in cohort_pids if p not in set(test_pids)]
        print(f"[split] LODO fold {args.lodo_fold}: holding out {held} "
               f"({len(test_pids)} labelled participants), training on "
               f"{[y for y in years if y != held]}")
    else:
        rest_cons, test_pids = balanced_pid_holdout(
            cohort_pids, pid_label, args.test_per_class, split_seed
        )
    if not test_pids:
        raise RuntimeError(f"Test holdout is empty; check --test-per-class and the '{args.cohort}' cohort.")
    # Windows carrying no label at all (y < 0) exist only in GLOBEM's weekly mode, where a
    # window whose date span contains no survey is documented as "UNLABELED -> used for
    # self-supervised pretraining only, never in the probe/test". That was the intent but not
    # the behaviour: such windows reached the probe, whose roc_auc_score then saw three classes
    # {-1, 0, 1} and raised "multi_class must be in ('ovo', 'ovr')". They are excluded from
    # every SCORED mask here and stay in `pretrain_mask`, which is exactly what the doc says.
    test_mask = np.isin(pids, list(test_pids)) & labelled       # scored windows of test pids
    pretrain_mask = ~np.isin(pids, list(test_pids))             # ALL non-test windows, labelled or not
    finetune_mask = np.isin(pids, list(rest_cons)) & labelled   # cohort minus test
    n_pos = sum(pid_label.get(p) == 1 for p in test_pids)
    # participants != windows: unlabeled pids are used in pretraining ONLY (SSL is label-free).
    all_pids = np.unique(pids)
    n_labeled = len(set(cohort_pids) & set(all_pids))
    n_unlabeled = len(all_pids) - n_labeled
    pretrain_w, ft_w = int(pretrain_mask.sum()), int(finetune_mask.sum())
    print(f"[split] participants: {len(all_pids)} total = {n_labeled} labeled "
          f"[{len(test_pids)} test ({n_pos} dep/{len(test_pids)-n_pos} non) + {len(rest_cons)} probe] "
          f"+ {n_unlabeled} unlabeled (pretrain-only)")
    print(f"[split] windows: pretrain={pretrain_w} (= {ft_w} labeled-non-test + {pretrain_w-ft_w} unlabeled) "
          f"| test-pids' windows={int(test_mask.sum())}")

    # 3. CoST self-supervised pretraining --------------------------------
    seq_len = X.shape[1]
    kernels = args.kernels if args.kernels is not None else paper_kernels(seq_len)
    # The CoST seasonal (Fourier) layer is sized to max_train_length, so it must
    # equal the window length T; clamp any larger request down to T.
    max_train_length = seq_len if args.max_train_length is None else min(args.max_train_length, seq_len)
    # clock/time-feature channels are appended after the sensors; route them into
    # the encoder's temporal-encoding path (not input_fc / the seasonal branch).
    n_time_features = int(X.shape[-1]) - int(data["n_sensors"])
    model = CoST(
        input_dims=X.shape[-1],
        n_time_features=n_time_features,
        kernels=kernels,
        alpha=args.alpha,
        max_train_length=max_train_length,
        output_dims=args.repr_dims,
        hidden_dims=args.hidden_dims,
        depth=args.depth,
        backbone=args.backbone,
        pe=args.pe,
        time2vec_dim=args.time2vec_dim,
        loss_balance=args.loss_balance,
        bins_per_day=(24 * 60 // args.bin_minutes),
        disentangle=args.disentangle,
        jitter_sigma=args.jitter_sigma,
        shift_sigma=args.shift_sigma,
        moco_k=args.moco_k,
        trend_pool=args.trend_pool,
        seasonal_bands=args.seasonal_bands,
        negatives=args.negatives,
        n_negatives=args.n_negatives,
        positive_pair=args.positive_pair,
        smooth_bins=args.smooth_bins,
        phase_readout=args.phase_readout,
        decomp_aug=args.decomp_aug,
        n_sensors=data["n_sensors"],
        mask_mode=args.mask_mode,
        mask_prob=args.mask_keep_prob,
        phase_mode=args.phase_encoding,
        device=device,
        lr=args.lr,
        batch_size=args.batch_size,
    )
    # Hold out a small slice of pretrain windows to monitor the SSL validation loss. The
    # holdout is by PARTICIPANT, not by window: windows of one person share their physiology
    # (and are already per-participant z-scored), so a window-level split would put the same
    # people on both sides and the val loss would measure "can I contrast windows of someone I
    # trained on" instead of generalisation to a new person. That loss drives the
    # best-checkpoint restore in CoST.fit(), so a contaminated signal selects the checkpoint.
    # (Not a test leak either way -- pretrain_mask already excludes every test participant.)
    rng = np.random.default_rng(split_seed)
    pre_pids = np.unique(pids[pretrain_mask])
    rng.shuffle(pre_pids)
    n_val_pids = int(round(args.pretrain_val_frac * len(pre_pids)))
    pre_val_mask = np.zeros_like(pretrain_mask)
    if n_val_pids >= 1:
        cand = pretrain_mask & np.isin(pids, pre_pids[:n_val_pids])
        # CoST.fit builds the val loader with drop_last=True, so a holdout smaller than one
        # batch yields zero batches (no val loss, no best-checkpoint). Keep all windows for
        # training in that case rather than silently losing them.
        if int(cand.sum()) >= args.batch_size:
            pre_val_mask = cand
    pre_train_mask = pretrain_mask & ~pre_val_mask
    print(f"[pretrain] DSSL backbone={args.backbone} pe={args.pe} on {int(pre_train_mask.sum())} "
          f"windows from {len(np.unique(pids[pre_train_mask]))} pids "
          f"(+{int(pre_val_mask.sum())} windows from "
          f"{len(np.unique(pids[pre_val_mask])) if pre_val_mask.any() else 0} disjoint pids "
          f"held out for val loss) ...")
    # Return value discarded on purpose: the curves are read later off model.loss_log /
    # model.val_loss_log, which fit() sets and which survive the best-checkpoint restore.
    model.fit(X[pre_train_mask],
              valid_data=X[pre_val_mask] if pre_val_mask.any() else None,
              n_epochs=args.epochs, n_iters=args.iters, verbose=True,
              pids=pids[pre_train_mask],
              valid_pids=pids[pre_val_mask] if pre_val_mask.any() else None)
    pretrain_seconds = time.time() - t_start
    # How often the subject-conditional sampler had fewer than --n-negatives slots for a
    # participant and had to draw with replacement. It never falls back to foreign negatives,
    # so a high rate does not invalidate the arm -- but it does thin the denominator, so the
    # number belongs in the record rather than in a log nobody reads.
    _short, _calls = model.cost.neg_short, model.cost.neg_calls
    neg_shortfall = (_short / _calls) if _calls else 0.0
    print(f"[pretrain] negatives={args.negatives} n={args.n_negatives} "
          f"| shortfall {_short:,}/{_calls:,} queries ({neg_shortfall:.1%})")

    # 4. representations + classifier ------------------------------------
    # pooled over the whole window (args.pool; default mean) rather than the last timestep.
    # The seasonal half uses --season-pool ('spec' by default; 'same' falls back to args.pool).
    season_pool = None if args.season_pool == "same" else args.season_pool
    reprs = model.encode(X, mode="forecasting", pool=args.pool,
                         season_pool=season_pool).squeeze(1)                 # (N, repr_dims)

    ft_pids = sorted(rest_cons)
    # --probe-unit decides what ONE probe sample is. All three modes keep the participant-level
    # split; they differ only in how a participant's windows reach the classifier.
    #
    #   last        one row per participant: its most recent window (closest to the endpoint
    #               survey). No pseudo-replication, but ~96% of the test windows go unused.
    #   all         every window is a sample carrying its participant's label. Uses all data,
    #               but the probe sees ~26 correlated copies per person, so the effective n is
    #               the participant count while the fit and its regularisation behave as if it
    #               were the window count, and participants with longer records dominate.
    #   persubject  one row per participant holding [mean | std] of that participant's window
    #               embeddings. Keeps every window (through the summary) AND keeps n equal to
    #               the participant count. The std half carries within-person variability,
    #               which a single window cannot express.
    #
    # persubject builds the aggregate for every window row and then selects one row per
    # participant, so all rows of a participant are identical and the existing mask machinery
    # (train/val/test, CV folds, participant_aggregate) keeps working untouched.
    if args.probe_unit == "persubject":
        # Rows come from the one canonical builder (_eval_protocols.persubject_rows) that the
        # RQ3 ladder and the Separability table also use, then are broadcast back so the mask
        # machinery below is untouched.
        X_ps, _, ps_ids = persubject_rows(reprs, pids, y, np.ones(len(pids), bool))
        agg = np.zeros((len(pids), reprs.shape[1] * 2), dtype=reprs.dtype)
        for row, p in zip(X_ps, ps_ids):
            agg[pids == p] = row
        reprs = agg
        probe_sel = last_window_mask          # representative row; every row of a pid is equal
    elif args.probe_unit == "last":
        probe_sel = last_window_mask
    else:                                      # "all"
        probe_sel = np.ones(len(pids), bool)
    print(f"[probe] unit={args.probe_unit} -> {int(probe_sel.sum())} probe rows "
          f"of dim {reprs.shape[1]}" + (f", PCA<={args.probe_pca}" if args.probe_pca else ""))

    # A single participant-level train/val split of the probe pool. Used by the probe in
    # non-CV mode, and ALWAYS by the auxiliary supervised/rhythm analyses below (which need
    # their own held-out split for early stopping / threshold selection).
    rem_pids, val_pids = stratified_pid_holdout(ft_pids, pid_label, args.val_frac, split_seed)
    train_mask = np.isin(pids, list(rem_pids)) & probe_sel
    val_mask = (np.isin(pids, list(val_pids)) & probe_sel) if val_pids else train_mask

    # Same participant split, but WITHOUT probe_sel. `probe_sel` exists to stop one
    # participant-level depression label from being pseudo-replicated across ~26 correlated
    # windows; the DRS probe (RQ1) predicts a per-window harmonic reference instead, so its
    # target varies row by row and there is nothing to pseudo-replicate. Restricting it would
    # only throw away ~96% of the fitting data -- and asymmetrically, since test_mask is
    # already unrestricted. The participant-disjoint split is what protects that analysis, and
    # it is preserved here.
    train_mask_all = np.isin(pids, list(rem_pids))
    val_mask_all = np.isin(pids, list(val_pids)) if val_pids else train_mask_all

    if args.cv_folds >= 2:
        # k-fold CV WITHIN the probe pool: threshold from pooled out-of-fold predictions
        # (stable, uses all pool participants) + final probe refit on ALL pool participants.
        clf, thr, cv_metrics = probe_cv_within_pool(
            reprs, y, pids, pid_label, ft_pids, probe_sel,
            args.cv_folds, args.probe_c, model_seed, probe_mode=args.probe_mode,
            n_pca=args.probe_pca, fold_seed=split_seed)
        n_probe_train = int((np.isin(pids, ft_pids) & probe_sel).sum())   # refit on all pool
        print(f"[probe] {cv_metrics['n_folds']}-fold CV within {cv_metrics['n_pool_participants']} "
              f"pool participants: internal OOF AUC={cv_metrics['auc_roc']:.3f} "
              f"BAcc={cv_metrics['balanced_accuracy']:.3f} MCC={cv_metrics['mcc']:.3f}; "
              f"threshold={thr:.3f}; final probe refit on all pool participants.")
    else:
        # single split: probe fit on the train split, threshold tuned on the val split
        clf = make_probe(args.probe_mode, args.probe_c, model_seed,
                          clamp_pca(args.probe_pca, int(train_mask.sum()), reprs.shape[1]))
        clf.fit(reprs[train_mask], y[train_mask])
        val_pid_prob, val_pid_lbl = participant_aggregate(
            pids[val_mask], clf.predict_proba(reprs[val_mask])[:, 1], y[val_mask])
        thr = best_threshold(val_pid_lbl, val_pid_prob)
        cv_metrics = None
        n_probe_train = int(train_mask.sum())

    def predict(mask):
        return clf.predict_proba(reprs[mask])[:, 1]

    # 5. evaluate on the held-out test set (last-week window per pid, or all windows).
    # With one window per pid, participant_aggregate is a no-op (win == pid).
    eval_mask = test_mask & probe_sel
    test_prob = predict(eval_mask)
    win = binary_metrics(y[eval_mask], test_prob, thr)
    pid_prob, pid_lbl = participant_aggregate(pids[eval_mask], test_prob, y[eval_mask])
    pid = binary_metrics(pid_lbl, pid_prob, thr)
    # Bootstrap CI over PARTICIPANTS for both. At participant level each row is already one
    # person, so this is an ordinary bootstrap; at window level it keeps a person's windows
    # together. With ~36 test participants the interval is wide -- that is the honest width,
    # and it is what the AUROC should be quoted with.
    # Resampled over the TEST participants, so it belongs to the split side: a given cohort
    # gets the same CI resampling regardless of which model_seed is being evaluated on it.
    uniq_eval = np.unique(pids[eval_mask])
    win["auc_ci"] = participant_bootstrap_auc(y[eval_mask], test_prob, pids[eval_mask],
                                              seed=split_seed)
    pid["auc_ci"] = participant_bootstrap_auc(pid_lbl, pid_prob, uniq_eval, seed=split_seed)

    # --- prevalence transport, fixed operating points, calibration --------------------
    # The test cohort is 50/50 by construction (--test-per-class) and `thr` was tuned to
    # maximise balanced accuracy. AUROC, sensitivity, specificity and balanced accuracy
    # survive that unchanged; accuracy / F1 / MCC / PPV above are quoted at an implied 50%
    # base rate and do NOT transfer to a deployment population. Everything below exists so
    # the headline numbers cannot be read as if they did.
    labeled_cohort = [p for p in sorted(set(cohort_pids) & set(all_pids))
                      if pid_label.get(p) in (0, 1)]
    cohort_prev = float(np.mean([pid_label[p] for p in labeled_cohort])) if labeled_cohort else 0.5
    prevalences = list(args.target_prevalence) if args.target_prevalence else [round(cohort_prev, 4)]
    # Operating points fixed WITHOUT looking at test: the two sensitivity-anchored thresholds
    # come from val / out-of-fold predictions, and 0.5 is committed to a priori.
    if cv_metrics is not None:
        sel_prob = np.asarray(cv_metrics["oof_prob"]); sel_lbl = np.asarray(cv_metrics["oof_label"])
    else:
        sel_prob, sel_lbl = np.asarray(val_pid_prob), np.asarray(val_pid_lbl)
    fixed_thr = {"tuned_balanced_acc": float(thr), "nominal_0.5": 0.5}
    for s in (0.80, 0.90):
        fixed_thr[f"sens{int(s * 100)}_on_val"] = threshold_at_sensitivity(sel_lbl, sel_prob, s)

    for m, yt, yp in ((win, y[eval_mask], test_prob), (pid, pid_lbl, pid_prob)):
        m["at_prevalence"] = {f"{p:g}": prevalence_transport(m["sensitivity"], m["specificity"], p)
                              for p in prevalences}
        m["calibration"] = calibration_metrics(yt, yp)
        # The probe uses class_weight='balanced', so its scores are anchored at a 50% prior;
        # prior_shift re-anchors them before calibration is quoted at the target base rate.
        m["calibration_at_prevalence"] = {
            f"{p:g}": calibration_metrics(yt, prior_shift(yp, 0.5, p)) for p in prevalences}
        m["operating_points"] = operating_point_report(yt, yp, fixed_thr, prevalences)
    pid["cohort_prevalence"] = cohort_prev
    pid["test_prevalence"] = float(np.mean(pid_lbl)) if len(pid_lbl) else float("nan")

    _p0 = prevalences[0]
    _tp = pid["at_prevalence"][f"{_p0:g}"]
    print(f"[prevalence] test cohort is {pid['test_prevalence']:.0%} positive by construction; "
          f"observed cohort base rate {cohort_prev:.1%}")
    print(f"[prevalence] participant-level @thr={thr:.2f}: sens={pid['sensitivity']:.3f} "
          f"spec={pid['specificity']:.3f} (transfer) | F1={pid['f1']:.3f} acc={pid['accuracy']:.3f} "
          f"(at 50% only)")
    print(f"[prevalence] the same model at {_p0:.1%} prevalence: PPV={_tp['ppv']:.3f} "
          f"NPV={_tp['npv']:.3f} F1={_tp['f1']:.3f} acc={_tp['accuracy']:.3f}")
    print(f"[calibration] Brier={pid['calibration']['brier']:.4f} "
          f"ECE={pid['calibration']['ece']:.4f}")

    # results_hrd/<run_id>/<backbone>_<pe>_seed<seed>/{metrics.json, pretrain_loss.npy}
    run_id = args.run_id or os.environ.get("SLURM_JOB_ID") or "local"
    clock_tag = "_clock" if args.with_clock_features else ""
    dis_tag = "" if args.disentangle else "_plain"     # plain SSL (no disentangler) baseline
    # Tag only non-default seasonal readouts, so the 'same' (MESOR-only) ablation cannot
    # overwrite the default 'spec' run of the same variant.
    sp_tag = "" if args.season_pool == "spec" else f"_sp-{args.season_pool}"
    # Crossed runs need distinct folders, but the common case (one seed for both) must keep
    # producing the historical `_seed<n>` name so archived result paths stay valid.
    seed_tag = (f"_seed{model_seed}" if split_seed == model_seed
                else f"_seed{model_seed}_split{split_seed}")
    variant_dir = (out_dir / run_id /
                   f"{args.backbone}_{args.pe}{dis_tag}{seed_tag}{clock_tag}{sp_tag}")
    variant_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "backbone": args.backbone,
        "pe": args.pe,
        "window_level": win,
        "participant_level": pid,
        # internal k-fold CV within the probe pool (test untouched); null when --cv-folds<2
        "cv_internal": cv_metrics,
        # --- participants (the true statistical unit) ---
        "n_participants_total": int(len(all_pids)),
        "n_labeled_participants": int(n_labeled),
        "n_test_participants": int(len(test_pids)),
        "n_probe_participants": int(len(rest_cons)),
        # WHICH participants, not just how many. The split is a deterministic function of
        # --seed, but reproducing it after the fact needs the post-windowing pid pool, so it
        # was not recoverable from a finished run. Recorded here so seed-to-seed differences
        # (e.g. the cosinor baseline moving 0.525 -> 0.608 across seeds in run 18975686) can
        # be traced to the actual test cohort.
        "test_pids": sorted(map(str, test_pids)),
        "test_pids_depressed": sorted(str(p) for p in test_pids if pid_label.get(p) == 1),
        "n_unlabeled_participants": int(n_unlabeled),
        # --- samples ACTUALLY used (== participants when --probe-last-window) ---
        "n_probe_train_samples": int(n_probe_train),
        "n_test_samples": int(eval_mask.sum()),
        # --- windows (reference): pretrain excludes ALL test-pid windows; unlabeled pids
        #     appear in pretrain only (SSL is label-free) ---
        "n_pretrain_windows": int(pretrain_mask.sum()),
        "n_test_pid_windows": int(test_mask.sum()),
        "n_finetune_pid_windows": int(finetune_mask.sum()),
        "pretrain_seconds": pretrain_seconds,
        "neg_shortfall_rate": neg_shortfall,
        "n_params": int(sum(p.numel() for p in model.net.parameters())),
        "n_params_trainable": int(sum(p.numel() for p in model.cost.parameters()
                                      if p.requires_grad)),
        "seq_len": int(seq_len),
        "n_features": int(X.shape[-1]),
        "gradnorm_final_weights": ([float(w) for w in model.loss_w_log[-1]]
                                   if getattr(model, "loss_w_log", None) else None),
        # Record the RESOLVED seeds, not the raw None defaults, so a collector can group runs
        # by cohort (split_seed) vs optimisation (model_seed) without re-deriving the fallback.
        "config": {**vars(args), "split_seed": split_seed, "model_seed": model_seed},
    }
    # Human-readable companion, so the balanced-cohort / tuned-threshold / seed-confounding
    # caveats travel WITH the numbers instead of living only in the code.
    write_run_report(variant_dir, args, result, split_seed, model_seed)
    print(f"[report] {variant_dir/'report.md'}")
    (rq_path(variant_dir, "metrics.json")).write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    np.save(rq_path(variant_dir, "pretrain_loss.npy"), np.asarray(model.loss_log))
    np.save(rq_path(variant_dir, "loss_iters.npy"), np.asarray(model.iters_log))
    # Encoder weights, ~235 MB each -- 59% of that is the complex SFD table alone (337x320x160
    # at 8 bytes). Keeping every variant of a 52-task sweep costs ~12 GB, so this is OFF by
    # default and run.sh enables it for one seed: enough to redo any latent analysis without a
    # 3.4 h retrain, at a quarter of the storage.
    if args.save_encoder:
        model.save(rq_path(variant_dir, "encoder.pt"))
    if getattr(model, "loss_w_log", None):                      # GradNorm task-weight trajectory
        np.save(rq_path(variant_dir, "gradnorm_weights.npy"), np.asarray(model.loss_w_log))
    if model.val_loss_log:
        np.save(rq_path(variant_dir, "val_loss.npy"), np.asarray(model.val_loss_log))
    # one depressed + one non-depressed test window: raw signal + encoded rep, for the
    # cross-technique before/after-embedding figure (plot_position_similarity.py --signalviz)
    try:
        save_signal_embedding(model, X, pids, test_pids, pid_label, variant_dir,
                              data["sensor_cols"], data["n_sensors"])
    except Exception as e:
        print(f"[signal-embed] skipped ({e})")
    save_loss_curves(model.iters_log, model.loss_log, model.val_loss_log,
                     variant_dir, tag=f"{args.backbone} / {args.pe}")

    print("\n========== HELD-OUT TEST RESULTS ==========")
    print(f"backbone = {args.backbone}  pe = {args.pe}")
    print(f"window-level       AUC={win['auc_roc']:.3f}  F1={win['f1']:.3f}  Acc={win['accuracy']:.3f}"
          f"  BAcc={win['balanced_accuracy']:.3f}  MCC={win['mcc']:.3f}")
    print(f"participant-level  AUC={pid['auc_roc']:.3f}  F1={pid['f1']:.3f}  Acc={pid['accuracy']:.3f}"
          f"  BAcc={pid['balanced_accuracy']:.3f}  MCC={pid['mcc']:.3f}")
    _ci = pid.get("auc_ci") or {}
    if np.isfinite(_ci.get("lo", float("nan"))):
        print(f"participant AUC 95% CI = [{_ci['lo']:.3f}, {_ci['hi']:.3f}]  "
              f"(bootstrap over {_ci['n_participants']} test participants)")
    print(f"decision threshold = {thr:.3f}  (tuned on {'pooled OOF CV' if cv_metrics else 'val split'})")
    print(f"saved -> {rq_path(variant_dir, 'metrics.json')}")

    # --- Optional: emotional-energy downstream, REUSING this pretrained encoder --------
    # No second pretraining -- the frozen `model` is probed for daily emotional_energy on the
    # SAME held-out test participants (already excluded from pretraining -> leakage-free).
    # Sliding trailing windows, one per labelled day. Non-fatal: never abort the depression run.
    if args.energy_probe:
        try:
            from data_processing.data_preprocessing import (prepare_hrd_energy_sliding,
                                                clear_hrd_cache)
            from tasks.energy import run_energy_tasks
            print("[energy] emotional-energy probe (reusing frozen encoder, sliding windows) ...")
            _t = time.time()
            edata = prepare_hrd_energy_sliding(
                args.sensor_csv, window_hours=args.window_hours,
                bin_minutes=args.bin_minutes, build_pretrain=False,
                max_missing=args.max_missing,          # same cleaning as the depression path,
                                                       # so the cached CSV parse is reused
                energy_stride=args.energy_stride,
                clock_features=args.with_clock_features,   # match the encoder's channel count
                calendar_index=args.pe in CALENDAR_PES)
            clear_hrd_cache()                          # last consumer -- free the ~1 GiB tables
            Xe, eee, pe = edata["X"], edata["ee"], edata["pids"]
            print(f"[energy] {len(Xe):,} probe windows (every {args.energy_stride} labelled "
                  f"day(s)) in {time.time() - _t:.0f}s")
            labe = np.isfinite(eee)
            # test = the SAME participants held out of pretraining; train/val from the rest
            te_pids_e = set(test_pids) & set(pe[labe])
            rest = np.array(sorted(set(pe[labe]) - te_pids_e))
            rng_e = np.random.default_rng(split_seed); rng_e.shuffle(rest)   # a split -> split_seed
            n_val = max(1, int(round(0.15 * len(rest))))
            va_pids_e, tr_pids_e = set(rest[:n_val]), set(rest[n_val:])
            te_e = np.isin(pe, list(te_pids_e)) & labe
            va_e = np.isin(pe, list(va_pids_e)) & labe
            tr_e = np.isin(pe, list(tr_pids_e)) & labe
            # The "_energy" suffix is load-bearing, do not drop it. This directory holds the
            # SAME file names as the depression variant directory (metrics.json,
            # hrd_rhythm.json, hrd_rhythm_separability.*, frequency_*, the t-SNE/UMAP PNGs),
            # because run_energy_tasks and run_hrd_rhythm_analysis are shared between the two
            # downstreams. With identical directory names, copying one results tree into the
            # other silently overwrites 13 depression files with their energy counterparts and
            # reports no error. Making the directory name unique means the two trees can be
            # merged, moved or mis-copied without any file ever colliding.
            e_out = (Path(args.energy_output_dir) / run_id /
                     f"{args.backbone}_{args.pe}{dis_tag}{seed_tag}{clock_tag}{sp_tag}_energy")
            mode_desc = ("**Mode B (sliding), encoder REUSED from the depression run**: one "
                         "trailing 7-day window per labelled day ([D-6, D] -> EE(D)); test = the "
                         "participants already held out of pretraining, so leakage-free.")
            # Same encoder-based controls the depression ladder carries, so the two
            # downstreams are compared against the same set. The plain twin is pretrained here
            # with the SAME cache path experiment_q1/q3 use, so it is trained once per variant
            # and reloaded there -- moving the cost earlier, not adding one.
            extra = {}
            try:
                # module, so a top-level edge back would be a cycle.
                from tasks._experiment_common import PLAIN_REF
                # NO "Cosinor (paper)" rung here, and the omission is deliberate.
                #
                # The design's ladder does call for it, and it was added and then removed after
                # measurement. `prepare_hrd_energy_sliding` returns no `window_ids`, so
                # baselines/cosinor.py::_start_bins falls back to zeros and warns that "phase
                # parameters are NOT clock-anchored and not comparable across participants".
                # Every phase would then be relative to an arbitrary per-participant hour --
                # constant within a person, meaningless between people, which is precisely the
                # comparison a participant-split probe makes. The rung would have produced
                # numbers, and they would have been wrong in a direction no plot reveals.
                #
                # It also collided in the cache. Without window_ids the key is positional
                # (`arange(N)@0`), and the energy rhythm analysis writes the SAME file from a
                # SUBSET of the windows (Xe[sub]), so index 5 means a different window in each
                # call and _load_cache would return one window's fit for another.
                #
                # To restore it: have prepare_hrd_energy_sliding emit window_ids in the
                # depression path's format, f"{pid}_{window_start.isoformat()}", then pass them
                # here AND at the run_hrd_rhythm_analysis call below. Both, or the cache
                # collision comes back.
                extra["Random-init"] = random_init_repr(vars(args), Xe, data['n_sensors'], device, model_seed,
                                              pool=args.energy_pool)
                if ((args.backbone, args.pe) in PLAIN_REF and args.disentangle
                        and not args.no_plain_ssl):
                    from baselines.plain_ssl import encode_plain, plain_ssl_encoder
                    _plain = plain_ssl_encoder(X, pretrain_mask, vars(args), data["n_sensors"],
                                               device, seed=model_seed, pids=pids,
                                               cache_path=rq_path(variant_dir, "plain_encoder.pt"))
                    extra["DSSL plain (no disentangle)"] = encode_plain(_plain, Xe, vars(args))
            except Exception as e:
                print(f"[energy] extra ladder rungs SKIPPED (non-fatal): {type(e).__name__}: {e}")
            # Top rung: end-to-end supervised, trained on the ENERGY label. One training per
            # binary task, and each one costs roughly what the depression supervised rung
            # costs -- budget for it before trusting the 3 h wall clock.
            def _energy_supervised(ybin, tr_m, va_m, te_m, te_pids_, tag):
                from baselines.supervised import supervised_baseline_row
                from tasks._eval_protocols import (binary_metrics,
                                                   participant_bootstrap_auc)
                # From Xe, not X: the energy windows carry their own channel count.
                _, tprob, thr = supervised_baseline_row(
                    Xe, ybin, pe, tr_m, va_m, te_m, args.backbone, args.pe,
                    f"Supervised ({tag})",
                    n_time_features=int(Xe.shape[-1]) - int(data["n_sensors"]),
                    hidden_dims=args.hidden_dims, depth=args.depth,
                    output_dims=args.repr_dims, device=device, seed=model_seed,
                    return_window_scores=True)
                m = binary_metrics(ybin[te_m], tprob, thr)
                m["auc_ci"] = participant_bootstrap_auc(ybin[te_m], tprob, te_pids_,
                                                        seed=model_seed)
                return m

            _t = time.time()
            run_energy_tasks(model, Xe, eee, pe, data["n_sensors"], tr_e, va_e, te_e,
                             args.energy_pool, args.energy_threshold, model_seed,
                             e_out, mode_desc,
                             {**vars(args), "split_seed": split_seed, "model_seed": model_seed},
                             season_pool=season_pool, ee_win=edata.get("ee_win"),
                             extra_reprs=extra,
                             supervised_fn=None if args.no_energy_supervised
                             else _energy_supervised)
            print(f"[energy] probe tasks done in {time.time() - _t:.0f}s")
            # Same rhythm figures as depression, but split by energy PER DAY (high- vs
            # low-energy days). subject_aggregate=False: each window is its own unit (a person
            # spans both classes), so the per-subject figures/trajectory are skipped. Only in
            # disentangle mode (needs the trend/seasonal branches). Non-fatal.
            if args.disentangle and not args.no_rhythm_viz:
                try:
                    from tasks.rhythm import run_hrd_rhythm_analysis
                    y_e = (eee >= args.energy_threshold).astype(int)
                    # Rhythm analysis runs on a NON-OVERLAPPING subsample of the energy windows,
                    # because it builds the seasonal amplitude/phase views at 337x120 = 40,440
                    # features per window: ~3.2 GB per view over 20k windows, ~13 GB across the
                    # four of them, which is what OOM-killed 17 of 28 tasks in array 18535364.
                    # Target spacing is one window per window_hours/24 days (the same
                    # non-overlapping spacing the depression path uses). The probe windows are
                    # ALREADY --energy-stride days apart, so step only the remaining factor --
                    # stepping the full 7 again would subsample 21 days apart and needlessly
                    # shrink the figures. Sorted by day explicitly, not by construction order.
                    stride = max(1, math.ceil((args.window_hours // 24) / args.energy_stride))
                    days_e = edata["days"]
                    sub = np.zeros(len(pe), bool)
                    for _p in np.unique(pe):
                        _i = np.where(pe == _p)[0]
                        sub[_i[np.argsort(days_e[_i], kind="stable")][::stride]] = True
                    print(f"[energy] rhythm figures split by high-vs-low-energy day "
                          f"({int(sub.sum()):,} of {len(pe):,} windows, every {stride}th day) ...")
                    _t = time.time()
                    run_hrd_rhythm_analysis(
                        model, Xe[sub], y_e[sub], pe[sub], tr_e[sub], te_e[sub], e_out,
                        seq_len=seq_len, bin_minutes=args.bin_minutes, seed=model_seed,
                        sensor_cols=data["sensor_cols"],
                        label_names={0: f"low-energy day (EE<{args.energy_threshold:g})",
                                     1: f"high-energy day (EE>={args.energy_threshold:g})"},
                        batch_size=args.batch_size, val_mask=va_e[sub], pool=args.energy_pool,
                        season_pool=(None if args.season_pool == "same" else args.season_pool),
                        # the same two ladder controls the depression table carries, already
                        # encoded above for the EE probe -- indexed to the subsample so every
                        # row of this table is scored on identical windows
                        extra_views={
                            **{k: np.asarray(v)[sub] for k, v in extra.items()},
                            "Majority": np.ones((int(sub.sum()), 1), dtype=np.float32),
                            "Handcrafted (mean/std)": handcrafted_features(
                                Xe[sub], data["n_sensors"])},
                        probe_c=args.probe_c, paper_cosinor_topk=args.paper_cosinor_topk,
                        baseline_by_pid=None, subject_aggregate=False,
                        label_noun="emotional energy", table_tag="energy",
                        headline_unit="all")   # a per-day label: every window is a sample
                    print(f"[energy] rhythm figures -> {e_out} ({time.time() - _t:.0f}s)")
                except Exception as e:
                    import traceback
                    print(f"[energy] rhythm figures FAILED (non-fatal): {e}")
                    traceback.print_exc()
        except Exception as e:
            import traceback
            print(f"[energy] FAILED (non-fatal): {e}")
            traceback.print_exc()

    # Steps 6a-6c are DISENTANGLEMENT diagnostics: they read a trend branch and a
    # seasonal branch out of the latent. In plain mode there is a single representation
    # (no branches), so these are meaningless and are skipped -- the plain run produces
    # only the prediction result (metrics.json) so it can be compared, on the SAME
    # downstream probe, against the disentangled CoST run. This is the whole point of the
    # ablation: does the trend/seasonal-disentangled embedding predict better than a plain
    # single-representation SSL embedding, or is learning just as good without it?
    if not args.disentangle:
        print("[plain] single-representation SSL (no trend/seasonal branch): "
              "disentanglement/rhythm eval skipped -> prediction metrics.json only.")
    else:
        # 6a. supervised end-to-end baselines (plain TCN + Transformer-sinusoidal, no SSL):
        #     same backbone size as this run, trained directly on the label, added as baseline
        #     rows in the separability table next to Cosinor. Non-fatal.
        baseline_rows = []
        # One try PER RUNG. The try used to wrap the whole loop and reset baseline_rows in
        # its handler, so when the transformer rung ran out of memory it also discarded the
        # TCN rung that had already trained successfully -- the ladder lost both supervised
        # rows when only one had failed (run 1608369, all 22 tasks).
        for bb, pe_, nm in [("tcn", "none", "TCN (supervised)"),
                            ("transformer", "sinusoidal", "Transformer-sin (supervised)")]:
            try:
                print(f"[baseline] training supervised {nm} ...")
                baseline_rows.append(supervised_baseline_row(
                    X, y, pids, train_mask, val_mask, eval_mask, bb, pe_, nm,
                    n_time_features, args.hidden_dims, args.depth, args.repr_dims,
                    device=device, seed=model_seed, batch_size=args.batch_size))
            except Exception as e:
                import traceback
                print(f"[baseline] supervised {nm} FAILED (non-fatal): {e}")
                traceback.print_exc()
            # Outside the handler, so the traceback -- which holds the failed frame and with
            # it the half-built net and its activations -- has already been released.
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()

        # 6a-bis. the two ladder CONTROLS, encoded once and reused by both separability
        #     tables. Random-init isolates what TRAINING bought; the plain twin isolates what
        #     DISENTANGLING bought. The plain encoder is cached to plain_encoder.pt, so when
        #     the energy path has already built it this costs a load, not a pretraining.
        # The full RQ3 ladder, so the headline table and the utility table rank the same
        # things. `Majority` is a CONSTANT column rather than a hardcoded 0.5: run through the
        # identical probe it must come out at AUC 0.500, so any other value is a leak in the
        # probe pipeline, not a property of a representation. That check is worth more than
        # writing the constant down.
        ladder_views = {"Majority": np.ones((len(X), 1), dtype=np.float32),
                        "Handcrafted (mean/std)": handcrafted_features(X, data["n_sensors"])}
        # Each rung is guarded on its OWN, so one failure costs one row. A single try around
        # the whole block used to reset ladder_views to {}, which meant a missing plain twin
        # silently took Majority, Handcrafted and Random-init down with it -- i.e. the run
        # most in need of a floor was the one that lost it.
        try:
            ladder_views["Random-init"] = random_init_repr(
                vars(args), X, data["n_sensors"], device, model_seed, pool=args.pool)
        except Exception as e:
            print(f"[ladder] Random-init FAILED (non-fatal): {type(e).__name__}: {e}")
        if args.no_plain_ssl:
            print("[ladder] plain-SSL twin SKIPPED (--no-plain-ssl)")
        else:
            try:
                from baselines.plain_ssl import encode_plain, plain_ssl_encoder
                _pl = plain_ssl_encoder(X, pretrain_mask, vars(args), data["n_sensors"],
                                        device=device, seed=model_seed, pids=pids,
                                        cache_path=rq_path(variant_dir, "plain_encoder.pt"))
                ladder_views["DSSL plain (no disentangle)"] = encode_plain(_pl, X, vars(args))
            except Exception as e:
                print(f"[ladder] plain twin FAILED (non-fatal): {type(e).__name__}: {e}")

        # 6b. on-real-data rhythm analysis (test set): t-SNE coloured by endpoint +
        #    seasonal amplitude/phase profiles + a per-representation separability
        #    table. Non-fatal: a failure here must not discard the results above.
        if not args.no_rhythm_viz:
            try:
                from tasks.rhythm import run_hrd_rhythm_analysis
                print("[rhythm] HRD test-set amplitude/phase analysis ...")
                run_hrd_rhythm_analysis(
                    model, X, y, pids, train_mask, test_mask, variant_dir,
                    seq_len=seq_len, bin_minutes=args.bin_minutes, seed=model_seed,
                    sensor_cols=data["sensor_cols"],
                    label_names={0: "non-depressed (0)", 1: "depressed (1)"},
                    batch_size=args.batch_size, val_mask=val_mask, baseline_rows=baseline_rows,
                    extra_views=ladder_views,
                    window_ids=data.get("window_ids"), pool=args.pool,
                    season_pool=(None if args.season_pool == "same" else args.season_pool),
                    probe_sel=probe_sel, probe_c=args.probe_c,
                    paper_cosinor_topk=args.paper_cosinor_topk,
                    baseline_by_pid=data.get("baseline_by_pid"),
                    table_tag="depression", headline_unit=args.probe_unit,
                )
                print(f"[rhythm] saved figures + table -> {variant_dir}")
            except Exception as e:
                import traceback
                print(f"[rhythm] FAILED (non-fatal): {e}")
                traceback.print_exc()
                try:                          # surface the reason IN the results folder, not only the SLURM log
                    (rq_path(variant_dir, "hrd_rhythm.FAILED.txt")).write_text(
                        f"run_hrd_rhythm_analysis failed: {type(e).__name__}: {e}\n\n"
                        + traceback.format_exc(), encoding="utf-8")
                except Exception:
                    pass

        # 6c. trend/seasonal Decomposition Recovery Score (DRS): read-only probe of the
        #     frozen latents against per-channel harmonic-regression references. Non-fatal.
        if not args.no_rhythm_viz:
            try:
                from tasks.decomposition import run_decomposition_recovery
                print("[decomp] trend/seasonal recovery (DRS) ...")
                agg = run_decomposition_recovery(
                    model, X, train_mask_all, test_mask, variant_dir,
                    bin_minutes=args.bin_minutes,
                    sensor_cols=data["sensor_cols"], seed=model_seed, batch_size=args.batch_size,
                    val_mask=val_mask_all, pids=pids)
                print(f"[decomp] Full recovery: tau={agg['rec_full_trend']:.3f} "
                      f"sigma={agg['rec_full_rhythm']:.3f}  (DIS={agg['DIS']:.3f}) -> {variant_dir}")
            except Exception as e:
                import traceback
                print(f"[decomp] FAILED (non-fatal): {e}")
                traceback.print_exc()
                try:                          # surface the reason IN the results folder, not only the SLURM log
                    (rq_path(variant_dir, "decomposition_recovery.FAILED.txt")).write_text(
                        f"run_decomposition_recovery failed: {type(e).__name__}: {e}\n\n"
                        + traceback.format_exc(), encoding="utf-8")
                except Exception:
                    pass

    print(f"total time = {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
