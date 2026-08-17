"""Emotional-energy probes on a frozen CoST representation.

Library first, runnable second::

    python -m tasks.energy --sensor-csv datasets/HRD_RAW_MinuteLevel.csv \
        --output-dir results_hrd_energy --run-id local

``run_energy_tasks`` takes an already-fitted model and returns / writes metrics; it is
shared with ``train_hrd.py --energy-probe`` so both paths probe identically. ``main``
below only adds the pretraining and the argument wiring.
"""

import argparse
import json
import time

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tasks._eval_protocols import (best_threshold, binary_metrics,
                                   participant_bootstrap_auc)

# Imported inside main() rather than here: tasks/ is a library first, and these pull in
# torch + the 3.4 GB CSV reader that an importing analysis script has no use for.



# --------------------------------------------------------------------------- #
# split + features
# --------------------------------------------------------------------------- #
def split_pids(pids_with_label, seed, val_frac, test_frac):
    """Random participant-level split -> (train, val, test) sets of pids."""
    rng = np.random.default_rng(seed)
    p = np.array(sorted(pids_with_label))
    rng.shuffle(p)
    n = len(p)
    n_test = max(1, int(round(test_frac * n)))
    n_val = max(1, int(round(val_frac * n)))
    return set(p[n_test + n_val:]), set(p[n_test:n_test + n_val]), set(p[:n_test])


def handcrafted_features(X, n_sensors):
    """Baseline (b): per-window mean & std of each sensor channel -> (N, 2*n_sensors)."""
    s = X[:, :, :n_sensors]
    return np.concatenate([s.mean(axis=1), s.std(axis=1)], axis=1)


# --------------------------------------------------------------------------- #
# probes (window-level metrics; split is by participant)
# --------------------------------------------------------------------------- #
def probe_binary(feat, ybin, tr, va, te, seed, te_pids=None):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=1.0, max_iter=3000,
                                           class_weight="balanced", random_state=seed))
    clf.fit(feat[tr], ybin[tr])
    thr = best_threshold(ybin[va], clf.predict_proba(feat[va])[:, 1])
    prob = clf.predict_proba(feat[te])[:, 1]
    m = binary_metrics(ybin[te], prob, thr)
    if te_pids is not None:
        m["auc_ci"] = participant_bootstrap_auc(ybin[te], prob, te_pids, seed=seed)
    return m


def _rho(a, b):
    return spearmanr(a, b).correlation


def probe_regression(feat, y, tr, te, te_pids=None):
    reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    reg.fit(feat[tr], y[tr])
    pred = reg.predict(feat[te])
    rho = spearmanr(y[te], pred).correlation
    m = {"spearman": float(rho) if np.isfinite(rho) else 0.0,
         "mae": float(np.mean(np.abs(y[te] - pred)))}
    if te_pids is not None:
        # Same clustered bootstrap as the binary tasks. Without it rho is a point estimate on
        # ~1,800 correlated windows from 36 people, which reads far more precise than it is.
        m["rho_ci"] = participant_bootstrap_auc(y[te], pred, te_pids, seed=0, stat=_rho)
    return m


def majority_binary(ybin, tr, te, te_pids=None):
    """Chance baseline: always predict the training-majority class."""
    prob = np.full(int(te.sum()), float(ybin[tr].mean()))
    m = binary_metrics(ybin[te], prob, 0.5)
    if te_pids is not None:
        # A constant score gives AUROC 0.5 in every draw, so the interval is degenerate by
        # construction. Reported anyway so the row reads consistently with the others.
        m["auc_ci"] = participant_bootstrap_auc(ybin[te], prob, te_pids, seed=0)
    return m


def mean_regression(y, tr, te):
    m = float(y[tr].mean())
    return {"spearman": 0.0, "mae": float(np.mean(np.abs(y[te] - m)))}


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _ci_str(m, key="auc_ci"):
    ci = m.get(key)
    if not ci or not np.isfinite(ci.get("lo", float("nan"))):
        return "-"
    return f"[{ci['lo']:.3f}, {ci['hi']:.3f}]"


def binary_table(title, rows):
    """rows: list of (name, metrics_dict). Returns a markdown table string."""
    head = (f"### {title}\n\n"
            "| Representation | AUROC | 95% CI (participant bootstrap) | Acc | BAcc | MCC | F1 |\n"
            "|---|---|---|---|---|---|---|\n")
    body = "".join(
        f"| {n} | {m['auc_roc']:.3f} | {_ci_str(m)} | {m['accuracy']:.3f} | "
        f"{m['balanced_accuracy']:.3f} | {m['mcc']:.3f} | {m['f1']:.3f} |\n"
        for n, m in rows)
    return head + body


def regression_table(title, rows):
    head = (f"### {title}\n\n"
            "| Representation | Spearman rho | 95% CI (participant bootstrap) | MAE |\n"
            "|---|---|---|---|\n")
    body = "".join(f"| {n} | {m['spearman']:.3f} | {_ci_str(m, 'rho_ci')} | {m['mae']:.3f} |\n"
                   for n, m in rows)
    return head + body


def run_energy_tasks(model, X, ee, pids, n_sensors, tr, va, te,
                     pool, energy_threshold, seed, out_dir, mode_desc, config,
                     season_pool=None, ee_win=None, extra_reprs=None):
    """Encode X with the FROZEN `model` and run the 3 emotional-energy tasks on the given
    participant-level masks (tr/va/te), writing report.md + metrics.json to out_dir.

    Shared by the standalone script and the reuse-inside-train_hrd path (train_hrd.py
    --energy-probe), so the encoder is pretrained ONCE and both entry points get identical
    probing. Returns the metrics dict."""
    lab = np.isfinite(ee)
    pids_with_label = set(pids[lab])
    reprs = model.encode(X, mode="forecasting", pool=pool,
                         season_pool=season_pool).squeeze(1)                    # (N, repr_dims)
    hc = handcrafted_features(X, n_sensors)                                     # (N, 2*n_sensors)
    # Ladder rungs, in the design's order. `extra_reprs` carries the encoder-based controls the
    # depression ladder already has (random-init, plain SSL) so the two downstreams are probed
    # against the SAME comparison set; the caller builds them because only it has the config.
    reps = {"Handcrafted (mean/std)": hc}
    reps.update(extra_reprs or {})
    reps["CoST (SSL repr)"] = reprs

    y_reg = ee
    y_hi = (ee >= energy_threshold).astype(int)                                 # task 1
    # Task 3: "is this a high day FOR THIS PERSON". EE is a 1-5 INTEGER scale, so a median
    # split by comparison is not a median split at all: on HRD `ee >= median` puts every tie
    # in the positive class (72% positive) and `ee > median` puts them all in the negative
    # one (22%). Neither is the 50/50 contrast the task claims, and the imbalance was
    # visible in the reported majority accuracy (0.68-0.74 where it should be ~0.50).
    # Rank within the participant instead -- 50/50 by construction whatever the ties.
    y_wp = np.zeros(len(ee), dtype=int)
    for p in pids_with_label:
        m = (pids == p) & lab
        y_wp[m] = (rankdata(ee[m], method="average") / int(m.sum()) > 0.5).astype(int)

    # Participants of the test rows: the unit the bootstrap CI resamples, because a person's
    # ~150 sliding windows overlap by six days and are not independent observations.
    tep = pids[te]
    t1 = ([("Majority (chance)", majority_binary(y_hi, tr, te, tep))]
          + [(n, probe_binary(R, y_hi, tr, va, te, seed, tep)) for n, R in reps.items()])
    t2 = ([("Mean predictor", mean_regression(y_reg, tr, te))]
          + [(n, probe_regression(R, y_reg, tr, te, tep)) for n, R in reps.items()])
    t3 = ([("Majority (chance)", majority_binary(y_wp, tr, te, tep))]
          + [(n, probe_binary(R, y_wp, tr, va, te, seed, tep)) for n, R in reps.items()])

    # Tasks 4-5: the SAME input scored against a window-matched target (mean EE over the
    # days the window spans) instead of EE on its last day alone. Tasks 1-3 keep the
    # day-level target; reporting both is what turns the timescale mismatch from a
    # confound into a measurement.
    # Deliberately regression-only. A threshold on a window MEAN is not the same cut as the
    # same threshold on a single day -- averaging shrinks the variance, so `>= 4` selects a
    # small extreme tail -- and any other value would be a free parameter. Spearman needs no
    # cut and discards nothing; the binary EE questions are already covered by Tasks 1 and 3.
    t4 = None
    if ee_win is not None:
        y_win = np.asarray(ee_win, dtype=float)
        okw = np.isfinite(y_win)
        trw, tew = tr & okw, te & okw
        yw0 = np.nan_to_num(y_win, nan=float(np.nanmean(y_win)))
        tepw = pids[tew]
        t4 = ([("Mean predictor", mean_regression(yw0, trw, tew))]
              + [(n, probe_regression(R, yw0, trw, tew, tepw)) for n, R in reps.items()])

    n_tr, n_va, n_te = (len(set(pids[tr])), len(set(pids[va])), len(set(pids[te])))
    pos_rate = float(y_hi[te].mean())
    tbl1 = binary_table(f"Task 1 - High-energy day (EE >= {energy_threshold:g})  "
                        f"[test positive rate = {pos_rate:.1%}]", t1)
    tbl2 = regression_table("Task 2 - Ordinal regression (EE 1-5)", t2)
    tbl3 = binary_table("Task 3 - Within-person high day (rank split at participant's median)", t3)
    tbl45 = ""
    if t4 is not None:
        tbl45 = "\n" + regression_table(
            "Task 4 - Regression on mean EE over the window  [window-matched target; pair "
            "with Task 2 -- identical features, metric, split and pooling, only the "
            "target's timescale differs]", t4)

    out_dir.mkdir(parents=True, exist_ok=True)
    report = (
        f"# Emotional-energy downstream report\n\n"
        f"**pool** `{pool}` | **seed** {seed}\n\n"
        f"- {mode_desc}\n"
        f"- Evaluation unit = window (a labelled day); split is participant-level "
        f"({n_tr}/{n_va}/{n_te} pids). NB: with sliding windows, windows of one participant "
        f"are correlated, so the independent unit is the participant.\n"
        f"- Labelled probe windows: {int(lab.sum()):,} of {len(ee):,} "
        f"(test = {int(te.sum()):,} windows).\n\n"
        f"{tbl1}\n{tbl2}\n{tbl3}\n"
        f"_A representation is useful when CoST beats both the majority/mean baseline "
        f"and the handcrafted mean/std summary._\n"
    )
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    metrics = {
        "config": config,
        "n_labelled_windows": int(lab.sum()),
        "n_test_windows": int(te.sum()), "test_positive_rate": pos_rate,
        "n_pids": {"train": n_tr, "val": n_va, "test": n_te},
        "task1_high_energy": {n: m for n, m in t1},
        "task2_regression": {n: m for n, m in t2},
        "task3_within_person": {n: m for n, m in t3},
        **({"task4_regression_week": {n: m for n, m in t4}} if t4 is not None else {}),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\n" + report)
    print(f"[energy] saved -> {out_dir/'report.md'} , {out_dir/'metrics.json'}")
    return metrics

# --------------------------------------------------------------------------- #
# CLI: python -m tasks.energy --sensor-csv ... --output-dir ... --run-id ...
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Emotional-energy probe on CoST representations")
    p.add_argument("--sensor-csv", required=True)
    p.add_argument("--backbone", default="transformer",
                   choices=["tcn", "transformer"])
    p.add_argument("--pe", default=None)
    p.add_argument("--pool", choices=["last", "mean", "max", "meanmax"], default="last",
                   help="How the 7-day representation is collapsed before the probe. "
                        "'last' (default) = final timestep, closest to the labelled day.")
    p.add_argument("--season-pool", choices=["spec", "spec_amp", "spec_phase", "same"],
                   default="spec",
                   help="Readout of the SEASONAL half only; see train_hrd.py --season-pool. "
                        "'same' = use --pool for it too (the DC/MESOR-only ablation).")
    p.add_argument("--energy-threshold", type=float, default=4.0,
                   help="EE >= this = 'high-energy day' (default 4 -> ~47%% positive, balanced).")
    p.add_argument("--window-hours", type=int, default=168)
    p.add_argument("--bin-minutes", type=int, default=15)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--repr-dims", type=int, default=320)
    p.add_argument("--hidden-dims", type=int, default=64)
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output-dir", default="./results_hrd_energy")
    p.add_argument("--run-id", default="local")
    return p.parse_args()


def main():
    from pathlib import Path

    from cost import CoST
    from data_processing.data_preprocessing import prepare_hrd_energy_sliding
    from train_hrd import paper_kernels
    from utils import init_dl_program

    args = parse_args()
    if args.pe is None:
        args.pe = "sinusoidal" if args.backbone == "transformer" else "none"
    device = init_dl_program(args.gpu if args.gpu >= 0 else "cpu", seed=args.seed)
    t0 = time.time()

    # 1. data: SLIDING trailing windows -- one 7-day window per labelled day ([D-6, D] ->
    # EE(D)), stride 1 day (Day1-7->EE7, Day2-8->EE8, ...). X / ee / pids = PROBE windows;
    # X_pre / pids_pre = the label-free non-overlapping PRETRAIN windows (same CSV read).
    data = prepare_hrd_energy_sliding(args.sensor_csv, window_hours=args.window_hours,
                                      bin_minutes=args.bin_minutes)
    X, ee, pids = data["X"], data["ee"], data["pids"]
    X_pre, pids_pre = data["X_pretrain"], data["pids_pretrain"]
    n_sensors = data["n_sensors"]
    lab = np.isfinite(ee)                                   # windows with a last-day EE label
    pids_with_label = set(pids[lab])

    # 2. participant-level split -----------------------------------------------------
    tr_pids, va_pids, te_pids = split_pids(pids_with_label, args.seed,
                                           args.val_frac, args.test_frac)
    tr = np.isin(pids, list(tr_pids)) & lab
    va = np.isin(pids, list(va_pids)) & lab
    te = np.isin(pids, list(te_pids)) & lab
    pretrain_mask = ~np.isin(pids_pre, list(te_pids))      # label-free; excludes test pids only
    print(f"[split] participants: {len(tr_pids)} train / {len(va_pids)} val / {len(te_pids)} test")
    print(f"[split] labelled probe windows: {int(tr.sum())} train / {int(va.sum())} val / "
          f"{int(te.sum())} test | pretrain windows: {int(pretrain_mask.sum())}")

    # 3. CoST self-supervised pretraining (unchanged; label-free) --------------------
    seq_len = X.shape[1]
    model = CoST(input_dims=X.shape[-1], n_time_features=X.shape[-1] - n_sensors,
                 kernels=paper_kernels(seq_len), alpha=0.0005, max_train_length=seq_len,
                 output_dims=args.repr_dims, hidden_dims=args.hidden_dims, depth=args.depth,
                 backbone=args.backbone, pe=args.pe,
                 bins_per_day=(24 * 60 // args.bin_minutes),
                 device=device, lr=args.lr, batch_size=args.batch_size)
    print(f"[pretrain] backbone={args.backbone} pe={args.pe} on {int(pretrain_mask.sum())} windows ...")
    model.fit(X_pre[pretrain_mask], n_epochs=args.epochs, n_iters=args.iters, verbose=True)

    # 4. freeze -> encode -> 3 energy tasks -> report (shared with train_hrd --energy-probe)
    mode_desc = ("**Sliding trailing windows**: one 7-day window per labelled day "
                 "([D-6, D] -> EE(D)), stride 1 day; windows overlap by 6 days, split stays "
                 "participant-level so test is out of pretraining.")
    out = Path(args.output_dir) / args.run_id / f"{args.backbone}_{args.pe}_seed{args.seed}"
    run_energy_tasks(model, X, ee, pids, n_sensors, tr, va, te,
                     args.pool, args.energy_threshold, args.seed, out, mode_desc, vars(args),
                     season_pool=None if args.season_pool == "same" else args.season_pool,
                     ee_win=data.get("ee_win"))
    print(f"total time = {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
