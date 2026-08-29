"""Emotional-energy probes on a frozen CoST representation.

A library, with no entry point of its own: ``run_energy_tasks`` takes an already-fitted model
and returns / writes the metrics. Both callers -- ``train_hrd_energy.py`` (pretrain, then
probe) and ``train_hrd.py --energy-probe`` (reuse the depression run's encoder) -- go through
it, so the two paths probe identically.
"""

import json

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tasks._eval_protocols import (best_threshold, binary_metrics, make_probe,
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
    clf = make_probe("supervised", 1.0, seed)
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
                     season_pool=None, ee_win=None, extra_reprs=None, supervised_fn=None):
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
    reps["DSSL (SSL repr)"] = reprs

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
    # Top rung of the design's ladder: end-to-end supervised, which sees the labels the frozen
    # probes never do and is therefore a CEILING, not a competitor. Built by the caller --
    # this module deliberately does not import torch (see the header note), and only the
    # caller holds the backbone/dims/device. Binary tasks only: the rung is a BCE classifier,
    # so it has no meaning for the ordinal-regression tasks 2 and 4, and inventing a
    # regression variant would make the two ladders non-comparable.
    def _sup(ybin, tag):
        if supervised_fn is None:
            return []
        try:
            return [("Supervised (end-to-end)", supervised_fn(ybin, tr, va, te, tep, tag))]
        except Exception as e:      # never let the ceiling rung take the whole probe down
            print(f"[energy] supervised rung SKIPPED for {tag} (non-fatal): "
                  f"{type(e).__name__}: {e}")
            return []

    t1 = ([("Majority (chance)", majority_binary(y_hi, tr, te, tep))]
          + [(n, probe_binary(R, y_hi, tr, va, te, seed, tep)) for n, R in reps.items()]
          + _sup(y_hi, "task1_high_energy"))
    t2 = ([("Mean predictor", mean_regression(y_reg, tr, te))]
          + [(n, probe_regression(R, y_reg, tr, te, tep)) for n, R in reps.items()])
    t3 = ([("Majority (chance)", majority_binary(y_wp, tr, te, tep))]
          + [(n, probe_binary(R, y_wp, tr, va, te, seed, tep)) for n, R in reps.items()]
          + _sup(y_wp, "task3_within_person"))

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
        f"{tbl1}\n{tbl2}\n{tbl3}\n{tbl45}"
        f"_A representation is useful when DSSL beats both the majority/mean baseline "
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
