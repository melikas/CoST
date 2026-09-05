"""Does day-to-day PHASE CONCENTRATION add anything the current readout does not already have?

The readout emits, per chronobiological harmonic, the amplitude and phase of the whole window:

    v = [ |Z_k| ; arg Z_k ]

which is what cosinor gives too, computed from a latent instead of from the signal. That is
why nothing so far has beaten cosinor, and why a random-init encoder matches a trained one:
both hand the probe the same marginal spectrum, and a linear probe is blind to the difference
between two bases of it.

The quantity below is not in that vector and cannot be built from it. Split the window into its
D days, take each day's own 24 h coefficient, and measure how tightly those daily phases agree:

    R_h = | (1/D) * sum_d  Z_h^(d) / |Z_h^(d)| |        in [0, 1]

R = 1 is a person whose rhythm peaks at the same hour every day; R = 0 is a phase that wanders.
Three properties make it the right thing to test:

  * it is exactly RQ2's construct -- within-person rhythm stability -- measured directly rather
    than inferred from a distance to a personal mean;
  * a LINEAR probe cannot recover it from [mean | sd] of (cos phi, sin phi), because it is a
    norm of a mean of unit vectors, not a linear function of them;
  * it is invariant to the basis rotation a random projection introduces, so a random-init
    encoder cannot reproduce a trained one's value by accident.

Cosinor does not have it either: cosinor pools all D days into ONE fit and the day-to-day
dispersion is exactly what that pooling discards.

This script decides whether the block is worth an architecture change, using the ENCODERS THAT
ALREADY EXIST. No training happens. Three questions, each with its own answer:

  Q1  is R new?            ridge from the current readout to R. High R^2 = redundant, stop.
  Q2  does it help RQ3?    participant AUC with and without the block, same probe, same split.
  Q3  does it help RQ1?    recovery of interdaily stability, the marker R is supposed to be.

The rule below is fixed HERE, before any of it has been run, so the verdict cannot be chosen
after seeing the numbers. The block earns an architecture change only if all three hold:

    Q1   median R^2(R | base) < 0.50        R is not already inside the readout
    Q2   mean d(AUC) > 0 for DSSL, and larger than the same delta on the random-init
         control                            the gain is not just extra columns
    Q3   mean d(R^2 on IS) > 0 for DSSL

Anything else and the design is rejected here, at ~1 CPU-hour per seed, instead of after a
GPU sweep.

Run (no GPU):
    sbatch --array=0-23%24 scripts/stability_gate.sh results_hrd/<run>   # one seed per task
    python rhythm_stability.py --aggregate results_hrd/<run>             # then the verdict
"""
import sys
from pathlib import Path

# Run as `python analysis/<name>.py` from the repository root: the interpreter puts
# this file's own directory on sys.path, not the project root, so the shared modules
# would not import. scripts/ already does this; the pattern is the same.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from pathlib import Path

import numpy as np

from tasks._stats import paired

# torch, scikit-learn and the task package are imported inside the functions that
# need them. --aggregate reads nothing but the per-seed JSON already on disk, and
# making it import torch to run a t-test meant it could not run on a login node.

ALPHAS = (0.01, 0.1, 1, 10, 100, 1000, 10000, 100000)


def daily_phase_concentration(model, X, bins_per_day, batch=128, harmonics=(1, 2, 3, 4)):
    """R_h per latent dimension, from the seasonal branch's own per-day phases.

    The window is reshaped to (days, bins_per_day) and each day is transformed on its own, so
    harmonic h of a DAY is h cycles per 24 h: h=1 is circadian, h=2 the 12 h harmonic, and so
    on. Taking the phase per day is the whole point -- the window-level FFT the readout uses
    averages those days together and cannot see whether they agreed.

    Days whose amplitude is numerically zero contribute no direction and are dropped from that
    dimension's mean rather than being given an arbitrary one.
    """
    import torch
    org = model.net.training
    model.net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).float().to(model.device)
            _, season = model.net(xb)                       # (b, T, d)
            b, T, d = season.shape
            D = T // int(bins_per_day)
            day = season[:, :D * int(bins_per_day)].reshape(b, D, int(bins_per_day), d)
            Z = torch.fft.rfft(day.float(), dim=2)          # (b, D, bins/2+1, d)
            cols = []
            for h in harmonics:
                if h >= Z.shape[2]:
                    continue
                z = Z[:, :, h, :]                           # (b, D, d) complex
                mag = z.abs()
                unit = torch.where(mag > 1e-12, z / mag.clamp(min=1e-12),
                                   torch.zeros_like(z))
                live = (mag > 1e-12).float().sum(1).clamp(min=1.0)   # days that voted
                cols.append((unit.sum(1).abs() / live))     # (b, d) in [0, 1]
            out.append(torch.cat(cols, dim=-1).cpu().numpy())
    model.net.train(org)
    return np.concatenate(out)


def oof_r2(F, Y, groups, n_splits=5):
    """Out-of-fold R^2 per column, participant-grouped, penalty chosen inside the fold."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pred = np.full_like(Y, np.nan, dtype=float)
    cv = GroupKFold(n_splits=int(min(n_splits, len(np.unique(groups)))))
    for tr, te in cv.split(F, Y[:, 0], groups):
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(F[tr], Y[tr])
        # RidgeCV returns (n,) for a single-column target and (n, c) otherwise; reshape so
        # both land in the (n, c) buffer instead of raising a broadcast error.
        pred[te] = np.asarray(m.predict(F[te])).reshape(len(te), -1)
    ok = np.isfinite(pred[:, 0])
    ss = ((Y[ok] - pred[ok]) ** 2).sum(0)
    tot = ((Y[ok] - Y[ok].mean(0)) ** 2).sum(0).clip(1e-12)
    return 1 - ss / tot


def probe_auc(feat, ctx):
    """Participant AUC on the run's own held-out participants, with the canonical probe."""
    from sklearn.metrics import roc_auc_score
    from tasks._eval_protocols import fit_persubject_probe, persubject_rows
    clf = fit_persubject_probe(feat, ctx.pids, ctx.y, ctx.train_mask, ctx.val_mask, ctx.seed)
    Xs, ys, _ = persubject_rows(feat, ctx.pids, ctx.y, ctx.test_mask)
    return float(roc_auc_score(ys, clf.predict_proba(Xs)[:, 1])) if len(set(ys)) > 1 else np.nan


def aggregate(run_dir):
    """Pool the per-seed JSONs into the one verdict, with the corrected resampled t-test.

    The 24 seeds are 24 DIFFERENT participant splits, not folds of one -- verified: 24 distinct
    test-participant sets. Their training sets therefore overlap heavily and the fold
    differences are positively correlated, which is precisely the case Nadeau & Bengio's
    correction exists for; a naive paired t here would be anti-conservative. The correction
    inflates the variance by (1/n + n_test/n_train), and `paired` parameterises that as
    1/(n_splits-1), so passing n_splits = 1 + n_train/n_test makes the two expressions equal.
    Both are participant counts, read from the runs themselves rather than assumed.
    """
    rows, ntest, ntrain = [], [], []
    for f in sorted(Path(run_dir).glob("*_seed*/RQ1/rhythm_stability.json")):
        rows.append(json.loads(f.read_text(encoding="utf-8")))
        m = json.loads((f.parent.parent / "metrics.json").read_text(encoding="utf-8"))
        ntest.append(m["n_test_participants"])
        ntrain.append(m["n_labeled_participants"] - m["n_test_participants"])
    if not rows:
        raise SystemExit(f"no rhythm_stability.json under {run_dir} -- run the array first")
    col = lambda k: np.array([r.get(k, np.nan) for r in rows], float)
    n_splits = 1.0 + float(np.mean(ntrain)) / float(np.mean(ntest))
    print(f"[agg] {len(rows)} seeds | {np.mean(ntest):.0f} test / {np.mean(ntrain):.0f} train "
          f"participants -> corrected t with n_splits={n_splits:.2f}")

    out = {"n_seeds": len(rows), "n_splits_equiv": n_splits}
    verdicts = {}
    for tag in ("DSSL", "Random-init"):
        q1 = col(f"{tag}/Q1_R2_of_R_from_base_median")
        q2 = paired(col(f"{tag}/Q2_auc_base_plus_R"), col(f"{tag}/Q2_auc_base"), n_splits)
        q3 = paired(col(f"{tag}/Q3_IS_R2_base_plus_R"), col(f"{tag}/Q3_IS_R2_base"), n_splits)
        out[tag] = {"Q1_median_R2": float(np.nanmean(q1)), "Q2_dAUC": q2, "Q3_dIS_R2": q3}
        verdicts[tag] = (q1, q2, q3)
        print("")
        print(f"  {tag}")
        print(f"    Q1  R2(R | base)        {np.nanmean(q1):+.4f}   "
              f"({'NEW -- pass' if np.nanmean(q1) < 0.50 else 'REDUNDANT -- fail'})")
        for name, r in (("Q2  d(AUC)      ", q2), ("Q3  d(IS R2)    ", q3)):
            print(f"    {name}    {r['diff']:+.4f}   p={r['p']:.4f} "
                  f"(naive {r['p_naive']:.4f})  wins {r['wins']}/{r['n']}  dz={r['dz']:+.2f}")

    q1d, q2d, q3d = verdicts["DSSL"]
    _, q2c, _ = verdicts["Random-init"]
    checks = {
        "Q1 R is not already in the readout": float(np.nanmean(q1d)) < 0.50,
        "Q2 the block raises the depression AUC": q2d["diff"] > 0,
        "Q2 it raises it MORE than on random-init": q2d["diff"] > q2c["diff"],
        "Q3 the block improves interdaily-stability recovery": q3d["diff"] > 0,
    }
    print("")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    ok = all(checks.values())
    out["checks"] = {k: bool(v) for k, v in checks.items()}
    out["verdict"] = ("BUILD IT -- R carries a construct the readout lacks and it helps where "
                      "it should; the block is worth training with a frozen backbone."
                      if ok else
                      "REJECT -- the block does not earn an architecture change. Rejected on "
                      "CPU, before a GPU sweep, on a rule fixed before the numbers existed.")
    print("")
    print(f"  VERDICT: {out['verdict']}")
    (Path(run_dir) / "rhythm_stability_summary.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"[saved] {Path(run_dir) / 'rhythm_stability_summary.json'}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant-dir")
    ap.add_argument("--aggregate", metavar="RUN_DIR",
                    help="pool the per-seed JSONs already written under this run into the "
                         "single verdict; reads no encoder and needs no dataset")
    ap.add_argument("--cache-dir", default=None)
    a = ap.parse_args()
    if a.aggregate:
        aggregate(a.aggregate)
        return
    if not a.variant_dir:
        ap.error("one of --variant-dir or --aggregate is required")

    import torch
    from model_build import encode_repr
    from tasks._experiment_common import (load_context, out_dir,
                                        random_init_model, save)
    from tasks.rhythm import _interdaily_stability

    ctx = load_context(a.variant_dir, a.cache_dir, gpu=-1)
    ctrl = random_init_model(ctx)
    print(f"[stab] {ctx.tag} seed={ctx.seed} | {len(ctx.X):,} windows | "
          f"{int(ctx.test_mask.sum())} test windows, "
          f"{len(np.unique(ctx.pids[ctx.test_mask]))} test participants")

    res = {"variant": ctx.tag, "seed": ctx.seed}
    for tag, model in (("DSSL", ctx.model), ("Random-init", ctrl)):
        with torch.no_grad():
            base = encode_repr(model, ctx.X, ctx.cfg)
        R = daily_phase_concentration(model, ctx.X, ctx.bins_per_day)
        both = np.hstack([base, R])
        print(f"  {tag}: base {base.shape[1]} dims, R {R.shape[1]} dims, "
              f"R range [{R.min():.3f}, {R.max():.3f}], mean {R.mean():.3f}")

        # ---- Q1: is R already inside the current readout? ----
        r2 = oof_r2(base, R, ctx.pids)
        res[f"{tag}/Q1_R2_of_R_from_base_median"] = float(np.median(r2))
        res[f"{tag}/Q1_R2_of_R_from_base_mean"] = float(np.mean(r2))

        # ---- Q2: does it move the depression AUC? ----
        auc_b, auc_r, auc_br = probe_auc(base, ctx), probe_auc(R, ctx), probe_auc(both, ctx)
        res[f"{tag}/Q2_auc_base"] = auc_b
        res[f"{tag}/Q2_auc_R_only"] = auc_r
        res[f"{tag}/Q2_auc_base_plus_R"] = auc_br

        # ---- Q3: does it recover interdaily stability better? ----
        # IS is computed from the RAW signal, so this asks whether R carries a construct the
        # base readout is missing rather than whether it re-encodes one it already has.
        IS = _interdaily_stability(ctx.X[:, :, :ctx.n_sensors], int(ctx.bins_per_day),
                                   per_channel=True)
        ok = np.isfinite(IS).all(1)
        is_b = oof_r2(base[ok], IS[ok], ctx.pids[ok])
        is_br = oof_r2(both[ok], IS[ok], ctx.pids[ok])
        res[f"{tag}/Q3_IS_R2_base"] = float(np.mean(is_b))
        res[f"{tag}/Q3_IS_R2_base_plus_R"] = float(np.mean(is_br))

        print(f"    Q1  R2 of R from the base readout      median {np.median(r2):.4f}  "
              f"(high = R is already there, block is redundant)")
        print(f"    Q2  participant AUC   base {auc_b:.4f} | R only {auc_r:.4f} | "
              f"base+R {auc_br:.4f}   ({auc_br - auc_b:+.4f})")
        print(f"    Q3  interdaily-stability R2   base {np.mean(is_b):.4f} -> "
              f"base+R {np.mean(is_br):.4f}   ({np.mean(is_br) - np.mean(is_b):+.4f})")

    save(out_dir(ctx, "rq1"), "rhythm_stability", res)


if __name__ == "__main__":
    main()
