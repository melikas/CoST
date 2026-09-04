"""Does the TRAINED encoder gain more from opening the readout than an untrained one?

Five independent measurements now say pretraining contributes nothing:

    HRD, frozen                        -0.0011
    GLOBEM, frozen                     -0.0019
    GLOBEM, fine-tuned vs supervised   -0.0019
    GLOBEM, frozen, paired             +0.0000   49/96   p=0.92
    GLOBEM, combined rungs             -0.0011   43/96   p=0.36

Every one of them was taken through the SHIPPED readout: the trend half mean-pooled over
672 (HRD) or 112 (GLOBEM) timesteps, the seasonal half truncated to five harmonic lines.
And separately, on HRD, opening the readout's time resolution was worth +0.0328 -- but that
was measured on RANDOM-INIT encoders only, because no trained ones were on disk at the time.

So the question was never asked. It has a reason to come out either way. A random encoder's
feature sequence has no structure in time, so keeping time resolution gives it nothing but
columns the penalised probe pays for. A trained one, if the contrastive objective organised
anything at all, organised it ACROSS TIME -- and the mean pool collapses exactly that axis.
If that is right, DSSL gains more than its control does, the gap opens for the first time in
this project, and the fix is a readout that costs no retraining. If both rise together, the
last untested route is closed.

`DSSL - Random-init` at each readout is the whole output; the arms differ in the weights and
in nothing else, since random_init_model mirrors the layout of the encoder it controls for.

    SCRIPT=readout_interaction.py NEED_ENC=1 sbatch --array=0-23%24 \
      scripts/stability_gate.sh results_globem/2240054
    python readout_interaction.py --aggregate results_globem/2240054
"""
import argparse
import glob
import json
from math import comb
from pathlib import Path

import numpy as np

NAME = "readout_interaction"


def arms(parts):
    """{readout: features}. 'PRODUCTION' is the shipped one and the reference for the rest."""
    from readout_sweep import SEGS, production_readout
    out = {"PRODUCTION": production_readout(parts)}
    for s in SEGS:
        k = f"seg {s:2d}"
        out[f"both {k}"] = np.concatenate([parts[f"trend {k}"], parts[f"season {k}"]], axis=1)
        out[f"trend {k} + spec"] = np.concatenate([parts[f"trend {k}"], parts["season spec"]],
                                                  axis=1)
    return out


def window_auc(feat, ctx):
    """AUROC at the WINDOW unit -- the unit the GLOBEM ladder is scored at, so these numbers
    sit beside that table rather than beside a differently-defined one."""
    from sklearn.metrics import roc_auc_score
    from tasks._eval_protocols import fit_window_probe, window_rows
    clf = fit_window_probe(feat, ctx.pids, ctx.y, ctx.train_mask, ctx.val_mask, ctx.seed)
    Xte, yte, _ = window_rows(feat, ctx.pids, ctx.y, ctx.test_mask)
    if len(set(yte)) < 2:
        return float("nan")
    return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))


def sign_p(d):
    n, k = len(d), int((np.asarray(d) > 0).sum())
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n)


def aggregate(run_dir):
    files = sorted(glob.glob(str(Path(run_dir) / "*" / "**" / f"{NAME}.json"), recursive=True))
    rows = [json.loads(Path(f).read_text()) for f in files]
    if not rows:
        raise SystemExit(f"no {NAME}.json under {run_dir}")
    names = [k for k in rows[0]["DSSL"]]
    print()
    print(f"  {len(rows)} variants, window-unit AUROC, the run's own probe and splits")
    print()
    print(f"  {'readout':22s} {'DSSL':>8s} {'Rand-init':>10s} {'diff':>9s} {'wins':>8s}"
          f" {'p':>8s}")
    base = None
    for n in names:
        a = np.array([r["DSSL"][n] for r in rows], float)
        b = np.array([r["Random-init"][n] for r in rows], float)
        d = a - b
        ok = ~np.isnan(d)
        if base is None:
            base = d[ok].mean()
        print(f"  {n:22s} {np.nanmean(a):8.4f} {np.nanmean(b):10.4f} {d[ok].mean():+9.4f}"
              f" {int((d[ok] > 0).sum()):4d}/{int(ok.sum())} {sign_p(d[ok]):8.4f}")
    print()
    print("  The column that matters is `diff`. If it GROWS as the readout opens, the")
    print("  objective organised something across time that the mean pool was discarding.")
    print(f"  At the shipped readout it is {base:+.4f}.")
    out = {"n_variants": len(rows),
           "diff": {n: float(np.nanmean([r["DSSL"][n] - r["Random-init"][n] for r in rows]))
                    for n in names}}
    best = max(names, key=lambda n: out["diff"][n])
    out["best_readout"], out["verdict"] = best, (
        f"OPEN THE READOUT -- {best} gives DSSL {out['diff'][best]:+.4f} over its control "
        f"against {base:+.4f} shipped; the objective did organise something in time."
        if out["diff"][best] - base > 0.02 else
        "REJECT -- opening the readout moves DSSL and its untrained control together, so "
        "there is nothing learned for a better readout to expose.")
    print()
    print(f"  VERDICT: {out['verdict']}")
    p = Path(run_dir) / f"{NAME}_summary.json"
    p.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"[saved] {p}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant-dir")
    ap.add_argument("--aggregate", metavar="RUN_DIR")
    ap.add_argument("--cache-dir", default=None)
    a = ap.parse_args()
    if a.aggregate:
        aggregate(a.aggregate)
        return
    if not a.variant_dir:
        ap.error("one of --variant-dir or --aggregate is required")

    from readout_sweep import readout_parts
    from tasks._experiment_common import load_context, out_dir, random_init_model, save

    ctx = load_context(a.variant_dir, a.cache_dir, gpu=-1)
    if not getattr(ctx, "trained", True):
        raise SystemExit(f"{a.variant_dir} has no trained encoder -- this experiment is a "
                         "contrast between a trained encoder and its untrained control, and "
                         "without the first it would silently compare two random ones")
    sp = ctx.cfg.get("season_pool") or "spec"
    res = {"variant": ctx.tag, "seed": ctx.seed}
    for tag, model in (("DSSL", ctx.model), ("Random-init", random_init_model(ctx))):
        parts = readout_parts(model, ctx.X, sp)
        res[tag] = {}
        for name, F in arms(parts).items():
            res[tag][name] = window_auc(F, ctx)
        print(f"  {tag}: " + "  ".join(f"{k}={v:.3f}" for k, v in res[tag].items()),
              flush=True)
    save(out_dir(ctx, NAME), NAME, res)


if __name__ == "__main__":
    main()
