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
import sys
from pathlib import Path

# Run as `python analysis/<name>.py` from the repository root: the interpreter puts
# this file's own directory on sys.path, not the project root, so the shared modules
# would not import. scripts/ already does this; the pattern is the same.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from tasks.sign_test import sign_summary

NAME = "readout_interaction"


def arms(parts):
    """{readout: features}. 'PRODUCTION' is the shipped one and the reference for the rest."""
    from analysis.readout_sweep import SEGS, production_readout
    out = {"PRODUCTION": production_readout(parts)}
    for s in SEGS:
        k = f"seg {s:2d}"
        out[f"both {k}"] = np.concatenate([parts[f"trend {k}"], parts[f"season {k}"]], axis=1)
        out[f"trend {k} + spec"] = np.concatenate([parts[f"trend {k}"], parts["season spec"]],
                                                  axis=1)
    return out


def window_auc(feat, ctx, families=("supervised", "forest")):
    """AUROC at the WINDOW unit, through the SAME probe the ladder uses.

    Both halves of that sentence are load-bearing, and the second one was wrong first. This
    file defaulted to `fit_window_probe`'s own default, which is logistic only, while the
    RQ3 ladder is run with --probe-family supervised forest -- the family chosen on
    validation. The two disagree about the readout, and not slightly: seg2 came out +0.018
    to +0.028 in all four folds under a logistic probe and -0.009 under the ladder's, which
    is how this file came to recommend a readout that made the ladder worse.

    The probe that decides is the one whose numbers are reported, so that is the default
    here now. An instrument advising a protocol has to measure under that protocol.
    """
    from sklearn.metrics import roc_auc_score
    from tasks._eval_protocols import fit_window_probe, window_rows
    clf = fit_window_probe(feat, ctx.pids, ctx.y, ctx.train_mask, ctx.val_mask, ctx.seed,
                           families=tuple(families))
    Xte, yte, _ = window_rows(feat, ctx.pids, ctx.y, ctx.test_mask)
    if len(set(yte)) < 2:
        return float("nan")
    return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))


def aggregate(run_dir):
    files = sorted(glob.glob(str(Path(run_dir) / "*" / "**" / f"{NAME}.json"), recursive=True))
    rows = [json.loads(Path(f).read_text()) for f in files]
    if not rows:
        raise SystemExit(f"no {NAME}.json under {run_dir}")
    names = [k for k in rows[0]["DSSL"]]
    dims = rows[0].get("dims", {})
    print()
    print(f"  {len(rows)} variants, window-unit AUROC, the run's own probe and splits")
    print()
    print(f"  {'readout':22s} {'dim':>6s} {'DSSL':>8s} {'Rand-init':>10s} {'diff':>9s}"
          f" {'wins':>8s} {'p':>8s}")
    base = None
    for n in names:
        a = np.array([r["DSSL"][n] for r in rows], float)
        b = np.array([r["Random-init"][n] for r in rows], float)
        d = a - b
        ok = ~np.isnan(d)
        if base is None:
            base = d[ok].mean()
        k, m, pv = sign_summary(d)
        print(f"  {n:22s} {dims.get(n, 0):6d} {np.nanmean(a):8.4f} {np.nanmean(b):10.4f}"
              f" {d[ok].mean():+9.4f} {k:4d}/{m} {pv:8.4f}")
    print()
    print("  The column that matters is `diff`. If it GROWS as the readout opens, the")
    print("  objective organised something across time that the mean pool was discarding.")
    print(f"  At the shipped readout it is {base:+.4f}.")

    # A SECOND question, which the diff column cannot answer: is any readout better than the
    # shipped one in absolute terms, for both arms alike? Reading that off the means is the
    # mistake this file already warns about, so it is paired per variant here.
    print()
    print("  and separately -- each readout against the shipped one, paired per variant")
    print()
    print(f"  {'readout':22s} {'DSSL':>18s} {'Random-init':>18s}")
    print(f"  {'':22s} {'delta  wins    p':>18s} {'delta  wins    p':>18s}")
    for n in names:
        if n == "PRODUCTION":
            continue
        cells = []
        for arm in ("DSSL", "Random-init"):
            d = np.array([r[arm][n] - r[arm]["PRODUCTION"] for r in rows], float)
            d = d[~np.isnan(d)]
            k, m, pv = sign_summary(d)
            cells.append(f"{d.mean():+.4f} {k:3d}/{m} {pv:.4f}")
        print(f"  {n:22s} {cells[0]:>18s} {cells[1]:>18s}")
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
    ap.add_argument("--probe-family", nargs="+", default=["supervised", "forest"],
                    help="Probe families the validation split chooses between. Must match "
                         "what the RQ3 ladder is run with, or this measures a different "
                         "protocol than the one it advises (default: %(default)s).")
    ap.add_argument("--widths", metavar="VARIANT_DIR",
                    help="print each readout's width from a four-window forward pass and "
                         "stop. Cheap enough to answer 'is that gain just dimensions?' "
                         "without re-running the sweep.")
    a = ap.parse_args()
    if a.aggregate:
        aggregate(a.aggregate)
        return
    if a.widths:
        from analysis.readout_sweep import readout_parts
        from tasks._experiment_common import load_context, random_init_model
        ctx = load_context(a.widths, a.cache_dir, gpu=-1)
        m = ctx.model if getattr(ctx, "trained", True) else random_init_model(ctx)
        parts = readout_parts(m, ctx.X[:4], ctx.cfg.get("season_pool") or "spec", batch=4)
        for k, v in arms(parts).items():
            print(f"  {k:22s} {v.shape[1]:6d}")
        return
    if not a.variant_dir:
        ap.error("one of --variant-dir or --aggregate is required")

    from analysis.readout_sweep import readout_parts
    from tasks._experiment_common import load_context, out_dir, random_init_model, save

    ctx = load_context(a.variant_dir, a.cache_dir, gpu=-1)
    if not getattr(ctx, "trained", True):
        raise SystemExit(f"{a.variant_dir} has no trained encoder -- this experiment is a "
                         "contrast between a trained encoder and its untrained control, and "
                         "without the first it would silently compare two random ones")
    sp = ctx.cfg.get("season_pool") or "spec"
    res = {"variant": ctx.tag, "seed": ctx.seed,
           "probe_family": list(a.probe_family)}
    for tag, model in (("DSSL", ctx.model), ("Random-init", random_init_model(ctx))):
        parts = readout_parts(model, ctx.X, sp)
        res[tag] = {}
        built = arms(parts)
        # Widths belong in the record. A sweep of the raw projection over 16..1760 dims
        # spans 0.6865 to 0.7198 on HRD, so two readouts of different width are not
        # comparable on score alone, and an earlier version of this file compared them
        # anyway because it never wrote the numbers down.
        res["dims"] = {k: int(v.shape[1]) for k, v in built.items()}
        for name, F in built.items():
            res[tag][name] = window_auc(F, ctx, families=a.probe_family)
        print(f"  {tag}: " + "  ".join(f"{k}={v:.3f}" for k, v in res[tag].items()),
              flush=True)
    save(out_dir(ctx, NAME), NAME, res)


if __name__ == "__main__":
    main()
