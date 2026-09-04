"""Is the depression signal a SECOND-ORDER statistic, and does the architecture omit it?

Every measurement in this project points the same way and none of them was designed to.
The residual carries the signal (0.6862) while trend+seasonal does not (0.6228). A forest
on the raw window beats a linear probe on it by 0.0558, so whatever carries the signal is
not a linear functional of the bins. Handcrafted mean/std scores 0.4901 -- but those are
whole-window aggregates. On GLOBEM, `Structured rhythm` -- the one arm with per-DAY
dispersion in it -- came first among the frozen arms.

The clinical literature says the same thing in words: depression shows up in passive sensing
as IRREGULARITY of routine, not as a shifted or flattened mean rhythm.

Irregularity is a second-order statistic. The encoder is a stack of linear filters (TCN),
a linear spectral layer (BandedFourierLayer) and a MEAN pool. Nothing in that chain can
compute a variance. If the signal is dispersion, the architecture cannot represent it and
no amount of pretraining will find it -- which would explain every flat ladder we have.

This script measures the ceiling of that route before any of it is built, using the same
probe, the same splits and the same held-out participants as the ladder itself. The arms:

  raw projection / raw window        what we already have, as the reference
  residual                           x minus daily trend minus daily harmonics
  day means only                     per-day FIRST-order stats -- the control that says
                                     whether it is dispersion or merely day resolution
  day dispersion (raw / residual)    per-day std, mean-abs-successive-difference and IQR,
                                     pooled across days by mean AND std
  dispersion + projection            does dispersion add to what we already read?

Local, no GPU, no encoder:
    python dispersion_ceiling.py --npz hrd_2224103.npz
"""
import argparse
import json

import numpy as np

from local_context import local_context
from tasks.decompose import decompose
from random_init_audit import _probe_auc, raw_projection


def day_features(A, bins_per_day, kinds=("level", "level_var", "within")):
    """Per-day statistics, split by ORDER and by TIMESCALE so the arms can disagree.

    level      mean of the daily means -- pure first order, essentially the window mean.
    level_var  spread of the daily means ACROSS days: does this person do the same amount
               every day? Second order at the day scale.
    within     spread WITHIN each day (std, bin-to-bin roughness, IQR), pooled across days
               by mean and by std. Second order at the bin scale, plus its own variability.

    None of `level_var` or `within` is reachable by the encoder as it stands: a TCN is a
    bank of linear filters, BandedFourierLayer is linear, and the readout is a MEAN pool.
    Nothing in that chain squares anything, so a variance cannot appear at the probe.
    """
    n, T, C = A.shape
    d = T // bins_per_day
    D = A[:, :d * bins_per_day].reshape(n, d, bins_per_day, C)
    daily = D.mean(axis=2)                                        # (n, d, C)
    out = []
    if "level" in kinds:
        out.append(daily.mean(axis=1))
    if "level_var" in kinds:
        out.append(daily.std(axis=1))
    if "within" in kinds:
        w = np.concatenate([D.std(axis=2),
                            np.abs(np.diff(D, axis=2)).mean(axis=2),
                            np.percentile(D, 75, axis=2) - np.percentile(D, 25, axis=2)],
                           axis=2)                                # (n, d, 3C)
        out += [w.mean(axis=1), w.std(axis=1)]
    return np.concatenate(out, axis=1).astype(np.float32)


def arms(ctx):
    bpd, ns = ctx.bins_per_day, ctx.n_sensors
    _, _, resid = decompose(ctx.X, bpd, ns)
    raw = np.nan_to_num(np.asarray(ctx.X[:, :, :ns], dtype=float), nan=0.0)
    proj = raw_projection(ctx.X, ns, 320, ctx.seed)
    within_res = day_features(resid, bpd, kinds=("within",))
    all_res = day_features(resid, bpd)
    return {
        "raw projection": proj,
        "raw window": raw.reshape(len(raw), -1).astype(np.float32),
        "residual window": resid.reshape(len(resid), -1).astype(np.float32),
        "day level (1st order)": day_features(raw, bpd, kinds=("level",)),
        "day-to-day level spread": day_features(raw, bpd, kinds=("level_var",)),
        "within-day spread (raw)": day_features(raw, bpd, kinds=("within",)),
        "within-day spread (resid)": within_res,
        "all day features (resid)": all_res,
        "day features + projection": np.concatenate([all_res, proj], axis=1),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default="dispersion_ceiling.json")
    a = ap.parse_args()

    seeds = [int(s) for s in np.load(a.npz, allow_pickle=True)["seeds"]]
    rows = []
    for i, sd in enumerate(seeds):
        ctx = local_context(a.npz, sd)
        r = {"seed": sd}
        for name, F in arms(ctx).items():
            r[name] = _probe_auc(F, ctx)
        rows.append(r)
        print(f"[{i + 1:2d}/{len(seeds)}] seed {sd:3d}  "
              + "  ".join(f"{k.split()[0]}={v:.3f}" for k, v in r.items() if k != "seed"),
              flush=True)
    json.dump(rows, open(a.out, "w"), indent=1)

    names = [k for k in rows[0] if k != "seed"]
    ref = "raw projection"
    print(f"\n  {len(rows)} seeds, HRD, participant-level AUROC, same probe and splits\n")
    print(f"  {'arm':28s} {'AUC':>7s} {'vs raw proj':>12s} {'wins':>6s}")
    for n in sorted(names, key=lambda n: -float(np.nanmean([r[n] for r in rows]))):
        v = np.array([r[n] for r in rows], float)
        d = v - np.array([r[ref] for r in rows], float)
        print(f"  {n:28s} {np.nanmean(v):7.4f} {np.nanmean(d):+12.4f} "
              f"{int(np.nansum(d > 0)):4d}/{len(rows)}")


if __name__ == "__main__":
    main()
