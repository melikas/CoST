"""The encoder-free half of the random-init audit -- seconds per seed instead of 45 minutes.

Three forward passes over 3890 windows of 672 steps are the entire cost of
`random_init_audit.py` on a laptop, and none of them is needed to separate the explanations
that matter:

  a leak                 permute the participant-to-label map. Tests the probe and the split,
                         not any particular feature map, so it runs on whichever arm is at
                         hand -- here the banded projection, which scores in the same range as
                         Random-init.
  generic random         a plain Gaussian projection of the RAW window, no encoder anywhere.
  features               If it matches Random-init, the number is about dimensionality.
  the banding            the same projection restricted to the circadian rFFT bands the
                         Fourier layer actually uses. If THIS matches Random-init and the
                         unrestricted one does not, the number is about the inductive bias.

Random-init's own AUC is not recomputed: every run already records it per seed in
`rq3_utility.csv`, and recomputing it is the part that costs 45 minutes. Point `--ladder-csv`
at those files to put all the arms in one table.

    python fast_audit.py --npz hrd_2166049.npz
    python fast_audit.py --npz hrd_2166049.npz --ladder-csv "results_hrd/2166049/*/RQ3/rq3_utility.csv"
"""
import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

from local_context import local_context, seeds
from random_init_audit import _probe_auc, raw_projection
from models.encoder import seasonal_band_edges


def readout_width(ctx):
    """The readout width from the config: trend, plus amplitude and phase at five harmonics."""
    d = int(ctx.cfg["repr_dims"]) // 2
    D = max(1, ctx.seq_len // int(ctx.bins_per_day))
    n_h = len([f for f in (1, D, 2 * D, 3 * D, 4 * D) if f < ctx.seq_len // 2 + 1])
    return d + 2 * n_h * d


def bands_of(ctx):
    """The rFFT ranges the seasonal layer would use for this config."""
    if ctx.cfg.get("seasonal_bands") == "harmonics":
        return seasonal_band_edges(ctx.seq_len, ctx.bins_per_day)
    return [(0, (ctx.seq_len // 2) + 1)]


def permuted_null(feat, ctx, n_perm, rng):
    """AUC when the participant-to-label map is scrambled. A clean pipeline returns 0.5.

    Windows with no label stay unlabelled: the permutation is meant to destroy the association
    with the signal, not to invent labels for windows that never had one.
    """
    y = np.asarray(ctx.y)
    labelled = sorted({p for p in np.unique(ctx.pids) if (y[ctx.pids == p] >= 0).any()})
    lab = [int(y[(ctx.pids == p) & (y >= 0)][0]) for p in labelled]
    out = []
    for _ in range(n_perm):
        m = dict(zip(labelled, rng.permutation(lab)))
        yp = np.where(y >= 0, np.array([m.get(p, -1) for p in ctx.pids]), -1)
        out.append(_probe_auc(feat, ctx, y=yp))
    return np.array(out, float)


def ladder_from_csv(pattern):
    """Per-seed ladder AUCs a run already wrote, so the slow arms need not be recomputed."""
    per = {}
    for f in sorted(glob.glob(pattern)):
        seed = int(f.split("seed")[1].split("\\")[0].split("/")[0])
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if r.get("role") == "ladder":
                per.setdefault(seed, {})[r["representation"]] = float(r["auc"])
    return per


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--ladder-csv", default=None,
                    help="glob for the run's rq3_utility.csv files, to show DSSL and "
                         "Random-init beside the projection arms")
    ap.add_argument("--n-perm", type=int, default=20)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ladder = ladder_from_csv(a.ladder_csv) if a.ladder_csv else {}
    rows = []
    for sd in seeds(a.npz):
        ctx = local_context(a.npz, sd)
        rng = np.random.default_rng(sd + 7919)
        w, bands = readout_width(ctx), bands_of(ctx)
        raw = raw_projection(ctx.X, ctx.n_sensors, w, sd)
        band = raw_projection(ctx.X, ctx.n_sensors, w, sd, bands=bands)
        nul = permuted_null(band, ctx, a.n_perm, rng)
        tr = set(np.unique(ctx.pids[ctx.train_mask]))
        te = set(np.unique(ctx.pids[ctx.test_mask]))
        r = {"seed": sd, "width": w, "bands": [list(map(int, b)) for b in bands],
             "overlap_train_test": len(tr & te),
             "auc/Raw random projection": _probe_auc(raw, ctx),
             "auc/Banded random projection": _probe_auc(band, ctx),
             "auc/permuted null": float(np.nanmean(nul)),
             "auc/permuted null max": float(np.nanmax(nul))}
        r.update({f"auc/{k}": v for k, v in ladder.get(sd, {}).items()})
        rows.append(r)
        print(f"  seed {sd:>4}  raw {r['auc/Raw random projection']:.4f}  "
              f"banded {r['auc/Banded random projection']:.4f}  "
              f"null {r['auc/permuted null']:.4f}"
              + (f"  | Random-init {ladder[sd]['Random-init']:.4f}"
                 f"  DSSL {ladder[sd]['DSSL (frozen)']:.4f}" if sd in ladder else ""))

    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2, default=float), encoding="utf-8")
        print(f"[saved] {a.out}")
    summarise(rows)


def summarise(rows):
    from tasks._stats import paired
    keys = sorted({k[4:] for r in rows for k in r if k.startswith("auc/")})
    col = lambda k: np.array([r.get(f"auc/{k}", np.nan) for r in rows], float)
    bad = sum(r["overlap_train_test"] for r in rows)
    print(f"\n[agg] {len(rows)} seeds | participant-disjoint split: "
          f"{'OK' if bad == 0 else f'*** {bad} OVERLAPS ***'}\n")
    for k in sorted(keys, key=lambda x: -np.nanmean(col(x))):
        print(f"  {k:34s} {np.nanmean(col(k)):.4f}")
    ref = "Random-init" if "Random-init" in keys else "Banded random projection"
    print("")
    for k in keys:
        if k == ref or "max" in k:
            continue
        r = paired(col(k), col(ref), 3.17)
        print(f"  {k[:30]:30s} - {ref[:22]:22s} {r['diff']:+.4f}  p={r['p']:.4f}  "
              f"wins {r['wins']}/{r['n']}")


if __name__ == "__main__":
    main()
