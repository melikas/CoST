"""Is the raw random projection's score real, or an artifact of width and a flexible probe?

The claim it supports is the one everything else in this project rests on: a penalised
classifier on a random projection of the raw window reaches 0.7198, above every learned
representation and above the rhythm parameterisations. That is not a network with random
weights beating cosinor -- a Gaussian projection is close to a lossless compression
(Johnson-Lindenstrauss), so the arm is really "a classifier on the raw data", and the
comparison is against "a classifier on a 12-parameter sinusoidal summary of the same data".
Put that way it is unremarkable. But it has been checked for exactly one thing -- leakage,
via a permuted-label null at 0.4967, 0/24, p=0.0015 -- and two others could produce it:

  WIDTH      the projection is 512 columns and the rhythm blocks are 78. More columns with a
             validation-selected probe is not obviously an advantage, but the comparison was
             never dimension-matched, so it has to be.

  NOTHING    if the probe scores 0.72 on data whose temporal structure has been destroyed,
             it is not reading the signal at all and every number here is meaningless. This
             is the sharper of the two, and it is the one that would prove the objection
             right.

Three destructions, each removing something specific:

  shuffle time      permute the time axis per window: every marginal survives, all temporal
                    structure dies. A rhythm reader must collapse to chance.
  shuffle labels    the participant-to-label map is permuted. Already run elsewhere; repeated
                    here so the two nulls sit in one table.
  gaussian noise    same shape, no data at all. The floor.

    python analysis/projection_sanity.py --npz hrd_2224103.npz
"""
import sys
from pathlib import Path

# Run as `python analysis/<name>.py` from the repository root: the interpreter puts
# this file's own directory on sys.path, not the project root, so the shared modules
# would not import. scripts/ already does this; the pattern is the same.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

import numpy as np


def project(F, width, seed):
    """A Gaussian projection of an ALREADY FLAT feature matrix.

    raw_projection takes (N, T, C) windows and flattens them itself, so passing it a
    2-D block raises. This is the same map for a block that is already flat, and it is what
    lets the rhythm arm be compared at the projection's width instead of its own.
    """
    F = np.asarray(F, dtype=float)
    F = (F - F.mean(0)) / (F.std(0) + 1e-8)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1.0 / np.sqrt(width), (F.shape[1], int(width)))
    return (F @ W).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default="results/projection_sanity.json")
    a = ap.parse_args()

    from analysis.local_context import local_context
    from analysis.random_init_audit import _probe_auc, raw_projection
    from structured_rhythm import structured_features

    seeds = [int(s) for s in np.load(a.npz, allow_pickle=True)["seeds"]]
    ctx0 = local_context(a.npz, seeds[0])
    ns = ctx0.n_sensors
    S = np.nan_to_num(np.asarray(ctx0.X[:, :, :ns], dtype=float), nan=0.0).astype(np.float32)
    rhythm = structured_features(ctx0.X, ctx0.bins_per_day, ns)
    W = rhythm.shape[1]
    print(f"[sanity] {len(S)} windows | rhythm block is {W} columns", flush=True)

    rng = np.random.default_rng(0)
    # Time destroyed, marginals kept: the SAME values, reordered within each window. Any
    # reader of rhythm, trend or dispersion must lose everything; a reader of per-channel
    # marginals keeps everything.
    idx = np.argsort(rng.random((len(S), S.shape[1])), axis=1)
    S_shuf = np.take_along_axis(S, idx[:, :, None].repeat(ns, axis=2), axis=1)
    noise = rng.normal(size=(len(S), 512)).astype(np.float32)

    rows = []
    for i, sd in enumerate(seeds):
        ctx = local_context(a.npz, sd)
        r = {"seed": sd}
        # dimension-matched both ways, so width cannot be what separates them
        r[f"raw projection ({W}d)"] = _probe_auc(raw_projection(S, ns, W, sd), ctx)
        r["raw projection (512d)"] = _probe_auc(raw_projection(S, ns, 512, sd), ctx)
        r[f"structured rhythm ({W}d)"] = _probe_auc(rhythm, ctx)
        r["structured rhythm (512d)"] = _probe_auc(project(rhythm, 512, sd), ctx)
        # controls
        r["CONTROL time-shuffled raw (512d)"] = _probe_auc(
            raw_projection(S_shuf, ns, 512, sd), ctx)
        r["CONTROL gaussian noise (512d)"] = _probe_auc(noise, ctx)
        y = np.asarray(ctx.y).copy()
        pids = np.asarray(ctx.pids)
        uniq = np.array(sorted(set(pids)))
        lab = np.array([int(y[pids == p][0]) for p in uniq])
        keep = lab >= 0
        perm = lab.copy()
        perm[keep] = np.random.default_rng(1000 + sd).permutation(lab[keep])
        pos = {p: j for j, p in enumerate(uniq)}
        r["CONTROL permuted labels (512d)"] = _probe_auc(
            raw_projection(S, ns, 512, sd), ctx, y=perm[[pos[p] for p in pids]])
        rows.append(r)
        json.dump(rows, open(a.out, "w"), indent=1)
        print(f"[{i + 1:2d}/{len(seeds)}] seed {sd:3d}  "
              + "  ".join(f"{k[:14]}={v:.3f}" for k, v in r.items() if k != "seed"),
              flush=True)

    names = [k for k in rows[0] if k != "seed"]
    print()
    print(f"  {len(rows)} seeds, HRD, the run's own probe, splits and participants")
    print()
    print(f"  {'arm':36s} {'AUC':>8s}")
    for n in names:
        print(f"  {n:36s} {np.nanmean([r[n] for r in rows]):8.4f}")
    print()
    ctrl = max(np.nanmean([r[n] for r in rows]) for n in names if n.startswith("CONTROL"))
    print("  Every CONTROL must sit at chance. If one does not, the probe is reading")
    print(f"  something other than the signal and no number here means anything.")
    print(f"  worst control: {ctrl:.4f}"
          + ("   -- OK" if ctrl < 0.58 else "   -- PROBLEM, stop and investigate"))


if __name__ == "__main__":
    main()
