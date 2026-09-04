"""Two properties every candidate positive pair must have, and no implemented one has both.

A contrastive objective is defined by what it calls a positive. That single choice fixes two
things at once, and they pull against each other:

  DIFFICULTY  -- can the task be solved before training? pretext_difficulty.py measures it.
                 With the window pair, MoCo top-1 on an untrained encoder is 1.000: the loss
                 starts at zero and the gradient teaches nothing. That is why the frozen DSSL
                 representation has never separated from its own random-init control, across
                 five protocols and two datasets.

  CEILING     -- what survives the pairing. A contrastive branch can only represent the
                 invariant subspace of its positive pair, so the predictive content of that
                 subspace is the most the branch can ever carry.

Measured so far, and the two implemented options sit at opposite corners:

    window pair        trivial at init (top-1 1.000)      ceiling 0.7202
    participant pair   hard                               ceiling 0.5856

The easy task keeps the signal and teaches nothing; the hard task teaches something and
destroys the signal. Nothing in this project has looked between them, and `day-disjoint` --
already implemented -- sits exactly there: two views of ONE window rebuilt from disjoint
halves of its own days, so they share no raw day but do share the person, the week and the
time of day.

This script measures the CEILING axis for all three, through the same probe, splits and
held-out participants as every other number here. pretext_difficulty.py measures the other.
A pair that is hard AND keeps its ceiling above the window pair's is the sweep worth running;
if none exists, the contrastive framing is bounded by these two numbers and that is worth
knowing before more GPU is spent.

    python positive_pair_ceiling.py --npz hrd_2224103.npz
"""
import argparse
import json

import numpy as np


def invariant_window(X, cfg, n_sensors, draws, seed):
    """E[transform(x)] -- what survives the augmentations, with shift handled exactly.

    `shift` adds a constant per channel, so the functions it cannot change are exactly the
    functions of the mean-removed signal. Estimating that by averaging leaves Monte Carlo
    noise of order shift_sigma/sqrt(draws) -- 0.036 at sigma=0.5 with 64 draws -- which would
    depress the ceiling for a reason that has nothing to do with the augmentation.
    """
    import torch
    from cost import PretrainDataset
    ds = _dataset(X, cfg, n_sensors, "window", seed)
    out = np.zeros_like(np.asarray(X, dtype=np.float32))
    with torch.no_grad():
        for i in range(len(X)):
            x = torch.from_numpy(np.asarray(X[i], dtype=np.float32))
            acc = torch.zeros_like(x)
            for _ in range(draws):
                acc += ds.transform(x)
            out[i] = (acc / draws).numpy()
    return out - out.mean(axis=1, keepdims=True)


def invariant_day_disjoint(X, cfg, n_sensors, draws, seed):
    """The average of the two day-disjoint views, over `draws` independent day splits.

    The dataset's own `_day_views` builds them, so this measures the pairing that ships
    rather than a lookalike of it. Both views are averaged because the objective can only
    rely on what they agree about.
    """
    import torch
    ds = _dataset(X, cfg, n_sensors, "day-disjoint", seed)
    out = np.zeros_like(np.asarray(X, dtype=np.float32))
    with torch.no_grad():
        for i in range(len(X)):
            acc = None
            for _ in range(draws):
                a, b = ds._day_views(i)
                v = (a + b) / 2
                acc = v if acc is None else acc + v
            out[i] = (acc / draws).numpy()
    return out - out.mean(axis=1, keepdims=True)


def invariant_participant(X, pids):
    """Each window replaced by its participant's mean: two windows of one person must land
    together, so nothing that separates them can survive."""
    X = np.asarray(X, dtype=np.float32)
    out = np.empty_like(X)
    for p in np.unique(pids):
        m = pids == p
        out[m] = X[m].mean(axis=0)
    return out


def _dataset(X, cfg, n_sensors, positive, seed):
    import random
    import torch
    from cost import PretrainDataset
    torch.manual_seed(seed)
    random.seed(seed)
    # A TENSOR, not an array. PretrainDataset calls self.data.size(0) in __len__ and
    # x.size(-1) inside _offset, so a numpy array reaches _day_views and fails on `.size`
    # being an int attribute -- while `transform`, which the caller hands a tensor
    # explicitly, works fine and hides it.
    return PretrainDataset(
        torch.from_numpy(np.asarray(X, dtype=np.float32)),
        jitter_sigma=cfg.get("jitter_sigma", 0.1), shift_sigma=cfg.get("shift_sigma", 0.5),
        n_sensors=n_sensors, bins_per_day=24 * 60 // cfg["bin_minutes"],
        smooth_bins=cfg.get("smooth_bins", 0), positive=positive)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--draws", type=int, default=16)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--out", default="positive_pair_ceiling.json")
    a = ap.parse_args()

    from local_context import local_context
    from random_init_audit import _probe_auc, raw_projection

    seeds = [int(s) for s in np.load(a.npz, allow_pickle=True)["seeds"]]
    ctx = local_context(a.npz, seeds[0])
    ns = ctx.n_sensors
    S = np.nan_to_num(np.asarray(ctx.X[:, :, :ns], dtype=float), nan=0.0).astype(np.float32)

    print(f"[pair] {len(S)} windows, {ns} channels, {a.draws} draws each", flush=True)
    variants = {
        "no pairing (upper bound)": S - S.mean(axis=1, keepdims=True),
        "window pair": invariant_window(S, ctx.cfg, ns, a.draws, 0),
        "day-disjoint pair": invariant_day_disjoint(S, ctx.cfg, ns, a.draws, 0),
        "participant pair": invariant_participant(S, np.asarray(ctx.pids)),
    }
    for k, v in variants.items():
        print(f"  {k:26s} rms {np.sqrt((v ** 2).mean()):.4f}", flush=True)

    rows = []
    for i, sd in enumerate(seeds):
        c = local_context(a.npz, sd)
        r = {"seed": sd}
        for name, F in variants.items():
            r[name] = _probe_auc(raw_projection(F, ns, a.width, sd), c)
        rows.append(r)
        json.dump(rows, open(a.out, "w"), indent=1)
        print(f"[{i + 1:2d}/{len(seeds)}] seed {sd:3d}  "
              + "  ".join(f"{k.split()[0][:9]}={v:.3f}" for k, v in r.items() if k != "seed"),
              flush=True)

    names = [k for k in rows[0] if k != "seed"]
    base = np.array([r["no pairing (upper bound)"] for r in rows], float)
    print()
    print(f"  {len(rows)} seeds, HRD, the run's own probe, splits and participants")
    print()
    print(f"  {'positive pair':26s} {'ceiling':>9s} {'vs no pairing':>14s}")
    for n in names:
        v = np.array([r[n] for r in rows], float)
        print(f"  {n:26s} {np.nanmean(v):9.4f} {np.nanmean(v - base):+14.4f}")
    print()
    print("  Read this beside pretext_difficulty.py. A pair is worth pretraining on only if")
    print("  it is HARD at init and keeps its ceiling; the two implemented options each have")
    print("  one of those properties and not the other.")


if __name__ == "__main__":
    main()
