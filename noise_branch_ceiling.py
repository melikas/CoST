"""If the decomposition got its third branch, how much would be left for it to represent?

The project's own hypothesis names three parts and the model implements two:

    signal = trend + seasonal + noise
             V^T   + V^S      +  nothing

The third is discarded, and the residual is where the signal measurably is -- probed alone
it scores 0.6862 against 0.6228 for trend and seasonal together, on 24 seeds. So a third
contrastive branch V^N completes the design rather than abandoning it.

Before building it: a contrastive branch can only represent what survives its own positive
pair. Its ceiling is the predictive content of that invariant subspace, and the residual is
high-frequency by construction while the shipped augmentation set includes a smoother. If
`smooth` erases the residual, V^N trained that way has nothing to hold and the branch needs
different augmentations -- which is worth knowing before a GPU sweep, not after.

The augmentations come from cost.py's own PretrainDataset, not from a reimplementation here:
measuring a lookalike of the augmentation would answer about the lookalike.

The invariant part is estimated by averaging K independent views of each window. For jitter
and shift -- both additive and zero-mean -- that estimator returns the input, which is the
correct answer: neither can force a model to discard anything. Smoothing is the one that
bites, and the average over its random widths is what a contrastive objective can rely on.

    python noise_branch_ceiling.py --npz hrd_2224103.npz
"""
import argparse
import json

import numpy as np

from tasks.decompose import decompose


def invariant(X, cfg, n_sensors, draws=32, seed=0):
    """What `transform` leaves an objective able to rely on, per window.

    Two of the three augmentations are handled by averaging views and one analytically.

    `shift` adds a constant per channel, so the functions it cannot change are exactly the
    functions of the mean-removed signal -- that is its invariant subspace, known without
    sampling. Estimating it by averaging instead leaves Monte Carlo noise of order
    shift_sigma / sqrt(draws), measured at 0.036 with sigma=0.5 and 64 draws, which would
    depress the ceiling for a reason that has nothing to do with the augmentation.

    `jitter` and `smooth` have no such closed form here (smooth draws a random width each
    time), so they are averaged. Jitter is zero-mean and washes out; smoothing is the one
    that bites, and its average over widths is what survives.
    """
    import torch
    from cost import PretrainDataset
    torch.manual_seed(seed)
    import random as _r
    _r.seed(seed)
    ds = PretrainDataset(
        np.asarray(X, dtype=np.float32),
        jitter_sigma=cfg.get("jitter_sigma", 0.1), shift_sigma=cfg.get("shift_sigma", 0.5),
        n_sensors=n_sensors, bins_per_day=24 * 60 // cfg["bin_minutes"],
        smooth_bins=cfg.get("smooth_bins", 0))
    out = np.zeros_like(np.asarray(X, dtype=np.float32))
    with torch.no_grad():
        for i in range(len(X)):
            x = torch.from_numpy(np.asarray(X[i], dtype=np.float32))
            acc = torch.zeros_like(x)
            for _ in range(draws):
                acc += ds.transform(x)
            out[i] = (acc / draws).numpy()
    # shift, exactly: drop the per-channel DC it is free to move
    return out - out.mean(axis=1, keepdims=True)


def per_participant_mean(F, pids):
    """Each window replaced by its participant's average -- the invariant a participant-level
    positive pair imposes, since two windows of one person must map to the same place."""
    out = np.empty_like(F)
    for p in np.unique(pids):
        m = pids == p
        out[m] = F[m].mean(axis=0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--draws", type=int, default=16)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--out", default="noise_branch_ceiling.json")
    a = ap.parse_args()

    from local_context import local_context
    from random_init_audit import _probe_auc, raw_projection

    seeds = [int(s) for s in np.load(a.npz, allow_pickle=True)["seeds"]]
    ctx0 = local_context(a.npz, seeds[0])
    ns, bpd = ctx0.n_sensors, ctx0.bins_per_day
    S = np.nan_to_num(np.asarray(ctx0.X[:, :, :ns], dtype=float), nan=0.0)
    _, _, R = decompose(ctx0.X, bpd, ns)

    print(f"[ceiling] {len(S)} windows, {ns} channels, {a.draws} views each", flush=True)
    Sa = invariant(S, ctx0.cfg, ns, a.draws)
    Ra = invariant(R, ctx0.cfg, ns, a.draws)
    print(f"[ceiling] residual rms {np.sqrt((R ** 2).mean()):.4f}"
          f" -> after augmentation {np.sqrt((Ra ** 2).mean()):.4f}"
          f"  ({100 * np.sqrt((Ra ** 2).mean()) / np.sqrt((R ** 2).mean()):.0f}% kept)",
          flush=True)

    variants = {
        "full signal": S,
        "full signal, window-invariant": Sa,
        "residual": R,
        "residual, window-invariant": Ra,
        "residual, participant-invariant": per_participant_mean(R, ctx0.pids),
    }
    rows = []
    for i, sd in enumerate(seeds):
        ctx = local_context(a.npz, sd)
        r = {"seed": sd}
        for name, F in variants.items():
            # every arm through the same projection at the same width: the comparison is
            # about what survives, not about how many columns it survives into
            r[name] = _probe_auc(raw_projection(F, ns, a.width, sd), ctx)
        rows.append(r)
        json.dump(rows, open(a.out, "w"), indent=1)
        print(f"[{i + 1:2d}/{len(seeds)}] seed {sd:3d}  "
              + "  ".join(f"{k.split(',')[0][:9]}={v:.3f}" for k, v in r.items()
                          if k != "seed"), flush=True)

    names = [k for k in rows[0] if k != "seed"]
    print()
    print(f"  {len(rows)} seeds, HRD, the run's own probe, splits and participants")
    print()
    print(f"  {'what a branch could hold':34s} {'AUC':>8s}")
    for n in names:
        print(f"  {n:34s} {np.nanmean([r[n] for r in rows]):8.4f}")
    keep = np.nanmean([r["residual, window-invariant"] for r in rows])
    base = np.nanmean([r["residual"] for r in rows])
    print()
    print(f"  The shipped augmentations leave a residual branch {keep:.4f} of the {base:.4f}")
    print(f"  the residual carries unaugmented -- a loss of {base - keep:+.4f}.")
    print("  BUILD IT" if keep > 0.62 else "  REJECT -- nothing left for the branch to hold")


if __name__ == "__main__":
    main()
