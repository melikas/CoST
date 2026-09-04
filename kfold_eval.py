"""The HRD ladder under participant-level k-fold, and the margin that design requires.

Two things, and the order matters. --null runs first and fixes the bar; the ladder runs
after and is read against it. Running them the other way round, or quoting a margin derived
from the same numbers being tested, would be choosing the threshold after seeing the result.

WHY the design changes at all: the holdout evaluates 36 of 152 labelled participants per
seed and the 24 seeds overlap (rho = 0.46), so the corrected test needs 0.10 to 0.13 AUC
before it calls anything separable -- wider than the gap between the best and worst arm in
the project. That is a property of the evaluation, not of any model, and no architecture
can clear it.

WHAT --null does: permute the participant-to-label map, run the whole k-fold pipeline
unchanged, and record the difference between two arms. Under permutation there is no signal,
so the spread of that difference IS the noise the design leaves. The 95th percentile of its
absolute value is the margin a real difference has to exceed. Nothing about the arms or the
model enters it.

Every arm here is computable without a trained encoder, which is the point: the margin and
the HRD combination ladder -- the gap flagged as the most serious, since the combined rungs
had only ever run on GLOBEM -- both land before a GPU is needed.

    python kfold_eval.py --npz hrd_2224103.npz --null
    python kfold_eval.py --npz hrd_2224103.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np

from tasks.kfold import participant_folds, split_masks


def arms(ctx, width=512, cache_dir="_randinit_cache", npz=None):
    """{name: (n_windows, d)} -- everything measurable without pretraining."""
    from random_init_audit import raw_projection
    from structured_rhythm import structured_features
    from tasks.energy import handcrafted_features
    out = {}
    skip = raw_projection(ctx.X, ctx.n_sensors, width, 0)
    out["Raw skip only"] = skip
    out["Handcrafted (mean/std)"] = handcrafted_features(ctx.X, ctx.n_sensors)
    try:
        out["Structured rhythm"] = structured_features(ctx.X, ctx.bins_per_day, ctx.n_sensors)
    except Exception as e:                        # never lose the rest of the table
        print(f"[kfold] Structured rhythm SKIPPED: {type(e).__name__}: {e}")
    f = Path(cache_dir) / f"randinit_{Path(npz).stem}_{ctx.seed}.npy"
    if f.exists():
        out["Random-init"] = np.load(f)
    rhythm = [out[k] for k in ("Structured rhythm",) if k in out]
    if rhythm:
        def cat(*b):
            return np.concatenate([np.asarray(x, np.float32) for x in b], axis=1)
        out["rhythm + raw skip"] = cat(*rhythm, skip)
        if "Random-init" in out:
            out["Random-init + rhythm + raw skip"] = cat(out["Random-init"], *rhythm, skip)
    return out


def kfold_auc(feat, ctx, y, k, seed, families=("supervised", "forest")):
    """One participant-level AUC, pooled over all k held-out folds.

    Predictions from different folds come from different probes, so they are z-scored within
    each fold before pooling. Without that the pooled ranking mixes scores on incomparable
    scales and the AUC measures the offsets between folds as much as the signal.
    """
    from sklearn.metrics import roc_auc_score
    from tasks._eval_protocols import fit_persubject_probe, persubject_rows
    pids = np.asarray(ctx.pids)
    pid_label = {p: int(y[pids == p][0]) for p in np.unique(pids)}
    scores, labels = [], []
    for tr_p, te_p in participant_folds(pids, pid_label, k=k, seed=seed):
        tr, va, te = split_masks(pids, tr_p, te_p, y, seed=seed)
        clf = fit_persubject_probe(feat, pids, y, tr, va, seed, families=families)
        Xte, yte, _ = persubject_rows(feat, pids, y, te)
        p = clf.predict_proba(Xte)[:, 1]
        scores.append((p - p.mean()) / (p.std() + 1e-9))
        labels.append(yte)
    s, lab = np.concatenate(scores), np.concatenate(labels)
    return float(roc_auc_score(lab, s)) if len(set(lab)) > 1 else float("nan")


def run_null(ctx, A, a):
    """The margin this design requires, from labels that carry no signal."""
    # Two arms of the SAME shape, so the null describes the design and not a mismatch
    # between what the two can express. Their real difference is irrelevant here --
    # permuting the labels destroys it, which is the point.
    left = "rhythm + raw skip" if "rhythm + raw skip" in A else list(A)[0]
    right = "Raw skip only"
    rng = np.random.default_rng(0)
    pids = np.asarray(ctx.pids)
    uniq = np.array(sorted(set(pids)))
    pos = {p: i for i, p in enumerate(uniq)}
    base = np.array([int(ctx.y[pids == p][0]) for p in uniq])
    keep = base >= 0
    idx = np.array([pos[p] for p in pids])
    d = []
    for i in range(a.null_draws):
        perm = base.copy()
        perm[keep] = rng.permutation(base[keep])       # permuted at the PARTICIPANT level
        y = perm[idx]
        d.append(kfold_auc(A[left], ctx, y, a.k, 1000 + i)
                 - kfold_auc(A[right], ctx, y, a.k, 1000 + i))
        if (i + 1) % 10 == 0:
            v = np.abs(np.array(d, float))
            print(f"  [{i + 1:3d}/{a.null_draws}] 95th pct of |delta| so far "
                  f"{np.nanpercentile(v, 95):.4f}", flush=True)
    v = np.abs(np.array(d, float))
    margin = float(np.nanpercentile(v, 95))
    print()
    print(f"  {a.null_draws} participant-level label permutations, k={a.k}")
    print(f"  contrast: {left}  minus  {right}")
    print(f"  null difference: mean {np.nanmean(d):+.4f}, sd {np.nanstd(d):.4f}")
    print(f"  REQUIRED MARGIN (95th pct of |delta|): {margin:.4f}")
    print("  the holdout design this replaces requires 0.10 - 0.13")
    json.dump({"k": a.k, "draws": a.null_draws, "margin": margin,
               "null_sd": float(np.nanstd(d)), "contrast": [left, right]},
              open(a.out or "kfold_null.json", "w"), indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5,
                    help="independent k-fold partitions (default: %(default)s)")
    ap.add_argument("--null", action="store_true",
                    help="permute the participant labels and report the margin this design "
                         "requires. Run this BEFORE the ladder and write the number down.")
    ap.add_argument("--null-draws", type=int, default=200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from local_context import local_context
    seeds = [int(s) for s in np.load(a.npz, allow_pickle=True)["seeds"]]
    ctx = local_context(a.npz, seeds[0])
    A = arms(ctx, npz=a.npz)
    print(f"[kfold] {len(ctx.X):,} windows, {len(np.unique(ctx.pids))} participants, "
          f"k={a.k} | arms: {', '.join(A)}", flush=True)

    if a.null:
        run_null(ctx, A, a)
        return

    rows = []
    for rep in range(a.repeats):
        r = {"repeat": rep}
        for name, F in A.items():
            r[name] = kfold_auc(F, ctx, ctx.y, a.k, rep)
        rows.append(r)
        json.dump(rows, open(a.out or "kfold_eval.json", "w"), indent=1)
        print(f"[{rep + 1}/{a.repeats}] "
              + "  ".join(f"{k[:14]}={v:.3f}" for k, v in r.items() if k != "repeat"),
              flush=True)
    print()
    print(f"  {a.repeats} k-fold partitions, k={a.k}, HRD, every labelled participant tested")
    print()
    print(f"  {'arm':34s} {'AUC':>8s} {'sd':>7s}")
    for n in sorted(A, key=lambda n: -np.nanmean([r[n] for r in rows])):
        v = np.array([r[n] for r in rows], float)
        print(f"  {n:34s} {np.nanmean(v):8.4f} {np.nanstd(v):7.4f}")


if __name__ == "__main__":
    main()
