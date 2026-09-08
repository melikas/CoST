"""Build the GLOBEM window cache that data_loader.load_npz consumes.

    python scripts/build_globem_cache.py --csv datasets/GLOBEM_REDUCED.csv \
        --out globem_windows.npz

One-off ETL, not part of the pipeline: it turns the segment-level CSV into the same npz
shape the HRD cache already has, so `train.py --npz` and `eval.py --npz` need no GLOBEM
special case at all.

WHY THE MASKS EXIST, AND WHAT THEY ARE NOT. `data_loader` identifies the labelled cohort as
the union of the train/val/test masks, and never reads `y` to decide who is labelled. That
is not a stylistic choice -- `globem_preprocessing` assigns `label_by_pid.get(pid, 0)`, so a
participant with NO label is written into `y` as a 0, exactly as HRD writes its 38
unlabelled participants as 0. Folding on `y` would file every unlabelled participant as a
confirmed negative and corrupt the class balance. The masks carry the authoritative
`labeled_pids` set instead.

These masks are NOT the evaluation split. `cv.make_folds` does all splitting downstream,
from the master seed, and ignores them entirely. They are written for three seeds with
different internal partitions but an identical union, so that `data_loader`'s
seed-invariance assertion is actually exercised rather than vacuously satisfied.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_processing.globem_preprocessing import prepare_globem_dataset   # noqa: E402

SEG_PER_DAY = 4
MASK_SEEDS = (7, 13, 29)


def build_masks(pids, labelled, seeds=MASK_SEEDS, val_frac=0.2, test_frac=0.2):
    """train/val/test window masks per seed, whose UNION is exactly the labelled windows."""
    is_lab = np.isin(pids, sorted(labelled))
    out = {}
    for s in seeds:
        rng = np.random.default_rng(s)
        who = np.array(sorted(labelled))
        rng.shuffle(who)
        n_te = max(1, int(len(who) * test_frac))
        n_va = max(1, int(len(who) * val_frac))
        te, va = set(who[:n_te]), set(who[n_te:n_te + n_va])
        in_te = is_lab & np.isin(pids, sorted(te))
        in_va = is_lab & np.isin(pids, sorted(va))
        out[f"test_mask/{s}"] = in_te
        out[f"val_mask/{s}"] = in_va
        out[f"train_mask/{s}"] = is_lab & ~in_te & ~in_va
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="datasets/GLOBEM_REDUCED.csv")
    p.add_argument("--out", default="globem_windows.npz")
    p.add_argument("--window-days", type=int, default=28)
    p.add_argument("--stride-days", type=int, default=7)
    p.add_argument("--min-window-coverage", type=float, default=0.5)
    p.add_argument("--weekly-labels", action="store_true",
                   help="time-varying weekly labels instead of one endpoint per participant; "
                        "off by default so the cohort matches the archived GLOBEM ladder")
    a = p.parse_args(argv)

    print(f"[globem] reading {a.csv}")
    d = prepare_globem_dataset(a.csv, window_days=a.window_days, stride_days=a.stride_days,
                               min_window_coverage=a.min_window_coverage,
                               weekly_labels=a.weekly_labels)

    X, y, pids, wid = d["X"], d["y"], d["pids"], d["window_ids"]
    labelled = set(d["labeled_pids"])

    # Fail here rather than in eval: data_loader.participants() requires one label per
    # participant, and a pid carrying two would silently become whichever it saw first.
    for q in sorted(labelled):
        v = np.unique(y[pids == q])
        if len(v) != 1:
            raise SystemExit(f"participant {q} carries {len(v)} distinct labels {v}; "
                             "rebuild with --weekly-labels off, or aggregate first")

    masks = build_masks(pids, labelled)
    np.savez_compressed(
        a.out, X=X, y=y, pids=pids, window_ids=wid,
        n_sensors=int(d["n_sensors"]), bins_per_day=SEG_PER_DAY,
        sensor_cols=np.asarray(d["sensor_cols"]), **masks)

    lab = np.array([int(y[pids == q][0]) for q in sorted(labelled)])
    size = Path(a.out).stat().st_size / 1e6
    print(f"\n[globem] wrote {a.out}  ({size:.1f} MB)")
    print(f"  windows        {X.shape}  ({X.shape[1]} steps = {X.shape[1]//SEG_PER_DAY} days "
          f"at {SEG_PER_DAY}/day)")
    print(f"  participants   {len(set(pids.tolist()))} total, {len(labelled)} labelled")
    print(f"  prevalence     {lab.mean():.4f}  ({lab.sum()} pos / {len(lab)-lab.sum()} neg)")
    print(f"  channels       {int(d['n_sensors'])} sensors, {X.shape[-1]} total")
    print(f"  window ids     e.g. {wid[0]!r}")
    print(f"  mask seeds     {list(MASK_SEEDS)} (union identical by construction)")

    # Prove the cache is loadable by the real loader before anything is submitted.
    from data_loader import load_npz
    c = load_npz(a.out)
    ids, ylab = c.participants()
    assert set(ids) == labelled, "loader's cohort disagrees with labeled_pids"
    print(f"\n[check] data_loader.load_npz -> {len(ids)} labelled participants, "
          f"prevalence {ylab.mean():.4f}, seq_len {c.seq_len}, bins/day {c.bins_per_day}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
