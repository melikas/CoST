"""Dump a run's windows and splits to one npz, so evaluation can leave the cluster.

Parsing the 53.5M-row HRD CSV needs more memory than a laptop has, but the thing every probe
and readout question actually consumes is far smaller: the window tensor is 3890 x 672 x 4
float32, about 40 MB, and the per-seed masks are a few hundred kilobytes. Everything that does
not train a network or read trained weights -- the random-init audit, marker recovery, probe
and readout comparisons, permutation controls -- runs from this file alone.

The splits are copied from each variant's own metrics.json through the same `load_context`
every experiment uses, so a local run scores exactly the participants the cluster scored. They
are not recomputed here; recomputing them would be a second implementation of the split and a
second chance to disagree with the first.

    python dump_context.py --run-dir results_hrd/2166049 --out hrd_2166049.npz
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", default=None,
                    help="parse the CSV once for all seeds instead of once per seed")
    a = ap.parse_args()

    from pathlib import Path
    from tasks._experiment_common import load_context

    dirs = sorted(d for d in Path(a.run_dir).iterdir()
                  if d.is_dir() and "_seed" in d.name and (d / "metrics.json").exists())
    if not dirs:
        raise SystemExit(f"no variant directories with metrics.json under {a.run_dir}")
    print(f"[dump] {len(dirs)} variants under {a.run_dir}")

    out, X = {}, None
    seeds, cfgs = [], {}
    for d in dirs:
        ctx = load_context(d, a.cache_dir, gpu=-1, require_encoder=False)
        if X is None:
            X = ctx.X
            out["X"] = np.asarray(X, dtype=np.float32)
            out["y"] = np.asarray(ctx.y)
            out["pids"] = np.asarray(ctx.pids).astype(str)
            out["sensor_cols"] = np.asarray(ctx.sensor_cols).astype(str)
            out["n_sensors"] = np.int64(ctx.n_sensors)
            out["bins_per_day"] = np.int64(ctx.bins_per_day)
            out["bin_minutes"] = np.int64(ctx.bin_minutes)
            if ctx.window_ids is not None:
                out["window_ids"] = np.asarray(ctx.window_ids).astype(str)
        elif X.shape != ctx.X.shape:
            # Every seed of one run must see the same windows; a mismatch means the seeds were
            # not built from the same dataset and nothing downstream would be comparable.
            raise SystemExit(f"{d.name}: windows {ctx.X.shape} != {X.shape}")
        s = int(ctx.seed)
        seeds.append(s)
        cfgs[str(s)] = ctx.cfg
        for m in ("train_mask", "val_mask", "test_mask", "last_mask", "pretrain_mask"):
            out[f"{m}/{s}"] = np.asarray(getattr(ctx, m), dtype=bool)
        print(f"  seed {s:>4}: {int(ctx.test_mask.sum()):>5} test windows, "
              f"{len(np.unique(ctx.pids[ctx.test_mask])):>3} test participants")

    out["seeds"] = np.asarray(seeds, dtype=np.int64)
    out["configs_json"] = np.asarray(json.dumps(cfgs))
    np.savez_compressed(a.out, **out)

    from pathlib import Path as P
    print(f"[dump] windows {out['X'].shape} | {len(seeds)} seeds -> {a.out} "
          f"({P(a.out).stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
