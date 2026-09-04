"""Aggregate the GLOBEM LODO RQ3 ladder across folds, at whatever unit the runs used.

`folds` is the column to read first: an arm that did not write in every fold has a mean
over a different set of people than the arm beside it, and the two cannot be compared.
"""
import csv
import glob
import sys
from collections import defaultdict

import numpy as np

pat = sys.argv[1] if len(sys.argv) > 1 else "results_globem/*/*/RQ3/rq3_utility.csv"
files = sorted(glob.glob(pat))
acc, folds, stale = defaultdict(list), defaultdict(set), defaultdict(int)
used = []
for f in files:
    run = f.split("/")[1]
    d = csv.DictReader(open(f))
    # A file written before the balanced-accuracy column existed cannot be set beside one
    # written after it. Raising on the missing key was the lucky outcome; the failure mode
    # to avoid is averaging an older run in silently, which is how the previous table came
    # to describe a run nobody had launched. Name them and drop them.
    if "balanced_acc" not in (d.fieldnames or []):
        stale[run] += 1
        continue
    used.append(f)
    for r in d:
        if r["role"] != "ladder" or not r["auc"]:
            continue
        acc[r["representation"]].append((float(r["auc"]),
                                         float(r["balanced_acc"] or "nan")))
        folds[r["representation"]].add(run)

for run, n in sorted(stale.items()):
    print(f"  SKIPPED {n:3d} variant dirs in {run}: no balanced_acc column (older run)")
print(f"{len(used)} variant dirs,"
      f" {len(set(f.split(chr(47))[1] for f in used))} runs")
print()
print(f"  {'arm':32s} {'bal acc':>8s} {'AUROC':>8s} {'n':>5s} {'folds':>6s}")
for name in sorted(acc, key=lambda n: -np.nanmean([b for _, b in acc[n]])):
    v = np.array(acc[name], float)
    print(f"  {name:32s} {np.nanmean(v[:, 1]):8.4f} {np.nanmean(v[:, 0]):8.4f}"
          f" {len(v):5d} {len(folds[name]):6d}")
print("""
  published, same protocol / unit / metric:
    Reorder, leave-one-dataset-out    0.547 +/- 0.008
    Chikersal et al., cross-dataset   0.536 +/- 0.002
    majority                          0.500""")
