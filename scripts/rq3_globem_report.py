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
acc, folds = defaultdict(list), defaultdict(set)
for f in files:
    run = f.split("/")[1]
    for r in csv.DictReader(open(f)):
        if r["role"] != "ladder" or not r["auc"]:
            continue
        acc[r["representation"]].append((float(r["auc"]),
                                         float(r["balanced_acc"] or "nan")))
        folds[r["representation"]].add(run)

print(f"{len(files)} variant dirs, {len(set(f.split('/')[1] for f in files))} runs\n")
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
