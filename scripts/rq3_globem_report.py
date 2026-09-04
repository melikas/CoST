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
pair = defaultdict(dict)   # arm -> {variant dir: balanced accuracy}
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
        pair[r["representation"]][f] = float(r["balanced_acc"] or "nan")
        folds[r["representation"]].add(run)

for run, n in sorted(stale.items()):
    print(f"  SKIPPED {n:3d} variant dirs in {run}: no balanced_acc column (older run)")
print(f"{len(used)} variant dirs,"
      f" {len(set(f.split(chr(47))[1] for f in used))} runs")
print()
print(f"  {'arm':32s} {'bal acc':>8s} {'AUROC':>8s} {'n':>5s} {'folds':>6s}")
def _ba(n):
    """Sort key. Majority carries no balanced accuracy -- it is not a fitted probe --
    and nanmean over an all-NaN slice warns instead of saying so."""
    v = [b for _, b in acc[n] if not np.isnan(b)]
    return -float(np.mean(v)) if v else 1.0


for name in sorted(acc, key=_ba):
    v = np.array(acc[name], float)
    ba = np.nanmean(v[:, 1]) if not np.isnan(v[:, 1]).all() else np.nan
    print(f"  {name:32s} {ba:8.4f} {np.nanmean(v[:, 0]):8.4f}"
          f" {len(v):5d} {len(folds[name]):6d}")
print("""
  published, same protocol / unit / metric:
    Reorder, leave-one-dataset-out    0.547 +/- 0.008
    Chikersal et al., cross-dataset   0.536 +/- 0.002
    majority                          0.500""")



def sign_p(d):
    """Two-sided sign test. With 96 paired variant dirs a 0.001 gap is not a result until
    something says how often it holds, and the mean alone never will."""
    from math import comb
    n = len(d)
    k = int((d > 0).sum())
    tail = sum(comb(n, i) for i in range(min(k, n - k) + 1))
    return min(1.0, 2 * tail / 2 ** n)


CONTRASTS = [
    # the only one that says whether the LEARNED half contributes: the two arms differ in
    # the V block and in nothing else
    ("DSSL + rhythm + raw skip", "Random-init + rhythm + raw skip"),
    # does combining beat the best single arm at all
    ("DSSL + rhythm + raw skip", "Cosinor (paper)"),
    ("Random-init + rhythm + raw skip", "Cosinor (paper)"),
    # and the standing contrast, unchanged
    ("DSSL (frozen)", "Random-init"),
]

print()
print("  paired over the variant dirs both arms wrote, balanced accuracy")
print()
print(f"  {'contrast':58s} {'delta':>8s} {'wins':>8s} {'p':>8s}")
for a_name, b_name in CONTRASTS:
    shared = sorted(set(pair.get(a_name, {})) & set(pair.get(b_name, {})))
    if not shared:
        print(f"  {a_name + ' - ' + b_name:58s} {'(absent)':>8s}")
        continue
    d = np.array([pair[a_name][f] - pair[b_name][f] for f in shared], float)
    d = d[~np.isnan(d)]
    print(f"  {a_name + ' - ' + b_name:58s} {d.mean():+8.4f}"
          f" {int((d > 0).sum()):4d}/{len(d)} {sign_p(d):8.4f}")