"""Pair two RQ3 ladder tables, arm by arm, in the same variant directory.

Changing the readout rewrites rq3_utility.csv in place, so the run it replaced has to be
kept beside it under another name. Both files then describe the SAME encoder weights, the
same splits and the same held-out people -- only the way the per-timestep output is
collapsed differs -- which is what makes this a paired comparison rather than two tables
printed near each other. Reading the difference off two sets of means is the mistake this
project keeps making; on a 0.02 effect it is also the mistake that would look fine.

    python scripts/compare_readouts.py rq3_utility_meanpool.csv rq3_utility.csv
"""
import argparse
import csv
import glob
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tasks.sign_test import sign_summary

METRICS = ("balanced_acc", "auc")


def read(pattern, base):
    """{arm: {variant dir: {metric: value}}} for one CSV basename."""
    out = defaultdict(dict)
    for f in sorted(glob.glob(pattern.replace("rq3_utility.csv", base))):
        d = csv.DictReader(open(f))
        if "balanced_acc" not in (d.fieldnames or []):
            continue
        for r in d:
            if r["role"] != "ladder" or not r["auc"]:
                continue
            out[r["representation"]][str(Path(f).parent.parent)] = {
                m: float(r[m] or "nan") for m in METRICS}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before", help="CSV basename of the readout being replaced")
    ap.add_argument("after", nargs="?", default="rq3_utility.csv",
                    help="CSV basename of the new one (default: %(default)s)")
    ap.add_argument("--glob", default="results_globem/*/*/RQ3/rq3_utility.csv")
    a = ap.parse_args()

    A, B = read(a.glob, a.before), read(a.glob, a.after)
    arms = [n for n in B if n in A]
    if not arms:
        raise SystemExit(f"no arm appears in both {a.before} and {a.after} under {a.glob}")
    print()
    print(f"  {a.after}  minus  {a.before}, paired per variant directory")
    print()
    print(f"  {'arm':34s} {'bal acc':>28s} {'AUROC':>28s}")
    print(f"  {'':34s} {'after   delta  wins     p':>28s} {'after   delta  wins     p':>28s}")
    for n in sorted(arms, key=lambda n: -np.nanmean([v["balanced_acc"]
                                                     for v in B[n].values()])):
        shared = sorted(set(A[n]) & set(B[n]))
        cells = []
        for m in METRICS:
            d = np.array([B[n][v][m] - A[n][v][m] for v in shared], float)
            after = np.nanmean([B[n][v][m] for v in shared])
            k, cmp_n, p = sign_summary(d)
            cells.append(f"{after:.4f} {np.nanmean(d):+.4f} {k:3d}/{cmp_n:<3d} {p:.4f}")
        print(f"  {n:34s} {cells[0]:>28s} {cells[1]:>28s}")
    missing = [n for n in B if n not in A] + [n for n in A if n not in B]
    if missing:
        print()
        print("  arms present in only one of the two files, so not compared: "
              + ", ".join(sorted(set(missing))))


if __name__ == "__main__":
    main()
