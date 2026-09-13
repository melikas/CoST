"""Paired DeLong between two RQ3 rungs that may live in DIFFERENT runs.

    python scripts/cross_run_delong.py RUN_A "RUNG A" RUN_B "RUNG B"

    python scripts/cross_run_delong.py results_h12 "Raw + DSSL angle_contracted [fusion]" \
        results_eq_h "Raw + Random-init (angle) [fusion]"

The comparison is paired only if both runs scored the same participants with the same labels
in every repeat -- true when they share --master-seed, since the folds are then identical.
That is asserted per repeat rather than assumed. Output follows eval.py's pooled estimator:
one AUROC per repeat over the whole labelled cohort, DeLong per repeat, and the mean
difference with a 95% interval from the mean SE.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cv import delong_test          # noqa: E402
from eval import _pooled_oof        # noqa: E402


def load(run):
    """(pooled oof, {repeat: labels}) for one run directory."""
    per_fold = {}
    for p in sorted(Path(run).glob("*/*/eval.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        per_fold[r["fold"]] = r
    if not per_fold:
        raise SystemExit(f"no eval.json under {run}")
    labels = {}
    for tag, r in per_fold.items():
        labels.setdefault(int(tag[1:tag.index("f")]), {}).update(
            r.get("rq3", {}).get("_labels", {}))
    return _pooled_oof(per_fold), labels


def main(argv):
    if len(argv) != 4:
        raise SystemExit(__doc__)
    run_a, rung_a, run_b, rung_b = argv
    (pa, la), (pb, lb) = load(run_a), load(run_b)
    rows = []
    for r in sorted(set(pa) & set(pb)):
        if rung_a not in pa[r] or rung_b not in pb[r]:
            continue
        if la[r] != lb[r]:
            raise SystemExit(f"repeat {r}: the runs scored different participants or labels, "
                             "so the comparison would not be paired")
        (y, sa), (_, sb) = pa[r][rung_a], pb[r][rung_b]
        rows.append(delong_test(y, sa, sb))
    if not rows:
        raise SystemExit(f"no repeat holds both rungs: {rung_a!r} in {run_a}, "
                         f"{rung_b!r} in {run_b}")
    md, mse = np.mean([d["diff"] for d in rows]), np.mean([d["se"] for d in rows])
    print(f"A  {run_a}: {rung_a}\nB  {run_b}: {rung_b}")
    print("AUROC A  " + " ".join(f"{d['auc1']:.4f}" for d in rows)
          + f"  mean {np.mean([d['auc1'] for d in rows]):.4f}")
    print("AUROC B  " + " ".join(f"{d['auc2']:.4f}" for d in rows)
          + f"  mean {np.mean([d['auc2'] for d in rows]):.4f}")
    print(f"A - B    {md:+.4f}  95% CI [{md - 1.96 * mse:+.4f}, {md + 1.96 * mse:+.4f}]  "
          f"z per repeat " + " ".join(f"{d['z']:+.2f}" for d in rows)
          + f"  (n={rows[0]['n']}, {len(rows)} repeats)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
