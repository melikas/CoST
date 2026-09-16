"""Score the minimal-repair ablation arms on the mechanistic criteria, from per-fold files.

    python scripts/ablation_report.py [arm ...]        # default: the five configs/ablation arms

`summarize` needs the full seed x fold matrix, so it cannot read a 2-fold ablation. This reads
each arm's fold directories directly and reports exactly the criteria the repair has to meet:

    trend      MoCo top-1 at the last update            must NOT be ~1.0 (saturated)
    RQ1        recovery error, DSSL vs its untrained control (positive gain = training helps)
    intensity  RQ2 amplitude concordance per fraction    must be > 0.5 and rise with the fraction
    phase      own-branch vs leakage R2, per channel     own > leak, especially Steps and screen
    MESOR      own-branch vs leakage R2                  must stay separated
    timing     RQ2 timing concordance per shift          must stay high
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARMS = sys.argv[1:] or ['a0_current', 'a1_trend', 'a2_amplitude', 'a3_phase', 'a4_combined']
FAMILIES = ('amplitude', 'phase_hours', 'IS', 'IV', 'RA')


def r2(frame):
    """Held-out R2 floored at 0, over pooled rows; two components when the target is circular."""
    truth = np.column_stack([frame.truth, frame.truth_sin.fillna(0)])
    pred = np.column_stack([frame.prediction, frame.prediction_sin.fillna(0)])
    sse = ((truth - pred) ** 2).sum()
    sst = ((truth - truth.mean(0)) ** 2).sum()
    return max(0.0, 1 - sse / sst) if sst > 0 else float('nan')


def arm_metrics(arm):
    folds = sorted((ROOT / 'results' / 'hrd' / arm / 'tcn_none').glob('seed_*/fold_*'))
    folds = [f for f in folds if (f / 'complete.json').exists()]
    if not folds:
        return None
    out = {'arm': arm, 'folds': len(folds)}
    out['trend_top1'] = round(float(np.mean(
        [json.loads((f / 'dssl_model.json').read_text())['history']['top1'][-1] for f in folds])), 3)
    recovery = pd.concat([pd.read_csv(f / 'rq1_recovery.csv') for f in folds])
    marker = recovery.assign(marker=recovery.marker.replace({'phase_cos': 'phase', 'phase_sin': 'phase'}))
    per = marker[marker.method.isin(['dssl', 'untrained'])].groupby(
        ['method', 'marker']).normalized_absolute_error.mean().unstack('method')
    if {'dssl', 'untrained'} <= set(per.columns):
        families = [m for m in per.index if m in FAMILIES or m == 'phase']
        out['rq1_gain_vs_untrained'] = round(float((per.loc[families, 'untrained']
                                                    - per.loc[families, 'dssl']).mean()), 4)
        out['rq1_families_improved'] = int((per.loc[families, 'untrained'] > per.loc[families, 'dssl']).sum())
        out['rq1_families'] = len(families)
    rq2 = pd.concat([pd.read_csv(f / 'rq2_personalized.csv', dtype={'participant': str}) for f in folds])
    for perturbation, key in (('amplitude', 'intensity'), ('phase', 'timing')):
        g = rq2[(rq2.perturbation == perturbation) & (rq2.method == 'dssl')]
        by_level = {}
        for level, rows in g.groupby('level'):
            valid = rows[rows.raw_delta != 0]
            by_level[float(level)] = round(float(np.mean(np.where(
                valid.representation_delta == 0, .5,
                (np.sign(valid.representation_delta) == np.sign(valid.raw_delta)).astype(float)))), 3) if len(valid) else None
        out[key] = by_level
    ent = pd.concat([pd.read_csv(f / 'rq1_disentanglement.csv', dtype={'participant': str}) for f in folds])
    ent = ent[ent.method == 'dssl']
    for target in ('MESOR', 'amplitude', 'acrophase'):
        for channel in sorted(ent.channel.unique()):
            sub = ent[(ent.target == target) & (ent.channel == channel)]
            if len(sub):
                own, leak = sub[sub.role == 'own'], sub[sub.role == 'leakage']
                out[f'{target}/{channel}'] = (round(r2(own), 3), round(r2(leak), 3))
    return out


rows = [m for m in (arm_metrics(a) for a in ARMS) if m]
if not rows:
    raise SystemExit(f'No completed ablation folds under results/hrd/<arm>/tcn_none/ for {ARMS}')
print(f'{"arm":14s} {"folds":>5s} {"top-1":>6s} {"RQ1 gain":>9s} {"improved":>9s}  intensity by fraction')
for m in rows:
    print(f'{m["arm"]:14s} {m["folds"]:5d} {m["trend_top1"]:6.3f} '
          f'{m.get("rq1_gain_vs_untrained", float("nan")):9.4f} '
          f'{m.get("rq1_families_improved", 0):4d}/{m.get("rq1_families", 0):<4d} {m["intensity"]}')
print()
print(f'{"arm":14s} timing by shift (hours)')
for m in rows:
    print(f'{m["arm"]:14s} {m["timing"]}')
print()
print(f'{"arm":14s} own/leak R2 - acrophase Steps, acrophase screen, MESOR HR')
for m in rows:
    print(f'{m["arm"]:14s} {m.get("acrophase/Steps")}  {m.get("acrophase/screen")}  {m.get("MESOR/HR")}')
out = ROOT / 'results' / 'hrd' / 'ablation_summary.json'
out.write_text(json.dumps(rows, indent=1) + '\n')
print('\nwritten to', out)
print('\nCriteria: trend top-1 well below 1.0; RQ1 gain > 0; intensity above 0.5 and rising with '
      'the fraction; acrophase own > leak for Steps and screen; MESOR own >> leak; timing unchanged.')
