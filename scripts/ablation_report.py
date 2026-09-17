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


def fold_headline(fold):
    """The numbers that decide a candidate, for ONE fold, so their spread is visible.

    Pooling folds hides exactly what matters at this sample size: whether an arm's advantage
    holds in every fold or comes from one.
    """
    recovery = pd.read_csv(fold / 'rq1_recovery.csv')
    marker = recovery.assign(marker=recovery.marker.replace({'phase_cos': 'phase',
                                                             'phase_sin': 'phase'}))
    per = marker[marker.method.isin(['dssl', 'untrained'])].groupby(
        ['method', 'marker']).normalized_absolute_error.mean().unstack('method')
    gain = float('nan')
    if {'dssl', 'untrained'} <= set(per.columns):
        families = [m for m in per.index if m in FAMILIES or m == 'phase']
        gain = float((per.loc[families, 'untrained'] - per.loc[families, 'dssl']).mean())
    rq2 = pd.read_csv(fold / 'rq2_personalized.csv', dtype={'participant': str})
    amplitude = rq2[(rq2.perturbation == 'amplitude') & (rq2.method == 'dssl')]
    intensity = float('nan')
    if len(amplitude):
        strongest = amplitude[(amplitude.level == amplitude.level.max()) & (amplitude.raw_delta != 0)]
        if len(strongest):
            intensity = float(np.mean(np.where(
                strongest.representation_delta == 0, .5,
                (np.sign(strongest.representation_delta) == np.sign(strongest.raw_delta)).astype(float))))
    ent = pd.read_csv(fold / 'rq1_disentanglement.csv', dtype={'participant': str})
    phase = {}
    for channel in ('Steps', 'screen'):
        rows = ent[(ent.method == 'dssl') & (ent.target == 'acrophase') & (ent.channel == channel)]
        phase[channel] = tuple(r2(rows[rows.role == role]) if len(rows) else float('nan')
                               for role in ('own', 'leakage'))
    mesor = ent[(ent.method == 'dssl') & (ent.target == 'MESOR')]
    mesor_own = r2(mesor[mesor.role == 'own']) if len(mesor) else float('nan')
    top1 = json.loads((fold / 'dssl_model.json').read_text())['history']['top1'][-1]
    return gain, intensity, phase, mesor_own, top1


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
                out[f'{target}/{channel}'] = (round(float(r2(own)), 3), round(float(r2(leak)), 3))
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
print()
print(f'{"arm":14s} {"fold":>4s} {"top-1":>6s} {"RQ1 gain":>9s} {"intens.":>8s} '
      f'{"Steps own/leak":>15s} {"screen own/leak":>16s} {"MESOR own":>9s}'
      '   (per fold: does it hold everywhere, or come from one fold?)')
per_fold = {}
for arm in ARMS:
    folds = [f for f in sorted((ROOT / 'results' / 'hrd' / arm / 'tcn_none').glob('seed_*/fold_*'))
             if (f / 'complete.json').exists()]
    for fold in folds:
        gain, intensity, phase, mesor_own, top1 = fold_headline(fold)
        (s_own, s_leak), (c_own, c_leak) = phase['Steps'], phase['screen']
        per_fold.setdefault(arm, []).append(
            dict(fold=fold.name, top1=round(top1, 3), rq1_gain=round(gain, 4),
                 intensity_strongest=round(intensity, 3),
                 steps_own=round(float(s_own), 3), steps_leak=round(float(s_leak), 3),
                 screen_own=round(float(c_own), 3), screen_leak=round(float(c_leak), 3),
                 mesor_own=round(float(mesor_own), 3)))
        print(f'{arm:14s} {fold.name.replace("fold_", ""):>4s} {top1:6.3f} {gain:9.4f} '
              f'{intensity:8.3f} {s_own:7.3f}/{s_leak:<7.3f} {c_own:8.3f}/{c_leak:<7.3f} '
              f'{mesor_own:9.3f}')
for m in rows:
    m['per_fold'] = per_fold.get(m['arm'], [])

out = ROOT / 'results' / 'hrd' / 'ablation_summary.json'
out.write_text(json.dumps(rows, indent=1) + '\n')
print('\nwritten to', out)
print('\nCriteria: trend top-1 well below 1.0; RQ1 gain > 0; intensity above 0.5 and rising with '
      'the fraction; acrophase own > leak for Steps and screen; MESOR own >> leak; timing unchanged.')
