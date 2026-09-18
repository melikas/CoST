"""Rank development-ladder arms on the label-free criteria. No locked test data is read.

    python scripts/dev_scoreboard.py <run> [<run> ...]

Reads each arm's development SUMMARY tables and prints one comparison across both cohorts:

  RQ1        marker recovery vs untrained and vs raw (the fixed anchor)
  RQ1-D      own-vs-leakage for MESOR, amplitude, acrophase, and vs untrained
  RQ2        timing and strength concordance, and vs untrained
  RQ3-dev    reported LAST and marked, because architecture selection must not use it:
             it is a development probe on development participants, and the locked RQ3
             test stays unread until the architecture is frozen.

Every arm must carry dev_cohort=true in its manifests or the script refuses to score it.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = sys.argv[1:]
if not RUNS:
    raise SystemExit(__doc__)
FAMILIES = ('amplitude', 'phase_hours', 'IS', 'IV', 'RA')


def read(run, dataset, name):
    path = ROOT / 'results' / dataset / run / 'tcn_none' / name
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def check_dev(run, dataset):
    """Refuse to score a run that is not a development run."""
    for manifest in (ROOT / 'results' / dataset / run / 'tcn_none').glob('seed_*/fold_*/manifest.json'):
        if not json.loads(manifest.read_text()).get('dev_cohort'):
            raise SystemExit(f'{run}/{dataset}: manifest says dev_cohort=false -- this is not a '
                             f'development run and must not be used for architecture selection')
        return True
    return False


for dataset in ('hrd', 'globem'):
    present = [r for r in RUNS if (ROOT / 'results' / dataset / r / 'tcn_none').is_dir()]
    if not present:
        continue
    for run in present:
        check_dev(run, dataset)
    print(f'\n{"=" * 100}\n{dataset.upper()} development cohort ({len(present)} arms)\n{"=" * 100}')

    print('\nRQ1 -- marker recovery, DSSL minus control (positive favours DSSL)')
    fam = {r: read(r, dataset, 'RQ1_families.csv') for r in present}
    for control in ('untrained', 'raw'):
        print(f'  vs {control}:')
        for run in present:
            f = fam[run]
            if f is None or control not in f.columns:
                print(f'    {run:24s} unavailable')
                continue
            row = f.set_index('marker')[control].reindex([m for m in FAMILIES if m in set(f.marker)])
            print(f'    {run:24s} ' + '  '.join(f'{m}={str(v)[:22]}' for m, v in row.items()))

    print('\nRQ1-D -- own minus leakage (higher = better separation)')
    for run in present:
        d = read(run, dataset, 'RQ1_disentanglement.csv')
        if d is None:
            print(f'  {run:24s} unavailable')
            continue
        dssl = d[d.method == 'dssl'] if 'method' in d.columns else d
        print(f'  {run:24s} ' + '  '.join(
            f"{r.target[:9]}={str(getattr(r, 'own_minus_leakage', getattr(r, 'own R²', '?')))[:18]}"
            for r in dssl.itertuples()))

    print('\nRQ2 -- concordance (HRD only; chance 0.5)')
    for run in present:
        s = read(run, dataset, 'rq2_summary.csv')
        if s is None or not len(s):
            print(f'  {run:24s} not applicable')
            continue
        d = s[s.method == 'dssl'].set_index('perturbation').concordance
        u = s[s.method == 'untrained'].set_index('perturbation').concordance
        print(f'  {run:24s} ' + '  '.join(
            f'{p}: dssl {d.get(p, float("nan")):.3f} vs untrained {u.get(p, float("nan")):.3f}'
            for p in d.index))

    print('\nRQ3-dev -- DEVELOPMENT PROBE ONLY, must NOT drive architecture selection')
    for run in present:
        t = read(run, dataset, 'summary_by_seed.csv')
        if t is None:
            continue
        g = t[t.probe == 'logistic'].groupby('method').auroc.mean()
        print(f'  {run:24s} dssl {g.get("dssl", float("nan")):.3f} | raw {g.get("raw", float("nan")):.3f} '
              f'| untrained {g.get("untrained", float("nan")):.3f}')

print('\nSelect on the label-free criteria (RQ1, RQ1-D, RQ2). RQ3-dev is context; the locked RQ3 '
      'test is evaluated once, after the architecture is frozen.')
