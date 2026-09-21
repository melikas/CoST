"""Per-participant baseline and endpoint depression status for HRD, from the raw export.

    python scripts/hrd_depression_status.py      # writes datasets/cache/hrd_status_v1.csv

The minute-level export repeats each participant's survey fields on every row. This reads
only those fields and keeps one row per participant. A field that takes more than one value
for one participant is an error, not something to resolve silently. The window cache stores
only the endpoint, so the baseline status needed for the transition groups has to come from
here. The endpoint read here must equal the cache label for every labelled participant.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datautils import load_npz

FIELDS = {'depression_status_baseline': 'baseline', 'depression_status_endpoint': 'endpoint',
          'ces_d_baseline_score': 'cesd_baseline', 'ces_d_endpoint_score': 'cesd_endpoint',
          'depression_trajectory': 'trajectory'}
GROUPS = {(0, 0): 'stable_non_depressed', (1, 1): 'stable_depressed',
          (0, 1): 'became_depressed', (1, 0): 'recovered'}


def single(values):
    values = set(values)
    if len(values) > 1:
        raise ValueError(f'conflicting survey values within one participant: {sorted(values)}')
    return next(iter(values), np.nan)


def main():
    seen = {}
    for chunk in pd.read_csv(ROOT / 'datasets/HRD_RAW_MinuteLevel.csv', chunksize=5_000_000,
                             usecols=['pid', *FIELDS], dtype={'pid': str, 'depression_trajectory': str}):
        for pid, g in chunk.groupby('pid'):
            row = seen.setdefault(pid, {k: set() for k in FIELDS})
            for k in FIELDS:
                vals = g[k].dropna().unique()
                row[k] |= {v if k == 'depression_trajectory' else float(v) for v in vals}
    table = pd.DataFrame([{'participant': p, **{FIELDS[k]: single(v) for k, v in row.items()}}
                          for p, row in sorted(seen.items())])
    cohort = load_npz(ROOT / 'datasets/cache/hrd_rescue_v1.npz')
    ids, y = cohort.participants()
    table['labelled'] = table.participant.isin(ids)
    lab = table.set_index('participant').loc[ids]
    if not np.array_equal(lab.endpoint.to_numpy(), y):
        raise ValueError('raw endpoint status disagrees with the cache label')
    table['group'] = [GROUPS.get((int(b), int(e))) if l and np.isfinite(b) and np.isfinite(e) else None
                      for b, e, l in zip(table.baseline, table.endpoint, table.labelled)]
    out = ROOT / 'datasets/cache/hrd_status_v1.csv'
    table.to_csv(out, index=False)
    print(table[table.labelled].group.value_counts(dropna=False).to_string())
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
