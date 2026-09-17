"""RQ3 for ablation arms, from folds already on disk. No GPU, no retraining.

    python scripts/ablation_rq3.py <arm> [<arm> ...]

Every arm in the repair programme was scored on RQ1/RQ2 proxies only, yet RQ3 -- "does SSL add
endpoint discrimination against raw and the untrained encoder" -- is the study's own primary
condition and no configuration has ever met it. Each arm holds all 5 folds of seed 1, i.e. one
complete out-of-fold pass over the participants, so RQ3 can be computed exactly as
evaluation_protocol.summarize does it, with one seed instead of three: pool the per-fold
out-of-fold probabilities, then score participant-level AUROC and the paired participant
bootstrap of DSSL minus each control.

Single seed: no seed-to-seed spread, so treat it as a screen, not a substitute for the full run.
Arms run with --skip-reference have no cost_reference_adapter rung.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import paired_auc_interval

ROOT = Path(__file__).resolve().parents[1]
ARMS = sys.argv[1:]
if not ARMS:
    raise SystemExit(__doc__)
PRIMARY = ('raw', 'untrained')          # the protocol's two required RQ3 comparisons
ORDER = ('dssl', 'raw', 'untrained', 'cost_reference_adapter', 'pca', 'random_projection',
         'yan_cosinor', 'handcrafted', 'handcrafted_stack', 'nonparametric', 'distribution',
         'training_mean', 'training_prevalence')

for arm in ARMS:
    folds = sorted((ROOT / 'results' / 'hrd' / arm / 'tcn_none').glob('seed_*/fold_*'))
    folds = [f for f in folds if (f / 'complete.json').exists()]
    if not folds:
        print(f'{arm}: no completed folds')
        continue
    pred = pd.concat([pd.read_csv(f / 'rq3_predictions.csv', dtype={'participant': str})
                      .assign(seed=int(f.parent.name.split('_')[1])) for f in folds],
                     ignore_index=True)
    pred = pred[pred.probe == 'logistic']
    if pred.duplicated(['seed', 'method', 'participant']).any():
        raise ValueError(f'{arm}: folds are not disjoint within a seed')
    labels = pred.groupby('participant').label.first()
    y = labels.to_numpy()
    seeds = sorted(int(s) for s in pred.seed.unique())
    print(f'\n=== {arm}: {len(folds)} folds, seeds {seeds}, {len(y)} participants, '
          f'{int(y.sum())} positive')
    scores = {}
    for method, g in pred.groupby('method'):
        matrix = g.pivot(index='seed', columns='participant', values='probability')
        matrix = matrix.reindex(index=seeds, columns=labels.index)
        if matrix.isna().any().any():
            continue
        scores[method] = matrix.to_numpy()
    print(f'{"method":24s} {"AUROC":>7s} {"BAcc":>7s} {"MF1":>7s}   DSSL minus this [95% CI]')
    for method in [m for m in ORDER if m in scores] + [m for m in scores if m not in ORDER]:
        m = scores[method]
        auroc = float(np.mean([roc_auc_score(y, s) for s in m]))
        bacc = float(np.mean([balanced_accuracy_score(y, s >= .5) for s in m]))
        mf1 = float(np.mean([f1_score(y, s >= .5, average='macro', zero_division=0) for s in m]))
        line = f'{method:24s} {auroc:7.3f} {bacc:7.3f} {mf1:7.3f}'
        if method != 'dssl' and 'dssl' in scores:
            r = paired_auc_interval(y, scores['dssl'], m, n_boot=2000)
            lo, hi = r['ci95']
            verdict = ('favours DSSL' if lo > 0 else 'favours control' if hi < 0 else 'inconclusive')
            flag = '  <-- primary' if method in PRIMARY else ''
            line += f'   {r["mean_auc_difference"]:+.3f} [{lo:+.3f}, {hi:+.3f}] {verdict}{flag}'
        print(line)
    if 'dssl' in scores:
        met = [paired_auc_interval(y, scores['dssl'], scores[c], n_boot=2000)['ci95'][0] > 0
               for c in PRIMARY if c in scores]
        print(f'   RQ3 primary conditions (DSSL above raw AND above untrained): '
              f'{"met" if met and all(met) else "NOT met"}')
