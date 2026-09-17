"""Adversarial audit of an RQ3 ablation screen, from saved predictions only. No GPU.

    python scripts/audit_rq3_arms.py <baseline_arm> <arm> [<arm> ...]

Asks whether a one-seed RQ3 number is trustworthy, not whether it is encouraging:
  1. per-fold AUROC and class counts        - is a pooled gain broad or one lucky fold?
  2. leave-one-fold-out pooled AUROC        - does dropping any single fold erase it?
  3. fold-block bootstrap                   - uncertainty when folds, not just people, resample
  4. exact pair decomposition vs baseline   - which participants flip, and how few carry the gain
  5. calibration at the fixed 0.5 threshold - does AUROC agree with BAcc, and which C was chosen
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
ARMS = sys.argv[1:]
if len(ARMS) < 2:
    raise SystemExit(__doc__)
METHODS = ('dssl', 'raw', 'untrained')


def load(arm):
    folds = [f for f in sorted((ROOT / 'results' / 'hrd' / arm / 'tcn_none').glob('seed_*/fold_*'))
             if (f / 'complete.json').exists()]
    pred = pd.concat([pd.read_csv(f / 'rq3_predictions.csv', dtype={'participant': str})
                      .assign(seed=int(f.parent.name.split('_')[1]), fold=int(f.name.split('_')[1]))
                      for f in folds], ignore_index=True)
    sel = pd.concat([pd.read_csv(f / 'probe_selection.csv') for f in folds], ignore_index=True)
    pred = pred[pred.probe == 'logistic']
    seeds = sorted(pred.seed.unique())
    if len(seeds) > 1:
        print(f'   NOTE {arm}: seeds {seeds} present; auditing seed {seeds[0]} only, '
              f'one out-of-fold pass per participant')
        pred = pred[pred.seed == seeds[0]]
        folds = [f for f in folds if f.parent.name == f'seed_{seeds[0]}']
    return pred, sel[sel.probe == 'logistic'], folds


def wide(pred, method):
    g = pred[pred.method == method].sort_values('participant')
    return g.participant.to_numpy(), g.label.to_numpy(), g.probability.to_numpy(), g.fold.to_numpy()


base_arm = ARMS[0]
base_pred, _, _ = load(base_arm)
_, base_y, base_p, _ = wide(base_pred, 'dssl')
base_ids, _, _, _ = wide(base_pred, 'dssl')

for arm in ARMS:
    pred, sel, folds = load(arm)
    ids, y, _, fold_of = wide(pred, 'dssl')
    print(f'\n=== {arm}  ({len(folds)} folds, {len(y)} participants, {int(y.sum())} positive)')

    print('   per-fold AUROC and class counts')
    header = f'      {"fold":>4s} {"n":>4s} {"pos":>4s} {"neg":>4s}' + ''.join(f' {m:>11s}' for m in METHODS)
    print(header)
    for fold in sorted(set(fold_of)):
        row = f'      {fold:>4d}'
        m = fold_of == fold
        row += f' {m.sum():>4d} {int(y[m].sum()):>4d} {int((1 - y[m]).sum()):>4d}'
        for method in METHODS:
            _, yy, pp, ff = wide(pred, method)
            sel_m = ff == fold
            auc = roc_auc_score(yy[sel_m], pp[sel_m]) if len(set(yy[sel_m])) == 2 else np.nan
            row += f' {auc:11.3f}'
        print(row)

    scores = {m: wide(pred, m) for m in METHODS}
    pooled = {m: roc_auc_score(v[1], v[2]) for m, v in scores.items()}
    print(f'   pooled: ' + '  '.join(f'{m} {pooled[m]:.3f}' for m in METHODS))

    print('   leave-one-fold-out pooled AUROC (dssl, and dssl - raw / dssl - untrained)')
    for fold in sorted(set(fold_of)):
        keep = fold_of != fold
        d = roc_auc_score(y[keep], scores['dssl'][2][keep])
        r = roc_auc_score(y[keep], scores['raw'][2][keep])
        u = roc_auc_score(y[keep], scores['untrained'][2][keep])
        print(f'      drop fold {fold}: dssl {d:.3f}   dssl-raw {d - r:+.3f}   dssl-untrained {d - u:+.3f}')

    rng = np.random.default_rng(0)
    all_folds = np.array(sorted(set(fold_of)))
    draws = {'raw': [], 'untrained': []}
    for _ in range(2000):
        pick = rng.choice(all_folds, len(all_folds), replace=True)
        idx = np.concatenate([np.flatnonzero(fold_of == f) for f in pick])
        if len(set(y[idx])) < 2:
            continue
        for control in draws:
            draws[control].append(roc_auc_score(y[idx], scores['dssl'][2][idx])
                                  - roc_auc_score(y[idx], scores[control][2][idx]))
    for control, d in draws.items():
        lo, hi = np.quantile(d, [.025, .975])
        print(f'   fold-block bootstrap dssl - {control}: {pooled["dssl"] - pooled[control]:+.3f} '
              f'[{lo:+.3f}, {hi:+.3f}]  (resamples FOLDS, unlike the participant bootstrap)')

    if arm != base_arm:
        # Exact decomposition: AUROC = fraction of (pos, neg) pairs ranked correctly, so the gain
        # over the baseline arm is a count of pairs that flip, attributable to participants.
        assert (ids == base_ids).all() and (y == base_y).all(), 'arms must share participants'
        p_new, p_old = scores['dssl'][2], base_p
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        def pair_matrix(p):
            return (p[pos][:, None] > p[neg][None, :]).astype(float) + \
                   0.5 * (p[pos][:, None] == p[neg][None, :])
        flip = pair_matrix(p_new) - pair_matrix(p_old)
        total = flip.sum()
        print(f'   vs {base_arm}: AUROC {roc_auc_score(y, p_old):.3f} -> {pooled["dssl"]:.3f} '
              f'(= {total:+.0f} of {len(pos) * len(neg)} pairs)')
        contribution = np.concatenate([flip.sum(1), flip.sum(0)])
        order = np.argsort(-np.abs(contribution))
        share = np.cumsum(np.abs(contribution[order])) / np.abs(contribution).sum()
        n80 = int(np.searchsorted(share, 0.8) + 1)
        print(f'      participants carrying 80% of the movement: {n80} of {len(y)}; '
              f'largest single contributor {contribution[order[0]]:+.0f} pairs')
        print(f'      probability rank correlation with {base_arm}: '
              f'{pd.Series(p_new).corr(pd.Series(p_old), method="spearman"):.3f}')

    print('   calibration at the fixed 0.5 threshold (never tuned)')
    for method in METHODS:
        _, yy, pp, _ = wide(pred, method)
        c = sel[sel.method == method].parameter.value_counts().to_dict()
        print(f'      {method:10s} AUROC {roc_auc_score(yy, pp):.3f}  BAcc '
              f'{balanced_accuracy_score(yy, pp >= .5):.3f}  p in [{pp.min():.3f}, {pp.max():.3f}] '
              f'sd {pp.std():.3f}  frac>=0.5 {(pp >= .5).mean():.2f}  C {c}')
