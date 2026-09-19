"""RQ4: how does the choice of sequence backbone affect RQ1-RQ3 under one fixed framework?

Reads the outputs the ordinary pipeline already wrote for each backbone and assembles one
comparison table. It runs no model and introduces no objective, metric definition or
protocol of its own:

  RQ1  primary  DSSL vs the untrained encoder, per marker family (rq1_family_intervals.csv)
  RQ1-D primary own - leakage per target (rq1_disentanglement_intervals.csv)
  RQ2  primary  concordance per perturbation, vs untrained (rq2_summary.csv)
  RQ3  primary  participant AUROC, and balanced accuracy / macro-F1 at the fixed 0.5 rule

Sensitivity and specificity are computed here from the same out-of-fold probabilities and
the same fixed 0.5 threshold that produced the balanced accuracy already reported. They are
a decomposition of that metric, not a new one: balanced accuracy is their mean, and the
script asserts that identity holds to 1e-9 for every row it prints.

Usage:
    python scripts/rq4_summary.py --dataset hrd --run-name rq4
    python scripts/rq4_summary.py --dataset hrd --run-name rq4 --method untrained
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
# Backbones in the order the comparison is discussed; any present are reported.
ORDER = ['tcn', 'transformer', 'mamba', 'lstm', 'mlp']


def rq3_metrics(variant: Path, method: str) -> dict | None:
    """Per-seed participant-level metrics, averaged over seeds, as `summarize` reports them."""
    path = variant / 'oof_predictions.csv'
    if not path.exists():
        return None
    frame = pd.read_csv(path, dtype={'participant': str})
    frame = frame[(frame.probe == 'logistic') & (frame.method == method)]
    if frame.empty:
        return None

    per_seed = []
    for seed, group in frame.groupby('seed'):
        wide = group.pivot(index='seed', columns='participant', values='probability')
        ids = wide.columns
        y = group.groupby('participant').label.first().reindex(ids).to_numpy()
        score = wide.to_numpy()[0]
        decision = score >= 0.5
        positive, negative = y == 1, y == 0
        per_seed.append(dict(
            auroc=roc_auc_score(y, score),
            balanced_accuracy=balanced_accuracy_score(y, decision),
            macro_f1=f1_score(y, decision, average='macro', zero_division=0),
            sensitivity=float(decision[positive].mean()),
            specificity=float((~decision[negative]).mean()),
            n=len(ids), positives=int(positive.sum())))

    table = pd.DataFrame(per_seed)
    out = {k: float(table[k].mean()) for k in
           ('auroc', 'balanced_accuracy', 'macro_f1', 'sensitivity', 'specificity')}
    out['auroc_seed_sd'] = float(table.auroc.std(ddof=0))
    out['n'], out['positives'], out['seeds'] = int(table.n.iloc[0]), int(table.positives.iloc[0]), len(table)
    # Balanced accuracy is by definition the mean of sensitivity and specificity; if this
    # fails, the decomposition is not describing the metric that was reported.
    assert abs(out['balanced_accuracy'] - (out['sensitivity'] + out['specificity']) / 2) < 1e-9
    return out


def rq1_summary(variant: Path) -> dict:
    """How many marker families favour DSSL over the untrained encoder, and which fail."""
    path = variant / 'rq1_family_intervals.csv'
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    frame = frame[frame.control == 'untrained']
    if frame.empty:
        return {}
    wins = frame[frame.low > 0].marker.tolist()
    losses = frame[frame.high < 0].marker.tolist()
    return dict(rq1_families=len(frame), rq1_wins=len(wins), rq1_losses=len(losses),
                rq1_loss_markers=','.join(sorted(losses)) or '-')


def rq1d_summary(variant: Path) -> dict:
    """Own-minus-leakage per target for DSSL: how many of the three are separated."""
    path = variant / 'rq1_disentanglement_intervals.csv'
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    frame = frame[(frame.method == 'dssl') & (frame.quantity == 'own_minus_leakage')]
    if frame.empty:
        return {}
    return dict(rq1d_separated=int((frame.low > 0).sum()), rq1d_targets=len(frame))


def rq2_summary(variant: Path) -> dict:
    """Concordance per perturbation for DSSL, and the untrained encoder it is judged against."""
    path = variant / 'rq2_summary.csv'
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    out = {}
    for perturbation, key in (('phase', 'timing'), ('amplitude', 'strength')):
        block = frame[frame.perturbation == perturbation]
        for method, tag in (('dssl', ''), ('untrained', '_untrained')):
            rows = block[block.method == method]
            if not rows.empty:
                out[f'rq2_{key}{tag}'] = float(rows.concordance.mean())
    return out


def collect(dataset: str, run: str, method: str) -> pd.DataFrame:
    base = ROOT / 'results' / dataset / run
    if not base.exists():
        raise SystemExit(f'no results at {base}')
    rows = []
    for variant in sorted(base.iterdir()):
        if not variant.is_dir() or '_' not in variant.name:
            continue
        backbone, encoding = variant.name.rsplit('_', 1)
        rq3 = rq3_metrics(variant, method)
        if rq3 is None:
            print(f'  skipped {variant.name}: no logistic/{method} predictions')
            continue
        row = dict(backbone=backbone, encoding=encoding, **rq3)
        row.update(rq1_summary(variant))
        row.update(rq1d_summary(variant))
        row.update(rq2_summary(variant))
        rows.append(row)
    if not rows:
        raise SystemExit('no backbone variants with predictions were found')
    frame = pd.DataFrame(rows)
    frame['rank'] = frame.backbone.apply(lambda b: ORDER.index(b) if b in ORDER else len(ORDER))
    return frame.sort_values(['rank', 'backbone']).drop(columns='rank').reset_index(drop=True)


def markdown(frame: pd.DataFrame, dataset: str, run: str, method: str) -> str:
    n = int(frame.n.iloc[0])
    seeds = int(frame.seeds.iloc[0])
    lines = [f'# RQ4 - backbone comparison ({dataset}, run `{run}`, method `{method}`)', '',
             f'Every row is the identical framework with only the sequence backbone changed: same '
             f'preprocessing, self-supervised objective, augmentations, optimiser, participant '
             f'splits, embedding extraction and evaluation. {seeds} seeds x 5 participant-disjoint '
             f'folds, {n} evaluated participants.', '',
             '## RQ3 - downstream endpoint prediction', '',
             '| Backbone | AUROC | Balanced Accuracy | F1 | Sensitivity | Specificity |',
             '|---|---:|---:|---:|---:|---:|']
    for _, r in frame.iterrows():
        lines.append(f'| {r.backbone} | {r.auroc:.3f} | {r.balanced_accuracy:.3f} | '
                     f'{r.macro_f1:.3f} | {r.sensitivity:.3f} | {r.specificity:.3f} |')
    lines += ['', 'AUROC is the pre-registered primary metric. Balanced accuracy, macro-F1, '
                  'sensitivity and specificity are all at the fixed 0.5 decision rule; balanced '
                  'accuracy is the mean of the last two.', '']

    if 'rq1_wins' in frame.columns:
        lines += ['## RQ1 - rhythm recovery and disentanglement', '',
                  '| Backbone | Families beating untrained | Families losing | Which lose | '
                  'Targets disentangled |', '|---|---:|---:|---|---:|']
        for _, r in frame.iterrows():
            d = (f'{int(r.rq1d_separated)}/{int(r.rq1d_targets)}'
                 if 'rq1d_separated' in frame.columns and pd.notna(r.get('rq1d_separated')) else '-')
            lines.append(f'| {r.backbone} | {int(r.rq1_wins)}/{int(r.rq1_families)} | '
                         f'{int(r.rq1_losses)} | {r.rq1_loss_markers} | {d} |')
        lines += ['', 'RQ1 is supported only if a backbone beats the untrained encoder in every '
                      'eligible family. Disentanglement counts targets whose own-minus-leakage '
                      'interval excludes zero.', '']

    if 'rq2_timing' in frame.columns:
        lines += ['## RQ2 - personalised rhythmic phenotyping', '',
                  '| Backbone | Phase shift | vs untrained | Amplitude scaling | vs untrained |',
                  '|---|---:|---:|---:|---:|']
        for _, r in frame.iterrows():
            def cell(key):
                v, u = r.get(f'rq2_{key}'), r.get(f'rq2_{key}_untrained')
                if pd.isna(v):
                    return '-', '-'
                return f'{v:.3f}', ('-' if pd.isna(u) else f'{v - u:+.3f}')
            t, td = cell('timing')
            s, sd = cell('strength')
            lines.append(f'| {r.backbone} | {t} | {td} | {s} | {sd} |')
        lines += ['', 'Chance is 0.500. RQ2 requires both perturbations above chance and above the '
                      'untrained encoder.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', choices=['hrd', 'globem'], required=True)
    parser.add_argument('--run-name', default='rq4')
    parser.add_argument('--method', default='dssl',
                        help='representation to report; the controls are also available')
    args = parser.parse_args()

    frame = collect(args.dataset, args.run_name, args.method)
    out = ROOT / 'results' / args.dataset / args.run_name
    frame.to_csv(out / 'RQ4_backbone_table.csv', index=False)
    text = markdown(frame, args.dataset, args.run_name, args.method)
    (out / 'RQ4_SUMMARY.md').write_text(text, encoding='utf-8')
    print(text)
    print(f'\nWrote {out / "RQ4_SUMMARY.md"} and {out / "RQ4_backbone_table.csv"}')


if __name__ == '__main__':
    main()
