"""Choose the representation width (output_dims) per seed x fold, without touching test data.

    python scripts/select_dims.py --dataset hrd --run-name narval_v3 [--prefix tcn_none] [--smoke]

Reads every completed sweep variant <prefix>_d<N> of the run. For each seed x fold, the score of
width N is the inner cross-validation AUROC of the DSSL logistic probe (probe_selection.csv). That
AUROC is computed by stratified 3-fold cross-validation over the fold's training participants only;
the encoder itself never sees a label. The width with the highest score is selected, and ties go to
the smaller width. Writes <run>/<prefix>_selection.csv, which run_experiment.py --output-dims selected
reads.
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['hrd', 'globem'], required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--prefix', default='tcn_none')
    parser.add_argument('--smoke', action='store_true', help='select inside <run>_smoke_cpu')
    args = parser.parse_args()
    base = ROOT / 'results' / args.dataset / (f'{args.run_name}_smoke_cpu' if args.smoke else args.run_name)
    sweep = sorted(base.glob(f'{args.prefix}_d*'), key=lambda p: int(p.name.rsplit('_d', 1)[1]))
    if len(sweep) < 2:
        raise ValueError(f'need at least two sweep variants {args.prefix}_d<N> in {base}')
    scores = []
    for variant in sweep:
        dims = int(variant.name.rsplit('_d', 1)[1])
        for fold_dir in sorted(variant.glob('seed_*/fold_*')):
            if not (fold_dir / 'complete.json').exists():
                raise ValueError(f'incomplete sweep fold: {fold_dir}')
            probe = pd.read_csv(fold_dir / 'probe_selection.csv')
            row = probe[(probe.probe == 'logistic') & (probe.method == 'dssl')]
            if len(row) != 1:
                raise ValueError(f'no DSSL logistic inner score in {fold_dir}')
            scores.append(dict(seed=int(fold_dir.parent.name.split('_')[1]),
                               fold=int(fold_dir.name.split('_')[1]), dims=dims,
                               inner_auc=float(row.inner_auc.iloc[0])))
    frame = pd.DataFrame(scores)
    counts = frame.groupby(['seed', 'fold']).dims.nunique()
    if (counts != len(sweep)).any():
        raise ValueError(f'every seed x fold needs all {len(sweep)} widths: {counts[counts != len(sweep)]}')
    wide = frame.pivot_table(index=['seed', 'fold'], columns='dims', values='inner_auc')
    best = wide.max(axis=1)
    # Ties go to the smaller width: the first column (ascending width) reaching the maximum.
    selected = wide.apply(lambda r: next(d for d in wide.columns if r[d] == r.max()), axis=1)
    out = wide.add_prefix('inner_auc_d').assign(selected_dims=selected, selected_inner_auc=best).reset_index()
    path = base / f'{args.prefix}_selection.csv'
    out.to_csv(path, index=False)
    print(out.round(4).to_string(index=False))
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
