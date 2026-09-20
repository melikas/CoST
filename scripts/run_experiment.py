"""Train one variant for one seed x fold, then evaluate RQ1-RQ3 on the same frozen features.

    python scripts/run_experiment.py --dataset hrd --seed 1 --fold 0              # the config's variant
    python scripts/run_experiment.py --dataset hrd --seed 1 --fold 0 --reference  # CoST reference only
    python scripts/run_experiment.py --dataset hrd --seed 1 --fold 0 \
        --backbone transformer --temporal-encoding sinusoidal                    # an RQ4 variant
    python scripts/run_experiment.py --dataset hrd --summarize [--backbone ...]

A variant writes results/<dataset>/<run>/<backbone>_<encoding>/seed_<s>/fold_<f>/. The CoST
reference adapter is trained once per seed x fold into results/<dataset>/<run>/cost_reference/
and reused by every variant; a variant trains it itself only when it is missing.
"""
import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import torch
from cost import DSSL, band_support, REFERENCE_SHARED, exact_numerics
from datautils import load_npz, make_folds
from evaluation_protocol import disentanglement, evaluate, summarize, write_json
from tasks.personalized import personalized_records
from tasks.projection import RawProjection
from tasks.rhythm import individual_markers, window_rhythm
from tasks.yan_cosinor import yan_cosinor_features

RQ2_COLUMNS = ['method', 'participant', 'window_id', 'perturbation', 'level',
               'representation_delta', 'raw_delta']
NORMALIZATIONS = ('within_person', 'training_global')
# Execution-only model for --smoke: exercises every code path, carries no scientific meaning.
SMOKE_MODEL = dict(output_dims=16, hidden_dims=8, tcn_depth=0, n_layers=1, batch_size=4,
                   moco_k=8, trend_kernel_cap=2)
# Source files whose hashes pin a result; the reference depends only on the first group.
REFERENCE_SOURCES = ['cost.py', 'scripts/run_experiment.py', 'tasks/personalized.py',
                     'tasks/rhythm.py', 'models/*.py']
VARIANT_SOURCES = REFERENCE_SOURCES + ['datautils.py', 'utils.py', 'evaluation_protocol.py',
                                       'result_report.py', 'tasks/*.py', 'data_processing/*.py']


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def code_hashes(patterns):
    paths = sorted({p for pattern in patterns for p in ROOT.glob(pattern)})
    return {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in paths}


def _git(*command):
    return subprocess.run(['git', *command], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def code_version():
    """Git commit and whether tracked files differ from it. On a copy without .git (the
    cluster upload), the CODE_VERSION stamp written by scripts/stamp_version.py."""
    try:
        if Path(_git('rev-parse', '--show-toplevel')).resolve() != ROOT:
            raise OSError('the enclosing git repository is not this checkout')
        return dict(git_commit=_git('rev-parse', 'HEAD'),
                    git_dirty=bool(_git('status', '--porcelain', '--untracked-files=no')),
                    source='git')
    except (OSError, subprocess.CalledProcessError):
        stamp = ROOT / 'CODE_VERSION'
        if stamp.exists():
            return dict(json.loads(stamp.read_text()), source='CODE_VERSION')
        return dict(git_commit=None, git_dirty=None, source='unavailable')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', choices=['hrd', 'globem'], required=True)
    parser.add_argument('--config', type=Path, default=None, help='default: configs/<dataset>.json')
    parser.add_argument('--backbone', default=None, help='override model.backbone')
    parser.add_argument('--temporal-encoding', default=None, help='override model.temporal_encoding')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--reference', action='store_true',
                        help='train (or load) only the CoST reference adapter for this seed x fold')
    parser.add_argument('--dev-cohort', action='store_true',
                        help='architecture development: restrict every participant to the TRAINING '
                             'set of locked protocol fold 0, then run the normal seed x fold '
                             'machinery inside it. Fold 0 test participants are never seen and the '
                             'locked evaluation is untouched')
    parser.add_argument('--skip-reference', action='store_true',
                        help='ABLATIONS ONLY: evaluate without the CoST reference rung, which '
                             'otherwise costs a second full training per task. The scientific '
                             'matrix always includes it; a run started this way is not comparable '
                             'to one that has it')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--run-name', default='narval_v2', help='versioned result namespace')
    args = parser.parse_args()
    allowed = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-')
    if not args.run_name or not set(args.run_name) <= allowed:
        parser.error('run-name requires letters, digits, underscores or hyphens')
    return args


def load_config(args):
    cfg = json.loads(Path(args.config or ROOT / 'configs' / f'{args.dataset}.json').read_text())
    missing = [k for k in ('cache', 'seeds', 'folds', 'split_seed', 'steps', 'val_frac',
                           'normalization', 'marker_coverage', 'model') if k not in cfg]
    if missing:
        raise ValueError(f'config is missing {missing}')
    if cfg['normalization'] not in NORMALIZATIONS:
        raise ValueError(f"normalization must be one of {NORMALIZATIONS}")
    cfg['model'] = dict(cfg['model'])
    if args.backbone:
        cfg['model']['backbone'] = args.backbone
    if args.temporal_encoding:
        cfg['model']['temporal_encoding'] = args.temporal_encoding
    return cfg


def normalize(policy, c, used, raw, obs, pids, training):
    """Model input. within_person: the cache's X, each participant standardised on their own
    full record (the reported pipeline). training_global: per-channel moments of observed
    training values, every training participant weighted equally."""
    if policy == 'within_person':
        if c.metadata.get('normalization') != 'retrospective_full_participant_record':
            raise ValueError('within_person normalization needs a cache standardised within participant')
        return c.X[used].astype(np.float32), dict(policy='within_person', source='cache X')
    means, seconds = [], []
    for p in np.unique(pids[training]):
        x = raw[pids == p].astype(np.float64)
        m = obs[pids == p]
        count = m.sum(axis=(0, 1))
        total = np.where(m, x, 0).sum(axis=(0, 1))
        means.append(np.divide(total, count, out=np.full(x.shape[-1], np.nan), where=count > 0))
        seconds.append(np.divide(np.where(m, x * x, 0).sum(axis=(0, 1)), count,
                                 out=np.full(x.shape[-1], np.nan), where=count > 0))
    mean = np.nanmean(means, axis=0)
    scale = np.sqrt(np.maximum(0, np.nanmean(seconds, axis=0) - mean ** 2))
    scale = np.where(scale < 1e-8, 1., scale)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError('A sensor has no observed training values; investigate rather than drop it')
    return (((raw - mean) / scale).astype(np.float32),
            dict(policy='training_global', mean=mean.tolist(), scale=scale.tolist()))


def pretext_split(training, val_frac, model_seed):
    """(fit, monitor) window indices: a random `val_frac` of the non-test windows only
    monitors the pretext loss, as in the reported runs; no model selection uses it."""
    pre = np.flatnonzero(training)
    perm = np.random.default_rng(model_seed).permutation(len(pre))
    n_val = int(len(pre) * val_frac)
    return pre[perm[n_val:]], pre[perm[:n_val]]


def train_encoder(kwargs, label, out, data, steps):
    """Fit (resuming from the checkpoint), encode every window, collect RQ2 records, save and
    re-load the encoder to check the saved weights reproduce the representation."""
    X, fit_idx, val_idx = data['X'], data['fit_idx'], data['val_idx']
    model = DSSL(**kwargs)
    checkpoint = out / f'{label}_training.pt'
    if checkpoint.exists():
        model.load_training(checkpoint)
    model.fit(X[fit_idx], n_iters=steps, val_data=X[val_idx] if len(val_idx) else None,
              checkpoint_path=checkpoint, checkpoint_every=min(100, steps),
              log_every=max(1, min(100, steps)))
    features = model.encode(X, batch_size=16)
    rows, _ = personalized_records(model, label, X, data['raw'], data['pids'], data['window_ids'],
                                   data['test_ids'], data['bins_per_day'], data['bin_minutes'])
    model.save(out / f'{label}_encoder.pt')
    check = DSSL(**kwargs).load(out / f'{label}_encoder.pt')
    # atol above rtol=1e-5's usual floor: a batch of 2 windows and the full batch run different
    # conv paths, an ordinary ~1e-7 float32 difference (same weights, same inputs) that is inert
    # everywhere except a seasonal bin whose true amplitude is near spectral_readout's eps -- there
    # atan2 is ill-conditioned and the same ~1e-7 noise reaches ~8e-4 rad. A real bug -- wrong
    # weights, wrong normalization -- differs by orders of magnitude more (this check caught
    # exactly that on 2026-09-15, a TF32 training bug: >1e-1).
    np.testing.assert_allclose(check.encode(X[:2], batch_size=16), features[:2], rtol=1e-5, atol=2e-3)
    write_json(out / f'{label}_model.json', dict(config=model.config,
               parameters=sum(p.numel() for p in model.net.parameters()), history=model.history))
    return features, rows, (model.blocks(), model.pair_block())


def cost_reference(base, common, model_cfg, data, steps, device):
    """The CoST reference adapter for this seed x fold: loaded when a matching complete copy
    exists, otherwise trained here under an exclusive lock and cached for every variant."""
    out = base / 'cost_reference' / f"seed_{common['seed']}" / f"fold_{common['fold']['fold']}"
    shared = {k: model_cfg[k] for k in REFERENCE_SHARED if k in model_cfg}
    kwargs = dict(input_dims=data['n_sensors'], seq_len=data['seq_len'],
                  bins_per_day=data['bins_per_day'], method='cost_reference', device=device,
                  model_seed=common['model_seed'], **shared)
    config = DSSL(**{**kwargs, 'device': 'cpu'}).config
    manifest = dict(common, resolved_model=config, code_sha256=code_hashes(REFERENCE_SOURCES))
    if (out / 'complete.json').exists():
        if json.loads((out / 'manifest.json').read_text()) != manifest:
            raise ValueError(f'cached CoST reference differs in data, settings or code: {out}')
        stored = np.load(out / 'representations.npz', allow_pickle=True)
        if not np.array_equal(stored['window_ids'].astype(str), np.asarray(data['window_ids']).astype(str)):
            raise ValueError(f'cached CoST reference covers different windows: {out}')
        rows = pd.read_csv(out / 'rq2_personalized.csv',
                           dtype={'participant': str, 'window_id': str}).to_dict('records')
        layout = json.loads((out / 'reference.json').read_text())
        return (stored['cost_reference_adapter'], rows,
                (layout['blocks'], tuple(layout['pair_block']) if layout['pair_block'] else None))
    out.mkdir(parents=True, exist_ok=True)
    lock = out / '.lock'
    try:
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f'another task is training the CoST reference ({lock}); submit the '
                           'reference stage first, and delete the lock only if no task holds it') from None
    try:
        os.write(handle, f'{socket.gethostname()} pid {os.getpid()}\n'.encode())
        os.close(handle)
        if (out / 'manifest.json').exists() and json.loads((out / 'manifest.json').read_text()) != manifest:
            raise ValueError(f'existing CoST reference manifest differs; use a new run name: {out}')
        write_json(out / 'manifest.json', manifest)
        features, rows, layout = train_encoder(kwargs, 'cost_reference_adapter', out, data, steps)
        np.savez_compressed(out / 'representations.npz', pids=data['pids'],
                            window_ids=data['window_ids'], cost_reference_adapter=features)
        pd.DataFrame(rows, columns=RQ2_COLUMNS).to_csv(out / 'rq2_personalized.csv', index=False)
        write_json(out / 'reference.json', dict(blocks=layout[0], pair_block=layout[1]))
        write_json(out / 'complete.json', dict(status='execution_smoke_only' if common['smoke'] else 'complete',
                                             windows=int(len(data['X']))))
    finally:
        lock.unlink(missing_ok=True)
    return features, rows, layout


def main():
    args = parse_args()
    cfg = load_config(args)
    model_cfg = cfg['model']
    variant = f"{model_cfg['backbone']}_{model_cfg['temporal_encoding']}"
    base = ROOT / 'results' / args.dataset / (f'{args.run_name}_smoke_{args.device}'
                                              if args.smoke else args.run_name)
    seeds = [1] if args.smoke else cfg['seeds']
    folds = 2 if args.smoke else cfg['folds']
    if args.summarize:
        summarize(base / variant, seeds, folds, smoke=args.smoke)
        return
    if args.seed not in seeds or not 0 <= args.fold < folds:
        raise ValueError('Requested seed/fold is outside the declared matrix')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu for local smoke tests')
    version = code_version()
    if not args.smoke and (version['git_commit'] is None or version['git_dirty']):
        raise ValueError('scientific runs need committed code: commit, or run '
                         'scripts/stamp_version.py on a clean checkout before uploading')
    torch.set_num_threads(min(4, int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))))
    exact_numerics()
    c = load_npz(ROOT / cfg['cache'])
    if c.raw_X is None or c.observed is None:
        raise ValueError('Rebuild rescue cache: physical inputs and observation masks are required')
    ids, y = c.participants()
    eligible = np.ones(len(c.X), dtype=bool)
    year = None
    if args.dataset == 'globem':
        # No cross-year identity linkage exists. A single cohort prevents that leakage.
        years = c.participant_years()
        year = min(years.values())
        eligible = np.array([years[p] == year for p in c.pids])
        take = np.array([years[p] == year for p in ids])
        ids, y = ids[take], y[take]
    if args.dev_cohort:
        # Architecture development runs inside the TRAINING participants of locked protocol fold 0.
        # That fold's test participants are never seen here, and the locked evaluation keeps its
        # own folds, so comparability with every previous run is preserved. Selection on these
        # results must use the label-free criteria (RQ1, own-vs-leakage, RQ2); the RQ3 computed
        # here is a development probe on development people, never the locked test.
        locked = make_folds(ids, y, n_folds=cfg['folds'], n_repeats=1,
                            master_seed=cfg['split_seed'])[0]
        keep = np.isin(ids, locked.train_pids)
        ids, y = ids[keep], y[keep]
        eligible &= np.isin(c.pids, ids)
        print(f'development cohort: {len(ids)} participants (fold 0 training set); '
              f'{len(locked.test_pids)} locked-fold-0 test participants excluded')
    if args.smoke:
        # Execution-only subset: prefer broad observed coverage and, for HRD, a contiguous
        # six-week run so the central personalized-baseline path is genuinely exercised.
        def smoke_block(pid):
            idx = np.flatnonzero(c.pids == pid)
            if args.dataset != 'hrd':
                return idx[np.argsort(-c.observed[idx].mean(1).min(1), kind='stable')[:6]]
            dates = np.array([np.datetime64(str(c.window_ids[i]).rsplit('_', 1)[1]) for i in idx])
            candidates = [idx[j:j + 6] for j in range(max(0, len(idx) - 5))
                          if np.all(np.diff(dates[j:j + 6]) == np.timedelta64(7, 'D'))]
            return (max(candidates, key=lambda a: c.observed[a].mean((0, 1)).min())
                    if candidates else np.array([], int))
        blocks = {p: smoke_block(p) for p in ids}
        coverage = np.array([c.observed[blocks[p]].mean((0, 1)).min() if len(blocks[p]) else -1
                             for p in ids])
        selected = np.concatenate([np.flatnonzero(y == label)[np.argsort(-coverage[y == label],
                                                                         kind='stable')[:6]]
                                   for label in (0, 1)])
        if args.dataset == 'hrd' and any(len(blocks[ids[i]]) < 6 for i in selected):
            raise ValueError('Smoke subset lacks six contiguous weeks for personalized RQ2')
        ids, y = ids[selected], y[selected]
        eligible &= np.isin(c.pids, ids)
    fold = make_folds(ids, y, n_folds=folds, n_repeats=1, master_seed=cfg['split_seed'])[args.fold]
    used = np.concatenate([blocks[p] for p in ids]) if args.smoke else np.flatnonzero(eligible)
    raw, obs, pids, labels = c.raw_X[used], c.observed[used], c.pids[used], c.y[used]
    window_ids = np.asarray(c.window_ids[used])
    training = ~np.isin(pids, fold.test_pids)
    if set(pids[training]) & set(fold.test_pids):
        raise AssertionError('Participant leakage')
    X, normalization = normalize(cfg['normalization'], c, used, raw, obs, pids, training)
    steps = cfg['steps']
    if args.smoke:
        model_cfg.update(SMOKE_MODEL)
        steps = 2
    model_seed = int(np.random.SeedSequence([cfg['split_seed'], args.seed, args.fold]).generate_state(1)[0])
    fit_idx, val_idx = pretext_split(training, cfg['val_frac'], model_seed)
    data = dict(X=X, raw=raw, pids=pids, window_ids=window_ids, fit_idx=fit_idx, val_idx=val_idx,
                test_ids=fold.test_pids, n_sensors=c.n_sensors, seq_len=c.seq_len,
                bins_per_day=c.bins_per_day, bin_minutes=c.bin_minutes)
    common = dict(dataset=args.dataset, seed=args.seed, fold=fold.as_dict(), model_seed=model_seed,
                  steps=steps, val_frac=cfg['val_frac'], smoke=args.smoke, device=args.device,
                  cohort_year=year, cache_sha256=digest(ROOT / cfg['cache']),
                  normalization=normalization, code_version=version,
                  training_ids=sorted(map(str, np.unique(pids[training]))),
                  test_ids=list(fold.test_pids), windows=int(len(X)),
                  fit_windows=int(len(fit_idx)), monitor_windows=int(len(val_idx)),
                  dev_cohort=args.dev_cohort)
    if args.skip_reference and args.reference:
        raise ValueError('--reference trains only the reference; --skip-reference omits it')
    reference = None if args.skip_reference else cost_reference(base, common, model_cfg, data,
                                                                steps, args.device)
    if args.reference:
        print(f"CoST reference ready: {base / 'cost_reference'}")
        return

    out = base / variant / f'seed_{args.seed}' / f'fold_{args.fold}'
    out.mkdir(parents=True, exist_ok=True)
    kwargs = dict(input_dims=c.n_sensors, seq_len=c.seq_len, bins_per_day=c.bins_per_day,
                  method='dssl', device=args.device, model_seed=model_seed, **model_cfg)
    untrained = DSSL(**kwargs)
    manifest = dict(common, variant=variant, skip_reference=args.skip_reference, config=cfg,
                    resolved_model=untrained.config,
                    # Derived from the geometry, not part of the model's identity: how much of
                    # this run's own seasonal readout the band structure can actually support.
                    readout_support=band_support(untrained.net)[1],
                    code_sha256=code_hashes(VARIANT_SOURCES),
                    versions=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                                  **{name: __import__('importlib.metadata', fromlist=['version']).version(name)
                                     for name in ['scipy', 'scikit-learn', 'pandas', 'matplotlib',
                                                  'CosinorPy', 'statsmodels', 'seaborn', 'joblib']}),
                    scientific_scope='retrospective endpoint; GLOBEM first cohort only; no prospective or cross-year claim')
    if (out / 'manifest.json').exists():
        if json.loads((out / 'manifest.json').read_text()) != manifest:
            raise ValueError(f'Existing manifest differs: preserve it and use a new version directory: {out}')
        if (out / 'complete.json').exists():
            print(f'Already complete: {out}')
            return
    write_json(out / 'manifest.json', manifest)
    features = {'raw': X.reshape(len(X), -1),
                'distribution': np.concatenate([raw.mean(1), raw.std(1)], axis=1),
                # HRD reference-paper rhythm model; before GPU training, so a missing CosinorPy fails early.
                'yan_cosinor': yan_cosinor_features(raw, window_ids, pids, c.bin_minutes, np.isin(pids, ids))}
    # Manuscript random-projection control, scaled on training windows only; also an RQ2 control.
    projection = RawProjection(c.n_sensors, model_seed).fit(X[training])
    features['random_projection'] = projection.encode(X)
    rq2_rows, rq2_status = personalized_records(projection, 'random_projection', X, raw, pids, window_ids,
                                                fold.test_pids, c.bins_per_day, c.bin_minutes)
    if reference is not None:
        features['cost_reference_adapter'], reference_rows, reference_layout = reference
        rq2_rows.extend(reference_rows)
    features['untrained'] = untrained.encode(X, batch_size=16)
    rows, rq2_status = personalized_records(untrained, 'untrained', X, raw, pids, window_ids,
                                            fold.test_pids, c.bins_per_day, c.bin_minutes)
    rq2_rows.extend(rows)
    # Branch column ranges and circular pairs, per encoder: (DSSL.blocks(), DSSL.pair_block()).
    layouts = {'untrained': (untrained.blocks(), untrained.pair_block())}
    if reference is not None:
        layouts['cost_reference_adapter'] = reference_layout
    del untrained
    features['dssl'], rows, layouts['dssl'] = train_encoder(kwargs, 'dssl', out, data, steps)
    rq2_rows.extend(rows)
    pairs = {m: pair for m, (_, pair) in layouts.items() if pair}
    np.savez_compressed(out / 'representations.npz', pids=pids, window_ids=window_ids,
                        **{k: v for k, v in features.items() if k not in ('raw', 'distribution')})
    targets = individual_markers(raw, obs, c.bins_per_day, c.sensor_cols, cfg['marker_coverage'])
    write_json(out / 'target_definition.json', targets['definition'])
    pd.DataFrame(rq2_rows, columns=RQ2_COLUMNS).to_csv(out / 'rq2_personalized.csv', index=False)
    write_json(out / 'rq2_status.json', rq2_status)
    evaluate(features, targets['values'], pids, labels, np.asarray(fold.train_pids),
             np.asarray(fold.test_pids), c.sensor_cols, out, fold.probe_seed, pair_blocks=pairs)
    disentanglement(features, layouts, window_rhythm(X, c.bins_per_day), pids, window_ids,
                    fold.test_pids, c.sensor_cols, out)
    write_json(out / 'complete.json', dict(status='execution_smoke_only' if args.smoke else 'complete',
                                         windows=len(X), training_windows=int(training.sum())))
    print(f'Completed: {out}')


if __name__ == '__main__':
    main()
