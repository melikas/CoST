"""Small, explicit participant-level evaluations for the corrected RQ1--RQ3 protocol."""
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr, mannwhitneyu
from tasks.rhythm import stratum_pairs, concordance
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Target-derived or rhythm-model inputs are RQ3 baselines, never RQ1 recovery of their own targets.
NOT_RQ1 = {'distribution', 'nonparametric', 'yan_cosinor', 'handcrafted'}
# Manuscript super learner: physical distribution, nonparametric descriptors, Yan et al. cosinor.
STACK_BLOCKS = ('distribution', 'nonparametric', 'yan_cosinor')
RQ1_CONTROLS = ('training_mean', 'untrained', 'raw', 'pca', 'random_projection', 'cost_reference_adapter')
RQ2_CONTROLS = ('untrained', 'random_projection', 'cost_reference_adapter')


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def participant_mean(values, pids, ids):
    """Equal weight per participant after averaging that participant's windows."""
    return np.stack([np.mean(values[pids == p], axis=0) for p in ids])


class IsotropicPairScaler(BaseEstimator, TransformerMixin):
    """StandardScaler, except that a circular phase readout's (cos, sin) pairs share a scale.

    Columns [start, start + width) are the cos half and [start + width, start + 2*width) the
    sin half; column i of one pairs with column i of the other. Each pair is centred per
    column and scaled by the RMS of its two standard deviations, a translation plus a uniform
    scaling, so the unit circle stays a circle and Euclidean distance stays monotone in the
    angular gap. Every other column is standardised as usual.
    """

    def __init__(self, start, width):
        self.start = start
        self.width = width

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale < 1e-12] = 1.0
        a, b = int(self.start), int(self.start) + int(self.width)
        c = b + int(self.width)
        if c > X.shape[1]:
            raise ValueError(f'phase pair block [{a}:{c}) exceeds {X.shape[1]} columns')
        shared = np.sqrt((scale[a:b] ** 2 + scale[b:c] ** 2) / 2.0)
        shared[shared < 1e-12] = 1.0
        scale[a:b] = scale[b:c] = shared
        self.scale_ = scale
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_


def feature_scaler(pair=None):
    """Per-column standardisation, or isotropic pair scaling when `pair` = (start, width)."""
    return IsotropicPairScaler(*pair) if pair else StandardScaler()


def logistic_probe(C, seed, pair=None):
    """Primary RQ3 probe: training-fitted imputation and scaling, class-balanced L2 logistic."""
    return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), feature_scaler(pair),
                         LogisticRegression(C=C, class_weight='balanced', max_iter=3000, random_state=seed))


def forest_probe(leaf, seed, pair=None):
    """`pair` is accepted for a uniform signature; trees are scale-invariant."""
    return _forest(leaf, seed)


def _forest(leaf, seed):
    """Secondary ladder (manuscript): 400 balanced trees, minimum leaf round(1/C), C in {0.01,0.1,1}."""
    return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                         RandomForestClassifier(n_estimators=400, min_samples_leaf=leaf, class_weight='balanced',
                                                random_state=seed,
                                                n_jobs=int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))))


PROBES = {'logistic': (logistic_probe, (0.001, 0.01, 0.1, 1.0)), 'forest': (forest_probe, (100, 10, 1))}


def evaluate(features, markers, pids, labels, train_ids, test_ids, channels, output, seed,
             pair_blocks=None):
    """All tuning uses training people; every prediction row is a held-out person.

    RQ1 predicts each person's mean window marker (phase uses circular components).
    Undefined targets remain missing. This tests between-person preservation explicitly;
    it does not establish within-person monitoring or natural emotional-energy validity.
    `pair_blocks` maps a method to the (start, width) of its circular (cos, sin) readout
    columns, which RQ1 and the logistic probe scale isotropically.
    """
    pairs = dict(pair_blocks or {})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ids = np.r_[train_ids, test_ids]
    if len(np.unique(ids)) != len(ids) or not set(ids) <= set(pids):
        raise ValueError('Training/test participant IDs must be unique, disjoint and present')
    if any(len(np.unique(labels[pids == p])) != 1 for p in ids):
        raise ValueError('Endpoint labels must be constant within each participant')
    n = len(train_ids)
    y = np.array([labels[pids == p][0] for p in ids], dtype=int)
    if not np.isin(y,[0,1]).all() or len(np.unique(y[n:])) != 2:
        raise ValueError('Known binary endpoints and both held-out classes are required')
    # Average finite marker windows, retaining missingness instead of inventing targets.
    targets = {}
    for name, values in markers.items():
        a = []
        for p in ids:
            v = values[pids == p]
            count = np.isfinite(v).sum(0)
            a.append(np.divide(np.nansum(v, axis=0), count,
                               out=np.full(v.shape[1], np.nan), where=count > 0))
        targets[name] = np.stack(a)
    phase_norm = np.hypot(targets['phase_cos'], targets['phase_sin'])
    for key in ('phase_cos','phase_sin'):
        targets[key][phase_norm < 1e-6] = np.nan
    aggregated = {name: participant_mean(v, pids, ids) for name, v in features.items()}
    # PCA is a reconstruction control fitted exclusively on training people.
    width = min(aggregated['dssl'].shape[1], n - 1, aggregated['raw'].shape[1])
    pca = PCA(n_components=width, svd_solver='full').fit(aggregated['raw'][:n])
    aggregated['pca'] = pca.transform(aggregated['raw'])
    aggregated['training_mean'] = np.zeros((len(ids),1))
    # Handcrafted blocks use physical signals: distribution (runner), nonparametric
    # descriptors and all targets (here), and the Yan et al. cosinor (runner).
    aggregated['nonparametric'] = np.concatenate([targets[k] for k in ('IS', 'IV', 'RA')], axis=1)
    aggregated['handcrafted'] = np.concatenate([aggregated['distribution']] + list(targets.values()), axis=1)
    recovery, clinical, predictions, choices = [], [], [], []
    errors = {}
    # Ridge penalty fixed in advance: avoids window-wise CV and costly target-wise searches.
    # Standardization fits training rows only; physical targets retain their units.
    for method, z in aggregated.items():
        if method in NOT_RQ1:
            continue  # target-derived or rhythm-model features are not learned target recovery
        # One scaler per method, fitted on every training person. Refitting it on each target's
        # eligible subset (as few as 4 people) let near-constant columns explode predictions.
        z = feature_scaler(pairs.get(method)).fit(z[:n]).transform(z)
        error_columns = []
        for target, truth in targets.items():
            for c, channel in enumerate(channels):
                ok = np.isfinite(truth[:n, c])
                if ok.sum() < 4:
                    continue
                pred = Ridge(alpha=1.0).fit(z[:n][ok], truth[:n, c][ok]).predict(z[n:])
                scale = float(np.std(truth[:n, c][ok]))
                e = np.abs(pred - truth[n:, c]) / scale if scale > 1e-8 else np.full(len(test_ids), np.nan)
                if target not in ('MESOR', 'phase_cos', 'phase_sin'):
                    error_columns.append(e)
                for j, p in enumerate(test_ids):
                    if np.isfinite(truth[n+j, c]):
                        recovery.append(dict(method=method, participant=str(p), marker=target,
                                             channel=channel, truth=truth[n+j, c], prediction=pred[j],
                                             normalized_absolute_error=e[j], training_sd=scale))
        if error_columns:
            es = np.stack(error_columns, axis=1)
            count = np.isfinite(es).sum(1)
            errors[method] = np.divide(np.nansum(es, axis=1), count,
                                       out=np.full(len(test_ids), np.nan), where=count > 0)
    # Classifier training and tuning are at the same participant unit as final scoring.
    folds = min(3, int(np.bincount(y[:n], minlength=2).min()))
    if folds < 2:
        raise ValueError('Classification tuning requires >=2 training people per class')
    split = list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(np.zeros(n), y[:n]))
    scores, selected = {}, {}
    for probe_name, (build, grid) in PROBES.items():
        scores[probe_name] = {}
        for method, z in aggregated.items():
            pair = pairs.get(method)
            inner = {value: float(np.mean([roc_auc_score(y[b], build(value, seed, pair).fit(z[a], y[a]).predict_proba(z[b])[:, 1])
                                           for a, b in split])) for value in grid}
            value = max(grid, key=inner.get)   # first grid value among ties
            selected[probe_name, method] = value
            scores[probe_name][method] = build(value, seed, pair).fit(z[:n], y[:n]).predict_proba(z[n:])[:, 1]
            choices.append(dict(probe=probe_name, method=method, parameter=value, inner_auc=inner[value]))
    # Super learner: NNLS weights on training-only out-of-fold block probabilities, normalized
    # so the stack stays a probability for the fixed 0.5 rule; block probes refit on all training.
    oof = np.zeros((n, len(STACK_BLOCKS)))
    for j, block in enumerate(STACK_BLOCKS):
        for a, b in split:
            oof[b, j] = logistic_probe(selected['logistic', block], seed).fit(
                aggregated[block][a], y[a]).predict_proba(aggregated[block][b])[:, 1]
    weight = nnls(oof, y[:n].astype(float))[0]
    weight = weight / weight.sum() if weight.sum() > 0 else np.full(len(STACK_BLOCKS), 1 / len(STACK_BLOCKS))
    scores['logistic']['handcrafted_stack'] = np.column_stack([scores['logistic'][b] for b in STACK_BLOCKS]) @ weight
    choices.append(dict(probe='logistic', method='handcrafted_stack', inner_auc=np.nan,
                        parameter=json.dumps(dict(zip(STACK_BLOCKS, np.round(weight, 6).tolist())))))
    scores['logistic']['training_prevalence'] = np.full(len(test_ids), y[:n].mean())
    metrics = {}
    for probe_name, probe_scores in scores.items():
        for method, score in probe_scores.items():
            decision = score >= 0.5
            metrics[f'{probe_name}/{method}'] = dict(auroc=float(roc_auc_score(y[n:], score)),
                                   balanced_accuracy=float(balanced_accuracy_score(y[n:], decision)),
                                   macro_f1=float(f1_score(y[n:], decision, average='macro', zero_division=0)))
            for j, p in enumerate(test_ids):
                q = np.clip(score[j], 1e-7, 1 - 1e-7)
                predictions.append(dict(probe=probe_name, method=method, participant=str(p), label=int(y[n+j]),
                                        probability=score[j], log_loss=-y[n+j]*np.log(q)-(1-y[n+j])*np.log(1-q),
                                        rhythm_error=errors.get(method, np.full(len(test_ids), np.nan))[j]))
    # Save raw clinical evidence once per held-out person; aggregate across folds later.
    for target, values in targets.items():
        for c, channel in enumerate(channels):
            for j, p in enumerate(test_ids):
                clinical.append(dict(participant=str(p), label=int(y[n+j]), marker=target,
                                     channel=channel, value=values[n+j, c]))
    pd.DataFrame(recovery).to_csv(output/'rq1_recovery.csv', index=False)
    pd.DataFrame(clinical).to_csv(output/'secondary_endpoint_markers.csv', index=False)
    pd.DataFrame(predictions).to_csv(output/'rq3_predictions.csv', index=False)
    pd.DataFrame(choices).to_csv(output/'probe_selection.csv', index=False)
    write_json(output/'metrics.json', dict(unit='participant', metrics=metrics, pca_width=width,
                                         ridge_alpha=1.0, threshold=0.5))


# ---- RQ1 disentanglement: each window target from its own branch, and from the other branch
# (leakage). target: (own blocks, leakage blocks), each a contiguous run of cost.DSSL.blocks().
BRANCHES = {'MESOR': (('trend',), ('amplitude', 'phase')),
            'amplitude': (('amplitude',), ('trend',)),
            'acrophase': (('phase',), ('trend',))}


def disentanglement(features, layouts, window, pids, window_ids, test_ids, channels, output):
    """Predict every held-out window's targets (tasks.rhythm.window_rhythm) from one branch of
    a frozen representation. Scaling and the ridge (alpha 1, as in RQ1) are fitted on all
    other windows; no label is read. `layouts` maps a method to (DSSL.blocks(), pair_block())."""
    test = np.isin(pids, test_ids)
    frames = []
    for method, (blocks, pair) in layouts.items():
        for target, roles in BRANCHES.items():
            y = np.asarray(np.stack([window['phase_cos'], window['phase_sin']], -1) if target == 'acrophase'
                           else window[target][..., None], dtype=float)      # windows x channels x 1|2
            for role, names in zip(('own', 'leakage'), roles):
                lo, hi = blocks[names[0]][0], blocks[names[-1]][1]
                local = (pair[0] - lo, pair[1]) if pair and lo <= pair[0] < hi else None
                z = features[method][:, lo:hi].astype(float)             # encode() returns float32
                z = feature_scaler(local).fit(z[~test]).transform(z)
                for c, channel in enumerate(channels):
                    ok = np.isfinite(y[:, c]).all(1)
                    fit, score = ok & ~test, ok & test
                    pred = Ridge(alpha=1.0).fit(z[fit], y[fit, c]).predict(z[score]).reshape(-1, y.shape[2])
                    truth = y[score, c]
                    frames.append(pd.DataFrame(dict(
                        method=method, target=target, role=role, channel=channel,
                        participant=pids[score].astype(str), window_id=np.asarray(window_ids)[score].astype(str),
                        truth=truth[:, 0], prediction=pred[:, 0],
                        truth_sin=truth[:, -1] if y.shape[2] == 2 else np.nan,
                        prediction_sin=pred[:, -1] if y.shape[2] == 2 else np.nan)))
    pd.concat(frames, ignore_index=True).to_csv(Path(output)/'rq1_disentanglement.csv', index=False)


def disentanglement_intervals(frame, rng, n_boot):
    """Held-out R², floored at 0, of the own and the leakage branch per method x target: pooled
    over folds within a seed, averaged over channels, then over seeds. One participant
    bootstrap (the same draws for every method, target and seed) gives the intervals of the
    own and leakage R², their difference, and DSSL's difference minus each control's."""
    f = frame.fillna({'truth_sin': 0., 'prediction_sin': 0.})
    f = f.assign(n=1., sse=(f.truth - f.prediction) ** 2 + (f.truth_sin - f.prediction_sin) ** 2,
                 ss=f.truth ** 2 + f.truth_sin ** 2, s_cos=f.truth, s_sin=f.truth_sin)
    stats = f.groupby(['seed', 'method', 'target', 'role', 'channel', 'participant'])[
        ['n', 'sse', 'ss', 's_cos', 's_sin']].sum().unstack('participant', fill_value=0.)
    people = stats.shape[1] // 5
    draws = rng.integers(people, size=(n_boot, people))
    weights = np.vstack([np.ones(people)] + [np.bincount(d, minlength=people) for d in draws])
    n, sse, ss, s_cos, s_sin = np.einsum('bp,ksp->sbk', weights, stats.to_numpy().reshape(len(stats), 5, people))
    with np.errstate(divide='ignore', invalid='ignore'):
        r2 = pd.DataFrame(np.clip(1 - sse / (ss - (s_cos ** 2 + s_sin ** 2) / n), 0, 1).T, index=stats.index)
    r2 = r2.groupby(level=['seed', 'method', 'target', 'role']).mean().groupby(level=['method', 'target', 'role']).mean()
    own, leakage = r2.xs('own', level='role'), r2.xs('leakage', level='role')
    gap = own - leakage
    rows = []

    def add(method, target, quantity, values):
        rows.append(dict(method=method, target=target, quantity=quantity, estimate=values[0],
                         low=np.nanquantile(values[1:], .025), high=np.nanquantile(values[1:], .975)))

    for method, target in own.index:
        add(method, target, 'own', own.loc[(method, target)].to_numpy())
        add(method, target, 'leakage', leakage.loc[(method, target)].to_numpy())
        add(method, target, 'own_minus_leakage', gap.loc[(method, target)].to_numpy())
        if method != 'dssl' and ('dssl', target) in gap.index:
            add(method, target, 'dssl_minus_method', (gap.loc[('dssl', target)] - gap.loc[(method, target)]).to_numpy())
    # Acrophase error in hours (point estimates), for reading the acrophase R² on a clock.
    a = frame[frame.target == 'acrophase']
    hours = np.abs(np.angle(np.exp(1j * (np.arctan2(a.prediction_sin, a.prediction)
                                         - np.arctan2(a.truth_sin, a.truth))))) * 12 / np.pi
    hours = a.assign(hours=hours).groupby(['seed', 'method', 'role']).hours.mean().groupby(['method', 'role']).mean()
    rows += [dict(method=m, target='acrophase', quantity=f'{role}_error_hours', estimate=v, low=np.nan, high=np.nan)
             for (m, role), v in hours.items()]
    channels = pd.DataFrame(np.clip(1 - sse[0] / (ss[0] - (s_cos[0] ** 2 + s_sin[0] ** 2) / n[0]), 0, 1),
                            index=stats.index, columns=['R2']).groupby(level=['method', 'target', 'role', 'channel']).R2.mean()
    return pd.DataFrame(rows), channels.unstack('role').reset_index()


def paired_interval(diff, rng, n_boot):
    """Participant bootstrap of a mean paired difference, conditional on the fitted models."""
    boot = diff[rng.integers(len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    return dict(n=len(diff), difference=float(diff.mean()),
                low=float(np.quantile(boot, .025)), high=float(np.quantile(boot, .975)))


def summarize(root, seeds, folds, smoke=False):
    """Require complete OOF data; paired participant bootstrap, never treat folds as people."""
    from utils import paired_auc_interval
    from result_report import LADDER, write_report
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)
    n_boot = 200 if smoke else 2000
    order = [m for m, *_ in LADDER]
    frames, recovery, clinical, personalized, disentangled = [], [], [], [], []
    invariant = None
    for seed in seeds:
        for fold in range(folds):
            path = root / f'seed_{seed}' / f'fold_{fold}'
            if not (path/'complete.json').exists():
                raise ValueError(f'Missing completed fold: {path}')
            manifest=json.loads((path/'manifest.json').read_text())
            current={k:manifest[k] for k in ['dataset','variant','config','resolved_model','steps','smoke',
                                            'cohort_year','cache_sha256','normalization','code_sha256',
                                            'code_version','versions']}
            # Legitimately per fold: the model seed and training-fitted normalization moments.
            current['resolved_model'] = {k: v for k, v in current['resolved_model'].items()
                                         if k != 'model_seed'}
            current['normalization'] = current['normalization']['policy']
            if manifest['smoke'] != smoke:
                raise ValueError('Cannot mix smoke and scientific results')
            if invariant is None:
                invariant=current
            if current != invariant:
                raise ValueError('Runs have different data, code, settings or dependencies')
            for name, dest in [('rq3_predictions', frames), ('rq1_recovery', recovery),
                               ('secondary_endpoint_markers', clinical), ('rq2_personalized', personalized),
                               ('rq1_disentanglement', disentangled)]:
                frame = pd.read_csv(path/f'{name}.csv', dtype={'participant': str})
                frame['seed'], frame['fold'] = seed, fold
                dest.append(frame)
    # ---- RQ3: one pooled out-of-fold score per participant, seed and probe.
    pred = pd.concat(frames, ignore_index=True)
    if pred.duplicated(['seed', 'probe', 'method', 'participant']).any():
        raise ValueError('Duplicate held-out people: folds are not disjoint')
    pred.to_csv(root/'oof_predictions.csv', index=False)
    label_table = pred.groupby('participant').label
    if (label_table.nunique() != 1).any():
        raise ValueError('Endpoint changed across seeds or methods')
    rows, rq3_intervals, reference_ids = [], [], None
    for probe_name, part in pred.groupby('probe', sort=False):
        matrices = {}
        for method, g in part.groupby('method', sort=False):
            matrix = g.pivot(index='seed', columns='participant', values='probability').reindex(seeds)
            if matrix.isna().any().any():
                raise ValueError('Incomplete participant/seed matrix')
            if reference_ids is None:
                reference_ids = matrix.columns
            if not matrix.columns.equals(reference_ids):
                raise ValueError('Methods have different evaluated participants')
            y = label_table.first().reindex(reference_ids).to_numpy()
            matrices[method] = matrix.to_numpy()
            for seed, score in zip(seeds, matrices[method]):
                rows.append(dict(probe=probe_name, method=method, seed=seed, auroc=roc_auc_score(y, score),
                                 balanced_accuracy=balanced_accuracy_score(y, score >= .5),
                                 macro_f1=f1_score(y, score >= .5, average='macro', zero_division=0)))
        for method, matrix in matrices.items():
            if method != 'dssl':
                r = paired_auc_interval(y, matrices['dssl'], matrix, n_boot=n_boot)
                rq3_intervals.append(dict(probe=probe_name, control=method, n=r['n_participants'],
                                          difference=r['mean_auc_difference'], low=r['ci95'][0], high=r['ci95'][1]))
    table = pd.DataFrame(rows)
    table.to_csv(root/'summary_by_seed.csv', index=False)
    rq3_intervals = pd.DataFrame(rq3_intervals, columns=['probe','control','n','difference','low','high'])
    rq3_intervals.to_csv(root/'rq3_paired_intervals.csv', index=False)
    # ---- RQ1: paired per-person errors; phase is combined geometrically, not scored as two angles.
    rec = pd.concat(recovery, ignore_index=True)
    keys = ['seed','fold','method','participant','channel']
    phase = rec[rec.marker.isin(['phase_cos', 'phase_sin'])].pivot(
        index=keys, columns='marker', values=['truth','prediction'])
    phase_rows = []
    for key, row in phase.dropna().iterrows():
        actual = np.arctan2(row['truth','phase_sin'], row['truth','phase_cos'])
        estimated = np.arctan2(row['prediction','phase_sin'], row['prediction','phase_cos'])
        hours = abs(np.angle(np.exp(1j*(estimated-actual))))*12/np.pi
        phase_rows.append(dict(zip(keys, key), marker='phase_hours', error=hours, family_error=hours))
    scalar = rec[~rec.marker.isin(['phase_cos','phase_sin'])].copy()
    scalar['error'] = abs(scalar.prediction-scalar.truth)
    scalar['family_error'] = scalar.normalized_absolute_error
    err = pd.concat([scalar, pd.DataFrame(phase_rows)], ignore_index=True)
    err.to_csv(root/'rq1_errors.csv', index=False)
    rng = np.random.default_rng(20260914)

    def paired(frame, value):
        return frame.groupby(['participant','method'])[value].mean().unstack()

    # Positive differences are control error minus DSSL error: they favour DSSL.
    cells, families, display = [], [], []
    for (marker, channel), group in err.groupby(['marker','channel']):
        p, q = paired(group, 'error'), paired(group, 'family_error')
        for control in RQ1_CONTROLS:
            if not {'dssl', control} <= set(p.columns):
                continue
            diff = (p[control]-p['dssl']).dropna().to_numpy()
            if len(diff) >= 4:
                cells.append(dict(marker=marker, channel=channel, control=control, **paired_interval(diff, rng, n_boot)))
            diff = (q[control]-q['dssl']).dropna().to_numpy()
            if control == 'untrained' and len(diff) >= 4:
                display.append(dict(marker=marker, channel=channel, **paired_interval(diff, rng, n_boot)))
    # Marker families: channels averaged within person (training-SD units; phase in hours).
    per_person = err.groupby(['seed','method','participant','marker']).family_error.mean().reset_index()
    for marker, group in per_person.groupby('marker'):
        p = paired(group, 'family_error')
        for control in RQ1_CONTROLS:
            if {'dssl', control} <= set(p.columns):
                diff = (p[control]-p['dssl']).dropna().to_numpy()
                if len(diff) >= 4:
                    families.append(dict(marker=marker, control=control, **paired_interval(diff, rng, n_boot)))
    cells = pd.DataFrame(cells, columns=['marker','channel','control','n','difference','low','high'])
    families = pd.DataFrame(families, columns=['marker','control','n','difference','low','high'])
    cells.to_csv(root/'rq1_paired_intervals.csv',index=False)
    families.to_csv(root/'rq1_family_intervals.csv',index=False)
    ent_intervals, ent_channels = disentanglement_intervals(pd.concat(disentangled, ignore_index=True),
                                                            np.random.default_rng(20260914), n_boot)
    ent_intervals.to_csv(root/'rq1_disentanglement_intervals.csv', index=False)
    ent_channels.to_csv(root/'rq1_disentanglement_channels.csv', index=False)
    # ---- RQ2: each seed separately, then participant concordances; windows never cross seeds.
    rq2 = pd.concat(personalized, ignore_index=True)
    rq2_summary, rq2_person = [], []
    for (method, perturbation, seed), group in rq2.groupby(['method','perturbation','seed']):
        if perturbation == 'phase':
            keys=(group.participant.astype(str)+'|'+group.level.astype(str)).to_numpy()
            pairs=stratum_pairs(group.representation_delta.to_numpy(),group.raw_delta.to_numpy(),keys)
            score, count = concordance(pairs), sum(v[1] for v in pairs.values())
            for pid,g in group.groupby('participant'):
                pk=(g.participant.astype(str)+'|'+g.level.astype(str)).to_numpy()
                value=concordance(stratum_pairs(g.representation_delta.to_numpy(),g.raw_delta.to_numpy(),pk))
                if np.isfinite(value):
                    rq2_person.append(dict(method=method,perturbation=perturbation,seed=seed,participant=pid,concordance=value))
        else:
            valid=group[group.raw_delta!=0]
            agreement=np.where(valid.representation_delta==0,.5,
                               (np.sign(valid.representation_delta)==np.sign(valid.raw_delta)).astype(float))
            score, count = (float(np.mean(agreement)) if len(agreement) else np.nan), len(agreement)
            temp=pd.DataFrame(dict(participant=valid.participant.to_numpy(),concordance=agreement))
            for pid,g in temp.groupby('participant'):
                rq2_person.append(dict(method=method,perturbation=perturbation,seed=seed,participant=pid,concordance=g.concordance.mean()))
        people=[r['concordance'] for r in rq2_person if (r['method'],r['perturbation'],r['seed'])==(method,perturbation,seed)]
        rq2_summary.append(dict(method=method,perturbation=perturbation,seed=seed,
                                concordance=np.mean(people) if people else np.nan,
                                pair_weighted_concordance=score,participants=len(people),comparisons=int(count)))
    rq2_summary=pd.DataFrame(rq2_summary,columns=['method','perturbation','seed','concordance',
        'pair_weighted_concordance','participants','comparisons'])
    rq2_person=pd.DataFrame(rq2_person,columns=['method','perturbation','seed','participant','concordance'])
    rq2_summary.to_csv(root/'rq2_summary.csv',index=False);rq2_person.to_csv(root/'rq2_participant_concordance.csv',index=False)
    rq2_intervals=[]
    rng2=np.random.default_rng(20260914)
    for perturbation,g in rq2_person.groupby('perturbation'):
        paired2=g.groupby(['participant','method']).concordance.mean().unstack()
        for method in paired2.columns:
            diff=(paired2[method]-.5).dropna().to_numpy()
            if len(diff)>=4:
                rq2_intervals.append(dict(perturbation=perturbation,comparison=f'{method} - chance',**paired_interval(diff,rng2,n_boot)))
        for control in RQ2_CONTROLS:
            if {'dssl',control} <= set(paired2.columns):
                diff=(paired2.dssl-paired2[control]).dropna().to_numpy()
                if len(diff)>=4:
                    rq2_intervals.append(dict(perturbation=perturbation,comparison=f'dssl - {control}',**paired_interval(diff,rng2,n_boot)))
    pd.DataFrame(rq2_intervals,columns=['perturbation','comparison','n','difference','low','high']).to_csv(root/'rq2_paired_intervals.csv',index=False)
    # ---- Secondary: raw participant markers between endpoint groups, Holm-adjusted.
    clin = pd.concat(clinical).drop_duplicates(['participant','marker','channel'])
    associations = []
    for (marker,channel), g in clin.groupby(['marker','channel']):
        g=g.dropna(subset=['value'])
        a,b=g[g.label==1].value.to_numpy(),g[g.label==0].value.to_numpy()
        if min(len(a),len(b)) < 2:
            continue
        u,p=mannwhitneyu(a,b,alternative='two-sided')
        associations.append(dict(marker=marker,channel=channel,n_positive=len(a),n_negative=len(b),
                                 rank_biserial=2*u/(len(a)*len(b))-1,p=float(p)))
    assoc=pd.DataFrame(associations)
    if len(assoc):
        order_p=np.argsort(assoc.p.to_numpy()); adjusted=np.maximum.accumulate(
            (len(assoc)-np.arange(len(assoc)))*assoc.p.to_numpy()[order_p])
        assoc['p_holm']=np.nan; assoc.loc[order_p,'p_holm']=np.minimum(adjusted,1)
    assoc.to_csv(root/'secondary_endpoint_associations.csv',index=False)
    bridges=[]
    # Primary probe only. Average repeated predictions first; class-stratified association.
    for (method,label),g in pred[pred.probe=='logistic'].groupby(['method','label']):
        g=g.groupby('participant')[['rhythm_error','log_loss']].mean().dropna()
        if len(g)>=5 and g.rhythm_error.nunique()>1 and g.log_loss.nunique()>1:
            r,p=spearmanr(g.rhythm_error,g.log_loss)
            bridges.append(dict(method=method,label=int(label),n=len(g),spearman=float(r),p_exploratory=float(p)))
    write_json(root/'rq3_exploratory_bridge.json',bridges)
    # ---- Figures, all from the saved tables above.
    primary = table[table.probe == 'logistic']
    shown = [m for m in order if m in set(primary.method)]
    fig,(ax,bx)=plt.subplots(1,2,figsize=(15,5))
    for i,m in enumerate(shown):
        v=primary[primary.method==m].auroc.to_numpy()
        ax.scatter(np.full(len(v),i),v,s=14,color='grey')
        ax.errorbar(i,v.mean(),yerr=v.std(ddof=1) if len(v)>1 else 0,fmt='ko',capsize=4)
    ax.axhline(.5,color='grey',ls=':')
    ax.set(xticks=range(len(shown)),ylabel='Participant AUROC',title='RQ3 primary probe (points: seeds; black: mean ± SD)')
    ax.set_xticklabels(shown,rotation=40,ha='right')
    iv=rq3_intervals[rq3_intervals.probe=='logistic'].set_index('control').reindex([m for m in shown if m!='dssl']).dropna()
    d,lo,hi=iv.difference.to_numpy(),iv.low.to_numpy(),iv.high.to_numpy()
    bx.errorbar(d,range(len(iv)),xerr=[d-lo,hi-d],fmt='o',capsize=3)
    bx.axvline(0,color='grey')
    bx.set(yticks=range(len(iv)),yticklabels=iv.index,xlabel='DSSL minus control AUROC (paired participant bootstrap 95% CI)',
           title='Positive favours DSSL')
    fig.tight_layout(); fig.savefig(root/'rq3_auroc.png',dpi=180,bbox_inches='tight'); plt.close(fig)
    if len(assoc):
        fig,ax=plt.subplots(figsize=(9,max(4,len(assoc)*.18)))
        ax.barh([f'{r.marker}: {r.channel}' for r in assoc.itertuples()],assoc.rank_biserial)
        ax.set(xlabel='Rank-biserial association with exported endpoint');fig.tight_layout()
        fig.savefig(root/'secondary_endpoint_associations.png',dpi=180);plt.close(fig)
    if len(rq2_summary):
        fig,ax=plt.subplots(figsize=(8,4))
        pivot=rq2_summary.pivot_table(index='method',columns='perturbation',values='concordance')
        pivot.plot.bar(ax=ax);ax.axhline(.5,color='grey');ax.set(ylabel='Within-person concordance (mean over seeds)',ylim=(0,1))
        ax.tick_params(axis='x',rotation=25);fig.tight_layout();fig.savefig(root/'rq2_personalized.png',dpi=180);plt.close(fig)
    if len(families):
        labels=sorted(families.marker.unique())
        controls=[c for c in RQ1_CONTROLS if c in set(families.control)]
        fig,ax=plt.subplots(figsize=(10,max(4,len(labels)*.9)))
        for k,control in enumerate(controls):
            g=families[families.control==control]
            y=[labels.index(m)+(k-(len(controls)-1)/2)*.12 for m in g.marker]
            d,lo,hi=g.difference.to_numpy(),g.low.to_numpy(),g.high.to_numpy()
            ax.errorbar(d,y,xerr=[d-lo,hi-d],fmt='o',capsize=2,label=control)
        ax.axvline(0,color='grey');ax.legend(fontsize=8)
        ax.set(yticks=range(len(labels)),yticklabels=labels,title='RQ1 marker families: positive favours DSSL',
               xlabel='Control minus DSSL error, paired 95% CI (phase: hours; others: training-SD units)')
        fig.tight_layout();fig.savefig(root/'rq1_families.png',dpi=180,bbox_inches='tight');plt.close(fig)
    if display:
        # Channels differ in physical units: scalar panels use training-SD units, phase hours.
        display=pd.DataFrame(display)
        panels=sorted(display.marker.unique())
        fig,axes=plt.subplots(len(panels),1,figsize=(13,max(4,len(panels)*3.5)),squeeze=False)
        for ax,marker in zip(axes[:,0],panels):
            values=display[display.marker==marker].reset_index(drop=True)
            d,lo,hi=values.difference.to_numpy(),values.low.to_numpy(),values.high.to_numpy()
            ax.errorbar(d,values.index.to_numpy(),xerr=[np.maximum(d-lo,0),np.maximum(hi-d,0)],fmt='o')
            ax.set(yticks=values.index,yticklabels=values.channel,title=marker,
                   xlabel='Untrained minus DSSL MAE; paired 95% CI ('+('hours' if marker=='phase_hours' else 'training target SD units')+')')
            ax.tick_params(axis='y',labelsize=7);ax.axvline(0,color='grey')
        fig.tight_layout();fig.savefig(root/'rq1_recovery.png',dpi=180,bbox_inches='tight');plt.close(fig)
    write_report(root, smoke=smoke)
