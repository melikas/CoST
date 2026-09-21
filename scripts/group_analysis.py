"""How the learned representation differs between depression groups, measured directly.

    python scripts/group_analysis.py      # writes results/hrd/narval_v2/tcn_none/group_analysis/

Reads only saved outputs of the canonical run (representations.npz, rq1_recovery.csv,
oof_predictions.csv), the window cache and datasets/cache/hrd_status_v1.csv
(scripts/hrd_depression_status.py). Nothing is retrained and no label reaches an encoder:
the encoders are self-supervised, and labels enter only the descriptive statistics below.

Groups (HRD, baseline and endpoint CES-D status): stable non-depressed (SN), stable
depressed (SD), became depressed (ND), recovered (DN). "Endpoint" compares the study-end label
over all labelled participants (the RQ3 target).

Analyses:
  1. markers      -- measured rhythm markers by group, and the same markers decoded from the
                     representation (RQ1's out-of-fold ridge predictions).
  2. blocks       -- share of between-participant variance in each representation block that
                     group membership explains (R^2 of a one-way PERMANOVA on Euclidean distance
                     of standardised coordinates), averaged over the 15 encoders, with a
                     label-permutation null.
  3. axis         -- each participant's position on the stable-depressed vs stable-non-depressed
                     axis, computed only by encoders that never saw that participant and with that
                     participant left out of the axis.
  4. trajectories -- the same axis applied to every weekly window, for change over the study.
  5. variability  -- week-to-week dispersion of each participant's representation.
  6. classifier   -- the RQ3 probe's out-of-fold probability by group.
Effect sizes are Hedges' g (acrophase: circular mean difference in hours) with stratified
participant-bootstrap 95% intervals; p-values are label-permutation tests, Holm-corrected within
each family.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datautils import load_npz
from tasks.rhythm import individual_markers

RUN = ROOT / 'results/hrd/narval_v2/tcn_none'
OUT = RUN / 'group_analysis'
SEEDS, FOLDS = (1, 2, 3), range(5)
N_BOOT, N_PERM = 2000, 5000
CHANNELS = {'HR': 'Heart rate', 'Steps': 'Steps', 'is_asleep': 'Sleep', 'screen': 'Screen'}
MARKERS = ['MESOR', 'amplitude', 'acrophase', 'IS', 'IV', 'RA']
SPACES = ('dssl', 'untrained')
# HRD embedding layout (cost.DSSL.blocks, band_support): 160 trend | 5 bins x 160 amplitude |
# 5 bins x 160 phase, bin-major. Readout bins (cycles per week) 1, 7, 14, 21, 28 = weekly,
# 24 h, 12 h, 8 h, 6 h. Rhythm channels 40b..40b+39 belong to band b; a channel carries signal only
# at the bin inside its band, so only those 200 of 800 coordinates per block are used here.
TREND, AMP0, PH0, NCH = 160, 160, 960, 160
SUPPORTED = {0: range(0, 40), 1: range(0, 40), 2: range(40, 80), 3: range(80, 120), 4: range(120, 160)}
BLOCKS = {'Context (trend)': ('trend', None),
          '24 h strength': ('amp', [1]), '24 h timing': ('phase', [1]),
          'Day-shape strength (12, 8, 6 h)': ('amp', [2, 3, 4]),
          'Day-shape timing (12, 8, 6 h)': ('phase', [2, 3, 4]),
          'Weekly strength': ('amp', [0]), 'Weekly timing': ('phase', [0])}


# ----------------------------------------------------------------------------- statistics
def hedges_g(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (1 - 3 / (4 * (na + nb) - 9)) * (a.mean() - b.mean()) / sp


def circ_diff_hours(a, b):
    """Circular mean of a minus circular mean of b, angles in radians, result in hours."""
    return np.angle(np.exp(1j * a).mean() / np.exp(1j * b).mean()) * 12 / np.pi


def compare(a, b, rng, stat=hedges_g):
    """Effect, stratified bootstrap 95% interval and two-sided permutation p for group a vs b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    est = stat(a, b)
    boot = np.array([stat(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(N_BOOT)])
    if stat is circ_diff_hours:          # centre on the estimate before taking percentiles
        boot = est + (boot - est + 12) % 24 - 12
    pooled = np.concatenate([a, b])
    null = np.array([abs(stat(*np.split(rng.permutation(pooled), [len(a)]))) for _ in range(N_PERM)])
    return dict(n_a=len(a), n_b=len(b), effect=est, low=np.percentile(boot, 2.5),
                high=np.percentile(boot, 97.5), p=(1 + (null >= abs(est)).sum()) / (1 + N_PERM))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1., running)
    return adj


def with_holm(rows, family):
    frame = pd.DataFrame(rows)
    for _, idx in frame.groupby(family).groups.items():
        frame.loc[idx, 'p_holm'] = holm(frame.loc[idx, 'p'])
    return frame


# ----------------------------------------------------------------------------- inputs
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs():
    status = pd.read_csv(ROOT / 'datasets/cache/hrd_status_v1.csv', dtype={'participant': str})
    status = status[status.labelled].set_index('participant')
    cohort = load_npz(ROOT / 'datasets/cache/hrd_rescue_v1.npz')
    keep = np.isin(cohort.pids, status.index)
    windows = pd.DataFrame({'participant': cohort.pids[keep].astype(str),
                            'window_id': cohort.window_ids[keep].astype(str)})
    start = pd.to_datetime(windows.window_id.str.rsplit('_', n=1).str[1])
    windows['week'] = ((start - start.groupby(windows.participant).transform('min')).dt.days // 7).astype(int)
    reps, heldout = {}, {}
    for s in SEEDS:
        for f in FOLDS:
            path = RUN / f'seed_{s}' / f'fold_{f}'
            z = np.load(path / 'representations.npz', allow_pickle=True)
            if not np.array_equal(z['window_ids'].astype(str)[keep], windows.window_id.to_numpy()):
                raise ValueError(f'{path}: window order differs from the cache')
            reps[s, f] = {m: z[m][keep] for m in SPACES}
            heldout[s, f] = set(json.loads((path / 'manifest.json').read_text())['test_ids'])
    raw = {'raw_X': cohort.raw_X[keep], 'observed': cohort.observed[keep],
           'sensor_cols': list(cohort.sensor_cols), 'bins_per_day': cohort.bins_per_day}
    return status, windows, reps, heldout, raw


def block_features(V, block):
    """Window x feature matrix for one block: log amplitude, phase as (cos, sin)."""
    kind, bins = BLOCKS[block]
    if kind == 'trend':
        return V[:, :TREND]
    cols = [b * NCH + c for b in bins for c in SUPPORTED[b]]
    if kind == 'amp':
        return np.log(V[:, AMP0 + np.array(cols)])
    angle = V[:, PH0 + np.array(cols)]
    return np.concatenate([np.cos(angle), np.sin(angle)], axis=1)


def all_features(V):
    return np.concatenate([block_features(V, b) for b in BLOCKS], axis=1)


def participant_means(F, participants, order):
    frame = pd.DataFrame(F).groupby(participants).mean()
    return frame.loc[order].to_numpy()


def standardise(M, ref=None):
    ref = M if ref is None else ref
    sd = ref.std(0)
    ok = sd > 1e-12
    return (M[:, ok] - ref.mean(0)[ok]) / sd[ok]


# ----------------------------------------------------------------------------- analyses
def measured_markers(raw, windows):
    """Window-level measured markers and participant-level summaries (acrophase circular)."""
    m = individual_markers(raw['raw_X'], raw['observed'], raw['bins_per_day'], raw['sensor_cols'])['values']
    rows = {}
    for c, name in enumerate(raw['sensor_cols']):
        for key in ('MESOR', 'amplitude', 'IS', 'IV', 'RA'):
            if np.isfinite(m[key][:, c]).any():
                rows[key, name] = m[key][:, c]
        rows['phase_cos', name], rows['phase_sin', name] = m['phase_cos'][:, c], m['phase_sin'][:, c]
    return pd.DataFrame(rows, index=windows.index)


def marker_table(values_by_source, status, rng):
    """values_by_source: {source: DataFrame participant x (marker, channel)} (acrophase in rad)."""
    rows = []
    comparisons = {'Stable depressed vs stable non-depressed': ('stable_depressed', 'stable_non_depressed', 'group'),
                   'Depressed vs non-depressed at endpoint': (1, 0, 'endpoint')}
    for source, table in values_by_source.items():
        for (marker, channel) in table.columns:
            stat = circ_diff_hours if marker == 'acrophase' else hedges_g
            for label, (a, b, col) in comparisons.items():
                x = table[(marker, channel)]
                ga = x[status.loc[x.index, col] == a].dropna()
                gb = x[status.loc[x.index, col] == b].dropna()
                rows.append(dict(source=source, comparison=label, marker=marker,
                                 channel=CHANNELS[channel], **compare(ga, gb, rng, stat)))
    return with_holm(rows, ['source', 'comparison'])


def participant_marker_table(frame, participants):
    g = frame.groupby(participants).mean()
    out = {}
    for (key, ch) in g.columns:
        if key == 'phase_cos':
            out['acrophase', ch] = np.arctan2(g['phase_sin', ch], g['phase_cos', ch])
        elif key != 'phase_sin':
            out[key, ch] = g[key, ch]
    return pd.DataFrame(out)


def decoded_markers(status):
    """RQ1's out-of-fold predictions, averaged over seeds; acrophase from mean (cos, sin)."""
    frames = [pd.read_csv(RUN / f'seed_{s}' / f'fold_{f}' / 'rq1_recovery.csv', dtype={'participant': str})
              for s in SEEDS for f in FOLDS]
    rec = pd.concat(frames)
    rec = rec[rec.participant.isin(status.index)]
    out = {}
    for method in SPACES:
        p = rec[rec.method == method].groupby(['participant', 'marker', 'channel']).prediction.mean().unstack(['marker', 'channel'])
        table = {}
        for ch in CHANNELS:
            for key in ('MESOR', 'amplitude', 'IS', 'IV', 'RA'):
                if (key, ch) in p.columns:
                    table[key, ch] = p[key, ch]
            table['acrophase', ch] = np.arctan2(p['phase_sin', ch], p['phase_cos', ch])
        out[f'{method}_decoded'] = pd.DataFrame(table)
    return out


def block_r2(reps, windows, status, rng):
    """R^2 of group membership per block, averaged over encoders, with a permutation null."""
    rows = []
    designs = {'Stable depressed vs stable non-depressed':
                   status.index[status.group.isin(['stable_depressed', 'stable_non_depressed'])],
               'Depressed vs non-depressed at endpoint': status.index}
    for label, ids in designs.items():
        groups = (status.loc[ids, 'group'] if label.startswith('Stable') else status.loc[ids, 'endpoint']).to_numpy()
        codes = pd.factorize(groups)[0]
        perms = [codes] + [rng.permutation(codes) for _ in range(999)]
        onehots = np.stack([np.eye(codes.max() + 1)[c] for c in perms])      # (P, n, g)
        counts = onehots.sum(1)                                               # (P, g)
        for space in SPACES:
            for block in BLOCKS:
                r2 = np.zeros(len(perms))
                for rep in reps.values():
                    Z = standardise(participant_means(block_features(rep[space], block),
                                                      windows.participant.to_numpy(), ids))
                    sums = np.einsum('png,nd->pgd', onehots, Z)
                    between = ((sums ** 2).sum(-1) / counts).sum(-1)
                    r2 += between / (Z ** 2).sum() / len(reps)
                rows.append(dict(comparison=label, space=space, block=block, n=len(ids), r2=r2[0],
                                 null_mean=r2[1:].mean(), null_95=np.percentile(r2[1:], 95),
                                 p=(1 + (r2[1:] >= r2[0]).sum()) / len(perms)))
    return with_holm(rows, ['comparison', 'space'])


def axis_scores(reps, heldout, windows, status, measured_window):
    """Participant and window positions on the stable-depressed minus stable-non-depressed axis.

    For participant i, only the encoders that never saw i are used (i is in their test fold),
    and the axis is computed from the stable participants of that encoder excluding i. Position
    0 is the stable non-depressed centroid and 1 the stable depressed centroid."""
    ids = status.index.to_numpy()
    parts = windows.participant.to_numpy()
    stable_sn = status.index[status.group == 'stable_non_depressed']
    stable_sd = status.index[status.group == 'stable_depressed']
    window_rows, person_rows = [], []

    spaces = {s: {} for s in SPACES}
    for key, rep in reps.items():
        for space in SPACES:
            spaces[space][key] = all_features(rep[space])
    # measured markers: one "encoder", every participant scored with themself left out
    spaces['measured'] = {('m', 0): measured_window}
    for space, encoders in spaces.items():
        per_window = np.zeros(len(windows))
        used = np.zeros(len(windows))
        for key, F in encoders.items():
            F = np.where(np.isfinite(F), F, np.nanmean(F, 0))
            P = pd.DataFrame(F).groupby(parts).mean()
            mu, sd = P.mean(0).to_numpy(), P.std(0).to_numpy()
            sd[sd < 1e-12] = np.inf
            P = (P - mu) / sd
            Fz = (F - mu) / sd
            targets = ids if space == 'measured' else [p for p in ids if p in heldout[key]]
            m_sn, m_sd = P.loc[stable_sn].mean(0), P.loc[stable_sd].mean(0)
            n_sn, n_sd = len(stable_sn), len(stable_sd)
            for p in targets:
                a, b = m_sn, m_sd
                if p in stable_sn:
                    a = (m_sn * n_sn - P.loc[p]) / (n_sn - 1)
                if p in stable_sd:
                    b = (m_sd * n_sd - P.loc[p]) / (n_sd - 1)
                w = (b - a).to_numpy()
                rows = parts == p
                per_window[rows] += (Fz[rows] - a.to_numpy()) @ w / (w @ w)
                used[rows] += 1
        if (used == 0).any():
            raise ValueError(f'{space}: some windows were never scored by a held-out encoder')
        per_window /= used
        frame = windows.assign(space=space, score=per_window)
        window_rows.append(frame)
    W = pd.concat(window_rows, ignore_index=True)
    W['group'] = status.loc[W.participant, 'group'].to_numpy()
    person = W.groupby(['space', 'participant']).score.mean().reset_index()
    person['group'] = status.loc[person.participant, 'group'].to_numpy()
    person['endpoint'] = status.loc[person.participant, 'endpoint'].to_numpy()
    return W, person


def axis_effects(person, rng):
    rows = []
    pairs = [('stable_depressed', 'stable_non_depressed'), ('became_depressed', 'stable_non_depressed'),
             ('recovered', 'stable_depressed'), ('recovered', 'stable_non_depressed'),
             ('became_depressed', 'stable_depressed')]
    for space, g in person.groupby('space'):
        for a, b in pairs:
            xa, xb = g.score[g.group == a].to_numpy(), g.score[g.group == b].to_numpy()
            r = compare(xa, xb, rng)
            # Robustness: g after removing any one participant, and after trimming each group's
            # 10% most extreme scores.
            loo = [hedges_g(np.delete(xa, i), xb) for i in range(len(xa))] + \
                  [hedges_g(xa, np.delete(xb, i)) for i in range(len(xb))]
            trim = lambda x: x[(x >= np.percentile(x, 5)) & (x <= np.percentile(x, 95))]
            rows.append(dict(space=space, group_a=a, group_b=b, **r, loo_min=min(loo), loo_max=max(loo),
                             trimmed_g=hedges_g(trim(xa), trim(xb)),
                             mean_a=xa.mean(), mean_b=xb.mean()))
    return with_holm(rows, ['space'])


def trajectory_effects(W, rng, span=4):
    """Per participant: mean position in the last `span` weeks minus the first `span`."""
    rows, change = [], []
    for (space, p), g in W.groupby(['space', 'participant']):
        g = g.sort_values('week')
        if len(g) >= 2 * span:
            change.append(dict(space=space, participant=p, group=g.group.iloc[0],
                               change=g.score.iloc[-span:].mean() - g.score.iloc[:span].mean(),
                               weeks=int(g.week.iloc[-1])))
    change = pd.DataFrame(change)
    for space, g in change.groupby('space'):
        for grp, h in g.groupby('group'):
            x = h.change.to_numpy()
            boot = [rng.choice(x, len(x)).mean() for _ in range(N_BOOT)]
            flips = np.array([(x * rng.choice([-1, 1], len(x))).mean() for _ in range(N_PERM)])
            rows.append(dict(space=space, test=f'change within {grp}', n_a=len(x), n_b=np.nan,
                             effect=x.mean(), low=np.percentile(boot, 2.5), high=np.percentile(boot, 97.5),
                             p=(1 + (abs(flips) >= abs(x.mean())).sum()) / (1 + N_PERM)))
        for a, b in [('became_depressed', 'stable_non_depressed'), ('recovered', 'stable_depressed')]:
            r = compare(g.change[g.group == a], g.change[g.group == b], rng)
            rows.append(dict(space=space, test=f'change: {a} vs {b} (g)', **r))
    return change, with_holm(rows, ['space'])


def variability(reps, windows, status, measured_window, rng):
    """Within-person dispersion of weekly representations around the person's own mean, in
    units of the between-window SD (averaged over encoders)."""
    parts = windows.participant.to_numpy()
    disp = {}
    for space in SPACES:
        for block in BLOCKS:
            acc = 0.
            for rep in reps.values():
                F = block_features(rep[space], block)
                F = (F - F.mean(0)) / np.where(F.std(0) > 1e-12, F.std(0), np.inf)
                dev = F - pd.DataFrame(F).groupby(parts).transform('mean').to_numpy()
                acc = acc + pd.Series(np.sqrt((dev ** 2).mean(1))).groupby(parts).mean() / len(reps)
            disp[space, block] = acc
    F = measured_window.to_numpy()
    F = np.where(np.isfinite(F), F, np.nanmean(F, 0))
    F = (F - F.mean(0)) / F.std(0)
    dev = F - pd.DataFrame(F).groupby(parts).transform('mean').to_numpy()
    disp['measured', 'All measured markers'] = pd.Series(np.sqrt((dev ** 2).mean(1))).groupby(parts).mean()
    rows = []
    for (space, block), s in disp.items():
        g = status.loc[s.index, 'group']
        rows.append(dict(space=space, block=block,
                         **compare(s[g == 'stable_depressed'], s[g == 'stable_non_depressed'], rng)))
    return with_holm(rows, ['space'])


def classifier(status, rng):
    pred = pd.read_csv(RUN / 'oof_predictions.csv', dtype={'participant': str})
    pred = pred[(pred.probe == 'logistic') & pred.participant.isin(status.index)]
    rows, aucs = [], []
    for method, g in pred.groupby('method'):
        prob = g.groupby('participant').probability.mean()
        grp = status.loc[prob.index, 'group']
        for name in ['stable_non_depressed', 'stable_depressed', 'became_depressed', 'recovered']:
            x = prob[grp == name].to_numpy()
            boot = [rng.choice(x, len(x)).mean() for _ in range(N_BOOT)]
            rows.append(dict(method=method, group=name, n=len(x), mean_probability=x.mean(),
                             low=np.percentile(boot, 2.5), high=np.percentile(boot, 97.5)))
        known = grp.notna()
        st = grp.isin(['stable_non_depressed', 'stable_depressed'])
        tr = grp.isin(['became_depressed', 'recovered'])
        end, base = status.loc[prob.index, 'endpoint'], status.loc[prob.index, 'baseline']
        for label, mask, target in [('endpoint, all labelled', slice(None), end),
                                    ('endpoint, stable only', st, end),
                                    ('endpoint, changers only', tr, end),
                                    ('baseline status, all with baseline', known, base)]:
            y, s = target[mask].to_numpy(), prob[mask].to_numpy()
            boot = []
            pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
            for _ in range(N_BOOT):
                idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
                boot.append(roc_auc_score(y[idx], s[idx]))
            aucs.append(dict(method=method, target=label, n=len(y), positives=int(y.sum()),
                             auroc=roc_auc_score(y, s), low=np.percentile(boot, 2.5),
                             high=np.percentile(boot, 97.5)))
    return pd.DataFrame(rows), pd.DataFrame(aucs)


def main():
    rng = np.random.default_rng(20260921)
    OUT.mkdir(parents=True, exist_ok=True)
    status, windows, reps, heldout, raw = load_inputs()
    grouped = status.group.notna()
    print(f'{len(status)} labelled participants; groups: {status.group.value_counts().to_dict()}')

    measured_window = measured_markers(raw, windows)
    measured_person = participant_marker_table(measured_window, windows.participant.to_numpy())
    markers = marker_table({'measured': measured_person, **decoded_markers(status)}, status, rng)
    markers.to_csv(OUT / 'markers_by_group.csv', index=False)

    status_g = status[grouped]
    block_r2(reps, windows, status_g, rng).to_csv(OUT / 'block_r2.csv', index=False)

    keep = windows.participant.isin(status_g.index).to_numpy()
    win_g = windows[keep].reset_index(drop=True)
    reps_g = {k: {s: v[s][keep] for s in SPACES} for k, v in reps.items()}
    mw = measured_window[keep].reset_index(drop=True)
    W, person = axis_scores(reps_g, heldout, win_g, status_g, mw.to_numpy())
    W.to_csv(OUT / 'window_scores.csv', index=False)
    person.to_csv(OUT / 'participant_scores.csv', index=False)
    axis_effects(person, rng).to_csv(OUT / 'axis_effects.csv', index=False)
    change, traj = trajectory_effects(W, rng)
    change.to_csv(OUT / 'trajectory_change.csv', index=False)
    traj.to_csv(OUT / 'trajectory_effects.csv', index=False)
    variability(reps_g, win_g, status_g, mw, rng).to_csv(OUT / 'variability.csv', index=False)
    by_group, aucs = classifier(status, rng)
    by_group.to_csv(OUT / 'classifier_by_group.csv', index=False)
    aucs.to_csv(OUT / 'classifier_auroc_by_target.csv', index=False)
    (OUT / 'manifest.json').write_text(json.dumps(dict(
        script='scripts/group_analysis.py', script_sha256=digest(__file__),
        status_sha256=digest(ROOT / 'datasets/cache/hrd_status_v1.csv'),
        cache_sha256=digest(ROOT / 'datasets/cache/hrd_rescue_v1.npz'),
        run=str(RUN.relative_to(ROOT)), seeds=list(SEEDS), folds=list(FOLDS),
        n_boot=N_BOOT, n_perm=N_PERM, rng_seed=20260921,
        groups=status.group.value_counts(dropna=False).to_dict()), indent=2, default=str))
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
