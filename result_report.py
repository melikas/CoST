"""Human-readable RQ tables and summaries from saved results; no training or hard-coded study scores."""
from pathlib import Path
import html
import json
import numpy as np
import pandas as pd

# The evaluation ladder, in reading order: (method, category, definition).
LADDER = [
    ('training_prevalence', 'Trivial', 'Training prevalence as every score; majority decision at 0.5'),
    ('training_mean', 'Trivial', 'Constant input: population-mean model (RQ1), uninformative probe (RQ3)'),
    ('raw', 'Primary control', 'Participant mean of the flattened, training-normalized window'),
    ('pca', 'Compression', 'Training-only PCA of raw; achieved width in each fold metrics.json'),
    ('random_projection', 'Compression', 'Fixed 512-d Gaussian map of the training-standardized window'),
    ('distribution', 'Handcrafted', 'Physical-unit channel mean and SD'),
    ('nonparametric', 'Handcrafted', 'Interdaily stability, intradaily variability, activity RA'),
    ('yan_cosinor', 'Reference paper (HRD)', 'Yan et al. periodogram cosinor: 12 parameters x top-2 periods x channel'),
    ('handcrafted', 'Handcrafted', 'Mean/SD, 24-h cosinor, IS, IV and RA together'),
    ('handcrafted_stack', 'Handcrafted', 'NNLS super learner of distribution, nonparametric and Yan cosinor'),
    ('untrained', 'Primary control', 'Identical encoder at initialization (architecture without SSL)'),
    ('cost_reference_adapter', 'Published SSL method', 'CoST reference adapter: same data, budget and readout'),
    ('supervised', 'Supervised control', 'Same encoder and readout trained end to end on labelled training participants only'),
    ('dssl', 'Proposed', 'DSSL frozen representation'),
]
EXTERNAL = [
    ('Yan et al. 2022 cosinor (HRD paper, Zhao et al. 2025)', 'computed as yan_cosinor',
     'Rhythm model reproduced on these windows; the paper compares group rhythms and reports no classifier score to compare'),
    ('GLOBEM benchmark (Xu et al. 2022): Canzian, Saeb, Farhan, Wahle, Lu, Wang, Xu 2019/2021, Chikersal; '
     'ERM, Mixup, DANN, IRM, CSD, MLDG, MASF, Siamese, Reorder', 'NOT REPRODUCIBLE FROM THIS EXPORT',
     'Defined on the full RAPIDS feature set (incl. calls) and scored on weekly labels under the '
     "benchmark's own splits; GLOBEM_REDUCED.csv keeps 14 of those features at four day segments, "
     'so run here they would be different methods, and RQ3 predicts the end-point label. '
     'Published scores are not inserted'),
    ('Archived fusion, [agg], [dev] and supervised ladders', 'NOT IN PROTOCOL',
     'The governing protocol excludes unrestricted probe or feature-fusion searches'),
]
FAMILIES = ('amplitude', 'phase_hours', 'IS', 'IV', 'RA')        # MESOR is supplementary
PERTURBATION = {'phase': 'timing (0.5-4 h shifts)', 'amplitude': 'strength (24-h amplitude x(1±α))'}
# RQ1 disentanglement: target -> (label, own branch, leakage branch); see evaluation_protocol.BRANCHES.
DISENTANGLEMENT = {'MESOR': ('MESOR (window mean)', 'trend', 'seasonal amplitude + phase'),
                   'amplitude': ('24-h amplitude', 'seasonal amplitude', 'trend'),
                   'acrophase': ('24-h acrophase', 'seasonal phase', 'trend')}
DISENTANGLED = ('dssl', 'untrained', 'cost_reference_adapter', 'supervised')
NOTES = ('Intervals are paired participant-bootstrap 95% intervals, conditional on the fitted models '
         '(no retraining uncertainty) and not simultaneous across rows. "inconclusive" means the '
         'interval includes 0: it is not evidence of equivalence. Missing estimates are unavailable, '
         'not zero. Conditions restate the protocol\'s pre-specified evidence requirements; meeting '
         'them does not by itself establish a clinical or causal claim.')


def verdict(low, high, positive='favours DSSL', negative='favours control'):
    if pd.isna(low) or pd.isna(high):
        return 'unavailable'
    return positive if low > 0 else negative if high < 0 else 'inconclusive (CI includes 0)'


def estimate(difference, low, high):
    return 'unavailable' if pd.isna(difference) else f'{difference:+.3f} [{low:+.3f}, {high:+.3f}]'


def markdown(frame):
    text = frame.copy()
    for column in text:
        if text[column].dtype.kind == 'f':
            text[column] = text[column].map(lambda v: 'unavailable' if pd.isna(v) else f'{v:.3f}')
    text = text.fillna('unavailable').astype(str)
    return '\n'.join(['| ' + ' | '.join(map(str, text.columns)) + ' |', '|' + '---|' * len(text.columns)]
                     + ['| ' + ' | '.join(row) + ' |' for row in text.to_numpy()])


def condition(label, low):
    return f'- {label}: **{"met" if pd.notna(low) and low > 0 else "not met"}**'


def ladder_sort(frame, column='method'):
    rank = {m: i for i, (m, *_) in enumerate(LADDER)}
    return frame.sort_values(column, key=lambda s: s.map(rank), kind='stable').reset_index(drop=True)


def write_report(root, smoke=False):
    root = Path(root)
    dataset, run = root.parents[1].name, root.parent.name
    tables = []
    # ---- RQ1
    errors = pd.read_csv(root/'rq1_errors.csv')
    cells, families = pd.read_csv(root/'rq1_paired_intervals.csv'), pd.read_csv(root/'rq1_family_intervals.csv')
    for frame in (cells, families):
        frame['evidence'] = [verdict(a, b) for a, b in zip(frame.low, frame.high)]
        frame['DSSL gain [95% CI]'] = [estimate(*v) for v in zip(frame.difference, frame.low, frame.high)]
    mae = (errors.groupby(['method','marker','channel','seed']).error.mean()
           .groupby(['method','marker','channel']).mean().unstack('method'))
    rq1 = mae[[m for m, *_ in LADDER if m in mae.columns]].add_prefix('MAE ').reset_index()
    index = pd.MultiIndex.from_frame(rq1[['marker','channel']])
    rq1.insert(2, 'unit', np.where(rq1.marker == 'phase_hours', 'hours', 'marker units'))
    rq1.insert(3, 'participants', errors.groupby(['marker','channel']).participant.nunique().reindex(index).to_numpy())
    for control in ('training_mean', 'untrained'):
        c = cells[cells.control == control].set_index(['marker','channel']).reindex(index)
        rq1[f'DSSL gain vs {control} [95% CI]'] = c['DSSL gain [95% CI]'].fillna('unavailable').to_numpy()
        rq1[f'evidence vs {control}'] = c.evidence.fillna('unavailable').to_numpy()
    family_table = families.assign(result=families['DSSL gain [95% CI]'] + ' ' + families.evidence).pivot(
        index='marker', columns='control', values='result')
    family_table = family_table.reindex(index=[f for f in (*FAMILIES, 'MESOR') if f in family_table.index],
                                        columns=[m for m, *_ in LADDER if m in family_table.columns]).reset_index()
    tables += [('RQ1 — Marker families: control error minus DSSL error (positive favours DSSL)', 'RQ1_families.csv', family_table),
               ('RQ1 — Individual rhythm recovery per marker and channel (MAE, lower is better)', 'RQ1_table.csv', rq1),
               ('RQ1 — All paired marker/channel comparisons', 'RQ1_comparisons.csv', cells)]
    # ---- RQ1 disentanglement audit: own branch vs leakage per target (held-out R², floored at 0).
    ent = pd.read_csv(root/'rq1_disentanglement_intervals.csv').set_index(['method', 'target', 'quantity'])

    def cell(method, target, quantity):
        return ent.loc[(method, target, quantity)] if (method, target, quantity) in ent.index else None

    audit, versus = [], []
    for target, (label, own, leak) in DISENTANGLEMENT.items():
        for method in DISENTANGLED:
            gap = cell(method, target, 'own_minus_leakage')
            if gap is None:
                continue
            hours = [cell(method, target, f'{r}_error_hours') for r in ('own', 'leakage')]
            audit.append({'target': label, 'method': method, 'own branch': own,
                          'own R²': cell(method, target, 'own').estimate,
                          'leakage branch': leak, 'leakage R²': cell(method, target, 'leakage').estimate,
                          'own − leakage [95% CI]': estimate(gap.estimate, gap.low, gap.high),
                          'evidence': verdict(gap.low, gap.high, 'separated', 'entangled'),
                          'acrophase error own / leakage (h)': ' / '.join(f'{h.estimate:.2f}' for h in hours)
                          if all(h is not None for h in hours) else ''})
            d = cell(method, target, 'dssl_minus_method')
            if d is not None:
                versus.append({'target': label, 'comparison': f'dssl − {method}',
                               'difference in own − leakage [95% CI]': estimate(d.estimate, d.low, d.high),
                               'evidence': verdict(d.low, d.high)})
    audit, versus = pd.DataFrame(audit), pd.DataFrame(versus)
    tables += [('RQ1 — Disentanglement audit: each target from its own branch vs the other branch (held-out windows)',
                'RQ1_disentanglement.csv', audit),
               ('RQ1 — Disentanglement: DSSL minus each control', 'RQ1_disentanglement_comparisons.csv', versus)]
    # ---- RQ2
    rq2, rq2i = pd.read_csv(root/'rq2_summary.csv'), pd.read_csv(root/'rq2_paired_intervals.csv')
    rq2i['evidence'] = [verdict(r.low, r.high, 'above chance', 'below chance') if r.comparison.endswith('chance')
                        else verdict(r.low, r.high) for r in rq2i.itertuples()]
    rq2i['estimate [95% CI]'] = [estimate(*v) for v in zip(rq2i.difference, rq2i.low, rq2i.high)]
    if len(rq2):
        rq2t = rq2.groupby(['perturbation','method']).agg(
            concordance=('concordance','mean'), seed_SD=('concordance','std'),
            participants=('participants','max'), comparisons=('comparisons','sum')).reset_index()
        chance = rq2i[rq2i.comparison.str.endswith(' - chance')].assign(
            method=lambda f: f.comparison.str.replace(' - chance', '', regex=False))
        rq2t = rq2t.merge(chance[['perturbation','method','estimate [95% CI]','evidence']], how='left').rename(
            columns={'estimate [95% CI]': 'concordance - 0.5 [95% CI]', 'evidence': 'vs chance'})
        rq2t = ladder_sort(rq2t).sort_values('perturbation', kind='stable', ascending=False)
        rq2t['perturbation'] = rq2t.perturbation.map(PERTURBATION)
    else:
        rq2t = pd.DataFrame([dict(status='NOT APPLICABLE', reason='Overlapping 28-day windows and 6-h bins cannot '
                                  'express the declared weekly HRD protocol; no surrogate shifts are generated')])
    tables += [('RQ2 — Personalized rhythmic phenotyping (chance = 0.5)', 'RQ2_table.csv', rq2t),
               ('RQ2 — Paired participant comparisons', 'RQ2_comparisons.csv', rq2i)]
    # ---- RQ3
    by_seed, rq3i = pd.read_csv(root/'summary_by_seed.csv'), pd.read_csv(root/'rq3_paired_intervals.csv')

    def rq3_table(probe, secondary=False):
        s = by_seed[by_seed.probe == probe].groupby('method')
        i = rq3i[rq3i.probe == probe].set_index('control')
        out = []
        for method, category, definition in LADDER:
            if method not in s.groups:
                continue
            g = s.get_group(method)
            row = dict(method=method, category=category, AUROC=g.auroc.mean(), AUROC_seed_SD=g.auroc.std())
            if not secondary:
                row.update(balanced_accuracy=g.balanced_accuracy.mean(), macro_F1=g.macro_f1.mean())
            if method in i.index:
                r = i.loc[method]
                row.update({'DSSL minus method [95% CI]': estimate(r.difference, r.low, r.high),
                            'evidence': verdict(r.low, r.high)})
            out.append(row | ({} if secondary else dict(definition=definition)))
        return pd.DataFrame(out)

    rq3 = rq3_table('logistic')
    tables += [('RQ3 — Participant endpoint prediction, primary logistic probe', 'RQ3_table.csv', rq3),
               ('RQ3 — Secondary ladder: random-forest probe (AUROC only; not used for RQ3 claims)',
                'RQ3_forest_table.csv', rq3_table('forest', secondary=True))]
    coverage = pd.DataFrame([(m, c, d, 'computed') for m, c, d in LADDER] + [(n, 'External', d, s) for n, s, d in EXTERNAL],
                            columns=['method','category','definition','status'])
    tables.append(('Baseline and evaluation-ladder coverage', 'baseline_coverage.csv', coverage))
    for _, filename, table in tables:
        table.to_csv(root/filename, index=False)
    # ---- Plain-language summary with the pre-specified evidence conditions.
    title = (f'EXECUTION SMOKE — no scientific conclusions ({dataset})' if smoke
             else f'{dataset.upper()} — DSSL {root.name} RQ1–RQ3 results ({run})')
    fam = families.set_index(['marker','control'])
    eligible = [f for f in FAMILIES if (f, 'untrained') in fam.index]
    lines = [f'# {title}', '', NOTES, '', f'Evaluated participants (RQ3): {rq3i.n.max() if len(rq3i) else "unavailable"}; '
             f'seeds x folds: {by_seed.seed.nunique()} x {len(set(pd.read_csv(root/"oof_predictions.csv").fold))}.', '',
             '## RQ1 — Does SSL preserve individual rhythms beyond the population pattern and random architecture?', '',
             'Error reduction of DSSL relative to each control, per marker family (channels averaged within person).', '',
             markdown(family_table), '',
             condition('DSSL beats the population mean in every eligible family (' + ', '.join(eligible) + ')',
                       min([fam.loc[(f, 'training_mean'), 'low'] for f in eligible if (f, 'training_mean') in fam.index]
                           or [np.nan]) if all((f, 'training_mean') in fam.index for f in eligible) else np.nan),
             condition('DSSL beats the untrained encoder in every eligible family',
                       min([fam.loc[(f, 'untrained'), 'low'] for f in eligible] or [np.nan])),
             '', 'Per-channel detail: `RQ1_table.csv`, `rq1_recovery.png`, `rq1_families.png`.', '',
             '### Disentanglement audit (window level, held-out participants)', '',
             'Every held-out window\'s MESOR, 24-h amplitude and 24-h acrophase is predicted from its own branch '
             'and, as leakage, from the other branch (ridge; held-out R², floored at 0; channels averaged, then '
             'seeds). "separated" means own minus leakage is above 0 with its 95% CI.', '',
             markdown(audit), '', markdown(versus), '',
             condition('DSSL: own branch above leakage for MESOR, 24-h amplitude and 24-h acrophase',
                       np.min([getattr(cell('dssl', t, 'own_minus_leakage'), 'low', np.nan) for t in DISENTANGLEMENT])),
             condition('DSSL separates the branches better than the untrained encoder on all three targets',
                       np.min([getattr(cell('untrained', t, 'dssl_minus_method'), 'low', np.nan) for t in DISENTANGLEMENT])),
             '', 'Per-channel R²: `rq1_disentanglement_channels.csv`.', '',
             '## RQ2 — Do unlabeled personal baselines detect within-person rhythmic deviations?', '']
    if len(rq2):
        def low(comparison, perturbation):
            r = rq2i[(rq2i.comparison == comparison) & (rq2i.perturbation == perturbation)]
            return r.low.iloc[0] if len(r) else np.nan
        compare = rq2i[rq2i.comparison.str.startswith('dssl - ') & ~rq2i.comparison.str.endswith('chance')]
        lines += [markdown(rq2t[['perturbation','method','concordance','seed_SD','participants',
                                 'concordance - 0.5 [95% CI]','vs chance']]), '',
                  markdown(compare.assign(perturbation=compare.perturbation.map(PERTURBATION))[
                      ['perturbation','comparison','n','estimate [95% CI]','evidence']]), '']
        lines += [condition(f'DSSL {PERTURBATION[p]} concordance above chance', low('dssl - chance', p)) for p in PERTURBATION]
        lines += [condition(f'DSSL beats the untrained encoder on {PERTURBATION[p]}', low('dssl - untrained', p)) for p in PERTURBATION]
        lines += ['', 'Timing and strength are both necessary; one arm cannot compensate for the other.']
    else:
        lines += ['**Not applicable** for this dataset: ' + rq2t.reason.iloc[0] + '.']
    primary = rq3i[rq3i.probe == 'logistic'].set_index('control')
    lines += ['', '## RQ3 — Does SSL add endpoint discrimination?', '',
              markdown(rq3[[c for c in rq3.columns if c != 'definition']]), '',
              condition('DSSL AUROC above raw', primary.low.get('raw', np.nan)),
              condition('DSSL AUROC above untrained encoder', primary.low.get('untrained', np.nan)),
              '', 'Both primary conditions are required (a conjunction). Other rows contextualize the result.',
              'Secondary random-forest ladder: `RQ3_forest_table.csv`.']
    bridge = [b for b in json.loads((root/'rq3_exploratory_bridge.json').read_text()) if b['method'] == 'dssl']
    if bridge:
        lines += ['', 'Exploratory bridge (DSSL; Spearman of rhythm-recovery error with log loss, by endpoint class; '
                  'observational only): ' + '; '.join(f'class {b["label"]}: rho {b["spearman"]:+.3f} (n={b["n"]})' for b in bridge)]
    lines += ['', '## Baselines', '', markdown(coverage), '', '## Figures', '',
              '| Figure | Question it answers | How to read it |',
              '|---|---|---|',
              '| `rq1_recovery.png` | How large is the rhythm-recovery error, per marker family? | Bars are mean error, lower is better; phase in hours, other markers in training-SD units (`rq1_absolute_error.csv`) |',
              '| `rq1_families.png` | Does DSSL recover markers better than each control? | Right of the zero line favours DSSL; bars are paired 95% CIs (`RQ1_families.csv`) |',
              '| `rq1_disentanglement.png` | Does each rhythm property come from its own branch? | Solid = own branch, hatched = the other branch; solid should exceed hatched (`RQ1_disentanglement.csv`) |',
              '| `rq2_personalized.png` | Does the representation move with the size of a known rhythm change? | Concordance vs perturbation size; above the 0.5 chance line and rising is correct (`rq2_by_level.csv`) |',
              '| `rq3_auroc.png` | Which representation predicts the endpoint, and is DSSL ahead? | Left: AUROC per method, points are seeds; right: DSSL minus each method with 95% CI (`RQ3_table.csv`) |',
              '| `secondary_endpoint_associations.png` | Which raw rhythm markers differ by endpoint group? | Red bars are Holm-significant; negative = lower in the endpoint-positive group (`secondary_endpoint_associations.csv`) |',
              '', 'Full tables: `REPORT.html`.']
    (root/'SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    content = [f'<h1>{html.escape(title)}</h1>', f'<p>{html.escape(NOTES)}</p>']
    for heading, filename, table in tables:
        content += [f'<h2>{html.escape(heading)}</h2><p><code>{filename}</code></p>',
                    table.to_html(index=False, float_format=lambda v: f'{v:.4f}', na_rep='unavailable', escape=True)]
    content.append('<h2>Figures</h2>')
    for name in ['rq1_recovery.png', 'rq1_families.png', 'rq1_disentanglement.png',
                 'rq2_personalized.png', 'rq3_auroc.png', 'secondary_endpoint_associations.png']:
        if (root/name).exists():
            content.append(f'<p>{name}</p><img src="{name}" style="max-width:100%">')
    (root/'REPORT.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font-family:sans-serif;'
                                    'max-width:1500px;margin:30px auto}td,th{padding:5px;text-align:left}'
                                    'table{border-collapse:collapse;font-size:13px}</style>' + ''.join(content), encoding='utf-8')
    # One file for every dataset of this run, rebuilt whenever a dataset is summarized.
    summaries = sorted(p for p in root.parents[2].glob(f'*/{run}/*/SUMMARY.md')
                       if p.parent.name != 'cost_reference')
    (root.parents[2]/f'SUMMARY_{run}.md').write_text('\n\n---\n\n'.join(
        p.read_text(encoding='utf-8') for p in summaries), encoding='utf-8')
