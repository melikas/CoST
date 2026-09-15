# Essential rescue deliverables

The corrected initial experiment is executable and preserves the original scientific
intent. The entire PhD study is **not** declared complete: dynamic/natural-energy
analyses, clinical label provenance and actual Narval GPU execution remain unresolved.
Use [RUNNING.md](RUNNING.md) for commands and artifacts, and the
[audit](audit/README.md) and [repair report](audit/REPAIRS.md) for source-level evidence.

## Scientific design

DSSL learns window-instance representations under two augmented views. A TCN feeds
multiscale trend convolutions and a full-spectrum seasonal transformation. The loss is
`trend_InfoNCE + 0.005 * (amplitude_contrast + phase_contrast) / 2`.
The trend term uses a momentum queue; seasonal positives are corresponding views and
negatives are other batch windows. The phase loss uses circular geometry in DSSL.
The reference adapter retains original raw-angle contrast and augmentation policy.

Our wearable readout uses mean trend features and seasonal amplitudes/circular phases
at resolvable weekly/daily harmonics. Participants receive mean window features.
This creates a structural reason to test for rhythmic information, not proof that
the representation is identifiable, disentangled, clinically valid, or causal.
RQ1 provides the recovery test; RQ2 tests unlabeled within-person personal-baseline
deviation detection; RQ3 tests downstream usefulness.

## RQ table

The early manuscript commit `0120be4` does not contain the complete numbering; the author
explicitly confirmed the governing original RQ2 wording used below. The active manuscript
is [Main.tex](../SSL_Rhythmicity/Main.tex), which includes the dated corrected protocol
and excludes the superseded result claims from compilation.

| RQ | Hypothesis | Initial experiment | Baseline/control | Metric | Statistical comparison | Supporting evidence |
|---|---|---|---|---|---|---|
| 1 | SSL preserves individual rhythms beyond a population pattern and random architecture | Held-out participant-profile recovery of amplitude, phase, RA, IS, IV; supplementary MESOR | Population mean, untrained encoder, raw, training-fitted PCA, random projection, reference adapter | Per-marker/channel MAE; circular phase hours | Paired participant bootstrap of error differences; descriptive, not simultaneous intervals | Positive error reduction across the original eligible marker families; one favorable channel is insufficient |
| 2 | Unlabeled personal baselines detect within-person rhythmic deviations | Held-out HRD weeks compared with each person's four preceding contiguous weeks; controlled phase shifts and paired 24-hour amplitude changes | Untrained encoder, random projection and CoST reference adapter on identical weeks/perturbations; raw cosinor deviation supplies ordering ground truth | Timing and strength concordance, overall and per participant | Paired participant bootstrap of DSSL-minus-control concordance | Above-chance concordance and improvement over untrained for both timing and strength; one arm cannot compensate for failure of the other |
| 3 | SSL adds endpoint discrimination and preservation accompanies prediction quality | Identical frozen participant logistic probes; exploratory recovery-error versus log-loss association | Raw, untrained, PCA, random projection, distribution, nonparametric, Yan et al. cosinor (HRD reference paper), handcrafted, NNLS handcrafted stack, reference adapter, training-prevalence decision; secondary random-forest ladder. Published GLOBEM benchmark algorithms are not reproducible from the export | Primary AUROC; balanced accuracy and macro-F1 at 0.5; exploratory class-stratified Spearman | Paired stratified participant bootstrap averaged across seeds; bridge reported separately | Improvement against both raw and untrained; positive error–error relationship is only observational supporting evidence |

Endpoint-group rhythm associations are retained as a secondary analysis, not RQ2.
Failure to obtain this evidence leaves the corresponding claim unsupported. Failure
to reject zero is not equivalence. No score-selected margin, channel, seed or
configuration can redefine success. Controlled RQ2 does not by itself establish
natural emotional or clinical change; within-person recovery under RQ1 remains separate.

## Dataset audit at delivery

| Property | HRD | GLOBEM |
|---|---|---|
| Raw/export identifiers | 166 | 705 participant-year IDs, not verified unique people |
| Corrected cache | 151 IDs, 3,803 windows | 702 participant-year IDs, 6,804 windows |
| Labelled cache | 113, including 51 positive | 669, including 269 positive |
| Input channels | HR, Steps, sleep indicator, screen | All 14 exported activity/sleep/screen/location/Bluetooth/Wi-Fi features |
| Resolution/window | 15 minutes; 7 days; 672 bins | 6 hours; 28 days; 112 bins; 7-day stride |
| Missingness | Full elapsed minute grid; only short interior runs interpolated; original observation masks retained | Existing within-person completion retained; original per-channel masks retained; heavily missing sleep targets explicitly unavailable |
| Model normalization | Observed training-participant channel moments, equal person weight | Same policy, within the permitted cohort |
| Rhythm targets | Completed physical signals; ≥70% original observed coverage; activity-only RA | Same coverage rule; standard RA unavailable at 6-hour resolution |
| Split unit | Participant; RQ2 uses only held-out participant histories | Participant within 2018 only; all other years excluded from SSL and probes |
| Endpoint | Exported binary value; unknown is −1 | Exported binary value; unknown is −1 |
| Unresolved | Instrument/cutoff/timing and timezone provenance; retrospective completion | Same, plus cross-year identity linkage |

The run manifest records the exact experiment IDs, normalization, source hashes,
cohort restriction and actual software versions. Historical and new cache counts
are compared in `audit/repair_validation.json`. No raw datasets were deleted.

## Matrix and readiness

TCN × no positional encoding × HRD / GLOBEM-2018 × seeds {1,2,3} × five fixed folds.
Two trained methods per seed × fold: the paper's DSSL and the CoST reference adapter,
the latter trained once and shared by every backbone variant. The shared budget is
6,000 updates, batch 64, width 320, hidden width 64, TCN depth covering the window
(7 on HRD, 4 on GLOBEM), SGD/cosine learning rate 5e-4. RQ2 runs on HRD only; GLOBEM is
explicitly not applicable to its non-overlapping weekly protocol. No RQ4 sweep or large
job was run.

The tested local environment is recorded in `audit/runtime_versions.json`.
`tests/` holds 27 unit tests: preprocessing, split isolation, rhythm definitions,
finite optimization, checkpoint round trips, exact training continuation, the backbone
registry, the circular pair scaler, statistics, held-out label isolation, and the
paper-model conformance test. Both datasets have a separate two-fold CPU smoke run,
including the reference stage, structured predictions, aggregation and saved-data
figures. These are execution artifacts, not scientific results.

Follow [RUNNING.md](RUNNING.md): `slurm/setup_env.sh` once, the GPU smoke job
`slurm/smoke.sbatch`, then `slurm/submit.sh`, which chains both arrays to automatic summaries.
Bash syntax is checked; Narval submission and full-budget GPU execution have not been performed here. No local TeX compiler was available, so the
updated manuscript source still needs compilation in Overleaf/TeX Live.

## Cleanup boundary

The pre-repair source snapshot, original staged/unstaged patches and archive manifest
preserve prior work. Superseded launchers and evaluation ladders are archived, not
silently destroyed. Unique historic results/checkpoints and every raw dataset remain.
Low-priority refactoring and deletion of large historical result trees were deliberately
avoided. The supported workflow is small and documented even though historical research
artifacts remain available for provenance.
