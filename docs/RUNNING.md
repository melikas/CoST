# Corrected RQ1–RQ3 experiments: design, outputs and running on Narval

The authoritative design is `docs/SCIENTIFIC_PROTOCOL.md` (including its dated execution
amendments). Historical entry points are not part of this protocol. **Every Narval command, in
order, is in [CLUSTER.md](../CLUSTER.md).**

## The model

`configs/hrd.json` and `configs/globem.json` instantiate the paper's DSSL, and
`tests/test_paper_model.py` pins it:

| Component | Setting |
|---|---|
| Backbone | Dilated TCN, depth = smallest receptive field covering the window: 7 on HRD (RF 1021 bins), 4 on GLOBEM (RF 125) |
| Width | output 320 (trend 160 + seasonal 160), hidden 64 |
| Trend branch | Causal convolution experts at kernels 1, 2, 4, …, T/8 (1–64 on HRD, 1–8 on GLOBEM); MoCo, queue 4096 |
| Seasonal branch | One banded Fourier layer per daily harmonic: HRD bins (1,10), (10,17), (17,24), (24,31); GLOBEM (1,42), (42,57) |
| Seasonal loss | Amplitude-weighted circular phase contrast; contracted weights w_T 0.277, w_A 0.148, w_Φ 0.852; α = 0.005 |
| Level equivariance | w_eq = 1 (trend readout predicts the per-channel level offset between views) |
| Augmentation | Jitter σ 0.1, level shift σ 0.5, circular smoothing up to 75 min (5 bins on HRD, off on GLOBEM's 6-h bins) |
| Optimisation | SGD, lr 5e-4 with cosine decay, batch 64, 6,000 updates; 10% of training windows only monitor the pretext loss |
| Readout | Trend time-mean, amplitude and angle at bins 1, 7, 14, 21, 28 (HRD) / 4, 28 (GLOBEM); 1,760 columns on HRD |
| Input | Within-person standardisation (each participant's full record, label-free), as in the reported runs |

The HRD encoder has 7,452,048 parameters with the rescue cache's 4 channels (7,451,984 with 3).
With the same seed it is bit-identical to the reported code (commit `0a88999`): initial
weights, augmented views, the loss and gradients on a fixed batch, and the encodings.

`normalization: "training_global"` in a config switches the input to per-channel moments of
the training participants. With `phase_readout: "circular"`, each (cos, sin) pair is scaled
isotropically in RQ1 and the logistic probe (`IsotropicPairScaler`), so the circle is not
stretched into an ellipse; the default angle readout has no pairs.

The **CoST reference adapter** is upstream CoST behind the same readout, trained on the same
data and budget: one full-spectrum seasonal band, trend kernels up to T/2, raw-phase contrast,
weights 1 / 0.5 / 0.5, no equivariance, scaling, jitter and shift at 0.5, no smoothing. It
takes only the shared settings from the config (`cost.REFERENCE_SHARED`). On HRD its encoder
has 44,036,288 parameters.

## What runs

The matrix is one variant (default TCN, no added temporal encoding) × HRD / GLOBEM-2018 ×
seeds 1, 2, 3 × five fixed participant folds = 15 tasks per stage and dataset. Two stages:

```text
reference  CoST reference adapter, 6,000 updates → results/<dataset>/<run>/cost_reference/
           seed_<s>/fold_<f>/  (trained once per seed × fold, reused by every variant)
variant    within-person input → Yan cosinor + random projection baselines → untrained encoder
           → DSSL, 6,000 updates (checkpoint every 100) → saved encoders and representations
           → RQ2 perturbation records (HRD) → RQ1 ridge recovery → RQ1 disentanglement audit
           → RQ3 logistic and forest probes → complete.json
```

A variant loads the reference for its seed × fold and refuses it if its data, settings or code
differ. After all 15 variant tasks succeed, a CPU job (`--summarize`) pools the out-of-fold
records, computes paired participant-bootstrap intervals, and writes tables, figures and
plain-language summaries. `slurm/submit.sh` chains the three steps. By default each GPU task
uses one `a100_3g.20gb` slice, 4 CPUs, 32 GB and at most 3 h (`TIME=` and `GPU=` override);
the GPU smoke job reports the measured time per update of both full-size models.

GLOBEM uses only its earliest cohort (2018), because cross-year identity linkage is unavailable.
RQ2 applies to HRD only: GLOBEM's overlapping 28-day windows and 6-hour bins cannot express the
weekly protocol, so it is reported as not applicable rather than given surrogate shifts.

## Evaluation ladder

Every feature set gets the identical participant-level probe (inner stratified participant folds
select the penalty; the test fold never selects anything).

| Rung | RQ1 | RQ2 | RQ3 | Definition |
|---|---|---|---|---|
| `training_prevalence` | | | ✓ | Training prevalence as every score |
| `training_mean` | ✓ | | ✓ | Population-mean model (RQ1); constant input (RQ3) |
| `raw` | ✓ | | ✓ | Flattened model-input window, participant mean |
| `pca` | ✓ | | ✓ | Training-only PCA of `raw` |
| `random_projection` | ✓ | ✓ | ✓ | Fixed 512-d N(0, 1/TC) map of the training-standardized window |
| `distribution` | | | ✓ | Physical-unit channel mean and SD |
| `nonparametric` | | | ✓ | IS, IV, activity RA |
| `yan_cosinor` | | | ✓ | Yan et al. periodogram cosinor (HRD reference paper), `tasks/yan_cosinor.py` |
| `handcrafted` | | | ✓ | Mean/SD, 24-h cosinor, IS, IV, RA together |
| `handcrafted_stack` | | | ✓ | NNLS super learner of distribution, nonparametric, Yan cosinor |
| `untrained` | ✓ | ✓ | ✓ | Identical encoder at initialization |
| `cost_reference_adapter` | ✓ | ✓ | ✓ | CoST reference adapter, same data/budget/readout |
| `dssl` | ✓ | ✓ | ✓ | Proposed method |

RQ3 is reported for the primary logistic probe and, separately, for the manuscript's secondary
random-forest ladder (AUROC only; never used for RQ3 claims). The GLOBEM benchmark algorithms
(Xu et al. 2022) are defined on the full RAPIDS feature set and scored on weekly labels under
the benchmark's own splits; `GLOBEM_REDUCED.csv` keeps 14 of those features, so they are listed
as not reproducible and no published score is inserted.

## Local check (CPU)

```bash
python -m unittest discover -s tests -t . -v
python scripts/run_experiment.py --dataset hrd --smoke --device cpu --fold 0
python scripts/run_experiment.py --dataset hrd --smoke --device cpu --fold 1
python scripts/run_experiment.py --dataset hrd --smoke --device cpu --summarize
```

(Same for `--dataset globem`.) Smoke runs use an execution-sized model and 2 updates, write to
`results/<dataset>/<run>_smoke_cpu/`, and are labelled as having no scientific meaning.

## What to open

Start with `results/SUMMARY_narval_v1.md`: every dataset and variant, one section per RQ, each
with its headline table and the protocol's pre-specified evidence conditions marked met / not met.

Per dataset and variant, `results/<dataset>/narval_v1/<backbone>_<encoding>/` (`tcn_none` by
default) holds:

| File | Content |
|---|---|
| `SUMMARY.md` | Plain-language summary of RQ1–RQ3 for this dataset |
| `REPORT.html` | Every table below plus all figures on one page |
| `RQ1_families.csv` | Error reduction of DSSL vs each control, per marker family (headline) |
| `RQ1_table.csv` | MAE of every method per marker × channel, DSSL gains vs population mean and untrained |
| `RQ1_comparisons.csv` | Every paired marker/channel comparison with its interval |
| `RQ1_disentanglement.csv` | Disentanglement audit: held-out R² of MESOR, 24-h amplitude and 24-h acrophase from the own branch vs the other branch, for DSSL, untrained and CoST reference (per channel: `rq1_disentanglement_channels.csv`) |
| `RQ1_disentanglement_comparisons.csv` | DSSL minus each control in own − leakage, with intervals |
| `RQ2_table.csv` | Timing and strength concordance per method, vs chance |
| `RQ2_comparisons.csv` | DSSL minus each control, per perturbation |
| `RQ3_table.csv` | AUROC (mean, seed SD), balanced accuracy, macro-F1, DSSL minus each rung |
| `RQ3_forest_table.csv` | Secondary random-forest ladder |
| `baseline_coverage.csv` | Every rung and each external benchmark's status |
| `rq1_families.png`, `rq1_recovery.png`, `rq2_personalized.png`, `rq3_auroc.png`, `secondary_endpoint_associations.png` | Figures |
| `oof_predictions.csv`, `summary_by_seed.csv`, `rq1_errors.csv`, `rq*_intervals.csv`, `rq2_*`, `rq3_exploratory_bridge.json`, `secondary_endpoint_associations.csv` | Machine-readable records behind the tables |

Each `seed_<s>/fold_<f>/` holds `manifest.json` (settings, split IDs, data/code hashes, git
commit, normalization, versions), `dssl_encoder.pt`, `dssl_training.pt`, `dssl_model.json`,
`representations.npz` (untrained, DSSL, CoST reference, random projection, Yan cosinor features
with participant and window IDs), `rq1_recovery.csv`, `rq2_personalized.csv`, `rq2_status.json`,
`rq3_predictions.csv`, `probe_selection.csv` (including the stack weights), `metrics.json`,
`target_definition.json`, `secondary_endpoint_markers.csv` and `complete.json`. The reference's
own encoder, manifest and representations sit under `cost_reference/seed_<s>/fold_<f>/`.

Reading rules: intervals are paired participant-bootstrap 95% intervals conditional on the fitted
models and are not simultaneous; "inconclusive" means the interval includes zero, not equivalence;
timing and strength are both required for RQ2; RQ3 requires improvement over both raw and untrained.

## Active files

| File | Purpose |
|---|---|
| `scripts/run_experiment.py` | Single training/evaluation entry point (`--reference`, `--summarize`) |
| `evaluation_protocol.py` | Participant probes, circular pair scaler, ladder, statistics, figures |
| `result_report.py` | Tables, `SUMMARY.md`, `REPORT.html`, combined summary |
| `cost.py` | DSSL and the CoST reference adapter: objective, training, readout |
| `models/encoder.py`, `models/backbones.py`, `models/dilated_conv.py`, `models/losses.py` | Encoder heads, backbone registry (TCN, Transformer, Mamba), TCN, loss terms |
| `tasks/yan_cosinor.py` | Yan et al. cosinor baseline (CosinorPy) |
| `tasks/personalized.py`, `tasks/rhythm.py` | RQ2 personal baselines/perturbations; rhythm markers |
| `tasks/projection.py` | Training-fitted random projection |
| `datautils.py`, `scripts/build_cache.py`, `data_processing/` | Caches, validation, participant folds |
| `configs/hrd.json`, `configs/globem.json` | The paper model, budgets and split settings |
| `scripts/stamp_version.py`, `scripts/time_steps.py` | Commit stamp for the cluster copy; measured time per update |
| `slurm/setup_env.sh`, `slurm/env.sh` | Narval environment |
| `slurm/smoke.sbatch`, `slurm/rq123.sbatch`, `slurm/summarize.sbatch`, `slurm/submit.sh` | Jobs |
| `tests/` | Unit tests, including the paper-model conformance test |
| `docs/audit/` | Audit record |
