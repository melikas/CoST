# Disentangled self-supervised rhythm representations for wearable data

Self-supervised representations of multi-channel wearable time series, with the
trend/seasonal split of **CoST** re-targeted at *chronobiology*: circadian
amplitude, acrophase and rhythm regularity, and whether an unlabelled personal
baseline built from them detects a within-person rhythm deviation.

Two cohorts, one code path:

| | HRD | GLOBEM |
|---|---|---|
| signal | 4 minute-level sensor channels | 12 daily behavioural features |
| resolution | 15 min bins, 96/day | 6 h segments, 4/day |
| window | 7 days (`T=672`) | 28 days (`T=112`) |
| participants | 152 (114 labelled) | 702 (669 labelled) |

## Provenance

This repository is **derived from [salesforce/CoST](https://github.com/salesforce/CoST)**
(Woo et al., ICLR 2022, BSD 3-Clause), which in turn vendors the dilated-convolution
backbone and the instance-contrastive loss of
[TS2Vec](https://github.com/zhihanyue/ts2vec) (Yue et al., AAAI 2022, MIT).
`NOTICE` names every derived file and component; `third_party/` holds both licences.

Read that file before citing this work. What follows is the boundary between
what was inherited and what was added, stated so a reader can check it against
the code rather than take it on trust.

### What is upstream

The backbone (`models/dilated_conv.py`), the encoder and Fourier layer
(`CoSTEncoder`, `BandedFourierLayer`), the MoCo trend branch with its momentum
encoder and key queue, the within-batch seasonal contrastive term, and the
calendar covariates — all CoST or TS2Vec, modified but not invented here.

### What is ours

**Model.** Each of these is absent upstream and each is selectable, so the
contribution can be ablated rather than asserted:

| change | flag | why |
|---|---|---|
| Frequency-domain seasonal readout at the chronobiological harmonics — circaseptan, circadian, and its 2nd–4th harmonics | `--season-pool spec` | the seasonal branch is an irFFT, so time-averaging it returns exactly the `f=0` coefficient and every rhythm integrates to zero. Upstream pools over time. |
| Unit-circle phase embedding inside the seasonal loss, `φ → [sin φ; cos φ]` | `--phase-encoding circular` | the loss scores pairs with a dot product, and on raw `atan2` output that is not a similarity between angles: two identical phases score 0 at `φ=0` but `π²` at `φ=π`. |
| Fourier layer banded on the circadian harmonics, one sub-representation each | `--seasonal-bands harmonics` | upstream runs one band over the whole spectrum; at a 7-day window the harmonic bands cover bins 1-31 of 337, so the 90.8% above them is sub-6h content this project's hypothesis calls noise. |
| Positive pair drawn from the same participant rather than the same window | `--positive-pair participant` | measured: with the shipped pairing, top-1 retrieval on an *untrained* encoder is 3808× chance, so the objective starts solved and its gradient teaches nothing. |
| Subject-conditional negatives, with the count matched across modes | `--negatives subject` | isolates whether participant identity is the shortcut. Measured: it is not. |
| Trend term contrasting the pooled vector the probes actually read | `--trend-pool mean` | upstream contrasts one random timestep through a head that inference discards. |
| Decomposition-consistent views: share this window's seasonal component, swap the trend, resample the residual | `--decomp-aug` | makes "augmentation removes noise" literal instead of assumed. |
| Augmentation set re-derived for this domain | `--jitter-sigma` | `scale` removed outright: amplitude is the discriminative circadian feature, so contrasting scaled views would force amplitude-invariance and erase the signal. |

**Evaluation.** This is the larger contribution, and none of it exists upstream:

- **RQ1** — does the representation carry established cyclic and trend structure?
  Ridge read-out of cosinor amplitude, MESOR, acrophase and interdaily stability
  computed from the *raw* signal, out-of-fold and grouped by participant, against
  a PCA of the raw window **at the latent's own width** and against a random-init
  encoder of the same architecture.
- **RQ2** — can an unlabelled personal baseline detect a within-person rhythm
  deviation? A strictly causal personal baseline, a standardised-Euclidean
  deviation score, synthetic perturbations of known magnitude, and a stratified
  Mann-Whitney concordance against handcrafted, cosinor and raw references.
- **RQ3** — do the representations improve depression detection, how, and when
  does performance drop? A utility ladder from majority through handcrafted,
  cosinor, random-init, plain SSL, this model and an end-to-end supervised
  ceiling; branch and channel ablations; and degradation curves over duration,
  granularity, missingness and channel count.

**A random-init control on every rung.** The same architecture with weights
never trained, built by the same constructor and — since the layout is copied
from the encoder it is the control for — guaranteed to match it. This control is
rare in the SSL literature and is the reason several results here are negative.

**A difficulty gate for the pretext task** (`pretext_difficulty.py`): top-1
retrieval on an untrained encoder, measured before a sweep is submitted. A task
already solved at initialisation has no gradient to give, whatever its loss curve
looks like.

## Results, stated honestly

The measured findings, at n=24 seeds with architecture-matched controls:

- The representation beats a dimension-matched PCA of the raw window on **97.4%**
  of chronobiological markers, and recovers acrophase to ~15 minutes.
- The unlabelled deviation score reaches concordance **C=0.79** against a
  handcrafted baseline's 0.51.
- It does **not** beat its own random-init control at any of the three levels,
  and neither does an end-to-end supervised model with full label access. On this
  cohort the architecture's inductive bias, not the training, carries the signal.

The third point is a result, not an omission. Nothing here is configured to hide
it, and the control that produces it is part of the contribution.

## Running it

The pipeline is eight modules: `data_loader` → `model` + `objective` → `cost` → `train` →
`eval`, with `cv` supplying the protocol and `probe` the read-out.

```bash
pip install -r requirements.txt          # CosinorPy is needed for the cosinor baseline

# Resolve the whole run -- geometry, arms, folds, detectable margin -- before any GPU time.
python train.py --npz hrd_2224103.npz --out runs/oneshot --dry-run

# The pre-registered 4-arm run, locally
python train.py --npz hrd_2224103.npz --out runs/oneshot
```

The design is a 2×2: `phase_readout ∈ {angle, circular}` × objective weights ∈ `{paper,
contracted}`, over 10-fold × 3 repeats on all 114 labelled participants. It costs **60
encoder fits, not 120** — `phase_readout` is applied at encode time and never enters the
training path, so one encoder is read out both ways, which also makes that contrast exactly
paired.

`scripts/oneshot.sh` shards it across a SLURM array, one `(weights, fold)` pair per task:

```bash
sbatch --array=0,30%2  scripts/oneshot.sh    # stage 0 -- time it before committing
sacct -j <jobid> --format=JobID%20,State,Elapsed,MaxRSS
sbatch --array=0-59%12 scripts/oneshot.sh    # the full run

NPZ=globem_windows.npz DATASET=globem OUT=results_oneshot_globem \
    sbatch --array=0-59%12 scripts/oneshot.sh
```

Every task passes the same `--master-seed`; `cv.make_folds` is deterministic, so all tasks
reconstruct the same partition and `--only-fold` selects one. Changing it between tasks
voids every paired comparison. See `CLUSTER.md` for upload, monitoring and cleanup.

Evaluation shards the same way, and writes one publication-ready report:

```bash
python eval.py --run runs/oneshot --npz hrd_2224103.npz --only-fold r0f0   # per fold
python eval.py --run runs/oneshot --aggregate                              # results_summary.txt
```

RQ2 re-encodes phase-perturbed windows, so it needs the encoders, not just the stored
representations — run training with `--save-encoder` (`scripts/oneshot.sh` passes it) or
`--skip-rq2`.

`--sensor-csv` is not yet wired: cohort building from the raw CSV still lives in the
pre-cleanup `data_processing/`, so runs go through an `--npz` window cache.
`KEEP_ENC_ALL=1` keeps every task's encoder, which is what lets a readout or probe
question be re-tested later without retraining.

Participant data is **not** in this repository and is not redistributable.

## Licence

The added code has no licence file yet — choose one before publishing. Whatever
you choose, the BSD 3-Clause terms on the CoST-derived files travel with them:
the copyright notice and disclaimer must be retained, and Salesforce's name must
not be used to endorse this work. `NOTICE` and `third_party/` satisfy that as
long as they ship with the code.

## Citing

Cite CoST and TS2Vec for the method this builds on. See `NOTICE` for the exact
references.
