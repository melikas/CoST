# DSSL rhythmicity research 

This repository studies whether CoST-derived self-supervised wearable representations
retain **individual rhythmic characteristics** and support mental-health endpoints.

**Start with [the experiment guide](docs/RUNNING.md).** It specifies supported commands,
the model matrix, output files, cluster submission and remaining scientific limitations.

- [Scientific protocol and adjudication](docs/SCIENTIFIC_PROTOCOL.md)
- [Delivery status, RQ table and dataset audit](docs/RESCUE_STATUS.md)
- [Repository discrepancy audit](docs/audit/README.md)
- [Applied repairs](docs/audit/REPAIRS.md)
- [Corrected manuscript](SSL_Rhythmicity/Main.tex) and [dated methods/RQ amendment](SSL_Rhythmicity/sections/rescue-protocol.tex)

The active entry point is `scripts/run_experiment.py`. It trains each fold once and
evaluates RQ1–RQ3 from the same frozen features. `configs/hrd.json` and
`configs/globem.json` instantiate the paper's DSSL (harmonic bands, trend experts up to
T/8, contracted weights, level equivariance, within-person input); `--backbone` and
`--temporal-encoding` select other variants, each written to its own
`results/<dataset>/<run>/<backbone>_<encoding>/`. The CoST reference adapter is trained
once per seed × fold and shared by every variant.

```bash
python -m unittest discover -s tests -t . -v
python scripts/run_experiment.py --dataset hrd --smoke --device cpu --fold 0
python scripts/run_experiment.py --dataset hrd --smoke --device cpu --fold 1
python scripts/run_experiment.py --dataset hrd --smoke --device cpu --summarize
```

Repeat with `--dataset globem`. Scientific runs use three seeds and five fixed
participant folds, with a separately named CoST reference adapter, untrained, raw, PCA,
handcrafted and trivial controls. GLOBEM initially uses **2018 only** because cross-year
person linkage is unavailable. HRD steps and all 14 GLOBEM channels are preserved.

The manuscript contains no claimed results from the corrected experiment. RQ1 initially
tests individual profiles; RQ2 implements unlabeled personal baselines and controlled
within-person rhythm deviations on HRD. Endpoint associations remain secondary, while
natural emotional-energy validation remains pending, as do label-derivation verification
and Narval GPU execution. These limitations must accompany scientific conclusions.

Historical source and unique results are preserved under Git, the rescue archive, and
their original result directories. Historical scores are exploratory and must not be
pooled with `results/<dataset>/rescue_v1/`. See the guide before starting GPU jobs.
