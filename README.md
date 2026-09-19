# DSSL rhythmicity research 

This repository studies whether CoST-derived self-supervised wearable representations
retain **individual rhythmic characteristics** and support mental-health endpoints.

**Start with [the full report](REPORT_TO_PROFESSOR.md).** It is the single authoritative
document: the objective, the data, the final architecture layer by layer, the RQ1–RQ3
results, how the model was chosen, the limitations and exact reproduction commands.
For day-to-day commands see [the experiment guide](docs/RUNNING.md).

- [**Full report — architecture, results, reproduction**](REPORT_TO_PROFESSOR.md)
- [**Experiments tested and not retained**](FAILED_EXPERIMENTS.md)
- [Scientific protocol and adjudication](docs/SCIENTIFIC_PROTOCOL.md)
- [Pre-registration of the canonical run](docs/VALIDATION_PRECOMMIT.md) and [its outcome](docs/VALIDATION_RESULT.md)
- [**Where every output comes from**](docs/OUTPUT_PROVENANCE.md) — the single results location, what `manifest.json` records, and the one upload/download procedure
- [Consolidation manifest](docs/CONSOLIDATION_MANIFEST.md) — what was removed and why
- [Inventory of removed legacy outputs](docs/REMOVED_LEGACY_OUTPUTS.md)
- [Delivery status, RQ table and dataset audit](docs/RESCUE_STATUS.md)
- [Repository discrepancy audit](docs/audit/README.md)
- [Applied repairs](docs/audit/REPAIRS.md)
- [Corrected manuscript](SSL_Rhythmicity/Main.tex); the dated methods/RQ amendments live in [the scientific protocol](docs/SCIENTIFIC_PROTOCOL.md)

The active entry point is `scripts/run_experiment.py`. It trains each fold once and
evaluates RQ1–RQ3 from the same frozen features. `configs/hrd.json` and
`configs/globem.json` are the **one canonical configuration per dataset** (harmonic bands,
trend experts up to T/8, contracted weights, raw seasonal readout, within-person input);
`--backbone` and
`--temporal-encoding` select other variants, each written to its own
`results/<dataset>/<run>/<backbone>_<encoding>/`. RQ1 includes a disentanglement audit: each
window's MESOR, 24-h amplitude and acrophase from its own branch versus the other branch. The CoST reference adapter is trained
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
