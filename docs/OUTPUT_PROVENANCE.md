# Where every output in this repository comes from

Written 2026-09-18, after an audit found that experiment outputs were scattered across four
different top-level locations with no consistent rule for which was authoritative, and that one of
them carried no provenance metadata at all.

**The rule now, stated once:** every experiment output lives under `results/<dataset>/<run>/`, and
every fold within it carries a `manifest.json` that records the exact git commit, the data cache
hash, the resolved model configuration and the participant split. Nothing else in the tree is an
experiment output. If a directory of results does not carry that manifest, it is not trustworthy and
does not belong here.

---

## 1. The only authoritative location: `results/`

```
results/<dataset>/<run>/<backbone>_<encoding>/
├── SUMMARY.md                  <- the human-readable results document
├── RQ1_table.csv, RQ2_table.csv, RQ3_table.csv, ...
├── *.png                       <- figures
└── seed_<S>/fold_<F>/
    ├── manifest.json           <- PROVENANCE: git commit, cache hash, config, split
    ├── complete.json           <- written last; marks the fold as finished
    ├── metrics.json            <- this fold's numbers
    ├── dssl_encoder.pt         <- frozen encoder weights
    ├── dssl_training.pt        <- full training state (optimiser + RNG)
    └── representations.npz     <- encoded windows
```

### What `manifest.json` records

| Field | Purpose |
|---|---|
| `code_version` | `{git_commit, git_dirty, source}` — the exact commit, and whether the tree was dirty |
| `code_sha256` | Hash of the source files that actually affect this result |
| `cache_sha256` | Hash of the input data cache, so a silent data change is detectable |
| `resolved_model` | The full model configuration after all defaults were resolved |
| `config`, `seed`, `fold`, `steps`, `val_frac`, `normalization` | The experiment settings |
| `training_ids`, `test_ids` | The exact participant split, so leakage is auditable |
| `smoke`, `scientific_scope` | Whether this run has scientific meaning at all |

### Every run currently present, with its commit

All 181 folds record `git_dirty: false` — no result was produced from a modified working tree.

| Dataset | Run | Commit | Folds | What it is |
|---|---|---|---:|---|
| hrd | `narval_v1` | `49bad6e` | 30 | The pre-repair model (3 seeds × 5 folds × 2 arms) |
| globem | `narval_v1` | `49bad6e` | 30 | ″ |
| hrd | `narval_v2_split` | `b6e4f8b` | 15 | Rejected readout variant (amplitude raw, phase normalised) |
| globem | `narval_v2_split` | `b6e4f8b` | 15 | ″ |
| hrd | `narval_v2_timestep` | `b6e4f8b` | 15 | Rejected readout variant (upstream per-timestep L2) |
| globem | `narval_v2_timestep` | `b6e4f8b` | 15 | ″ |
| hrd | `narval_v3_decomposed` | `0e3bdc7` | 15 | Rejected pre-encoder decomposition |
| globem | `narval_v3_decomposed` | `0e3bdc7` | 15 | ″ |
| hrd | `dev_a0_shared` | `ffee9be` | 10 | Development capacity ladder, shared encoder |
| hrd | `dev_a1_decomposed` | `ffee9be` | 10 | Development capacity ladder, decomposed |
| hrd | `dev_a1lo_decomposed` | `ffee9be` | 10 | Development capacity ladder, reduced capacity |
| hrd | `a2check_smoke_cpu` | `ffee9be` | 1 | A local smoke check — **no scientific meaning** |

**Missing, and known to be missing:** the canonical run `narval_v2` itself exists only on the
cluster. Its numbers are quoted from `docs/VALIDATION_RESULT.md`, which is committed to git. See
`REPORT_TO_PROFESSOR.md` §M.1 and section 3 below for how to bring it down.

---

## 2. What is NOT an experiment output

These locations are sometimes mistaken for results. They are not, and nothing should be read from
them as a scientific number.

| Path | What it actually is |
|---|---|
| `datasets/cache/*.npz` | **Input** data caches, not output. `configs/*.json` point at them by name, and every `manifest.json` records their hash |
| `datasets/*.csv` | The raw exports the caches are built from |
| `archive/rescue_20260914/before_repairs.zip` | Source files preserved before the 2026-09-14 repairs, with a hash manifest. Cited in `docs/SCIENTIFIC_PROTOCOL.md` |
| `archive/rescue_20260914/retired/` | Old launchers and notes retired during the rescue, with a manifest of original paths. Cited in `docs/audit/REPAIRS.md` |
| `archive/failed_experiments/` | Two rejected implementations kept for reference, not their outputs |
| `docs/` | The written scientific record — `VALIDATION_RESULT.md` is the source of the canonical numbers |

---

## 3. The one consistent procedure for moving results

Previously results arrived by three different routes and landed in three different places
(`results/`, an extracted `rescue_results_*/` tree, and loose `.npz` files at the repository root).
There is now one route in each direction.

### Uploading code to the cluster

The bundle is built from git, never from the working tree, so what runs on the cluster is always a
named commit:

```bash
python scripts/stamp_version.py                    # writes CODE_VERSION from git HEAD
git archive --format=tar.gz -o rescue_upload.tgz \
    --prefix=rhythmssl_rescue/datasets/cache/ --add-file=datasets/cache/hrd_rescue_v1.npz \
    --prefix=rhythmssl_rescue/ --add-file=CODE_VERSION HEAD
scp rescue_upload.tgz <user>@narval.alliancecan.ca:~/projects/def-plago/<user>/projects/
```

`CODE_VERSION` is what every `manifest.json` later reports as `code_version`. This is why a result
can always be traced back to a commit. `CLUSTER.md` holds the full procedure.

### Downloading results from the cluster

**Download directly into `results/`. Do not extract into a separate tree.** The earlier practice of
unpacking into `rescue_results_narval_v1/` produced a 2.1 GB duplicate that was a strict subset of
`results/` and made it ambiguous which copy was authoritative.

```bash
# On the cluster: pack one run, excluding checkpoints
cd ~/projects/def-plago/<user>/projects/rhythmssl_rescue
tar czf ~/narval_v2.tgz --exclude='*.pt' results/hrd/narval_v2 results/globem/narval_v2

# Locally: extract in place, so it lands directly in results/<dataset>/<run>/
scp <user>@narval.alliancecan.ca:~/narval_v2.tgz .
tar xzf narval_v2.tgz          # unpacks into results/hrd/narval_v2 and results/globem/narval_v2
rm narval_v2.tgz
```

Then regenerate the tables locally so the summary matches the local files:

```bash
python scripts/run_experiment.py --dataset hrd    --summarize --run-name narval_v2
python scripts/run_experiment.py --dataset globem --summarize --run-name narval_v2
```

### Verifying a downloaded run

```bash
python -c "
import json, glob, collections
c = collections.Counter()
for m in glob.glob('results/*/narval_v2/**/manifest.json', recursive=True):
    d = json.load(open(m))
    c[(d['code_version']['git_commit'][:7], d['code_version']['git_dirty'])] += 1
print(c)"
```

A trustworthy run reports **one** commit and `git_dirty: False`. More than one commit means the run
was assembled from different code versions and its folds are not comparable.

---

## 4. `.gitignore` guarantees

`results/`, `datasets/cache/`, `rescue_upload.tgz`, `rhythmssl_rescue/` and `rescue_results_*/` are
all ignored. This matters beyond tidiness: `oof_predictions.csv` contains participant-level
predictions, so an accidental `git add -A` must not be able to stage it.

---

## 5. What was removed to reach this state

| Path | Size | Why it had to go |
|---|---:|---|
| `results_hrd/`, `results_hrd_energy/` | 9.7 GB | The pre-rescue generation, 45 + 22 SLURM job directories. **Zero `manifest.json` files** — no directory could be tied to a code version, data cache or configuration, so no number in them is verifiable. Inventory preserved in `REMOVED_LEGACY_OUTPUTS.md` |
| `results_globem/` | 15 MB | Same generation, same absence of provenance |
| `_evaljson/` | 568 KB | Loose `eval.json` files from a one-shot evaluation, outside any run structure |
| `eq_h_full.npz` | 744 MB | Cached representations from the rejected level-equivariance line, whose code no longer exists |
| `hrd_2224103.npz`, `globem_windows.npz`, `ee_windows.npz` | 16 MB | Pre-rescue data caches (3-channel HRD, 12-channel GLOBEM) superseded by `datasets/cache/` |
| `rescue_results_narval_v1/` | 2.1 GB | Strict subset of `results/`; 0 unique files |
| `rhythmssl_rescue/`, `rescue_upload.tgz` | 80 MB | Local copies of the upload bundle, regenerable from commit `49bad6e` |
| `archive/rescue_20260914/plot_review_smoke/`, `pre_rq2_correction_smoke/` | 22 MB | Archived **smoke** output. The project's own convention labels smoke runs as having no scientific meaning; archiving them permanently contradicted it |

Total removed: approximately 12.7 GB, none of it tracked by git, none of it cited in
`REPORT_TO_PROFESSOR.md`, `FAILED_EXPERIMENTS.md` or `docs/VALIDATION_RESULT.md`.
