# Consolidation manifest (2026-09-18)

Every removal below is justified by a completed experiment, and every removed file's scientific
content is preserved in `FAILED_EXPERIMENTS.md` or `REPORT_TO_PROFESSOR.md` before deletion.
Nothing is removed because it "looks unused": each symbol was traced through code, configs, SLURM
scripts and documentation first (`git grep`).

## Canonical model retained

The configuration recorded in `configs/hrd.json` and `configs/globem.json` — shared encoder,
`readout_norm="none"` (now hard-wired), `w_eq=0` (now removed), `trend_views="same"` (now removed).
This is the configuration evaluated at protocol grade as run `narval_v2`. Section 1 of
`REPORT_TO_PROFESSOR.md` gives the evidence for choosing it.

## A. Configuration families removed

| Path | What it was | Why obsolete | Preserved in |
|---|---|---|---|
| `configs/ablation/hrd_a0..a4*.json` | First failure-diagnosis batch: masking/queue, amplitude readout, phase mode | All tested at 2-fold screen; only `readout_norm=none` survived and is now the default | `FAILED_EXPERIMENTS.md` §1–3 |
| `configs/ablation/hrd_b0..b3*.json` | Equivariance-weight dose-response (`w_eq` 1.0 / 0.277 / 0.05 / 0) | Established `w_eq=0`; the coefficient is now removed from the model entirely | `FAILED_EXPERIMENTS.md` §4 |
| `configs/ablation/hrd_c1_trend_days.json` | Disjoint-day trend pairing | Tested at 5 folds, rejected (RQ1 worse in 5/5 folds) | `FAILED_EXPERIMENTS.md` §5 |
| `configs/readout/*.json` | `timestep` and `split` readout variants | Both evaluated at full protocol grade and rejected | `FAILED_EXPERIMENTS.md` §6 |
| `configs/arch/*_decomposed.json` | Pre-encoder decomposition, 0.54× capacity | Evaluated at full protocol grade and rejected | `FAILED_EXPERIMENTS.md` §7 |
| `configs/v2/*.json` | Development ladder round 1 (a0/a1lo/a1) and round 2 (a2/a3/a3s) | Round 1 completed and rejected decomposition at matched capacity; round 2 arms were never evaluated | `FAILED_EXPERIMENTS.md` §7–8 |

## B. SLURM scripts removed

| Path | Purpose | Why obsolete |
|---|---|---|
| `slurm/ablation.sbatch`, `ablation_confirm.sbatch`, `ablation_eq.sbatch`, `ablation_eq_zero.sbatch`, `ablation_trend.sbatch` | Launchers for the ablation batches above | Their configs are removed; findings recorded |
| `slurm/submit_readout.sh` | Launcher for the readout comparison | Variants rejected |
| `slurm/submit_decomposed.sh`, `slurm/submit_dev_ladder.sh` | Launchers for the decomposition and development ladder | Architecture rejected; ladder configs removed |

Retained: `submit.sh` (the locked protocol matrix), `rq123.sbatch`, `summarize.sbatch`,
`smoke.sbatch`, `env.sh`, `setup_env.sh` — these run the canonical model.

## C. Analysis scripts removed

| Path | Purpose | Why obsolete | Preserved in |
|---|---|---|---|
| `scripts/ablation_report.py` | Scored the 2-fold ablation screens | Screens superseded by protocol-grade runs | `FAILED_EXPERIMENTS.md` |
| `scripts/ablation_rq3.py` | RQ3 screen over ablation arms | Superseded; its method (pooling per-fold out-of-fold predictions) is the same one `summarize` uses | `FAILED_EXPERIMENTS.md` §9 |
| `scripts/audit_rq3_arms.py` | Adversarial audit of a one-seed RQ3 screen | Its conclusion (the screen did not survive fold resampling) is recorded | `FAILED_EXPERIMENTS.md` §9 |
| `scripts/dev_scoreboard.py` | Ranked development-ladder arms | Ladder concluded | `FAILED_EXPERIMENTS.md` §7 |

Retained: `run_experiment.py`, `build_cache.py`, `stamp_version.py`, `time_steps.py`.

## D. Documentation consolidated

Six documents are merged into `FAILED_EXPERIMENTS.md` and then deleted, because each records one
episode of a search whose conclusion is now settled:

| Path | Content | Merged into |
|---|---|---|
| `docs/READOUT_SPLIT_PRECOMMIT.md`, `docs/READOUT_SPLIT_RESULT.md` | Split-readout pre-registration and its rejection | `FAILED_EXPERIMENTS.md` §6 |
| `docs/DECOMPOSITION_PRECOMMIT.md`, `docs/DECOMPOSITION_RESULT.md` | Decomposition pre-registration and its rejection | `FAILED_EXPERIMENTS.md` §7 |
| `docs/DEV_ROUND1_RESULT.md` | Development ladder round 1 | `FAILED_EXPERIMENTS.md` §7 |
| `docs/DSSL_V2_DESIGN.md` | Proposed next-generation design | `FAILED_EXPERIMENTS.md` §10 (open directions) |

Retained unchanged: `docs/SCIENTIFIC_PROTOCOL.md` (governing protocol — never edited after
results), `docs/VALIDATION_PRECOMMIT.md` and `docs/VALIDATION_RESULT.md` (the pre-registration and
outcome of the canonical run), `docs/RUNNING.md`, `docs/RESCUE_STATUS.md`, `docs/audit/`.

## E. Model code removed

Each symbol was traced to confirm it is unreachable in the canonical configuration.

| Symbol | Evidence it is dead | Verification performed |
|---|---|---|
| `w_eq`, `head_eq`, `equivariance_loss`, the `delta` third element of each training batch | `w_eq=0` in both canonical configs; `head_eq` is only constructed when `w_eq>0`, so the term never executes. Dose-response over 5 folds established 0 as best | Confirmed `head_eq is None` and the loss branch is skipped; removing the `delta` return changes no random draw (the offset is drawn inside `shift()` regardless) |
| `readout_norm` flag, `"timestep"` and `"split"` branches | Three variants evaluated at protocol grade; `"none"` retained | Behaviour of `"none"` hard-wired; outputs unchanged |
| `trend_views` flag, `_disjoint_days`, `_visible_mean`, `queue_ids`, `neg_mask` | `disjoint_days` tested at 5 folds and rejected | `"same"` path retained verbatim; `queue_ids` only ever existed in the rejected mode |
| `w_ac`, `anticollapse_loss`, `ac_gamma`, `ac_queue`, `_log_readout_amp` | `w_ac=0` in every configuration ever run; the term has never executed in any experiment | Confirmed no config sets `w_ac`; no result depends on it |

## F. Code archived rather than deleted

`archive/failed_experiments/dssl_v2_decomposition/` retains the pre-encoder decomposition and the
seasonal-equivariance objective, because their status differs:

- **Decomposition** (`DecomposedEncoder`, `daily_average_kernel`, `decompose_daily`, the encoder's
  `branch` option) was tested at protocol grade and at matched capacity, and rejected. It is kept
  because it is the only implementation of a result that is reported quantitatively.
- **Seasonal equivariance** (`harmonic_transform`, `seasonal_equivariance_loss`, `seasonal_objective`,
  `seasonal_days`) was implemented but **never evaluated**. It is kept because deleting an untested
  idea would destroy work without evidence either way, and because it is the one open direction the
  measured failures actually point to.

Neither is imported by the canonical model.

## G. Result artefacts

`results/` is git-ignored, so nothing here is part of the repository's scientific record; the record
lives in `docs/` and in the summary tables quoted in `REPORT_TO_PROFESSOR.md`.

- **Not deleted:** every `SUMMARY.md` and the aggregate CSV tables of the locked runs, which are the
  primary source of the numbers reported.
- **Deletable safely:** `representations.npz` files (1.8 GB across 60 files). These are encoder
  outputs, regenerable by re-running the pipeline, and no reported number is read from them
  directly.
- **Smoke debris:** `results/*/‌*_smoke_cpu/` directories from local execution tests.

**Gap flagged, not silently resolved:** the locked run of the canonical model itself
(`results/hrd/narval_v2`, `results/globem/narval_v2`) is **not present locally** — only on the
cluster. Its numbers are quoted in `docs/VALIDATION_RESULT.md` and in this report from the cluster
summaries. Downloading it is listed as an outstanding action in `REPORT_TO_PROFESSOR.md` §M.

## H. Addendum, applied 2026-09-18

| Path | What it was | Why removed | Preserved in |
|---|---|---|---|
| `docs/report_to_professor.md` | Earlier architecture-comparison draft | **Actively misleading if kept**: presented `w_eq = 1.0` as a core design feature (since measured at 0 and removed), reported "29 tests" (now 30), and described the full Narval run as "pending" when it has completed. Keeping it alongside the new report would also reintroduce exactly the duplicate-final-document problem this consolidation removes | `REPORT_TO_PROFESSOR.md`. Its verified content was re-measured and folded in: the 30-of-337 bin figure and the 12-harmonic rejection (C.4), the 44.0M→7.45M parameter attribution (new Table C.4b), the `CONTRACTED` weight derivation from measured concordances and its "bet, not a correction" status (D.2), the circular-phase rationale and `IsotropicPairScaler` (D.2), and the anticipated-questions list (H.1) |

Every number carried across was re-verified against the running code before being written into the
new report; none was copied on trust.

**Correction, same day:** this entry originally stated the file was "recoverable from git history".
That is **wrong** — `docs/report_to_professor.md` was never committed (it was untracked at the start
of the consolidation), so deleting it was irreversible. Its verified content survives in
`REPORT_TO_PROFESSOR.md` as listed above, and every figure carried across was re-measured from the
running code rather than copied, so no quantitative claim was lost. But the draft prose itself is
gone, and the earlier sentence overstated the safety of that deletion.

### Defaults corrected

`--run-name` defaulted to `narval_v1` in `scripts/run_experiment.py`, `slurm/submit.sh`,
`slurm/rq123.sbatch` and `slurm/summarize.sbatch`. Since `narval_v2` is the canonical run, a
default invocation would have written into the superseded v1 namespace and overwritten an archived
comparison arm. All four defaults now read `narval_v2`.

### Defect fixed in `cost.py`

`spectral_readout`'s docstring contained a duplicated 11-line paragraph and still documented three
`readout_norm` modes ("none"/"timestep"/"split") after the option itself had been removed. Both
corrected; behaviour unchanged.

### Correction to section G: `representations.npz` is NOT freely deletable

Section G above listed all 60 `representations.npz` files (1.8 GB) as "deletable safely ... no
reported number is read from them directly". **Tracing the code before deleting showed that claim
is wrong, and nothing was deleted.**

`scripts/run_experiment.py:207` loads `representations.npz` from a **cached CoST reference**
directory whenever a variant run reuses a completed reference stage. The guard on that path checks
only for `complete.json`, so a missing `.npz` beside an existing `complete.json` raises
`FileNotFoundError` rather than gracefully retraining.

Of the 60 files, **30 sit beside a `reference.json` and are read back**; the other 30 are
variant-side and genuinely write-only. Deleting the read-back half would break the
`--skip-reference` / cached-reference workflow across every affected run.

**Decision: retained, all 60.** The disk saving does not justify breaking a working reuse path, and
distinguishing the two halves by filesystem heuristics is exactly the kind of "looks unused"
reasoning this manifest forbids. Only genuinely regenerable *smoke* output was removed.

## I. Working-directory duplication removed, 2026-09-18

None of these was ever tracked by git (`git ls-files` returns 0 for each), so nothing left the
repository's scientific record. They were local artefacts of the cluster round-trip documented in
`CLUSTER.md`, whose build recipe (`git archive ... -o rescue_upload.tgz`) is retained.

| Path | Size | What it was | Proof it was redundant |
|---|---:|---|---|
| `rhythmssl_rescue/` | 41 MB | Full working copy of the repository staged for upload | Snapshot of commit `49bad6e`, which is reachable in git history; regenerable with `git archive 49bad6e` |
| `rescue_upload.tgz` | 39 MB | The same tree, packed for `scp` | Same 117 entries as the directory above; `CLUSTER.md` retains the command that builds it |
| `rescue_results_narval_v1/` | 2.1 GB | Results downloaded back from the cluster | **Verified strict subset of `results/`: 0 files exist only in this copy.** `results/` additionally holds the v2/v3 arms and a newer `SUMMARY.md` format |
| `__pycache__/` | 172 KB | Bytecode cache | Build artefact |

Total reclaimed: ~4.2 GB (22 GB → 20 GB).

`.gitignore` previously covered only `rhythmssl_rescue/`. It now also covers `rescue_upload.tgz`
and `rescue_results_*/`, closing a gap through which the downloaded results — which contain
participant-level `oof_predictions.csv` — could have been staged.

### Retained deliberately, though they look like clutter

| Path | Size | Why kept |
|---|---:|---|
| `hrd_2224103.npz`, `globem_windows.npz`, `ee_windows.npz` | 16 MB | Caches of a superseded data generation (3-channel HRD, 12-channel GLOBEM). `tests/test_repairs.py:211` exercises the loader against them and `skipTest`s when absent, so they add real coverage on differently shaped real data for negligible cost |
| `eq_h_full.npz` | 744 MB | Cached representations from the rejected level-equivariance line. Not referenced by any code or final document, but the experiment line no longer exists in the tree, so it is not cheaply regenerable. Retained pending an explicit decision |
| `results_hrd/`, `results_hrd_energy/`, `results_globem/` | 9.7 GB | The previous project generation, 46 SLURM job directories. Not cited in `REPORT_TO_PROFESSOR.md` or `FAILED_EXPERIMENTS.md`, but the only raw source behind conclusions recorded elsewhere. Deleting is a research-history decision, not code hygiene; retained pending an explicit decision |

## J. Provenance consolidation, 2026-09-18

Experiment output was scattered across four top-level locations with no rule for which was
authoritative. An audit of provenance metadata settled it:

- `results/` — **181 folds, every one carrying `manifest.json`** with the git commit, code hash,
  data-cache hash, resolved configuration and participant split. All report `git_dirty: false`.
  Fully traceable, and now the single authoritative location.
- `results_hrd/`, `results_globem/`, `results_hrd_energy/` — **0 `manifest.json` files**. No
  directory could be tied to a code version, data cache or configuration.
- `_evaljson/`, `eq_h_full.npz` — loose evidence outside any run structure. `docs/audit/README.md`
  line 191 had independently flagged exactly these as "detached evidence ... needs a verified
  linkage", which was never established.

The rule and the single upload/download procedure are written in **`docs/OUTPUT_PROVENANCE.md`**.
An inventory of what was removed is in **`docs/REMOVED_LEGACY_OUTPUTS.md`**.

| Path | Size | Reason |
|---|---:|---|
| `results_globem/` | 15 MB | No provenance metadata |
| `_evaljson/` | 568 KB | Loose one-shot evaluation JSON outside any run |
| `eq_h_full.npz` | 744 MB | Representations from the rejected level-equivariance line; code no longer exists |
| `hrd_2224103.npz`, `globem_windows.npz`, `ee_windows.npz` | 16 MB | Pre-rescue caches (3-ch HRD, 12-ch GLOBEM) superseded by `datasets/cache/` |
| `archive/rescue_20260914/plot_review_smoke/` | 11 MB | Archived smoke output; the project labels smoke as having no scientific meaning |
| `archive/rescue_20260914/pre_rq2_correction_smoke/` | 11 MB | Same |
| `results/hrd/a2check_smoke_cpu/` | small | Smoke output inside the authoritative tree |
| `baselines/` | empty | Empty leftover directory |

`tests/test_repairs.py::test_real_cache_tiny_model_and_participant_probe` iterated over four data
caches, two of which were the removed pre-rescue ones. It now covers only the two canonical caches,
with a comment recording that the second window geometry is no longer exercised.

**Still present, blocked by the sandbox:** `results_hrd/` (8.5 GB) and `results_hrd_energy/`
(1.2 GB). Both have zero provenance and are inventoried in `REMOVED_LEGACY_OUTPUTS.md`; the deletion
was refused locally and must be run by hand.
