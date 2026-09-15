# Applied repairs and validation

The user clarified scientific authority on 2026-09-14: original scientific questions govern, while genuine later refinements are retained. [SCIENTIFIC_PROTOCOL.md](../SCIENTIFIC_PROTOCOL.md) records the resulting comparison. The two Phase A audit documents describe the **before-repair** state; they are preserved as evidence rather than rewritten to imply that defects never existed.

## Preservation and cleanup status

Before changing implementation, 80 source/manuscript/note files and staged/unstaged root patches were saved to `archive/rescue_20260914/before_repairs.zip`, with `manifest.json` containing file hashes. No historical dataset, cache, checkpoint, or result directory was deleted. Broad result relocation/deduplication remains pending; it must preserve unique run context. The working tree's pre-existing staged changes were not reset or committed.

## Data repairs actually applied

**Problem:** HRD dropped activity, miscounted absent minutes and partly interpolated long gaps. Its final complete week could be lost, and observation flags did not survive cache creation.

**Evidence:** Phase A counterexamples, original raw export, and cache-builder feature removal. The complete repaired build read all 53,575,181 rows / 166 participant blocks and found no invalid timestamps or interleaved participant blocks.

**Fix:** HRD now streams one participant at a time, inserts absent minute rows, retains original per-channel observation masks, fills only entire interior runs within the declared 30-minute limit, uses midnight window origins, accounts for the final minute's interval, and preserves steps. Original-unit completed windows are stored alongside normalized model inputs. The existing within-window completion rule remains explicit; it is not relabelled as observed data.

**Scientific consequence:** Eligibility and the cohort change. The repaired cache has 3,803 windows / 151 IDs / 113 labelled IDs / 51 positives versus 3,890 / 152 / 114 / 52 previously. One additional ID is excluded by the corrected missingness calculation; the machine record names it. There are 136 removed and 49 added window IDs, including changed alignment. Old results cannot be silently compared as though this were the same input experiment.

**Problem:** GLOBEM excluded two exported sensor features without a sufficient scientific rationale and discarded channel-specific missingness.

**Fix:** Preserve all 14 exported numeric features by default, including Bluetooth/Wi-Fi, and save per-channel masks and original-unit completed windows. Unknown label tokens and conflicting endpoint values now raise errors. The unsupported majority-of-weekly-labels endpoint mode is blocked.

**Scientific consequence:** GLOBEM has 6,804 windows versus 6,801 previously; the three additional windows meet the unchanged any-channel coverage rule when the restored channels are considered. IDs and endpoint labels are unchanged: 702 IDs / 669 labelled / 269 positive. Sensor exclusions are now explicit CLI ablations applied after common full-sensor window eligibility.

Both builds preserve unlabelled windows with `y=-1` and save the explicit labelled-ID set. The loader continues to support old mask-defined labelled cohorts for historical checks. Full-record participant normalization remains **retrospective**, and GLOBEM cross-year person linkage and endpoint derivation remain unresolved. These defects are not claimed fixed by metadata alone.

| New cache | Shape | Meaning |
|---|---|---|
| `datasets/cache/hrd_rescue_v1.npz` | 3,803 × 672 × 4 | HR, steps, sleep fraction, screen; 15-minute bins |
| `datasets/cache/globem_rescue_v1.npz` | 6,804 × 112 × 14 | All exported segment features; four bins/day |
| `datasets/cache/hrd_energy_rescue_v1.npz` | Aligned to HRD window IDs | Existing daily emotional-energy values averaged over each window's dates |

HRD physical-unit windows use per-bin means: heart rate in bpm, mean minute step count (not summed 15-minute steps), mean sleep indicator, and mean minute screen-event value. GLOBEM physical units are those of the supplied RAPIDS feature columns. Reconstructing HRD physical values from its linear normalization can introduce sub-micro-unit floating-point negatives around zero; the saved minimum step value is approximately −9.5e−7, not a negative raw step count.

Adjacent JSON files record raw-source SHA256, channel names, normalization scope, observation-mask meaning, counts and provenance limitations. [repair_validation.json](repair_validation.json) records exact old/new differences, target eligibility and strict-domain checks. The versioned cache builder refuses existing output paths.

## Evaluation repairs actually applied

- A LODO fold now records its held-out year and excludes **all** windows from that year from SSL, including unlabelled records. Rechecking the corrected real GLOBEM cache gives zero held-out-year training records in every fold. This establishes domain exclusion, not cross-year person disjointness.
- Classification inner splits require enough participant groups in each class and never fall back to a window split. Candidate penalties/families are scored after participant probability aggregation. Endpoint aggregation rejects conflicting labels.
- Random projections fit one fixed scaler and Gaussian map using explicit training inputs. Perturbing another evaluation row no longer changes an unchanged row's representation.
- Untrained controls reconstruct the saved full-spectrum versus harmonic-band choice and verify recorded band geometry.
- Personal references require sorted unique starts and consecutive reference-to-current intervals. A four-week reference followed by a 49-day gap is rejected. Fractional elapsed days are preserved.
- Phase shifts must be exactly representable; no surrogate perturbation levels are invented. The weekly perturbation diagnostic is explicitly unavailable for 28-day overlapping GLOBEM windows, and the old downstream deviation ladder is disabled for that geometry.
- Corrected fold tests now require the full set of finite paired estimates and use run-dependent t degrees of freedom. Degenerate variance no longer produces automatic significance. A paired participant bootstrap is available for fixed repeated OOF score matrices and explicitly excludes retraining/post-selection uncertainty.

The old RQ dispatch and repeated-DeLong reporting have now been retired. The corrected runner uses the participant protocol in `evaluation_protocol.py`, described below; historical utility behavior is not the primary protocol.

## Original rhythm targets restored

`tasks/rhythm.py::individual_markers` implements amplitude/phase, IS, IV and activity RA, with supplementary MESOR. IS is daily-profile variance divided by total variance; IV is successive-difference variance divided by total variance at the declared bin width. RA uses the most-active 10-hour and least-active 5-hour intervals of the average daily step-count profile. These definitions follow the [pyActigraphy reference documentation](https://ghammad.github.io/pyActigraphy/_modules/pyActigraphy/metrics/metrics.html); no actigraphy-specific binarization threshold is imported into the wearable features.

Targets are explicitly computed on completed physical-unit windows and require at least 70% originally observed bins for that channel. The 70% rule follows the existing HRD per-channel eligibility criterion; applying it to GLOBEM target reliability is a documented new analysis rule, not a claim about the source dataset. GLOBEM has very few qualifying sleep-channel windows, which is reported rather than hidden. Zero-variance IS/IV and numerically undefined phase are unavailable. Standard activity RA is unavailable at six-hour resolution and is not generalized to arbitrary signed/z-scored sensor channels.

These targets are now connected to participant-profile recovery, endpoint association and the exploratory recovery/prediction bridge. Within-person dynamic recovery and natural emotional-energy analysis remain pending and are not claimed implemented by these profile analyses.

## Commands run

```bash
python scripts/build_cache.py globem --csv datasets/GLOBEM_REDUCED.csv --out datasets/cache/globem_rescue_v1.npz
python scripts/build_cache.py hrd --csv datasets/HRD_RAW_MinuteLevel.csv --out datasets/cache/hrd_rescue_v1.npz --energy-out datasets/cache/hrd_energy_rescue_v1.npz
python -m unittest discover -s tests -t . -v
```

The HRD preprocessing pass completed in approximately 409 seconds before final cache serialization. These are local CPU/data-validation commands, not tested Narval submissions. The historical `validate_audit.py` (now `archive/docs_audit/`) and its original counterexamples reflect the archived pre-repair implementation; use `tests/` for the repaired invariants rather than overwriting Phase A evidence.

## Final essential implementation

### Problem

The later implementation used score-derived objectives, mixed evaluation units, full-person normalization that removed individual amplitude information, and questions narrowed to fit its evaluation code.

### Evidence

The manuscript history and initial discrepancy audit document those changes. `models/losses.py` explicitly records weights derived from previous concordance scores; historical scripts searched readouts/probe families. The cache metadata records full-person normalization. The original `0120be4` manuscript asks about individual amplitude, phase, RA, IS and IV and their relation to prediction.

### Fix

The active runner uses only the original equal seasonal-loss weighting, full-spectrum branches, no auxiliary heads or smoothing, and explicitly distinguishes DSSL's augmentation/circular-phase modifications from the CoST reference adapter. It fits normalization using observed training-participant moments, then freezes it. It averages windows to one feature row per participant and uses the same participant-level inner logistic protocol across feature methods. A training-population-mean control complements untrained/raw/PCA RQ1 controls. Per-person predictions, marker recovery, baseline choices, configuration and data/code hashes are saved.

`scripts/run_experiment.py` runs the three initial analyses from each fold's common frozen representations; `evaluation_protocol.py` produces paired OOF summaries and saved-data figures. Full checkpoints include optimizer, momentum encoder, negative queue, shuffled sample position and RNG states; exact interrupted-versus-uninterrupted CPU continuation is tested. TCN/Transformer encoding interfaces pass small forward/training checks; official Mamba remains optional and unverified. The initial scientific matrix uses only TCN/no encoding, five fixed folds, three initialization seeds, and equal 1,000-step budgets.

GLOBEM is restricted to the 2018 cohort in this initial matrix because missing cross-year identity linkage prevents a defensible all-year person-disjoint split. No other year, labelled or unlabelled, enters its pretraining. The complete all-year cache is preserved. This is an explicit limitation of scope, not a claim of leave-year-out benchmark reproduction.

The old root launchers, score-driven task ladders, stale Slurm scripts and loose experiment notes were moved to `archive/rescue_20260914/retired/`; its manifest records original paths and hashes. Twenty-nine explicitly named files/directories were archived, including intermediate development smoke outputs. No raw dataset, historical result directory or unique old checkpoint was deleted. A few regression helpers remain in `tasks/`, but the new runner does not invoke the archived ladders.

### Scientific consequence

The initial runs can test individual rhythm preservation and participant endpoint utility without quietly changing sensors or giving baselines different downstream protocols. They can fail meaningfully: good synthetic sensitivity or a favorable single seed cannot substitute for original marker recovery or clinical association. The manuscript now presents a dated experiment protocol, not unverified results.

## Remaining limits — do not label the whole PhD study complete

- RQ1 currently tests individual mean profiles; natural within-person dynamic recovery is not implemented in the initial runner.
- RQ2 has been corrected to the original Personalized Rhythmic Phenotyping question: unlabeled four-week personal baselines and controlled within-person timing/strength deviation detection on held-out HRD participants. Endpoint associations remain secondary. Natural emotional-energy/repeated-measures validation remains pending and is not silently substituted for RQ2.
- Endpoint instrument/cutoff/timing and timestamp timezone provenance still need authoritative confirmation. No labels or thresholds were invented.
- Cross-year GLOBEM identity linkage remains unknown, so cross-year generalization is deferred.
- CPU execution and Bash syntax checks do not verify Narval GPU execution, full-budget convergence or clinical validity. GPU smoke submission is the next cluster step, not the full array.
- The corrected LaTeX source is updated, but no local TeX compiler is available to verify PDF compilation.

See `docs/RUNNING.md` for exact commands, outputs, the file map and statistical limitations. No large GPU jobs have been launched and no scientific success claim has been made from execution tests.
