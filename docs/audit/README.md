# Scientific rescue audit — Phase A

Audit date: 2026-09-14. **The existing results are exploratory, and this repository is not yet ready for a confirmatory Narval run.** Basic model execution works; the scientific protocol, data provenance, controls, and result provenance do not yet support the manuscript's strongest claims.

This is an audit of an existing project, not a replacement implementation. No existing source, dataset, checkpoint, result, or staged change was removed or rewritten during this audit. The changes in this phase are this report, the [RQ specification](RQ_SPECIFICATION.md), the [audit program](validate_audit.py), and its [evidence](evidence/). In the findings below, **Fix means the recommended next change, not a change already applied**.

## 1. Scope and reference hierarchy

The working tree already contained a substantial staged rewrite when the audit began. Both root `HEAD` and `pre-cleanup` were `0a88999022cda79bcb7fab4311b396706d2e0bd5`. The root origin is `melikas/CoST`; it is the research fork, not the original Salesforce reference. The manuscript is a separate nested Git repository, at `a69aa4e` when inspected. Preserve both histories and the initial staged state: [status](evidence/initial_git_status.txt), [staged diff summary](evidence/initial_staged_diff.txt), [revisions](evidence/git_revisions.txt).

Completed inspection:

- Read the active Python implementation and entry points, current manuscript sections, local documentation, dependency declarations, and shell/SLURM scripts. The source index records 27 existing Python files / 5,242 lines, excluding the new audit program.
- Compared the initial Overleaf manuscript (`0120be4`) and subsequent manuscript history with the current RQs; inspected historical DSSL/backbone/encoding/baseline implementations at `pre-cleanup`.
- Read the original [CoST paper](https://arxiv.org/html/2202.01575) and [Salesforce repository](https://github.com/salesforce/CoST), including its [training implementation](https://raw.githubusercontent.com/salesforce/CoST/main/cost.py) and [encoder](https://raw.githubusercontent.com/salesforce/CoST/main/models/encoder.py). Local history also contains the upstream lineage.
- Streamed all 53,575,181 HRD CSV rows for schema, missingness, identifiers, labels, and timestamp deltas. Read the complete GLOBEM reduced table and rebuilt its cache exactly with the current preprocessing.
- Inventoried 38,723 files / 14,497,671,309 bytes, excluding Git internals, Python caches, and these audit files; parsed JSON schemas, hashed 34,572 text/table/figure files, and indexed result provenance.
- Executed targeted counterexamples and tiny CPU training, representation, checkpoint, and participant-level probe checks on both real caches.

Limits: this is not a visual review of every historical figure, a line-by-line review of every version of every deleted historical script, a re-execution of old experiments, a full HRD cache reconstruction, or a CUDA/Narval validation. Binary arrays/checkpoints were inventoried; their historical training provenance was not reconstructed by executing untrusted serialization. The missing checks are not implied to have passed.

### The manuscript is not an independent, unchanged specification

The first Overleaf import (`0120be4`) describes multichannel rhythm preservation, including amplitude, phase, relative amplitude (RA), interdaily stability (IS), and intradaily variability (IV), followed by depression prediction. It does not contain the present numbered H1–H4 specification. Later commit subjects explicitly include “Align experiments text with evaluation code,” “Restructure manuscript around RQ1–RQ3 frozen encoder scripts,” and “Add datasets/Q1–Q4 from evaluation code.” Another commit says it corrects fabricated Q1 metrics. Commit messages establish a history of revision; they do not independently prove which numbers were fabricated or who made each change.

The current manuscript narrows RQ1 to 24-hour cosinor recovery and RQ2 to a particular perturbation experiment, while documenting choices informed by previous outcome scores. We must not silently treat those revisions as the original research intent, or automatically restore the first version as correct. The earlier implementation also contains unequal backbone dimensions and incomplete experimental descriptions.

**Authority requiring clarification:** identify which manuscript/version is the scientific contract. Until then, the companion RQ document faithfully separates the current manuscript's questions from proposed repairs. The stable common core is CoST-inspired two-branch SSL, rhythm preservation/deviation, and frozen downstream evaluation—not a particular score-selected loss, sensor subset, or readout.

## 2. Intended model and actual model

### Reference implementation

CoST learns two temporal components from independently augmented views of the same input. A dilated convolutional feature extractor feeds causal convolutional trend experts and a learned Fourier seasonal layer. Trend learning uses an EMA encoder and queue-based contrast; seasonal learning compares amplitude and phase within the batch. The reference objective is

\[
L_{\mathrm{ref}}=L_{\mathrm{trend}}+\frac{\alpha}{2}(L_{\mathrm{amp}}+L_{\mathrm{phase}}).
\]

In the released code, the seasonal views both retain gradients through the query encoder. Seasonal outputs are normalized in the time domain and transformed back into frequency coefficients for the loss; the paper's frequency-layer description is not a substitute for inspecting that path. The forecasting export uses the final timestep's concatenated components. These details come from the [reference code](https://raw.githubusercontent.com/salesforce/CoST/main/cost.py), with the decomposition implemented in the [reference encoder](https://raw.githubusercontent.com/salesforce/CoST/main/models/encoder.py).

### Current implementation: a modified CoST-derived DSSL

`cost.py` still exposes a class called `CoST`, but it is a materially modified research method:

1. Input is a completed, participant-standardized sensor window: HRD `(672,3)` or GLOBEM `(112,12)` in the shipped caches.
2. Two views receive independent jitter, channelwise constant shifts, and optional circular smoothing. Scaling was removed. Smoothing is specified in minutes and is disabled at GLOBEM's coarse resolution.
3. A TCN feeds the trend and seasonal heads. The active tree has no working Transformer or Mamba alternative.
4. The seasonal layer is split into hand-selected frequency bands, rather than the reference full-spectrum layer. The default trend kernel cap is also different from the reference.
5. The loss is

\[
L=w_TL_T+\alpha(w_AL_A+w_PL_P)+w_{eq}L_{eq}+w_{ac}L_{ac}.
\]

The last two terms are optional modifications. The level term predicts the difference between augmentation offsets from the mean trend features; the anti-collapse term imposes a variance floor on amplitude features. Neither is an original CoST component.

6. The exported vector concatenates mean trend with amplitude and angle (or sine/cosine) of selected frequencies of the **normalized learned seasonal sequence**. For the usual 160+160 branch width, HRD exports 1,760 angle features or 2,560 circular features; GLOBEM exports 1,120 or 1,600. `output_dims=320` is not the downstream vector width.
7. RQ1 regresses signal-derived cosinor markers; RQ2 measures distance from a personal reference; RQ3 fits a frozen probe and averages window probabilities per participant.

Positive pairs are the two augmentations of one window. Trend negatives are queued keys; seasonal negatives are other windows in the batch at the corresponding frequency index. Windows from the same participant, and overlapping GLOBEM windows, can be negatives. This is an instance-discrimination design, not participant-level contrastive learning or proof that identity information was removed.

**Interpretation:** DSSL is encouraged to retain repeatable temporal structure that survives its augmentations. The Fourier branch supplies an inductive bias toward rhythmic structure. Neither the objective nor the architecture proves that learned “trend” is physiological trend, that learned amplitude equals sensor amplitude, that branches are statistically independent, or that depression information is retained. RQ1–RQ3 must supply that evidence.

## 3. Decisions and discrepancies

### A01 — FIX: HRD steps were removed on the basis of prediction scores

**Problem.** Activity is absent from the shipped input although present in the raw export and preprocessing.

**Evidence.** `scripts/build_cache.py:28` sets `HRD_DROP=("Steps",)` and line 55 applies it after windowing. `data_processing/hrd_dataset.py:167` explains the removal using a 24-seed AUC comparison (0.6884 to 0.7123). The current manuscript repeats the three-channel choice. Cache metadata confirms `HR`, `is_asleep`, `screen` only; raw steps missingness is just 0.879% of exported rows.

**Fix.** Restore steps in the default HRD sensor set as explicitly requested by the user; make any exclusion a named ablation. Rebuild a versioned cache and list the exact cohort/window changes. Preserve the old three-channel cache and results as historical evidence.

**Scientific consequence.** The existing result-selected subset cannot establish that activity is irrelevant to rhythmicity. Restoring it changes the experiment and invalidates direct reuse of old model checkpoints as the new default.

### A02 — FIX / VERIFY: GLOBEM domain and participant separation

**Problem.** Leave-one-year-out pretraining excludes labelled test IDs, but includes unlabelled IDs from the held-out year. Participant-year IDs also do not establish distinct people across years.

**Evidence.** `datautils.py`, `Cohort.fold_data`, uses the complement of test participant IDs for pretraining. The reproduced held-out-year inclusions are 13 IDs / 90 windows (2018), 14 / 123 (2019), 1 / 8 (2020), and 5 / 50 (2021): [cache evidence](evidence/caches.json). The [official GLOBEM description](https://github.com/UW-EXP/GLOBEM) reports 497 unique people across more than 700 person-years; the local CSV has 705 IDs.

**Fix.** Exclude the entire held-out domain, including unlabelled records, for a strict domain-generalization experiment. Obtain a stable cross-year person mapping and exclude all records of each held-out person where person-disjoint generalization is claimed. If linkage is unavailable, label results participant-year/domain evaluation with unresolved cross-year person overlap; do not claim person-disjoint validation.

**Scientific consequence.** Current LODO results can benefit from exposure to the supposedly unseen domain. Fixing domain exclusion alone does not resolve repeated-person exposure.

### A03 — FIX / VERIFY: missingness and normalization do not mean what the comments claim

**Problem.** HRD's short-gap operation partly fills long gaps; missing timestamps are not first inserted onto a complete minute grid. Later interpolation fills remaining interior gaps. Both datasets use statistics from the participant's complete record.

**Evidence.** `_interpolate_short_gaps` in `hrd_clean.py` uses pandas interpolation with `limit=30`; a 60-row missing run receives 30 filled values. Raw HRD timestamp differences include absent stretches up to 23,540,160 seconds. `hrd_windows.py` computes z-scores before completed-window construction; the manuscript describes standardization after it. GLOBEM `_prep_participant` interpolates across the full record and carries end values without a maximum gap. Its observation gate is “any channel present,” not per-channel coverage. [Counterexamples](evidence/counterexamples.json), [raw HRD evidence](evidence/hrd_raw.json), [raw GLOBEM evidence](evidence/globem_raw.json).

**Fix.** Establish the timestamp grid and original per-channel observation mask before filling. Explicitly identify whole short gaps, retain long-gap missingness for eligibility checks, and save masks/coverage. Distinguish retrospective full-record standardization from prospective historical-only normalization. Specify the available observation interval relative to the endpoint survey; do not invent a new cutoff.

**Scientific consequence.** Interpolated rhythms can become the target that the model is congratulated for recovering. Full-person statistics do not cross people, but can use future observations in a historical RQ2 claim; “leakage-free by construction” is too broad. Per-person scaling also removes absolute between-person levels and changes amplitude into relative units.

### A04 — FIX: window chronology and phase reference

**Problem.** A complete HRD final window can be dropped at a boundary; clock phase is not consistently anchored; RQ2 accepts a remote reference week.

**Evidence.** `_tag_samples_with_window_and_bin` computes complete windows from the last timestamp, not the end of its sampling interval: minute timestamps 0 through 10,079 describe one complete week but yield zero complete windows. HRD's default first-sample anchor produces 49 non-midnight windows according to the current manuscript. `personal_baseline` verifies only the span of the four reference starts; starts `[0,7,14,21,70]` days wrongly admit the current week at day 70. It relies on existing row order rather than sorting/asserting it.

**Fix.** Use explicit half-open time intervals, document clock origin/timezone and DST handling, sort/assert chronology, and require all four references plus the current week to be consecutive and non-overlapping for the stated RQ2. Do not silently apply this weekly protocol to overlapping 28-day GLOBEM windows.

**Scientific consequence.** Relative Fourier phase is not automatically clock acrophase. Remote or overlapping references test a different hypothesis from deviation from the preceding four contiguous weeks.

### A05 — FIX / VERIFY: labels and source attribution

**Problem.** Current H3 says endpoint CES-D ≥16, while the GLOBEM dataset section says BDI-II >13. The reduced CSV does not document how its boolean endpoint was generated. HRD's source/cohort narrative is also questionable.

**Evidence.** `SSL_Rhythmicity/sections/3-design.tex` versus `4-datasets.tex`; `globem_dataset.py` consumes `LABEL_ENDPOINT` rather than computing a scale threshold, and maps unknown strings to zero. The [official GLOBEM data documentation](https://raw.githubusercontent.com/UW-EXP/GLOBEM/main/data_raw/README.md) distinguishes weekly/end-term labels and questionnaire versions across years. The local HRD export spans September 2021–February 2023 and has 166 IDs, whereas the manuscript attributes it to a one-semester first-year cohort. The [Human Rhythms repository](https://github.com/HAI-lab-UVA/Human-Rhythms-Dataset) and [associated authors' paper](https://mariacardei.github.io/assets/pdf/rhythms_imwut.pdf) describe a longer study matching those broad characteristics; an exact export/source match still needs provenance.

**Fix.** Recover the export scripts or survey-to-label manifest, record instrument/version/threshold/date by dataset, assert valid labels and conflicts, and correct attribution using verified provenance. Keep weekly-label experiments outside the endpoint pipeline. Do not manufacture a common label by majority voting over surveys.

**Scientific consequence.** A depression endpoint is not interchangeable with another scale or a majority of time-varying symptoms. Current cache consistency is evidence about the export, not proof of clinical label derivation.

### A06 — FIX: model and control selection used the outcomes they later evaluate

**Problem.** Several method choices are justified by previous RQ scores, including removing an informative control.

**Evidence.** `models/losses.py` derives `CONTRACTED` from earlier RQ2 concordances. `tasks/recovery.py`, `run_rq1`, explicitly explains not scoring the untrained encoder by its earlier favorable comparison. `hrd_dataset.py` selects sensors using AUC. `tasks/report.py` selects an arm using the same pooled out-of-fold results before contrasts. The manuscript documents adoption decisions after RQ2 results and many additional readout families.

**Fix.** Retain outcome-informed runs as exploratory. Freeze the future method and primary contrasts before scoring; use training-only development or an explicitly independent confirmation design. Restore the untrained encoder as an RQ1 diagnostic of the contribution of SSL. Do not describe code comments as proof of preregistration.

**Scientific consequence.** A favorable PCA comparison does not show that pretraining improved the architecture. Reusing the same cohort after extensive method selection requires an explicit exploratory interpretation; merely changing the random seed does not create an untouched test population.

### A07 — KEEP / FIX / VERIFY: preserve CoST's core, separate every modification

**Problem.** Names such as `CoST` and `PAPER` can imply reference fidelity that the complete method lacks.

**Evidence.** The working implementation changes augmentation strength/type, trend kernel range, seasonal bands, phase similarity, objective weights, optional objectives, and export. The reference uses jitter/shift/scaling (sigma 0.5), a full spectrum including DC, and raw-angle seasonal contrast. `PAPER` restores only relative loss coefficients, not all reference behavior. `CONTRACTED` also reduces the trend coefficient to 0.277; renormalizing the seasonal pair does not keep the seasonal-to-trend balance unchanged. The reference's normal forward call defaults to all-true masking despite a binomial constructor option; enabling masking would itself change those calls.

**Fix.** Keep the two-branch backbone/head/contrast structure. Name the modified implementation DSSL. Provide an explicit reference configuration with pinned upstream provenance and a documented wearable readout adapter. Freeze one DSSL modification set; remove score-driven loss alternatives and the untrained anti-collapse experiment from the default path. Evaluate any necessary amendment as an explicit method change.

**Scientific consequence.** Reference reproduction, domain adaptation, and proposed scientific novelty become separable. The prior claim that all depth/band changes “fix defects wrong under any hypothesis” is not justified.

### A08 — FIX / VERIFY: spectral interpretation and amplitude preservation

**Problem.** Exported spectral magnitudes are treated too readily as physical rhythm strength, and frequency labels do not always match their periods.

**Evidence.** `spectral_readout` normalizes each timestep over seasonal channels before FFT. Multiplying the entire seasonal sequence by 3 changes its amplitude readout by only about 9.5e-7 in the audit example. This tests global gain of the learned sequence, not arbitrary raw-input scaling. Time-varying normalization can mix frequencies across the purported bands. `spectral_freqs` calls bin 1 “circaseptan,” although in a 28-day GLOBEM window it is a 28-day period; the weekly bin would be 4. Its GLOBEM Nyquist bin 56 is real-valued and does not have a freely varying phase. Circular box smoothing preserves phase only where its response is positive; it does not preserve every frequency's phase as the comment claims.

**Fix.** Document physical frequency/period for every selected bin and the effect of normalization. Limit phase claims to identifiable frequencies/amplitudes. Retain reference normalization in a reference control; any change to the DSSL amplitude path must be explicit and validated rather than silently removing normalization to obtain a desired result.

**Scientific consequence.** The representation may encode relative multichannel rhythm patterns while suppressing global amplitude. High rhythm-probe performance and poor amplitude-deviation performance are not logically contradictory, and neither proves a physical decomposition.

### A09 — FIX: evaluation selection unit and degenerate splits

**Problem.** Outer participant aggregation is present, but inner model selection does not consistently use that unit. Degenerate cases can break grouping.

**Evidence.** `tasks/_eval_protocols.py` selects logistic penalties using window-level AUC, then evaluates mean probabilities per participant. `selection_split` falls back to a split sharing a single group in the reproduced one-group example. `tasks/recovery.py` uses `RidgeCV`'s window-based leave-one-out selection rather than participant groups. Training pretext validation randomly splits windows; the reproduced first HRD fold shares 116 IDs between those subsets. That is not outer test leakage by itself, but cannot support a participant-generalization validation claim. Window counts vary substantially, so class weights alone do not give each participant equal training weight.

**Fix.** Validate split feasibility and fail explicitly when grouping cannot be honored. Select penalties with participant-disjoint inner data and the declared downstream aggregation. Decide and document whether each participant contributes equal total probe weight. Keep outer test participants outside SSL, tuning, and fitted preprocessing. Treat pretext-window validation only as optimization monitoring if retained.

**Scientific consequence.** Window-rich individuals must not silently determine a participant-level method comparison. Invalid inner partitions cannot be rescued by a correctly grouped outer score.

### A10 — FIX: baseline definitions and control reconstruction

**Problem.** There is no complete active original-CoST baseline, trivial baseline, or transparent endpoint handcrafted baseline. Some controls do not reconstruct the evaluated architecture correctly.

**Evidence.** The active `baselines/` implementations are staged for deletion; important variants survive only in Git/history/results. `tasks/_common.py` rebuilds an untrained `single`-band plan as the default multi-band encoder in the counterexample. `RawProjection.encode` refits scaling from every input batch/cohort; changing one row changes other rows' representations by up to 1.545 in the example. Historical `plain_ssl` also changes objectives and capacity; historical cosinor code can replace failed fits with zero features and uses cache identifiers without a content hash.

**Fix.** Keep explicit categories: training-majority classifier; named descriptive/cosinor features with valid-fit flags; raw flattened input/PCA; architecture-matched untrained DSSL; upstream CoST with a disclosed downstream adapter. Fit all data-dependent transforms on training data once. Reconstruct controls from complete saved architecture settings and verify exact parameter shapes/readout geometry. Do not restore every legacy baseline merely because it has a name.

**Scientific consequence.** Baselines isolate different questions: compression, architecture, handcrafted rhythms, and SSL method. A modified control or moving coordinate system can reverse the apparent value of representation learning.

### A11 — FIX: statistics and hypothesis verdicts

**Problem.** Some uncertainty calculations and success statements are not supported by the design.

**Evidence.** `utils.py` uses a hard-coded t multiplier 2.010 even for 30 estimates, whose conventional t-based 95% multiplier at 29 degrees of freedom is about 2.045. The correction uses planned fold/repeat counts even if records are missing. Degenerate zero-variance contrasts are declared significant without establishing a valid sampling model. The manuscript/report combines repeated DeLong differences by dividing their mean by the mean SE; this is not a justified variance estimate for a repeated-CV mean. Overlapping training sets and repeated participants require care beyond pairing two score vectors. The RQ2 minimum of six people is motivated by a sign test that is not the reported concordance test.

**Fix.** Require complete matched records and predeclared contrasts; report effect sizes, seed variability, and clearly scoped intervals. For fixed out-of-fold predictions, a paired participant-cluster bootstrap can describe conditional evaluation uncertainty, with all repeats for a person kept together; it does not include retraining uncertainty. Do not pool repeated participants as independent observations. Retain corrected resampling tests only with explicit applicable assumptions, actual sample sizes, degrees of freedom, and degenerate-case handling. The [RQ specification](RQ_SPECIFICATION.md) distinguishes these proposals from the manuscript.

**Scientific consequence.** The displayed RQ1 medians are worse than PCA for MESOR (0.998 vs 1.000) and phase error (0.606 vs 0.592 h); with no declared tolerance/test, “H1 holds” is unsupported. H2 requires timing **and** strength, so timing improvement and strength deterioration do not support H2 as a whole. RQ3 AUROC 0.6993 vs untrained 0.7159 and raw 0.6944 does not establish H3. Nonsignificance is not evidence of parity/equivalence. These are manuscript-reported numbers, not newly reproduced results.

### A12 — FIX: result provenance, aggregation, and resumption

**Problem.** Outputs do not reliably identify a complete experiment, and parallel jobs can corrupt the effective experiment manifest.

**Evidence.** Among 1,750 indexed `metrics.json`, `fold.json`, `eval.json`, and `plan.json` records, none exposes `git_commit` or `data_hash` at the top level or inside `config` (the exact fields inspected). This does not rule out information elsewhere. `train.py` performs non-atomic read/modify/write updates of a shared plan, and fold paths can be shared across weighting shards. Plan records omit important optimizer/augmentation/configuration details. Evaluation accepts row-index alignment without verifying dataset content/window identities and merges partial results. `cost.py` saves inference weights/readout/iteration count, but not complete optimizer, EMA/queue, scheduler, and RNG state for resumed SSL.

**Fix.** Save an immutable resolved configuration, code/data/split hashes and explicit window IDs; use one writer per run directory and an atomic completion marker. Aggregation must reject duplicates, mismatched splits, changed data, and incomplete expected folds. Separate an inference checkpoint from a full resumable training checkpoint. Save predictions and per-unit scores, not just summaries.

**Scientific consequence.** A plausible metric file is not evidence that the named method ran on the intended cohort. Existing outputs cannot be promoted to reproducible final results without a source-to-result manifest.

### A13 — REMOVE from the active path / VERIFY before deletion: accumulated experiments

**Problem.** Historical sweeps, exploratory readout ladders, and duplicate artifacts obscure the core questions.

**Evidence.** The inventory includes 11,902 PNGs in `results_hrd` and 1,880 in `results_hrd_energy`. Hashing found 2,876 exact duplicate groups / 3,787 extra copies across audited text/table/figure files. Two JSON files are empty/invalid: `results_hrd/19937323/tcn_time2vec_seed42/frequency_spectrum.json` and `hrd_rhythm.json`. Four `.pt` checkpoints were found, all in `results_hrd/20093940`, for `tcn_none_seed42` and `tcn_circular_seed42` (encoder and plain encoder). The exact `results_eq_h` run directory named by the current manuscript is absent; detached evidence such as `eq_h_full.npz`, `_evaljson`, and text summaries needs a verified linkage.

**Fix.** Archive whole historical runs with an inventory and hashes before deduplicating anything. Preserve unique checkpoints, representations, predictions, source snapshots, and manuscripts. Remove obsolete entry points, optional anti-collapse/readout sweeps, bridge/ranking scripts and disconnected manuscript sections from the supported workflow once their unique information is accounted for. Keep one implementation per primary task. A byte-identical figure is a deletion candidate, not authorization to discard its run context blindly.

**Scientific consequence.** Archival preserves negative and abandoned results and prevents selective retention of attractive figures. No file has been deleted in Phase A.

### A14 — KEEP core interface / VERIFY future variants: backbones and temporal encodings

**Problem.** The desired comparison is not currently executable or controlled.

**Evidence.** The staged rewrite removes `model_build.py` and `models/positional_encoding.py`; active training constructs only a TCN. Legacy Transformer/TCN settings used different dimensions, and tokenization can alter the entire head geometry. Legacy encodings include sinusoidal, learned, Time2Vec, relative, convolutional/stochastic, disentangled, and calendar variants. Some relative encodings hard-code 96 bins/day; ConvSPE draws random noise during evaluation. No active Mamba implementation exists. The manuscript's complex resonant S4D-style construction is not the same as [reference Mamba-1](https://raw.githubusercontent.com/state-spaces/mamba/main/mamba_ssm/modules/mamba_simple.py).

**Fix.** First freeze TCN with no added temporal encoding for RQ1–RQ3. Later implement a direct common `(B,T,C) -> (B,T,D)` backbone contract, unsupported-variant errors, deterministic inference, and measured parameter/update budgets. Keep heads, readout, information, augmentations, splits, and probe fixed. A few justified temporal encodings are sufficient; do not restore all historical variants. Any custom SSM must be named as a modification.

**Scientific consequence.** This separates backbone effects from capacity, tokenization, readout width, clock covariates, and training budget. The current manuscript calls RQ4 packages not parameter-matched; that conflicts with the user's requested controlled comparison and must be revised explicitly later.

### A15 — FIX: Narval execution is not ready

**Problem.** Existing scripts can start the wrong experiment, overwrite shared files, or fail before logging; they cannot resume complete SSL state.

**Evidence.** `scripts/oneshot.sh` embeds an account and personal project path, creates `logs` only inside the job although SLURM opens output earlier, merely prints CUDA availability, and defaults to 10 folds and four arms. Defaults differ from the manuscript's selected one-arm, level-equivariant model. Changing `DATASET` does not automatically establish strict GLOBEM LODO. Dependencies are not fully pinned. The A100 MIG resource spelling is supported by Alliance/SHARCNET documentation; it was not classified as a defect merely because it is unusual.

**Fix.** Generate a fixed run matrix after the protocol is settled; create log directories before `sbatch`; pass account/project through explicit environment/arguments; validate GPU, data/config hashes, task bounds and dependencies; give every array task a unique run directory; save resumable state. Run the final entry points in smoke mode before publishing tested submission commands.

**Scientific consequence.** Cluster success requires the intended experiment and complete artifacts, not merely SLURM's `COMPLETED` status. No large array was submitted, and no new `sbatch` command is represented as tested in this audit.

## 4. Dataset audit: measured current inputs

| Property | HRD | GLOBEM |
|---|---|---|
| Local source | `datasets/HRD_RAW_MinuteLevel.csv` | `datasets/GLOBEM_REDUCED.csv` |
| Export size | 53,575,181 rows; 166 IDs | 278,408 rows; 705 participant-year IDs |
| Current cache | `hrd_2224103.npz` | `globem_windows.npz` |
| Retained IDs / labelled / positive | 152 / 114 / 52 | 702 / 669 / 269 |
| Label consumed | `depression_status_endpoint`; raw export has 118 labelled IDs | `LABEL_ENDPOINT`; derivation requires source manifest |
| Input shape | 3,890 × 672 × 3 | 6,801 × 112 × 12 |
| Sampling | Minute export, averaged to 15-minute bins | Four ordered six-hour feature segments/day |
| Windows | Seven days; non-overlapping; first-sample anchor | 28 days; seven-day stride; Monday anchor |
| Normalization | Full participant record mean/sample SD before bin/window completion | Full participant observed values mean/population SD, then interpolation/edge carry |
| Missingness handling | Wear filter, partial limited interpolation, bin/window gate, interior/edge completion | Continuous grid; unlimited interpolation/edge carry; absent channel -> zero; ≥50% timesteps with any observation |
| Observation masks saved in shipped cache | No | No |
| Current split policy | Labelled participant-disjoint repeated folds; all non-test IDs available to SSL | Participant-year folds or LODO; held-out-year unlabelled exposure confirmed |
| Interpretation limit | Subject-relative completed signals; endpoint observation cutoff and source attribution unresolved | Coarse aggregate features; repeated-person linkage and endpoint derivation unresolved |

HRD raw channels include heart rate, floors, fairly/lightly/very active and sedentary minutes, steps, sleep stage, screen, and call. The loader selects heart rate, steps, derived sleep and screen; cache construction then removes steps. Floors, intensity/sedentary counts, and calls need an explicit semantic/redundancy decision. Keeping useful activity does not require treating all overlapping derived counts as independent sensors.

HRD row-wise missingness is 17.13% heart rate, 0.879% steps/activity/floors, 72.43% sleep-stage field, and 0% exported screen/call. Sleep-stage absence is not directly a non-wear rate: the loader infers awake when heart rate is present. Zero-filled event exports also do not prove continuous phone observation. These rates exclude absent timestamp rows.

GLOBEM keeps three steps features (step sum, active-bout duration, sedentary episodes), three sleep features (asleep duration, awake duration, sleep ratio), two screen features (unlock count/duration), and four location features (home time, entropy, distance, transitions). Bluetooth scan count and Wi-Fi unique devices are excluded with an unsupported blanket claim that proximity cannot reflect behavioral rhythm. Their inclusion/exclusion should be explicit and justified. Detailed column names and rates are in [globem_raw.json](evidence/globem_raw.json).

GLOBEM missingness is approximately 38.5–44.8% steps, 66.9–68.9% sleep, 30.7% screen, 28.7% location, 48.6% Bluetooth, and 58.7% Wi-Fi. No duplicated `(pid,date,segment)` rows or conflicting nonmissing endpoint values were found. Rebuilding the existing GLOBEM pipeline reproduced both window IDs and `X` bit for bit; that validates reproducibility of the current transformation, not its scientific appropriateness. Neither cache has nonfinite inputs or duplicate window IDs.

## 5. Repository map and proposed small active structure

Current important files, with disposition:

| File | Purpose and disposition |
|---|---|
| `cost.py` | Augmentations, SSL trainer and readout; KEEP core, rename/clarify DSSL and simplify optional experiments. |
| `models/encoder.py` | TCN plus trend/seasonal heads; KEEP core, explicitly distinguish reference and harmonic variants. |
| `models/dilated_conv.py` | Dilated residual convolution blocks; KEEP. |
| `models/losses.py` | Contrastive losses and experimental weight/equivariance/variance terms; KEEP essentials, isolate modifications. |
| `train.py` | Fold-based SSL and representation export; FIX config, manifests, resumption and collisions. |
| `eval.py` | RQ dispatch and result merging; FIX validation and immutable output semantics. |
| `datautils.py` | Cache/cohort abstraction and folds; KEEP small interface, FIX split contracts. |
| `data_processing/hrd_config.py` | HRD schema and preprocessing defaults; KEEP explicit settings, reduce redundant aliases. |
| `data_processing/hrd_clean.py` | HRD sensor/label cleaning; FIX time grid and gap handling. |
| `data_processing/hrd_windows.py` | HRD binning, normalization and windows; FIX boundary, masks and time origin. |
| `data_processing/hrd_dataset.py` | HRD orchestration and metadata; KEEP shared return contract, remove score-driven default sensor removal. |
| `data_processing/globem_dataset.py` | Segment-grid GLOBEM preprocessing; FIX provenance, labels/missingness contracts. |
| `scripts/build_cache.py` | Shared dataset cache CLI; FIX explicit sensor lists and metadata/hash persistence. |
| `tasks/recovery.py` | RQ1 cosinor ridge probes/PCA; FIX grouped selection, untrained diagnostics and per-channel outputs. |
| `tasks/rhythm.py` | Cosinor math, perturbations and personal references; KEEP mathematics, FIX chronology and unsupported resolution behavior. |
| `tasks/deviation.py` | RQ2 timing/strength evaluation; KEEP separate arms, FIX reference eligibility and per-unit results. |
| `tasks/depression.py` | RQ3 primary and extensive secondary probes; KEEP primary, archive unsupported/exploratory ladder defaults. |
| `tasks/_eval_protocols.py` | Fitted probe pipelines and selection; FIX selection unit and degenerate groups. |
| `tasks/_common.py` | Representation controls/loading/index helpers; FIX moving random projection and incomplete reconstruction. |
| `tasks/report.py` | Aggregate reports and statistical comparisons; FIX completeness, post-selection and claims. |
| `utils.py` | Paired statistical calculations; FIX scope, sample sizes, intervals and edge cases. |
| `scripts/rq3_bridge.py` | Natural emotional-energy relation analysis; preserve as exploratory RQ2 context, clarify naming. |
| `scripts/backbone_rank.py` | Historical backbone score aggregation; archive pending controlled RQ4. |
| `scripts/cross_run_delong.py` | Cross-run AUC comparison; archive/replace with validated paired result comparison. |
| `scripts/oneshot.sh` | Training array; replace after the final protocol is frozen. |
| `scripts/eval_array.sh` | Evaluation array; replace with config-aware task matrix and complete artifact checks. |
| `scripts/bridge_array.sh` | Auxiliary emotional-energy array; remove from primary workflow. |
| `README.md`, `CLUSTER.md`, `CC.md` | Existing usage and experiment notes; preserve history, consolidate supported instructions. |
| `requirements.txt`, `NOTICE`, `third_party/` | Environment and upstream attribution; KEEP attribution, pin supported environments. |
| `SSL_Rhythmicity/` | Manuscript and its independent Git history; preserve and reconcile scientific authority. |
| `docs/audit/validate_audit.py` | Reproduces this audit's inventory, data summaries, counterexamples and tiny execution checks. |

A suitable next structure needs only dataset modules, a DSSL module, an encoder module, losses/augmentations, three RQ modules, shared evaluation/statistics, one training CLI, one RQ CLI and a SLURM dispatcher. Avoid a top-level Python package named `ssl`, which can shadow Python's standard-library `ssl`. Moving working files solely to resemble a template would add churn without fixing the science.

## 6. Validation performed and remaining gates

These commands were executed locally from the repository root:

```powershell
python docs/audit/validate_audit.py --raw-hrd
python docs/audit/validate_audit.py --section checks
python docs/audit/validate_audit.py --section provenance
```

The first command inventories and validates the existing data/model, including a streaming HRD pass. The second reruns focused checks; the third requires the inventory and hashes existing artifacts. Evidence is written only under `docs/audit/evidence/`. Re-running refreshes evidence but preserves the initial Git snapshots. These are audit commands, not final experiment commands.

| Check | Result |
|---|---|
| Read HRD and GLOBEM real caches, dimensions/finite values | PASS for current caches |
| GLOBEM reconstruction from complete reduced CSV | PASS: exact `X` and window IDs |
| Tiny augment/forward/loss/backward on both dataset geometries | PASS: finite losses, gradients and representations |
| Save/load tiny inference checkpoint | PASS: exact representation round-trip |
| Tiny participant-disjoint downstream logistic evaluation | PASS: 8 training / 4 test participants, no overlap |
| Save structured audit/probe outputs | PASS under `docs/audit/evidence/` |
| Adversarial invariants for gaps, chronology, controls, split fallback | FAILURES REPRODUCED as documented above |
| Full HRD cache rebuild with corrected preprocessing | NOT RUN |
| Production-size DSSL / Transformer / Mamba training | NOT RUN; latter two not active |
| Complete experiment CLI and output-layout smoke test | NOT YET AVAILABLE |
| Full-state restart equivalence | NOT IMPLEMENTED |
| CUDA/Narval execution | NOT RUN |

Environment: Python 3.13.5, PyTorch 2.9.1+cpu, NumPy 2.3.3, pandas 2.3.3, SciPy 1.16.2, scikit-learn 1.7.2. The tiny model used a reduced width/depth, one or two optimizer steps, and real cached windows. Its downstream scores are deliberately labelled `execution_only_not_scientific_result`; they must not enter the manuscript.

## 7. Next phases and deliverable status

Phase A has produced a discrepancy report and reproducible evidence, with the scope limits above. Phases B–H remain. Proceed in this order:

1. Resolve manuscript authority and endpoint/person-linkage provenance; freeze the RQ specification without choosing rules from test scores.
2. Archive uniquely valuable legacy code/results with manifests; simplify the supported path while preserving staged work and Git history.
3. Correct and version data transformations; restore HRD steps; verify timestamp/mask/window/label/split assertions and rebuild caches with exclusion summaries.
4. Establish a minimal reference-derived DSSL and faithful CoST control, with explicit modifications and complete checkpoints.
5. Repair primary RQ evaluation, per-unit artifacts and statistics; verify synthetic signal recovery/perturbation semantics as well as execution.
6. Add the small future backbone interface without activating an RQ4 sweep.
7. Run final end-to-end smoke commands and only then supply the Narval matrix and exact submission commands.

The first confirmatory matrix should contain one frozen TCN/no-added-encoding DSSL configuration per eligible dataset, its untrained control, and required raw/handcrafted/reference controls, on identical splits and several recorded initialization seeds. Three seeds are a reasonable minimum execution plan, not evidence of sufficient statistical power. Transformer × Mamba × every historical encoding is not an immediate RQ1–RQ3 matrix. GLOBEM RQ2 is not scheduled under the current manuscript; silently making coarse surrogate shifts would create a different experiment.

Proposed final output convention (not implemented yet): `results/<dataset>/<protocol_id>/<method>_<backbone>_<encoding>/seed_<seed>/fold_<fold>/`, containing resolved configuration, split manifest, training checkpoint, representations with window IDs, per-unit predictions/targets, metrics, and completion status. RQ summaries/figures derive only from complete matching per-fold artifacts; SLURM logs use `%A_%a` in a pre-created `logs/<protocol_id>/` directory. Sharing one pretrained encoder across RQs is appropriate when the split and training contract are identical; copying/retraining it separately for each RQ is unnecessary.

**Remaining explicit inputs:** which manuscript version defines the intended study; the GLOBEM person-year linkage, if available; and source/export records establishing endpoint instruments, thresholds and observation cutoffs. These are scientific inputs, not permission for routine reversible cleanup.
