# RQ1–RQ3: manuscript extraction and protocol repair

Status: scientific audit specification, **not a preregistration and not an implemented final protocol**. Sources are the current `SSL_Rhythmicity/sections/3-design.tex`, `4d-questions.tex`, `4a-setup.tex`, `4b-baselines.tex`, and `4c-metrics.tex` at manuscript commit `a69aa4e`. The initial Overleaf version `0120be4` has a broader rhythm-preservation motivation but not these exact numbered hypotheses. See [the discrepancy report](README.md) before adopting this document as the final contract.

“Current” below means manuscript/current-code behavior. “Proposed repair” is explicit research-method advice, not an invented claim that the manuscript already specifies it. No new mental-health labels or RQ4 experiments are introduced here.

## 1. RQ table

| RQ | Current hypothesis | Experiment | Baseline/control | Metric | Statistical test/status | Evidence supporting the hypothesis |
|---|---|---|---|---|---|---|
| RQ1: rhythm recovery | Frozen features linearly encode each window/channel's 24-hour MESOR, amplitude and acrophase at least as well as same-width raw PCA. | Train linear ridge readouts on training participants; score the frozen readouts on held-out participants' windows. | Current: training-fitted raw PCA. Proposed diagnostic: identically initialized untrained DSSL and a disclosed CoST comparator. | MESOR/amplitude R²; circular phase MAE in hours; per-channel results and participant-aware summaries. | Current manuscript explicitly descriptive, no paired test. Proposed: paired participant-cluster intervals; formal “at least as well” requires a prospectively defined tolerance/equivalence or noninferiority design, or strict improvement evidence. | Reliable recovery of all stated markers relative to PCA. Current point estimates and absent tolerance do not establish the full H1. An untrained comparison separately identifies the contribution of SSL. |
| RQ2: within-person deviation | Representation distance from the person's four preceding weeks follows raw 24-hour rhythm changes more often than an untrained encoder, for both timing and strength. | Perturb current week only; keep four contiguous earlier weeks and all learned parameters fixed. | Architecture/readout-matched untrained DSSL; explicit raw-cosinor and level-only calibrations. | Timing concordance and paired amplitude agreement, reported separately with trained-minus-untrained effects. | Current: corrected fold comparison. Proposed: retain participant pairing, save individual contributions, and use an explicitly scoped paired cluster analysis; do not treat windows/perturbations as independent. | Positive, reproducible trained-minus-untrained effects for **both** arms, with interpretable uncertainty and adequate eligible reference windows. Timing-only gain does not establish H2. |
| RQ3: depression detection | A frozen linear probe separates endpoint-positive participants better than the same probe on untrained features and raw windows. Current H3 names CES-D ≥16; GLOBEM's dataset section names another scale. | Grouped SSL/probe training and inner model selection; participant probability is mean of window probabilities. | Current primary: untrained and raw. Add training-majority, explicitly defined handcrafted/cosinor features, and original-CoST-derived features with a documented wearable adapter. | Primary participant AUROC; secondary participant balanced accuracy and macro-F1 at the fixed rule. | Current repeated-DeLong combination is not validated. Proposed: paired participant-cluster intervals for fixed OOF predictions, with seed variability separate and prespecified primary contrasts. | Positive, practically meaningful AUROC effects versus **both** primary controls across seeds, with uncertainty that supports the claim; secondary metrics describe consequences. Highest observed AUC alone is insufficient. |

## 2. Shared controls

- Freeze data export versions, sensor names/order/units, missingness rules, window alignment/stride, observation interval, normalization policy and labels before final evaluation. All arms use the same eligible cohort and windows. Keep signal targets in the declared units; participant z-scores are not physical units.
- Split at the participant level for HRD. For GLOBEM, a participant-year key is insufficient to guarantee person-disjoint evaluation; strict held-out-year SSL must exclude all held-out-year data, including unlabelled records. Resolve person linkage before claiming unseen-person performance.
- Record train/inner-validation/test person IDs and window start/end timestamps explicitly. Assert no forbidden overlap at every stage, including SSL and fitted feature transformations.
- Use identical splits, training update counts, batch size, optimizer/schedule, representation dimensions/readout policy, and augmentation settings for learned methods wherever the method comparison permits it. Record necessary reference-method differences rather than hiding them.
- Distinguish initialization seeds from split seeds/repeats. Repeated folds are not independent participants. Report mean ± SD across independently initialized runs on matched splits; record the full seed list and every planned run before launch.
- Retain the training-majority prediction rule for classification sanity checks. Permutation or random-score results, if used, must have explicit randomization/repetition rules; one random draw is not a meaningful baseline.
- Do not select backbones, encodings, sensors, loss weights, readout families, or decision thresholds from outer-test results. Extensive reuse of the existing cohorts for method development remains a limitation even after code repair.
- A metric being undefined due to absent classes, near-zero rhythm, or insufficient personal history must be recorded as unavailable with its reason and count, not converted into zero or silently omitted.

## 3. RQ1 specification

**Scientific hypothesis (current).** H1 concerns linear recoverability of each window's 24-hour rhythm markers, not depression discrimination, reconstruction of the entire waveform, or proof of independence between learned branches.

For a channel, fit the fixed-period model

\[
x_t=M+a\cos(2\pi t/B)+b\sin(2\pi t/B)+\epsilon_t,
\qquad A=\sqrt{a^2+b^2}.
\]

Define phase consistently with the implementation's complex coefficient convention and state whether it is a time-of-maximum or Fourier angle. Predict its sine and cosine and score circular distance; never subtract wrapped angles as ordinary real numbers. A synthetic shifted sinusoid must verify the sign convention and time origin.

**Experimental variable.** Frozen representation supplied to the same ridge-readout protocol: DSSL versus PCA. Untrained DSSL is an additional diagnostic of training benefit, not a substitute for the manuscript's PCA comparator.

**Controls.** Same windows/channels/targets, participant splits, probe penalty candidates and fitting budget. PCA and any feature scaling are fitted on training data only. Record both requested and achieved PCA width: `min(requested width, training rows - 1, input width)` can be smaller than the DSSL width. A capacity-matched claim must acknowledge that cap.

**Unit of analysis and split.** Prediction target: window × channel. Generalization unit: held-out participant. Windows from a participant must stay together in outer and inner splits. Current code's ridge leave-one-window-out tuning must be replaced by grouped tuning. Report each channel separately; collapsing all channels into a variance-weighted R² can hide a failed sensor.

**Metrics.** R² for MESOR and amplitude; circular phase MAE in hours. Report counts, distributions, negative-R² folds, and uncertainty. Phase becomes undefined at zero amplitude; a target reliability rule must be based on a stated scientific criterion or training-only calibration and shared by all methods. Record excluded targets. Current targets come from imputed, standardized cache values; observed-signal and physical-unit interpretations cannot be asserted without rebuilding the target pipeline.

**Statistical test (proposed repair).** Keep the current descriptive analysis honest if no margin is available. Save paired predictions and derive intervals by resampling whole participants, preserving channels, windows and repeats together. For a formal “no worse” claim, specify a scientifically defensible loss tolerance before evaluation for each endpoint and the multiplicity/conjunction rule. The manuscript supplies no such tolerances; this audit does not choose them from existing differences. Intervals conditional on existing predictions do not measure uncertainty from retraining on a new cohort.

**Success criterion.** All three marker families satisfy the declared comparison to PCA. To claim benefit specifically from SSL, DSSL must also improve the relevant comparison to untrained features. This second statement is stronger than current H1 and must be named separately.

**Failure / inconclusive criterion.** A marker fails the prespecified comparator criterion, or intervals cannot distinguish acceptable from unacceptable loss. High absolute R² alone does not establish noninferiority. In the current manuscript, MESOR and phase medians are slightly worse than PCA and no tolerance is given; “H1 holds” is not established.

**Required visualization.** Per-channel paired marker-performance plots for DSSL/PCA/untrained, with seed points and participant-aware intervals; phase truth-versus-prediction plotted circularly for a prespecified subset. Include target-amplitude/coverage distributions so interpolation and ill-defined phase cannot be hidden by a single mean.

**Dataset applicability.** HRD is primary. GLOBEM has a sampled daily component at four bins/day, so coarse daily rhythm recovery can be studied, but precise subsegment timing of the original behavior is not directly observed. Do not turn a regression output's decimal-hour precision into a claim of sensor timing resolution.

## 4. RQ2 specification

**Scientific hypothesis (current).** H2 tests whether frozen representation distance tracks both timing and strength changes relative to the same participant's recent rhythm better than an untrained encoder. Natural emotional energy is contextual/exploratory evidence, not the target that defines the two perturbation arms.

For a current week i, let the reference be exactly weeks i−4 through i−1. The current implementation computes featurewise reference mean and SD, then a standardized representation distance. Freeze these reference quantities for every perturbation of i. The raw comparator uses the complex mean of reference cosinor coefficients, retaining both amplitude and phase.

**Experimental variable.** Current-week timing shift or 24-hour amplitude modification, crossed with trained versus untrained representation. Do not change reference weeks, model weights, preprocessing fit, or distance scaling when perturbing the current week.

**Timing arm (current).** Circular shifts of 0.5, 1, 2, 3 and 4 hours on HRD sensor channels. Classify whether a perturbation moves raw cosinor distance closer to or farther from the personal reference; score stratified concordance of representation-distance change with that direction. Strata are participant × shift magnitude. This is not simply “larger shift gives larger distance.”

**Strength arm (current).** Make paired copies with only the 24-hour component multiplied by 1−α and 1+α for α in {0.05, 0.1, 0.2, 0.3, 0.5}. Compare which copy is farther from the same raw/reference rhythm and whether representation distance agrees. The equal perturbation norm is a useful control against a generic magnitude-of-change detector. These perturbations can make some standardized channels unlike physically admissible raw measurements; interpret them as signal-level counterfactual tests, not simulated clinical events.

**Controls.** Identical frozen architecture/readout at initialization; raw-cosinor calibration; a level-only null. Random projection is optional after fixing its moving scaler. Keep any channel selection, SD floor, weighting, amplitude reliability rule, perturbation levels and eligibility policy fixed in advance. Report actual shift bins/hours, never an automatically rounded label.

**Unit of analysis and split.** Held-out participant is the independent evaluation cluster. Repeated weeks and perturbations are nested observations. Require five chronologically ordered, contiguous, non-overlapping weekly windows; reference windows must end before the current window. Fit prospective normalization using only available past data if prospective monitoring is the intended claim. The existing full-record normalization supports only a explicitly retrospective interpretation until repaired.

**Metrics.** Timing concordance and amplitude agreement separately, their trained-minus-untrained differences, eligible people/weeks/strata and pair counts. The current timing score weights valid pairs; amplitude averages eligible week/level comparisons. Keep this distinction visible. A participant-weighted alternative changes the estimand and must be documented rather than silently substituted. Natural energy concordance uses absolute change from the same historical reference and remains secondary.

**Statistical test (proposed repair).** Save participant contributions and resample paired participant clusters, keeping their repeats and perturbations together. Define whether the interval targets pair-weighted or participant-weighted concordance. A corrected fold test may be reported only with justified resampling assumptions and complete paired folds. The six-participant sign-test argument in current code does not validate its concordance inference. Report instability/eligibility rather than manufacturing a universal significance gate from that argument.

**Success criterion.** Both timing and amplitude show positive trained-minus-untrained effects under the prespecified inference rule. Report practical effect magnitudes, not only thresholded significance. A positive timing effect with a negative amplitude effect is a qualified mechanism finding, not overall support for H2.

**Failure / inconclusive criterion.** Either arm shows deterioration, no credible improvement, or insufficient eligible participants/reference windows. “Not measurable” is not “model failure” and must not disappear from summaries. Current manuscript results support a possible timing gain but show amplitude deterioration, so H2 as a conjunction is not supported.

**Required visualization.** Two separate paired effect panels for timing/strength, with seeds and participant-cluster intervals; sensitivity across fixed perturbation magnitudes; a few deterministically selected participant timelines showing raw rhythm, reference interval, representation distance and eligibility gaps. Generate all panels from saved per-window/per-perturbation records.

**Dataset applicability.** The current manuscript explicitly excludes GLOBEM RQ2. Its six-hour grid and overlapping 28-day windows do not implement this HRD weekly experiment. Current code's auto-generated `[4,12,18]` hour levels even labels a rounded one-bin shift as four hours although one bin is six hours. A GLOBEM-specific study would require a separately justified protocol; it is not scheduled here.

## 5. RQ3 specification

**Scientific hypothesis (current).** Frozen DSSL features improve endpoint discrimination relative to both untrained features and raw input. This does not ask for the most accurate classifier obtainable from an unrestricted readout search.

**Experimental variable.** Representation supplied to the same class-balanced L2 logistic probe: trained DSSL, matched untrained DSSL, raw flattened window. Other baseline categories are reported separately and named accurately.

**Controls.** Same labelled people, windows, splits, feature-scaling fit, inner penalty grid and participant aggregation. Primary current grid is C in {0.001, 0.01, 0.1, 1}. Select on participant-level validation AUROC after averaging window probabilities, matching the outer estimand. Do not select C on window-level AUC. Record the treatment of variable window counts; equal total contribution per participant is a proposed protocol amendment if adopted.

**Unit of analysis and split.** One endpoint outcome and one final score per person. Outer and inner splits are person-disjoint, subject to the unresolved GLOBEM person-year mapping. The score is the arithmetic mean of the participant's eligible window probabilities; hard classification uses p≥0.5. This is retrospective aggregation over the declared pre-endpoint observation interval, not a per-week forecast. A different aggregation rule requires a separate comparison.

**Metrics.** AUROC primary; balanced accuracy and macro-F1 secondary under the manuscript's decision rule. Report prevalence, number of positive/negative participants, and confusion counts. Do not mix macro-F1 with positive-class F1 or window AUC with person AUC. Accuracy may be a sanity check, not the primary outcome. Regression/ordinal endpoints are outside this immediate binary RQ until their labels and hypotheses are specified.

**Baseline categories.** Training-majority predictions establish trivial performance. Simple distribution and explicitly specified cosinor features test whether learned representations add beyond transparent summaries. Raw/PCA test compression. Untrained DSSL isolates SSL from architecture. A pinned original-CoST implementation with a disclosed fixed wearable readout provides the reference-method comparison. None of these should be relabelled as another published method. The raw cosinor target itself is not a meaningful RQ1 recovery competitor, but cosinor features can be a meaningful RQ3 classifier input.

**Statistical test (proposed repair).** Predeclare DSSL−untrained and DSSL−raw as primary paired contrasts. Save one out-of-fold score per participant per repeat/initialization; keep repeated observations together in resampling. Report paired participant-cluster bootstrap intervals conditional on fitted models, seed-level mean/SD, and multiplicity handling for separately claimed comparisons. If a truly independent fixed holdout is available, paired DeLong on its fixed scores is an option; do not claim that averaging per-repeat SEs establishes inference for repeated cross-validation. Full training-procedure uncertainty would require a design that includes retraining, with its computational cost stated.

**Success criterion.** DSSL improves AUROC versus both primary controls with uncertainty and effect size consistent with the declared scientific claim, and the effect is not confined to one seed. If a minimum useful effect is required, specify it prospectively; this manuscript supplies no numeric minimum. Secondary metrics describe whether any gain is useful at the fixed operating point.

**Failure / inconclusive criterion.** DSSL does not reliably outperform either primary control, or evaluation/provenance cannot support the comparison. A broad confidence interval is inconclusive, not evidence of equality. Current reported AUROCs (DSSL 0.6993, untrained 0.7159, raw 0.6944) do not establish H3; they also do not prove the methods equivalent.

**Required visualization.** Paired participant-AUROC differences across initialization seeds and contrasts with clearly scoped intervals; participant-level ROC curves from saved OOF predictions with repeat handling stated; confusion matrices/operating-point metrics at the fixed threshold. Avoid picking the most favorable seed or plotting a standard error over correlated folds as if they were independent replications.

**Dataset applicability.** HRD's documented target is endpoint CES-D ≥16, subject to verifying export and survey timing. GLOBEM's local boolean endpoint must be traced to its instrument/version/threshold; the current manuscript's BDI-II description conflicts with H3's universal CES-D wording. Analyze the datasets separately; neither pooling endpoints nor changing their labels is authorized by this ambiguity.

## 6. Minimal run matrix and release gate

This is a proposed execution budget, not a finalized submission matrix:

| Dataset | Questions | DSSL backbone × encoding | Initializations | Required split status |
|---|---|---|---|---|
| HRD | RQ1, RQ2, RQ3 | TCN × none | At least three explicit seeds on shared fixed splits | Participant-disjoint; all chronology, data and label checks passed |
| GLOBEM | RQ1, RQ3 | TCN × none | Same declared seed policy | Strict held-out-year exclusion; person-linkage limitation resolved or explicitly scoped |
| GLOBEM | RQ2 | Not scheduled | Not applicable | Current manuscript says this experiment is not measurable |
| Both | RQ4 | Deferred | Deferred | Do not add Transformer/Mamba/encoding sweeps to rescue RQ1–RQ3 |

Each DSSL fit yields an exact initialization control and supports all applicable frozen evaluations on that split. The CoST reference uses the same data/probe protocol and separately declared method-specific settings. Trivial/raw/handcrafted baselines do not need unnecessary GPU training. Seeds cannot compensate for a small cohort, missing source labels, or reused test data.

Release requires the corrected real-data validation, synthetic rhythm/chronology checks, exact control reconstruction, finite training and gradients, inference and full-state checkpoint tests, participant-level downstream execution, immutable saved configuration/splits/predictions, and one complete end-to-end smoke run through the final CLI/output structure for each dataset. Until these pass, no large Narval array or final success claim is justified.
