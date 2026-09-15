# Governing scientific protocol

Decision date: 2026-09-14; corrected after the author's explicit RQ2 clarification. The original scientific questions are authoritative, while later scientifically justified refinements may be retained. This supersedes the unresolved-authority note in `docs/audit/README.md`. The audit and its extracted current-manuscript specification remain historical records, not the governing protocol.

## Scientific intent and numbering

The early repository manuscript (`SSL_Rhythmicity`, commit `0120be4`, `sections/4b-representation.tex`) asks whether embeddings retain **individual** amplitude, phase, relative amplitude (RA), interdaily stability (IS), and intradaily variability (IV), and how rhythm preservation relates to prediction. That snapshot does not contain the complete authoritative numbering. The author has explicitly restored the original RQ2 wording: **“RQ2 — Personalized Rhythmic Phenotyping: Can unlabeled personal baselines derived from these representations detect within-person rhythmic deviations?”** This wording governs the code and manuscript. The prior rescue mapping of RQ2 to between-person clinical association was a scientific change and is retracted.

| RQ | Governing question | Evidence required | Relation to later manuscript |
|---|---|---|---|
| RQ1 — individual rhythm preservation | Does the frozen representation preserve individual rhythmic characteristics beyond a common daily pattern? | Held-out-person recovery of amplitude, phase, RA, IS and IV; compare trained/untrained/PCA; separate between-person and within-person information. | Retain per-channel cosinor probes and circular phase errors; restore omitted nonparametric markers and untrained controls. MESOR is supplementary, not a replacement for rhythmicity. |
| RQ2 — personalized rhythmic phenotyping | Can unlabeled personal baselines derived from the representations detect within-person rhythmic deviations? | For each held-out person, compare representation distance from four preceding contiguous weeks with raw 24-hour rhythm deviation under controlled timing and strength changes. | Restore the original personalization question. Keep paired, bidirectional perturbations and strict chronological eligibility as refinements; endpoint labels never enter RQ2. |
| RQ3 — downstream utility and the rhythm–prediction link | Does SSL yield useful depression representations, and does better preservation of individual rhythm accompany better prediction? | Matched frozen probes against raw, untrained, handcrafted and reference-CoST features; a prespecified association between held-out rhythm-recovery quality and prediction error. | Retain participant-level discrimination and grouped tuning; remove unrestricted readout searches from the primary experiment. |

RQ4 remains deferred: backbone/temporal encoding comparisons must not become a search for a successful RQ1–RQ3 result.

The original text's informal `corr(person_amp, AUC)` is not a literal implementable statistic: AUROC is defined over a set of positive/negative people, not one person. Use per-person held-out rhythm reconstruction error and per-person proper prediction loss for an explicitly observational bridge analysis. Do not correlate a single cohort AUROC with a vector of individual amplitudes. A later cross-configuration association can be descriptive under RQ4; configurations and seeds are not independent populations. No bridge analysis establishes causation.

## Later changes: adjudication

| Change | Decision | Scientific reason |
|---|---|---|
| Separate channels for rhythm targets | KEEP | Pooled phase across sleep/activity/heart rate has no consistent physiological referent. |
| Circular phase prediction/error | KEEP | Angles wrap; a linear difference gives incorrect errors at midnight. |
| Window targets plus held-out-person evaluation | KEEP AND EXTEND | Tests local rhythm fidelity; add individual-profile summaries so a common daily pattern cannot satisfy the original question alone. |
| Replace RA/IS/IV with only three cosinor markers | REJECT | Narrows the original construct to match available code. Restore the omitted measures with explicit domain/coverage rules. |
| Exclude the untrained RQ1 control because it performs well | REJECT | That control is needed to distinguish learning from architectural bias. |
| Fixed historical references and bidirectional, equal-norm amplitude perturbations | KEEP WITH REPAIRS | A meaningful controlled test of rhythm sensitivity; references must actually be preceding and contiguous. |
| Declare synthetic perturbation sensitivity sufficient clinical evidence | REJECT | RQ2 tests personalized rhythmic deviation detection, not clinical validity. Synthetic changes give controlled ground truth but do not establish natural clinical change. |
| Participant-disjoint splits and participant-level predictions | KEEP AND REPAIR | Match person-level generalization and endpoints; resolve GLOBEM person-year linkage separately. |
| Grouped inner tuning, trained-only feature transforms | KEEP AND REPAIR | Prevents tuning leakage and matches the outer prediction unit. |
| HRD steps removal selected by AUC | REJECT | Activity is central to the original rhythmicity question; restore the default activity channel. |
| Hand-selected harmonic bands, score-derived loss weights, extra equivariance/anti-collapse modules | DEFER FROM DEFAULT | Multiple unisolated changes do not follow from the original hypothesis; preserve as exploratory historical variants. |
| Omit amplitude scaling from DSSL augmentation | KEEP AS OUR MODIFICATION | The original manuscript deliberately preserves rhythm strength; disclose this difference from CoST. |
| Add smoothing | DEFER FROM DEFAULT | Not required by the original augmentation definition, and its transfer function changes rhythm magnitude. |
| Circular rather than raw-angle phase similarity | KEEP AS AN EXPLICIT MODIFICATION | Corrects angular geometry independently of performance; retain the original behavior in the reference-CoST control. |
| GLOBEM cannot support the exact HRD sub-hour perturbation protocol | KEEP LIMITATION | Do not generate or mislabel coarse surrogate shifts. Other original clinical/rhythm questions still apply where their measurements are identifiable. |
| Treat CES-D and BDI endpoints as one interchangeable label | REJECT | Use the exported endpoint without inventing thresholds; verify source instrument, date and derivation for each dataset. |
| “No significant improvement” means parity | REJECT | Equivalence needs a prospectively justified margin and appropriate inference. |
| Preregistration asserted by comments written around results | REJECT | Existing results remain exploratory; this dated protocol is also a post-audit amendment, not retroactive preregistration. |

## Operational constraints

1. Do not change labels. Preserve unknown labels explicitly and reject malformed/conflicting endpoint values. GLOBEM person-year linkage and survey-to-label provenance remain external uncertainties; report them instead of guessing.
2. Keep raw-unit window values and original per-channel observation masks available. RA requires a meaningful nonnegative signal and cannot be calculated from arbitrary z-scored values; do not report standard activity RA for every sensor by default. IS/IV require explicit sampling and valid-data rules. Zero variance, zero denominator, and insufficient cycles produce unavailable targets, not fabricated zeros.
3. Keep preprocessing changes versioned and report cohort/window/channel differences. The first repair addresses established code defects; full-record participant standardization remains explicitly retrospective until a separate normalization decision is implemented. Never call it prospective merely because it does not cross person IDs.
4. Keep the original two-branch CoST-derived architecture understandable. A reference CoST configuration and DSSL must be named separately, with augmentation, phase-loss and readout differences recorded. No result-selected harmonic/weight/regularizer sweep belongs in the initial matrix.
5. Freeze one TCN/no-added-encoding DSSL condition, shared splits and several explicit initialization seeds for the first experiments. Raw/PCA/untrained/handcrafted/CoST controls answer different questions; none may be removed for winning.
6. Report per-channel/per-person outcomes and paired effects with participant-aware uncertainty; do not use one favorable metric, seed, channel or condition to declare a compound hypothesis supported. Clinical association is not clinical diagnosis or causation.
7. RQ2 is the HRD phase/strength personalized-baseline experiment. It must remain label-free and within person. GLOBEM's overlapping 28-day windows and six-hour bins do not support the same declared experiment, so it is marked not applicable rather than given surrogate shifts.

## What counts as evidence

- RQ1: demonstrate held-out individual marker recovery and the contribution of SSL against the untrained control, rather than only recovery of the population daily template. Report each marker/channel and its uncertainty. No numeric noninferiority tolerance will be derived from old scores.
- RQ2: report timing and strength concordance with raw rhythmic deviation, contributing held-out people/windows, and paired effects against the untrained control. Timing success cannot cancel a strength failure. Synthetic sensitivity supports controlled deviation detection but does not establish natural or clinical validity.
- RQ3: report participant-level AUROC with balanced accuracy/macro-F1 for the frozen, matched probe; compare to both untrained and raw controls. Analyze the rhythm-recovery/prediction-loss relationship with person-level pairing, conditioning/reporting endpoint class where needed so different error distributions are not mistaken for a mechanism. Treat this bridge as exploratory unless an independent confirmation design is available.

The operational amendment in `docs/RUNNING.md`, `configs/hrd.json`, `configs/globem.json`, and `SSL_Rhythmicity/sections/rescue-protocol.tex` fixes the initial execution protocol. It restores label-free personalized rhythmic phenotyping as RQ2. Between-person endpoint association is retained only as a secondary analysis. Narval GPU verification remains a separate prerequisite.

## Operational amendment, 2026-09-14

- Fit normalization on observed training-participant channel moments, with equal participant weights. Apply it unchanged to test people. Full-record per-person z-scores in caches are retained for historical compatibility but not used by the corrected runner, because they remove individual amplitude information.
- Use only GLOBEM's earliest study cohort (2018) in the initial matrix. This avoids cross-year person leakage without inventing a linkage map; it limits external validity and is not leave-year-out evaluation. Preserve all cohorts in the cache.
- RQ1 initially evaluates individual mean profiles, including training-population-mean, untrained, raw and PCA controls. The common-template baseline explicitly checks whether the encoder captures individual differences. Within-person recovery is not yet evaluated.
- Freeze participant-level feature aggregation before probing. This replaces the prior window-fitted classifier/mean-probability protocol; the change gives each participant one training row and matches the outcome unit. Inner tuning is stratified at that same unit.
- Use five fixed outer participant folds, three initialization seeds and 1,000 updates per learned method. These settings are a disclosed resource-bounded starting protocol, not reference benchmark defaults or an optimized budget. There is no automatic superiority flag.
- RQ2 uses four preceding contiguous HRD weeks as an unlabeled personal baseline and tests controlled phase/strength deviation concordance against raw 24-hour rhythm. GLOBEM is not applicable to this exact weekly protocol. Raw participant marker associations with endpoints remain a secondary analysis with Holm correction. RQ3's exploratory bridge remains separate and does not establish causality.
- Execution amendment (same date, before any Narval run; no study results existed): the matched ladder adds the manuscript-designed rungs that the runner lacked — random projection (RQ1–RQ3, scaled on training windows), nonparametric descriptors, the Yan et al. cosinor of the HRD reference paper, the NNLS handcrafted super learner, and the secondary random-forest probe (reported separately). RQ1 ridge scaling is fitted once on all training participants, because refitting it on a target's eligible subset (as few as four people) produced errors above 10⁵ training SDs for GLOBEM raw features. The MoCo cross-entropy is computed as −log-softmax of the positive, identical in value, because CUDA NLLLoss is disallowed under deterministic algorithms.

## Model amendment, 2026-09-14 (investigator decision, before any Narval run)

This supersedes the normalization and budget items of the operational amendment above.

- The proposed method is the paper's DSSL, not a single-band variant: four harmonic seasonal bands, causal trend experts up to T/8, contracted seasonal weights (w_T 0.277, w_A 0.148, w_Φ 0.852), amplitude-weighted circular phase contrast, level equivariance w_eq = 1, smoothing up to 75 min, angle readout. `configs/*.json` instantiate it; `tests/test_paper_model.py` pins every setting; with the same seed it is bit-identical to the reported code (commit `0a88999`) in initial weights, augmentation, loss, gradients and encodings.
- Input normalization reverts to within-person standardisation on each participant's full record, as in the reported runs; it is label-free but uses each test person's own record. `normalization: "training_global"` remains available as a config switch.
- Each learned method gets 6,000 updates, the reported budget; 10% of training windows only monitor the pretext loss.
- The CoST reference adapter keeps upstream CoST's settings and is trained once per seed × fold, then shared by every backbone variant.
- With a circular phase readout, RQ1 and the logistic probe scale each (cos, sin) pair by one shared factor so phase geometry is preserved; the default angle readout has no pairs.

## Preservation

Before repairs, 80 current source/manuscript/note files and the root staged/unstaged patches were preserved in `archive/rescue_20260914/before_repairs.zip`; `manifest.json` records source hashes. Raw datasets, historical result directories, caches and checkpoints remain intact. Git retains the reference and older model implementations.
