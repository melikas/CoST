# Experiments tested and not retained

This is the single authoritative record of everything we changed, measured, and decided against.
It replaces six separate pre-registration and result documents. Each entry states what was changed,
why we expected it to help, what it was compared against, the actual numbers, and why it was
rejected.

**How to read the metrics used throughout.** All are defined once here so the entries stay readable.

| Term | Meaning | Direction |
|---|---|---|
| **RQ1 recovery** | How well a ridge regression, fitted on training participants, predicts a held-out participant's rhythm markers (24-h amplitude, acrophase in hours, interdaily stability IS, intradaily variability IV, relative amplitude RA, MESOR) from the frozen representation. Reported as *error of the control minus error of DSSL*, so positive favours DSSL | ↑ better |
| **"vs untrained"** | The same architecture with random weights, never trained. The strictest control: it isolates what *training* contributed, separately from what the architecture provides | ↑ better |
| **"vs raw"** | Participant mean of the flattened input window. A fixed anchor that does not change when the model changes | ↑ better |
| **own-branch R²** | Variance of a rhythm property explained by the branch that is *supposed* to carry it (amplitude from the seasonal-amplitude block, acrophase from the seasonal-phase block, MESOR from the trend block) | ↑ better |
| **leakage R²** | Variance of the same property explained by the *other* branch. If the trend branch can predict the 24-h acrophase, the branches are not specialised | ↓ better |
| **own − leakage** | Branch specialisation. Positive with a confidence interval above zero means the property really is carried by its own branch | ↑ better |
| **RQ2 timing / strength** | A week is perturbed in a known way — shifted in time by 0.5–4 h ("timing"), or its 24-h amplitude scaled up and down ("strength"). Concordance is the fraction of cases where the representation's distance from that person's own 4-week baseline moves in the same direction as the true change. 0.5 is chance | ↑ better |
| **RQ3 AUROC** | Participant-level area under the ROC curve for predicting the depression endpoint from the frozen representation, with a logistic probe | ↑ better |

Unless stated otherwise, "protocol grade" means the locked evaluation: 3 seeds × 5 participant-disjoint
folds, both cohorts, paired participant-bootstrap 95% confidence intervals.

---

## 1. Restoring upstream CoST's masking and queue size (rejected)

**What changed.** Two settings were returned to the values used by the original CoST paper:
timestep masking during training (`mask_mode="binomial"`, which randomly zeroes half the timesteps
of each view) and a much smaller memory queue of negatives (`moco_k` 4096 → 256).

**Why we tested it.** The trend branch is trained by instance discrimination: given a window, pick
its own augmented copy out of a queue of other windows. We measured that this task was solved
almost immediately — top-1 retrieval accuracy reached 1.000 by roughly iteration 1,600 in all 15
folds, meaning the remaining ~75% of training provided that branch with no gradient. We also found
the queue (4,096) was larger than the number of training windows (~2,900), so a window's own earlier
copies sat in its negatives. Upstream CoST uses both settings differently, so restoring them was the
most conservative possible fix.

**Compared against.** The same model with the shipped settings, HRD, seed 1, folds 0–1.

**Result.** Top-1 retrieval stayed at 0.992 — still effectively solved, so the task did not become
harder in any meaningful sense. Worse, the change damaged phase specialisation: own-branch R² for
the 24-h acrophase of the Steps channel collapsed from 0.765 to 0.000 while the trend branch still
predicted it at 0.522.

**Conclusion.** The masking and queue settings were not the reason the trend task was trivial, and
changing them harmed a property they do not touch. The damage reaching the phase branch was the
first concrete evidence that the two branches are not independent: they read from one shared
encoder, so a change aimed at one of them propagates to the other.

**Decision.** Rejected. Not retained in any later configuration.

---

## 2. Removing amplitude weighting from the phase loss (rejected)

**What changed.** The seasonal phase loss weights each representation dimension by its own
amplitude (`phase_mode="circular_amp"`). We switched to the unweighted form (`"circular"`).

**Why we tested it.** Weak-rhythm channels (Steps, screen use) had poor phase recovery. Because
weighting by amplitude gives low-amplitude dimensions almost no gradient, we hypothesised those
channels were simply never trained on.

**Compared against.** Same model with amplitude weighting, HRD, seed 1, folds 0–1.

**Result.** Phase recovery did not improve — Steps and screen own-branch acrophase R² both stayed at
0.000 — and this arm produced the worst overall rhythm recovery of any arm tested in that batch
(mean RQ1 gain over the untrained control −0.0595, with 0 of 5 marker families improved).

**Conclusion.** The amplitude weighting was not the cause of poor phase recovery.

**Decision.** Rejected.

---

## 3. Per-timestep normalisation in the readout destroys amplitude (fix retained)

**What changed.** The frozen representation's seasonal block was read out after L2-normalising each
timestep across channels — inherited from upstream CoST. We added the option to read the seasonal
sequence without that normalisation (`readout_norm="none"`) and made it the default.

**Why we tested it.** RQ2's strength arm was *below chance* (0.438, where 0.5 is chance), meaning
that when a person's 24-h amplitude was artificially increased, the representation systematically
moved as if it had decreased. Amplitude information appeared to be inverted rather than merely weak.

**Compared against.** The same encoder read out both ways. Crucially this was first established on
an **untrained** encoder, which proves the cause is the readout arithmetic and not training.

**Result.** Scaling a window's true 24-h amplitude by 0.5×, 1.0× and 1.5× moved the amplitude
feature to 8.06 / 12.20 / 14.09 under normalisation — compressive and nearly flat — versus
1.34 / 2.67 / 3.99 without it, which is exactly proportional. The mechanism: DSSL's harmonic bands
start at frequency bin 1, so the seasonal sequence has no constant component to anchor the norm, and
normalising drives it toward a square wave. At protocol grade the fix moved RQ2 strength from 0.438
(below chance, failing) to **0.713** (above chance, and beating the untrained control by +0.054).

**Conclusion.** This was a genuine defect in the readout, diagnosed mechanically rather than by
search, and it is the single change that moved a research question from "not supported" to
"supported".

**Decision.** **Retained.** It is now the hard-wired behaviour; the configuration flag has been
removed because the alternatives were tested and rejected (§6).

---

## 4. The level-equivariance loss coefficient `w_eq` (removed entirely)

**What changed.** The objective contained a fourth term rewarding the trend branch for predicting
the artificial level offset between two augmented views, with coefficient `w_eq`. We tested
`w_eq` = 1.0, 0.277, 0.05 and 0.

**Why we tested it.** Measuring the objective at initialisation showed this term contributed 44.9%
of the total loss — more than the amplitude (27.3%) and phase (27.6%) terms — while the trend term
contributed 0.3%. It was also the only term not scaled by the shared coefficient `alpha`, so its
weight had never been calibrated against the others.

**Compared against.** Each other, HRD, 5 folds, paired by fold.

**Result.** RQ1 gain over the untrained control improved monotonically as the coefficient fell:
**−0.0343 → −0.0136 → +0.0096** for 1.0 → 0.05 → 0, and the ordering held in **every one of the 5
folds**. Removing the term entirely was best.

**Conclusion.** The term was actively harming rhythm recovery. It was introduced to stop the trend
branch discarding a person's baseline level, but the level turned out to be recoverable from the
trend branch anyway (MESOR own-branch R² = 1.000 with the term switched off).

**Decision.** `w_eq = 0` adopted, and the coefficient, its projection head, its loss function and
the per-batch level offsets it consumed have now been **deleted from the code**, because a
permanently-zero coefficient is dead configuration.

---

## 5. Disjoint-day pairing for the trend task (rejected)

**What changed.** The trend branch's training pair was built from two *disjoint halves of the days*
of the same week, so the two views shared no timesteps, and a week's own earlier copies were
excluded from its negatives.

**Why we tested it.** We measured that the existing pair was almost a copy of itself: using the raw
input alone, the correct partner could be identified out of all 3,803 weeks with 89.1% accuracy.
Any encoder that preserves input identity therefore solves the task, which explains the saturation
in §1 and means the objective rewards memorising the window rather than representing its trend.

**Compared against.** The same model with the standard pairing, HRD, 5 folds.

**Result.** The task did become hard — top-1 retrieval fell from 1.000 to 0.006 — and rhythm leakage
into the trend branch fell substantially (Steps acrophase leakage down in all 5 folds). But rhythm
recovery got **worse in all 5 folds**, with RQ1 gain over untrained falling from +0.0096 to −0.0164.

**Conclusion.** The task went from trivially solved to essentially unsolvable. A contrastive
objective whose positive pair cannot be matched provides little useful learning signal, and the
harder task did not translate into a better representation.

**Decision.** Rejected. `trend_views` has been removed from the code.

---

## 6. Two alternative readouts: "timestep" and "split" (both rejected)

**What changed.** Having established §3, we asked whether a different split of the readout could
keep amplitude fidelity *and* recover phase quality. Three readouts were compared, all reading the
**same trained encoders** so that only the readout differed: `timestep` (both amplitude and phase
from the normalised sequence — the original), `none` (both from the raw sequence — the retained
default), and `split` (amplitude from the raw sequence, phase from the normalised one).

**Why we tested it.** The `none` readout, while fixing RQ2 strength, had made phase recovery worse
(RQ1 phase versus the untrained control moved from +0.033 to −0.023). `split` was designed to take
each quantity from whichever source measured it better.

**Compared against.** Each other, at protocol grade, on identical encoders.

**Result.**

| Readout | RQ2 strength ↑ (chance 0.5) | RQ2 strength vs untrained ↑ | RQ1 phase vs untrained ↑ | Acrophase own − leakage ↑ |
|---|---|---|---|---|
| `timestep` (original) | 0.399 — *below chance* | −0.075 (control wins) | **+0.033** | −0.097 |
| **`none` (retained)** | **0.713** | **+0.054** | −0.023 | −0.083 |
| `split` | 0.557 | −0.081 (control wins) | −0.019 | −0.097 |

*Interpretation.* `split` did not recover phase — the change from −0.023 to −0.019 is negligible and
still favours the untrained control — and it undid most of the amplitude gain, dropping RQ2 strength
from 0.713 to 0.557 and losing to the untrained control.

**An instructive detail.** The amplitude half of the representation was *bit-identical* between
`none` and `split`, yet RQ2 strength still collapsed. The reason is that the RQ2 distance is a mean
over all 1,760 representation dimensions, of which the phase block is 45%. Changing the phase block
alone changes the distance arithmetically. This is a property of the evaluation metric, not evidence
that amplitude and phase are entangled in the encoder.

**Decision.** Both rejected; `readout_norm` removed from the code and `none` hard-wired. Recorded as
a selection among three variants.

---

## 7. Pre-encoder decomposition into separate trend and rhythm towers (rejected)

**What changed.** Instead of one shared encoder feeding two readout heads, the *input* was split
first: a zero-phase 24-hour moving average produced a "trend view", its complement produced a
"rhythm view", and each fed its own independent encoder. The 24-h filter has exact zeros at the
daily frequency and all its harmonics, so the trend tower physically cannot see the daily rhythm.

**Why we tested it.** We measured that branch leakage is *geometric, not learned*: at random
initialisation the trend block already predicted 24-h amplitude at R² 0.827 and acrophase at 0.642.
Since the property exists before any training, no change to the loss can remove it. The shared
encoder's receptive field (1,021 steps) exceeds the window (672 steps), so every timestep encodes
the whole window, and the time-averaged trend readout inherits everything.

**Compared against.** The shared encoder, first at protocol grade (at 0.54× the parameter count),
then on a development cohort at **matched** parameter count to remove the capacity confound.

**Result — the mechanism worked.** Leakage fell exactly as predicted, and by nearly the same amount
in trained and untrained models, confirming it is structural:

| Leakage R² ↓ | Shared, trained | Shared, untrained | Decomposed, trained | Decomposed, untrained |
|---|---|---|---|---|
| 24-h amplitude | 0.843 | 0.827 | **0.454** | **0.451** |
| 24-h acrophase | 0.669 | 0.642 | **0.016** | **0.024** |

**Result — but the model got worse where it matters.** At protocol grade, RQ1 amplitude recovery
versus the untrained control flipped from +0.029 (favouring DSSL) to −0.039 (favouring the control),
phase recovery fell from −0.023 to −0.054, and RQ3 AUROC fell from 0.704 to 0.665.

**Result — and capacity was not the explanation.** We initially attributed the recovery loss to the
0.54× parameter budget. A controlled development run tested this directly by comparing the
decomposed architecture at 4.03M against the same architecture at 7.51M parameters:

| RQ1 metric vs untrained ↑ | Decomposed 4.03M | Decomposed 7.51M | Shared 7.45M |
|---|---|---|---|
| 24-h amplitude | −0.027 | **−0.027** | **+0.046** |
| Acrophase (hours) | −0.056 | −0.039 | **−0.003** |
| Acrophase own-branch R² ↑ | 0.464 | **0.416** | 0.515 |

*Interpretation.* Nearly doubling the parameters changed amplitude recovery not at all and made
acrophase recovery slightly worse. **The earlier capacity explanation was wrong.** The decomposition
itself costs recovery.

**Conclusion.** Disentanglement and recovery are in genuine tension, and the tension is structural
rather than a defect: the RQ1 probe reads the *whole* representation, so the unrestricted access
that creates leakage is the same access the probe exploits. Restricting a branch's input necessarily
removes information the probe was using.

**Decision.** Rejected as the canonical architecture. The implementation is archived rather than
deleted (see §10), because it is the only implementation of a quantitative result we report.

---

## 8. A finding that constrains every future attempt: training degrades phase

Across every architecture tested, self-supervised training made the representation **worse** than
random initialisation at recovering the 24-h acrophase, and the effect grew when the branch was
isolated:

| Architecture | Acrophase own-branch R², untrained ↑ | Same, after training ↑ | Cost of training |
|---|---|---|---|
| Shared encoder | 0.680 | 0.515 | **−0.165** |
| Decomposed, 4.03M | 0.658 | 0.464 | −0.194 |
| Decomposed, 7.51M | 0.621 | 0.416 | **−0.205** |

*Interpretation.* The seasonal objective is contrastive: it asks two augmented views of the same
window to produce the *same* amplitude and phase, i.e. it trains those quantities to be **invariant**
to augmentation. RQ1 and RQ2 measure the opposite property — whether amplitude and phase *track*
real changes. The objective is therefore optimising against the criteria, which is why an untrained
encoder can beat a trained one. This is consistent with the codebase's own reasoning for the trend
branch, which notes that an augmentation defines what a representation discards.

This finding is the strongest available argument that the remaining problem is the **training
objective**, not the wiring — and it is the basis of the open direction in §10.

---

## 9. An RQ3 screening result that did not survive audit (retained as a cautionary record)

A single-seed screen suggested the `w_eq=0` configuration reached RQ3 AUROC 0.726 against a raw
baseline of 0.695. An adversarial audit of that number found:

- **Most of the apparent gain over the untrained control was the control moving**, not the model
  improving. The untrained control's own AUROC dropped from 0.713 to 0.657 when the readout changed,
  because the control shares the model's readout. Measured against the fixed `raw` anchor, the model
  moved only +0.019.
- **The advantage rested on single folds.** Dropping one fold reduced the margin over `raw` from
  +0.032 to +0.006; dropping a different fold reduced the margin over the untrained control from
  +0.069 to +0.009.
- **Resampling folds rather than participants put both intervals across zero**: +0.032
  [−0.017, +0.119] and +0.069 [−0.027, +0.185].
- At three seeds the number fell to 0.704 and remained statistically indistinguishable from `raw`.

*Lesson recorded:* at 113 participants with ~22 per fold, per-fold AUROC ranges from 0.51 to 0.87 for
every method. Differences of ~0.03 cannot be established at this sample size, and a confidence
interval that holds the fold structure fixed will understate the uncertainty.

---

## 10. Open directions, implemented but never evaluated

Kept in `archive/failed_experiments/dssl_v2_decomposition/` and **not** part of the canonical model.
Neither has been tested, so neither is claimed to work.

- **Seasonal equivariance objective.** Instead of asking two views to agree, it applies a known
  scaling and rotation to one input harmonic and requires the latent coefficient to transform
  correspondingly. It targets §8 directly: it would train the representation to *track* amplitude and
  phase rather than discard them. Implemented with the transformation deliberately applied to the
  12-hour harmonic, because RQ2 probes the 24-hour amplitude and whole-window timing — keeping RQ2 an
  independent test rather than a restatement of the training objective.
- **Day-local receptive field for the rhythm tower.** Caps the rhythm encoder's receptive field
  below the window length so it cannot encode week identity, addressing §5's shortcut structurally
  instead of through the loss.

Both would need evaluation on the development cohort before any claim is made.
