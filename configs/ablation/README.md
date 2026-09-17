# Minimal-repair ablations

One mechanism changed at a time from `configs/hrd.json`. Run with `slurm/ablation.sbatch` /
`slurm/ablation_eq.sbatch` (`--skip-reference`: the mechanistic criteria only need DSSL vs its
own untrained control, not a second full CoST-reference training) and scored with
`scripts/ablation_report.py <arm> [<arm> ...]`.

## Batch A — the three originally diagnosed failures (2026-09-16, job 3183155, HRD seed 1 folds 0-1)

| Arm | Changed | Verdict |
|---|---|---|
| `hrd_a0_current.json` | nothing (control) | top-1 1.000 (saturated); RQ1 gain −0.0068; intensity below 0.5 and falling |
| `hrd_a1_trend.json` | `mask_mode=binomial`, `moco_k=256` (upstream CoST defaults) | **REJECTED.** top-1 0.992 — not meaningfully less saturated. Also has a real, unpredicted cost: Steps' own-branch acrophase R² collapses 0.765→0.0 even though nothing here touches the phase readout or loss — the shared TCN backbone is the only plausible path, since trend and seasonal are both just readouts of one `feature_extractor` output |
| `hrd_a2_amplitude.json` | `readout_norm=none` | **ADOPTED** as the new default (`configs/hrd.json`). Fixes RQ2 intensity cleanly (0.58→0.697, rising, above chance). Confirmed to be a pure readout-time change (training is identical to a0 — `seasonal_loss` always normalises regardless of this flag), yet on the *same trained weights* it also collapses Steps' own R² 0.765→0.0. RQ1 proxy got slightly worse than a0 (−0.0194 vs −0.0068) on this 2-fold slice. Kept anyway: the RQ2 fix is unambiguous and mechanism-proven; the RQ1/Steps trade-offs are separate open problems, not caused by training |
| `hrd_a3_phase.json` | `phase_mode=circular` (remove amplitude weighting) | **REJECTED.** Did not recover Steps/screen phase (still 0.0/0.0); worst RQ1 proxy of any arm (−0.0595, 0/5 families improved) |
| `hrd_a4_combined.json` | all three | Inherits a1's and a3's failures; not adopted |

Kept for the record, not deleted: they are the evidence for what does and doesn't work. See
memory `ablation-a0-a4-hrd-seed1.md` for the full numbers and the diagnostic that ruled out
a probe bug (predictions have real variance; the R²=0.0 values are genuine, not floored noise).

## Batch B — loss/gradient-weight balance (2026-09-16, on top of the adopted `readout_norm=none`)

Diagnosed cause: at HRD initialisation the weighted objective contributes trend 0.3%, amplitude
27.3%, phase 27.6%, **equivariance 44.9%**. Trend and seasonal go through `total_loss`'s single
`alpha=0.005`, which is what tames amplitude's huge raw scale (≈193) down to a comparable share.
`w_eq` is added *outside* `total_loss` with no such scaling (`total = total_loss(...) + w_eq *
L_eq`), so at `w_eq=1.0` it ends up the largest single term despite its raw value (≈0.235) being
smaller than phase's (≈34). Two single-scalar probes of the same hypothesis:

| Arm | `w_eq` | Rationale |
|---|---|---|
| `hrd_b0_readout_fix.json` | 1.0 (= `hrd_a2_amplitude.json`, reuse its results, no rerun) | current default, baseline for this batch |
| `hrd_b1_eq_peer.json` | 0.277 | equal to `w_trend` -- peer to the term it regularises, not allowed to outweigh it |
| `hrd_b2_eq_small.json` | 0.05 | clearly subordinate to amp+phase's combined ~55% share |

Monitored: RQ1 gain vs untrained, RQ2 timing and intensity, phase own/leak for Steps and screen,
MESOR own/leak. Everything here inherits `readout_norm=none` and its open Steps-phase issue;
this batch is not expected to change that on its own, since `w_eq` never touches the readout.

**Result (5 folds, HRD seed 1):** RQ1 gain w_eq 1.0 / 0.05 / 0 = −0.0343 / −0.0136 / **+0.0096**,
ordered that way in every fold; phase leakage into the trend branch rises as w_eq falls (Steps leak
b3 > b2 > a2 in 5/5 folds). **w_eq=0 adopted as the working default** (`configs/hrd.json`,
`configs/globem.json`, `DSSL`'s own default).

## Batch C — the trend branch, Failure A (2026-09-17, on top of w_eq=0, readout_norm=none)

Diagnosis (measured, no training):
- **The trend positive is a near-copy.** Two augmented views of the same full week: raw input alone
  retrieves the pair at top-1 **0.891** among all 3,803 HRD weeks (a 24 h slice: 0.737). Any encoder
  that preserves input identity solves it; training learns exactly that and saturates (full run:
  top-1 ≥0.98 by ~1,600 iterations). Stronger augmentation is not the cause: the CoST reference, at
  jitter/scale/shift 0.5, saturates too (0.985).
- **The trend term is not weak at initialisation.** An earlier probe put it at 0.3% of the loss and a
  5.2e-4 gradient — an artifact of the randomly initialised queue. Against real negatives it is the
  largest term (weighted 1.39 vs amplitude 0.14, phase 0.14; gradient 1.64). Its signal disappears
  during training, once the copy shortcut is learned.
- **Its own window is among its negatives.** Queue 4,096 > ~2,900 training weeks: ~1.4 keys per query
  come from the query's own week (and ~38 from its own participant).

| Arm | Change | Rationale |
|---|---|---|
| `b3_eq_zero` | (baseline, already run at 5 folds) | identical to the working default |
| `hrd_c1_trend_days.json` | `trend_views=disjoint_days` | query = days A of view 1, key = a disjoint half of the days of view 2, each read out as the mean trend over its visible bins (what the frozen trend block reports); a week's own earlier keys are masked out of its negatives. Views share no timestep, so copying cannot solve it: simple cross-day features retrieve at 0.002–0.016; untrained encoder with real negatives 0.688 → 0.062. One extra encoder pass: +23% per step |

`trend_views="same"` (default) is bit-identical to commit a238a37, on which `b3_eq_zero` ran.

**Pass only if all hold, c1 vs b3, 5 folds:** RQ1 gain stays > 0 or improves; trend top-1 no longer
≈1.0; Steps and screen phase leakage improve or do not worsen; intensity above 0.5 and rising; timing
preserved; MESOR own intact. Risk named in advance: across disjoint days the most identifying cue
measured was the daily-rhythm profile (0.016, vs 0.002–0.004 for level/variability), so the trend
branch could learn rhythm and leakage could *rise*.

**Result (5 folds, HRD seed 1, job 3277080): c1 FAILS the rule — not adopted.**

| Condition | b3_eq_zero | c1_trend_days | |
|---|---|---|---|
| trend top-1 not saturated | 1.000 | **0.006** (chance 0.00024) | met, but the task is now barely learned at all |
| **RQ1 gain > 0 or better** | **+0.0096** | **−0.0164** | **FAILED — worse in 5/5 paired folds, mean −0.026** |
| Steps leakage not worse | 0.664 | **0.622** | met: down in 5/5 folds (mean −0.041) |
| screen leakage not worse | 0.507 | **0.490** | met: down in 4/5 folds |
| intensity > 0.5, rising | 0.654→0.730 | 0.645→0.730 | met, unchanged |
| timing preserved | 0.693→0.972 | 0.686→0.974 | met, unchanged |
| MESOR own intact | 1.000 (5/5) | 1.000 (5/5) | met |

The mechanism worked and still lost: making the trend pair unmatchable by copying **did** drive rhythm
out of the trend branch (leakage down in 5/5 Steps folds), which confirms the shared-backbone story.
But top-1 fell from 1.000 to ~0.006, i.e. from trivially solved to essentially unsolved, and RQ1
recovery regressed in every fold. It also produced new own-branch collapses (Steps fold 4
0.705→0.000; screen folds 1 and 4 0.308→0.000 and 0.147→0.000), so branch separation counted over
folds did not improve (Steps own>leak 4/5 → 3/5).

Two points now bracket the difficulty knob — copy-easy (top-1 1.000, RQ1 +0.0096) and
disjoint-hard (top-1 0.006, RQ1 −0.0164) — and RQ1 is better at the easy end. Note c1 bundled three
changes (disjoint days, visible-mean readout, own-window keys masked), so it does not attribute the
regression to disjointness alone. `trend_views` stays `"same"` by default; Failure A remains open and
is documented as a limitation rather than fixed.

## RQ3 screen of every arm (2026-09-17, `scripts/ablation_rq3.py`, no GPU)

Each arm holds all 5 folds of seed 1 — one complete out-of-fold pass — so RQ3 is computed with the
protocol's own aggregation, one seed instead of three. Controls are identical across arms (they do
not depend on the DSSL config); at seed 1, raw = 0.695 and untrained = 0.657.

| Arm | DSSL AUROC | vs raw (primary) | vs untrained (primary) |
|---|---|---|---|
| `a2_amplitude` (w_eq=1.0) | 0.674 | −0.021 [−0.093, +0.053] | +0.017 [−0.054, +0.094] |
| `b2_eq_small` (w_eq=0.05) | 0.661 | −0.034 [−0.097, +0.033] | +0.004 [−0.067, +0.081] |
| **`b3_eq_zero` (w_eq=0)** | **0.726** | +0.032 [−0.016, +0.084] | **+0.069 [+0.007, +0.136] favours DSSL** |
| `c1_trend_days` | 0.710 | +0.015 [−0.037, +0.071] | +0.053 [−0.007, +0.120] |
| (narval_v1, 3 seeds) | 0.697 | +0.002 | +0.006 |

`b3_eq_zero` is the best configuration on RQ3 as well as on RQ1 and RQ2, and is the **first time
DSSL has beaten the untrained encoder on RQ3 with an interval excluding 0**. The conjunction is
still **NOT met**: raw (0.695) is not beaten. Caveats, both material: one seed, so the interval is a
participant bootstrap conditional on the fitted models and carries no seed-to-seed spread (untrained
alone moves 0.657 → 0.690 between seed 1 and the 3-seed mean); and the w_eq order for RQ3 is not
monotone (0.674 / 0.661 / 0.726), unlike RQ1. Treat the RQ3 gain as promising, not established.
