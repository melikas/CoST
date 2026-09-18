# Outcome of the full protocol validation (narval_v2)

Read against `docs/VALIDATION_PRECOMMIT.md`, which was committed (8bc9b6e) before the run. Candidate:
`readout_norm="none"`, `w_eq=0`, `trend_views="same"`. 3 seeds x 5 folds, both cohorts, full ladder.
Compared against `narval_v1` (the same protocol, pre-repair model).

## Every pre-specified criterion

| Criterion | narval_v1 | narval_v2 | Verdict |
|---|---|---|---|
| **HRD RQ1(a)** beats population mean, every family | met | met | **met** |
| **HRD RQ1(b)** beats untrained, every family | not met | amplitude +0.029, IV +0.035, MESOR +0.040 favour DSSL; **phase_hours −0.023 and IS −0.019 favour control**; RA inconclusive | **not met** |
| **HRD RQ1-D(i)** own > leakage, all three targets | not met | MESOR +0.544, amplitude +0.134 separated; **acrophase −0.083 inconclusive** | **not met** |
| **HRD RQ1-D(ii)** separates better than untrained | not met | MESOR +0.077 favours DSSL; **amplitude −0.018 and acrophase −0.209 favour control** | **not met** |
| **HRD RQ2 timing** above chance | met | 0.759 | **met** |
| **HRD RQ2 strength** above chance | **not met (0.438, below chance)** | **0.713** | **met** |
| **HRD RQ2 timing** beats untrained | met | +0.049 [+0.030, +0.069] | **met** |
| **HRD RQ2 strength** beats untrained | **not met** | **+0.054 [+0.042, +0.066]** | **met** |
| **HRD RQ3** above raw | not met | +0.009 [−0.035, +0.056] | **not met** |
| **HRD RQ3** above untrained | not met | +0.028 [−0.009, +0.067] | **not met** |
| **GLOBEM RQ1(a)** beats population mean | met | **amplitude +0.030 and IV −0.009 inconclusive** | **not met (regression)** |
| **GLOBEM RQ1(b)** beats untrained | not met | phase, IS, MESOR favour control | **not met** |
| **GLOBEM RQ1-D(i)** own > leakage | met | all three separated | **met** |
| **GLOBEM RQ1-D(ii)** better than untrained | not met | all three favour control | **not met** |
| **GLOBEM RQ2** | not applicable | not applicable | **not applicable** |
| **GLOBEM RQ3** above raw / untrained | not met | +0.020 / −0.026, both inconclusive | **not met** |

## Reading, per the pre-registered rules

- **RQ1: partially supported.** Population mean met on HRD, not on GLOBEM; untrained not met on either.
- **RQ1-D: not supported on HRD** (both clauses); on GLOBEM clause (i) holds, (ii) does not.
- **RQ2: fully supported on HRD** — all four conditions, the first time in this project. Not
  applicable on GLOBEM.
- **RQ3: not supported on either cohort.** The conjunction fails; DSSL beating `cost_reference_adapter`
  (+0.080) and `pca` (+0.081) on HRD is context, not a substitute (rule 4).

## What the repair did, including its costs

**RQ2 strength is the result of the programme, and it was predicted before it was measured.** The
mechanism was established on an *untrained* encoder (per-timestep L2 normalisation destroys amplitude),
the fix was adopted for that reason, and the pre-commitment stated the expectation in advance.
Concordance went 0.438 (below chance) -> 0.713, and DSSL now beats untrained on both RQ2 arms.

**Regressions caused by the same changes, reported as required:**
- HRD phase recovery got worse: vs raw −0.052 -> **−0.142**; vs untrained +0.031 (favoured DSSL) ->
  **−0.023 (favours control)**. This is the protocol-grade appearance of the phase cost already seen
  mechanistically, where `readout_norm="none"` makes `atan2` ill-conditioned for near-zero-amplitude bins.
- HRD acrophase separation vs untrained worsened: −0.156 -> **−0.209 (favours control)**.
- GLOBEM RQ1(a) went from **met to not met**.
- HRD RQ2 timing fell 0.789 -> 0.759 and lost its edge over the CoST reference (+0.030 favouring DSSL ->
  −0.016 inconclusive). On RQ2 strength the reference now beats DSSL (−0.045 favours control).

**Improvements besides RQ2:** HRD amplitude disentanglement own R² 0.845 -> **0.977** (own − leakage
+0.051 -> +0.134); HRD MESOR separation now beats untrained (+0.077); HRD RQ1 amplitude, IV and MESOR
now favour DSSL over untrained.

## Control fairness (rule 6)

`raw` is a fixed anchor at 0.695 in both runs. Against it, DSSL moved 0.697 -> **0.704 (+0.007)**.
`untrained` fell 0.690 -> 0.676 because it shares the DSSL readout, so part of the widened
DSSL-vs-untrained gap (+0.006 -> +0.028) is the control degrading, not the representation improving.
Both comparisons remain inconclusive.

## The ablation screen did not replicate (rule 7)

The seed-1 screen reported DSSL 0.726 on HRD RQ3. At 3 seeds it is **0.704** (seed SD 0.021), and the
comparison against raw stays inconclusive. The adversarial audit predicted this: the screen's advantage
was concentrated in one fold and vanished under fold resampling. No conclusion rests on it.

## Standing limitations

Trend saturation is unfixed. The HRD endpoint is largely trait-like (baseline CES-D alone AUROC 0.876;
80.4% of labels match baseline status), and at n=113 with ~22 participants per fold, a negative RQ3 is a
plausible true result. RQ3 is not redefined; incremental-over-baseline analysis remains future work.
