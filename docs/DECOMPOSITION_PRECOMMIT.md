# Pre-commitment: pre-encoder decomposition (development experiment)

Written before the run. Nothing about the RQs, endpoint, metrics, thresholds, baselines, splits or
success criteria is redefined. `dscore` and every protocol metric are untouched.

## What changes, and only this

```
x(t) ──┬── MA24h(x)      ──► trend encoder    ──► trend readout
       └── x − MA24h(x)  ──► seasonal encoder ──► amplitude + phase
```

- The 24-h filter is a **centred moving average**, `[0.5, 1, …, 1, 0.5]/N` over `N+1` taps
  (HRD N=96, GLOBEM N=4), circularly padded. Odd and symmetric, so it is **zero phase** — a
  phase-shifted trend would corrupt its complement's phase, which is the quantity this exists to
  protect. Verified: transfer function real to 6e-08, gain 1.000 at DC, **< 1e-6 at 24 h and every
  faster harmonic**, reconstruction exact to 1e-5, missing bins preserved as NaN in both views.
- Two independent encoders, **no shared parameters**, each half width (160), each building only its
  own head. Readout geometry is unchanged: trend 160, amplitude 800, phase 800, blocks identical.
- Unchanged: Fourier machinery, `readout_norm="none"`, `w_eq=0`, `trend_views="same"`, augmentations,
  objective, probes, folds, seeds, and the RQ2 distance.
- Not included, deliberately: no orthogonality, adversarial or branch-weighting losses, no residual
  branch, no new harmonic bank, no readout variant.

**Parameter count: 4,028,496 vs 7,452,048 (0.54×).** The backbone width and depth per branch are
identical to the baseline's; what halves is the trend head, because each branch's feature dimension
is 160 rather than 320. Matching 7.45M would require widening the backbone from 64 to ~200, which
would add a second variable. The reduction biases **against** the new architecture and is recorded
so it cannot be claimed as an advantage later.

## Pre-flight evidence (untrained, no training, full HRD cache)

Same probe applied to both architectures at initialisation. Absolute values are not comparable to
the protocol's audit (single participant split, channels pooled into one regression); only the
within-probe contrast is.

| target | shared own/leak | decomposed own/leak |
|---|---|---|
| MESOR | 1.000 / 0.000 | 1.000 / 0.000 |
| amplitude | 0.965 / **0.810** | 0.976 / **0.392** |
| acrophase | 0.471 / **0.501** | 0.386 / **0.006** |

Acrophase separation flips from −0.029 to +0.380 with no training, which is what the input-access
hypothesis predicts. Note the cost already visible here: acrophase own-recovery falls 0.471 → 0.386.

## Pass/fail, frozen. Adopt only if ALL hold, against narval_v2 on the same folds and seeds

1. RQ2 strength supported and approximately preserving v2 (0.713, beats untrained by +0.054).
2. RQ2 timing no material regression from v2 (0.759).
3. RQ1 phase vs untrained materially improves over v2 (−0.023, favouring control).
4. RQ1 amplitude recovery not materially regressed (v2: +0.029 vs untrained).
5. Own-vs-leakage improves for **both** amplitude (v2: +0.134) and acrophase (v2: −0.083).
6. The improvement is not produced only by weakening the matched untrained control — the untrained
   decomposed model is reported alongside, and leakage reduction there is expected, not hidden.
7. Consistent across folds, not driven by one.
8. GLOBEM does not materially regress against v2.

**RQ3 is not a development criterion and will not be read until the architecture is frozen.**

## Stated in advance

- The untrained control is expected to gain separation too, because the mechanism is structural.
  If criterion 5 passes while RQ1-D(ii) ("separates better than untrained") still fails, the honest
  conclusion is that branch semantics are enforced by **structure, not by learning**.
- Acrophase own-recovery may fall (pre-flight: 0.471 → 0.386). If separation improves only because
  own and leakage both collapse, that is not a success.
- If a criterion fails, the first question is whether the 0.54× parameter count explains it. A
  capacity-driven regression does **not** falsify the input-access hypothesis; criterion 5 is the
  direct test of it, and the pre-flight already supports it at initialisation.
- No expectation is claimed for RQ3.
