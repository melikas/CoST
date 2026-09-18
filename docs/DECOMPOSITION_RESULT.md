# Outcome: pre-encoder decomposition (development experiment) — REJECTED

Read against `docs/DECOMPOSITION_PRECOMMIT.md`. RQ3 was visible in the same `SUMMARY.md` file
read for RQ1/RQ2 and is disclosed as seen, but did not enter the eight criteria below; it is
reported last, unread until the verdict was fixed.

## Trained vs untrained leakage (the direct test of the hypothesis)

| Target | v2 shared: dssl leak | v2 shared: untrained leak | decomposed: dssl leak | decomposed: untrained leak |
|---|---:|---:|---:|---:|
| HRD amplitude (trend block) | 0.843 | 0.827 | **0.454** | **0.451** |
| HRD acrophase (trend block) | 0.669 | 0.642 | **0.016** | **0.024** |
| GLOBEM amplitude | 0.352 | 0.350 | 0.112 | 0.128 |
| GLOBEM acrophase | 0.000 | 0.001 | 0.000 | 0.000 |

**Leakage drops by almost the same amount for the trained and the untrained model**, on both
markers, both cohorts. This is exactly what the pre-flight predicted (0.810→0.392, 0.501→0.006,
no training) and it replicates at protocol grade. It confirms the mechanism is structural, not
learned — stated as the expected outcome in the pre-commitment.

## Fold-by-fold consistency (condition 7), computed from the 15 individual seed×fold files

| | HRD RQ2 strength | HRD RQ2 timing | HRD amplitude own−leak | **HRD acrophase own−leak** |
|---|---|---|---|---|
| Range over 15 cells | 0.663 – 0.739 | 0.826 – 0.878 | +0.119 – +0.148 | **−0.478 – +0.195** |
| Consistent? | yes | yes | yes | **no** |

RQ2 and amplitude separation are stable across every fold. **Acrophase separation is not**: 11 of
15 cells are positive and consistent (+0.11 to +0.20), but **4 cells — seed 2 and seed 3, folds 1
and 4 — are strongly negative**, down to −0.478. Fold assignment is identical across seeds
(verified earlier in this project), so folds 1 and 4 are specific participant groups on which
acrophase separation fails to replicate. The pooled "+0.489, separated" figure in `SUMMARY.md` is
real but is carried by the majority of folds, not all of them — this is precisely what condition 7
was written to catch. GLOBEM's acrophase own−leak is far more stable (14/15 cells positive,
+0.133 to +0.341; one mild negative at −0.047).

## Pass/fail table

| # | Criterion | v2 | Decomposed | Verdict |
|---|---|---|---|---|
| 1 | RQ2 strength ≈ preserved | 0.713, +0.054 vs untrained | 0.687, +0.029 vs untrained | **met** (−0.026, inside the 0.076 fold-to-fold range) |
| 2 | RQ2 timing no regression | 0.759 | 0.768 | **met** (improved) |
| 3 | RQ1 phase vs untrained materially improves | −0.023 (favours control) | **−0.054 (favours control)** | **FAILS — regressed** |
| 4 | RQ1 amplitude not materially regressed | +0.029 (favours DSSL) | **−0.039 (favours control)** | **FAILS — sign flip** |
| 5 | Own−leak improves, amplitude & acrophase | +0.134 / −0.083 | **+0.520 / +0.489** | met, pooled |
| 6 | Not from weakening untrained | — | untrained own-R² unchanged; leak drops equally | **met** — but reveals #5 is structural, not SSL-specific |
| 7 | Consistent across folds | — | RQ2 & amplitude: yes. **Acrophase: no (4/15 strongly negative)** | **FAILS for acrophase** |
| 8 | GLOBEM not materially regressed | RQ1-D acrophase −0.083 vs untrained | −0.045 vs untrained | **met** (stable to mildly improved) |

Three of eight fail (3, 4, and 7). The pre-registered rule requires all eight.

## Why 3 and 4 fail: capacity, not the hypothesis

RQ1's ridge-on-full-representation recovery got worse against **both** untrained and the fixed
anchor `raw` (phase vs raw: −0.142→−0.169; amplitude vs raw: −0.047→−0.137). Both move the same
direction against a config-independent control, so this is not the untrained control shifting —
DSSL's absolute recovery quality declined. The likely cause, named in advance: **the 0.54× parameter
budget**. Each branch lost half its backbone width, and the full-representation ridge probe is
exactly where lost capacity would show first, since it draws on the whole encoder rather than one
branch's separation.

## Verdict

**REJECT**, applying the pre-registered all-eight rule literally.

**The hypothesis is not falsified — it is confirmed, more precisely than the pre-flight could show.**
Restricting input access cuts leakage by the same amount whether the encoder is trained or not, on
both amplitude and acrophase, on both cohorts, at protocol grade. What fails is this specific,
capacity-constrained implementation: it trades RQ1 full-representation quality for branch
separation, and the separation gain itself is fold-fragile on HRD acrophase specifically.

Do not conclude the pre-encoder approach is wrong. Two distinct, nameable problems block it, not
one diffuse failure: (a) the parameter cut, and (b) fold-4/fold-1 acrophase instability. Per the
user's standing instruction not to immediately add another mechanism, no follow-up architecture is
proposed here. The next decision is the user's: widen the branches to close the capacity gap (a
second controlled experiment, matching the current parameter budget), investigate what is specific
to folds 1 and 4, or hold at `readout_norm="none"` with the shared encoder as the frozen candidate.
