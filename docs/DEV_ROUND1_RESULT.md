# Development round 1 (HRD dev cohort, 89 participants, 2 seeds x 5 folds)

Arms: `a0_shared` 7.45M | `a1lo_decomposed` 4.03M | `a1_decomposed` 7.51M. All `dev_cohort=true`;
locked fold-0 test participants never seen. Selection uses the label-free criteria only.

## The capacity hypothesis is FALSIFIED

`docs/DECOMPOSITION_RESULT.md` attributed the v3 RQ1 regression to the 0.54x parameter cut, calling
that "the likely cause". Round 1 tested it directly and it is wrong:

| RQ1 vs untrained | a1lo (4.03M) | a1 (7.51M) |
|---|---|---|
| amplitude | −0.027 | **−0.027** |
| phase_hours | −0.056 | −0.039 |
| IS | −0.052 | −0.042 |
| MESOR | −0.033 | −0.026 |
| acrophase own R² | 0.464 | **0.416** (worse) |

Nearly doubling the parameters changed nothing, and acrophase own-recovery got slightly *worse*.
**Capacity was never the explanation.** The decomposition itself costs RQ1 recovery.

## The real trade-off, measured at matched capacity (a0 vs a1)

| | a0 shared | a1 decomposed | winner |
|---|---|---|---|
| RQ1 amplitude vs untrained | **+0.046 favours DSSL** | −0.027 favours control | shared |
| RQ1 phase vs untrained | **−0.003 inconclusive** | −0.039 favours control | shared |
| RQ1 MESOR vs untrained | **+0.032 favours DSSL** | −0.026 inconclusive | shared |
| amplitude leak | 0.828 | **0.442** | decomposed |
| acrophase leak | 0.652 | **0.007** | decomposed |
| amplitude own−leak | +0.141 | **+0.518** | decomposed |
| acrophase own−leak | **−0.137 (fails)** | **+0.409 (separated)** | decomposed |
| RQ2 timing | 0.775 | 0.779 | tie |
| RQ2 strength | **0.733** | 0.701 | shared |

**Disentanglement and recovery are in tension, and the tension is structural.** The RQ1 probe reads
the whole representation, so the same unrestricted access that creates leakage also gives the probe
more to work with. Restricting access necessarily removes information the probe was using. This is
not an implementation defect to engineer away; it is what the decomposition *does*.

## The finding that matters most for the next step

Training degrades acrophase own-recovery in **every** arm, and slightly more once the branch is
isolated:

| | untrained | trained | cost of training |
|---|---|---|---|
| a0 shared | 0.680 | 0.515 | −0.165 |
| a1lo decomposed | 0.658 | 0.464 | −0.194 |
| a1 decomposed | 0.621 | 0.416 | **−0.205** |

Isolating the seasonal branch makes the contrastive objective *more* effective at destroying the
quantity it is applied to — exactly what an invariance-seeking loss should do once nothing else
dilutes it. This strengthens the A3 hypothesis rather than weakening it: the objective, not the
wiring, is what stops training from adding value.

## Round 2, one change per arm

| Arm | Baseline | Single change | Params |
|---|---|---|---|
| `a2_daylocal` | a1 | day-local oscillatory receptive field | 1.007x |
| `a3_equivar` | a1 | seasonal equivariance objective | 1.008x |
| `a3s_shared_eq` | a0 | seasonal equivariance objective | 1.000x |

`a3s_shared_eq` matters because a0 currently wins RQ1 recovery: if the objective is the real
problem, fixing it on the *shared* encoder may beat fixing it on the decomposed one. Round 1 gives
no reason to assume the decomposed tower is the right host for the new objective.

Pre-committed reading: an arm is only interesting if it makes **trained beat untrained** on
acrophase own-recovery — that is the failure no architecture has yet fixed. Better separation with
unchanged or worse own-recovery is not progress.
