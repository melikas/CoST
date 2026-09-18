# DSSL-v2: architecture design from the measured failures

Development-phase design document. The final held-out evaluation stays locked; everything here is
selected on label-free criteria using development participants only (§5).

## 1. Architectural assumptions now contradicted by evidence

**A1. "A shared high-capacity encoder can serve branch-specific semantics through readout heads."**
Refuted. At *random initialisation* the shared model's trend block predicts 24-h amplitude at
R² 0.827 and acrophase at 0.642. Leakage is geometric — a receptive field of 1021 over a 672-step
window means every timestep encodes the whole window, and the time-averaged trend readout inherits
it. No objective can remove a property present before training.

**A2. "Window instance discrimination is a meaningful trend objective."** Refuted. Raw input alone
retrieves the positive at top-1 0.891 among 3,803 weeks; MoCo top-1 reaches 1.000 by ~iteration
1600 in 15/15 folds. The task is solved by preserving window identity — which *includes* rhythm,
so the objective actively drives entanglement while its gradient vanishes.

**A3. "Amplitude and phase can share one tensor, separated at readout by |·| and atan2."** Refuted.
They are two views of one complex quantity with zero parameters between them, and the readout
choice trades them against each other: `timestep` gave RQ1 phase +0.033 vs untrained but RQ2
strength 0.399 (below chance); `none` gave RQ2 strength 0.713 but RQ1 phase −0.023. Three readout
variants were tested; none dominates, because the conflict is structural.

**A4. "Concatenating trend | amplitude | phase is a sound downstream geometry."** Refuted. `dscore`
is a mean over all 1,760 standardised dimensions — amplitude 45%, phase 45%, trend 9%. Changing
only the phase block dropped RQ2 strength 0.713 → 0.557 with the amplitude block **bit-identical**.
The metric is dominated by dimension count, not information content.

**A5. "SSL training improves the representation."** Refuted on every measured branch target:

| own-branch R² | v2 trained | v2 untrained | v3 trained | v3 untrained |
|---|---|---|---|---|
| amplitude | 0.977 | 0.979 | 0.973 | 0.980 |
| acrophase | **0.587** | **0.769** | **0.506** | **0.768** |

Training never helps and destroys phase, in both architectures. This is the deepest failure and it
is an **objective** failure, not a wiring failure: contrastive instance discrimination seeks
*invariance*, while RQ1 and RQ2 measure *equivariance* — whether the representation moves
correctly when amplitude or phase moves. The objective is optimising against the criteria.

**A6. "Better rhythm fidelity implies better downstream utility."** Not supported. RQ2 strength went
from below chance (0.438) to 0.713 while RQ3 stayed flat (0.697 → 0.704, inconclusive vs raw).

## 2. What the decomposition experiment actually established

Not a failure — a confirmed mechanism plus a diagnosis of *what did not* cause it:

- Input restriction removes leakage: acrophase leak 0.669 → 0.016 (trained), 0.642 → 0.024
  (untrained); amplitude 0.843 → 0.454 / 0.827 → 0.451.
- **The decomposition discards no recoverable information**: the untrained decomposed model recovers
  amplitude at 0.980 and acrophase at 0.768, statistically identical to the untrained shared model
  (0.979 / 0.769). The RQ1 regression therefore came from training and the 0.54× capacity cut, not
  from the architecture removing signal.

So: keep the input decomposition, restore capacity, and change the objective.

## 3. Candidate architectures

| Architecture | Core idea | Why it could fix the measured failures | Main risk | RQ1 | RQ2 | RQ3 potential | Complexity |
|---|---|---|---|---|---|---|---|
| **B1. Capacity-matched two-tower** | v3 decomposition, sized to 7.45M, late learned fusion | Fixes A1 + A5-capacity directly; keeps the proven leakage fix | Leaves A2, A3, A5-objective untouched — training will still degrade phase | recovery restored, separation kept | preserved | none | low |
| **B2. Complex-valued seasonal pathway + equivariance objectives** | Seasonal latent stays complex; train the representation to *transform correctly* under known shifts/scalings rather than be invariant | Fixes A3 (timing = rotation, intensity = radial scaling are orthogonal actions on one complex number) and A5-objective (equivariance is what RQ1/RQ2 measure) | RQ2 becomes partly a manipulation check if trained on the same transform family — must be declared and mitigated | phase recovery should stop degrading | strong by construction | unknown | medium |
| **B3. Hierarchical day → week → participant** | Encode each day, then aggregate across days, then across weeks | IS, IV and RA are *by definition* cross-day statistics; the flat 672-step encoder computes them implicitly. Also natural for GLOBEM's 28 days | More stages, more places to lose signal; day-level encoders see less context | IS/IV/RA should improve | day-level phase helps timing | plausible: exposes within-person variability that participant-mean aggregation currently destroys | medium |
| **B4. Predictive (masked-day parameter forecasting)** | Hide a day, predict its rhythm parameters from the rest | Structured pretext with no identity shortcut, unlike A2 | Pixel-level MAE was already tested on this data and lost to contrastive; parameter-level may inherit that | uncertain | uncertain | uncertain | medium |

**Eliminations.** B4 is deprioritised: the closest tested relative (MAE across full sweeps) lost to
contrastive on both cohorts, and nothing in the new evidence changes that prior. B1 alone is
insufficient — it fixes capacity and leakage but leaves the objective that *destroys* phase
(0.769 → 0.587), so it would reproduce A5. B2 and B3 attack different failures and are compatible:
B2 fixes the amplitude/phase conflict and the invariance/equivariance mismatch; B3 fixes the
cross-day markers and the downstream geometry. **DSSL-v2 = B1 ∪ B2 ∪ B3**, built on the confirmed
input decomposition.

## 4. DSSL-v2

```
  x(t)   C channels, T steps, B bins/day, NaN-preserving throughout
    │
    ├──────────────── MA24h(x) ──────────────┐         zero-phase, exact nulls at
    │                                        │         1/day and every harmonic
    │                                        ▼
    │                            ┌───────────────────────┐
    │                            │  TREND ENCODER        │  input decimated to 4/day
    │                            │  dilated TCN, RF ~7d  │  (band-limited: no loss)
    │                            └───────────┬───────────┘
    │                                        │  per-day trend latent  t_d ∈ R^dt
    │
    └──── x − MA24h(x) ──────────┐
                                 ▼
                     ┌───────────────────────┐
                     │  OSCILLATORY ENCODER  │  full resolution
                     │  dilated TCN          │  RF ≈ 1.5 days  (NOT whole window:
                     │  day-local            │   blocks week-identity shortcuts)
                     └───────────┬───────────┘
                                 │
                                 ▼   per day d, per harmonic h ∈ {1..H}
                     ┌───────────────────────┐
                     │ COMPLEX HARMONIC HEAD │  z_{d,h} ∈ C^k   (never split into
                     │ learned complex proj  │   |·| and atan2 blocks)
                     └───────────┬───────────┘
                                 │
        ┌────────────────────────┴────────────────────────┐
        ▼                                                 ▼
┌─────────────────┐                            ┌──────────────────────┐
│ WEEK AGGREGATOR │  over days:                │  (same for trend)    │
│  • |z| mean/SD  │→ amplitude level & IV-like │  • level, slope      │
│  • complex mean │→ vector strength = IS-like │  • day-to-day var    │
│  • circular var │→ phase stability           │                      │
└────────┬────────┘                            └──────────┬───────────┘
         └───────────────────┬───────────────────────────┘
                             ▼
                  ┌─────────────────────┐
                  │   FUSION HEAD       │  small MLP → window representation
                  │  (learned, not      │  branch-calibrated, equal weight PER
                  │   concatenation)    │  BRANCH not per dimension
                  └─────────────────────┘
```

**Components.**
- *Decomposition*: as validated — centred 24-h MA, zero phase, circular, NaN preserved, exact
  reconstruction. Unchanged from the tested implementation.
- *Trend encoder*: operates on the decimated trend (legitimate: it is band-limited below 1/day), so
  a week-long receptive field costs little. Output per day.
- *Oscillatory encoder*: **day-local receptive field (~1.5 days)**, deliberately *not* whole-window.
  This is what removes A2's identity shortcut at the architectural level rather than by changing the
  loss: an encoder that cannot see the week cannot encode week identity.
- *Harmonics*: H = min(4, B//2). HRD gets 24/12/8/6 h; **GLOBEM resolves only 24 h** (6-h bins put
  Nyquist at a 12-h period), so H=1 there. The design degrades correctly rather than inventing
  unresolvable bands.
- *Complex representation*: amplitude and phase are modulus and argument of one stored complex
  vector. Timing perturbation = rotation `e^{-i2πfΔ}`; intensity perturbation = radial scaling.
  Orthogonal group actions on one object instead of two competing real blocks.
- *Week aggregator*: complex mean gives vector strength (an IS analogue), modulus SD gives an IV
  analogue, both computed structurally rather than implicitly.
- *Fusion*: a learned head, with branch contributions equalised **per branch**, fixing A4's
  dimension-count domination.

**Objectives (one per branch, matched to what each should represent).**
- *Trend*: masked-day level prediction — hold out a day, predict its trend level from the others.
  Non-trivial, structured, no identity shortcut. Replaces window MoCo entirely.
- *Oscillatory*: **equivariance**. Apply a known transformation and require the complex latent to
  transform predictably: `z(shift_Δ x) ≈ e^{-i2πfΔ} z(x)`, `z(scale_s x) ≈ s·z(x)`. Loss is complex
  MSE against the predicted transform.

**RQ2 independence — stated before running.** Training equivariance on the same family RQ2 probes
would make RQ2 a manipulation check rather than evidence. Mitigation: train equivariance on
transformations *disjoint* from the RQ2 probes — per-day random shifts and scaling of the **12-h**
harmonic — while RQ2 evaluates whole-window uniform shifts and **24-h** amplitude scaling. The
mechanism generalises; the specific transformation does not overlap. If we ever train on the RQ2
family itself, RQ2 must be re-designated a manipulation check and RQ1/RQ3 carry the evidence.

**Budget**: total ≈ 7.45M ± 10%, matching the current model. Trend encoder is cheap (decimated
input), so the oscillatory tower takes the majority. A capacity-control arm is mandatory (§6).

**Missingness**: NaN through the filter as validated; day-level aggregation weights each day by its
observed fraction; a day below a coverage threshold is dropped from the week aggregate rather than
imputed.

**Downstream**: participant representation = fusion output aggregated over that person's windows,
as now (so it stays comparable to every existing baseline).

**RQ2 deviation score**: defined now, before evaluation — per-branch standardised distances combined
with **equal weight per branch**, analytically fixed, never tuned:
`D = (D_trend + D_amp + D_phase)/3`, each D standardised on the personal baseline. This is a
pre-specified change for the new model only; the existing results keep the existing metric.

## 5. Why it should outperform — mechanism, not hope

**RQ1.** Phase degradation is caused by an invariance objective acting on the quantity being
measured (0.769 → 0.587). Equivariance training removes that pressure: the loss now *requires* the
latent to track phase. Recovery starts from the untrained level (0.768 — the decomposition preserves
it fully) instead of being trained away from it. IS/IV/RA improve because the week aggregator
computes cross-day statistics structurally rather than hoping a flat encoder induces them.

**RQ2.** Timing and intensity stop competing: in complex coordinates they are rotation and radial
scaling of the same object. The per-branch distance stops letting a 45%-of-dimensions phase block
dilute an amplitude-only perturbation — the mechanism measured directly when a bit-identical
amplitude block still lost 0.713 → 0.557.

**RQ3.** *No mechanism is claimed.* Prior evidence constrains the expectation: aggregation
alternatives, probe capacity and multi-week trajectory were each tested and negative; the endpoint
is largely trait-like (baseline CES-D alone AUROC 0.876, 80.4% of labels match baseline status); the
whole ladder spans 0.049 AUROC. The one untested path is that participant-mean aggregation over
windows discards within-person variability that a hierarchical representation would expose. That is
a hypothesis with a low prior, not a justification. **The architecture is justified by RQ1, RQ1-D
and RQ2; RQ3 stays an open question evaluated once, at the end.**

## 6. Development protocol and first experiment

**Clean separation.** Architecture selection uses **only label-free criteria** — RQ1 recovery,
own-vs-leakage, RQ2 timing and strength. No endpoint label is read during development, so the final
RQ3 evaluation stays label-clean by construction. Development runs on the **training participants of
protocol fold 0** with an inner participant split; fold 0's test participants are never seen. The
final evaluation remains the existing locked 3 seeds × 5 folds on both cohorts, unchanged, so every
prior run stays comparable.

**First experiment — a four-arm ladder at matched capacity (~7.45M), one structural change per step:**

| Arm | Change from previous | Isolates |
|---|---|---|
| A0 | current shared v2 | baseline |
| A1 | + input decomposition, capacity-matched | decomposition without the capacity confound |
| A2 | + day-local RF and complex seasonal latent | the amplitude/phase conflict (A3) |
| A3 | + equivariance and masked-day objectives | the invariance/equivariance mismatch (A5) |

Plus a **capacity control**: A0 re-run at the decomposed model's parameter count, so no result can be
attributed to capacity. Pass criteria per arm are the eight already used, evaluated on development
folds. Any arm that regresses RQ2 strength below chance or fails to recover phase is dropped and the
reason recorded.
