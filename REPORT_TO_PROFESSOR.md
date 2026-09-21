# DSSL — Disentangled Self-Supervised Learning for Circadian Rhythm Representation

**A complete technical and scientific report.**
Prepared 18 September 2026. Written to be read without any prior knowledge of this project's
history. Every internal term is defined at first use, and every number is traced to the file it
was read from.

---

## How to read this document

The report is organised so that you can stop after any section and have a complete, correct
picture at that level of detail.

| Section | Question it answers |
|---|---|
| **A** | What were we trying to find out, and how did we decide what counts as an answer? |
| **B** | What data did we use, and what exactly did we do to it before the model saw it? |
| **C** | What is the final model, layer by layer? |
| **D** | How was it trained, and how were checkpoints chosen? |
| **E** | RQ1 — does the model preserve individual rhythms? (Result: **partially**) |
| **F** | RQ2 — can it detect within-person rhythm changes? (Result: **yes, on HRD**) |
| **G** | RQ3 — does it predict depression better than the controls? (Result: **no**) |
| **H** | How did we arrive at this model, and what did we compare it against? |
| **I** | What did we try that did not work? |
| **J** | What are the limitations of everything above? |
| **K** | Exact identifiers: config files, commits, checkpoints. |
| **L** | Exact commands to reproduce every number. |
| **M** | What is still outstanding. |

**The headline, stated once, plainly.** We built a self-supervised encoder that separates a
person's wearable signal into a *level* component and a *rhythm* component. On our primary
dataset it succeeded at the task it was designed for — detecting when a person's own 24-hour
rhythm has shifted or weakened relative to their own recent history (RQ2). It did **not** improve
depression prediction over simple baselines (RQ3), and it only partially preserved individual
rhythm markers (RQ1). We report this as a partially positive, partially negative result. We did
not redefine any research question after seeing results.

---

## A. Research objective and the three research questions

### A.1 The scientific motivation

Human physiology runs on a roughly 24-hour cycle driven by an internal pacemaker. In depression
research, two properties of that cycle are repeatedly reported as clinically meaningful:

- **Amplitude** — how *strongly* the rhythm expresses itself. A "blunted" or compressed rhythm
  (the person's day and night look more alike than they should) is associated with depression.
- **Acrophase** — *what time of day* the rhythm peaks. A delayed phase (peaking later than
  typical) is likewise associated with depression.

A third quantity matters too:

- **MESOR** — the person's average level around which the rhythm oscillates (for example, their
  mean heart rate or mean activity over the window). This is a *level*, not a rhythm.

Consumer wearables (heart rate, step counts, sleep state, phone screen activity) record these
quantities continuously and cheaply. The question this project asks is whether a *self-supervised*
model — one trained without any depression labels — can learn a representation of a person's
wearable data in which these three quantities are cleanly separated and individually preserved,
and whether such a representation is clinically useful.

**Why self-supervised.** Labelled mental-health data is scarce (we have 113 labelled participants).
Unlabelled wearable data is comparatively plentiful. If self-supervised pretraining can produce a
rhythm-aware representation from unlabelled windows, the scarce labels are spent only on a small
downstream classifier rather than on training a whole encoder.

**Why "disentangled".** A representation is *disentangled* here if the part of it that is supposed
to encode the rhythm does not also silently encode the level, and vice versa. This matters because
a clinician asking "is this person's rhythm blunted?" needs an answer that is not contaminated by
"this person is simply less active overall."

### A.2 The base model we started from and why we changed it

We started from **CoST** (Salesforce Research, BSD-3 licensed), a general-purpose time-series
encoder that separates a signal into a *trend* component and a *seasonal* (periodic) component.
CoST was designed for industrial forecasting — electricity demand, traffic counts — where you do
not know in advance which periodicities matter, so it searches the whole frequency spectrum.

That genericness is a liability for us for three reasons:

1. **We already know the answer to the search.** Human physiology is built around a 24-hour
   pacemaker. Letting a model rediscover that from 113 people wastes capacity on frequency bins
   that are sensor noise.
2. **CoST's trend branch is built to discard the baseline level.** For us the level (MESOR) is
   itself a clinical marker, so discarding it is backwards.
3. **CoST treats phase as an ordinary real number.** Phase is circular — 23:59 and 00:01 are
   adjacent, not maximally distant — and treating it linearly is simply wrong for time-of-day.

Our model, **DSSL** (Disentangled Self-Supervised Learning), keeps CoST's overall two-branch
architecture and its contrastive training scheme, but injects the biological prior: dedicated
frequency bands at the daily harmonics, a circular phase treatment, and a readout that preserves
level and amplitude. Section C specifies it completely; Section H.1 lists the four changes and the
evidence for each.

**Crucially, the original CoST is kept in the codebase and run as a paired control in every single
experiment** (`method="cost_reference"` in `cost.py`). So every claim of the form "our change
helped" is something we measured by switching it off, not something we asserted.

### A.3 The three research questions, as pre-registered

These were fixed in `docs/SCIENTIFIC_PROTOCOL.md` and `docs/audit/RQ_SPECIFICATION.md` **before**
the reported runs, and were not modified afterwards.

> **RQ1 — Rhythm preservation.** Do the frozen representations linearly encode each window's
> MESOR, 24-hour amplitude and 24-hour acrophase, for held-out participants, better than control
> representations do?
>
> **RQ1-D — Branch disentanglement** (an amendment recorded 14 Sep 2026, before any reported run).
> Is each marker predicted better from *its own* branch than from *the other* branch?
>
> **RQ2 — Personalised rhythmic phenotyping.** When a person's current week is artificially
> perturbed, does the representation's distance from that person's *own* four preceding weeks grow
> in step with the size of the perturbation — more reliably than an untrained encoder achieves?
> Two arms: **timing** (the rhythm is shifted in time) and **strength** (the rhythm's amplitude is
> scaled up or down).
>
> **RQ3 — Downstream clinical utility.** Does a frozen linear probe on DSSL features separate
> endpoint-depressed from endpoint-non-depressed participants better than the same probe on
> (a) raw windows and (b) an untrained encoder of identical architecture?

### A.4 The decision rules, fixed in advance

These are what make the results below binding rather than interpretable after the fact. They are
quoted from `docs/VALIDATION_PRECOMMIT.md`, which was committed (`8bc9b6e`) **before** the run was
launched.

1. **RQ1 is fully supported only if** DSSL beats *both* the population mean *and* the untrained
   encoder in every eligible marker family. Beating the population mean alone = **partially
   supported**, and must be reported as such.
2. **RQ1-D requires both** (i) own-branch prediction above cross-branch leakage for MESOR,
   amplitude *and* acrophase, and (ii) better separation than the untrained encoder on all three.
3. **RQ2 requires timing *and* strength.** Timing passing while strength fails = **partially
   supported**. Timing cannot compensate for strength.
4. **RQ3 requires DSSL above raw *and* above untrained.** Anything else = **not supported**,
   regardless of how DSSL compares to CoST, cosinor, handcrafted features, PCA or random
   projection — those are context only, never substitutes.

**Expected outcome, recorded in advance.** The pre-commitment stated that RQ3 was expected **not**
to be supported, that RQ2 strength was expected to improve, and that RQ1 was genuinely uncertain.
Recording this in advance is why the negative RQ3 below cannot be presented as a surprise, nor the
positive RQ2 as a mere confirmation of an assumption.

---

## B. Datasets and preprocessing

Two cohorts were used. The second exists to test whether anything found on the first generalises.

### B.1 HRD — the primary cohort

| Property | Value |
|---|---|
| Participants with labels | 113 |
| Windows after preprocessing | 3,803 |
| Sampling resolution | 15-minute bins (96 bins per day) |
| Window length | 7 days = **672 timesteps** |
| Window stride | 7 days (non-overlapping weeks) |
| Channels | **4** |
| Depression label | `ces_d_endpoint_score >= 16` (the standard CES-D cut-off) |

**The four channels, and why each was chosen:**

| Channel | Source | Nature | Cleaning treatment |
|---|---|---|---|
| `HR` | Fitbit heart rate | Wear-dependent continuous | Gaps up to 30 min linearly interpolated; longer gaps left missing |
| `Steps` | Fitbit step count per minute | Wear-dependent count | Same |
| `is_asleep` | Derived binary from Fitbit `sleep_status` | Wear-dependent binary | `asleep/light/deep/rem` → 1; `awake/wake/restless` → 0; unknown resolved by wear |
| `screen` | Phone screen unlock events | Event stream | A missing value means "no event", which is a genuine 0 — **not** interpolated |

Excluded on purpose: Fitbit active-minute intensity levels, sedentary minutes, floors climbed, and
call logs. The distinction between *wear-dependent* signals (missing = we don't know) and *event
streams* (missing = nothing happened) is the reason the two groups get different cleaning; treating
an absent screen event as an unknown to be interpolated would fabricate phone usage.

**Preprocessing pipeline, in order:**

1. **Streaming read with participant-boundary enforcement.** The raw CSV is read in chunks; if a
   participant's rows are non-contiguous the loader raises rather than silently interleaving
   people. (Tested: `tests/test_repairs.py`, `test_streaming_preserves_participant_boundaries…`.)
2. **Minute grid.** Each participant is placed on a complete minute-resolution grid; absent
   minutes become explicit rows, and an `observed/<channel>` mask records which values were real.
3. **Short-gap interpolation.** Wear-dependent channels only; gaps ≤ 30 minutes are linearly
   interpolated, gaps > 30 minutes are left missing and eventually cause window rejection.
4. **Binning to 15 minutes.**
5. **Windowing.** Contiguous 7-day windows. A window is dropped if fewer than 70% of its timesteps
   carried any observation (`marker_coverage: 0.7` in the config).
6. **Normalisation: `within_person`.** Each participant's channels are z-scored using that
   participant's own statistics. This is deliberate and consequential — see the limitation in J.4.

### B.2 GLOBEM — the generalisation cohort

| Property | Value |
|---|---|
| Participants | 155 (142 labelled), 2018 cohort only; the full release has 702 participant-years (669 labelled; ~497 unique people) across 2018–2021 |
| Sampling resolution | 4 day-segments per day (6-hour bins) |
| Window length | 28 days = **112 timesteps** |
| Window stride | 7 days |
| Channels | **14** (RAPIDS-derived features: steps, sleep, screen, call-free subset, location) |
| Cohort policy | Earliest study year (2018) only; other years, including unlabelled, excluded, because the release does not link a returning student's identifiers across years, so a pooled split cannot be guaranteed person-disjoint |

GLOBEM's features are sparse (sleep is present only ~33% of the time). Because the encoder's FFT
layer cannot accept `NaN`, and because the encoder masks a timestep only when *every* channel is
missing, per-participant linear interpolation plus nearest-value end-extension is applied — but
**after** computing the z-score statistics from observed values only, so no imputed value can skew
the scale. That ordering is what keeps the imputation leakage-free.

**A structural caveat that matters for interpreting every GLOBEM result:** at 6-hour bins there are
only 4 samples per day. The 24-hour rhythm sits at the very edge of what is resolvable, and
sub-daily harmonics are not resolvable at all (only 2 bands exist, versus 4 on HRD). GLOBEM is
therefore a weak test of a circadian model, and its RQ2 is marked **not applicable** by protocol
rather than being replaced with a surrogate.

### B.3 Splitting, seeds, and leakage control

| Element | Value | Why |
|---|---|---|
| Cross-validation | **5 participant-disjoint folds** | No participant ever appears in both train and test — the leakage risk that matters here is person identity, not time |
| Seeds | **3** (1, 2, 3) | Quantifies model-initialisation variability |
| Total runs per arm | 3 × 5 = **15** | |
| Split seed | `20260914`, fixed | Folds are **identical across seeds**, so seed effects and fold effects do not confound each other |
| Validation fraction | 0.1, drawn inside the training participants | Used only for monitoring the pretext loss; never for selection against test data |
| Uncertainty | Paired participant bootstrap, 95% intervals | Resamples *participants*, not windows, because windows from one person are not independent |

**What the bootstrap intervals do and do not cover.** They are computed *conditional on the fitted
models* and hold the fold structure fixed. They therefore capture participant-sampling variability
but **exclude** seed and fold variability. Seed variability is reported separately as `seed_SD` in
every results table. Where a conclusion depends on a single fold, we say so.

---

## C. The final architecture, layer by layer

This section specifies the canonical model completely. Every number below was printed from the
instantiated model, not copied from notes.

### C.1 Overview diagram

```
INPUT  x : (B, 672, 4)            B = batch, 672 timesteps (7 days x 96 bins), 4 sensor channels
  │
  │  NaN handling: timesteps where every channel is NaN are zeroed and masked out
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ input_fc : Linear(4 -> 64)                          320 params  │
└─────────────────────────────────────────────────────────────────┘
  │  (B, 672, 64)
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ SHARED BACKBONE — Dilated TCN, 8 blocks                         │
│   blocks 0..6 : dilation 1,2,4,8,16,32,64   (24,704 params ea.) │
│   block   7   : dilation 128, widens 64 -> 320  (390,080 params)│
│   depth = 7 dilation levels, RECEPTIVE FIELD = 1021 timesteps   │
│                                              563,008 params     │
└─────────────────────────────────────────────────────────────────┘
  │  z : (B, 672, 320)
  ├──────────────────────────────┬──────────────────────────────────┐
  ▼                              ▼                                  │
┌──────────────────────────┐   ┌──────────────────────────────────┐ │
│ TREND BRANCH             │   │ SEASONAL BRANCH                  │ │
│ 7 causal Conv1d experts  │   │ 4 BandedFourierLayers            │ │
│ kernels 1,2,4,8,16,32,64 │   │ band (1,10)  -> 40 dims  24 h     │ │
│ each 320 -> 160 channels │   │ band (10,17) -> 40 dims  12 h     │ │
│ outputs left-trimmed     │   │ band (17,24) -> 40 dims   8 h     │ │
│   (causal: no future)    │   │ band (24,31) -> 40 dims   6 h     │ │
│ then MEAN over experts   │   │ concatenated -> 160 dims          │ │
│         6,503,520 params │   │ then Dropout(p=0.1)               │ │
│                          │   │              385,200 params       │ │
└──────────────────────────┘   └──────────────────────────────────┘ │
  │  V^T : (B, 672, 160)          │  V^S : (B, 672, 160)            │
  ▼                               ▼                                 │
  mean over time                  rFFT over time at bins [1,7,14,21,28]
  │                               │   (1 = weekly, 7/14/21/28 = 24/12/8/6 h)
  │  (B, 160)                     ├── amplitude = sqrt(Re² + Im²)  -> (B, 5, 160) -> (B, 800)
  │                               └── phase     = atan2(Im, Re)    -> (B, 5, 160) -> (B, 800)
  ▼                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ FROZEN REPRESENTATION  (B, 1760)                                │
│   columns    0 ..  159  : trend      (160)                      │
│   columns  160 ..  959  : amplitude  (800)                      │
│   columns  960 .. 1759  : phase      (800)                      │
└─────────────────────────────────────────────────────────────────┘
```

### C.2 Layer-by-layer table with tensor shapes

**Table C.2 — Complete HRD forward pass. Shapes are `(batch, …)`; parameter counts are from
`sum(p.numel() for p in module.parameters())` on the instantiated model.**

| # | Module | Input shape | Output shape | Parameters | Notes |
|---|---|---|---|---:|---|
| 0 | Input window | — | (B, 672, 4) | 0 | 7 days × 96 bins/day; 4 sensor channels |
| 1 | `input_fc` `Linear(4→64)` | (B, 672, 4) | (B, 672, 64) | 320 | Bias included |
| 2 | `temporal_encoding` = `none` | (B, 672, 64) | (B, 672, 64) | 0 | Identity in the canonical model (see C.5) |
| 3 | TCN block 0, dilation 1 | (B, 672, 64) | (B, 672, 64) | 24,704 | Residual, causal |
| 4 | TCN block 1, dilation 2 | (B, 672, 64) | (B, 672, 64) | 24,704 | |
| 5 | TCN block 2, dilation 4 | (B, 672, 64) | (B, 672, 64) | 24,704 | |
| 6 | TCN block 3, dilation 8 | (B, 672, 64) | (B, 672, 64) | 24,704 | |
| 7 | TCN block 4, dilation 16 | (B, 672, 64) | (B, 672, 64) | 24,704 | |
| 8 | TCN block 5, dilation 32 | (B, 672, 64) | (B, 672, 64) | 24,704 | |
| 9 | TCN block 6, dilation 64 | (B, 672, 64) | (B, 672, 64) | 24,704 | |
| 10 | TCN block 7, dilation 128 | (B, 672, 64) | (B, 672, **320**) | 390,080 | Output projection block |
| — | **Backbone subtotal** | | (B, 672, 320) | **563,008** | Receptive field **1021** |
| 11 | Trend expert `Conv1d(320→160, k=1)` | (B, 320, 672) | (B, 160, 672) | 51,360 | |
| 12 | Trend expert `k=2` | (B, 320, 672) | (B, 160, 672) | 102,560 | Left-trimmed to 672 (causal) |
| 13 | Trend expert `k=4` | ″ | ″ | 204,960 | |
| 14 | Trend expert `k=8` | ″ | ″ | 409,760 | |
| 15 | Trend expert `k=16` | ″ | ″ | 819,360 | |
| 16 | Trend expert `k=32` | ″ | ″ | 1,638,560 | |
| 17 | Trend expert `k=64` | ″ | ″ | 3,276,960 | Largest kernel = 672/8 ≈ 84, rounded down to 64 |
| 18 | Mean over the 7 experts | 7 × (B, 672, 160) | (B, 672, 160) | 0 | `V^T` |
| — | **Trend subtotal** | | | **6,503,520** | 87% of all encoder parameters |
| 19 | `BandedFourierLayer` band (1,10) | (B, 672, 320) | (B, 672, 40) | 115,560 | Complex weights; 24-h fundamental |
| 20 | `BandedFourierLayer` band (10,17) | ″ | (B, 672, 40) | 89,880 | 12-h harmonic |
| 21 | `BandedFourierLayer` band (17,24) | ″ | (B, 672, 40) | 89,880 | 8-h harmonic |
| 22 | `BandedFourierLayer` band (24,31) | ″ | (B, 672, 40) | 89,880 | 6-h harmonic |
| 23 | Concatenate + `Dropout(0.1)` | 4 × (B, 672, 40) | (B, 672, 160) | 0 | `V^S` |
| — | **Seasonal subtotal** | | | **385,200** | |
| — | **TOTAL ENCODER** | | | **7,452,048** | 4 input channels |

**Interpretation.** The trend branch holds 87% of the parameters, because a kernel-64 convolution
from 320 to 160 channels is large. This imbalance was tested directly: cutting total capacity to
4.03M parameters produced an *identical* RQ1 amplitude result (−0.027 either way), so the
parameter count is **not** what drives performance here. That test is Section I.3.

### C.3 Receptive field — why 1021

The TCN's receptive field is the number of input timesteps that can influence one output timestep.
With 8 blocks at dilations 1…128 and kernel width 3, it is **1021 timesteps**. The window is 672
timesteps. Because 1021 > 672, **every output timestep can see the entire window**. The depth is
chosen automatically as the smallest depth satisfying this (`depth_for_window`), so the model never
has a blind spot and never pays for depth it cannot use.

### C.4 The frequency bands — the central design decision

`rhythm_bands()` computes, from the sampling rate alone, which rFFT bin each daily harmonic falls
in, and gives each harmonic its own dedicated learned filter with everything outside its band
zeroed before the inverse transform.

For HRD (672 timesteps, 96 bins/day): the 24-hour cycle sits at bin 672/96 = **7**, and harmonic
*k* at bin 7*k*. Band edges are midpoints between neighbouring harmonics.

**Table C.4 — Harmonic bands on HRD, and what each captures physiologically.**

| Band | Centre bin | Period | rFFT bin range | Output dims | Physiological content |
|---|---:|---|---|---:|---|
| 1st harmonic | 7 | **24 h** | 1 – 10 | 40 | The fundamental circadian rhythm: the main sleep/wake cycle |
| 2nd harmonic | 14 | **12 h** | 10 – 17 | 40 | Bimodal day structure, e.g. the post-lunch activity dip |
| 3rd harmonic | 21 | **8 h** | 17 – 24 | 40 | Finer within-day structure tied to meals and activity bouts |
| 4th harmonic | 28 | **6 h** | 24 – 31 | 40 | The sharpest sub-daily structure the model may use |

Bins above 31 (periods shorter than ~4.6 hours) are **discarded entirely** — the model is
structurally forbidden from fitting them. Bin 0, the window mean, is excluded from the first band
(it starts at 1) so that the constant level belongs unambiguously to the trend branch. That
exclusion is the architectural basis of the trend/seasonal split.

**How much this narrows the search.** A 7-day HRD window has **337 usable frequency bins**. Our
four harmonic bands look at **30 of them (8.9%)**. The remaining 91% — every periodicity with no
known circadian or ultradian referent — is architecturally zeroed out of the seasonal branch.
Upstream CoST, by contrast, gives a single learned filter all 337 bins and asks it to discover,
from 113 people, which ones matter.

**Is 4 harmonics itself a tuned choice?** It was tested. Widening to 12 harmonic bands did not
help — it slightly *hurt* the personalised-rhythm signal (RQ2) rather than adding discriminative
information — so 4, matched to well-established ultradian harmonics, is the default. The count is a
config knob (`harmonics`), not a hard-coded constant.

For GLOBEM (112 timesteps, 4 bins/day) only two harmonics are resolvable, giving bands
(1, 42) and (42, 57) — one for the 24-hour cycle and one for its 12-hour harmonic.

### C.4b Why the model is 6× smaller than upstream CoST

A natural question: if the representation width is unchanged (320 dims either way), why does the
parameter count fall from 44.0M to 7.45M? The answer is that **two** of the design decisions above
each remove roughly half of the gap. Each row below was measured by instantiating the encoder with
exactly one setting swapped and everything else held fixed.

**Table C.4b — Where the 36.6M parameter difference from upstream CoST comes from. Measured, not
estimated. Baseline = the canonical DSSL encoder at 7,452,048 parameters (4 input channels).**

| Setting swapped to upstream's value | Total parameters | Added vs. DSSL | Share of the gap |
|---|---:|---:|---:|
| *(none — canonical DSSL)* | 7,452,048 | — | — |
| Seasonal filter sees all 337 bins instead of 30 | 24,375,168 | **+16,923,120** | **46.3%** |
| Trend kernels reach T/2 = 336 instead of T/8 = 64 | 27,113,168 | **+19,661,120** | **53.7%** |
| **Both together (= upstream CoST geometry)** | **44,036,288** | **+36,584,240** | **100.0%** |

**Interpretation.** The two contributions sum to exactly 100% of the gap, so nothing else is
involved. The two cuts are justified differently, and the distinction matters:

- **The spectrum cut is a *content* argument.** The ~307 discarded bins correspond to no known
  circadian or ultradian rhythm, so a filter there spends weights modelling structure with no
  physiological referent.
- **The trend-kernel cut is a *redundancy* argument.** The backbone's receptive field (1021) already
  exceeds the window (672), so every trend expert already sees the entire window. Kernels stretching
  across multiple days add parameters but **no additional context**.

**The honest summary:** the model did not shrink by making the representation smaller — the
representation is 320 dimensions either way. It shrank by removing a frequency range with no
circadian meaning and convolutional memory the backbone already makes redundant, in roughly equal
measure. With ~113–142 participants, a right-sized, biologically scoped model is a better match to
the data than 44M generic parameters searching 337 frequencies.

### C.5 Options that exist in the code but are **not** part of the canonical model

These are retained deliberately and are exercised by the test suite. They are *available
alternatives*, not dead code, and they are what makes the architecture claims falsifiable.

| Option | Canonical value | Alternatives retained | Why retained |
|---|---|---|---|
| `backbone` | `tcn` | `transformer`, `mamba` | Backbone-comparison ladder; `mamba` raises a clear `ImportError` if `mamba-ssm` is absent |
| `temporal_encoding` | `none` | `sinusoidal`, `time2vec` | Tested; neither is canonical |
| `seasonal_bands` | `harmonics` | `single` | `single` **is** upstream CoST's layer — required for the CoST control |
| `phase_readout` | `angle` | `circular` (emits cos, sin → 2560-dim readout) | Alternative circular readout; kept for the phase-representation comparison |
| `band_readout` | `False` | `True` | Restricts each band's dims to harmonics inside that band. Helps at 12 harmonics, *hurts* at 4 — hence off |
| `mask_mode` | `none` | `binomial` | Timestep masking; tested and rejected (Section I.1) |
| `method` | `dssl` | `cost_reference` | **The paired control in every experiment** |

---

## D. The training pipeline

### D.1 The self-supervised objective in words

DSSL is trained with **contrastive learning**: two randomly augmented views of the *same* window
should produce similar representations, while different windows should produce dissimilar ones. No
labels are used at any point in pretraining.

The two branches are trained differently, following CoST:

- **Trend branch — MoCo.** A momentum-updated copy of the encoder produces "key" representations,
  which are stored in a queue of 4,096 past keys. The loss asks the query representation to match
  its own key against those 4,096 negatives. The momentum copy (coefficient 0.999) keeps the queue's
  entries mutually consistent even though the encoder is changing.
- **Seasonal branch — within-batch instance discrimination in the frequency domain.** Both views
  go through the *same* encoder with gradients (no momentum copy, no queue), the sequence is
  Fourier-transformed, and amplitude and phase are contrasted **separately**. Because there is no
  queue, there are no stale keys to keep consistent, and the loss is symmetric in its two arguments
  — detaching one side would zero half the gradient path. This asymmetry with the trend branch is
  intentional and matches upstream CoST.

### D.2 The loss, term by term

```
L = w_trend · L_trend  +  alpha · ( w_amp · L_amp  +  w_phase · L_phase )
```

**Table D.2 — Every loss coefficient in the canonical model, with its value and role.**

| Symbol | Config key | Value | What it weights |
|---|---|---:|---|
| `w_trend` | `weights: "contracted"` | **0.277** | The MoCo trend term |
| `w_amp` | ″ | **0.14821124361158433** | The seasonal amplitude contrast (= 0.174/1.174) |
| `w_phase` | ″ | **0.85178875638841570** | The seasonal phase contrast (= 1.0/1.174) |
| `alpha` | `alpha` | **0.005** | Global scale on the whole seasonal half of the loss |
| `K` | `moco_k` | **4096** | MoCo queue length |
| `m` | (fixed) | **0.999** | MoCo momentum |
| `T` | (fixed) | **0.07** | Contrastive temperature |

**Where the `contracted` weights come from, and why they are flagged as a bet.** Upstream CoST's
preset, named `paper` (1.0 / 0.5 / 0.5), weights the three terms equally-ish and is what the CoST
control uses. We measured how much personalised-rhythm signal each *block* of the representation
carries on its own, using RQ2 concordance (chance = 0.5):

| Representation block | Measured concordance | Signal above chance |
|---|---:|---:|
| Seasonal **phase** | **0.8802** | 0.3802 |
| **Trend** | 0.6054 | 0.1054 |
| Seasonal **amplitude** | 0.5661 | 0.0661 |

Upstream's `1 : 0.5 : 0.5` recipe weights these in close to the **wrong order** — it spends the most
gradient on trend and equal effort on amplitude, which carries the least signal. `CONTRACTED`
re-allocates the weights in proportion to the measured signal-above-chance, normalised so phase = 1
(giving trend 0.277, amp 0.174, phase 1.000), and then renormalises the seasonal pair to sum to 1 so
that `alpha` keeps its original meaning as the seasonal-versus-trend scale. The result puts roughly
**5.75× more weight on phase than on amplitude**.

**This is explicitly a bet, not a correction, and it is the only one among the four changes.** The
band and depth changes fix defects that are wrong under any hypothesis; this re-weighting is
inferred from concordances that were themselves measured under one configuration. That is precisely
why the `paper` preset is retained and run as the paired control in every experiment.

**`phase_mode: "circular_amp"`** — the circular phase treatment, and one of the four substantive
changes from upstream CoST. Phase is *time of day*, and time of day wraps. Treated as plain numbers,
23:59 (23.98) and 00:01 (0.02) look nearly a full day apart under a Euclidean loss, even though the
two clock hands are almost touching; two identical phases score 0 at φ=0 but π² at φ=π. We map every
angle onto the unit circle as `(sin φ, cos φ)` before comparing views, so similarity becomes
`cos(Δφ)` — a function of the true angular gap. We additionally weight each channel's embedding by
that channel's **amplitude**: if a channel's 24-hour rhythm is essentially flat, its phase is
meaningless noise (like asking the time from a clock with no hands), so it is down-weighted rather
than allowed to vote as loudly as a clear reading.

This carries into evaluation, not just training: RQ1 phase error is always computed through the same
`(cos, sin)` representation as circular error in hours, and the probe's feature scaler
(`IsotropicPairScaler` in `evaluation_protocol.py`) was written specifically so that scaling a
`(cos, sin)` pair does not warp the circle into an ellipse — which a naive per-column
`StandardScaler` applied separately to `cos` and `sin` would do.

### D.3 Augmentations

Two views of each window are produced by independently sampling the following, each applied with
probability 0.5:

**Table D.3 — Augmentations in the canonical model.**

| Augmentation | Parameter | Value | Effect on the rhythm |
|---|---|---:|---|
| Circular box smoothing | `smooth_minutes` | **75.0** (= 5 bins on HRD, 0 on GLOBEM) | Random odd width; circular and odd, so window length and **every rhythm's phase are exactly preserved** |
| Additive Gaussian jitter | `jitter_sigma` | **0.1** | Per-element noise |
| Constant per-channel offset | `shift_sigma` | **0.5** | Changes **only** the f=0 bin — i.e. the MESOR — leaving all harmonics untouched |
| Random scaling | `scale_sigma` | **0.0** (off) | Off in DSSL; **on at 0.5** in the CoST control, which is upstream's setting |

The design logic: smoothing perturbs high-frequency content without touching phase; the shift
perturbs level without touching rhythm. Scaling is switched off in DSSL precisely because it would
perturb amplitude, which is a quantity we want the representation to *track*, not be invariant to.
On GLOBEM, `smooth_minutes=75` yields 0 bins because no odd box ≥ 3 fits inside 6-hour bins, so
smoothing is automatically inactive there.

### D.4 Optimisation

**Table D.4 — Training configuration.**

| Setting | Value | Notes |
|---|---|---|
| Optimiser | SGD, momentum 0.9, weight decay 1e-4 | |
| Learning rate | **5e-4** | |
| Schedule | Half-cycle cosine decay to 0 over the full horizon | `adjust_learning_rate` |
| Batch size | **64** | Must divide the MoCo queue length (4096/64 = 64) |
| Iterations | **6000** | Fixed in advance in the config, per fold per seed |
| Epoch multiplier | 10 | Each window is revisited 10× per epoch with fresh augmentations |
| Dropout | 0.1, on the seasonal branch output only | |

**Checkpoint selection rule — stated explicitly because it is a common source of leakage.**
There is **no** checkpoint selection. Training runs for exactly 6000 iterations and the **final**
model is used. The held-out validation fraction (10% of *training* participants) is used only to
log the pretext loss for monitoring; it never selects a checkpoint, and test participants are never
touched during training. This removes an entire class of selection leakage at the cost of possibly
not using the single best iterate.

### D.5 Determinism and numerical reproducibility

This required real engineering effort and is worth stating, because it affected earlier results.

- By default an A100 GPU runs convolutions in **TF32** (~1e-3 relative precision) with a kernel
  chosen by batch shape. That meant a window's encoding depended on which batch it was encoded in —
  measured at up to **0.37% drift** (Narval job 3068780) — and a reloaded encoder did not reproduce
  its own outputs.
- The fix (`exact_numerics()` / `tf32_convolutions()`): **training** uses TF32 convolutions, which
  is what the reported runs always did and which keeps training at a workable 8.6 s/update; but
  **every encoding** is made in full float32 with deterministic kernels. Training batches always
  have one shape, so runs stay deterministic.
- `spectral_readout` uses `eps = 1e-3` inside the `atan2` and the amplitude `sqrt`. This is not
  cosmetic: reading the seasonal sequence raw leaves many bins genuinely near zero amplitude, where
  `atan2` is ill-conditioned. Float32 non-associativity shifted a bin's seasonal output by 7.45e-08
  — negligible everywhere else — yet swung that bin's *angle* by up to **0.65 radians**. `eps=1e-3`
  is ~10,000× the measured noise floor and negligible against any real signal (≥0.01 in every
  measurement), so a near-zero bin now reports a stable, uninformative angle of π/4 instead of noise.
- Full training state — optimiser, RNG states for Python/NumPy/Torch/CUDA, the exact shuffled
  sampling position — is checkpointed atomically, and resuming reproduces uninterrupted training
  **bit for bit**. This is verified by a test
  (`test_complete_training_resume_matches_uninterrupted_training`).

---

## E. RQ1 — Does the model preserve individual rhythms?

### E.1 What was tested and how

**The question.** Take a held-out participant's window. Compute its true rhythm markers directly
from the raw signal. Now ask: can a simple linear model, fitted on *training* participants only,
recover those markers from the frozen representation of that window?

**The markers** ("families"), each computed per channel and then averaged within a person:

| Marker | Meaning | Units |
|---|---|---|
| `amplitude` | Strength of the 24-hour cosinor component | z-score units |
| `phase_hours` | Time of day of the rhythm peak (acrophase) | hours |
| `MESOR` | The window mean, i.e. the level the rhythm oscillates around | z-score units |
| `IS` | Interdaily stability — how consistent the pattern is from day to day | unitless, 0–1 |
| `IV` | Intradaily variability — how fragmented the rhythm is within a day | unitless |
| `RA` | Relative amplitude, from the most/least active hours | unitless, 0–1 |

**The metric.** *Error reduction* relative to a control: positive means DSSL makes **smaller**
errors than that control (**↑ better**). Reported with paired participant-bootstrap 95% intervals.
"Inconclusive" means the interval includes zero — it is **not** evidence of equivalence.

**The controls, and why each exists:**

| Control | What it isolates |
|---|---|
| `training_mean` | The population pattern. Beating it means the model captures *something individual* |
| `untrained` | **The critical control.** An identically-initialised, identically-shaped, *never-trained* encoder. Beating it means the *training* helped, not merely the architecture |
| `raw` | The raw window itself. Beating it means the representation adds something over the input |
| `pca`, `random_projection` | Generic compressions of matched width |
| `cost_reference_adapter` | Upstream CoST behind the identical readout |

### E.2 Results

**Table E.2 — RQ1 on HRD: error reduction of DSSL relative to each control, canonical run
`narval_v2`, 3 seeds × 5 folds. Positive = DSSL better (↑). Source:
`docs/VALIDATION_RESULT.md`.**

| Marker family | vs. population mean | vs. **untrained encoder** | Verdict on this family |
|---|---|---|---|
| `amplitude` | beats it | **+0.029 ↑ favours DSSL** | DSSL wins |
| `MESOR` | beats it | **+0.040 ↑ favours DSSL** | DSSL wins |
| `IV` (fragmentation) | beats it | **+0.035 ↑ favours DSSL** | DSSL wins |
| `phase_hours` (acrophase) | beats it | **−0.023 ↓ favours control** | **DSSL loses** |
| `IS` (day-to-day stability) | beats it | **−0.019 ↓ favours control** | **DSSL loses** |
| `RA` | beats it | inconclusive (interval spans 0) | Undetermined |

**Pre-registered criteria:**
- RQ1(a) — beats the population mean in **every** eligible family: **MET** on HRD.
- RQ1(b) — beats the untrained encoder in **every** eligible family: **NOT MET** (phase and IS fail).

**Plain interpretation.** The representation genuinely encodes individual rhythm information —
it is far better than simply predicting the population average for every person. But for two of the
six markers, an *untrained* network of the same shape does it as well or better. In other words, a
substantial part of what looks like "learned rhythm structure" is actually structure the
architecture possesses at random initialisation, before any training. Under our pre-registered
conjunction rule, **RQ1 is partially supported**, not supported.

**On GLOBEM, RQ1(a) is NOT met** — amplitude (+0.030) and IV (−0.009) are inconclusive against the
population mean. This is a regression relative to the earlier run and is reported as a **failure of
cross-cohort generalisation**, not omitted.

### E.3 RQ1-D — Branch disentanglement

**The question.** Is each marker predicted better from *its own* branch than from the *other*
branch? If the trend block can predict the 24-hour amplitude just as well as the seasonal block
can, the branches are not actually separated, whatever we call them.

**The metric.** For each target we fit a ridge regression from the own branch and, separately,
from the other branch ("leakage"), and report held-out R² for each. "**own − leakage**" is the
difference; "separated" means it is above 0 with its 95% interval.

**Table E.3 — Branch separation on HRD, canonical run `narval_v2`. Higher own−leakage = cleaner
separation (↑). Source: `docs/VALIDATION_RESULT.md`.**

| Target | own − leakage (DSSL) | Separated? | DSSL − untrained | Which is better? |
|---|---:|---|---:|---|
| MESOR | **+0.544** | Yes | **+0.077 ↑** | DSSL |
| 24-h amplitude | **+0.134** | Yes | −0.018 ↓ | Untrained |
| 24-h acrophase | **−0.083** | **No — inconclusive** | **−0.209 ↓** | Untrained |

**Pre-registered criteria:**
- RQ1-D(i) — own > leakage for all three: **NOT MET** (acrophase fails).
- RQ1-D(ii) — better separation than untrained on all three: **NOT MET** (two of three fail).

**Plain interpretation, and the most important mechanistic finding in the project.** The level
(MESOR) is very cleanly separated — the trend branch owns it, decisively. Amplitude is separated
but by less than the untrained control achieves. Acrophase is **not** separated at all: the trend
branch predicts the rhythm's timing about as well as the seasonal branch does.

We traced *why*, and the answer is not what we expected. **The leakage is geometric, not learned.**
At random initialisation — before any training whatsoever — the trend block already predicts 24-hour
amplitude at R² = 0.827 and acrophase at R² = 0.642. A time-pooled convolutional summary of a
rhythmic signal is simply *not independent* of that signal's rhythm; the two branches read the same
backbone output, and pooling does not erase periodic structure. Training does not create this
leakage, and in our experiments training did not remove it either.

We verified this is fixable in principle: an architecture that decomposes the *input* before the
encoder drove acrophase leakage from 0.669 to **0.016** when trained (and 0.642 → 0.024 untrained).
That architecture was nevertheless rejected — it was worse on nearly everything else. See I.3.

---

## F. RQ2 — Can the model detect within-person rhythm changes?

**This is the research question the model was designed for, and the one it answers positively.**

### F.1 What was tested and how

The setup is a controlled perturbation experiment, and it is worth walking through carefully
because it is the most convincing evidence in this report.

1. Take a participant's current week and their **four preceding weeks**, all real data.
2. Compute the representation of each. The four preceding weeks define that person's **personal
   baseline**: a mean `mu` and a standard deviation `sd`, *per person*, not across the cohort.
   Windows without a full four-week contiguous reference are **never scored**.
3. Artificially perturb the current week by a known amount, in one of two ways:
   - **Timing arm** — circularly shift the sensor channels by 0.5 to 4 hours. Because the window is
     a whole number of days, this is *exactly* a rotation of the 24-hour cosinor coefficient: the
     rhythm's timing changes, its strength and level do not.
   - **Strength arm** — multiply *only* the 24-hour cosinor component by (1 ± α). Over a window of
     whole days that component is orthogonal to the mean and to every other harmonic, so the
     rhythm's strength changes *exactly* and its level and timing do not.
4. Compute the perturbed week's distance from the personal baseline:
   `dscore = sqrt(mean(((V − mu)/sd)²))` — a person-normalised distance across all 1760 dimensions.
5. **Concordance** = the fraction of perturbation pairs where the larger perturbation produced the
   larger `dscore`. **0.5 = chance.** Higher is better (↑).

**Why this design is strong.** The perturbation is applied to real data, the ground-truth ordering
is known by construction, all model parameters are frozen, and the comparison is *within person* —
so between-person confounds cannot produce a positive result.

### F.2 Results

**Table F.2 — RQ2 concordance on HRD. 0.5 = chance; higher is better (↑). Canonical run
`narval_v2`, 3 seeds × 5 folds; 102 participants (timing) and 106 (strength). Sources:
`docs/VALIDATION_RESULT.md`; the pre-repair figures from `results/hrd/narval_v1/`.**

| Arm | Before the repair (`narval_v1`) | **Canonical (`narval_v2`)** | vs. chance | vs. untrained encoder |
|---|---:|---:|---|---|
| **Timing** (0.5–4 h shifts) | 0.789 | **0.759** | above chance ✓ | **+0.049 [+0.030, +0.069] ↑ favours DSSL** |
| **Strength** (24-h amplitude ×(1±α)) | **0.438 — below chance** ✗ | **0.713** | above chance ✓ | **+0.054 [+0.042, +0.066] ↑ favours DSSL** |

**All four pre-registered RQ2 conditions are MET. This is the first time in the project that any
research question was fully supported.**

**Plain interpretation.** Given a person's own recent history, the frozen representation reliably
detects both *when* their rhythm has shifted in time and *how much* its strength has changed —
and it does so better than an untrained network of the same architecture, which is the control
that isolates the contribution of training. The strength arm previously performed *below chance*
(0.438), meaning the representation was systematically ranking bigger amplitude changes as
*smaller* deviations. That is now 0.713.

### F.3 Why the strength arm was broken, and why the fix counts as a prediction rather than a
tuning exercise

This is methodologically the most defensible result in the report, so the sequence matters:

1. **The mechanism was identified on an *untrained* encoder**, with no labels and no test data
   involved. Upstream CoST L2-normalises each timestep across channels before the readout. DSSL's
   harmonic bands start at bin 1, so the seasonal sequence has **no constant component to anchor
   that norm** and it saturates toward a square wave — destroying amplitude information.
2. **It was measured.** Scaling a window's true 24-hour amplitude by 0.5 / 1.0 / 1.5 moved the
   amplitude feature to **8.06 / 12.20 / 14.09** under normalisation (compressive, and nearly
   unable to distinguish the cases) versus **1.34 / 2.67 / 3.99** without it (cleanly proportional).
3. **The fix — read the seasonal sequence raw — was adopted for that reason, and written into
   `docs/VALIDATION_PRECOMMIT.md` and committed (`8bc9b6e`) BEFORE the protocol run.** The
   pre-commitment explicitly states the expectation that RQ2 strength would move above chance.
4. **Then the run was executed, and 0.438 → 0.713 was observed.**

The prediction preceded the measurement, and the mechanism was established without touching the
evaluation data. Section H explains why this also means the canonical model was **not** selected
by looking at the comparison table.

### F.4 The costs of the same change, reported as required

The pre-commitment required that regressions caused by the adopted change be reported alongside
its benefit. They were:

| Quantity | Before (`narval_v1`) | After (`narval_v2`) | Direction |
|---|---:|---:|---|
| RQ1 phase recovery vs. raw | −0.052 | **−0.142** | ↓ worse |
| RQ1 phase recovery vs. untrained | +0.031 (favoured DSSL) | **−0.023 (favours control)** | ↓ worse |
| RQ1-D acrophase separation vs. untrained | −0.156 | **−0.209** | ↓ worse |
| RQ2 timing concordance | 0.789 | **0.759** | ↓ worse |
| RQ2 timing vs. CoST reference | +0.030 (favoured DSSL) | −0.016 (inconclusive) | ↓ worse |
| RQ2 strength vs. CoST reference | — | −0.045 (favours control) | ↓ CoST is better here |
| GLOBEM RQ1(a) | met | **not met** | ↓ worse |
| RQ1-D amplitude own R² | 0.845 | **0.977** | ↑ better |
| RQ1 amplitude / IV / MESOR vs. untrained | did not favour DSSL | **all now favour DSSL** | ↑ better |

**There is a genuine trade-off here and we do not hide it.** Reading the sequence raw restores
amplitude (which fixed RQ2 strength) at the direct cost of phase, because raw reading is exactly
what makes `atan2` ill-conditioned near zero amplitude. We attempted to have both — a "split"
readout taking amplitude raw and phase normalised — and evaluated it at full protocol grade. It
failed: see I.2.

---

## G. RQ3 — Does the model improve depression prediction?

### G.1 What was tested and how

**The metric.** Participant-level **AUROC** (area under the ROC curve) for the endpoint label
`CES-D ≥ 16`. 0.5 = chance, 1.0 = perfect; higher is better (↑). A participant's probability is
the mean of their windows' probabilities. Balanced accuracy and macro-F1 at a **fixed 0.5 decision
threshold** are secondary. The probe is a frozen **linear** logistic model. No threshold was tuned
at any point.

**The two primary controls (a conjunction — both must be beaten):** `raw` windows, and the
`untrained` encoder.

### G.2 Results

**Table G.2 — RQ3 participant AUROC on HRD (↑ better; 0.5 = chance). Canonical run `narval_v2`,
3 seeds × 5 folds, 113 participants. Sources: `docs/VALIDATION_RESULT.md` and the baseline ladder
in `results/hrd/*/tcn_none/SUMMARY.md`.**

| Method | Category | AUROC | DSSL − method | Evidence |
|---|---|---:|---:|---|
| `training_prevalence` | Trivial | 0.492 | +0.21 | favours DSSL |
| **`raw`** (window itself) | **Primary control** | **0.695** | **+0.009 [−0.035, +0.056]** | **inconclusive** |
| **`untrained`** (same architecture, no training) | **Primary control** | **0.676** | **+0.028 [−0.009, +0.067]** | **inconclusive** |
| `yan_cosinor` (reference paper's method) | Reference | 0.697 | ≈ −0.00 | inconclusive |
| `handcrafted_stack` | Handcrafted | 0.695 | ≈ −0.00 | inconclusive |
| `handcrafted` | Handcrafted | 0.676 | ≈ +0.03 | inconclusive |
| `nonparametric` | Handcrafted | 0.670 | ≈ +0.03 | inconclusive |
| `random_projection` | Compression | 0.675 | ≈ +0.03 | inconclusive |
| `pca` | Compression | 0.623 | +0.081 | favours DSSL |
| `cost_reference_adapter` | Upstream CoST | 0.624 | +0.080 | favours DSSL |
| **`dssl`** | **Proposed** | **0.704** (seed SD 0.021) | — | — |

**Pre-registered criteria:**
- DSSL above `raw`: **NOT MET** (interval includes 0).
- DSSL above `untrained`: **NOT MET** (interval includes 0).
- **RQ3 is NOT SUPPORTED on HRD.** It is also not supported on GLOBEM (+0.020 vs. raw, −0.026 vs.
  untrained, both inconclusive).

**Plain interpretation.** DSSL's 0.704 is the highest number in the table, and it beats upstream
CoST (+0.080) and PCA (+0.081) conclusively. But it does **not** beat the two controls that the
protocol designated as primary. The honest statement is: *the learned representation is no better
at predicting this endpoint than the raw data it was computed from.* Under rule 4, beating CoST and
PCA is context, not a substitute. We do not report this as a success.

### G.3 Control fairness — an accounting requirement we imposed on ourselves

Rule 6 of the pre-commitment requires decomposing any DSSL-vs-untrained gap into "the
representation improved" versus "the control got worse", because `untrained`, `pca` and
`cost_reference_adapter` all *share the DSSL readout* and therefore move whenever the DSSL
configuration moves. `raw`, `yan_cosinor`, `handcrafted` and the others are **fixed anchors**
(seed SD exactly 0.000).

| Quantity | `narval_v1` | `narval_v2` | Change |
|---|---:|---:|---:|
| `raw` — fixed anchor | 0.695 | 0.695 | 0.000 |
| `dssl` | 0.697 | **0.704** | **+0.007** |
| `untrained` — moves with DSSL | 0.690 | 0.676 | **−0.014** |
| Apparent DSSL−untrained gap | +0.006 | +0.028 | +0.022 |

**Interpretation.** The DSSL-minus-untrained gap appears to have widened four-fold. But against the
*fixed* anchor, DSSL improved by only +0.007. **About two-thirds of the apparent widening is the
control degrading, not the representation improving.** Both comparisons remain inconclusive either
way. We flag this because reporting only the +0.028 would have been materially misleading.

### G.4 An earlier promising result that did not replicate

A single-seed screen reported DSSL at **0.726** on HRD RQ3. An adversarial audit predicted it would
not survive, because the advantage was concentrated in one fold. At 3 seeds it is **0.704** (seed SD
0.021) and the comparison against raw stays inconclusive. **No conclusion in this report rests on
the 0.726 figure.** It is recorded here only so that it cannot resurface as evidence later.

### G.5 Why RQ3 may be unanswerable with this dataset — stated as a limitation, not an excuse

This was recorded in the pre-commitment **before** the result was known, which is what entitles us
to raise it now.

**The endpoint is largely a trait, not a state.** The label is `ces_d_endpoint_score ≥ 16`. The
participant's *baseline* depression status already matches that endpoint label for **80.4%** of
participants (90 of 112); only **22 participants switch status** over the study. And baseline CES-D
score *alone* predicts the endpoint at **AUROC 0.876** — against 0.49–0.73 for the *entire*
wearable ladder, DSSL included.

So RQ3, as written, asks the wearable representation to predict something that is largely
determined before the wearable data was collected. We did **not** redefine RQ3 in response to this.
Incremental-over-baseline analysis is explicitly future work and cannot be substituted for RQ3.

**Statistical power.** With n = 113 and roughly 22 participants per fold, per-fold AUROC swings
between 0.51 and 0.87 for *every* method. A prior learning-curve analysis put representation-side
gains at roughly +0.02–0.03 AUROC per doubling of labelled participants. **A negative RQ3 under this
protocol is a plausible true result, not necessarily a defect in the model.**

---

## H. How the final model was chosen

### H.1 The four changes from upstream CoST, and the evidence for each

| # | Change | Upstream CoST | DSSL | Evidence it was the right call |
|---|---|---|---|---|
| 1 | **Frequency bands** | One band over the whole spectrum (337 bins) | 4 dedicated bands at the daily harmonics | Beats CoST on RQ1 MESOR separation (+0.332) and RQ3 (+0.080) |
| 2 | **Phase treatment** | Raw real-valued angles | Circular, amplitude-weighted (`circular_amp`) | Loss weight on phase is 5.75× amplitude; RQ2 timing 0.759 vs. chance 0.5 |
| 3 | **Trend kernels** | Up to T/2 (= 336 on HRD) | Up to T/8 (= 64) | The backbone's receptive field (1021) already covers the window, so longer kernels add parameters and no context |
| 4 | **Readout** | Per-timestep L2 normalisation | **Raw seasonal sequence** | The decisive one: RQ2 strength 0.438 → 0.713 (Section F.3) |

A fifth change, **loss-term re-weighting** (`paper` 1:0.5:0.5 → `contracted` 0.277:0.148:0.852), is
listed separately because it is explicitly a **bet rather than a correction** (D.2). Changes 1–4 fix
things that are wrong under any hypothesis; the re-weighting is inferred from measured concordances
and is therefore always run against the `paper` preset as a paired control.

Changes 1 and 3 also apply to the augmentations: DSSL turns *scaling* off (it would destroy the
amplitude signal we want to track) and adds *smoothing*, which is circular and odd-width so it
preserves every rhythm's phase exactly. Together, changes 1 and 3 are also what reduce the model
from 44.0M to 7.45M parameters, in near-equal measure (Table C.4b).

**Anticipated questions, answered in one line each:**

- *"Why not use CoST as-is?"* — Two of its mechanisms work directly against what we need: it treats
  clock time as a plain number (which breaks at midnight), and its trend branch is trained to
  discard baseline level, which is clinically informative. These are structural mismatches, not
  tuning issues.
- *"Isn't fixing 24/12/8/6 h a strong assumption?"* — Yes, deliberately. It is a well-established
  physiological assumption, and it is a far safer bet than asking an unconstrained filter to find
  structure across 337 frequencies using 113 people.
- *"Did shrinking to 7.45M parameters hurt the model?"* — The representation is 320 dimensions
  either way; the shrinkage removes frequency bins and convolutional memory that carry no circadian
  information (C.4b). Separately, an explicit capacity ladder found that halving parameters to 4.03M
  changed RQ1 amplitude not at all (I.4), so capacity is not the binding constraint here.
- *"Is the re-weighted loss a fix or a guess?"* — Explicitly a data-driven hypothesis. The original
  `1:0.5:0.5` weighting is run in parallel as a control so this one choice can be judged alone.

### H.2 The candidates evaluated at full protocol grade

Four complete configurations were run at full protocol grade (3 seeds × 5 folds, both cohorts,
full baseline ladder). Their differences are all in the **readout** or the **branch structure**;
the backbone, objective, optimiser and data are identical throughout.

**Table H.2 — All protocol-grade candidates on HRD. Best value per column in bold. "vs. untrained"
compares against an untrained encoder sharing each arm's own readout, so the control moves with the
arm. Sources: `results/hrd/<run>/tcn_none/SUMMARY.md`; canonical row from
`docs/VALIDATION_RESULT.md`.**

| Configuration | What it does differently | RQ2 timing ↑ | RQ2 strength ↑ | RQ1 amplitude vs. untrained ↑ | RQ1 phase vs. untrained ↑ | RQ3 AUROC ↑ | RQ2 fully met? |
|---|---|---:|---:|---:|---:|---:|---|
| **`narval_v2` — CANONICAL** | Seasonal sequence read **raw** | 0.759 | **0.713** | +0.029 | −0.023 | 0.704 | **YES** |
| `narval_v2_timestep` | Per-timestep L2 normalisation (upstream CoST's readout) | 0.796 | **0.399 — below chance** | **+0.035** | **+0.033** | **0.712** | NO |
| `narval_v2_split` | Amplitude raw, phase normalised | **0.800** | 0.557 (loses to untrained, −0.081) | +0.020 | −0.019 | 0.696 | NO |
| `narval_v3_decomposed` | Input decomposed *before* the encoder | 0.768 | 0.687 | −0.039 | −0.054 | 0.665 | **YES** |
| `narval_v1` (pre-repair) | The earlier model, for reference | 0.789 | **0.438 — below chance** | −0.004 | +0.031 | 0.697 | NO |

### H.3 **No configuration dominates every metric. We state this plainly.**

The table above makes the trade-off explicit, and it would be dishonest to present the canonical
model as a clean winner:

- **`narval_v2_timestep` is the best on three columns** — RQ3 AUROC (0.712), RQ1 amplitude (+0.035)
  and RQ1 phase (+0.033). It is the only arm where DSSL beats the untrained control on *both* RQ1
  markers shown. But its RQ2 strength concordance is **0.399, below the 0.5 chance level** — the
  representation systematically ranks larger amplitude perturbations as smaller deviations. It
  fails RQ2 outright.
- **`narval_v2_split` has the best RQ2 timing (0.800)**, but its strength arm loses to its own
  untrained control (−0.081), so it also fails RQ2.
- **`narval_v3_decomposed` also fully satisfies RQ2** (timing +0.049, strength +0.029 vs. untrained)
  and beats the canonical model on timing (0.768 vs. 0.759). But it is worse on RQ1 amplitude
  (−0.039 vs. +0.029), RQ1 phase (−0.054 vs. −0.023), RQ2 strength (0.687 vs. 0.713) and RQ3 (0.665
  vs. 0.704).

**Two configurations therefore satisfy RQ2 fully: the canonical model and the decomposed one.** The
canonical model is preferred over the decomposed one because it wins on four of the five remaining
columns and loses on none of them.

### H.4 Why the canonical model was retained — and why this is not post-hoc selection

The reasoning matters as much as the conclusion:

1. **The choice was pre-registered on mechanistic grounds, before the comparison existed.** The
   decision to read the seasonal sequence raw was made from an *untrained*-encoder measurement
   (Section F.3), written into `docs/VALIDATION_PRECOMMIT.md`, and committed as `8bc9b6e` **before**
   the protocol run was launched. Table H.2 is *confirmatory context*, not the selection mechanism.
2. **RQ3 was never used to select the architecture.** This is a standing constraint of the project,
   and it is why we did **not** adopt `narval_v2_timestep` despite its higher RQ3 AUROC (0.712 vs.
   0.704). Selecting on the held-out downstream metric would have contaminated the only clean
   downstream evaluation we have. Note also that the 0.712 vs. 0.704 difference is well within the
   seed standard deviation (0.021) and is not a meaningful difference in any case.
3. **RQ2 is the question this model exists to answer**, and the canonical configuration is the one
   with the largest margin on the arm that was previously broken (strength 0.713, +0.054 over its
   own untrained control).
4. **The alternatives fail a pre-registered criterion, not a preference.** `timestep` and `split`
   fail RQ2 under rule 3, which was fixed in advance. That is a binding disqualification.

**What we give up by this choice, stated honestly:** phase recovery. The canonical model is the
*worse* of the two viable options on RQ1 phase (−0.023 vs. `timestep`'s +0.033). We accept that
cost because RQ2 is a conjunction that `timestep` fails outright, and because the phase cost has a
known, documented mechanism (`atan2` conditioning) rather than being an unexplained regression.

---

## I. Negative findings — what did not work

This section exists because these experiments consumed real effort and their conclusions constrain
what anyone should try next. The full detail, including exact configurations and per-fold numbers,
is in **`FAILED_EXPERIMENTS.md`**; this is the summary.

**A note on evidence grades**, used throughout:
- **A — mechanistic:** measured directly on an untrained encoder or in closed form. No test data.
- **B — ablation screen:** 1 seed, 2–5 folds. Indicative, not conclusive.
- **C — protocol grade:** 3 seeds × 5 folds, full ladder, pre-registered. Conclusive.

### I.1 Timestep masking and MoCo queue restoration — rejected (grade B)

**Hypothesis:** the trend branch's MoCo top-1 accuracy saturating at 1.000 means the contrastive
task is too easy, so restoring upstream's binomial timestep masking would make it harder and
improve the representation. **Result:** no improvement on the RQ1 proxy. **Decision:** rejected;
`mask_mode` stays `none`. **Note:** trend saturation remains *unfixed* and is reported as a
standing limitation (J.1), not as something we resolved.

### I.2 Alternative readouts — rejected (grade C, full protocol)

Three readout variants were run at full protocol grade: raw (canonical), per-timestep normalised
(`timestep`), and a split taking amplitude raw and phase normalised (`split`). The split was
designed specifically to get amplitude *and* phase — amplitude needs the unnormalised magnitude,
phase needs bins held away from zero to keep `atan2` conditioned.

**Result:** the split did not recover the amplitude signal. RQ2 strength was 0.557 and **lost to its
own untrained control** (−0.081). **Conclusion:** the amplitude/phase trade-off is not separable by
choosing a different normalisation per block. **Decision:** rejected; the `readout_norm` option has
been removed from the code entirely and the raw behaviour is hard-wired.

### I.3 Pre-encoder decomposition — rejected (grade C, full protocol)

**Hypothesis:** the acrophase leakage found in RQ1-D (E.3) is geometric — the two branches read the
same backbone output, so pooling cannot remove periodic structure. Decomposing the *input* into a
daily-average component and a residual, before the encoder, should eliminate it.

**Result on the stated hypothesis: confirmed, decisively.** Acrophase leakage fell from 0.669 to
**0.016** when trained, and from 0.642 to **0.024** untrained. The mechanism was exactly as
diagnosed.

**Result on everything else: worse.** RQ1 amplitude −0.039 (vs. +0.029 canonical), RQ1 phase −0.054
(vs. −0.023), RQ2 strength 0.687 (vs. 0.713), RQ3 0.665 (vs. 0.704).

**Decision:** rejected. **Conclusion: disentanglement trades against recovery.** Forcing the
branches apart at the input removes leakage but costs the representation's ability to recover the
markers at all. The implementation is preserved in
`archive/failed_experiments/dssl_v2_decomposition/` because it is the only implementation of a
result we report quantitatively.

### I.4 The capacity explanation — falsified (grade C). **This was our own error, corrected.**

When the decomposed architecture underperformed, we initially wrote that its 0.54× parameter
reduction was "the likely cause". **That claim was wrong, and we tested it rather than leaving it
standing.** A within-architecture capacity ladder compared **7.51M** against **4.03M** parameters
and obtained an *identical* RQ1 amplitude result: **−0.027 in both cases**.

**Conclusion:** capacity is not what limits this model. The earlier statement is retracted. (An
earlier attempt at this control was itself badly designed — we tried to shrink the model via
`hidden_dims=42` and got 7.33M parameters anyway, because the trend head's size does not depend on
`hidden_dims`. The corrected ladder replaced it.)

### I.5 Level-equivariance loss term (`w_eq`) — rejected (grade B, 5 folds)

**Hypothesis:** adding a loss term rewarding the trend branch for *tracking* level changes rather
than being invariant to them would help. **Result — dose-response over 5 folds**, RQ1 proxy:

| `w_eq` | 1.0 | 0.05 | **0** |
|---|---:|---:|---:|
| RQ1 proxy ↑ | −0.0343 | −0.0136 | **+0.0096** |

The ordering held in **all 5 folds**. **Decision:** `w_eq = 0`. The coefficient, its loss function,
its projection head and the third element of every training batch have all been **removed from the
code**, because a coefficient established at zero is not a hyperparameter, it is dead weight.

### I.6 Disjoint-day trend pairing — rejected (grade B, 5 folds)

**Hypothesis:** building the two contrastive views from *non-overlapping days* would remove the
MoCo trend saturation. **Result:** it did remove the saturation, but **regressed RQ1 in 5 of 5
folds**. **Decision:** rejected; `trend_views` stays `same` and the option is removed.

### I.7 Training degrades phase structure — an observation, not an experiment

Across **every** architecture we tried, self-supervised training made acrophase recovery *worse*
than the same architecture at random initialisation:

| Architecture | Untrained acrophase R² | Trained acrophase R² | Change |
|---|---:|---:|---:|
| Shared encoder (canonical) | 0.680 | 0.515 | **−0.165 ↓** |
| Pre-encoder decomposed | 0.621 | 0.416 | **−0.205 ↓** |

**Interpretation, and the most likely root cause of the whole negative-RQ3 picture.** There is a
fundamental mismatch between the objective and the evaluation. A **contrastive** objective trains
the representation to be *invariant* to the difference between two augmented views. But RQ1 and RQ2
ask the representation to *track* amplitude and phase — that is, to be **equivariant**, not
invariant. We are training for one property and measuring another. The two augmentations that touch
rhythm structure (smoothing, and CoST's scaling which we disabled) make this concrete.

This is the single most important open direction, and it is why we did not simply tune the existing
objective further. Our one attempt at an explicit equivariance term (I.5) was a crude version of
this idea and failed; a seasonal-equivariance objective was implemented but **never evaluated**, and
is preserved unevaluated in `archive/failed_experiments/dssl_v2_decomposition/equivariance.py`
precisely because deleting an untested idea would destroy work without evidence either way.

---

## J. Limitations

### J.1 Trend branch saturation is unfixed
The MoCo top-1 accuracy on the trend branch reaches **1.000**: the contrastive task on that branch
becomes trivially solvable, so it stops producing a useful gradient. Two principled fixes were
attempted (I.1, I.6) and both failed. We report this as an open defect, not as something resolved.

### J.2 The disentanglement claim is only partly earned
"Disentangled" is in the model's name, but by our own pre-registered criteria RQ1-D is **not
supported on HRD**. MESOR is cleanly separated; amplitude is separated but less than the untrained
control achieves; acrophase is not separated at all. The leakage is geometric and present at random
initialisation (E.3). The name describes the design intent, not a verified property.

### J.3 The HRD endpoint is largely trait-like
80.4% of endpoint labels match baseline status, only 22 participants switch, and baseline CES-D
alone achieves AUROC 0.876. RQ3 as written asks for total prediction from wearable data alone. This
may not be an answerable question on this dataset. See G.5.

### J.4 Within-person normalisation removes absolute levels
Each participant's channels are z-scored using their own statistics. This makes windows comparable
across people but **erases between-person level differences** — so the MESOR the model recovers is
a *person-relative* level, not a physiological one. A separate gate confirmed this choice is the
right one for downstream signal (person-relative input scored 0.6791 against 0.6656 and 0.6526 for
physical and cohort-standardised alternatives), but it does bound what "MESOR" means in every RQ1
result above.

### J.5 Statistical power
n = 113 (HRD) and 142 (GLOBEM); ~22 participants per test fold. Per-fold AUROC swings 0.51–0.87 for
every method. Bootstrap intervals are conditional on the fitted models and **exclude** seed and fold
variability; seed SD is reported separately.

### J.6 GLOBEM is a weak circadian test
At 4 samples per day the 24-hour rhythm is barely resolvable and sub-daily harmonics are not
resolvable at all. GLOBEM RQ2 is marked not applicable by protocol. GLOBEM's failure to replicate
HRD's RQ1(a) should be read with this in mind — it is a genuine failure of generalisation, but the
cohort is not an equally strong test.

### J.7 Multiple comparisons
The results tables report many quantities. Intervals are **not simultaneous across rows**. The
pre-registered primary criteria (A.4) are the ones that carry inferential weight; everything else is
descriptive.

### J.8 The canonical run's raw artefacts are on the cluster, not in this repository
See M.1. The reported numbers come from `docs/VALIDATION_RESULT.md`, which was written from the
cluster summaries. This is a reproducibility gap and is flagged rather than glossed over.

---

## K. Canonical model and exact identifiers

**Table K — Everything needed to identify the canonical model unambiguously.**

| Item | Value |
|---|---|
| Model class | `DSSL` in [cost.py](cost.py), `method="dssl"` (the default) |
| Encoder class | `CoSTEncoder` in [models/encoder.py](models/encoder.py) |
| Loss functions | [models/losses.py](models/losses.py) |
| **HRD config** | [configs/hrd.json](configs/hrd.json) |
| **GLOBEM config** | [configs/globem.json](configs/globem.json) |
| Canonical run name | **`narval_v2`** (now the default for `--run-name`) |
| Pre-registration commit | `8bc9b6e` (`docs/VALIDATION_PRECOMMIT.md`, committed before the run) |
| Governing protocol | [docs/SCIENTIFIC_PROTOCOL.md](docs/SCIENTIFIC_PROTOCOL.md) |
| Results of record | [docs/VALIDATION_RESULT.md](docs/VALIDATION_RESULT.md) |
| Rejected experiments | [FAILED_EXPERIMENTS.md](FAILED_EXPERIMENTS.md) |
| HRD data cache | `datasets/cache/hrd_rescue_v1.npz` |
| GLOBEM data cache | `datasets/cache/globem_rescue_v1.npz` |
| Split seed | `20260914` |
| Seeds | 1, 2, 3 |
| Folds | 5, participant-disjoint |
| Encoder checkpoints | `results/<dataset>/narval_v2/tcn_none/seed_<S>/fold_<F>/dssl_encoder.pt` |
| Full training state | ″`/dssl_training.pt` |
| Per-fold metrics | ″`/metrics.json` |

**The canonical configuration in full** (identical in both config files except for the cache path;
geometry is derived from the data, not set here):

```json
{
  "backbone": "tcn",           "temporal_encoding": "none",
  "tcn_depth": null,           "n_layers": 4,
  "n_heads": 4,                "bidirectional": true,
  "output_dims": 320,          "hidden_dims": 64,
  "seasonal_frac": 0.5,        "seasonal_bands": "harmonics",
  "harmonics": 4,              "trend_kernel_cap": null,
  "phase_readout": "angle",    "phase_mode": "circular_amp",
  "weights": "contracted",     "alpha": 0.005,
  "moco_k": 4096,              "jitter_sigma": 0.1,
  "shift_sigma": 0.5,          "smooth_minutes": 75.0,
  "lr": 0.0005,                "batch_size": 64
}
```

`tcn_depth: null` and `trend_kernel_cap: null` mean *derive from the window length* — depth becomes
the smallest value whose receptive field covers the window (7 on HRD, 4 on GLOBEM), and the largest
trend kernel becomes the largest power of two ≤ T/8 (64 on HRD, 8 on GLOBEM). This is why one config
file serves two very different geometries.

**There is exactly one canonical configuration per dataset.** There are no `_final`, `_best`,
`_v2` or `_latest` variants anywhere in `configs/`.

---

## L. Exact reproduction commands

Every command below is literal and complete. `$ACCOUNT` is your Compute Canada allocation.

### L.1 Environment

```bash
bash slurm/setup_env.sh                 # creates the venv and installs pinned dependencies
source slurm/env.sh                     # activates it and exports the module environment
```

### L.2 Build the data caches (once)

```bash
python scripts/build_cache.py --dataset hrd
python scripts/build_cache.py --dataset globem
```
Produces `datasets/cache/hrd_rescue_v1.npz` and `datasets/cache/globem_rescue_v1.npz`.

### L.3 Verify the installation before spending GPU time

```bash
python -m unittest discover -s tests -t .              # 30 tests, ~5 min on CPU
```
This checks the architecture against the written specification (parameter count 7,451,984 for 3
channels, receptive field 1021, bands `[[1,10],[10,17],[17,24],[24,31]]`, trend kernels
`[1,2,4,8,16,32,64]`), the loss constants, the readout geometry, the preprocessing invariants, and
bit-exact training resumption. **Expected: `Ran 30 tests ... OK`.**

```bash
python scripts/run_experiment.py --dataset hrd --smoke --device cpu --seed 1 --fold 0
```
A single tiny fold end-to-end on CPU. Writes to `results/hrd/narval_v2_smoke_cpu/`, never to the
canonical namespace. Use this to confirm the full pipeline runs before submitting jobs.

### L.4 Reproduce the canonical run (the numbers in Sections E, F and G)

**Whole matrix, both cohorts, 3 seeds × 5 folds, on the cluster:**

```bash
bash slurm/submit.sh $ACCOUNT narval_v2
```

This submits, per dataset, in dependency order: the CoST reference array → the DSSL variant array
(starts only when every reference task succeeded) → a CPU summarisation job (starts only when every
variant task succeeded). Default resources are a `gpu:a100_3g.20gb` slice and a 3-hour limit per
task. To vary them:

```bash
TIME=12:00:00 bash slurm/submit.sh $ACCOUNT narval_v2       # longer limit per task
GPU=gpu:a100:1 bash slurm/submit.sh $ACCOUNT narval_v2      # whole A100s instead of slices
```

**A single seed/fold, e.g. for debugging one cell of the matrix:**

```bash
python scripts/run_experiment.py --dataset hrd    --seed 1 --fold 0 --device cuda --run-name narval_v2
python scripts/run_experiment.py --dataset globem --seed 1 --fold 0 --device cuda --run-name narval_v2
```
Repeat for `--seed` in {1,2,3} and `--fold` in {0,1,2,3,4} — 15 runs per dataset.

### L.5 Regenerate the results tables from completed runs

```bash
python scripts/run_experiment.py --dataset hrd    --summarize --run-name narval_v2
python scripts/run_experiment.py --dataset globem --summarize --run-name narval_v2
```

This aggregates the 15 per-fold `metrics.json` files and writes, per dataset, into
`results/<dataset>/narval_v2/tcn_none/`:

| File | Contents |
|---|---|
| `SUMMARY.md` | **The human-readable results document — every number in Sections E, F and G** |
| `RQ1_table.csv` | Per-channel rhythm recovery |
| `rq1_disentanglement_channels.csv` | Per-channel own-vs-leakage R² |
| `rq2_associations.csv` | Per-participant RQ2 concordance contributions |
| `RQ3_table.csv` | The full RQ3 baseline ladder |
| `RQ3_forest_table.csv` | Secondary random-forest ladder |
| `oof_predictions.csv` | Out-of-fold participant probabilities |
| `rq1_recovery.png`, `rq1_families.png`, `rq2_associations.png`, `rq3_auroc.png` | Figures |

### L.6 Reproduce the specific claims in this report

| Claim | Where to find it | Command |
|---|---|---|
| Architecture: 1021 receptive field, 7.45M params, the 4 bands | Section C | `python -m unittest tests.test_paper_model -v` |
| RQ1 / RQ1-D results (Section E) | `results/hrd/narval_v2/tcn_none/SUMMARY.md`, "RQ1" section | L.5 above |
| RQ2 results (Section F) | same file, "RQ2" section | L.5 above |
| RQ3 results (Section G) | same file, "RQ3" section, plus `RQ3_table.csv` | L.5 above |
| Model-selection comparison (Table H.2) | `results/hrd/{narval_v1,narval_v2_split,narval_v2_timestep,narval_v3_decomposed}/tcn_none/SUMMARY.md` | Present in the repository (see M.1) |
| Rejected experiments (Section I) | `FAILED_EXPERIMENTS.md` | — |

### L.7 Timing

```bash
python scripts/time_steps.py --dataset hrd     # measures per-update time on the current device
```
Reference: HRD at 6000 iterations on a `3g.20gb` A100 slice fits comfortably inside the 3-hour
default limit; the CoST reference measured 8.6 s/update in full float32, which is why training uses
TF32 convolutions (D.5).

---

## M. Outstanding actions

These are the things that are *not* done, listed so that nothing is silently missing.

### M.1 The canonical run's raw artefacts are not in this working copy
`results/hrd/narval_v2/` and `results/globem/narval_v2/` exist **on the cluster only**. Present
locally are `narval_v1`, `narval_v2_split`, `narval_v2_timestep` and `narval_v3_decomposed` — the
comparison arms used for Table H.2.

Every canonical number in Sections E, F and G is quoted from `docs/VALIDATION_RESULT.md`, which is
committed to git and was written from the cluster summaries. That document is the record; the raw
artefacts should nevertheless be downloaded so the tables can be regenerated locally:

```bash
scp -r $CLUSTER:~/CoST/results/hrd/narval_v2    results/hrd/
scp -r $CLUSTER:~/CoST/results/globem/narval_v2 results/globem/
```

### M.2 Local `results/` holds 1.8 GB of `representations.npz` — deliberately retained
An earlier version of the cleanup plan listed all 60 of these files as safely deletable. Tracing the
code before deleting showed that is **wrong**: `scripts/run_experiment.py:207` reads
`representations.npz` back whenever a variant run reuses a **cached CoST reference** stage, and the
guard on that path checks only for `complete.json` — so a missing `.npz` beside an existing
`complete.json` raises `FileNotFoundError` instead of gracefully retraining. **30 of the 60 files
are read back this way.** They were therefore kept, and the manifest was corrected. No number in
this document is read from them, but the reuse path is.

One genuinely regenerable smoke directory remains and can be removed:

```bash
rm -rf results/hrd/a2check_smoke_cpu
```

### M.3 Open scientific directions, in priority order
1. **An equivariant objective.** Section I.7 identifies the invariance/equivariance mismatch as the
   most likely root cause of the negative results. The one implementation of a seasonal-equivariance
   objective was never evaluated and is preserved in
   `archive/failed_experiments/dssl_v2_decomposition/equivariance.py`.
2. **Fix trend-branch saturation** (J.1) — two approaches have already failed, so a third should
   differ in kind, not degree.
3. **An incremental-over-baseline downstream analysis** — explicitly *future work*, and explicitly
   **not** a substitute for RQ3 as pre-registered (G.5).

---

## Summary of findings

| Research question | Pre-registered criterion | Result | Verdict |
|---|---|---|---|
| **RQ1** rhythm preservation | Beat population mean **and** untrained, every family | Beats population mean on HRD; loses to untrained on phase and IS; population mean not met on GLOBEM | **Partially supported** |
| **RQ1-D** disentanglement | Own > leakage on all three **and** better than untrained | MESOR separated cleanly; amplitude separated but below untrained; acrophase not separated | **Not supported on HRD** (clause (i) holds on GLOBEM, (ii) does not) |
| **RQ2** rhythmic phenotyping | Timing **and** strength, both above chance and above untrained | Timing 0.759 (+0.049 vs. untrained); strength 0.713 (+0.054 vs. untrained) | **FULLY SUPPORTED on HRD** |
| **RQ3** clinical utility | Above raw **and** above untrained | +0.009 vs. raw, +0.028 vs. untrained — both intervals include 0 | **Not supported on either cohort** |

**The honest one-paragraph conclusion.** DSSL does the specific thing it was designed to do: given a
person's own recent history, its frozen representation reliably detects both when their 24-hour
rhythm has shifted in time and how much its strength has changed, better than an untrained network
of identical architecture. It does not improve depression prediction over raw windows or an
untrained encoder, and its branch separation is real for level but not for rhythm timing. The most
likely reason, supported by the observation that training *degrades* phase recovery in every
architecture we tried, is that a contrastive objective trains for invariance while our research
questions measure equivariance. Fixing that mismatch is the clearest next step, and it is a change
of objective, not of architecture.
