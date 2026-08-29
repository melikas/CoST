# Methodology — Self-Supervised Rhythm Representations for Depression Endpoint Prediction

**Status: protocol specification, derived from the codebase only. No results are reported.**

This document describes what the code *does*, not what any run produced. Every numeric claim
cites `file:LINE`. Where a parameter is not fixed in code or config it is written
`[UNSPECIFIED — needs confirmation]` rather than inferred from convention. Where the code
contradicts a project document, both are stated and the code is treated as ground truth.

Two datasets are covered: **HRD** (minute-level Fitbit + phone, the primary cohort) and
**GLOBEM** (segment-level RAPIDS features, four annual college cohorts). They share one
training and evaluation implementation — `train_globem.py` is a thin argv wrapper that calls
`train_hrd.main()` ([train_globem.py:20-35](../train_globem.py#L20-L35)) — so everything in
§4.3–§4.4 applies to both unless stated otherwise.

---

## 1. Notation

| Symbol | Meaning | HRD | GLOBEM |
|---|---|---|---|
| $p$ | participant identifier | — | — |
| $N$ | number of windows in the dataset | — | — |
| $T$ | timesteps (bins) per window | $672$ | $112$ (default) / $28$ (as configured via `DATASET=globem`) |
| $C$ | input channels | $4$ | $12$ |
| $C_s$ | sensor channels (`n_sensors`) | $4$ | $12$ |
| $C_\pi$ | appended temporal channels | $0$, $7$ (clock) or $2$ (calendar index) | $0$ or $7$ |
| $\Delta$ | bin width | 15 min | 6 h (one day-segment) |
| $B$ | pretraining batch size | $64$ | $64$ |
| $C_h$ | encoder hidden width (`hidden_dims`) | $64$ (TCN) / $48$ (Transformer) | same |
| $C_o$ | representation width (`repr_dims`) | $320$ (TCN) / $240$ (Transformer) | same |
| $d$ | component width, $C_o/2$ | $160$ / $120$ | same |
| $L$ | encoder depth | $10$ (TCN) / $4$ (Transformer) | same |
| $x_{p,t}\in\mathbb{R}^{T\times C}$ | one window of participant $p$ ending at $t$ | — | — |
| $f_\theta$ | SSL encoder, frozen after pretraining | — | — |
| $z^{(T)},z^{(S)}\in\mathbb{R}^{T\times d}$ | trend / seasonal branch sequences | — | — |
| $V^{(T)},V^{(S)}$ | pooled branch vectors | — | — |
| $V^{(F)}=[V^{(T)};V^{(S)}]$ | full window representation | — | — |
| $\tau,\sigma\in\mathbb{R}^{T\times C_s}$ | closed-form polynomial-trend / harmonic reference | — | — |
| $\pi$ | temporal reference frame (positional encoding) | — | — |
| $M,A,\phi$ | cosinor MESOR, amplitude, acrophase | — | — |
| $\mathrm{IS}$ | interdaily stability | — | — |
| $\rho(V\!\to\!u)$ | held-out ridge $R^2$ of a linear probe $V\mapsto u$ | — | — |
| $s_{p,t}$ | within-person robust-$z$ deviation score | — | — |
| $\delta^{*}$ | minimum detectable perturbation | — | — |
| $c^{*}$ | breakdown point of a degradation factor | — | — |
| $\lambda$ | ridge penalty; $\lambda^{*}$ its validation-selected value | — | — |
| $\mathcal{P}_{\text{tr}},\mathcal{P}_{\text{va}},\mathcal{P}_{\text{te}}$ | participant-disjoint splits | — | — |

$T = 168\,\text{h}\times 60/15 = 672$ from
[data_preprocessing.py:173-174](../data_processing/data_preprocessing.py#L173-L174).
$C_o$ and $L$ from [train_hrd.py:489-491](../train_hrd.py#L489-L491) and the per-backbone
override in [scripts/run.sh](../scripts/run.sh) (`HID=64 REPR=320 DEPTH=10` for TCN,
`48/240/4` for Transformer).

---

## 2. Study design & data provenance

### 2.1 HRD

**Source table.** A single CSV, `datasets/HRD_RAW_MinuteLevel.csv`, holding minute-level
sensor rows with the participant-level depression labels repeated on every row
([data_preprocessing.py:1-22](../data_processing/data_preprocessing.py#L1-L22)). The file is
excluded from version control.

**Channels used (4).** `HR`, `Steps`, `is_asleep`, `screen`
([data_preprocessing.py:65-77](../data_processing/data_preprocessing.py#L65-L77)).
`is_asleep` is *derived*, not read: sleep stages `{asleep, light, deep, rem}` → 1,
`{awake, wake, restless}` → 0, and anything else is resolved by wear — HR present means the
watch was worn but nothing was scored → 0; HR also missing → NaN
([data_preprocessing.py:278-296](../data_processing/data_preprocessing.py#L278-L296)).
Deliberately excluded: `floors`, `sedentary_minutes`, `call`, and the Fitbit
`fairly/lightly/very_active_minutes` intensity levels
([data_preprocessing.py:69-72](../data_processing/data_preprocessing.py#L69-L72)).

**Labels.** `depression_status_endpoint` is the default target
([train_hrd.py:353](../train_hrd.py#L353)); `depression_status_baseline` is carried alongside,
as are `depression_trajectory` (the four Case-1 groups Pre1_Post1 … Pre2_Post2) and
`ces_d_baseline_score` / `ces_d_endpoint_score`
([data_preprocessing.py:95-105](../data_processing/data_preprocessing.py#L95-L105)).
A daily self-reported `emotional_energy` (1–5) supports a secondary task
([data_preprocessing.py:107-110](../data_processing/data_preprocessing.py#L107-L110)).

> **The following are not recoverable from the repository and must be supplied by the study
> team before this section can be considered complete:** device manufacturer and firmware;
> native sampling rate and how the vendor derived `sleep_status`; recruitment setting, period
> and geography; inclusion/exclusion criteria; the CES-D instrument version and the **cutoff
> that binarises `depression_status_*`**; the baseline→endpoint interval; comorbidity,
> medication and shift-work status; ethics approval and consent procedure; and the
> demographic composition of the cohort. All are `[UNSPECIFIED — needs confirmation]`.
> Without the CES-D cutoff in particular, the label's clinical meaning — screening positive
> versus diagnosed disorder — is undetermined, and every AUROC in this study inherits that
> ambiguity.

**Exclusions, in pipeline order.**

| Stage | Rule | Source |
|---|---|---|
| Row | unparseable timestamp dropped | [data_preprocessing.py:271-272](../data_processing/data_preprocessing.py#L271-L272) |
| Sample | HR outside $[20,250]$ bpm → NaN | [:136](../data_processing/data_preprocessing.py#L136) |
| Sample | step counts clipped to $\ge 0$ | [:306](../data_processing/data_preprocessing.py#L306) |
| Participant | wear-channel missingness $>30\%$ (mean over HR/Steps/is_asleep) → dropped | [:140](../data_processing/data_preprocessing.py#L140), [:327-335](../data_processing/data_preprocessing.py#L327-L335) |
| Window | any channel missing $>30\%$ of its bins → dropped | [:150](../data_processing/data_preprocessing.py#L150), [:593-606](../data_processing/data_preprocessing.py#L593-L606) |
| Window | any channel with leading/trailing gap $>30$ min → dropped | [:162](../data_processing/data_preprocessing.py#L162), [:575-590](../data_processing/data_preprocessing.py#L575-L590) |
| Window | $\le 10$ raw samples → dropped | [:166](../data_processing/data_preprocessing.py#L166) |

The window gate is Case 1 of Algorithm 1 in Yan et al. 2022 (ACM TIST 13(3), Art. 47,
§3.4.1/4.1.1), with one deliberate divergence: because the tensor needs a fixed channel count,
an unusable *feature* disqualifies the whole *window*, rather than the feature being dropped
per the paper ([data_preprocessing.py:593-606](../data_processing/data_preprocessing.py#L593-L606)).
The edge-gap rule replaced an earlier "first and last bin must be observed" rule that
discarded 14.0% of candidate windows and fell harder on the depressed group (15.0% vs 13.0%)
([data_preprocessing.py:152-161](../data_processing/data_preprocessing.py#L152-L161)) — a
differential-exclusion bias that was measured and removed. That measurement should appear in
the manuscript; it is the kind of thing reviewers ask for and few studies have.

**Unit of analysis.** Non-overlapping 7-day windows; the label is participant-level and
constant across a participant's windows. See §4.5 for why this makes the probe unit a
first-order design decision rather than a detail.

### 2.2 GLOBEM

**Source table.** `datasets/GLOBEM_REDUCED.csv` — RAPIDS behavioural aggregates from Fitbit
and phone, at 4 fixed day-segments (night/morning/afternoon/evening), with `LABEL_ENDPOINT`
per participant and a time-varying `LABEL_WEEKLY`
([globem_preprocessing.py:1-31](../data_processing/globem_preprocessing.py#L1-L31)).

**Channels used (12):** Fitbit steps (3), Fitbit sleep (3), phone screen (2), phone location
(4) ([globem_preprocessing.py:48-63](../data_processing/globem_preprocessing.py#L48-L63)).
Bluetooth and wifi are excluded as device-proximity counts rather than participant rhythms
([globem_preprocessing.py:44-47](../data_processing/globem_preprocessing.py#L44-L47)).

**Cohort structure.** Four annual cohorts (DS1–DS4, 155/218/137/195 = 705 participants in the
published release). The redistributed file renumbers pids globally, so cohort identity is
recovered from the calendar year of a participant's first record
([globem_preprocessing.py:358-362](../data_processing/globem_preprocessing.py#L358-L362)).
This is a genuine strength for external validity — GLOBEM supports a *leave-one-cohort-out*
evaluation that HRD cannot — but see §3.4: **that split is not implemented.**

**Sparsity.** Sleep features are ~33% present
([globem_preprocessing.py:24-31](../data_processing/globem_preprocessing.py#L24-L31)). A window
is kept only if at least `min_window_coverage = 0.5` of its timesteps carried any observation
([globem_preprocessing.py:265](../data_processing/globem_preprocessing.py#L265)).

**Unit of analysis.** Sliding windows, stride 7 days
([train_globem.py:22-23](../train_globem.py#L22-L23)), so consecutive windows of one
participant overlap by `window_days − 7` days. With the default 28-day window that is 75%
overlap; with the GLOBEM sweep's 7-day stride
([scripts/run.sh](../scripts/run.sh), `DATASET=globem`) windows are
non-overlapping. The label is *weekly* by default
([train_hrd.py:365](../train_hrd.py#L365)) — i.e. window-level, not participant-level, which
changes the correct probe unit (§4.5).

### 2.3 What limits generalization to a clinical population

Stated here so it is not buried in §8: the HRD test cohort is **balanced by construction**
(`--test-per-class 18`, [train_hrd.py:447](../train_hrd.py#L447)), which no deployment
population is. The code handles this correctly — sensitivity/specificity and AUROC are
prevalence-invariant and are transported to a target base rate by Bayes' rule
([_eval_protocols.py:205-238](../tasks/_eval_protocols.py#L205-L238)) — but every
threshold-dependent number (accuracy, F1, MCC, PPV) must be read at the transported
prevalence, never at the implied 50%.

---

## 3. Experiment inventory

### 3.1 Table

| # | Experiment | Data / input | Key settings | Hypothesis tested | Why necessary (claim that collapses without it) | Success criterion | Source |
|---|---|---|---|---|---|---|---|
| **E0** | Depression endpoint probe | frozen $V^{(F)}$, held-out participants | logistic probe, 5-fold CV in pool, participant bootstrap | The representation carries endpoint-relevant information | The entire utility claim | AUROC CI excludes 0.5 | [train_hrd.py:867-905](../train_hrd.py#L867-L905) |
| **E1.2** | Decomposition-recovery (DRS) | $V^{(F)},V^{(T)},V^{(S)}\to\tau,\sigma$ | ridge, $\lambda^*$ on participant-disjoint val | The latent linearly preserves polynomial-trend and harmonic subspaces | *Diagnostic only* (see §3.3) | reported vs. floor + random-init | [decomposition.py:302](../tasks/decomposition.py#L302) |
| **E1.3** | Chronobiology axis probe **(RQ1 headline)** | $V^{(F)}\to (A,\phi,\mathrm{IS})$ per channel | grouped 5-fold OOF ridge, $\lambda^*$ per fold | Established chronobiological constructs are linearly decodable from the latent | The claim "SSL encodes circadian biology" | $\phi$ MAE in hours; `gain_over_raw` $>0$ **and** `median_gain_over_random_init` $>0$ | [rhythm.py:1283](../tasks/rhythm.py#L1283) |
| **E1.4** | Temporal-frame contrast $\Delta_\pi$ | RQ1 headline per PE variant | paired seed×participant bootstrap, Holm | The temporal reference frame changes what is encoded | The second half of RQ1 | $\Delta_\pi$ CI excludes 0 after Holm | [collect_results.py:745](../scripts/collect_results.py#L745) |
| **E1.5** | Negative controls | random-init; per-window time permutation ×3; permutation-invariant floor; plain-SSL twin | frozen, inference-only except plain-SSL | The scores reflect *learned* structure, not architecture or trivially preserved statistics | **Every** RQ1 number's interpretation | trained $\gg$ random-init; shuffled $\approx$ floor | [experiment_q1.py:186-230](../experiment_q1.py#L186-L230) |
| **E2** | Rhythmic-deviation detection **(RQ2 headline)** | frozen $V$, held-out participants; causal personal baseline | $R=4$ reference windows, frozen; phase shift $\{0.5,1,2,3,4\}$ h; strata $=$ (participant, level) | The personalised distance rises when the window's *rhythmic* deviation rises, not merely when the input changes | The personalised-monitoring claim | $C$ CI excludes 0.5, and $\Delta$ vs Random-init CI excludes 0 | [experiment_q2.py](../experiment_q2.py) |
| **E3.A** | Baseline ladder | majority → handcrafted → cosinor → random-init → plain-SSL → **CoST** → supervised | paired participant bootstrap, $B=2000$ | The frozen SSL representation beats simpler and untrained alternatives | The utility claim's *comparative* content | $\Delta$AUC CI excludes 0 | [experiment_q3.py:161-236](../experiment_q3.py#L161-L236) |
| **E3.B** | Ablations | branch / channel-zeroing / circadian removal | probe fit once on intact input | *Which* component carries the signal | The mechanistic "how" | consistent with the whole | [experiment_q3.py:238-251](../experiment_q3.py#L238-L251) |
| **E3.C** | Degradation grid | duration, granularity, MCAR, block-missing, channels | inference-only; $c^{*}$ bootstrapped | Operating envelope | The deployment claim | $c^{*}$ defined per factor | [experiment_q3.py:253-306](../experiment_q3.py#L253-L306) |
| **E4** | Separability table | 15 representation views | one shared train/val/test split | Which *view* of the representation is most discriminative | Cross-view comparability (§5) | relative ordering, not absolute | [rhythm.py:680](../tasks/rhythm.py#L680) |
| **E5** | Emotional-energy probe | sliding trailing windows | own participant split | Generalization to a day-resolution target | Second downstream task | AUROC CI excludes 0.5 | [tasks/energy.py:124](../tasks/energy.py#L124) |

### 3.2 Per-experiment objective and the negative outcome

**E0.** Objective: establish that the frozen representation supports participant-level
endpoint classification. *If it came out opposite* — AUROC CI spanning 0.5 — the honest
reading is that 7-day passive-sensing windows at $n\approx36$ held-out participants do not
resolve this endpoint, not that the representation is empty; E1.3 and E5 would then carry the
paper, and E3.C becomes undefined by construction (§3.5).

**E1.2 (DRS).** Objective: measure linear preservation of the trend and harmonic subspaces.
*If it came out opposite* — low $R^2$ — it would mean the encoder discards a fixed linear
projection of its own input, which is a statement about compression, not about biology. This
is precisely why it is demoted (§3.3).

**E1.3 (headline).** Objective: show that amplitude, acrophase and interdaily stability —
nonlinear functionals that are the standard chronobiological constructs — are linearly
readable from the frozen latent, and better than from a dimensionality-matched PCA of the raw
window. *If it came out opposite*, the "learns circadian structure" framing is unsupported and
the paper must be rewritten as a negative/limits result.

**E1.4.** Objective: test whether the temporal reference frame $\pi$ matters. *If it came out
opposite* — $\Delta_\pi\approx 0$ — the positional-encoding contribution is null and the PE
sweep becomes a robustness check rather than a finding. **Note this is currently unanswerable:
see §3.4.**

**E1.5.** Objective: calibrate everything above. *If the controls match or beat the trained
encoder*, no RQ1 claim survives; the correct output is a negative result with these controls,
which is publishable and considerably more useful than a marginal positive.

**E2.** Objective: show that an unlabelled personal baseline detects a within-person *rhythmic* deviation — specifically that $\Delta d$ ranks windows whose raw 24-h cosinor deviation the perturbation **increased** above those whose it **decreased**. *If it came out opposite* — $C$ spanning 0.5, or Random-init matching CoST — the personalised distance is reading input-space change rather than rhythm, which is a legitimate and reportable finding.

**This experiment was redesigned; the previous version is retracted.** It labelled every perturbed window positive, which is false whenever the perturbation moves a window *toward* its own baseline — measured at up to 78 % of windows in one cell (`DSSL V^S amp`, $\alpha{=}0.9$; `paired_win_rate` 0.220). Under that labelling a **pure change-detector** $\lVert x'-x\rVert$ scores a perfect **1.000**, above cosinor; under $C$ it scores **0.501** and a phase-blind amplitude reader scores **0.509**. Verification detail in §3.7.

**E3.** Objective: comparative utility, mechanism, and operating envelope. *If the ladder is
flat*, the SSL pretraining is not contributing and the plain-SSL twin (which isolates the
disentangler alone) tells you whether the architectural claim or the pretraining claim failed.

**E4.** Objective: compare 15 views under identical probe settings. *If the seasonal views lose
to the trend views*, the trend/seasonal split is not doing what the architecture assumes.

**E5.** Objective: a second, day-resolution target on the same frozen encoder. *If it came out
opposite*, the representation is endpoint-specific rather than a general behavioural summary.

### 3.3 Decision — the RQ1 headline is **E1.3**, not E1.2

The two project documents disagree: `docs/RQ_Minimal_Experiment_Design.md` §E1.2 declares
Full→$\sigma$ the headline, while
[experiment_q1.py:11-16](../experiment_q1.py#L11-L16) declares E1.3 the headline and E1.2
"not, because tau/sigma are a FIXED LINEAR projection of the input." **I adopt E1.3.** The
reasoning, which should go into Methods verbatim:

$\tau$ and $\sigma$ are produced by one design matrix $M=[M_\tau\,|\,M_\sigma]$ that is
*identical for every window*, via $\hat\beta = M^{+}x$, $\tau=M_\tau\hat\beta_{1:4}$,
$\sigma=M_\sigma\hat\beta_{5:10}$
([decomposition.py:66-88](../tasks/decomposition.py#L66-L88)). Therefore
$\tau = (M_\tau M^{+})\,x$ and $\sigma = (M_\sigma M^{+})\,x$ are **fixed linear functionals
of the input**. Asking a *linear* ridge probe to recover them from $f_\theta(x)$ tests whether
$f_\theta$ preserves a particular 10-dimensional linear subspace of its input — a property any
sufficiently wide, near-isometric map has, trained or not. It cannot distinguish "encodes
circadian physiology" from "is approximately invertible." This is a structural property of the
target, not a small-sample artifact, and no amount of extra data fixes it.

By contrast, amplitude $A_c=\sqrt{a_c^2+b_c^2}$, acrophase $\phi_c=\operatorname{atan2}(b_c,a_c)$
and interdaily stability
$\mathrm{IS}=n_d\,\mathrm{SS}_{\text{between-bin}}/\mathrm{SS}_{\text{total}}$
([rhythm.py:923-943](../tasks/rhythm.py#L923-L943)) are **nonlinear** functionals, are the
constructs the chronobiology literature actually uses, and — critically — acrophase error is
reportable **in hours**, giving RQ1 one unit-bearing number. E1.3 also ships with the correct
within-experiment control: `gain_over_raw`, the same probe run on a PCA of the raw window at
the latent's own width ([rhythm.py:1344-1347](../tasks/rhythm.py#L1344-L1347)).

**Two changes are required before E1.3 can carry headline weight**, and they are not optional:

1. ~~E1.3 has no random-init control.~~ **Resolved during preparation of this document.**
   `rhythm_axis_probe` is now also run on a random-init encoder over the same markers, the
   same participant-grouped folds and the same cached cosinor fit, so the control costs one
   extra encode and no cosinor time ([experiment_q1.py:328-344](../experiment_q1.py#L328-L344)).
   Critically, the comparison is **paired per marker** —
   `median_gain_over_random_init` = median over markers of
   $R^2_{\text{trained}} - R^2_{\text{random-init}}$
   ([experiment_q1.py:348-360](../experiment_q1.py#L348-L360)) — so it answers *what
   pretraining added*, not merely *is the latent better than raw*. **Report
   `median_gain_over_random_init` and `frac_markers_beating_random_init` beside the acrophase
   error; the headline is not interpretable without them.** It still lacks a confidence
   interval (§7.3).
2. **E1.3 and E1.2 use different held-out protocols and this must be stated.** E1.2 fits on
   train participants and scores on test ([decomposition.py:144-152](../tasks/decomposition.py#L144-L152));
   E1.3 fits *out-of-fold within the test participants*, grouped by pid, 5 folds
   ([rhythm.py:1283-1284](../tasks/rhythm.py#L1283-L1284), [rhythm.py:1224](../tasks/rhythm.py#L1224)).
   Both are defensible; using the word "held-out" for both without qualification is not.

E1.2 remains in the paper as a **diagnostic of linear information preservation**, reported
beside the permutation-invariant floor (§4.6) and the random-init control — never as evidence
about biology.

### 3.4 E1.4 is answerable: the sweep carries one wall-clock arm

`scripts/run.sh` runs `VARIANTS=(tcn:none tcn:circular)` over `SEEDS=(42 82 22)`
([scripts/run.sh:100-103](../scripts/run.sh#L100-L103), [:62](../scripts/run.sh#L62)).
`pe_contrast` takes `ref=("tcn","none")` ([collect_results.py:745](../scripts/collect_results.py#L745)),
so `tcn:circular` is the contrast arm and $\Delta_\pi$ is defined.

The pair is the right one, and the isolation is exact. `circular` supplies **wall-clock
phase**, which is what the reference-frame question is about, and the two variants differ by
**320 parameters** out of 44.1 M — the `Linear(4, 64)` of `CircularCalendarPE` and its bias.
`input_fc` is `(64, 4)` in both, since the two calendar-index channels bypass it entirely
([models/encoder.py:194](../models/encoder.py#L194), [:203-206](../models/encoder.py#L203-L206)).
No performance difference can be attributed to capacity.

`CLOCK_FLAG=""` is **required**, not a limitation: `--pe circular` supplies its own calendar
encoding and `train_hrd.py` aborts if `--with-clock-features` is also passed
([train_hrd.py:627-630](../train_hrd.py#L627-L630)).

Both arms are in `PLAIN_REF` ([_experiment_common.py:31-32](../tasks/_experiment_common.py#L31-L32))
and in `Q23_VARIANTS` ([run.sh:410](../scripts/run.sh#L410)), so each pretrains its own plain
twin and each runs RQ2 and RQ3 in full. A variant outside `PLAIN_REF` would get a ladder one
rung shorter, making its $\Delta$AUC column non-comparable.

**Scope limit to state, not a defect.** With two variants, $\Delta_\pi$ has exactly one
contrast arm. That is a valid comparison but not a family, so the Holm correction in
`pe_contrast` is a no-op and the reference-frame claim rests on a single pair. The design
document's F0–F4 families are not instantiated. Adding `transformer:factorized` or
`tcn:factorized` would turn the pair into a family and separate the *anchor* from the *basis*;
both are implemented ([positional_encoding.py:33](../models/positional_encoding.py#L33)).

### 3.5 Two guards that make experiments legitimately return "undefined"

Both are correct behaviour and must be reported, not worked around:

- `breakdown_valid = (auc_full >= 0.60)` ([experiment_q3.py:295](../experiment_q3.py#L295)).
  At chance level every degradation trivially sits within 0.05 of the full AUC, which would
  read as perfect robustness. If E0 lands near chance, **all of E3.C is void** and the
  degradation analysis must be re-anchored to a target the representation demonstrably
  recovers (E1.3 or E5). This is a design decision to take *before* running, not after.
- `rhythm_axis_probe` returns `{}` below 3 participant groups or 20 windows
  ([rhythm.py:1318-1320](../tasks/rhythm.py#L1318-L1320)), surfaced as
  `EMPTY_TOO_FEW_PARTICIPANTS` rather than an empty dict
  ([experiment_q1.py:339-347](../experiment_q1.py#L339-L347)) — "could not be estimated" is
  distinguished from "estimated and found nothing."

### 3.6 GLOBEM-specific inventory defects (code as ground truth)

Three items in the GLOBEM path are inconsistent with the code that would run them:

1. **`--holdout` does not exist.** `scripts/run.sh:496` builds
   `HOLD_FLAG="--holdout $HOLD"` with `HOLDOUTS=(DS1 DS2 DS3 DS4)`
   ([run.sh (DATASET=globem):309](../scripts/run.sh#L309)) and passes it to `train_globem.py`,
   but `parse_args()` defines no such argument
   ([train_hrd.py:350-604](../train_hrd.py#L350-L604)). Argparse will abort every task. The
   leave-one-cohort-out evaluation — GLOBEM's single most valuable design property, and the
   thing `pid_ds` was built for ([globem_preprocessing.py:358-362](../data_processing/globem_preprocessing.py#L358-L362))
   — is **specified but unimplemented**.
2. **`vit:none` is not a valid backbone.** `--backbone` accepts only `{tcn, transformer}`
   ([train_hrd.py:439-440](../train_hrd.py#L439-L440)); `run.sh (DATASET=globem):328-335` lists
   `vit:none`.
3. **`tasks_globem/between_person.py` is dead code** — never imported anywhere in the
   repository. Its `between_person_rhythm` is a GLOBEM analogue of E1.3 and is arguably the
   right analysis for that dataset; either wire it in or delete it.

---

### 3.7 E2 was redesigned after verification — what was tested and what failed

The RQ2 protocol in `experiment_q2.py` is not the one in the design document. It was replaced
after direct measurement, and the measurements belong in Methods.

**The defect in the retracted design.** Layer 1 labelled every perturbed window a positive and
scored `AUC(d_clean, d_pert)` plus `TPR@FPR5`. That label is false whenever the perturbation
moves a window *toward* its own personal baseline. On the real cohort this is not rare: in
`results_hrd/1514691/tcn_circular_seed42`, `paired_win_rate` at `amplitude_damping` $\alpha{=}0.9$
is **0.220** for `DSSL V^S amp` and **0.375** for `DSSL` — the perturbation *reduced* the
distance in 78 % and 62 % of windows. Across all runs, **10 947 of 67 139** grid cells have
`paired_win_rate < 0.5`, concentrated at the mildest level of each perturbation. The response
is U-shaped in magnitude and reproduces across all five seeds, so the design document's stated
success criterion ("monotone increase with magnitude") was **falsified**.

**Why that inflates the old numbers.** For any encoder,
$\lVert f(x')-\mu\rVert^{2}-\lVert f(x)-\mu\rVert^{2}=2\langle f(x)-\mu,\delta\rangle+\lVert\delta\rVert^{2}$,
and the $\lVert\delta\rVert^{2}$ term is unconditionally positive — so $\Pr(\Delta d>0)\to1$
with perturbation magnitude whether or not the representation understands rhythm. Measured on
DSSL: 0.375 at $\alpha{=}0.9$ up to 0.943 at a 12-h shift.

**The verification.** Ground truth in the new design is a function of the raw signal only, so
the whole protocol was validated without a trained encoder, calling the repository's own
`perturb` / `cosinor_params` / `harmonic_reference` / `personal_baseline` / `dscore` on
realistic synthetic 7-day windows (36 participants x 30 windows, 5 channels, 96 bins/day,
within-person amplitude and phase drift, harmonics 2-3, polynomial trend, observation noise).

| check | result |
|---|---|
| Phase shift is exactly a rotation, $z'=z\,e^{i\theta}$ | relative error $5\times10^{-9}$ (float32 machine precision) |
| **Pure change-detector** $\lVert x'-x\rVert$ under $C$ | **0.501 / 0.515 / 0.491** over three seeds; bootstrap CI $[0.463,0.538]$ contains 0.5 |
| **Same detector under the retracted metric** | **1.000** — a perfect score, above cosinor |
| Phase-blind **amplitude reader** $\lvert z\rvert$ | **0.509** (rules out "ties cause the null": it has full variance) |
| Constant response (all ties) | 0.500 |
| Oracle ($\Delta d=\Delta g$) | 1.000 |
| Pure noise | 0.473 |
| Ceilings / random projection | cosinor 0.817, raw 0.832, random projection 0.742 |
| Power at 36 participants, $B{=}400$ | CI half-width $\approx0.02$; cosinor $[0.792,0.839]$ and randproj $[0.713,0.767]$ both exclude 0.5 |

**The amplitude arm was tested and rejected.** Damping's input magnitude is
$(1-\alpha)\lvert z\rvert$ and $\operatorname{sign}(\Delta g)=\operatorname{sign}[2\rho\cos(\psi-\beta)-(1+\alpha)\lvert z\rvert]$
also turns on $\lvert z\rvert$, so the two classes are **not** magnitude-matched: the
within-stratum correlation between input change and class is $-0.51$ to $-0.70$, against
$\approx0$ for phase. A pure change-detector scores **0.075** there instead of 0.5, and a
random linear projection scores **0.87-0.94**, beating both ceilings — the same pathology that
made the retracted Spearman layer unreadable. Sign-randomising the scaling (half damped, half
amplified) lifts the null only to 0.43 at $\alpha{=}0.9$ and 0.11-0.25 elsewhere, so it was
rejected too. Separately, $z'=\alpha z$ is only approximate (relative error $1.5\times10^{-3}$
to $1.5\times10^{-2}$) because `harmonic_reference` fits a degree-3 polynomial trend and three
harmonics *jointly*, leaking between them.

**Consequence for the claim.** RQ2 is now answered for rhythmic **phase** deviation only. That
is narrower than the design document promised and must be stated in Methods, not elided.

**Levels 6, 8 and 12 h were dropped**: 100 % of windows there have $\Delta g>0$, the negative
class is empty and the cell contributes zero pairs. The effective grid is
$\{0.5,1,2,3,4\}$ h, yielding ~12.8k pairs.

**One implementation defect was found and fixed in passing.** The retracted code encoded only
the eligible subset (`reps[name](Xp[elig])`) while the clean arm was encoded on the full array,
so the batch boundaries differed and floating-point reduction order flipped 2-5 % of windows.
This was visible as `paired_win_rate` $=0.488$ at a **zero-magnitude** perturbation, where it
must be exactly 0.500. The rewrite encodes the full array on both arms and slices.

## 4. End-to-end computational pipeline

### 4.1 HRD preprocessing — INPUT → OPERATION → OUTPUT

| # | Operation | Input | Output | Fit on |
|---|---|---|---|---|
| 1 | Read + rename columns | CSV, ~5.4×10⁷ rows | long frame | — |
| 2 | Parse timestamps, drop unparseable | long frame | long frame | — |
| 3 | HR range gate $[20,250]$ → NaN | `(rows,)` | `(rows,)` | none (fixed constant) |
| 4 | Clip counts $\ge0$ | `(rows,)` | `(rows,)` | none |
| 5 | Derive `is_asleep` from `sleep_status` + HR wear | `(rows, 2)` | `(rows,)` | none |
| 6 | Event channel NaN → 0 | `(rows,)` | `(rows,)` | none |
| 7 | Participant missingness gate (30%) | per-pid means | pid subset | **all of that participant's data** |
| 8 | Gap interpolation $\le 30$ min, `limit_area="inside"` | per-pid series | per-pid series | **that participant only** |
| 9 | Per-participant z-score | `(rows, 4)` | `(rows, 4)` | **that participant only** |
| 10 | Bin to $T{=}672$ × 15 min, mean within bin | variable rows | `(672, 4)` + observed mask | — |
| 11 | Algorithm-1 window gate | `(672, 4)` bool | keep/drop | none |
| 12 | Interior linear interp + edge nearest-fill | `(672, 4)` | `(672, 4)`, NaN-free | — |
| 13 | Optional temporal channels | `(672, 4)` | `(672, 4+C_\pi)` | **fixed constants, never fitted** |
| 14 | Stack | list | `X \in \mathbb{R}^{N\times672\times C}` float32 | — |

**Leakage status of each fitted step.** Steps 7–9 are fitted **within a single participant**,
so no statistic crosses the participant boundary and none touches the labels
([data_preprocessing.py:544-556](../data_processing/data_preprocessing.py#L544-L556)). Step 13
uses fixed constants derived from each calendar field's a-priori range
$\mu=(lo+hi)/2$, $\sigma=\sqrt{((hi-lo+1)^2-1)/12}$
([data_preprocessing.py:196-215](../data_processing/data_preprocessing.py#L196-L215)) rather
than an empirical scaler. That is a deliberate correction of two prior defects, both worth
stating in Methods: an empirical scaler was (i) transductive — fitted on pooled windows
including held-out participants (measured $\le 0.03\sigma$), and (ii) inconsistent between the
two entry points, so an encoder pretrained through one path saw clock channels on a different
scale through the other (measured up to $0.16\sigma$).

**Residual concern.** Step 9 uses each participant's **entire record**, including windows that
are chronologically *after* the window being normalized. There is no cross-participant or
label leakage, so E0/E1/E3 are unaffected. But RQ2 is framed prospectively ("could flag a
deviation in practice"), and a look-ahead normalizer is incompatible with that framing. See
§8.1.

### 4.2 GLOBEM preprocessing

| # | Operation | Input | Output | Fit on |
|---|---|---|---|---|
| 1 | Read, map segment→{0,1,2,3}, sort by (pid, date, seg) | CSV | long frame | — |
| 2 | Reindex to a complete (day × 4-segment) grid | per-pid | `(n_days·4, 12)` with NaN | — |
| 3 | Compute $\mu,\sigma$ from **observed values only** | `(n_days·4, 12)` | `(12,)`, `(12,)` | **that participant, observed only** |
| 4 | Interior linear interp + `limit_direction="both"`; fully-absent channel → 0 | `(n_days·4, 12)` | NaN-free | — |
| 5 | Apply z-score | | `(n_days·4, 12)` | (step 3 statistics) |
| 6 | Anchor window start to Monday | — | `start_step` | — |
| 7 | Slide windows, keep coverage $\ge 0.5$ | | `(T, 12)` per window | — |
| 8 | Optional 7 calendar channels, fixed scale | | `(T, 12+7)` | none |

Step 3 before step 4 is the correct order and is called out explicitly
([globem_preprocessing.py:26-31](../data_processing/globem_preprocessing.py#L26-L31)): the
scale is estimated from real observations, so imputed values cannot inflate or shrink it.

**Why imputation at all.** CoST's FFT layer cannot accept NaN, and the encoder masks a
timestep only when *every* channel is NaN
([models/encoder.py](../models/encoder.py)), so a per-channel mask would be inert. Whole-array
imputation is therefore forced by the architecture, not chosen — and that should be stated as a
limitation (§8.1), because with sleep ~33% present, roughly two thirds of the sleep channel is
interpolated.

**Phase-origin anchoring (GLOBEM only).** Every window is forced to start on Monday
([globem_preprocessing.py:104](../data_processing/globem_preprocessing.py#L104),
[:202-233](../data_processing/globem_preprocessing.py#L202-L233)). Without it the Fourier phase
origin differs per participant according to enrolment date, making the absolute phase of every
non-daily frequency bin incomparable across people. The realised weekday distribution is
printed at build time so a violation is visible
([globem_preprocessing.py:321-330](../data_processing/globem_preprocessing.py#L321-L330)).

> **Asymmetry to fix or to declare.** HRD does **not** do this. `prepare_hrd_dataset` defaults
> `align_midnight=False` ([data_preprocessing.py:787](../data_processing/data_preprocessing.py#L787))
> and `train_hrd.py` never passes it ([train_hrd.py:671-683](../train_hrd.py#L671-L683)), so
> HRD windows begin at each participant's first sample. The cosinor baseline compensates by
> recovering the wall-clock offset from `window_ids`
> ([baselines/cosinor.py:90](../baselines/cosinor.py#L90)), but the **seasonal branch's
> spectral readout does not** (§4.4): `_seasonal_spectral` reads absolute Fourier phase at
> fixed harmonic bins ([cost.py:671-694](../cost.py#L671-L694)), so on HRD those phases carry
> an arbitrary per-window offset. This is a concrete, cheap fix — pass `align_midnight=True`
> — and it is a genuine confound for any claim that reads phase from the seasonal branch.
> (Note the energy path already passes `align_midnight=True` for its pretrain windows,
> [data_preprocessing.py:1020](../data_processing/data_preprocessing.py#L1020), so the two
> HRD paths currently disagree with each other.)

### 4.3 Rhythm decomposition — two distinct decompositions

The study contains **two separate decompositions** which are easily confused and must be named
apart in the manuscript.

#### 4.3.1 The closed-form reference (analysis target, no learning)

For each window and channel, one least-squares fit against a fixed design matrix
([decomposition.py:66-88](../tasks/decomposition.py#L66-L88)):

$$
x_c(t)\;\approx\;\underbrace{\sum_{j=0}^{P}a_{c,j}\,\tilde t^{\,j}}_{\tau_c(t)}
\;+\;\underbrace{\sum_{k=1}^{K}\bigl[b_{c,k}\cos(k\omega t)+e_{c,k}\sin(k\omega t)\bigr]}_{\sigma_c(t)},
\qquad \omega=\frac{2\pi}{\text{period\_bins}},\quad \tilde t = t/T
$$

with $P=3$, $K=3$, and $\text{period\_bins}=24\cdot 60/\Delta = 96$ for HRD. Solved once by
pseudo-inverse for all $N\times C$ series simultaneously. **Physiological reading:** $\tau$ is
the multi-day drift of a channel's level (e.g. a week-long decline in activity); $\sigma$ is
the circadian waveform and its 12 h and 8 h harmonics, which is what lets a non-sinusoidal but
periodic profile — the usual shape of human rest–activity — be represented. **Band definitions
are study-specific**, not standard: $K=3$ harmonics of the 24 h fundamental is a modelling
choice, and the 8 h component in particular has no established generator.

#### 4.3.2 The learned decomposition (the model's own trend/seasonal split)

The encoder emits two sequences from one backbone output $V$:

- **TFD (trend)** — a mixture of causal autoregressive experts with kernel sizes
  $\{1,2,4,\dots,2^{L}\}$, $L=\lfloor\log_2(T/2)\rfloor$
  ([model_build.py:23-26](../model_build.py#L23-L26)). For $T=672$: $L=8$, giving 9 experts up
  to a 256-bin (64 h) receptive field.
- **SFD (seasonal)** — a `BandedFourierLayer`: rFFT over time, a learned complex linear map per
  frequency band, inverse rFFT. It is a *learned spectral filter*, not a fixed band-pass.

**Why decompose at all rather than feed the raw signal.** The two components support different
contrastive objectives (§4.4): the trend branch is trained with a MoCo InfoNCE over time
indices, the seasonal branch with an instance-discrimination loss in the *frequency* domain.
A single representation cannot be trained against both without the objectives interfering.
Whether the split succeeds is an empirical question the DRS and the plain-SSL twin are designed
to answer — and it is the single most falsifiable architectural claim in the paper.

**Time–frequency resolution cost.** One 7-day window at 15 min gives $T/2+1=337$ rFFT bins with
a frequency resolution of $1/168\,\text{h}^{-1}$; the circadian bin is $f{=}7$. There is no
overlap and no tapering, so a rhythm whose period drifts within the week is smeared across
neighbouring bins, and windowing sidelobes are not controlled. **For GLOBEM this is far more
severe:** at 4 samples/day the Nyquist limit is 2 cycles/day, so of the harmonics
`_seasonal_spectral` is designed to read $\{1\times, 2\times, 3\times, 4\times\}$ daily, only
the first two exist. With `run.sh (DATASET=globem)`'s $T{=}28$ the readout collapses to bins
$f\in\{1,7,14\}$ — circaseptan, circadian, and the 12 h harmonic sitting exactly at Nyquist,
where phase is not identifiable. **The 12 h bin should be dropped from GLOBEM's readout.**

### 4.4 Training

**Objective.** $\mathcal{L} = \mathcal{L}_{\text{temp}} + \alpha\,\mathcal{L}_{\text{seas}}$
with $\alpha=5\times10^{-4}$ by default ([train_hrd.py:480](../train_hrd.py#L480)) but
$\alpha=0.005$ as configured in `scripts/run.sh` — a 10× difference that must be reported as
the value actually used.

*Trend term* (MoCo InfoNCE at a random time index $r$, queue $Q\in\mathbb{R}^{d\times K}$,
$K=256$, $\tau_{\text{NCE}}=0.07$, momentum $m=0.999$):

$$
\mathcal{L}_{\text{temp}}=\frac{1}{B}\sum_{n=1}^{B}-\log\frac{\exp(q_n^\top k_n/\tau_{\text{NCE}})}{\exp(q_n^\top k_n/\tau_{\text{NCE}})+\sum_{j=1}^{K}\exp(q_n^\top Q_{:,j}/\tau_{\text{NCE}})}
$$

*Seasonal term* (both views encoded by the **trainable** encoder — no momentum encoder, no
queue; instance-discrimination applied separately to amplitude and phase):

$$
\mathcal{L}_{\text{seas}}=\tfrac12\bigl[\mathcal{L}_{\text{inst}}(A^q,A^k)+\mathcal{L}_{\text{inst}}(\phi^q,\phi^k)\bigr]
$$

**Phase must be embedded on the circle**, $\phi\mapsto[\sin\phi;\cos\phi]$
(`--phase-encoding circular`, the default, [train_hrd.py:485](../train_hrd.py#L485)). This is a
correction to upstream CoST and belongs in Methods: the contrastive loss scores pairs by dot
product, and on a raw `atan2` angle $\langle\phi_i,\phi_j\rangle$ depends on *where* the angles
sit rather than how far apart they are — two identical phases score 0 at $\phi{=}0$ but
$\pi^2$ at $\phi{=}\pi$, and the pair $(\pi-\epsilon,-\pi+\epsilon)$ scores the most negative
value possible. After the embedding the score is $\sum_c\cos(\phi_c^q-\phi_c^k)$, a function of
the angular difference alone. `circular_amp` additionally weights each channel by its own
(detached, RMS-normalised) amplitude so that channels whose phase is undefined noise do not
count as much as real rhythms.

**Augmentation.** Per-channel Gaussian jitter $\sigma_j=0.1$
([train_hrd.py:482](../train_hrd.py#L482)) plus a per-channel DC offset (`shift_sigma=0.5`,
[cost.py:497](../cost.py#L497)) — which moves only the MESOR/0-frequency bin and so preserves
every rhythm's amplitude and phase — plus timestep masking (`--mask-mode binomial`,
`--mask-keep-prob 0.5` in `run.sh`; the argparse default is `none`,
[train_hrd.py:515-525](../train_hrd.py#L515-L525)). **Random scaling is deliberately absent
because amplitude is a biomarker here**, and random cropping never fires because
`max_train_length == T`. Masking is training-only; `encode()` always runs unmasked.

**Optimizer.** TCN: SGD, momentum 0.9, weight decay $10^{-4}$. Transformer: AdamW,
$\beta=(0.9,0.999)$, weight decay $10^{-4}$, chosen because the Transformer trains unstably
under SGD ([cost.py:519-527](../cost.py#L519-L527)). `--lr` default $10^{-3}$
([train_hrd.py:538](../train_hrd.py#L538)); `run.sh` uses $5\times10^{-4}$ and
`--iters 6000`, `--batch-size 64`.

**Stopping and checkpointing.** Fixed iteration budget with **best-checkpoint restore on a
held-out SSL validation loss**. The SSL validation holdout is **by participant**, not by window
([train_hrd.py:764-775](../train_hrd.py#L764-L775)) — 10% of pretrain pids by default
([train_hrd.py:436](../train_hrd.py#L436)) — because a window-level split would put the same
people on both sides and the loss would measure "can I contrast windows of someone I trained
on." Since that loss selects the checkpoint, contaminating it selects the model. If the
holdout is smaller than one batch it is abandoned rather than silently yielding zero batches.

**Leakage assessment of training.** Clean. `pretrain_mask = ~test_mask`
([train_hrd.py:707](../train_hrd.py#L707)) excludes **every window of every test participant
from pretraining**, not merely from the probe — stronger than most published SSL wearable work
and worth stating explicitly. Unlabeled participants (those with no endpoint label) are used in
pretraining only. Validation participants *are* in pretraining, which is standard for SSL — no
labels are involved — but should be declared.

**Downstream readout.** $V^{(F)} = [\,\text{pool}(z^{(T)})\;;\;\text{spec}(z^{(S)})\,]$ with
`--pool mean` and `--season-pool spec` ([train_hrd.py:466-477](../train_hrd.py#L466-L477)).
The spectral readout exists because **time-domain pooling provably destroys the seasonal
branch**: $z^{(S)}$ is an inverse rFFT, so its mean over a whole window is *exactly* the $f{=}0$
(DC) coefficient — every oscillation integrates to zero — and `last` is one arbitrary phase
point at the edge ([cost.py:671-682](../cost.py#L671-L682)). Instead the sequence is
L2-normalised, rFFT'd, and amplitude+phase are read at chronobiological harmonics only.

**Resulting dimensions (HRD, TCN).** $d=160$; $D=\lfloor T/\text{bins\_per\_day}\rfloor=7$;
$f\in\{1,7,14,21,28\}$, all $\le T/2$, so $|f|=5$; spectral half $=2\times5\times160=1600$;
$V^{(F)}\in\mathbb{R}^{1760}$.

### 4.5 Decision — the probe unit is a function of label resolution

Three places in the repository disagree: the design document declares `persubject` primary,
`scripts/run.sh` sets `PROBE_UNIT="last"`, and
[experiment_q3.py:58-72](../experiment_q3.py#L58-L72) uses `persubject`. **I adopt the
following rule, which resolves all three and generalises to both datasets:**

> **The probe row must match the label's resolution.**
>
> | Label resolution | Primary unit | Sensitivity check | Excluded | Applies to |
> |---|---|---|---|---|
> | **Participant-level** | `persubject` | `last` | `all` | HRD depression endpoint; GLOBEM `--globem-label endpoint` |
> | **Window/period-level** | `all` | — | `last`, `persubject` | HRD emotional energy; GLOBEM `--globem-label weekly` (the default) |

**Why `persubject` for a participant-level label.** With ~24 windows per person all carrying
the same value, `all` is pseudo-replication: the effective $n$ is the participant count, but
the fit and its L2 penalty behave as if $n$ were the window count, so participants with longer
records dominate and any interval computed at row level is optimistically tight.
Aggregate-within-cluster is the textbook remedy when the estimator is not a mixed-effects or
GEE fit — and a logistic probe is neither. `last` avoids pseudo-replication but discards ~96%
of the windows and makes the estimate depend on one arbitrary week, which at $n\approx36$ is
pure variance. `persubject` — one row per participant holding $[\text{mean}\,|\,\text{std}]$ of
that participant's window embeddings ([train_hrd.py:812-821](../train_hrd.py#L812-L821)) —
keeps every window through the summary while keeping $n$ equal to the participant count. The
`std` half is not filler: within-person variability of the latent state is itself a candidate
marker and no single window can express it.

**Conditions attached to this choice**, all of which must be implemented:

1. `persubject` doubles $p$ (1760 → 3520 for the HRD TCN configuration). It **must** be paired
   with `--probe-pca` ([train_hrd.py:379](../train_hrd.py#L379)); the PCA sits inside the
   pipeline so it is refit per CV fold on that fold's training participants only
   ([_eval_protocols.py:55-68](../tasks/_eval_protocols.py#L55-L68)).
2. `last` is reported alongside as a pre-registered sensitivity check. If the two give the same
   ordering of representations, say so in one sentence and the choice stops being contestable.
3. **`run.sh` must change** from `PROBE_UNIT="last"` to `persubject`, and the separability table
   (§5) must be extended to emit the `persubject` unit — it currently emits only `all` and
   `last` ([rhythm.py:1649-1675](../tasks/rhythm.py#L1649-L1675)).

**Why `all` for a period-level label.** For emotional energy and GLOBEM's weekly label the
target varies *within* participant, so there is nothing to pseudo-replicate; `last` would score
one arbitrary day/week per person and `persubject` would average the label away entirely,
destroying the within-person signal that is the whole point. The code already reasons this way
([rhythm.py:1641-1651](../tasks/rhythm.py#L1641-L1651)) and reports its evidence for it.

**Note that RQ1's window-level targets are a third case.** The DRS target $\tau,\sigma$ varies
row by row, so the probe deliberately uses the participant split *without* the `probe_sel`
restriction ([train_hrd.py:840-841](../train_hrd.py#L840-L841)) — restricting it would only
discard fitting data. The participant-disjoint split is what protects that analysis.

### 4.6 Evaluation

**Splits.** All participant-level, all three datasets/tasks:

| Split | Rule | Source |
|---|---|---|
| Test | class-balanced, `--test-per-class 18` (HRD) / `50` (GLOBEM) | [train_hrd.py:56](../train_hrd.py#L56), [:421](../train_hrd.py#L421) |
| Pretrain | all non-test windows, incl. unlabeled participants | [train_hrd.py:707](../train_hrd.py#L707) |
| SSL val | 10% of pretrain participants, disjoint | [train_hrd.py:764-775](../train_hrd.py#L764-L775) |
| Probe pool | cohort minus test | [train_hrd.py:708](../train_hrd.py#L708) |
| Probe train/val | stratified, `--val-frac 0.25` | [train_hrd.py:830](../train_hrd.py#L830), [:434](../train_hrd.py#L434) |
| Probe CV | $k$-fold within pool, `CV_FOLDS=5` in run.sh | [train_hrd.py:236](../train_hrd.py#L236) |

Seeds are separable: `--split-seed` decides *who* goes where, `--model-seed` decides *how* it
trains ([train_hrd.py:552-559](../train_hrd.py#L552-L559)). Both default to `--seed`, and
`run.sh` leaves them unset — so the six sweep seeds vary cohort **and** optimisation together.
This is exploitable (§7.4).

**Metrics.**

| Metric | Formula | Range | Why appropriate here |
|---|---|---|---|
| AUROC | $P(\hat s_{+}>\hat s_{-})$, Mann–Whitney form ([_eval_protocols.py:72-82](../tasks/_eval_protocols.py#L72-L82)) | $[0,1]$, chance 0.5 | Threshold-free and **prevalence-invariant** — the only headline metric that survives the balanced test cohort |
| Balanced accuracy | $\tfrac12(\text{sens}+\text{spec})$ | $[0,1]$, chance 0.5 | Prevalence-invariant; used for threshold selection instead of F1-max, because F1 saturates at 0.667 for an all-positive predictor whereas BAcc gives 0.5 ([_eval_protocols.py:93-104](../tasks/_eval_protocols.py#L93-L104)) |
| MCC | $\frac{TP\cdot TN-FP\cdot FN}{\sqrt{(TP{+}FP)(TP{+}FN)(TN{+}FP)(TN{+}FN)}}$ | $[-1,1]$, chance 0 | 0 for a degenerate predictor, unlike F1 |
| Sensitivity / specificity | within-class rates | $[0,1]$ | The only threshold metrics that transport across base rates unchanged |
| PPV/NPV at target prevalence | Bayes transport ([_eval_protocols.py:205-238](../tasks/_eval_protocols.py#L205-L238)) | $[0,1]$ | What a clinician actually needs; at 10% prevalence a 90/90 classifier has PPV 0.5, not 0.9 |
| Brier / ECE / MCE | ([_eval_protocols.py:253-277](../tasks/_eval_protocols.py#L253-L277)) | $[0,1]$ | Two models with identical AUROC can differ entirely in whether $p{=}0.7$ means 70% |
| Ridge $R^2$ | $1-\|u-\hat u\|^2/\|u-\bar u_{\text{tr}}\|^2$, **unclipped** | $(-\infty,1]$ | Negative values are real and reportable; clipping would bias DIS upward (§5) |
| Circular $r$, MAE$_\phi$ | ([rhythm.py:1216-1222](../tasks/rhythm.py#L1216-L1222)) | $[-1,1]$; hours | Acrophase is an angle; an $R^2$ on raw radians is meaningless across the $0/2\pi$ wrap |

**Operating points are committed without seeing test.** Two sensitivity-anchored thresholds
(0.80, 0.90) come from validation or out-of-fold predictions, and 0.5 is fixed a priori
([train_hrd.py:887-905](../train_hrd.py#L887-L905),
[_eval_protocols.py:280-294](../tasks/_eval_protocols.py#L280-L294)). This is the correct
discipline and should be stated.

**Chance levels.** AUROC 0.5; the majority-class rung is explicitly entered at 0.5 rather than
at the majority rate, because a constant score has AUROC 0.5 by definition
([experiment_q3.py:185](../experiment_q3.py#L185)). For the time-shuffled control the chance
level is **not** 0 — see §5's discussion of the permutation-invariant floor.

---

## 5. Separability analysis

### 5.1 The quantity

For each representation view $R$, a scikit-learn pipeline
`StandardScaler → [PCA] → LogisticRegression(class_weight="balanced", max_iter=3000)` is fit on
the training participants and scored on the held-out ones
([rhythm.py:680-733](../tasks/rhythm.py#L680-L733)). Regularisation is view-dependent by
design: high-dimensional FFT views ($p>2000$) are kept full-dimensional with strong L2
($C{=}0.01$); low-dimensional views use $C = $ `--probe-c`. The reported cell is
$\mathrm{AUROC}$ (threshold-free) or a threshold metric at a per-view threshold tuned on the
**participant-aggregated validation split** ([rhythm.py:706-712](../tasks/rhythm.py#L706-L712)).

### 5.2 What the table compares

**Rows** — 15 views × 2 probe units, produced by
[rhythm.py:113-126](../tasks/rhythm.py#L113-L126) plus PCA counterparts:

| View | Dimension (HRD TCN) | What it is |
|---|---|---|
| `V (encoder pre-decomp)` | 320 | backbone output, mean-pooled, **before** the trend/seasonal split |
| `Full [V^(T);V^(S)]` | 320 | both branches, time-pooled |
| `Trend V^(T)` | 160 | trend branch only |
| `Season V^(S)` | 160 | seasonal branch, **time**-pooled |
| `Season V^(S) spectral` | 1600 | seasonal branch, frequency-domain readout |
| `Seasonal amp` | 337×160 | full rFFT amplitude spectrum |
| `Seasonal phase` | 337×160 | full rFFT phase spectrum |
| `Cosinor (paper)` | 96 | CosinorPy clone of Yan et al. 2022 — the classical benchmark |
| + PCA20 counterparts | 20 | dimensionality-matched comparison |

**Columns** — `Unit`, `Representation`, `Dim`, `Thr`, then window-level and participant-level
AUC / F1 / Acc / BAcc / MCC / Sens / Spec.

**One cell** = the held-out performance of a linear probe on that view, at that probe unit,
under that view's regularisation, on a **single** train/val/test split.

### 5.3 Why the table is necessary

It is the only analysis that holds probe, split, threshold rule and class weighting fixed while
varying *only the representation*. Every other result compares a model to a baseline; this
compares the model's own internal views to each other, which is what licenses statements of the
form "the seasonal branch carries the discriminative signal and the trend branch does not." It
is also the only place the classical cosinor benchmark meets the learned representations under
identical machinery — the fair comparison a chronobiology reviewer will demand.

### 5.4 How to read it

- **Chance is AUROC 0.5**, and with the balanced test cohort accuracy chance is also 0.5.
- **Rows are comparable to each other, not to the headline.** The table uses a single split
  while E0 uses $k$-fold CV refit on all pool participants
  ([rhythm.py:688-694](../tasks/rhythm.py#L688-L694)). **The code's own explanation is
  incomplete:** it attributes the gap to CV-vs-single-split, but there is a larger cause.
  `extract_representations` calls `encode` *without* `season_pool`
  ([rhythm.py:77-81](../tasks/rhythm.py#L77-L81)), so the table's `Full` row is a **320-dim**
  time-pooled vector, whereas the headline probe uses the **1760-dim** spectral readout
  ([train_hrd.py:790-791](../train_hrd.py#L790-L791)). These are different representations
  under the same name. **This must be fixed or renamed before publication** — as it stands the
  table's `Full` row does not describe the model the paper reports.
- **The `Cosinor (paper)` rows are one number printed twice.** `paper_cosinor_features` is
  called with `pids`, so it already collapses each participant to one vector upstream with
  phases averaged as angles; every window of a person therefore carries the same vector and the
  `last` and `persubject` rows come out bit-identical
  ([rhythm.py:1662-1678](../tasks/rhythm.py#L1662-L1678)). This is **not** evidence that the
  unit choice does not matter, and the table footnote says so.
- **The reported spread reflects neither seed nor fold variance.** A single split, a single
  seed, no interval. Any spread a reader infers from the table is between-view variation
  confounded with split noise. §7.3 specifies the fix.

### 5.5 What it supports — and what it does not

**Legitimately supports:** the relative ordering of representation views under a common probe;
the comparison of learned views against the classical cosinor benchmark; the claim that a
particular branch or readout carries more label-relevant information than another.

**Does NOT support, and must not be cited for:**

- an absolute performance estimate for the method — it is one split with no interval;
- a claim that any two rows *differ*, absent a paired test across seeds (§7.3);
- a claim about the headline model, because the `Full` row is a different representation (§5.4);
- any statement about the unit choice based on the Cosinor rows;
- a causal claim about *why* a view wins — that is E3.B's job, not this table's.

---

## 6. Figure-by-figure analysis

Every figure below is written to the variant directory. Interpretations are drafted so they can
drop into the manuscript once numbers exist.

### 6.1 `pretrain_loss.png` / `val_loss.png` ([train_hrd.py:291-316](../train_hrd.py#L291-L316))

**Axes:** iteration (x) vs SSL loss (y), training and participant-disjoint validation.
**Supports:** that pretraining is a non-trivial optimisation problem.
**If the hypothesis holds:** the loss decreases substantially and validation tracks training.
**Artifact signature:** a loss collapsing to near zero means the pretext task was *solved
outright* — the two augmented views became trivially matchable — and the resulting encoder can
be expected to underperform its own random initialisation. This is a documented failure mode of
this exact configuration and is the reason `--mask-mode binomial` exists (§4.4). **This figure
is a gate, not a supplementary plot: read it before trusting any downstream number.**
*Draft:* "Pretraining converged to a non-degenerate optimum (final InfoNCE ≈ [X]), with the
participant-disjoint validation loss tracking the training loss; the timestep-masking
augmentation was required to prevent the contrastive task collapsing."

### 6.2 `decomposition_recovery.png` ([decomposition.py:273-300](../tasks/decomposition.py#L273-L300))

**Axes:** sensor channel (x) vs held-out ridge $R^2$ (y); grouped bars for Full→$\tau$,
$V^{(T)}$→$\tau$, leak $V^{(S)}$→$\tau$ and the mirrored seasonal triple.
**Supports:** E1.2, the diagnostic.
**If the split holds:** $\rho(V^{(T)}\!\to\!\tau) > \rho(V^{(S)}\!\to\!\tau)$ and
$\rho(V^{(S)}\!\to\!\sigma) > \rho(V^{(T)}\!\to\!\sigma)$.
**Artifact/failure signature:** leak exceeding own-branch recovery means the trend branch
predicts the circadian component better than the seasonal branch does — the split has not
separated in the direction the architecture assumes, and DIS $\approx 0$ regardless of how
large the individual $R^2$s are.
*Draft:* "Both branches recovered the harmonic reference at $R^2 =$ [X]; the disentanglement
score was [X], indicating that the trend/seasonal split [did / did not] separate the two
components in the direction the architecture posits."

### 6.3 `rq1_controls.png` ([experiment_q1.py:236-253](../experiment_q1.py#L236-L253))

**Axes:** control condition (trained / plain-SSL / random-init / time-shuffled) on x, held-out
$R^2$ on y; two bars per group (Full→$\tau$, Full→$\sigma$); error bars are the SD over
`--n-shuffle` draws; **dashed horizontal lines are the permutation-invariant floor.**
**Supports:** E1.5 — the calibration for every RQ1 number.
**If the hypothesis holds:** trained $\gg$ random-init, and time-shuffled falls to the floor.
**Artifact signature, and it is the decisive one:** if random-init matches or exceeds trained,
the architecture rather than the pretraining is doing the work, and no RQ1 claim survives.
**Why the floor line matters:** because the window spans exactly 7 whole 24 h periods, the
harmonic design columns are orthogonal to the constant column and $\overline{\tau}=\overline{x}$
exactly; the window mean is permutation-invariant, so Full→$\tau$ **can never reach 0** and
"$R^2\to0$" is the wrong expectation for the trend half
([experiment_q1.py:89-109](../experiment_q1.py#L89-L109)). The floor is *measured*, using
per-channel window mean and std held constant across timesteps — a conservative floor, which is
the useful direction.
*Draft:* "Against a randomly initialised encoder of identical architecture and a per-window
time permutation scored against the original reference, the trained encoder recovered the
circadian component at [X] versus [Y] and [Z]; the permutation control fell to the measured
permutation-invariant floor, confirming that the recovery reflects temporal structure rather
than window-level marginal statistics."

### 6.4 `hrd_rhythm_separability_{depression,energy}.png` ([rhythm.py](../tasks/rhythm.py))

Rendered form of §5. Read it with §5.4's caveats attached.

**Both tables now carry the RQ3 utility ladder's two controls**, passed in as `extra_views`
and scored by the same probe, on the same rows, as every other view:

| rung | isolates |
|---|---|
| `Majority` | the floor - a constant column, so the probe fits an intercept only |
| `Handcrafted (mean/std)` | what a **representation** bought over per-channel summary stats |
| `Random-init` | what **training** bought - same architecture, same seed, weights never updated |
| `DSSL plain (no disentangle)` | what **disentangling** bought - the plain SSL twin |

`Majority` is a constant feature run through the identical probe rather than a written 0.5.
Measured that way it **must** come out at AUROC exactly 0.500 (verified: 0.500 on both units),
so any other value is a leak in the probe pipeline rather than a property of a representation
- a standing check the hardcoded constant could not provide. `handcrafted_features`
(tasks/energy.py) is now the single definition of that rung for the EE probe, the RQ3 utility
ladder and both separability tables; RQ3 previously inlined an identical copy (verified
element-wise identical before the refactor).

Without them the table ranked only representations that were all produced by a trained,
disentangled encoder, so it could say which branch was best but not whether either step bought
anything - which is exactly what RQ1 and RQ3 ask. The RQ3 utility table already had them; the
two separability tables, which are the headline objects for depression and for emotional
energy, did not. `_random_init_repr` (train_hrd.py) is now the single source of that floor for
all three, differing only in the pooling the caller uses; its docstring previously claimed the
depression ladder shared it, which was not true until this change.

Both get dimensionality-matched `(PCA)` twins like every other low-dim view, so the comparison
is not confounded by width. Neither adds a pretraining: the plain twin is cached to
`plain_encoder.pt` on the path the energy probe and `experiment_q1/q3` already share, and the
random-init encoder is never trained.

**PCA twins are gated on width.** Only views wider than `PCA_TARGET` (20) get one: below it
`n_comp` is clamped to the view's own width, so the "dimensionality-matched control" would be
the same row printed twice under a name promising otherwise, and PCA cannot run on the 1-column
`Majority` view at all. No pre-existing row is affected - the narrowest that had a twin is
`Cosinor (paper)` at 96.

### 6.5 `frequency_contrast.png` / `frequency_heatmap.png` ([rhythm.py:249-457](../tasks/rhythm.py#L249-L457))

**Axes:** period (x, log-spaced) vs seasonal-branch spectral power (y), one curve per class
(contrast) or a participant×frequency image (heatmap).
**Supports:** that the seasonal branch has learned physiologically meaningful bands.
**If the hypothesis holds:** peaks at 24 h and its harmonics, and a group difference
concentrated in the circadian band.
**Artifact signature:** a peak at exactly the window length (168 h) or at the bin edges is
spectral leakage from the un-tapered rectangular window, not physiology. A broadband difference
with no band structure indicates an amplitude/scale difference between groups surviving the
per-participant z-score, not a rhythm difference. **On HRD, absolute-phase structure in this
figure is confounded by the missing midnight alignment (§4.2).**

### 6.6 `participant_trajectory_*.png` ([rhythm.py:458-612](../tasks/rhythm.py#L458-L612))

**Axes:** time (x) vs latent trajectory (y) for individual participants selected by
(baseline, endpoint) status.
**Supports:** illustrative, not inferential.
**Caution:** hand-picked participants support no claim. Label the panels as illustrative in the
caption or a reviewer will read them as evidence.

### 6.7 `hrd_tsne_label.png`, `hrd_umap_label.png`, `hrd_tsne_clinical.png`, `*_subject.png` ([rhythm.py:888-973](../tasks/rhythm.py#L888-L973))

**Axes:** unitless embedding coordinates; colour = class label or a clinical marker.
**Supports:** qualitative structure only.
**Artifact signature:** apparent clustering in `*_label.png` that disappears in
`*_label_subject.png` means you are seeing **participant identity, not disease** — the single
most common failure in this literature, and the reason the subject-aggregated variant exists.
Report both or neither.
*Draft:* "Two-dimensional embeddings are shown for qualitative inspection only; the
subject-aggregated variant is included because window-level embeddings of longitudinal wearable
data separate participants rather than classes."

### 6.8 `circadian_similarity_depression.png` + `circadian_landscape.png`

A reproduction of **WavesFM Fig. 14** (Cao, Yang, Liu et al., Google Research, 2026,
arXiv:2605.09173) on this cohort, with its layout and, critically, its **fixed axes**.

**Rows are the pipeline, top to bottom**, mapping the paper's comparison onto this
architecture. WavesFM contrasts the *input* of its temporal encoder with that encoder's
*output* - before and after the step the paper contributes. The step this project contributes
is the **disentangling**, not the backbone, so the rows are:

| row | stage | what it answers |
|---|---|---|
| `Raw sensor bins` | model input | how much rhythm was in the signal to begin with |
| `Backbone h` | **before** disentangling | what the shared encoder produced (`tcn_output=True`) |
| `DSSL trend V^(T)` | **after** | what the TFD kept |
| `DSSL season V^(S)` | **after** | what the SFD kept - the branch that should carry rhythm |

The `Backbone h` row is what makes a collapse **attributable**. With only raw and the two
branches, two stages are folded into one and a degenerate `V^(S)` is indistinguishable between
"the backbone already lost the structure" and "the SFD destroyed it" - which, given that
`V^(S)` collapses in 21 of 23 runs, is the question that matters. It costs nothing: the encoder
already has an early exit at `tcn_output=True`, so there is no second forward pass and no model
change. The row is skipped under `disentangle=False`, where the backbone output already *is*
the single representation and the row would be an exact duplicate.

Clock features are excluded from the raw row: they are deterministic 24 h cosines and would
draw a perfect rhythm by construction.
**Columns** (a)-(c) are the paper's: (a) every intra-window pair against the time distance
between its two bins; (b) and (c) every bin against its position in the week, referenced to
that window's own Monday 00:00 and Monday 09:00 bin. **Two anchors, not one, is the paper's
own control** - a rhythm visible under only one reference hour is a property of the reference.

**Column (d), `periods present`, is an addition.** Panel (a) shows that *something* recurs
daily; (d) says at *which* period, as the spectrum of `s(lag)` normalised to a share of its
oscillating power, with the 24 h and 12 h shares printed above. The window is 168 h, so 24 h
falls on integer bin 7 and 12 h on bin 14 - both exact, no interpolation. This is what makes
the documented artefact signature checkable rather than a matter of eyeballing: banding at the
window length instead of at 24 h now shows as a peak in the wrong place.

The 168 h "weekly" line is deliberately **not** reported. One cycle inside a one-week window is
a trend, not a rhythm - nothing separates it from a slow drift - so a circaseptan number there
would be an artefact of the window length. The paper reads circaseptan structure the other way,
as a flattening of the Sat/Sun peaks, which stays visible in columns (b) and (c).

**A high 24 h share is not evidence of biological rhythm recovery.** It says the representation
repeats daily, which a clock does perfectly: on run 20007838 the degenerate `tcn/none` seed 22
would score a near-total 24 h share while carrying nothing about any participant. Column (d)
must be read next to `participant_var_frac`, which is the number that separates a recovered
biological rhythm from a clock. Per-person *marker* recovery - acrophase, amplitude, MESOR,
interdaily stability - is measured elsewhere and not by this figure: see the decomposition
recovery score (SS6.5) and the rhythm-axis probe.

**Every panel is fixed to a `[-1, 1]` cosine axis.** The paper fixes its own at 0.2-1.0 for the
same reason, and this is the single most consequential detail of the figure. The earlier
version let matplotlib autoscale each panel, and on run 20007838 that produced a diurnal swing
of 0.0035 (`tcn/none` seed 82) and one of 1.9895 (seed 22) drawn at *identical visual height*.
Runs differing by a factor of 568 looked alike, and a fully collapsed representation was drawn
as a textbook circadian rhythm.

**Read the figure only after the two numbers printed above each row.** WavesFM needed no such
numbers: it had one model that did not degenerate. These runs produce two *opposite*
degeneracies, and a similarity plot cannot tell them apart because both draw a clean curve:

| | `mean cos` | diurnal swing | `single_mode_r2` | `power_24h` | between-person var |
|---|---|---|---|---|---|
| collapse to one **direction** | ~1 | ~0 | - | any | ~0 |
| collapse to one **frequency** | ~0 | ~2 | ~1 | ~1 | ~0 |
| usable | mid | mid | < 1 | high | **substantial** |

Verified on four constructed regimes plus a `disentangle=False` model (which must yield
exactly two rows, not a duplicated backbone). In all four the backbone is built healthy, so the
figure has to attribute the collapse to the disentangler, and does: `h` holds 53.8%
between-person variance and stays unflagged while the degenerate `V^(S)` rows drop to 0.0%.
The fourth regime is a clean 12 h rhythm carrying real
per-person structure: it scores a 12 h share of 75.0% against a 24 h share of 0.1%, keeps 56.6%
between-person variance, and is correctly left unflagged - so column (d) identifies *which*
period recurs rather than only that something does, and `single_mode_r2` correctly falls to
0.000 there because the single-24 h-mode fit fails.

Only the last column separates them. A representation that serves RQ1-RQ3 must move with the
clock *and* still separate people at the same hour; a perfect clock scores a maximal swing and
is useless. Both fractions are one-way sums of squares - explained by that factor **alone** -
so they are not orthogonal and do not sum to 1. Read each against zero, never as a partition.
Rows meeting either degeneracy are labelled `COLLAPSED` in red on the figure. The test runs on
every learned row, the backbone's included - localising the collapse is the point - and only
the raw row is exempt, since it is whatever the sensors recorded.

**Measured on run 20007838** (`tcn/none`): `V^(S)` seed 22 fits a single 24 h cosine with
R^2 = 0.99981 - a perfect clock, identical for every participant - while seeds 42 and 82 sit at
mean cosine 0.9935 and 0.9959, i.e. every window pointing the same way. Both are collapse, onto
different Fourier modes. This is the same endemic `V^(S)` failure seen in 21 of 23 runs; the
figure now states it instead of hiding it.

**`circadian_landscape.png`** ([collect_results.py](../scripts/collect_results.py), written by
`collect_results.py` across a whole results directory) is **beyond the paper** - WavesFM
compares two models on one figure and never varies a seed, so it needs no cross-run summary.
Panel (a) plots clock-variance against participant-variance, one point per (variant, seed,
stage) with a marker per stage (backbone / trend / season): the usable corner is top-right, the
shaded band at the bottom is "a clock, or collapsed", and a variant whose backbone sits in the
usable corner while its season branch sits in the band has lost its structure *in the
disentangler*. Panel (b) is the diurnal swing per seed on a fixed `[0, 2]` axis, so between-seed
irreproducibility is visible rather than averaged away.

**Artifact signature:** banding at the window length rather than at 24 h; a rhythm that appears
under one anchor column but not the other; or, on HRD, apparent banding driven by the arbitrary
per-participant phase origin (SS4.2).

### 6.9 `rq2_concordance.png` + `rq2_concordance.csv` + `rq2_cells.csv` ([experiment_q2.py](../experiment_q2.py))

**Axes:** directional concordance $C$ (x) with its 95 % participant-bootstrap interval, one row per representation, dashed reference at chance $0.5$.
**Supports:** RQ2 as stated, and nothing else — this is the whole experiment now.
**If the hypothesis holds:** CoST's interval lies wholly above 0.5 and above Random-init's, with `delta_vs_DSSL` bootstrapped on **shared participant draws** so the difference can be significant even when the two marginal intervals overlap.

**Ceilings, not competitors.** The ground truth $g$ is defined in 24-h cosinor space, so `Cosinor` and `Raw (hourly)` encode the estimand explicitly and are labelled `[ceiling]` in the figure. This is the same circularity that put cosinor at the top of the retracted Spearman table ($
ho_arphi=0.232$ vs CoST $0.110$) — here it is declared rather than reported as a finding. **The comparison that carries the claim is CoST vs Random-init.**

**Built-in null control:** a representation that responds only to *how much the input moved* must score 0.5. Verified at 0.501 / 0.515 / 0.491 across three independent seeds, with the bootstrap interval containing 0.5 — see §3.7.
**Artifact signature:** Random-init matching or beating CoST means the distance is reading input-space geometry, not learned rhythm. `rq2_cells.csv` must be checked first: a level whose `n_strata_with_both_classes` is 0 contributes no evidence at all.

**Scope limit, stated not hidden:** the perturbation is a phase shift, so the claim is about rhythmic **phase** deviation. The amplitude arm was tested and rejected (§3.7).

### 6.11 `rq3_degradation.png` ([experiment_q3.py](../experiment_q3.py))

**Axes:** degradation level (x) vs participant AUROC (y), one panel per factor, with the
shaded participant bootstrap band. Four horizontal references: chance (0.5), the intact
DSSL AUROC, the design threshold intact $-0.05$, and the **smallest drop this sample can
resolve**. The header states the intact AUROC with its CI and the number of held-out
participants, so the claim can be accepted or rejected from the figure alone.
**Supports:** E3.C, the operating envelope.
**If the hypothesis holds:** a plateau then a drop, with $c^{*}$ inside a narrow interval.

**$c^{*}$ is withheld unless the data locate it.** It is one level of an ordered grid, and it
is reported only when both hold:

1. its participant-bootstrap CI covers **less than half** the tested levels - an interval
   still spanning the grid does not name a level; and
2. $c^{*}$ exists in **more than 90%** of bootstrap draws - one that vanishes in a tenth of
   them is not a stable quantity.

When either fails the panel is greyed and prints the reason instead of a green line. The
bootstrap was always paired (one participant draw shared by the intact reference and every
degraded level, which is what gives $c^{*}$ an interval at all); before this gate the interval
was computed, written to `rq3.json`, and then ignored while the point estimate was drawn as a
solid line.

**The resolution floor.** `DROP` = 0.05 is a design constant; whether a drop that size is
*measurable* is a property of the sample. The paired bootstrap gives the standard error of
(intact $-$ degraded) directly, and `min_resolvable_drop` = $1.96\times$ its median across the
grid is the smallest drop the run can distinguish from zero. **On run 20007838 seed 82 that
floor is 0.220 against a design threshold of 0.05** - a factor of 4.4. At 36 held-out
participants (18 depressed) the Hanley-McNeil SE of a single AUROC is 0.087-0.097, so 0.05 is
roughly half of one standard error and $c^{*}$ was the first crossing of a threshold buried
inside the noise. `cstar`'s own docstring had already recorded this; the gate now enforces it.

**Replayed on the run's real curves**, seed 82's five reported values ($c^{*}$ = 5 / 120 / 0.4
/ 0.2 / 3) are all withheld: the intervals cover 57%, 57%, 67%, 100% and 67% of their grids.
A positive control at n = 400 with a genuine cliff recovers all five with intervals covering
14-33% of the grid and no undefined draws, so the gate withholds for lack of power rather than
by construction.

**Artifact signature:** a *flat* curve across all levels including the most extreme means
either the probe is at chance (`breakdown_valid` False - §3.5) or the degradation is not
reaching the model. Non-monotonicity is the other tell: on seed 82, 80% block missingness
scored AUROC 0.741 against an intact 0.710, i.e. destroying most of the data improved the
result. Note that MCAR/block missingness is implemented as setting bins to 0.0, which after
per-participant z-scoring is the participant's **mean** - so these curves measure robustness to
mean-imputation, not to missingness, and there is no missingness-indicator channel
([experiment_q3.py:107-117](../experiment_q3.py#L107-L117)). State that in the caption.

### 6.12 Cross-variant figures ([collect_results.py](../scripts/collect_results.py), [scripts/plot_position_similarity.py](../scripts/plot_position_similarity.py))

`rhythm_vs_prediction.png` relates a rhythm-recovery axis to downstream AUROC across variants;
the position-similarity figures visualise what each positional encoding does to the
attention/position geometry. Both are cross-variant and require the full PE sweep restored
(§3.4) to carry any weight.

---

## 7. Statistical reporting audit

### 7.1 Tests used, and whether their assumptions hold

| Test | Where | Assumption status |
|---|---|---|
| Participant bootstrap (percentile) | [_eval_protocols.py:107-143](../tasks/_eval_protocols.py#L107-L143) | **Correct choice.** Rows are not independent — one person contributes many correlated windows — so resampling *participants* matches the claim "transfers to a new person." Percentile intervals are approximate at $n\approx36$; BCa would be better but the difference is second-order next to the sample size. |
| Paired bootstrap for $\Delta$AUC | [experiment_q3.py:222-236](../experiment_q3.py#L222-L236), [experiment_q2.py:246-261](../experiment_q2.py#L246-L261) | **Correct and above the norm.** Every representation is scored on the *same* participant draw, so the difference's interval can exclude zero even when the marginals overlap. Differencing two independent intervals cannot do this. |
| Wilcoxon signed-rank, one-sided | [experiment_q2.py:116-121](../experiment_q2.py#L116-L121) | Directional hypothesis is justified (a valid deviation score must correlate *positively*). Guarded to $\ge6$ participants because below that the statistic cannot reach $p<0.05$ at any effect size — a good guard. |
| Within-person label permutation | [experiment_q2.py:307-317](../experiment_q2.py#L307-L317) | **The right null.** Shuffling labels *inside* each person preserves that person's score distribution and window count, so the null isolates the association. |
| Kruskal–Wallis across trajectory groups | [experiment_q2.py:529](../experiment_q2.py#L529) | Distribution-free, appropriate. Requires $\ge3$ participants per group — enforced at [experiment_q2.py:525](../experiment_q2.py#L525). With four groups from ~36 test participants this test will be badly underpowered; report the effect size and group sizes, not just $p$. |
| Two-stage paired bootstrap for $\Delta_\pi$ | [collect_results.py:745-823](../scripts/collect_results.py#L745-L823) | **Well-motivated.** Resamples participants *and* seeds with replacement, carrying run-to-run variance as a random effect rather than averaging it away. The design's original "paired Wilcoxon over $S$ seeds" is correctly rejected: at $S{=}6$ the two-sided signed-rank test bottoms out at $p=0.031$, so after Holm nothing reaches 0.05 however large the effect. |

### 7.2 Multiple-comparison correction

Holm is applied to the $\Delta_\pi$ family ([collect_results.py:735](../scripts/collect_results.py#L735)).
**Three other families are uncorrected and should be:**

1. **E1.3 per-channel probes** — 3 markers × $C_s$ channels (12 tests on HRD, 36 on GLOBEM).
   The code's own docstring says to "treat the per-channel table as a family and correct for
   multiplicity before calling any single channel significant"
   ([rhythm.py:1300-1301](../tasks/rhythm.py#L1300-L1301)) — but no correction is implemented.
   Since E1.3 is now the headline (§3.3), **this is mandatory.**
   ([experiment_q2.py:504-512](../experiment_q2.py#L504-L512)).
3. **The separability table** — 21 views × 2 units, with no test at all (§5.4). The four
   ladder rungs added in §6.4 raise the view count; they do not change the fact that no cell
   carries an interval.

### 7.3 Effect sizes and confidence intervals

Present and correct for: E0 (participant bootstrap on AUROC and on window-level AUROC),
E2 ($C$ CI plus shared-draw $\Delta$ CI), E3.A (paired $\Delta$AUC CI), E3.C ($c^{*}$
with a bootstrap interval and the fraction of draws where $c^{*}$ is undefined —
[experiment_q3.py:302-306](../experiment_q3.py#L302-L306)), E1.4.

**Missing:** the separability table (§5.4) and the E1.3 headline itself, which reports a
point $R^2$/circular $r$ per marker — and now a paired random-init delta — all without an
interval. Since E1.3 is the headline, it
needs one. The machinery exists — `_probe_r2` already stores per-participant sufficient
statistics so a participant bootstrap is an exact reweighting rather than a refit
([decomposition.py:184-200](../tasks/decomposition.py#L184-L200)) — and the same pattern should
be applied to `rhythm_axis_probe`.

### 7.4 Seeds, folds, and whether the reported variance covers what matters

Three variance sources exist: **cohort** (which participants land in the test set),
**optimisation** (initialisation, augmentation draws, SGD noise), and **probe** (fold
assignment).

- Cohort variance is captured *within* a run by the participant bootstrap.
- Optimisation variance is captured *across* seeds — but `run.sh` leaves `SPLIT_SEED` and
  `MODEL_SEED` empty, so `--seed` drives both and the two sources are **confounded**. The
  crossed design that separates them is already implemented
  ([train_hrd.py:560-567](../train_hrd.py#L560-L567)) and is simply not used.
- Probe/fold variance is not reported separately.

**Recommendation, and it is the highest-leverage change in this document.** A single
36-participant test set gives AUROC intervals roughly $\pm0.20$ wide. Under the design's own
rule — a claim is licensed only when the CI excludes zero — E3.A cannot license *any* effect of
realistic size. But the six sweep seeds are already six different random participant splits.
Two options, in order of preference:

1. **Repeated participant-level holdout:** aggregate across seeds and report the across-split
   distribution as the headline, with the crossed `--split-seed`/`--model-seed` design used to
   separate cohort from optimisation variance.
2. **$k$-fold participant CV over all labeled participants with per-fold pretraining** — 5
   pretrainings, fewer than the 12 the current sweep already pays for, and it uses every
   labeled participant as test exactly once.

For GLOBEM, neither is the first move: **implement the leave-one-cohort-out split** (§3.6),
which gives four genuinely external test cohorts and is a far stronger generalization claim than
any within-cohort resampling.

---

## 8. Assumptions, limitations, weaknesses

Each item names a concrete control or ablation that would address it.

### 8.1 Methodological

| # | Limitation | Control / fix |
|---|---|---|
| M1 | ~~`robust_z` uses the participant's entire record, including future windows.~~ **Resolved by the E2 redesign:** `robust_z` is deleted — $C$ is a rank statistic computed *within* (participant, level), so no per-person standardisation is needed and nothing is read from the future. Within-person z-scoring in [data_preprocessing.py:544](../data_processing/data_preprocessing.py#L544) remains retrospective and still applies to E0/E1/E3. | Report the preprocessing z-scoring as a stated limitation of the *input pipeline*; the RQ2 headline no longer depends on it. |
| M2 | **HRD windows are not midnight-aligned**, so absolute Fourier phase in the seasonal branch carries an arbitrary per-window offset — while GLOBEM anchors to Monday (§4.2). | Set `align_midnight=True` in the HRD path and re-run E1.3 and the spectral separability views. Report both. One-line change. |
| M3 | **The separability table's `Full` row is a different representation from the headline model** (320-dim time-pooled vs 1760-dim spectral, §5.4). | Pass `season_pool` through `extract_representations`, or rename the row `Full (time-pooled)` and add the spectral variant. |
| M4 | **Missingness is mean-imputed with no indicator channel** ([experiment_q3.py:107-117](../experiment_q3.py#L107-L117)); the E3.C curves measure robustness to mean-imputation. Real non-wear is detectable at inference. | Add a missingness-mask channel and re-run the block-missing arm. This would strengthen the deployment claim materially. |
| M5 | **GLOBEM whole-array imputation**: with sleep ~33% present, most of that channel is interpolated, and the architecture forces this (§4.2). | Ablate by dropping the sleep channels entirely and comparing; report per-channel observed fractions in the paper. |
| M6 | **E1.3 uses a different held-out protocol from E1.2** — train→test vs out-of-fold within test (§3.3). The missing random-init control was added during preparation of this document ([experiment_q1.py:328-344](../experiment_q1.py#L328-L344)). | State both protocols explicitly wherever "held-out" appears. Add a participant bootstrap to the paired random-init delta (§7.3, S5). |
| M7 | **$\Delta_\pi$ rests on a single contrast arm.** The sweep carries the wall-clock arm (`tcn:none` vs `tcn:circular`, §3.4), so E1.4 is answerable, but one arm is not a family and the Holm correction is a no-op. | Add `factorized` to separate the calendar *anchor* from the circular *basis*. Not blocking: the existing pair already answers "does wall-clock anchoring help at all". |
| M8 | **GLOBEM's leave-one-cohort-out split is specified but unimplemented**; `--holdout` does not exist and `vit` is not a backbone (§3.6). | Implement `--holdout DS{1..4}`; remove `vit:none` from the runner. |
| M9 | **`MODEL_ARCHITECTURE.md` was DELETED** (it specified $C{=}15$ with 10 sensor channels and SGD/lr $10^{-3}$/600 iters/$\alpha{=}5	imes10^{-4}$, against the code's 4 sensors, 7 clock features, lr $5	imes10^{-4}$, 6000 iters, $\alpha{=}0.005$). Recoverable from git if ever needed. — it specifies $C{=}15$ with 10 sensor channels including `Floors`/`Sedentary`/`*_Active`/`calls`, and SGD/lr $10^{-3}$/600 iters/$\alpha{=}5\times10^{-4}$; the code has 4 sensors, 7 clock features, and `run.sh` uses lr $5\times10^{-4}$, 6000 iters, $\alpha{=}0.005$. | Regenerate from the code or mark superseded. **Do not cite it in the manuscript.** |
| M10 | **`is_asleep` is conditioned on HR wear**, so HR missingness partially determines a second channel's values ([data_preprocessing.py:278-296](../data_processing/data_preprocessing.py#L278-L296)) — the two channels are not independent measurements. | Report the fraction of `is_asleep` values set by the wear rule; ablate by treating unknown sleep as NaN throughout. |

### 8.2 Statistical

| # | Limitation | Control / fix |
|---|---|---|
| S1 | **$n{=}36$ held-out participants (18/class)** — intervals ~$\pm0.20$ AUROC; no realistic effect can clear the CI-excludes-zero bar. | §7.4: repeated participant holdout across the existing seeds, or 5-fold participant CV with per-fold pretraining. |
| S2 | **Cohort and optimisation variance are confounded** (one seed drives both). | Use the already-implemented crossed `--split-seed` / `--model-seed` design. |
| S3 | **Three uncorrected test families** (§7.2), including the new headline. | Holm or Benjamini–Hochberg within each family. |
| S4 | **The separability table has no interval and no test** (§5.4). | Repeat over seeds; report mean ± SD per cell and a paired test for any two rows being compared. |
| S5 | **The E1.3 headline has no confidence interval.** | Port the per-participant sufficient-statistics bootstrap from `_probe_r2` to `rhythm_axis_probe`. |
| S6 | **Kruskal–Wallis over four trajectory groups from ~36 participants is severely underpowered.** | Report group sizes and effect size; treat as exploratory and say so. |
| S7 | **$c^{*}$ is read off a noisy curve.** Correctly mitigated — bootstrapped on coherent per-draw curves with first-crossing rather than max ([experiment_q3.py:41-55](../experiment_q3.py#L41-L55)) — but note that at $n{=}36$ pointwise AUC noise is about twice the 0.05 threshold that defines $c^{*}$. | Report `frac_undefined` alongside every $c^{*}$; it is already computed. |

### 8.3 Biological / physiological

| # | Limitation | Control / fix |
|---|---|---|
| B1 | **The harmonic reference's bands are study-specific**, not standard chronobiology. $K{=}3$ harmonics of a 24 h fundamental is a modelling choice; the 8 h component has no established generator. | Sensitivity analysis over $K\in\{1,2,3\}$ and $P\in\{1,2,3\}$; report whether conclusions change. |
| B2 | **GLOBEM cannot resolve the harmonics the readout requests.** At 4 samples/day, Nyquist is 2 cycles/day; the 8 h and 6 h bins do not exist and the 12 h bin sits *at* Nyquist where phase is unidentifiable (§4.3). | Drop bins above 1 cycle/day from GLOBEM's `_seasonal_spectral`; report circadian and circaseptan only. |
| B3 | **`is_asleep` is a vendor-derived, proprietary-algorithm label**, not polysomnography; `restless` → awake is a study convention ([data_preprocessing.py:86-89](../data_processing/data_preprocessing.py#L86-L89)). | State it. Where possible, report agreement against any available reference; otherwise flag as a device-dependent construct. |
| B4 | **Per-participant z-scoring removes between-person amplitude differences** — but MESOR and amplitude are themselves clinical markers in chronobiology. The design compensates by not applying random scaling augmentation, but the normalisation still discards the between-person signal. | Ablate with `--no-zscore` (a supported flag, [train_hrd.py:437](../train_hrd.py#L437)); report whether between-person amplitude carries endpoint information. |
| B5 | **Preprocessing could manufacture the structure being reported.** Gap interpolation up to 30 min, edge nearest-fill, and window-level linear interpolation all inject smooth, low-frequency content into exactly the band the seasonal branch reads. | Report the per-window imputed fraction as a covariate; test whether recovery $R^2$ correlates with it. If it does, the "rhythm" is partly the imputer's. **This is the control a physiology reviewer will ask for and it is currently absent.** |
| B6 | **Sleep runs roughly antiphase to activity and heart rate**, so any pooled acrophase is a quantity with no chronobiological referent. Correctly handled — all claims use per-channel markers ([rhythm.py:1293-1305](../tasks/rhythm.py#L1293-L1305)) and the pooled version is explicitly display-only — but the manuscript must not slip into pooled language. | Keep per-channel throughout; state the reason once in Methods. |

### 8.4 Clinical translation

| # | Limitation | Control / fix |
|---|---|---|
| C1 | **The label's clinical meaning is undetermined** — the CES-D cutoff behind `depression_status_*` is not in the repository (§2.1). Screening-positive and diagnosed disorder are different targets with different base rates and different consequences for a false positive. | Supply the instrument, cutoff and validation reference. Non-negotiable for a clinical venue. |
| C2 | **The test cohort is balanced by construction**; deployment prevalence is not. | Already handled by prevalence transport ([_eval_protocols.py:205](../tasks/_eval_protocols.py#L205)) — but the manuscript must quote PPV at the deployment base rate, not at the implied 50%. |
| C3 | **A single cohort, single device, single geography** (HRD). | GLOBEM's leave-one-cohort-out split (§3.6) is the available external-validity test. Implement it. |
| C4 | **No comparison to a clinician or to a cheap questionnaire.** A model that does not beat a 2-minute PHQ-2 is not clinically actionable regardless of AUROC. | Add the available self-report as a baseline rung if any is collected; otherwise state the omission explicitly. |
| C5 | **No decision-analytic evaluation.** AUROC does not tell a clinician whether to deploy. | Add decision-curve analysis or a net-benefit calculation at the fixed operating points already computed ([train_hrd.py:887-905](../train_hrd.py#L887-L905)); the sensitivity/specificity pairs needed are present. |
| C6 | **Calibration is reported but not corrected.** A miscalibrated score cannot be given to a clinician as a probability. | Report the reliability curve; if ECE is large, fit an isotonic or Platt recalibration on validation only and report both. |
| C7 | **No subgroup analysis** (sex, age, device wear-time, baseline severity). Fairness and differential performance are unexamined. | Report AUROC by subgroup with participant bootstrap CIs, acknowledging that at $n{=}36$ these are descriptive only. This is a further argument for the larger evaluation of §7.4. |

---

## 9. Reproducibility notes

- **Environment is not pinned.** `requirements.txt` gives lower bounds only, and `run.sh`
  installs `--no-index` from a cluster wheelhouse whose versions come from the module. Emit
  `pip freeze` into each run directory.
- **Determinism is claimed but not enforced.** `run.sh` states that re-running the same seeds
  reproduces results bit for bit, but `init_dl_program` defaults `deterministic=False`
  ([utils.py:47](../utils.py#L47)) and `train_hrd.py:722` does not override it;
  `CUBLAS_WORKSPACE_CONFIG` is exported but `torch.use_deterministic_algorithms` is never
  called. Either enable determinism or soften the claim.
- **Failure semantics are strong and should be preserved.** Missing CosinorPy aborts by default
  rather than degrading silently ([experiment_q1.py:271-286](../experiment_q1.py#L271-L286)),
  and outcomes carry explicit `status` fields distinguishing `SKIPPED_NO_COSINORPY`, `FAILED`
  and `EMPTY_TOO_FEW_PARTICIPANTS`. **One gap:** `experiment_q3.py` skips the `Cosinor (paper)`
  and `Supervised (end-to-end)` rungs with only a `print`
  ([experiment_q3.py:173](../experiment_q3.py#L173), [:216](../experiment_q3.py#L216)), so a
  missing rung appears in the log but not in `rq3.json`. Apply the same `status` discipline.
- **Dataset caching is keyed on preprocessing-relevant config plus an explicit
  `_SCHEMA_VERSION`** ([_experiment_common.py:35-41](../tasks/_experiment_common.py#L35-L41)),
  bumped when the returned dict gains a *key*, not only when a value changes — the correct
  discipline, since none of the config keys move when a new field is added.
- **The experiment scripts never train.** `experiment_q1/q2/q3.py` reload the frozen
  `encoder.pt` and the split recorded in `metrics.json`
  ([_experiment_common.py:93-100](../tasks/_experiment_common.py#L93-L100)), so the held-out
  participants are bit-identical across RQ1–RQ3. The only exception is the plain-SSL twin,
  which is a genuine second pretraining, cached and shared between q1 and q3
  ([baselines/plain_ssl.py:38](../baselines/plain_ssl.py#L38)).

---

## Appendix A — Summary of code/document contradictions

| # | Document says | Code does | Resolution adopted here |
|---|---|---|---|
| 1 | Design doc: RQ1 headline is Full→$\sigma$ (E1.2) | `experiment_q1.py:11-16`: headline is E1.3 | **E1.3**, with the two fixes in §3.3 |
| 2 | Design doc: `persubject` is the primary probe unit | `run.sh`: `PROBE_UNIT="last"`; `experiment_q3.py`: persubject | **Unit follows label resolution** (§4.5) |
| 3 | Design doc: RQ2 reference is $W{=}28$ daily windows | `experiment_q2.py`: $R{=}4$ non-overlapping 7-day windows ($=28$ days), frozen and contiguity-checked | Code; self-documented |
| 4 | `MODEL_ARCHITECTURE.md` (now deleted): $C{=}15$, 10 sensors, SGD lr $10^{-3}$, 600 iters, $\alpha{=}5\times10^{-4}$ | 4 sensors, 7 clock features, lr $5\times10^{-4}$, 6000 iters, $\alpha{=}0.005$ | Code; mark the document superseded |
| 5 | `run.sh (DATASET=globem)`: `--holdout DS1..DS4`, `vit:none` | No `--holdout` argument; no `vit` backbone | Unimplemented (§3.6) |
| 6 | `globem_preprocessing.py:360` comment: "used by train_hrd's `--holdout`" | No such flag | Unimplemented |
| 7 | `rhythm.py:686-692`: table differs from headline "because CV vs single split" | Also a different representation (320 vs 1760 dim) | Both; the second is larger (§5.4) |
| 8 | `train_hrd.py` docstring: `--pe` families give a temporal-frame contrast | Sweep restricted to `tcn:none`/`tcn:time2vec`, clock channels off | Contrast not instantiated (§3.4) |

---

## Appendix B — Why DSSL never beat Random-init: the pretext task is solved at initialisation

### B.1 The observation this explains

Four runs on the current objective, all of which the ladder scored against a same-architecture
encoder whose weights were never updated:

| run | val InfoNCE | frozen DSSL vs Random-init |
|---|---|---|
| baseline | 0.074 | never ahead |
| `--shift-sigma 0` | 0.052 | never ahead |
| `--trend-pool mean` | hard, no gain | never ahead |
| `--moco-k 1024` | 0.05 scale-adjusted | never ahead |

A validation InfoNCE of 0.052 against a chance of $\ln(K{+}1) = \ln 4097 = 8.318$ was read as
"the objective converges". It is not evidence of convergence, and the three ablations that
targeted the augmentation, the readout and the queue size all failing to move the ladder is the
signature of a defect none of them touch.

### B.2 The measurement

The training curve conflates *can the model find the positive* with *how confident is it*. At
step 0 the MoCo queue holds `F.normalize(torch.randn(dim, K))` — random directions, not encoded
windows — so the positive scores $\approx 1$ against negatives scoring $\approx 0$ and the loss
is near zero for reasons that have nothing to do with the encoder.

The difficulty measure is therefore **top-1 retrieval with a queue of real keys, on an untrained
encoder**. A task a random encoder already solves cannot teach that encoder anything, whatever
the loss curve later shows.

Measured, 3 seeds, $K$ negatives drawn from other participants, chance $= 1/(K{+}1)$:

| pretext task | top-1 at init |
|---|---|
| **current** (both views = the same window) | **1.000 ± 0.000** |
| two different day-aligned crops, loss on the overlap | 1.000 ± 0.000 |
| per-window z-scoring | 1.000 ± 0.000 |
| positive = a different window of the **same participant** | **0.150 ± 0.029** |

`PretrainDataset.__getitem__` returns `transform(ts), transform(ts)` — the same window, the same
timesteps, twice, differing only by jitter ($\sigma{=}0.1$) and a per-channel DC offset
($\sigma{=}0.5$). That is a near-identity transform, while any two distinct 7-day windows are
far apart. Instance discrimination with a near-identity positive is solved by any non-degenerate
map, including a random projection. The gradient carries almost no information about temporal
structure, so the trained encoder stays where it started — which is exactly what the ladder
reports.

The single crop in `CoST.fit` is not a mitigation: it draws one `window_offset` and applies it
to **both** views, and it never executes anyway because `train_hrd` sets
`max_train_length = seq_len`, making the guard `x_q.size(1) > self.max_train_length` always false.

### B.3 The intervention, and what is and is not established

`--positive-pair participant` draws the second view from a different window of the same person.
The pair then shares only what is stable about that participant — circadian amplitude and phase —
and shares neither the noise nor the week, so matching it requires encoding the rhythm. It also
aligns the pretext unit with the evaluation unit, since the endpoint label is per participant.

**Established.** The task is no longer solved at initialisation (0.150 vs 1.000, chance 0.125),
and it is well-posed: hand-built circadian features (24 h and 12 h amplitude and phase — what the
model is meant to encode) retrieve the same-person window at top-1 0.360 ± 0.012 across 3 seeds,
about 3× chance. The information the objective asks for is genuinely present.

**Not established.** Whether the encoder learns it. On synthetic data at 100 iterations
($\approx$ 2 epochs) top-1 is 0.095, i.e. still chance — but a real run uses 6000 iterations, so
this is under-trained by roughly 50× and is not evidence either way. That question needs the
cluster; it could not be run on a CPU-only host with a full disk.

**Treat `--positive-pair participant` as a hypothesis with a repaired mechanism, not as a
demonstrated improvement.** The claim it earns is "the current objective is degenerate, and this
is the one tested change that makes it non-degenerate" — not "this raises AUROC".

### B.4 A negative result worth keeping

Giving the two views different day-aligned crops and taking the loss on their overlap
(TS2Vec contextual consistency, which CoST builds on) leaves top-1 at 1.000. It was implemented,
measured, and removed rather than left in as an option. Two constraints found while building it
are worth recording:

* the seasonal branch is a `BandedFourierLayer` whose weight has a fixed `num_freqs` derived from
  the training `length`, so it accepts exactly one sequence length — any crop-based augmentation
  can only touch the trend branch;
* a crop that is not a whole number of days puts the 24 h component between FFT bins, since
  `_seasonal_spectral` reads bins $[1, D, 2D, 3D, 4D]$ for a $D$-day span.
