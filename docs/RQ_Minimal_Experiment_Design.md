# Minimal Experiment Design for RQ1–RQ3

Goal of this document: for each research question, the **simplest defensible design** — one mathematical definition, one protocol, one headline metric, one negative control, one output (table/figure).

Governing rule: each RQ has **exactly one headline number**. Everything else is diagnostic.

---

## 0. Shared notation (define once, use three times)

| Symbol | Meaning |
|---|---|
| $x_{p,t}\in\mathbb{R}^{T\times C}$ | Window of participant $p$ ending on day $t$; $T=672$ bins of 15 min (7 days), $C$ sensor channels |
| $f_\theta$ | SSL encoder, **frozen** after pretraining |
| $z=f_\theta(x)\in\mathbb{R}^{T\times d}$ | Representation; $z=[z^{(T)};z^{(S)}]$ (trend / seasonal branch), $d=d_T+d_S$ |
| $v=\mathrm{pool}(z)\in\mathbb{R}^{d}$ | Window vector (mean or last) |
| $\pi$ | Temporal reference / positional encoding, $\pi\in\{$F0…F4$\}$ |
| $\mathcal{P}_{\text{tr}},\mathcal{P}_{\text{te}}$ | **Participant-disjoint** splits — always |

**Four fixed protocol rules** (state once in Methods, apply in all three RQs):

1. Test participants are excluded from **pretraining**, not just from the probe.
2. Every number = mean ± SD over $S=5$ seeds; 95% CI by bootstrap **over participants** (never over windows).
3. Probes are always **linear on frozen features** (Ridge / Logistic). Anything that needs a nonlinear probe weakens the claim "the representation encodes X".
4. **The probe row matches the label's resolution.** A participant-level label (depression endpoint) gets one row per participant; a day-level label (emotional energy) gets one row per day; a **window-level** target (the harmonic reference $\tau,\sigma$ of RQ1) gets one row per window. See §0.1 — getting this wrong is either pseudo-replication or discarded data, depending on which direction you err.

## 0.1 The probe unit, and why RQ1 does not share RQ3's

`train_hrd.py --probe-unit` decides what one probe row is. The three options are the three standard responses to clustered data, and each RQ needs a different one.

| Mode | One row = | Effective $n$ | Use for |
|---|---|---|---|
| `all` | every window, carrying its participant's label | participant count, but the fit and its penalty behave as if it were the window count | not recommended for a participant-level label |
| `last` | the participant's most recent window | participant count | sensitivity analysis (closest in time to the endpoint survey) |
| `persubject` | one row per participant, `[mean \| std]` of their window embeddings | participant count | **primary** for the depression endpoint |

For a **participant-level** label, ~26 windows per person all carry the same value, so `all` is pseudo-replication: long-record participants dominate the fit, and window-level intervals are optimistically tight. Aggregate-within-cluster (`persubject`) is the textbook remedy when the model is not a mixed-effects / GEE fit — and a logistic probe is neither. Its `std` half is not filler: within-person variability of the latent state is itself a candidate marker, and no single window can express it. Because $p$ doubles, pair it with `--probe-pca`. Report `last` alongside as a sensitivity check; if both give the same ordering of PE families, say so in one sentence and the choice stops being contestable.

**RQ1 is the exception, and it is not a close call.** The DRS target is the per-window harmonic reference $\tau,\sigma$ — it varies row by row, so there is no repeated label and nothing to pseudo-replicate. Restricting that probe to one window per participant only discards fitting data. The participant-disjoint split is what protects the analysis, and it is kept.

This was previously wired the other way and asymmetrically: `train_mask` carried the `probe_sel` restriction while `test_mask` did not, so with the default `--probe-unit last` the headline Full→σ probe was fit on ~1 window per participant and scored on all test windows. `train_hrd.py` now builds `train_mask_all` / `val_mask_all` — the same participant split without `probe_sel` — and passes those to `run_decomposition_recovery`. The window counts actually used are recorded in `decomposition_recovery.json` (`n_alpha_fit_windows`, `n_alpha_select_windows`, `n_test_windows`).

Unchanged on purpose: the person-level chronobiology probes of E1.3 (`rhythm_axis_probe`) and the depression separability table keep the restriction, because *their* targets really are one value per participant.

## 0.2 Relation to the original CoST evaluation protocol

The original repository (`salesforce/CoST`) evaluates in exactly this style, which is what licenses the design below:

* `cost.py::_eval_with_pooling` returns `torch.cat([out_t[:, -1], out_s[:, -1]], dim=-1)` — the trend and seasonal branches **concatenated**. That vector is our $V^{(F)}$.
* `tasks/forecasting.py` freezes the encoder and fits only `eval_protocols.fit_ridge(...)` — a **linear ridge probe on frozen features**.
* `tasks/_eval_protocols.py` selects the ridge penalty **on a validation split** from
  `alphas = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]`, minimizing RMSE + MAE, subsampling to `MAX_SAMPLES = 100000`.
* Metrics are MSE/MAE, reported both normalized and inverse-transformed to raw units.

Four deliberate divergences, each of which should be stated in Methods:

| Aspect | Original CoST | Here | Why |
|---|---|---|---|
| Probe target | future raw values (horizon $H$) | $\tau,\sigma$, cosinor markers, clinical labels | same machinery, different target |
| Split | temporal, single long series | **participant-disjoint**; test pids also excluded from pretraining | strictly harder; required for a cohort study |
| Pooling | `last` timestep | `mean` (+ spectral readout for the seasonal branch) | time-domain pooling collapses the seasonal branch to its DC coefficient, discarding the rhythm the branch exists to carry |
| Ridge $\lambda$ | selected on validation over 13 values | **same grid, same rule** (implemented in `tasks/decomposition.py`) | aligned |

**Implementation (done).** `_probe_r2` now sweeps `RIDGE_ALPHAS = (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)`: fit on train for every $\lambda$, score each on a held-out selection set with the CoST criterion (RMSE + MAE pooled over channels), then evaluate the test set with $\lambda^{*}$ alone. Each of the six probes selects its own $\lambda^{*}$, recorded in `decomposition_recovery.json` under `ridge_alpha` together with the grid, the selection source, and the fit/selection window counts, so the choice is reportable rather than implicit.

The selection set is the participant-disjoint `val_mask` that `train_hrd.py` already builds. When no validation participants exist, `train_hrd.py` sets `val_mask = train_mask`; selecting a penalty on the windows it was fit on would always return the smallest $\lambda$, so `_selection_split` detects that case and carves a participant-disjoint quarter out of train instead. Passing `alpha=<float>` pins a single penalty and reproduces the pre-sweep numbers.

The sweep is not 13× the cost: the $O(np^2)$ Gram is computed once and the whole grid is read off one eigendecomposition, $W(\lambda) = Q\,\mathrm{diag}(1/(\Lambda+\lambda))\,Q^{\top}\tilde F^{\top}\tilde Y$. Verified to match `make_pipeline(StandardScaler(), Ridge(alpha))` to $6.5\times10^{-15}$ relative error at matched precision.

**Watch for:** if $\lambda^{*}$ lands on the grid endpoint 1000 for the real runs, the grid is truncating and should be extended upward — a ceiling-pinned penalty means the probe wants more shrinkage than the grid offers, which itself is evidence about the representation.

Two things the original does **not** do, which are additions here and should be labeled as such:

1. It never probes the branches in isolation — always the concatenation. Hence `own` / `leak` / DIS are ours, another reason to keep DIS secondary.
2. It uses no negative controls; the random-init and time-shuffled encoders (E1.5) come from the standard probing literature, not from CoST.

Reporting note: the original reports errors in raw units (inverse-transformed MSE/MAE) so the number stays physically interpretable. The analogue here is acrophase error **in hours** (E1.3) and $\delta^{*}$ **in hours** (E2.2) — keep at least one such unit-bearing number per RQ.

---

# RQ1 — Do SSL representations encode cyclic and trend constructs?

> Simple idea: build a **closed-form reference** from the signal itself that the model never saw, then ask whether a *linear* map can reconstruct it from the representation.

## E1.1 Pseudo ground-truth reference

For each window and channel $c$, by least squares (closed form, no learning):

$$
x_c(t)\;\approx\;\underbrace{\sum_{j=0}^{P}a_{c,j}t^{j}}_{\tau_c(t)\ \text{trend}}
\;+\;\underbrace{\sum_{k=1}^{K}\bigl[b_{c,k}\cos(k\omega t)+e_{c,k}\sin(k\omega t)\bigr]}_{\sigma_c(t)\ \text{circadian}},
\qquad \omega=\frac{2\pi}{24\text{h}}
$$

with $P=3$, $K=3$. The target is neither the depression label nor a model output ⇒ independent, no leakage.

## E1.2 Headline metric: held-out linear recovery (Ridge $R^2$)

$$
\rho(V\!\to\!u)\;=\;1-\frac{\lVert u_{\text{te}}-\hat u_{\text{te}}\rVert^{2}}{\lVert u_{\text{te}}-\bar u_{\text{tr}}\rVert^{2}},
\qquad \hat u = V\hat W,\;\; \hat W=\arg\min_W\lVert u_{\text{tr}}-V_{\text{tr}}W\rVert^2+\lambda\lVert W\rVert^2
$$

Averaged over channels with variance weights $w_c=\mathrm{Var}(u_c)/\sum_{c'}\mathrm{Var}(u_{c'})$.

$\lambda$ is **never fixed by hand** — otherwise a low $R^2$ is confounded with a mis-set penalty. Fit on train for every $\lambda$ in the original CoST grid, pick

$$
\lambda^{*}=\arg\min_{\lambda}\ \Bigl[\underbrace{\sqrt{\tfrac1n\lVert \hat u_{\text{val}}(\lambda)-u_{\text{val}}\rVert^{2}}}_{\text{RMSE}}+\underbrace{\tfrac1n\lVert \hat u_{\text{val}}(\lambda)-u_{\text{val}}\rVert_{1}}_{\text{MAE}}\Bigr]
$$

on the **validation** set only, then score the test set with $\lambda^{*}$ alone (§0.2).

| Quantity | Definition | Role |
|---|---|---|
| **Full→σ** | $\rho(V^{(F)}\!\to\!\sigma)$ | **RQ1 headline number** |
| **Full→τ** | $\rho(V^{(F)}\!\to\!\tau)$ | the "trend" half of the question |
| own / leak | $\rho(V^{(S)}\!\to\!\sigma),\ \rho(V^{(T)}\!\to\!\sigma)$ and the mirror pair | diagnostic |
| DIS | $\tfrac12\bigl[(\rho^{T\to\tau}-\rho^{S\to\tau})+(\rho^{S\to\sigma}-\rho^{T\to\sigma})\bigr]$ | **secondary** — never replaces Full→σ |

## E1.3 Tether to classical chronobiology ("established constructs" half)

Per participant, compute standard parameters from the **raw signal** (cosinor + actigraphy):

$$
\hat x_c(t)=M_c+A_c\cos\!\bigl(\omega t-\phi_c\bigr),
\qquad
\mathrm{IS}=\frac{N\sum_{h=1}^{24}(\bar x_h-\bar x)^2}{24\sum_{i=1}^{N}(x_i-\bar x)^2}
$$

($M$ = MESOR, $A$ = amplitude, $\phi$ = acrophase, IS = interdaily stability; optionally IV and RA.)
Then predict them from $v$ with a linear probe:

* $A,\ \mathrm{IS},\ M$ → Ridge; report $R^2$ and Pearson $r$.
* $\phi$ is **circular** ⇒ regress the two outputs $(\cos\phi,\sin\phi)$, set $\hat\phi=\mathrm{atan2}$, and report

$$
r_{\text{circ}}=\frac{\sum\sin(\phi_i-\bar\phi)\sin(\hat\phi_i-\bar{\hat\phi})}{\sqrt{\sum\sin^2(\phi_i-\bar\phi)\sum\sin^2(\hat\phi_i-\bar{\hat\phi})}},
\qquad
\mathrm{MAE}_\phi=\mathrm{median}\bigl|\,(\phi_i-\hat\phi_i+\pi)\bmod 2\pi-\pi\,\bigr|\ \text{[hours]}
$$

Phase error **in hours** is the most reviewer-friendly number here, because it has a physical unit.

## E1.4 Effect of the temporal reference frame (second half of RQ1)

Hold everything fixed, vary only $\pi$ (families F0–F4). Effect:

$$
\Delta_\pi=\rho_\pi(\text{Full}\!\to\!\sigma)-\rho_{\text{F0}}(\text{Full}\!\to\!\sigma)
$$

Simple test: paired Wilcoxon signed-rank across the $S$ seeds (each seed = one pair), Holm correction over the number of families. CI by bootstrap over participants.

## E1.5 Two negative controls (cheap and essential — without them the $R^2$ means nothing)

| Control | Construction | Expectation |
|---|---|---|
| **Random-init encoder** | same architecture, no pretraining, frozen | Full→σ must drop substantially; if it does not, the architecture (not SSL) is doing the work |
| **Time-shuffled input** | permute the bins, $\tilde x=x_{\Pi(t)}$, keep $\sigma$ from the original $x$ | $R^2\to 0$ |

**RQ1 output:** one table (rows = variants grouped by family F0–F4; columns Full→τ, **Full→σ**, own/leak, DIS, $r_{\text{circ}}$, MAE$_\phi$) plus one bar figure of Full→σ by family with two horizontal lines for the controls.

---

# RQ2 — Can unlabeled personal baselines detect within-person rhythmic deviation?

> Simple idea: deviation = distance of today's me from **my own past** in latent space. No labels needed to *build* the score; labels are used only to *validate* it.

## E2.1 Deviation score (causal, no future leakage)

For participant $p$ and day $t$, the reference set is the previous $W$ days ($W=28$):

$$
\mu_{p,t}=\frac{1}{|\mathcal{R}_{p,t}|}\sum_{s\in\mathcal{R}_{p,t}}v_{p,s},
\qquad \mathcal{R}_{p,t}=\{t-W,\dots,t-1\}
$$

$$
d_{p,t}=\bigl\lVert v_{p,t}-\mu_{p,t}\bigr\rVert_{\Sigma_p^{-1}}
=\sqrt{(v_{p,t}-\mu_{p,t})^{\!\top}\Sigma_p^{-1}(v_{p,t}-\mu_{p,t})}
$$

$\Sigma_p$ = **diagonal** covariance over the reference set (simplest; use shrinkage if $d>|\mathcal{R}|$). An equally valid, even simpler choice: $d_{p,t}=1-\cos(v_{p,t},\mu_{p,t})$.

Robust within-person standardization (so people are comparable):

$$
s_{p,t}=\frac{d_{p,t}-\mathrm{median}_p(d)}{1.4826\cdot \mathrm{MAD}_p(d)},
\qquad \text{flag a deviation if } s_{p,t}>3
$$

## E2.2 Three-layer validation (from fully controlled to clinical)

**Layer 1 — synthetic perturbation recovery (the key RQ2 experiment, because it yields an interpretable unit).**
Inject perturbations of known magnitude into held-out windows and measure detection AUC:

| Perturbation | Formula | Levels |
|---|---|---|
| Phase shift | $x'(t)=x(t-\delta)$ | $\delta\in\{0.5,1,2,3,4\}$ h |
| Amplitude damping | $x'=\tau+\alpha\,\sigma$ | $\alpha\in\{1,0.9,0.7,0.5,0.3\}$ |
| Sleep fragmentation | $k$ random 30-min interruptions in `is_asleep` | $k\in\{0,1,2,4,8\}$ |

Headline number = **minimum detectable perturbation**:

$$
\delta^{*}=\min\{\delta:\ \mathrm{TPR}(\delta)\ge 0.80 \ \text{ at } \ \mathrm{FPR}=0.05\}
$$

This lets you write a sentence like "the representation detects phase shifts ≥ 1.5 h at 80% sensitivity" — exactly the answer to "detect within-person deviations".

**Layer 2 — convergent validity with classical constructs (still label-free).**
Compute the same deviation from the raw signal: $\Delta\phi_{p,t}=|\phi_{p,t}-\bar\phi_{p,\mathcal{R}}|$, $\Delta A$, $\Delta\mathrm{IS}$. Then correlate **within person**, aggregate across people:

$$
\rho_p=\mathrm{Spearman}\bigl(s_{p,\cdot},\ \Delta\phi_{p,\cdot}\bigr),
\qquad \text{report } \mathrm{median}_p(\rho_p)\ \text{and a Wilcoxon test against } 0
$$

**Layer 3 — behavioral / clinical tether (labels enter here).**

* Daily: within-person AUC of $s_{p,\cdot}$ against low-energy days (EE $<$ that person's own median) → report $\mathrm{median}_p \mathrm{AUC}_p$ and Wilcoxon against 0.5. Being within-person, stable individual traits cannot drive it.
* Study level: per-person summaries $\bar s_p$, $q_{95}(s_p)$, and the **slope** $\beta_p$ from $s_{p,t}=\alpha_p+\beta_p t$ → compare the four `depression_trajectory` groups (Pre1\_Post1 … Pre2\_Post2) with Kruskal–Wallis; directional hypothesis: the new-onset group (Pre1\_Post2) has the largest $\beta_p$.

## E2.3 Baselines (without them the claim is not comparative — same score, different space)

Exactly the same $s_{p,t}$, only $v$ changes:
(a) handcrafted statistics (per-channel mean/std), (b) paper cosinor parameters, (c) the downsampled raw signal, (d) random-init encoder.

**RQ2 output:** one "detection AUC vs perturbation magnitude" curve (three perturbations × representations), one table of $\delta^{*}$, $\mathrm{median}_p\rho_p$ and the within-person EE AUC, and one illustrative single-participant figure ($s_{p,t}$ over time with the reference band).

---

# RQ3 — Utility and limits

## Part A — Utility (does it improve?)

Three tasks, all with a linear probe on the frozen representation:

| Task | Unit | Metric |
|---|---|---|
| Depression endpoint | participant | AUROC |
| High-energy day (EE $\ge 4$) | day, participant-level split | AUROC / AUPRC |
| EE $\ge$ own median | day, within-person | within-person AUROC |

Baseline ladder (five rungs, increasing strength):
majority → handcrafted statistics → cosinor (paper) → random-init encoder → **frozen CoST** → end-to-end supervised.

Headline number and its test:

$$
\Delta\mathrm{AUC}=\mathrm{AUC}_{\text{SSL}}-\mathrm{AUC}_{\text{base}},\qquad
\text{95\% CI from bootstrap resampling \textbf{participants} } (B=2000)
$$

A claim is licensed only when the CI excludes zero.

## Part B — "How?" (three ablations, all inference-only — no re-pretraining)

| Level | What is removed / isolated | Question it answers |
|---|---|---|
| **Branch** | probe on $v^{(T)}$ only, $v^{(S)}$ only, and full $v$ | does the gain come from trend or rhythm? |
| **Channel** | zero out channel $c$ at the input: $\Delta_c=\mathrm{AUC}_{\text{full}}-\mathrm{AUC}_{\neg c}$ | which sensor carries the signal? |
| **Timescale** | feed the input with the circadian component removed, $x'=x-\sigma$ (and its mirror $x'=\tau+\sigma$) | how much of the utility is *rhythm*? |

Together these three complete the "how" story: which branch, which sensor, which timescale.

## Part C — Limits (degradation grid, one factor at a time from the full setting)

The encoder stays frozen; only the test-time input is degraded ⇒ very cheap.

| Factor | Values | Construction |
|---|---|---|
| **Duration** $L$ | 1, 2, 3, 7, 14, 28 days | keep the last $L$ days of the window, mask the rest |
| **Granularity** $\Delta$ | 1, 5, 15, 30, 60 min | average within $\Delta$-blocks, then resample back to $T$ (input shape stays fixed) |
| **Missingness** $m$ | 0, 10, 20, 40, 60% | two mechanisms: MCAR (random bins) and **block / non-wear** (contiguous gaps) — these *must* be reported separately |
| **Channels** | HR only / HR+Steps / all | drop channels at the input |

Headline number for "when does performance drop" — the **breakdown point**:

$$
c^{*}=\max\Bigl\{c:\ \mathrm{AUC}(c)\ \ge\ \mathrm{AUC}_{\text{full}}-0.05\Bigr\}
$$

This supports sentences like: "stable down to 3 days, 30-min granularity, and 40% MCAR missingness, but it breaks at 20% block missingness." No model fitting is needed — just the curve plus bootstrap CIs.

**RQ3 output:** one utility table (baseline ladder × 3 tasks), one 3-panel ablation figure (branch / channel / timescale), and one 4-panel degradation figure with the $-0.05$ dashed line and $c^{*}$ marked.

---

# Compute budget (why this design is "simple")

| RQ | Pretraining needed | Evaluation |
|---|---|---|
| RQ1 | $|\Pi|\times S$ (all PE families × seeds) — the main cost | linear probes, seconds |
| RQ1/RQ3 plain-SSL control | $|\Pi|\times S$ **again** — see below | trained once by `experiment_q1.py`, reloaded by `experiment_q3.py` |
| RQ2 | **none** (reuses the RQ1 encoders) | inference + synthetic perturbation |
| RQ3 | **none** for Parts B and C; the supervised baseline is trained separately | inference over the degradation grid |

Every control in this document is inference-only **except one**. The plain-SSL control — the same CoST pretrained with `disentangle=False`, `baselines/plain_ssl.py` — is a second self-supervised pretraining per (seed, variant), so enabling it doubles the sweep. It buys the one comparison nothing else can make: everything (backbone, PE, dims, augmentation, loss, iterations, pretrain windows) is held fixed and *only* the trend/seasonal split changes, which is what licenses the claim that the split itself is responsible for a difference. Its weights are cached to `plain_encoder.pt`, so the two consumers share a single training.

Reporting note: with one branch there is nothing to leak between, so the plain row carries **Full→τ / Full→σ only and DIS is undefined, not zero** — the `Ctrl` family of the RQ1 blueprint.

RQ2 and RQ3 should ideally run on the **two best RQ1 variants** only (best F0 and best clock-based). Running them on the full grid is affordable but widens the multiple-comparison surface, and the plain control doubles the pretraining cost on every variant it is enabled for (`--no-plain-ssl` disables it per script).

---

# One-line mapping: which number answers which question

| Question | Headline number | Control that protects the claim |
|---|---|---|
| RQ1 — is rhythm/trend encoded? | Full→σ and Full→τ (Ridge $R^2$) | random-init encoder, time-permuted input |
| RQ1 — does the temporal frame matter? | $\Delta_\pi$ + paired Wilcoxon across seeds | family F0 as the reference |
| RQ2 — are within-person deviations detectable? | $\delta^{*}$ (minimum detectable phase shift) | same score on handcrafted / cosinor / random-init features |
| RQ2 — is the deviation meaningful? | $\mathrm{median}_p\rho_p$ and within-person EE AUC | within-person day-label permutation |
| RQ3 — does it help? | $\Delta\mathrm{AUC}$ with participant bootstrap CI | five-rung baseline ladder |
| RQ3 — how? | ablation $\Delta$ for branch / channel / timescale | the parts should be consistent with the whole |
| RQ3 — when does it break? | $c^{*}$ per factor | MCAR vs block missingness, reported separately |
