# CoST — Layer-by-Layer Architecture & Mathematics (HRD experiments)

This document is a complete, layer-by-layer specification of the CoST model **as
implemented in this repo** and **as run on the HRD wearable dataset**. Every
component, every weight matrix (with its shape), and every formula is listed in
the exact order data flows from the raw CSV to the final classification metric.
All numeric examples use the *real* configuration from
`results_hrd/62749320/`.

---

## 0. Notation and the two real configurations

| Symbol | Meaning | TCN run | Transformer run |
|--------|---------|--------:|----------------:|
| `N`  | # windows (samples) | 4389 pretrain / 778 test | same |
| `B`  | batch size | 64 | 64 |
| `T`  | time steps per window (bins) | **672** | **672** |
| `C`  | input channels (`input_dims`) | **15** | **15** |
| `Ch` | hidden width (`hidden_dims`) | **64** | **48** |
| `Co` | output/representation width (`output_dims`,`repr_dims`) | **320** | **240** |
| `d`  | component width (`component_dims = Co/2`) | **160** | **120** |
| `L`  | encoder depth (`depth`) | **10** | **6** |
| `H`  | attention heads (`n_heads`) | — | **8** |
| `dh` | head width (`Co/H`) | — | **30** |
| `kernels` | AR-expert kernel sizes | `[1,2,4,8,16,32,64,128,256]` | same |
| `K`  | MoCo queue size | 256 | 256 |
| `m`  | momentum | 0.999 | 0.999 |
| `τ`  | InfoNCE temperature (`T`) | 0.07 | 0.07 |
| `α`  | seasonal-loss weight | 5e-4 | 5e-4 |

`T=672` comes from `window_hours=168` (7 days) and `bin_minutes=15`:
`T = 168·60/15 = 672`.
`C=15` = 10 sensor channels + 5 clock features.
`kernels` from `paper_kernels(672)`: `L=⌊log2(672/2)⌋=⌊log2 336⌋=8` →
`[2^0 … 2^8]` (9 experts).

---

## 1. Data layer — raw CSV → window tensor `X ∈ ℝ^{N×T×C}`

Source: `data_preprocessing.prepare_hrd_dataset` (file `data_preprocessing.py`).

### 1.1 Channels (`C = 15`, fixed order)
**10 sensor channels** (`SENSOR_COLS`):
`HR`, `Steps`, `Floors`, `Fairly_Active`, `Lightly_Active`, `Sedentary`,
`Very_Active`, `calls`, `screen`, `sleep_level`.
**5 clock channels** (`NUM_TIME_FEATURES`): see 1.3.

### 1.2 Per-participant cleaning + z-score (leakage-free)
For each participant `p` and each sensor channel `c`:

- HR clipped to `[20,250] bpm`; activity counts clipped to `≥0`; event/sleep NaN→0.
- Short non-wear gaps (≤30 min) linearly interpolated; long gaps stay NaN → window dropped.
- Per-participant standardization:

$$ \tilde{x}^{(p)}_{c}(t) = \frac{x^{(p)}_{c}(t) - \mu^{(p)}_{c}}{\sigma^{(p)}_{c}}, \qquad \mu^{(p)}_c,\ \sigma^{(p)}_c \text{ from participant } p \text{ only.}$$

`σ=0` is replaced by 1. **Stats are never shared across participants or with labels.**

### 1.3 Binning into windows
Window length `window_minutes = 168·60 = 10080`; bin `15` min →
`target_bins = 10080/15 = 672 = T`.
Each bin = mean of the raw minute samples falling in it; within-window gaps
interpolated (linear, both directions) then `nan→0`.

### 1.4 Clock features (`_clock_time_features`), bin index `b=0…671`
With `abs_min = start_min + b·15`, `hod = (abs_min mod 1440)/60`,
`dow = (abs_min mod 10080)/1440`:

$$ f_0 = \tfrac{b}{672},\quad f_1=\sin\!\tfrac{2\pi\,hod}{24},\quad f_2=\cos\!\tfrac{2\pi\,hod}{24},\quad f_3=\sin\!\tfrac{2\pi\,dow}{7},\quad f_4=\cos\!\tfrac{2\pi\,dow}{7}. $$

The window is `concat([sensors(672×10), clock(672×5)]) = 672×15`.

**Output of the data layer:** `X ∈ ℝ^{N×672×15}`, labels `y∈{0,1}^N`
(0=control, 1=depressed-endpoint), participant ids `pids`.

### 1.5 Numeric example (one window)
`X[i]` is a `672×15` matrix. Row `t=100` (i.e. bin 100, ≈ 25 h into the week)
might be:
`[HR=0.43, Steps=-0.21, …, sleep_level=-0.5 | f0=0.149, f1=sin(2π·1.0/24)=0.259, f2=0.966, f3=…, f4=…]`.
(Sensor entries are z-scores, so ~N(0,1); clock entries are bounded in [-1,1].)

---

## 2. Augmentation layer (pretraining only) — `PretrainDataset`

Each window produces **two** stochastic views `x_q, x_k` (each `T×C`), via
`transform = jitter∘shift∘scale` with `σ=0.5`, prob `p=0.5` each:

$$\text{scale: } x \mapsto x\odot(\sigma\,\varepsilon_s + 1),\quad \varepsilon_s\sim\mathcal N(0,I_C)$$
$$\text{shift: } x \mapsto x + \sigma\,\varepsilon_h,\quad \varepsilon_h\sim\mathcal N(0,I_C)$$
$$\text{jitter: } x \mapsto x + \sigma\,\varepsilon_j,\quad \varepsilon_j\sim\mathcal N(0,I_{T\times C})$$

(`scale`/`shift` draw one number per channel; `jitter` one per element.)
A batch is `x_q, x_k ∈ ℝ^{64×672×15}`.

---

## 3. Encoder `CoSTEncoder` (file `models/encoder.py`)

Shared front-end → backbone (TCN **or** Transformer) → two disentanglers
(Trend = TFD, Seasonal = SFD). Returns `(trend, season)`, each `B×T×d`.

### 3.1 Input projection (`input_fc`)
`Linear(C → Ch)`: weight `W_in ∈ ℝ^{Ch×15}`, bias `b_in ∈ ℝ^{Ch}`.

$$ h(t) = W_{in}\,x(t) + b_{in},\qquad H \in ℝ^{B×T×Ch}. $$

- TCN: `15→64` → `64×672×64`.  Transformer: `15→48` → `64×672×48`.

### 3.2 NaN handling + masking
NaN rows → 0, giving the non-wear mask `N∈{0,1}^{B×T}` (`N[b,t]=0` where any
channel is NaN). The applied mask is `M∧N`, and `H[b,t,:]←0` wherever it is 0.
Masking is **time-wise**, not channel-wise.

`M` is selected by `--mask-mode`, and the default is **`none`** (`M≡1`), so in
the default configuration **only the non-wear mask `N` is applied — there is no
masking augmentation**. This matches upstream salesforce/CoST, whose encoder
hard-defaulted its `mask` argument to `'all_true'` while the training loop never
passed one: its `mask_mode='binomial'` was unreachable, so no published CoST
number was produced with a masking augmentation. Every result in this repo up to
and including run `19229424` was likewise produced with `M≡1`.

`--mask-mode binomial` opts in to the TS2Vec-style mask (`M[b,t]∼Bernoulli(p)`
per timestep, `p=--mask-keep-prob`, default `0.5`); `continuous` opts in to
contiguous span dropout. Both apply **during training only** — at eval
`mask=None` resolves to `none`. Enabling either changes SSL results and is not
free here: `M` multiplies on top of the real non-wear gaps in `N`, and the SFD
branch (§ seasonal) contrasts the rFFT **amplitude and phase** of the
representation, which random timestep dropout perturbs directly.

### 3.3 (optional) Time2Vec — fed as an input feature (Kazemi et al. 2019)
Only if `pe='time2vec'` (the reported TCN run uses `pe='none'`, so this is
skipped). Faithful to the paper, Time2Vec is **fed as input** (not added):
`t2v(τ)∈ℝ^{T×m}` is **concatenated** to the sensor channels *before* `input_fc`,
so `x' = [x ‖ t2v(τ)]` and `input_fc: ℝ^{n_sensor+m} → ℝ^{Ch}`. With `τ=t`
(0…T-1, time-from-start), `v=τ·w+b`, `t2v[:,0]=v[:,0]` (linear),
`t2v[:,1:]=sin(v[:,1:])`, `w,b∈ℝ^{m}` learnable, `m=time2vec_dim` (=k+1,
default 64). Works with either backbone; the Transformer then uses **vanilla
attention** (no extra PE) since Time2Vec carries the time signal as input.

Then transpose to **channels-first** `H ∈ ℝ^{B×Ch×T}` for the backbone.

---

## 3A. Backbone option A — TCN (`DilatedConvEncoder`, file `models/dilated_conv.py`)

`channels = [Ch]*L + [Co] = [64,64,…,64, 320]` → **11 `ConvBlock`s**, block `i=0…10`.
Block `i`: `in = 64` (or `Ch` at i=0), `out = 64` (i≤9) / `320` (i=10, `final`),
`kernel=3`, `dilation = 2^i` (1,2,4,…,512,1024).

### 3A.1 `SamePadConv` (causal-length-preserving dilated conv)
receptive field `R=(k-1)·d+1 = 2·2^i+1`, padding `R//2 = 2^i`. `R` is odd → no
trailing crop. For input `u∈ℝ^{in×T}`, output channel `o`, position `t`:

$$ y[o,t] = b[o] + \sum_{c=0}^{in-1}\sum_{j=0}^{2} W[o,c,j]\;u\!\left[c,\; t + 2^i(j-1)\right],\qquad W\in ℝ^{out×in×3}. $$

Output length stays `T=672`.

### 3A.2 `ConvBlock` (residual, pre-activation, two convs)
With `g=GELU`:

$$ \text{res} = \begin{cases}u & \text{in=out, not final}\\ W_{1\times1}u & \text{else (1×1 conv } ℝ^{out×in×1})\end{cases} $$
$$ \text{Block}(u) = \text{SamePadConv}_2\big(g(\text{SamePadConv}_1(g(u)))\big) + \text{res}. $$

`GELU(z)=z·Φ(z)` (Φ = standard-normal CDF). Conv1, conv2 share `(k=3, d=2^i)`;
conv1 maps `in→out`, conv2 `out→out`. Only block 10 has a `1×1` projector
(64→320) because `in≠out`.

### 3A.3 Stacked dilation = exponential receptive field
Block `i` covers `±2^i` per conv; total receptive field after 11 blocks
≈ `2·Σ_{i=0}^{10} 2·2^i ≈ 2·2·(2^{11}-1) ≈ 8188 ≫ 672`, so the top layer sees the
whole week.

**TCN output:** `Z ∈ ℝ^{B×Co×T} = 64×320×672`.

---

## 3B. Backbone option B — Transformer (`TransformerFeatureExtractor`)

Input `H∈ℝ^{B×Ch×T}=64×48×672` → transpose → `B×T×Ch`.

### 3B.1 Token projection + positional encoding
`input_proj = Linear(Ch→Co)` (`48→240`): `E = H W_p^⊤ + b_p`, `E∈ℝ^{B×T×240}`.
Add absolute PE then dropout(0.1):

**Sinusoidal** (the reported run, `add_absolute_pe`):

$$ PE[t,2i]=\sin\!\Big(\tfrac{t}{10000^{2i/240}}\Big),\quad PE[t,2i+1]=\cos\!\Big(\tfrac{t}{10000^{2i/240}}\Big),\qquad E \leftarrow E + PE. $$

Other supported PEs (selectable via `--pe`): absolute family
`{learnable, tape}` added here; attention family
`{rpe, erpe, tupe, convspe, tpe}` injected inside attention (3B.2). `tape`
scales the sinusoid by `d/T`; `learnable` is an `Embedding(2048,240)`.
`time2vec` is **not** added here — it is fed as an input feature upstream
(concatenated before `input_fc`, §3.3), so with `--pe time2vec` the Transformer
runs vanilla attention with no absolute PE.

### 3B.2 `PETransformerEncoderLayer` ×6 (pre-norm)
Each layer, with `LN`=LayerNorm:

$$ x \leftarrow x + \text{Attn}(\text{LN}(x)),\qquad x \leftarrow x + \text{FF}(\text{LN}(x)). $$

**Multi-head self-attention** (`PESelfAttention`, `H=8`, `dh=30`):
`Q=xW_Q, K=xW_K, V=xW_V` with `W_Q,W_K,W_V∈ℝ^{240×240}`, reshaped to `B×8×T×30`.
For the absolute/sinusoidal case (vanilla scaled-dot-product):

$$ A = \mathrm{softmax}\!\Big(\tfrac{QK^\top}{\sqrt{30}}\Big),\qquad O = A\,V,\qquad \text{out}=O W_O,\ W_O\in ℝ^{240×240}. $$

Attention-PE variants modify the score matrix before softmax, e.g.
- `rpe`: `+ Q·R` with relative-key table `R∈ℝ^{(2T-1)×30}`;
- `erpe`: `+ bias[h, i−j]`, `bias∈ℝ^{8×(2T-1)}`, added **after** softmax;
- `tupe`: `(QKᵀ + (P W_{pq})(P W_{pk})ᵀ)/√(2·30)`;
- `convspe`: stochastic positional kernel from depthwise Conv1d over noise.
  **Training** uses the paper's Monte-Carlo estimate over `R=16` fresh noise
  realizations (the resampling is the method). **Eval** substitutes that
  estimator's exact expectation `Cq Ckᵀ` — its `R→∞` limit — so evaluation is
  deterministic and unbiased; see `PESelfAttention._convspe_pos`;
- `tpe`: `+ exp(−‖x_i−x_j‖²/2σ²)` Gaussian bias, `σ=exp(logσ)`.

**Feed-forward** (`ff_mult=4`):

$$ \text{FF}(x)=\big(\,\mathrm{GELU}(xW_1+b_1)\,\big)W_2+b_2,\quad W_1\in ℝ^{240×960},\ W_2\in ℝ^{960×240}. $$

After 6 layers: final `LayerNorm(240)`, transpose → `Z∈ℝ^{B×Co×T}=64×240×672`.

---

## 4. Trend disentangler **TFD** (`self.tfd`) — mixture of causal AR experts

`Co×T → d×T` via `len(kernels)=9` causal Conv1d experts. Expert `r` uses kernel
`k_r∈{1,2,4,…,256}`: `Conv1d(Co→d, k_r, padding=k_r−1)`, then crop the last
`k_r−1` steps → length `T` (strictly causal / backward-looking):

$$ \text{Trend}_r[o,t] = b_r[o] + \sum_{c=0}^{Co-1}\sum_{j=0}^{k_r-1} W_r[o,c,j]\; Z[c,\,t-j],\qquad W_r\in ℝ^{d×Co×k_r}. $$

Average the experts:

$$ \boxed{\;\text{Trend}[b,t,:] = \frac{1}{9}\sum_{r=1}^{9}\text{Trend}_r[b,t,:]\;}\qquad \text{Trend}\in ℝ^{B×T×d}=64×672×160. $$

Intuition: each expert is a learned auto-regressive moving-average over `k_r`
past steps; the mixture spans scales 1…256 bins (15 min … 64 h).

---

## 5. Seasonal disentangler **SFD** (`BandedFourierLayer`) — learned spectral filter

One band over the full spectrum. `total_freqs = T//2+1 = 337`. Per-frequency
**complex** weight `W∈ℂ^{337×Co×d}` and bias `b∈ℂ^{337×d}`.

1. Real FFT over time: `Z_f = rFFT(Z, axis=t) ∈ ℂ^{B×337×Co}`.
2. Per-frequency complex linear map (`einsum 'bti,tio->bto'`):

$$ \widehat{S}[b,f,:] = Z_f[b,f,:]\;W[f] + b[f],\qquad f=0,\dots,336,\quad \widehat S\in ℂ^{B×337×d}. $$

3. Inverse real FFT back to time: `S = irFFT(Ŝ, n=672, axis=t) ∈ ℝ^{B×672×d}`.
4. Dropout(0.1): `Season = Dropout(S)`.

Each weight is a learned complex gain+phase per frequency — a fully learnable
band-pass filter bank.

**Encoder output:** `Trend, Season ∈ ℝ^{B×672×160}` (TCN) / `…×120` (Transformer).

---

## 6. Downstream representation (`encode` / `_eval_with_pooling`)

At inference (`encode` puts the net in eval mode, so no masking augmentation is
applied regardless of `--mask-mode`; only the non-wear mask `N` of §3.2 acts),
take the **last timestep** of each component and concatenate:

$$ z = \big[\,\text{Trend}[:,T{-}1,:]\ \Vert\ \text{Season}[:,T{-}1,:]\,\big]\in ℝ^{B×Co}. $$

TCN: `160+160=320`. Transformer: `120+120=240`. Per window → vector `z∈ℝ^{Co}`.

This `z` is the input to the classifier (§8).

---

## 7. Self-supervised pretraining loss (`CoSTModel.forward`)

Total loss is **temporal (trend) MoCo loss + α · seasonal frequency loss**.

### 7.1 Projection heads
`head_q, head_k = Linear(d→d)→ReLU→Linear(d→d)` (`d=160` TCN / `120` Transf.).
`encoder_k`, `head_k` are momentum copies (no gradient).

> **The momentum encoder is used by the trend branch only.** `encoder_k`/`head_k`
> and the queue feed §7.2; the seasonal branch (§7.3) encodes **both** views with
> the trainable `encoder_q` and has no queue. Only §7.2 is MoCo — the model as a
> whole is *not* a symmetric two-branch MoCo, and describing it that way is wrong.

### 7.2 Trend / temporal loss — MoCo InfoNCE
Pick random time index `r`. From the two views:

$$ q = \text{normalize}\big(\text{head}_q(\text{Trend}^{q}[:,r,:])\big),\quad k = \text{normalize}\big(\text{head}_k(\text{Trend}^{k}[:,r,:])\big),\quad q,k\in ℝ^{B×d}. $$

With queue `Q∈ℝ^{d×K}` (`K=256`), positives `ℓ^+_n=q_n^\top k_n`, negatives
`ℓ^-_{n,:}=q_n^\top Q`:

$$ \mathcal L_{\text{temp}} = \frac{1}{B}\sum_{n=1}^{B} -\log \frac{\exp(q_n^\top k_n/\tau)}{\exp(q_n^\top k_n/\tau)+\sum_{j=1}^{K}\exp(q_n^\top Q_{:,j}/\tau)},\quad \tau=0.07. $$

Then `k` is enqueued (FIFO, size 256). Key encoder EMA update:
`θ_k ← m·θ_k + (1−m)·θ_q`, `m=0.999`.

### 7.3 Seasonal / frequency loss
**Both views are encoded by `encoder_q`** — `q_s` from `x_q` and `k_s` from `x_k`
(`_, k_s = self.encoder_q(x_k)`, a *third* encoder pass per step). The superscripts
`q`/`k` below denote the two **augmented views**, not the two encoders: there is no
momentum encoder and no queue here, and gradients flow through *both* sides.

This is intentional and matches `salesforce/CoST` upstream. `L_inst` is a
within-batch instance-discrimination loss, symmetric in its two arguments; the EMA
encoder exists in MoCo only to keep stale *queued* keys consistent, and with no
queue there is nothing to stabilise — routing one side through a frozen EMA copy
would instead zero out half of the symmetric objective's gradient path.

Normalize seasonal outputs of both views, FFT over time, split into amplitude and
phase:

$$ A,\;\phi:\quad \text{amp}=\sqrt{(\Re+\epsilon)^2+(\Im+\epsilon)^2},\quad \text{phase}=\operatorname{atan2}(\Im,\ \Re+\epsilon). $$

The phase is then embedded on the **unit circle**, `φ ↦ [\sin φ ; \cos φ]`
(`--phase-encoding circular`, the default), optionally weighted per channel by
that channel's amplitude (`--phase-encoding circular_amp`, see below). This is required because the
contrastive loss below scores pairs with a **dot product**, and on a raw `atan2`
angle that is not a similarity between angles at all: `⟨φ_i,φ_j⟩` depends on
*where* the angles sit, not how far apart they are. Two **identical** phases
score `0` at `φ=0` but `π²` at `φ=π`, and the pair `(π−ε, −π+ε)` — the same
angle either side of the branch cut — scores the most negative value possible.
With `C=160` channels ~64% of near-identical view pairs have at least one
channel straddling the cut. After the embedding the dot product becomes
`Σ_c cos(φ^q_c − φ^k_c)`: a function of the angular difference alone,
`2π`-periodic and monotone in the gap. `--phase-encoding raw` restores upstream
CoST's uncorrected angle, for reproducing archived runs only.

**`circular_amp`** additionally weights each channel by its own amplitude,
`φ_c ↦ w_c·[\sin φ_c ; \cos φ_c]`, so the score becomes an amplitude-weighted
phase coherence `Σ_c w^q_c w^k_c \cos(φ^q_c − φ^k_c)`. Motivation: unweighted,
every channel carries unit norm, so a channel with amplitude ≈ 0 — where the
phase is undefined noise — counts as much as a strong rhythm. The weight is
**RMS-normalised** (`mean(w²)=1`), not L2-normalised: the loss applies
`log_softmax` to these dot products with no temperature, so the embedding norm
*is* the effective temperature. L2 would collapse `‖emb‖` from `√C` to `1` — a
`√160 ≈ 12.6×` logit shrink that flattens the softmax and starves the phase
branch of gradient. RMS keeps `‖emb‖=√C` exactly, so the mode changes only the
*relative* channel weighting and reduces to `circular` bit-for-bit when the
amplitudes are equal. The weight is **detached**: amplitude is already trained
by its own contrastive term, and letting the phase loss push on it too would let
the model lower that loss by shrinking amplitudes instead of aligning phases.

Apply the **instance-wise contrastive loss** (TS2Vec style) separately to
amplitude and phase. For stacked views `z=[z_1;z_2]∈ℝ^{2B×T×·}`, per timestep
similarity `sim=z z^\top`, the loss pulls the two views of the same window
together against all other instances:

$$ \mathcal L_{\text{inst}}(z_1,z_2)=\frac{1}{2BT}\!\sum_{t}\Big(\text{xent}(t,\text{view}_1)+\text{xent}(t,\text{view}_2)\Big),\qquad \mathcal L_{\text{seas}}=\tfrac12\big(\mathcal L_{\text{inst}}(A^q,A^k)+\mathcal L_{\text{inst}}(\phi^q,\phi^k)\big). $$

### 7.4 Total
$$ \boxed{\;\mathcal L = \mathcal L_{\text{temp}} + \alpha\,\mathcal L_{\text{seas}},\qquad \alpha=5\times10^{-4}.\;} $$

Optimizer: SGD, lr 1e-3, momentum 0.9, weight-decay 1e-4, cosine decay; 600
iters; bf16 autocast.

---

## 8. Classification head (downstream, `train_hrd.py`)

Frozen encoder → `z∈ℝ^{N×Co}`. Then a scikit-learn pipeline:

$$ \hat z = \frac{z-\mu}{\sigma}\ (\text{StandardScaler}),\qquad p = \sigma\!\big(\hat z\,w + b\big)\ (\text{LogisticRegression, balanced, } \ell_2). $$

- Window-level prob `p∈[0,1]`; threshold `θ` tuned for best F1 on a
  participant-level validation split.
- Participant-level prob = **mean** of its windows' `p`.
- Metrics: AUROC, F1, Accuracy at both levels.

Real test numbers (`62749320`, seed 42): TCN/`none` participant AUC **0.591**,
Transformer/`sinusoidal` participant AUC **0.653** (27 test participants, 778
test windows).

---

## 9. End-to-end shape trace (one pretraining batch, TCN config)

| # | Stage | Operation | Output shape |
|--:|-------|-----------|--------------|
| 0 | data | window tensor | `64×672×15` |
| 1 | aug | two views `x_q,x_k` | `64×672×15` each |
| 2 | `input_fc` | Linear 15→64 | `64×672×64` |
| 3 | mask | binomial time-mask | `64×672×64` |
| 4 | transpose | to channels-first | `64×64×672` |
| 5 | TCN | 11 dilated ConvBlocks 64→…→320 | `64×320×672` |
| 6 | TFD | 9 causal AR experts, mean | `64×672×160` |
| 7a | SFD rFFT | rFFT over time (`T=672→337`) | `64×337×160` (complex) |
| 7b | SFD | complex linear `Co→d` → irFFT (`337→672`) | `64×672×160` |
| 8 | repr | last-step concat `[trend‖season]` | `64×320` |
| 9 | head | Linear 160→160→160 (+L2 norm) | `64×160` |
|9b | seas. key | `x_k` re-encoded by **`encoder_q`** (not `encoder_k`) | `64×672×160` |
|10a| seas. loss | rFFT season → amp, phase (`T=672→337`) | `64×337×160` each |
|10b| loss | MoCo InfoNCE (trend) + α·seasonal (amp/phase) | scalar |

Transformer config differs only in widths (the freq dim stays `337`): step 2
`15→48`, step 5 `TransformerFeatureExtractor 48→240` (`64×240×672`), steps 6–7b
use `d=120` so `64×672×120` (with `64×337×120` complex at 7a), step 8 `64×240`,
step 9 `120→120`, step 10a amp/phase `64×337×120` each.

---

## 10. Parameter inventory (the matrices to draw)

**Shared front-end:** `W_in (Ch×15), b_in (Ch)`.

**TCN backbone (per block i, two convs):**
`W1 (out×in×3), b1`, `W2 (out×out×3), b2`, and for block 10 a `1×1` projector
`(320×64×1)`. 11 blocks.

**Transformer backbone (per layer ×6):** `W_Q,W_K,W_V,W_O (240×240)`;
`LN1,LN2 (γ,β ∈ ℝ^{240})`; `W1 (240×960), W2 (960×240)`. Plus shared
`input_proj (240×48)` and final `LayerNorm(240)`. PE-specific extras as in §3B.2.

**TFD:** 9 real convs `W_r (160×320×k_r), b_r (160)`, `k_r∈{1,2,…,256}`.

**SFD:** complex `W (337×320×160)`, complex `b (337×160)`.

**Heads (pretraining only):** `head_q/head_k`: `(160×160)`×2 + biases.
**MoCo buffers:** queue `(160×256)`, pointer `(1,)`.

**Classifier (downstream):** StandardScaler `(μ,σ ∈ ℝ^{320})`,
LogisticRegression `w∈ℝ^{320}, b∈ℝ`.

---

### Figure-drawing summary (suggested boxes, left→right)

`X (N×672×15)` → **InputFC** `15→Ch` → **Mask** →
**Backbone** {TCN: 11×DilatedConvBlock | Transformer: PE + 6×EncoderLayer} `Ch→Co` →
split into **TFD** (9 causal AR-conv experts → mean, `Co→d`) and
**SFD** (rFFT → per-freq complex linear `Co→d` → irFFT) →
**[Trend ‖ Season]** last step → `z (Co)` →
{pretrain: **proj head** + **MoCo/InfoNCE** (trend) and **freq instance-contrastive** (season) | downstream: **StandardScaler → LogReg → p**}.
