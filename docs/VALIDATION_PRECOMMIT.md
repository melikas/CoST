# Pre-commitment for the full protocol validation run

Written and committed **before** the run is launched, so the reading of its results cannot be
adjusted afterwards. Governing criteria: `docs/SCIENTIFIC_PROTOCOL.md`. Nothing here redefines an
RQ, endpoint, metric, threshold, baseline, split or criterion.

## Frozen candidate

| Setting | Value | Basis |
|---|---|---|
| `readout_norm` | `"none"` | Mechanistic: per-timestep L2 normalisation destroys amplitude on an *untrained* encoder (0.5/1.0/1.5 -> 8.06/12.20/14.09 normalised vs 1.34/2.67/3.99 plain) |
| `w_eq` | `0` | Ablation screen: RQ1 proxy -0.0343 / -0.0136 / +0.0096 for w_eq 1.0 / 0.05 / 0, ordered in all 5 folds |
| `trend_views` | `"same"` | `disjoint_days` removed saturation but regressed RQ1 in 5/5 folds; rejected |
| everything else | as `configs/hrd.json`, `configs/globem.json` | unchanged |

No further tuning of any kind before or during the run. Seeds, folds, probes, thresholds and the
baseline ladder are those already in the protocol.

## Interpretation rules, fixed in advance

1. **RQ1** is fully supported only if DSSL beats the population mean **and** the untrained encoder in
   every eligible family. Population mean alone passing = **partially supported**, reported as such.
2. **RQ1-D** requires (i) own − leakage above 0 for MESOR, amplitude *and* acrophase, and (ii) better
   separation than untrained on all three. Failing either = not supported for that clause.
3. **RQ2** requires timing **and** strength. Timing passing while strength fails = **partially
   supported**. Timing cannot compensate for strength (protocol, "What counts as evidence").
4. **RQ3** requires DSSL above **raw** *and* above **untrained**. Anything else = **not supported**,
   regardless of how DSSL compares to CoST, Cosinor, handcrafted, PCA or random projection, which are
   context only.
5. **GLOBEM near chance** is reported as a lack of cross-cohort generalisation, not omitted. GLOBEM
   RQ2 stays "not applicable" by protocol; it is not replaced by a surrogate.
6. **Control fairness is reported with every RQ3 claim.** `untrained`, `cost_reference_adapter` and
   `pca` (its width is `min(dssl_width, n-1, raw_width)`) all move when the DSSL configuration moves.
   `raw`, `yan_cosinor`, `handcrafted`, `nonparametric`, `distribution` and `handcrafted_stack` are
   fixed anchors (seed SD 0.000). Any DSSL-vs-untrained gap is decomposed into trained improvement
   versus control degradation, and the fixed anchor `raw` is quoted alongside.
7. **Uncertainty is reported for what it covers.** The participant bootstrap is conditional on the
   fitted models and holds the fold structure fixed; it excludes seed and fold variability. Where a
   conclusion depends on a single fold, that is stated.

## Prohibited after seeing results

- Switching the primary metric, or promoting a secondary probe (forest) to primary.
- Selecting seeds, folds, channels or participants.
- Adding or re-tuning thresholds; the decision rule stays a fixed 0.5.
- Downgrading or removing a baseline because it performs well (protocol adjudication, "Exclude the
  untrained RQ1 control because it performs well: REJECT").
- Re-running with a changed configuration and reporting the better of the two runs.
- Describing a mechanistic (A) or ablation-screen (B) result as protocol-grade (C).

## Known limitations, recorded now

- **Trend saturation is unfixed** (MoCo top-1 -> 1.000). Two principled fixes failed; it is reported
  as a limitation, not presented as resolved.
- **The endpoint is largely trait-like**: label = `ces_d_endpoint_score >= 16`; baseline status
  matches it for 80.4% (90/112, 22 switchers); baseline CES-D alone gives AUROC 0.876 against
  0.49-0.73 for the entire wearable ladder. RQ3 as written asks for total prediction from wearable
  representations and is **not** redefined. Incremental-over-baseline analysis, if ever done, is
  exploratory/future work and cannot be substituted for RQ3.
- **n = 113 participants, ~22 per fold.** Per-fold AUROC swings 0.51-0.87 for every method; a prior
  learning curve put representation-side gains at roughly +0.02-0.03 AUROC per doubling of labelled
  participants. A negative RQ3 under this protocol is a plausible true result, not a defect.

## Expected outcome, stated in advance

RQ3 is expected **not** to be supported. RQ2 strength is expected to move above chance on the
strength of the mechanistic repair, and RQ1(b) and RQ1-D remain genuinely uncertain. Recording this
here so that a negative result cannot later be described as a surprise, nor a positive one as
confirmation of something already assumed.
