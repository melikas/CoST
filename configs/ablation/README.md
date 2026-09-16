# Minimal-repair ablations

One mechanism changed at a time from `configs/hrd.json`. Run with `slurm/ablation.sbatch` /
`slurm/ablation_eq.sbatch` (`--skip-reference`: the mechanistic criteria only need DSSL vs its
own untrained control, not a second full CoST-reference training) and scored with
`scripts/ablation_report.py <arm> [<arm> ...]`.

## Batch A — the three originally diagnosed failures (2026-09-16, job 3183155, HRD seed 1 folds 0-1)

| Arm | Changed | Verdict |
|---|---|---|
| `hrd_a0_current.json` | nothing (control) | top-1 1.000 (saturated); RQ1 gain −0.0068; intensity below 0.5 and falling |
| `hrd_a1_trend.json` | `mask_mode=binomial`, `moco_k=256` (upstream CoST defaults) | **REJECTED.** top-1 0.992 — not meaningfully less saturated. Also has a real, unpredicted cost: Steps' own-branch acrophase R² collapses 0.765→0.0 even though nothing here touches the phase readout or loss — the shared TCN backbone is the only plausible path, since trend and seasonal are both just readouts of one `feature_extractor` output |
| `hrd_a2_amplitude.json` | `readout_norm=none` | **ADOPTED** as the new default (`configs/hrd.json`). Fixes RQ2 intensity cleanly (0.58→0.697, rising, above chance). Confirmed to be a pure readout-time change (training is identical to a0 — `seasonal_loss` always normalises regardless of this flag), yet on the *same trained weights* it also collapses Steps' own R² 0.765→0.0. RQ1 proxy got slightly worse than a0 (−0.0194 vs −0.0068) on this 2-fold slice. Kept anyway: the RQ2 fix is unambiguous and mechanism-proven; the RQ1/Steps trade-offs are separate open problems, not caused by training |
| `hrd_a3_phase.json` | `phase_mode=circular` (remove amplitude weighting) | **REJECTED.** Did not recover Steps/screen phase (still 0.0/0.0); worst RQ1 proxy of any arm (−0.0595, 0/5 families improved) |
| `hrd_a4_combined.json` | all three | Inherits a1's and a3's failures; not adopted |

Kept for the record, not deleted: they are the evidence for what does and doesn't work. See
memory `ablation-a0-a4-hrd-seed1.md` for the full numbers and the diagnostic that ruled out
a probe bug (predictions have real variance; the R²=0.0 values are genuine, not floored noise).

## Batch B — loss/gradient-weight balance (2026-09-16, on top of the adopted `readout_norm=none`)

Diagnosed cause: at HRD initialisation the weighted objective contributes trend 0.3%, amplitude
27.3%, phase 27.6%, **equivariance 44.9%**. Trend and seasonal go through `total_loss`'s single
`alpha=0.005`, which is what tames amplitude's huge raw scale (≈193) down to a comparable share.
`w_eq` is added *outside* `total_loss` with no such scaling (`total = total_loss(...) + w_eq *
L_eq`), so at `w_eq=1.0` it ends up the largest single term despite its raw value (≈0.235) being
smaller than phase's (≈34). Two single-scalar probes of the same hypothesis:

| Arm | `w_eq` | Rationale |
|---|---|---|
| `hrd_b0_readout_fix.json` | 1.0 (= `hrd_a2_amplitude.json`, reuse its results, no rerun) | current default, baseline for this batch |
| `hrd_b1_eq_peer.json` | 0.277 | equal to `w_trend` -- peer to the term it regularises, not allowed to outweigh it |
| `hrd_b2_eq_small.json` | 0.05 | clearly subordinate to amp+phase's combined ~55% share |

Monitored: RQ1 gain vs untrained, RQ2 timing and intensity, phase own/leak for Steps and screen,
MESOR own/leak. Everything here inherits `readout_norm=none` and its open Steps-phase issue;
this batch is not expected to change that on its own, since `w_eq` never touches the readout.
