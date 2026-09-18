# Outcome: the split readout (third readout variant) — REJECTED

Read against the criteria frozen in `docs/READOUT_SPLIT_PRECOMMIT.md` before this ran. All three
arms read out the **same trained encoders** (`narval_v2`, via `--reuse-encoders`), so amplitude and
phase are the only things that differ between them.

| Arm | Amplitude from | Phase from |
|---|---|---|
| v1-readout (`narval_v2_timestep`) | normalised | normalised |
| v2 (`narval_v2`) | raw | raw |
| split (`narval_v2_split`) | raw | normalised |

## Comparison table (HRD)

| Metric | v1-readout | v2 | Split | Split vs v2 |
|---|---:|---:|---:|---:|
| RQ1 amplitude vs untrained | +0.035 [+0.012,+0.058] DSSL | +0.029 [+0.010,+0.049] DSSL | +0.020 [+0.004,+0.037] DSSL | −0.009, still favours DSSL |
| RQ1 phase_hours vs untrained | **+0.033 [+0.015,+0.051] DSSL** | −0.023 [−0.042,−0.005] control | −0.019 [−0.035,−0.003] control | +0.004, **still favours control** |
| RQ1 IS vs untrained | −0.040 control | −0.019 control | −0.013 [−0.030,+0.004] inconclusive | small improvement |
| RQ1 IV vs untrained | +0.039 DSSL | +0.035 DSSL | +0.040 DSSL | unchanged |
| RQ1 RA vs untrained | +0.043 inconclusive | +0.033 inconclusive | +0.013 inconclusive | unchanged (inconclusive throughout) |
| RQ1-D MESOR own−leak vs untrained | +0.015 inconclusive | +0.077 [+0.019,+0.149] DSSL | **+0.095 [+0.019,+0.165] DSSL** | improved |
| RQ1-D amplitude own R² | 0.859 | 0.977 | 0.977 | **identical** (bit-identical block) |
| RQ1-D amplitude own−leak vs untrained | −0.081 control | −0.018 control | −0.018 control | unchanged |
| RQ1-D acrophase own−leak | −0.097 inconclusive | −0.083 inconclusive | **−0.097 inconclusive** | **not improved** (back to v1's worse point estimate) |
| RQ1-D acrophase vs untrained | −0.236 inconclusive | −0.209 [−0.325,−0.004] control | **−0.236 [−0.344,+0.000] control** | **not improved** |
| RQ2 timing | 0.796, +0.296 | 0.759, +0.259 | **0.800, +0.300** | **improved** |
| RQ2 timing vs untrained | +0.072 DSSL | +0.049 DSSL | **+0.077 DSSL** | improved |
| **RQ2 strength** | 0.399, **below chance** | **0.713, +0.213 DSSL** | **0.557, +0.057** | **−0.156, collapses toward v1** |
| **RQ2 strength vs untrained** | −0.075 control | **+0.054 [+0.042,+0.066] DSSL** | **−0.081 [−0.102,−0.061] control** | **reverses from favouring DSSL to favouring control** |

RQ3 (reported after the decision, per rule 6): v2 DSSL 0.704 (vs raw +0.009, vs untrained +0.028,
both inconclusive); split DSSL 0.696 (vs raw +0.001, vs untrained +0.028, both inconclusive).
Essentially flat across all three readouts, as expected — RQ3 did not decide this and would not
have changed the verdict either way.

Matched controls, both required by the protocol, behaved consistently under the split readout
(condition 7): `untrained` strength dropped less than DSSL's (0.659→0.638, vs DSSL's 0.713→0.557),
and `cost_reference_adapter` strength also fell (0.680 in v1-readout vs 0.761 in split — its own
readout-coupling, expected since it inherits `readout_norm` via `REFERENCE_SHARED`). No artificial
DSSL-only advantage was created; the readout was applied identically everywhere it must be.

## Against the seven frozen conditions

1. **RQ2 strength remains supported** — **FAILS.** Still above chance (+0.057, CI excludes 0), but
   no longer beats untrained (−0.081, favours control) and regresses 0.713→0.557, more than half
   the v2 effect size gone. This is the single condition the whole experiment was designed to test.
2. **RQ2 timing, no material regression** — met; timing improved (0.759→0.800).
3. **RQ1 phase improves relative to v2** — **FAILS.** −0.023→−0.019 is not material, and it never
   leaves "favours control" — nowhere near recovering v1's +0.033.
4. **Acrophase disentanglement improves** — **FAILS.** Own−leak −0.083→−0.097 (worse point
   estimate); vs untrained −0.209→−0.236 (worse). Both stayed inconclusive/control-favouring
   throughout; neither moved in the intended direction.
5. **Amplitude-related gains preserved** — partially: RQ1/RQ1-D amplitude held (as guaranteed by
   the bit-identical amplitude block), but **RQ2 strength**, the main amplitude-related RQ2 result,
   did not survive — see 1.
6. **RQ3 excluded from the decision** — followed; reported above only as context.
7. **Applied fairly to untrained and the reference** — met, see above.

Two conditions pass (timing, fairness), one is partial (amplitude preserved in RQ1 but not RQ2),
and the three decisive conditions (1, 3, 4) all fail.

## Mechanistic note, not anticipated in the pre-commitment

The amplitude block is bit-identical between v2 and split (verified: `RQ1-D amplitude own R²`
0.977 = 0.977 exactly), yet **RQ2 strength still collapsed for DSSL** (0.713→0.557). RQ2's
personalised deviation score evidently uses the full representation vector, not the amplitude
block alone, so reverting only the phase block to the normalised readout contaminates the joint
distance used to detect an amplitude-only perturbation. `untrained`'s much smaller drop
(0.659→0.638) shows DSSL's representation is more entangled across blocks than the untrained
architecture's — a finding for the disentanglement story, not an artifact of this test.

## Context, not decisive for this call

GLOBEM RQ1(a) (population mean, all eligible families) is **met under v1-readout but not met under
either v2 or split** — the amplitude fix that helps HRD does not help this GLOBEM criterion, and the
split readout doesn't recover it either. Outside the scope of the seven frozen conditions above
(which govern only the v2-vs-split decision), but relevant to the wider record.

## Conclusion

**REJECT SPLIT READOUT.** It does not materially recover RQ1 phase or acrophase disentanglement —
the two things it was built to fix — and it substantially undoes v2's RQ2 strength result, which is
the strongest confirmed outcome of the whole repair programme. `readout_norm="none"` (v2) remains
the adopted configuration. No further readout variant is proposed; three have now been tested
(`timestep`, `none`, `split`) and that is disclosed as a selection over three configurations
wherever this result is reported.
