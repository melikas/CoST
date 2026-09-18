# Pre-commitment: the split-readout experiment (third readout variant)

Written before the run. This is the **third readout variant tested** (`timestep` -> `none` -> `split`),
and that must be disclosed wherever the result is reported. No RQ, endpoint, metric, threshold,
baseline, split or criterion is redefined.

## What is compared

All three arms read out **the same trained encoders** (`narval_v2`, `--reuse-encoders`), so the only
difference between them is the readout. Reuse is sound because the readout never enters training:
`seasonal_loss` normalises independently of `readout_norm`, and the only other use
(`_log_readout_amp`) is inert at `w_ac=0`. Verified on a smoke run: encoder weights bit-identical,
trend and amplitude blocks bit-identical to the unnormalised source, phase block different.

| Arm | `readout_norm` | Amplitude from | Phase from |
|---|---|---|---|
| v1-readout (`narval_v2_timestep`) | `timestep` | normalised | normalised |
| v2 (`narval_v2`, already run) | `none` | raw | raw |
| split (`narval_v2_split`) | `split` | raw | normalised |

**`narval_v1` is NOT this comparison.** It also had `w_eq=1.0` and `eps=1e-6`, so its numbers are
historical context, not a controlled readout arm.

## Hypothesis, with a correction to its stated mechanism

Tested hypothesis: *unnormalised latent magnitude is necessary for amplitude/intensity preservation,
while timestep-normalised features give better phase estimates and may recover the phase lost in v2.*

**Correction made before running, not after.** I originally justified this by `atan2` being
ill-conditioned near zero amplitude. That is no longer the operative mechanism: raising `eps` from
1e-6 to 1e-3 already conditions the raw readout, and a test now pins this — at the measured
batch-to-batch jitter (7.5e-08) every mode moves the angle by < 0.05 rad in the realistic regime
(‖z(t)‖ of order 1, individual bins silent). A second test shows normalisation can instead *amplify*
noise when the whole vector is tiny (3.03 rad), so neither mode is unconditionally better
conditioned.

What actually differs: per-timestep normalisation divides by the **time-varying** norm ‖z(t)‖, which
is not a positive scalar, so it reshapes the spectrum. The two modes compute the phase of genuinely
different signals. Which one better preserves acrophase is therefore an **empirical** question, and
that is all this experiment tests.

## Pass/fail, frozen

Adopt the split readout **only if all** hold against `narval_v2`:

1. **RQ2 strength still supported** — above chance, above the matched untrained control, no material
   regression from v2's 0.713.
2. **RQ2 timing still supported** — no material regression from v2's 0.759.
3. **RQ1 phase improves over v2** — particularly DSSL vs untrained (v2: −0.023, favouring control),
   ideally back toward v1's +0.031.
4. **Acrophase disentanglement improves** — own vs leakage (v2: −0.083) and DSSL vs untrained
   separation (v2: −0.209).
5. **Amplitude-related RQ1/RQ2 gains preserved** — RQ1 amplitude vs untrained (v2: +0.029) and
   RQ1-D amplitude own R² (v2: 0.977).
6. **RQ3 does not enter the decision.** It is reported afterwards as a secondary consequence only.
7. **The readout is applied to every config-coupled arm** — the matched untrained encoder and the
   CoST reference are read out the same way, so no artificial DSSL advantage is created.

Any material regression in a predefined criterion, or failure to recover phase, means
**REJECT SPLIT READOUT** and keep `readout_norm="none"`.

## Selection disclosure

Three readouts will have been evaluated at protocol grade. If the split is adopted, the report must
state that it was selected among three variants on RQ1/RQ2 criteria with RQ3 excluded from the
decision, and that the selection was made after seeing v1 and v2 results. That is a real multiplicity
cost and is not erased by the criteria being frozen in advance here.
