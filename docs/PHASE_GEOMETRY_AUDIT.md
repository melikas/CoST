# Circular phase geometry: verified defect, partial fix, and what remains

Date 2026-09-20. Code version at audit: `92cd7dc`. This document records only what was
measured or changed. It does not restate the McGrath audit's claims as findings; each claim
below was re-checked against the code and, where possible, against real data.

## 1. The claim, and whether it holds

`docs/MCGRATH_COMMENTS_AUDIT_2026-09-20.md` item 2 states that acrophase is treated
circularly in the loss and in the RQ1 targets, but linearly in the downstream paths.
**Verified true.** The canonical configuration sets `phase_readout="angle"`, so 800 of the
1760 HRD embedding columns are raw angles in $(-\pi, \pi]$, and:

| Path | Code | Operation on raw angles | Correct? |
|---|---|---|---|
| Participant aggregation | `evaluation_protocol.participant_mean` | arithmetic mean over windows | **no** |
| RQ2 personal baseline | `tasks.rhythm.personal_baseline` | arithmetic mean and SD of 4 preceding weeks | **no** |
| RQ2 deviation | `tasks.rhythm.dscore` | `(V - mu) / sd`, linear subtraction | **no** |
| Probe scaling | `IsotropicPairScaler` | circle-preserving, but only engages for `phase_readout="circular"` | correct, inactive |
| RQ1 phase targets | `individual_markers`, circular error in hours | `(cos, sin)` | correct |
| Seasonal loss | `phase_mode="circular_amp"` | unit circle, amplitude-weighted | correct |
| RQ2 ground truth | `cosinor_z` complex mean | complex resultant | correct |

The machinery to do this correctly already exists; the canonical readout simply does not use
it. Note that the RQ2 **ground truth** ordering is computed from the complex cosinor
coefficient and is therefore unaffected: the defect degrades the model's own score, not the
target it is scored against.

## 2. Magnitude, measured on real HRD data

Per participant and channel, the arithmetic mean of that participant's window acrophases was
compared with the circular mean, over all 3,803 windows of the 113 labelled participants
(452 participant x channel summaries).

| Statistic | Value |
|---|---:|
| Median absolute error | 0.003 h |
| Mean | 0.266 h |
| 90th percentile | 0.606 h |
| Maximum | **8.91 h** |
| Summaries off by > 0.5 h | 51 (11.3%) |
| Summaries off by > 3 h | 17 (3.8%) |

The distribution is heavy-tailed rather than uniformly bad: most participants are unaffected,
while a minority is corrupted almost maximally (the theoretical worst case is 12 h). 7.5% of
summaries have a true circular mean within 25% of the wrap, which is where the failure
concentrates.

**This matters beyond the average.** The affected summaries are not a random subset: they are
the ones whose acrophase sits near midnight, and the worst four are all the `screen` channel,
whose usage peaks late. Phase delay is precisely the phenotype a depression study is looking
for, so the defect preferentially corrupts the clinically interesting subgroup. No claim is
made here about the direction or size of any effect on a reported result; that requires a
controlled run (Section 4).

## 3. What was changed

Minimal and behaviour-preserving for the canonical configuration.

- `tasks.rhythm.dscore` takes an optional `pair=(start, width)`. Given it, each `(cos, sin)`
  column pair is divided by one shared scale, the RMS of its two standard deviations, instead
  of by two different per-column SDs. Per-column scaling turns the unit circle into an
  ellipse and the distance stops being monotone in the angular gap. This mirrors
  `IsotropicPairScaler`, which already did the same thing for the probe; without it the two
  consumers of the same representation would disagree about its geometry.
- `tasks.personalized` passes the model's `pair_block()` into every `dscore` call, and
  resolves to `None` for objects that have no such layout, such as the `RawProjection`
  control, for which no circular structure exists.

With `phase_readout="angle"`, `pair` is `None` and every number is bit-identical to before;
verified directly. **No historical result is invalidated by this change.**

Four regression tests (`tests/test_repairs.py::CircularPhaseGeometry`) pin the behaviour:
the linear mean of two angles straddling the wrap is ~12 h wrong; `participant_mean` on a
`(cos, sin)` readout recovers the circular mean exactly; `dscore` is pair-aware only when
told; and the RQ2 personal-baseline distance is invariant to rotating a participant's whole
history, which it must be, since a change of origin is not a change in within-person
deviation.

## 4. What this does NOT do, and what remains

The fix makes the circular path *correct when used*. It does not switch the canonical
configuration to use it, and switching is not a free correction:

- `phase_readout="circular"` changes the embedding from 1760 to 2560 columns on HRD. That is
  a different representation, not a repaired one, so results are not comparable to the
  existing run by inspection.
- Under a `(cos, sin)` readout the linear operations above become the correct circular ones
  automatically, because the linear mean of unit vectors is the resultant whose angle is the
  circular mean. That is the argument for it; it is not evidence that it scores better.

Deciding it therefore requires a controlled run in which the readout is the only variable,
under the existing protocol, reported for RQ1, RQ2 and RQ3 separately. That run has not been
performed. Per the project's standing rule, the configuration is not changed on the strength
of an argument, and nothing here should be read as a performance claim.

Remaining phase-related gaps not addressed here, in the order they affect interpretation:

1. Near-zero amplitude. The phase of a coefficient with no amplitude is undefined; `eps=1e-3`
   currently makes such a coordinate report a stable but meaningless angle. A reliability
   policy is needed that is defined without reference to test labels.
2. The Yan cosinor baseline aggregates phase circularly and then converts back to an angle
   before entering an ordinary probe, so it re-enters a linear pipeline.
3. Group-level descriptive reporting of acrophase still needs a circular mean and dispersion
   rather than separate tests on `phase_cos` and `phase_sin`.
