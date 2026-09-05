"""RQ2 -- Can an unlabelled personal baseline detect a WITHIN-PERSON RHYTHMIC deviation?

Runs on a FINISHED train_hrd.py variant directory: loads the frozen encoder, never trains.

ONE protocol, ONE metric. The question is not "did the input change" -- every representation
notices that -- but "when the perturbation genuinely made the window MORE rhythmically
deviant from its own past, did the personalised distance rise more than when it made it LESS
deviant". So:

  d      the project's existing personalised distance, UNCHANGED (standardised Euclidean to
         the mean of the R preceding windows). Baseline built on clean windows and FROZEN.
  dd     Delta d = d(x') - d(x), the same window against the same frozen baseline.
  dg     Delta g, the same difference computed on the RAW signal in 24-h cosinor space.
         This -- not "was perturbed" -- is the ground truth. A window with dg < 0 was pushed
         CLOSER to its own rhythm and is a NEGATIVE, not a missed positive.
  C      stratified Mann-Whitney concordance: within one (participant, shift level), does dd
         rank the dg > 0 windows above the dg < 0 ones. Null is exactly 0.5.

Why the previous design was replaced (measured, not argued -- see the verification in the
commit message): it labelled every perturbed window positive, so a representation that only
senses "the input moved" scored a PERFECT 1.000 -- above cosinor. Under C the same detector
scores 0.501, and a phase-blind amplitude reader scores 0.509. On the real cohort the old
labelling was wrong for up to 78% of windows in a single cell (DSSL V^S amp, alpha = 0.9).

PHASE SHIFT ONLY. The amplitude-damping arm was tested and REJECTED: the perturbation's input
magnitude is (1-alpha)*|z|, and sign(dg) also turns on |z|, so the two classes are not
magnitude-matched (within-stratum corr -0.5 to -0.7 against ~0 for phase). A pure
change-detector scores 0.075 there instead of 0.5, and a random linear projection scores
0.87-0.94 -- beating both ceilings, the same pathology that made the old Spearman layer
unreadable. Sign-randomising the scaling only lifts the null to 0.43 and was rejected too.
The claim this file supports is therefore about rhythmic PHASE deviation, and says so.

  python experiment_q2.py --variant-dir results_hrd/<run>/<backbone>_<pe>_seed<S>
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The project's single figure style, shared with scripts/collect_results.py so the paper's
# figures cannot drift apart. It used to live under scripts/, which is not an importable
# package, so all five of its consumers manipulated sys.path to reach it.
from tasks.style import ACCENT, BASE, GRID, INK, INK2, MUTED, SURFACE, strip
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from structured_rhythm import structured_features

from tasks._experiment_common import (encode, load_context, out_dir, random_init_model,
                                      save, write_csv)

# Levels that actually produce a comparison. 6, 8 and 12 h were dropped after measurement:
# at those magnitudes 100% of windows have dg > 0, the negative class is empty and the cell
# contributes ZERO pairs -- it costs an encode pass and returns nothing. 0.5-4 h yields
# ~12.8k usable pairs. 0.25 h is one bin at bin_minutes=15 and is the finest shift that exists.
PHASE_LEVELS = (0.5, 1.0, 2.0, 3.0, 4.0)


def cosinor_z(Xs, bpd):
    """Per-window, per-channel 24-h cosinor coefficient as ONE complex number, z = a + ib.

    Amplitude is |z| and acrophase is arg z, so a phase shift is exactly a rotation of z and
    the ground truth below needs no separate treatment of the two constructs. Verified: after
    perturb(..., "phase_shift_h", delta) the identity z' = z * exp(2i*pi*delta/24) holds to a
    relative error of 5e-9, i.e. float32 machine precision, because np.roll is circular and
    the window is a whole number of days.

    Never pooled across channels: sleep runs roughly antiphase to activity and heart rate, so
    a pooled circular mean lands between two different constructs.
    """
    t = 2 * np.pi * np.arange(Xs.shape[1]) / bpd
    Z = Xs - Xs.mean(1)[:, None]
    a = 2 * (Z * np.cos(t)[None, :, None]).mean(1)
    b = 2 * (Z * np.sin(t)[None, :, None]).mean(1)
    return a + 1j * b


def window_start_days(window_ids):
    """Elapsed days of each window's start, from the "pid_<isotime>" window id.

    Windows are indexed by POSITION everywhere else, and position proxies time only while a
    participant's windows are contiguous. They are not: the Algorithm-1 quality gate drops
    windows, leaving holes. Measured on the real cohort, 3.8% of consecutive stored windows
    are more than 7 days apart, median gap 21 days, maximum 35.
    """
    t = pd.to_datetime([str(w).rsplit("_", 1)[1] for w in window_ids])
    return ((t - t.min()).days).to_numpy().astype(float)


def personal_baseline(V, pids, R, tdays=None, max_span=None):
    """(mu, sd, ok) where each window's reference is its R PRECEDING windows of the same
    person. Windows without a full reference are never scored.

    `max_span` (days) additionally requires that reference to be CONTIGUOUS in time. The
    reference is selected by position, so a dropped window silently stretches it: measured on
    the real cohort, 8.6% of scored windows carry a reference spanning more than the nominal
    R-1 weeks, up to 77 days. A baseline drawn from 28 days and one drawn from 84 days are
    not the same quantity.
    """
    mu, sd, ok = np.zeros_like(V), np.ones_like(V), np.zeros(len(V), bool)
    n_skipped = 0
    for p in np.unique(pids):
        idx = np.where(pids == p)[0]
        for j in range(R, len(idx)):
            if max_span is not None and tdays[idx[j - 1]] - tdays[idx[j - R]] > max_span:
                n_skipped += 1                      # reference not contiguous -> not scored
                continue
            ref = V[idx[j - R:j]]
            mu[idx[j]], sd[idx[j]], ok[idx[j]] = ref.mean(0), ref.std(0) + 1e-6, True
    personal_baseline.n_skipped = n_skipped
    return mu, sd, ok


def dscore(V, mu, sd):
    """STANDARDISED EUCLIDEAN distance to the personal baseline -- each dimension divided by
    its own within-person SD, then a plain Euclidean norm. UNCHANGED from the previous design.

    Not a Mahalanobis distance: there is no inverse covariance, so correlated dimensions are
    counted once each. With the spectral seasonal readout the vector is 176-dimensional at
    --repr-dims 32 while a participant contributes ~26 windows, so a full covariance could not
    be estimated anyway.
    """
    return np.sqrt((((V - mu) / sd) ** 2).mean(1))


def raw_deviation(Zw, zbar):
    """g -- rhythmic deviation of a window from its own RAW-signal personal baseline.

    Deliberately the SAME functional form as `dscore`: mean of squared coordinate deviations
    from the mean of the R preceding windows, then a square root. So dd and dg are one object
    measured in two spaces, which is what makes their comparison interpretable.

    Not standardised per channel, unlike dscore: the pipeline z-scores every sensor channel
    (prepare_hrd_dataset, z_score=True), so the z_c are already commensurate, and only the
    SIGN of the difference is ever used.
    """
    return np.sqrt((np.abs(Zw - zbar) ** 2).mean(1))


def resolve_phase_levels(levels, bin_minutes):
    """The requested shifts, kept only where the sampling grid can actually express them.

    `phase_shift` rolls by k = round(hours * 60 / bin_minutes) bins, so any level below half a
    bin rolls by ZERO and perturbs nothing. Nothing downstream notices: the deviation score is
    computed on an unchanged window, the ground-truth change dg is exactly 0, and a stratum
    needs both a positive and a negative dg to contribute a pair -- so the level silently
    contributes no evidence at all.

    Run 2074344 hit exactly that. GLOBEM samples 4 segments a day, i.e. 360-minute bins, and
    the default grid (0.5, 1, 2, 3, 4 h) rolls by (0, 0, 0, 0, 1) bins: four of five levels
    were no-ops and the fifth moved every window identically, so n_pairs came out 0 in all 24
    variants and RQ2 measured nothing while reporting success.

    Levels that collide on the same k are also dropped -- two names for one perturbation would
    enter the stratified estimate twice. If nothing survives, a grid of 1..4 bins replaces it,
    which is the finest thing the data can represent.
    """
    seen, keep, dropped = set(), [], []
    for lv in levels:
        k = int(round(float(lv) * 60 / bin_minutes))
        if k == 0 or k in seen:
            dropped.append((float(lv), k))
            continue
        seen.add(k)
        keep.append(float(lv))
    # A single surviving level is no better than none. The stratified estimate needs strata
    # holding BOTH a positive and a negative dg, and one uniform shift moves every window the
    # same way -- which is the second half of what went wrong on GLOBEM: 4 h survived the
    # filter, rolled every window by the same 1 bin, and frac_dg_positive came out 1.0. Top up
    # from the finest shifts the grid can express until at least three distinct ones exist.
    k_extra = 1
    while len(keep) < 3:
        if k_extra not in seen:
            seen.add(k_extra)
            keep.append(k_extra * bin_minutes / 60)
        k_extra += 1
        if k_extra > 12:                       # nothing sensible left to add
            break
    keep = sorted(set(keep))
    if dropped:
        print(f"[rq2] dropped unrepresentable phase levels {[d[0] for d in dropped]} h "
              f"(they roll by {[d[1] for d in dropped]} bins at {bin_minutes} min/bin)")
    if bin_minutes >= 120:
        print(f"[rq2] NOTE: at {bin_minutes} min/bin the smallest expressible shift is "
              f"{bin_minutes/60:.1f} h, a {bin_minutes/60/24:.0%} fraction of the circadian "
              f"cycle. This perturbation is coarse by construction on this cohort.")
    return keep


def phase_shift(X, hours, n_sensors, bin_minutes):
    """Circular shift of the sensor channels by `hours`. Exactly a rotation of z (see
    cosinor_z). Calendar/clock channels beyond n_sensors are left alone -- shifting them would
    move the reference frame with the signal and cancel the perturbation."""
    Xp = X.copy()
    k = int(round(hours * 60 / bin_minutes))
    Xp[:, :, :n_sensors] = np.roll(X[:, :, :n_sensors], k, axis=1)
    return Xp


def stratum_pairs(dd, dg, keys):
    """Per-stratum (U, n_pairs) for the stratified Mann-Whitney.

    U is the count of concordant (dg>0, dg<0) pairs with ties at 1/2 -- computed from ranks
    rather than the O(n^2) double loop. Returned per stratum rather than summed so the
    participant bootstrap can resample strata without recomputing any rank.
    """
    out = {}
    for s in np.unique(keys):
        m = (keys == s) & np.isfinite(dd) & np.isfinite(dg)
        pos, neg = dd[m & (dg > 0)], dd[m & (dg < 0)]
        if len(pos) == 0 or len(neg) == 0:
            continue                                # no contrast in this cell -> no evidence
        r = rankdata(np.r_[pos, neg])
        out[s] = (float(r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2),
                  float(len(pos) * len(neg)))
    return out


def concordance(per_stratum, pid_of, draw=None):
    """C = sum(U) / sum(n_pairs), optionally over a bootstrap draw of PARTICIPANTS.

    `draw` is a list of pids WITH multiplicity: a participant drawn twice contributes its
    strata twice, which is what makes this a bootstrap rather than a subsample.
    """
    if draw is None:
        u = sum(v[0] for v in per_stratum.values())
        n = sum(v[1] for v in per_stratum.values())
    else:
        u = n = 0.0
        by = {}
        for s, v in per_stratum.items():
            by.setdefault(pid_of[s], []).append(v)
        for q in draw:
            for a, b in by.get(q, ()):
                u += a; n += b
    return u / n if n else np.nan


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant-dir", required=True)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--ref-windows", type=int, default=4,
                   help="Personal reference length (4 x 7d = 28d)")
    p.add_argument("--phase-levels", type=float, nargs="+", default=list(PHASE_LEVELS))
    p.add_argument("--n-boot", type=int, default=1000,
                   help="Participant bootstrap draws (shared across representations)")
    p.add_argument("--contiguous-reference", action=argparse.BooleanOptionalAction, default=True,
                   help="Score a window only when its R-window personal baseline is contiguous "
                        "in time (span <= 7*(R-1) days).")
    a = p.parse_args()

    ctx = load_context(a.variant_dir, a.cache_dir, a.gpu)
    d = out_dir(ctx, "rq2")
    R = a.ref_windows
    tdays = window_start_days(ctx.window_ids)
    max_span = float(7 * (R - 1)) if a.contiguous_reference else None
    Xs = ctx.X[:, :, :ctx.n_sensors]

    # --- ground truth, computed ONCE from the raw signal and independent of every encoder ---
    Z = cosinor_z(Xs, ctx.bins_per_day)
    zbar = np.full(Z.shape, np.nan, complex)
    ok = np.zeros(len(Z), bool)
    for q in np.unique(ctx.pids):
        idx = np.where(ctx.pids == q)[0]
        for j in range(R, len(idx)):
            if max_span is not None and tdays[idx[j - 1]] - tdays[idx[j - R]] > max_span:
                continue
            zbar[idx[j]] = Z[idx[j - R:j]].mean(0)
            ok[idx[j]] = True
    g_clean = raw_deviation(Z, zbar)

    # Held-out participants only. The deviation score is fit on nothing, but the ENCODER was
    # pretrained on every non-test window (train_hrd.py:759), so scoring train participants
    # would report a quantity the encoder has already seen.
    elig = ok & ctx.test_mask

    rand = random_init_model(ctx)
    dT = ctx.model.net.component_dims          # cost.py concatenates [trend | seasonal]

    # The seasonal block is [amp | phase] under phase_readout='angle' and [amp | cos | sin]
    # under 'circular', so the split is into halves or thirds depending on the run. Hard-coding
    # //2 sliced through the middle of a three-part block on any circular run, handing "amp"
    # part of the cosines and calling the rest "phase".
    n_blocks = 3 if getattr(ctx.model, "phase_readout", "angle") == "circular" else 2

    def cost_part(which):
        def f(A):
            V = encode(ctx.model, A, ctx.cfg)
            S = V[:, dT:]
            b = S.shape[1] // n_blocks
            return {"trend": V[:, :dT], "season": S,
                    "amp": S[:, :b], "phase": S[:, b:]}[which]
        return f

    def cosinor_rep(A):
        """Acrophase enters as cos/sin because it is an ANGLE: a raw radian would make the
        distance jump by 2*pi across the midnight wrap. Reported as a CEILING, not a
        competitor -- the ground truth g is defined in this same cosinor space, so a
        representation that encodes it explicitly is expected to win."""
        z = cosinor_z(A[:, :, :ctx.n_sensors], ctx.bins_per_day)
        m = A[:, :, :ctx.n_sensors].mean(1)
        return np.concatenate([m, np.abs(z), np.cos(np.angle(z)), np.sin(np.angle(z))], axis=1)

    def raw_rep(A, w=max(1, ctx.bins_per_day // 24)):
        """The window itself at hourly resolution, flattened. Also a CEILING: g is a linear
        functional of x, so raw space is structurally advantaged."""
        S = A[:, :, :ctx.n_sensors]
        n = (S.shape[1] // w) * w
        return S[:, :n].reshape(len(S), -1, w, S.shape[2]).mean(2).reshape(len(S), -1)

    reps = {"DSSL": lambda A: encode(ctx.model, A, ctx.cfg),
            "DSSL V^T": cost_part("trend"),
            "DSSL V^S": cost_part("season"),
            "Cosinor [ceiling]": cosinor_rep,
            "Handcrafted": lambda A: np.concatenate(
                [A[:, :, :ctx.n_sensors].mean(1), A[:, :, :ctx.n_sensors].std(1)], axis=1),
            "Raw (hourly) [ceiling]": raw_rep,
            "Random-init": lambda A: encode(rand, A, ctx.cfg),
            # Cosinor's estimator plus the three things cosinor cannot express -- waveform
            # harmonics, per-day phase/amplitude dispersion, and the channel-to-channel
            # acrophase difference. RQ2's perturbation IS a phase shift, so this is the arm
            # where the per-day block should matter most. Not a ceiling: it is fitted from
            # the same window every other arm sees, with no access to the perturbation.
            "Structured": lambda A: structured_features(A, ctx.bins_per_day, ctx.n_sensors)}
    # amp/phase exist only under season_pool='spec', where the seasonal readout really is
    # [amplitude | phase]. Under the other poolings, halving the block yields two arbitrary
    # slices that only LOOK like amplitude and phase.
    if ctx.cfg.get("season_pool") == "spec":
        reps["DSSL V^S amp"] = cost_part("amp")
        reps["DSSL V^S phase"] = cost_part("phase")

    # Baselines are built on the CLEAN windows and frozen for the rest of the run.
    base = {}
    for name, fn in reps.items():
        V = fn(ctx.X)
        mu, sd, _ = personal_baseline(V, ctx.pids, R, tdays, max_span)
        base[name] = (V, mu, sd, dscore(V, mu, sd))
    print(f"[rq2] {int(elig.sum())} scored held-out windows, "
          f"{len(np.unique(ctx.pids[elig]))} participants")

    # --- one pass per shift level; the perturbed input is shared by every representation ----
    per_stratum = {name: {} for name in reps}
    pid_of, cells = {}, []
    levels = resolve_phase_levels(a.phase_levels, ctx.bin_minutes)
    for lv in levels:
        Xp = phase_shift(ctx.X, lv, ctx.n_sensors, ctx.bin_minutes)
        dg = raw_deviation(cosinor_z(Xp[:, :, :ctx.n_sensors], ctx.bins_per_day), zbar) - g_clean
        keys = np.array([f"{q}|{lv}" for q in ctx.pids])
        for s in np.unique(keys[elig]):
            pid_of[s] = s.rsplit("|", 1)[0]
        for name, fn in reps.items():
            V, mu, sd, d_clean = base[name]
            # Only the scored window is perturbed; its mu/sd stay clean, so dd measures the
            # response to the perturbation and not a shifted baseline. The full array is
            # encoded and then sliced -- encoding only Xp[elig] changes the batch boundaries
            # and flips ~2-5% of windows by floating-point noise alone, which was visible in
            # the old design as paired_win_rate 0.488 at a ZERO-magnitude perturbation.
            dd = np.where(elig, dscore(fn(Xp), mu, sd) - d_clean, np.nan)
            for s, v in stratum_pairs(np.where(elig, dd, np.nan),
                                      np.where(elig, dg, np.nan), keys).items():
                per_stratum[name][s] = v
        m = elig & np.isfinite(dg)
        cells.append([lv, int(m.sum()), round(float((dg[m] > 0).mean()), 4),
                      int(sum(1 for s in np.unique(keys[elig]) if s in per_stratum["DSSL"])),
                      int(sum(per_stratum["DSSL"].get(s, (0, 0))[1]
                              for s in np.unique(keys[elig])))])

    write_csv(d, "rq2_cells", ["shift_hours", "n_windows", "frac_dg_positive",
                               "n_strata_with_both_classes", "n_pairs"], cells)

    # --- C, with a participant bootstrap on SHARED draws -----------------------------------
    # Shared draws are what let delta_vs_DSSL have a CI that can exclude zero even when the
    # two marginal CIs overlap; differencing two independent intervals cannot do that.
    uniq = np.unique(ctx.pids[elig])
    rng = np.random.default_rng(ctx.seed)
    draws = [rng.choice(uniq, len(uniq), replace=True) for _ in range(a.n_boot)]
    point = {n: concordance(per_stratum[n], pid_of) for n in reps}
    boot = {n: np.array([concordance(per_stratum[n], pid_of, dr) for dr in draws])
            for n in reps}

    ci = lambda b: [float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))]
    rows, res = [], {"variant": ctx.tag, "seed": ctx.seed, "ref_windows": R,
                     "perturbation": "phase_shift_h", "levels": list(levels),
                     "n_participants": int(len(uniq)), "n_windows": int(elig.sum()),
                     "metric": "stratified Mann-Whitney concordance C; null = 0.5",
                     "strata": "(participant, shift level)",
                     "concordance": {}}
    for n in reps:
        lo, hi = ci(boot[n])
        npair = sum(v[1] for v in per_stratum[n].values())
        row = {"C": round(point[n], 5), "ci": [round(lo, 5), round(hi, 5)],
               "n_pairs": int(npair), "excludes_null": bool(lo > 0.5)}
        if n != "DSSL":
            dl = boot["DSSL"] - boot[n]
            row["delta_vs_DSSL"] = round(point["DSSL"] - point[n], 5)
            row["delta_ci"] = [round(v, 5) for v in ci(dl)]
        res["concordance"][n] = row
        rows.append([n, row["C"], row["ci"][0], row["ci"][1], row["n_pairs"],
                     row.get("delta_vs_DSSL", ""), *(row.get("delta_ci") or ["", ""])])
    write_csv(d, "rq2_concordance",
              ["representation", "C", "ci_lo", "ci_hi", "n_pairs",
               "delta_vs_DSSL", "delta_ci_lo", "delta_ci_hi"], rows)
    res["reference_contiguity"] = {"enforced": bool(a.contiguous_reference),
                                   "max_span_days": max_span,
                                   "n_windows_skipped": int(getattr(personal_baseline,
                                                                    "n_skipped", 0))}
    save(d, "rq2", res)

    # --- one figure: C with its CI, chance-referenced ---------------------------------------
    order = sorted(reps, key=lambda n: point[n])
    fig, ax = plt.subplots(figsize=(8.2, 0.46 * len(order) + 2.1))
    for i, n in enumerate(order):
        if i % 2 == 0:
            ax.axhspan(i - .5, i + .5, color=GRID, alpha=.35, lw=0, zorder=0)
    ax.axvline(0.5, color=BASE, lw=1.3, ls=(0, (4, 3)), zorder=1)
    for i, n in enumerate(order):
        top = n == "DSSL"
        col = ACCENT if top else (BASE if "[ceiling]" in n else INK2)
        lo, hi = res["concordance"][n]["ci"]
        ax.plot([lo, hi], [i, i], color=col, lw=3.2 if top else 1.8,
                alpha=1 if top else .55, zorder=2, solid_capstyle="round")
        ax.scatter([point[n]], [i], s=84 if top else 40, color=col, zorder=3,
                   edgecolor=SURFACE, linewidth=.9)
        ax.annotate(format(point[n], ".3f"), (hi, i), textcoords="offset points",
                    xytext=(7, -3), fontsize=8, color=col,
                    fontweight="bold" if top else "normal")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    for tk, n in zip(ax.get_yticklabels(), order):
        tk.set_color(ACCENT if n == "DSSL" else INK)
        if n == "DSSL":
            tk.set_fontweight("bold")
    ax.set_ylim(-.5, len(order) - .5)
    ax.set_xlabel("directional concordance $C$   (0.5 = chance)", fontsize=9, color=MUTED)
    ax.tick_params(axis="y", length=0)
    strip(ax)
    ax.set_title("RQ2  Does the personal-baseline distance rise when the RHYTHMIC\n"
                 "deviation rises, rather than merely when the input changes?",
                 fontsize=11, color=INK, loc="left", pad=10)
    fig.text(0.01, 0.005,
             f"{ctx.tag}   |   phase shift {levels} h   |   personal reference = {R} "
             f"preceding windows, frozen   |   {int(elig.sum())} held-out windows, "
             f"{len(uniq)} participants\npositive = the shift increased the window's raw 24-h "
             "cosinor deviation from its own baseline; negative = it decreased it   |   "
             "strata = (participant, level)   |   bars = 95% participant bootstrap",
             fontsize=7.5, color=MUTED, ha="left")
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    fig.savefig(d / "rq2_concordance.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
