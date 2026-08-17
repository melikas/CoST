"""RQ2 -- Personalized rhythmic phenotyping: do unlabeled personal baselines detect
within-person rhythmic deviation?

Runs on a FINISHED train_hrd.py variant directory: loads the frozen encoder, never trains.

  E2.1  score      mu_p = mean of the R windows PRECEDING the scored one (strictly causal),
                   d = diagonal Mahalanobis distance to it, s = robust within-person z of d.
  E2.2  layer 1    synthetic perturbations of known magnitude -> detection AUC, and delta*,
                   the smallest perturbation caught at 80% TPR / 5% FPR.
        layer 2    convergent validity: within-person Spearman(s, |delta phi|) from the RAW
                   signal, aggregated across people.
        layer 3    per-person slope of s over the study, contrasted across the Case-1
                   depression_trajectory groups.
  E2.3  baselines  the identical score computed on handcrafted stats and on a random-init
                   encoder, so the claim is comparative.

Windows here are the project's non-overlapping 7-day windows, so the personal reference is
R windows (default 4 = 28 days), not 28 daily windows as in the design note.

  python experiment_q2.py --variant-dir results_hrd/<run>/<backbone>_<pe>_seed<S>
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kruskal, spearmanr, linregress, wilcoxon

from tasks._eval_protocols import fast_auc
from scipy.stats import rankdata
from tasks._experiment_common import (encode, load_context, out_dir, random_init_model,
                                      save, write_csv)
from tasks.decomposition import harmonic_reference
from tasks.rhythm import _interdaily_stability

PERTURBATIONS = ("phase_shift_h", "amplitude_damping", "sleep_fragmentation")

# delta* is defined on the perturbation MAGNITUDE (docs E2.2), not on the grid level. For phase
# and fragmentation the two coincide; for damping they are complements -- x' = x - (1-alpha)*sig,
# so alpha=0.9 is a 10% perturbation and alpha=0.0 a 100% one. Ranking by level therefore
# returned the STRONGEST damping caught instead of the mildest, and the resulting "delta* = 0.0"
# reads as near-perfect sensitivity while meaning the exact opposite. Kept explicit here rather
# than implied by the order of the level lists. (RQ3 solves the same problem with worse_is_larger.)
MAGNITUDE = {"phase_shift_h":       (lambda lv: lv,       "h"),
             "amplitude_damping":   (lambda lv: 1.0 - lv, "fraction of circadian amplitude"),
             "sleep_fragmentation": (lambda lv: lv,       "interruptions per 7-day window")}


def delta_star(got, kind, col, thr=0.80):
    """Smallest perturbation MAGNITUDE from which detection stays above `thr`.

    Sustained, not merely the first level that passes: with ~36 held-out participants a mild
    level can clear the bar by chance, and a one-off crossing is not a detection limit.
    `got` rows are (level, auc, tpr); `col` picks 1 = AUC or 2 = TPR. NaN never passes.
    Returns None when the grid never reaches `thr`.
    """
    mag, unit = MAGNITUDE[kind]
    pts = sorted(got, key=lambda r: mag(r[0]))                  # ascending magnitude
    for i, r in enumerate(pts):
        if all(p[col] >= thr for p in pts[i:]):
            return {"magnitude": round(mag(r[0]), 4), "level": r[0], "unit": unit}
    return None


def cosinor_params(Xs, bpd):
    """Per-window, PER-CHANNEL closed-form 24-h cosinor: MESOR, amplitude, acrophase. Each (N, C).

    The vectorised single-harmonic equivalent of baselines/cosinor.py::paper_cosinor_features,
    which fits every (window, channel, period) through CosinorPy and is far too slow to re-run
    inside the perturbation grid (~21 cells x ~900 windows -- experiment_q1.py caches it and
    calls it the slow part). Same model, closed form.

    Never pooled across channels: sleep runs roughly antiphase to activity and heart rate, so
    an amplitude-weighted circular mean lands between two different constructs and moves with
    the relative z-scored amplitudes of sleep vs activity rather than with physiology.
    """
    t = 2 * np.pi * np.arange(Xs.shape[1]) / bpd
    M = Xs.mean(1)
    Z = Xs - M[:, None]
    a = 2 * (Z * np.cos(t)[None, :, None]).mean(1)
    b = 2 * (Z * np.sin(t)[None, :, None]).mean(1)
    return M, np.hypot(a, b), np.arctan2(b, a)


def phase_deviation(acro, pids, R):
    """|acrophase - circular mean of the R preceding windows of the same person|, per window."""
    d = np.full(len(acro), np.nan)
    for p in np.unique(pids):
        idx = np.where(pids == p)[0]
        for j in range(R, len(idx)):
            ref = np.angle(np.exp(1j * acro[idx[j - R:j]]).mean())
            d[idx[j]] = np.abs(np.angle(np.exp(1j * (acro[idx[j]] - ref))))
    return d


def within_person_rho(s, dmark, pids, ok, min_n=5):
    """Median within-person Spearman(s, |d marker|), the share of positive rho, and the
    design's Wilcoxon signed-rank test of that distribution against 0 (E2.2 layer 2).

    The median alone cannot say whether the correlation is distinguishable from none, which is
    what the design asks for; the test is one-sided (greater) because the hypothesis is
    directional -- a deviation score that tracks a real chronobiological change must correlate
    POSITIVELY with it. Needs >= 6 participants: below that the signed-rank statistic cannot
    reach p < 0.05 at any effect size, so a number there would be meaningless rather than
    merely weak.
    """
    rho = []
    for p in np.unique(pids):
        m = ok & (pids == p) & np.isfinite(s) & np.isfinite(dmark)
        if m.sum() >= min_n:
            r = spearmanr(s[m], dmark[m]).correlation
            if np.isfinite(r):
                rho.append(r)
    w = None
    if len(rho) >= 6 and np.any(np.asarray(rho) != 0):
        w = float(wilcoxon(rho, alternative="greater").pvalue)
    return {"median_within_person_spearman": float(np.median(rho)) if rho else None,
            "n_participants": len(rho),
            "frac_positive": float(np.mean(np.asarray(rho) > 0)) if rho else None,
            "wilcoxon_p_greater": w}


def linear_deviation(v, pids, R):
    """|v_t - mean of the R preceding windows of the same person|, per window.

    The linear counterpart of `phase_deviation`, for markers that live on a line rather than a
    circle (amplitude, interdaily stability). Same causal reference window as the deviation
    score itself, so the two series are comparable window for window.
    """
    d = np.full(len(v), np.nan)
    for p in np.unique(pids):
        idx = np.where(pids == p)[0]
        for j in range(R, len(idx)):
            ref = v[idx[j - R:j]]
            if np.isfinite(ref).any() and np.isfinite(v[idx[j]]):
                d[idx[j]] = abs(v[idx[j]] - np.nanmean(ref))
    return d


def personal_baseline(V, pids, R):
    """(mu, sd, ok) where each window's reference is its R PRECEDING windows of the same
    person. Windows without a full reference are never scored."""
    mu, sd, ok = np.zeros_like(V), np.ones_like(V), np.zeros(len(V), bool)
    for p in np.unique(pids):
        idx = np.where(pids == p)[0]
        for j in range(R, len(idx)):
            ref = V[idx[j - R:j]]
            mu[idx[j]], sd[idx[j]], ok[idx[j]] = ref.mean(0), ref.std(0) + 1e-6, True
    return mu, sd, ok


def dscore(V, mu, sd):
    return np.sqrt((((V - mu) / sd) ** 2).mean(1))


def robust_z(d, pids, ok):
    """Within-person median/MAD standardisation, fitted on that person's scored windows."""
    s = np.full(len(d), np.nan)
    for p in np.unique(pids):
        m = ok & (pids == p)
        if m.sum() < 3:
            continue
        med = np.median(d[m])
        mad = 1.4826 * np.median(np.abs(d[m] - med)) + 1e-9
        s[m] = (d[m] - med) / mad
    return s


def perturb(X, kind, level, n_sensors, sig, sleep_ix, bpd, bin_minutes, rng):
    """Return a copy of X with a perturbation of known magnitude on the sensor channels."""
    Xp = X.copy()
    S = slice(None), slice(None), slice(0, n_sensors)
    if kind == "phase_shift_h":
        shift = int(round(level * 60 / bin_minutes))
        Xp[S] = np.roll(X[S], shift, axis=1)
    elif kind == "amplitude_damping":
        Xp[S] = X[S] - (1.0 - level) * sig          # damp only the circadian component
    elif kind == "sleep_fragmentation":
        if sleep_ix is None or level == 0:
            return Xp
        w = max(1, 30 // bin_minutes)               # 30-min interruptions
        for i in range(len(X)):
            awake = X[i, :, sleep_ix].min()
            for st in rng.integers(0, X.shape[1] - w, int(level)):
                Xp[i, st:st + w, sleep_ix] = awake
    return Xp


def detect_auc(clean, pert, calib=None):
    """Paired detection: same windows, unperturbed vs perturbed.

    The operating point comes from `calib`: clean scores of participants who are NOT scored
    here (the train cohort). Taking the 95th percentile of the very clean scores being
    evaluated pins the false-alarm rate at exactly 5% BY CONSTRUCTION and reports a TPR at a
    threshold tuned to that same sample, which is optimistic. With the threshold fixed
    elsewhere the realized FPR is free to miss 5%, and it is returned so any miss is visible
    rather than hidden. Falls back to the in-sample quantile only when calibration scores are
    unavailable -- there `fpr_realized` is 0.05 by construction and carries no information.

    Returns (auc, tpr, fpr_realized).
    """
    m = np.isfinite(clean) & np.isfinite(pert)
    if m.sum() < 10:
        return np.nan, np.nan, np.nan
    c = calib[np.isfinite(calib)] if calib is not None else np.empty(0)
    thr = float(np.percentile(c if len(c) >= 20 else clean[m], 95))
    return _auc(clean[m], pert[m]), float((pert[m] > thr).mean()), float((clean[m] > thr).mean())


def _auc(c, p):
    """Two-sample AUC: P(a perturbed window outranks a clean one), clean labelled 0."""
    return fast_auc(np.r_[np.zeros(len(c)), np.ones(len(p))], np.r_[c, p])


def detection_stats(scores, pids, calib=None, ref="CoST", n_boot=1000, seed=0):
    """Point estimates + participant-level CIs for ONE (perturbation, level) cell.

    Two things the design document asks for and the previous version skipped.

    (1) A CI at all (design rule 2). Rows are windows and one participant contributes many, so
        a row-level interval answers "how stable is this on these people" when the claim is
        "does this transfer to a NEW person". Whole participants are drawn with replacement
        instead -- the rule already used by _eval_protocols.participant_bootstrap_auc.

    (2) The pairing. The design is paired twice over and neither pairing was used. Every window
        supplies BOTH a clean and a perturbed score, so `paired_win_rate` = P(the same window
        scores higher when perturbed) drops the between-window variance the two-sample AUC
        carries. And every representation scores the SAME windows, so `delta_vs_ref` is
        bootstrapped on SHARED participant draws -- which is what lets its CI exclude zero even
        when the two marginal CIs overlap. Differencing two independent intervals cannot.

    `scores` = {name: (clean, pert)} on a common window index. Returns {name: {...}}, empty
    when fewer than 10 windows are scorable in every representation.
    """
    names = list(scores)
    m = np.all([np.isfinite(c) & np.isfinite(p) for c, p in scores.values()], axis=0)
    if m.sum() < 10:
        return {}
    pm = np.asarray(pids)[m]
    S = {n: (c[m], p[m]) for n, (c, p) in scores.items()}
    pt = {n: detect_auc(*S[n], calib=None if calib is None else calib.get(n)) for n in names}
    uniq = np.unique(pm)
    by = [np.where(pm == q)[0] for q in uniq]
    rng = np.random.default_rng(seed)
    boot = {n: [] for n in names}
    for _ in range(n_boot):
        ix = np.concatenate([by[q] for q in rng.integers(0, len(uniq), len(uniq))])
        for n in names:                                  # same draw for every representation
            boot[n].append(_auc(S[n][0][ix], S[n][1][ix]))
    ci = lambda b: [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    out = {}
    for n in names:
        c, p = S[n]
        out[n] = {"auc": pt[n][0], "ci": ci(boot[n]), "tpr_at_fpr5": pt[n][1],
                  "fpr_realized": pt[n][2],
                  "paired_win_rate": float((p > c).mean() + 0.5 * (p == c).mean()),
                  "n_windows": int(m.sum()), "n_participants": int(len(uniq))}
        if n != ref and ref in names:
            out[n]["delta_vs_ref"] = float(pt[ref][0] - pt[n][0])
            out[n]["delta_ci"] = ci(np.asarray(boot[ref]) - np.asarray(boot[n]))
    return out


def energy_within_person(s, ctx, ok, n_perm=1000, min_n=5):
    """E2.2 layer 3, clinical half: does the deviation score rank a person's LOW-energy
    windows above their high-energy ones, inside that person?

    The design says "low-energy DAYS", but RQ2 scores non-overlapping 7-day windows -- there
    are no days at this resolution -- so the target is the window's own mean emotional energy
    (`ctx.ee_win`), split at the participant's RANK median. A comparison split would not be a
    median split: EE is a 1-5 integer scale, so `<= median` sweeps every tie into one class.

    Deviation is expected to be HIGH on low-energy windows, so the label is "low energy" and
    a score above 0.5 supports the hypothesis. Reported with a within-person label permutation
    (the design's stated control): shuffling the labels inside each person destroys the
    association while leaving that person's score distribution and window count untouched, so
    the null it gives is the one this test actually needs.
    """
    ee = getattr(ctx, "ee_win", None)
    if ee is None:
        return {"status": "SKIPPED_NO_EE_WIN",
                "note": "dataset cache predates _SCHEMA_VERSION 2; rebuild it"}
    ee = np.asarray(ee, dtype=float)
    rng = np.random.default_rng(ctx.seed)

    def macro_auc(lab):
        per = []
        for p in np.unique(ctx.pids):
            m = ok & (ctx.pids == p) & np.isfinite(s) & np.isfinite(ee)
            if m.sum() >= min_n and 0 < lab[m].sum() < m.sum():
                per.append(fast_auc(lab[m], s[m]))
        per = [v for v in per if np.isfinite(v)]
        return per

    low = np.zeros(len(ee), dtype=int)
    for p in np.unique(ctx.pids):
        m = ok & (ctx.pids == p) & np.isfinite(ee)
        if m.sum():
            low[m] = (rankdata(ee[m], method="average") / int(m.sum()) <= 0.5).astype(int)

    per = macro_auc(low)
    if len(per) < 2:
        return {"status": "TOO_FEW_PARTICIPANTS", "n_participants": len(per)}

    obs = float(np.median(per))
    null = []
    for _ in range(n_perm):
        perm = low.copy()
        for p in np.unique(ctx.pids):
            m = ok & (ctx.pids == p) & np.isfinite(ee)
            if m.sum():
                perm[m] = rng.permutation(low[m])
        v = macro_auc(perm)
        if v:
            null.append(float(np.median(v)))
    p_perm = (1 + sum(v >= obs for v in null)) / (1 + len(null)) if null else None
    return {"status": "OK", "median_within_person_auc": obs,
            "n_participants": len(per),
            "frac_above_half": float(np.mean(np.asarray(per) > 0.5)),
            "wilcoxon_p_greater": (float(wilcoxon(np.asarray(per) - 0.5,
                                                  alternative="greater").pvalue)
                                   if len(per) >= 6 and np.any(np.asarray(per) != 0.5)
                                   else None),
            "permutation_p": p_perm, "n_permutations": len(null)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant-dir", required=True)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--ref-windows", type=int, default=4, help="Personal reference length (4 x 7d = 28d)")
    p.add_argument("--n-boot", type=int, default=1000,
                   help="Participant bootstrap draws per grid cell (~9 s for the whole grid)")
    # Grids must reach far enough for delta* (80% TPR at 5% FPR) to be DEFINED, otherwise the
    # RQ2 headline is None. Run 19606825 topped out at TPR 0.45 (4 h shift), 0.14 (alpha 0.3)
    # and 0.05 (8 interruptions), so all three are extended to their physiological limits:
    #   phase  -- 12 h is antiphase, the largest shift that exists (beyond it wraps back);
    #   amp    -- 0.0 removes the circadian component entirely, the strongest possible damping;
    #   frag   -- levels are interruptions per 7-DAY window, so 8 was ~1/night (normal sleep).
    #             Severe fragmentation is ~8-16/night, i.e. 56-112 per window.
    p.add_argument("--phase-levels", type=float, nargs="+",
                   default=[0.5, 1, 2, 3, 4, 6, 8, 12])
    p.add_argument("--amp-levels", type=float, nargs="+",
                   # 1.0 = magnitude 0, the free negative control: no perturbation at all, so
                   # AUC must be 0.5 and the realized FPR must land on 0.05. Anything else
                   # there is a leak in the scoring pipeline, not a property of the encoder.
                   default=[1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0])
    p.add_argument("--frag-levels", type=float, nargs="+",
                   default=[7, 14, 28, 56, 112, 224])
    a = p.parse_args()

    ctx = load_context(a.variant_dir, a.cache_dir, a.gpu)
    d = out_dir(ctx, "rq2")
    Xs = ctx.X[:, :, :ctx.n_sensors]
    _, sig = harmonic_reference(Xs, 24 * 60.0 / ctx.bin_minutes)
    sleep_ix = next((i for i, c in enumerate(ctx.sensor_cols) if "asleep" in c or "sleep" in c),
                    None)
    R = a.ref_windows

    # One representation FUNCTION per baseline, built once: the random-init control must keep
    # the same untrained weights across every perturbation level, and rebuilding it per level
    # would silently redraw them.
    rand = random_init_model(ctx)

    def cosinor_rep(A):
        """E2.3 baseline (b). Acrophase enters as cos/sin because it is an ANGLE: a raw radian
        would make the deviation score jump by 2*pi across the midnight wrap."""
        M, amp, ph = cosinor_params(A[:, :, :ctx.n_sensors], ctx.bins_per_day)
        return np.concatenate([M, amp, np.cos(ph), np.sin(ph)], axis=1)

    def raw_rep(A, w=max(1, ctx.bins_per_day // 24)):
        """E2.3 baseline (c): the window itself at hourly resolution, flattened."""
        S = A[:, :, :ctx.n_sensors]
        n = (S.shape[1] // w) * w
        return S[:, :n].reshape(len(S), -1, w, S.shape[2]).mean(2).reshape(len(S), -1)

    reps = {"CoST": lambda A: encode(ctx.model, A, ctx.cfg),
            "Cosinor": cosinor_rep,
            "Handcrafted": lambda A: np.concatenate(
                [A[:, :, :ctx.n_sensors].mean(1), A[:, :, :ctx.n_sensors].std(1)], axis=1),
            "Raw (hourly)": raw_rep,
            "Random-init": lambda A: encode(rand, A, ctx.cfg)}
    views = {name: fn(ctx.X) for name, fn in reps.items()}
    res = {"variant": ctx.tag, "seed": ctx.seed, "ref_windows": R}

    # --- E2.2 layer 1: detection of synthetic perturbations, on held-out participants -----
    levels = {"phase_shift_h": a.phase_levels, "amplitude_damping": a.amp_levels,
              "sleep_fragmentation": a.frag_levels}
    base = {}
    for name, V in views.items():
        mu, sd, ok = personal_baseline(V, ctx.pids, R)
        dall, elig = dscore(V, mu, sd), ok & ctx.test_mask
        # Last element: the clean scores of TRAIN participants. The 5%-FPR operating point is
        # fixed there and applied to the held-out cohort unchanged, so it is never tuned on the
        # windows it is scored on (tasks/_eval_protocols.py::threshold_at_sensitivity, same rule).
        base[name] = (V, mu, sd, elig, np.where(elig, dall, np.nan), dall[ok & ctx.train_mask])
        print(f"[rq2] {name}: {int(elig.sum())} scored held-out windows, "
              f"{int((ok & ctx.train_mask).sum())} calibration windows")

    # The perturbed input is built ONCE per (kind, level) and shared by every representation:
    # E2.3's claim is "same score, different space", so the space must be the only thing that
    # differs. Previously the loops nested the other way and one advancing RNG was threaded
    # through them, so each representation saw a different sleep_fragmentation draw (the only
    # kind that is random -- phase and damping are deterministic). Hoisting it also stops the
    # O(N x level) fragmentation loop from being recomputed once per representation.
    # The generator is derived per cell so each (kind, level) reproduces on its own, whatever
    # the rest of the grid or the set of representations happens to be.
    rows, curves = [], {}
    for ki, kind in enumerate(PERTURBATIONS):
        for lv in levels[kind]:
            Xp = perturb(ctx.X, kind, lv, ctx.n_sensors, sig, sleep_ix, ctx.bins_per_day,
                         ctx.bin_minutes, np.random.default_rng([ctx.seed, ki, int(lv * 1000)]))
            sc, cal = {}, {}
            for name, (V, mu, sd, elig, clean, calib) in base.items():
                # Only the scored window is perturbed; its reference mu/sd stays clean, so the
                # score measures detection rather than a shifted baseline.
                Vp = V.copy()
                Vp[elig] = reps[name](Xp[elig])
                sc[name], cal[name] = (clean, np.where(elig, dscore(Vp, mu, sd), np.nan)), calib
            for name, s in detection_stats(sc, ctx.pids, cal, n_boot=a.n_boot,
                                           seed=ctx.seed).items():
                dl = ([round(s["delta_vs_ref"], 4)] + [round(v, 4) for v in s["delta_ci"]]
                      if "delta_vs_ref" in s else ["", "", ""])        # blank on the reference
                rows.append([name, kind, lv, round(s["auc"], 4), round(s["ci"][0], 4),
                             round(s["ci"][1], 4), round(s["paired_win_rate"], 4),
                             round(s["tpr_at_fpr5"], 4), round(s["fpr_realized"], 4), *dl,
                             s["n_windows"], s["n_participants"]])
                curves.setdefault((name, kind), []).append((lv, s["auc"], *s["ci"]))

    write_csv(d, "rq2_detection",
              ["representation", "perturbation", "level", "auc", "ci_lo", "ci_hi",
               "paired_win_rate", "tpr_at_fpr5", "fpr_realized", "delta_vs_CoST",
               "delta_ci_lo", "delta_ci_hi", "n_windows", "n_participants"], rows)
    # Two thresholds, because the strict one is often undefined. delta* is the design's
    # headline -- 80% TPR at a 5% false-alarm rate, a demanding operating point that answers
    # "could this flag a deviation in practice". delta*_AUC is the smallest level at which the
    # score merely SEPARATES perturbed from clean (AUC >= 0.80); it is the weaker claim, but it
    # is defined far more often, so RQ2 still has a reportable number when delta* is None.
    # Report both -- a None delta* beside a finite delta*_AUC is itself the finding: the
    # deviation is visible but not at a usable false-alarm rate. Both carry their unit.
    res["delta_star"], res["delta_star_auc"] = {}, {}
    for name in views:
        for kind in PERTURBATIONS:
            got = [(r[2], r[3], r[7]) for r in rows if r[0] == name and r[1] == kind]
            res["delta_star"][f"{name}|{kind}"] = delta_star(got, kind, 2)      # TPR@FPR5
            res["delta_star_auc"][f"{name}|{kind}"] = delta_star(got, kind, 1)  # AUC

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, kind in zip(axes, PERTURBATIONS):
        for name in views:
            xs, ys, lo, hi = zip(*curves[(name, kind)])
            ln, = ax.plot(xs, ys, "o-", label=name)
            ax.fill_between(xs, lo, hi, color=ln.get_color(), alpha=0.15, lw=0)
        ax.axhline(0.5, ls=":", c="grey"); ax.set_ylim(0.4, 1.02)
        ax.set_xlabel(kind); ax.grid(alpha=0.25)
    axes[0].set_ylabel("detection AUC"); axes[0].legend(fontsize=8)
    fig.suptitle(f"RQ2 deviation detection -- {ctx.tag}", fontsize=11)
    fig.tight_layout(); fig.savefig(d / "rq2_detection.png", dpi=200); plt.close(fig)

    # --- E2.2 layer 2 + 3, on the CoST score ---------------------------------------------
    V = views["CoST"]
    mu, sd, ok = personal_baseline(V, ctx.pids, R)
    s = robust_z(np.where(ok, dscore(V, mu, sd), np.nan), ctx.pids, ok)
    # One convergent-validity row per CHANNEL: a phase drift in sleep and one in heart rate are
    # different findings, and averaging their angles first would report neither.
    _, AMP, ACRO = cosinor_params(Xs, ctx.bins_per_day)
    IS = _interdaily_stability(Xs, ctx.bins_per_day, per_channel=True)      # (N, C)
    names = list(ctx.sensor_cols[:ctx.n_sensors])
    # E2.2 layer 2 asks for THREE raw-signal deviations, not one: phase, amplitude and
    # interdaily stability. They are different chronobiological constructs -- a phase drift and
    # an amplitude collapse are separate findings -- so each is correlated with the latent
    # deviation score separately, per channel, and none is averaged into another.
    res["convergent_validity"] = {
        cname: within_person_rho(s, phase_deviation(ACRO[:, c], ctx.pids, R), ctx.pids, ok)
        for c, cname in enumerate(names)}
    res["convergent_validity_amplitude"] = {
        cname: within_person_rho(s, linear_deviation(AMP[:, c], ctx.pids, R), ctx.pids, ok)
        for c, cname in enumerate(names)}
    res["convergent_validity_interdaily_stability"] = {
        cname: within_person_rho(s, linear_deviation(IS[:, c], ctx.pids, R), ctx.pids, ok)
        for c, cname in enumerate(names)}

    res["energy_within_person"] = energy_within_person(s, ctx, ok)

    slopes, groups = {}, {}
    for p in np.unique(ctx.pids):
        m = ok & (ctx.pids == p) & np.isfinite(s)
        if m.sum() >= 4:
            slopes[p] = float(linregress(np.arange(m.sum()), s[m]).slope)
            groups[p] = ctx.trajectory_by_pid.get(p, ctx.trajectory_by_pid.get(str(p), "NA"))
    by_g = {}
    for p, sl in slopes.items():
        by_g.setdefault(groups[p], []).append(sl)
    by_g = {g: v for g, v in by_g.items() if g != "NA" and len(v) >= 3}
    res["trajectory_slope"] = {
        "median_slope_by_group": {g: float(np.median(v)) for g, v in by_g.items()},
        "n_by_group": {g: len(v) for g, v in by_g.items()},
        "kruskal_p": float(kruskal(*by_g.values()).pvalue) if len(by_g) >= 2 else None}

    # one illustrative participant: the held-out person with the most scored windows
    cand = [p for p in np.unique(ctx.pids[ctx.test_mask]) if (ok & (ctx.pids == p)).sum() >= 4]
    if cand:
        p0 = max(cand, key=lambda q: (ok & (ctx.pids == q)).sum())
        m = ok & (ctx.pids == p0)
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot(s[m], "o-", color="#0072B2")
        ax.axhline(3, ls="--", c="#D55E00", label="deviation flag (s=3)")
        ax.axhspan(-1, 1, color="grey", alpha=0.15, label="personal reference band")
        ax.set_xlabel("window index"); ax.set_ylabel("deviation score $s_{p,t}$")
        ax.set_title(f"participant {p0} -- {ctx.tag}", fontsize=10); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(d / "rq2_subject.png", dpi=200); plt.close(fig)

    save(d, "rq2", res)


if __name__ == "__main__":
    main()
