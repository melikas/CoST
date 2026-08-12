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
from scipy.stats import kruskal, spearmanr, linregress
from sklearn.metrics import roc_auc_score

from tasks._experiment_common import (encode, load_context, out_dir, random_init_model,
                                      save, write_csv)
from tasks.decomposition import harmonic_reference
from tasks.rhythm import _interdaily_stability

PERTURBATIONS = ("phase_shift_h", "amplitude_damping", "sleep_fragmentation")


def raw_markers(Xs, bpd):
    """Per-window 24-h cosinor amplitude and acrophase (amplitude-weighted circular mean over
    channels) plus interdaily stability -- computed from the RAW signal, never the model."""
    t = 2 * np.pi * np.arange(Xs.shape[1]) / bpd
    Z = Xs - Xs.mean(1, keepdims=True)
    a = 2 * (Z * np.cos(t)[None, :, None]).mean(1)
    b = 2 * (Z * np.sin(t)[None, :, None]).mean(1)
    A = np.hypot(a, b)
    return A.mean(1), np.angle((A * np.exp(1j * np.arctan2(b, a))).sum(1)), \
        _interdaily_stability(Xs, bpd)


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


def detect_auc(clean, pert):
    """Paired detection: same windows, unperturbed vs perturbed."""
    m = np.isfinite(clean) & np.isfinite(pert)
    if m.sum() < 10:
        return np.nan, np.nan
    auc = roc_auc_score(np.r_[np.zeros(m.sum()), np.ones(m.sum())], np.r_[clean[m], pert[m]])
    return auc, float((pert[m] > np.percentile(clean[m], 95)).mean())   # TPR at FPR=5%


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant-dir", required=True)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--ref-windows", type=int, default=4, help="Personal reference length (4 x 7d = 28d)")
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
                   default=[0.9, 0.7, 0.5, 0.3, 0.1, 0.0])
    p.add_argument("--frag-levels", type=float, nargs="+",
                   default=[7, 14, 28, 56, 112, 224])
    a = p.parse_args()

    ctx = load_context(a.variant_dir, a.cache_dir, a.gpu)
    d = out_dir(ctx, "rq2")
    rng = np.random.default_rng(ctx.seed)
    Xs = ctx.X[:, :, :ctx.n_sensors]
    _, sig = harmonic_reference(Xs, 24 * 60.0 / ctx.bin_minutes)
    sleep_ix = next((i for i, c in enumerate(ctx.sensor_cols) if "asleep" in c or "sleep" in c),
                    None)
    R = a.ref_windows

    # One representation FUNCTION per baseline, built once: the random-init control must keep
    # the same untrained weights across every perturbation level, and rebuilding it per level
    # would silently redraw them.
    rand = random_init_model(ctx)
    reps = {"CoST": lambda A: encode(ctx.model, A, ctx.cfg),
            "Handcrafted": lambda A: np.concatenate(
                [A[:, :, :ctx.n_sensors].mean(1), A[:, :, :ctx.n_sensors].std(1)], axis=1),
            "Random-init": lambda A: encode(rand, A, ctx.cfg)}
    views = {name: fn(ctx.X) for name, fn in reps.items()}
    res = {"variant": ctx.tag, "seed": ctx.seed, "ref_windows": R}

    # --- E2.2 layer 1: detection of synthetic perturbations, on held-out participants -----
    levels = {"phase_shift_h": a.phase_levels, "amplitude_damping": a.amp_levels,
              "sleep_fragmentation": a.frag_levels}
    rows, curves = [], {}
    for name, V in views.items():
        mu, sd, ok = personal_baseline(V, ctx.pids, R)
        elig = ok & ctx.test_mask
        clean = np.where(elig, dscore(V, mu, sd), np.nan)
        for kind in PERTURBATIONS:
            for lv in levels[kind]:
                Xp = perturb(ctx.X, kind, lv, ctx.n_sensors, sig, sleep_ix,
                             ctx.bins_per_day, ctx.bin_minutes, rng)
                # Only the scored window is perturbed; its reference mu/sd stays clean, so the
                # score measures detection rather than a shifted baseline.
                Vp = V.copy()
                Vp[elig] = reps[name](Xp[elig])
                pert = np.where(elig, dscore(Vp, mu, sd), np.nan)
                auc, tpr = detect_auc(clean, pert)
                rows.append([name, kind, lv, round(auc, 4), round(tpr, 4), int(elig.sum())])
                curves.setdefault((name, kind), []).append((lv, auc))
        print(f"[rq2] {name}: {int(elig.sum())} scored held-out windows")

    write_csv(d, "rq2_detection", ["representation", "perturbation", "level", "auc",
                                   "tpr_at_fpr5", "n_windows"], rows)
    # Two thresholds, because the strict one is often undefined. delta* is the design's
    # headline -- 80% TPR at a 5% false-alarm rate, a demanding operating point that answers
    # "could this flag a deviation in practice". delta*_AUC is the smallest level at which the
    # score merely SEPARATES perturbed from clean (AUC >= 0.80); it is the weaker claim, but it
    # is defined far more often, so RQ2 still has a reportable number when delta* is None.
    # Report both -- a None delta* beside a finite delta*_AUC is itself the finding: the
    # deviation is visible but not at a usable false-alarm rate.
    res["delta_star"], res["delta_star_auc"] = {}, {}
    for name in views:
        for kind in PERTURBATIONS:
            got = [(r[2], r[3], r[4]) for r in rows if r[0] == name and r[1] == kind]
            first = lambda hits: min(hits) if hits else None      # None = never reached
            res["delta_star"][f"{name}|{kind}"] = first(
                [lv for lv, _, tpr in got if tpr >= 0.80])
            res["delta_star_auc"][f"{name}|{kind}"] = first(
                [lv for lv, auc, _ in got if auc >= 0.80])

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, kind in zip(axes, PERTURBATIONS):
        for name in views:
            xs, ys = zip(*curves[(name, kind)])
            ax.plot(xs, ys, "o-", label=name)
        ax.axhline(0.5, ls=":", c="grey"); ax.set_ylim(0.4, 1.02)
        ax.set_xlabel(kind); ax.grid(alpha=0.25)
    axes[0].set_ylabel("detection AUC"); axes[0].legend(fontsize=8)
    fig.suptitle(f"RQ2 deviation detection -- {ctx.tag}", fontsize=11)
    fig.tight_layout(); fig.savefig(d / "rq2_detection.png", dpi=200); plt.close(fig)

    # --- E2.2 layer 2 + 3, on the CoST score ---------------------------------------------
    V = views["CoST"]
    mu, sd, ok = personal_baseline(V, ctx.pids, R)
    s = robust_z(np.where(ok, dscore(V, mu, sd), np.nan), ctx.pids, ok)
    amp, acro, IS = raw_markers(Xs, ctx.bins_per_day)

    dphi = np.full(len(s), np.nan)
    for p in np.unique(ctx.pids):
        idx = np.where(ctx.pids == p)[0]
        for j in range(R, len(idx)):
            ref = np.angle(np.exp(1j * acro[idx[j - R:j]]).mean())
            dphi[idx[j]] = np.abs(np.angle(np.exp(1j * (acro[idx[j]] - ref))))
    rho = []
    for p in np.unique(ctx.pids):
        m = ok & (ctx.pids == p) & np.isfinite(s) & np.isfinite(dphi)
        if m.sum() >= 5:
            r = spearmanr(s[m], dphi[m]).correlation
            if np.isfinite(r):
                rho.append(r)
    res["convergent_validity"] = {"median_within_person_spearman": float(np.median(rho)) if rho else None,
                                  "n_participants": len(rho),
                                  "frac_positive": float(np.mean(np.array(rho) > 0)) if rho else None}

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
