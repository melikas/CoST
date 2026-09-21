"""Figures 2-4 of the manuscript, drawn only from the canonical run's saved outputs.

    python scripts/paper_figures.py            # writes SSL_Rhythmicity/figures/*.pdf

Reads results/<dataset>/narval_v2/tcn_none/{rq1_family_intervals,rq2_by_level,
oof_predictions}.csv. Nothing is retrained.

The only statistic computed here that the pipeline does not already report is a 95% interval
on each method's OWN participant AUROC (the pipeline reports intervals on paired DSSL-minus-
method differences). It is a percentile participant bootstrap, stratified by label so every
draw contains both classes, with each draw's AUROC averaged over the seeds exactly as the
reported point estimate is. It is written to rq3_auroc_ci.csv beside the other tables.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "SSL_Rhythmicity" / "figures"
RUN = "narval_v2/tcn_none"

# Categorical slots in fixed order (validated: all six checks pass on a light surface).
# Colour follows the entity and is identical in every figure.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
NEUTRAL = "#8a8983"
INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#e6e5e0"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def recessive_grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------------------
# Figure 2 -- RQ1: marker recovery relative to the two required baselines
# --------------------------------------------------------------------------------------
MARKERS = [("MESOR", "MESOR"), ("amplitude", "24 h amplitude"), ("phase_hours", "Acrophase"),
           ("IS", "Interdaily stability"), ("IV", "Intradaily variability"),
           ("RA", "Relative amplitude")]


def figure_rq1():
    f = pd.read_csv(ROOT / "results/hrd" / RUN / "rq1_family_intervals.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3), sharey=True)
    for ax, control, title in ((axes[0], "training_mean", "vs. population-mean predictor"),
                               (axes[1], "untrained", "vs. untrained encoder")):
        s = f[f.control == control].set_index("marker")
        ys = np.arange(len(MARKERS))[::-1]
        for y, (key, _) in zip(ys, MARKERS):
            r = s.loc[key]
            conclusive = r.low > 0 or r.high < 0
            ax.plot([r.low, r.high], [y, y], color=BLUE, linewidth=1.6, solid_capstyle="round")
            ax.plot(r.difference, y, "o", markersize=5.5, color=BLUE,
                    markerfacecolor=BLUE if conclusive else "white", markeredgewidth=1.3)
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.set_yticks(ys)
        ax.set_yticklabels([label for _, label in MARKERS])
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("Error reduction (positive favours DSSL)")
        recessive_grid(ax)
    fig.text(0.0, -0.14, "Filled: 95% interval excludes zero.  Hollow: inconclusive.  "
             "HRD, 113 participants, 3 seeds x 5 folds.", color=INK2, fontsize=7)
    fig.savefig(OUT / "rq1_recovery.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Figure 3 -- RQ2: the perturbation design (top) and the detection result (bottom)
# --------------------------------------------------------------------------------------
RQ2_METHODS = [("dssl", "DSSL", BLUE, "o"), ("cost_reference_adapter", "CoST", AQUA, "D"),
               ("untrained", "Untrained", ORANGE, "s"),
               ("random_projection", "Random proj.", YELLOW, "^")]



def always_increase_floor():
    """Sign agreement of a rule that always predicts that amplification moves a week away from
    its reference, computed with the same pooled-per-level aggregation as rq2_by_level.csv so it
    is directly comparable with the plotted curves. The target sign does not depend on alpha,
    so the value is the same at every level."""
    import glob
    per_seed = []
    for path in sorted(glob.glob(str(ROOT / "results/hrd" / RUN / "seed_*/fold_*/rq2_personalized.csv"))):
        frame = pd.read_csv(path)
        frame = frame[(frame.method == "dssl") & (frame.perturbation == "amplitude")
                      & (frame.raw_delta != 0)]
        seed = path.replace("\\", "/").split("/seed_")[1].split("/")[0]
        per_seed.append(frame.assign(seed=seed))
    rows = pd.concat(per_seed, ignore_index=True)
    return float(rows.groupby("seed").raw_delta.apply(lambda d: (d > 0).mean()).mean())

def figure_rq2():
    fig = plt.figure(figsize=(6.6, 4.1))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.35], hspace=0.62, wspace=0.28)

    # Schematic: one channel's 24 h component, two days shown. This is an illustration of
    # the two perturbations, which are applied to the real windows; it plots no data.
    t = np.linspace(0, 48, 961)
    base = np.cos(2 * np.pi * (t - 14) / 24)
    a = fig.add_subplot(grid[0, 0])
    a.plot(t, base, color=INK2, linewidth=1.4, label="current week")
    a.plot(t, np.cos(2 * np.pi * (t - 14 - 3) / 24), color=BLUE, linewidth=1.8,
           label="phase shift (+3 h)")
    a.set_title("Phase shift", loc="left")
    b = fig.add_subplot(grid[0, 1], sharey=a)
    b.plot(t, base, color=INK2, linewidth=1.4, label="current week")
    b.plot(t, 0.6 * base, color=BLUE, linewidth=1.8, label=r"amplitude $\times(1-\alpha)$")
    b.set_title("Amplitude scaling", loc="left")
    for ax in (a, b):
        ax.set_xticks([0, 12, 24, 36, 48])
        ax.set_xlabel("Hours")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_ylim(-1.25, 1.9)
        ax.legend(frameon=False, loc="upper left", ncol=2, handlelength=1.6,
                  borderaxespad=0.0)

    levels = pd.read_csv(ROOT / "results/hrd" / RUN / "rq2_by_level.csv")
    floor = always_increase_floor()
    for col, (pert, xlabel, title, ylabel) in enumerate((
            ("phase", "Phase shift (hours)", "Detection of phase shifts", "Concordance"),
            ("amplitude", r"Amplitude change $\alpha$", "Detection of amplitude changes",
             "Sign agreement"))):
        ax = fig.add_subplot(grid[1, col])
        block = levels[levels.perturbation == pert]
        for key, label, colour, marker in RQ2_METHODS:
            g = block[block.method == key].groupby("level").concordance
            m, sd = g.mean(), g.std(ddof=0)
            ax.errorbar(m.index, m.values, yerr=sd.values, color=colour, marker=marker,
                        markersize=4.5, linewidth=1.6, capsize=0, elinewidth=1.0,
                        label=label, zorder=3 if key == "dssl" else 2)
        # The honest reference differs by arm: 0.5 for the rank statistic, but for sign agreement
        # a rule that always answers "away" already scores the share of weeks moved away.
        ref, ref_label = (0.5, " chance") if pert == "phase" else (floor, " always-increase rule")
        ax.axhline(ref, color=INK2, linewidth=0.8, linestyle=(0, (3, 2)))
        right = pert != "phase"   # amplitude: the open space is on the right, above the floor
        ax.text(ax.get_xlim()[1 if right else 0], ref + 0.005, ref_label + (" " if right else ""),
                color=INK2, fontsize=6.8, va="bottom", ha="right" if right else "left")
        ax.set_ylim(0.45, 0.9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        recessive_grid(ax, "y")
        ax.margins(x=0.08)
        if col == 0:
            handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 0.02), handlelength=1.8)
    fig.text(0.0, -0.085, "Points: mean over 3 seeds; bars: seed SD. HRD: 102 (phase) and 106 "
             "(amplitude) participants. All amplitude levels share one target sign per week.",
             color=INK2, fontsize=7)
    fig.savefig(OUT / "rq2_perturbation.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Figure 4 -- RQ3: participant AUROC with a 95% interval for every method, both cohorts
# --------------------------------------------------------------------------------------
RQ3_METHODS = [("dssl", "DSSL (proposed)", BLUE), ("raw", "Raw window", ORANGE),
               ("untrained", "Untrained encoder", ORANGE),
               ("yan_cosinor", "Cosinor, reimplemented", NEUTRAL),
               ("handcrafted", "Handcrafted rhythm", NEUTRAL),
               ("nonparametric", "Nonparametric (IS, IV, RA)", NEUTRAL),
               ("random_projection", "Random projection", NEUTRAL),
               ("cost_reference_adapter", "CoST", NEUTRAL),
               ("pca", "PCA", NEUTRAL)]


def auroc_interval(frame, draws=2000, seed=0):
    """Point AUROC (mean over seeds) and a stratified participant-bootstrap 95% interval."""
    wide = frame.pivot(index="participant", columns="seed", values="probability")
    y = frame.groupby("participant").label.first().reindex(wide.index).to_numpy()
    scores = wide.to_numpy()
    point = float(np.mean([roc_auc_score(y, scores[:, j]) for j in range(scores.shape[1])]))
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    stats = np.empty(draws)
    for i in range(draws):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        stats[i] = np.mean([roc_auc_score(y[idx], scores[idx, j])
                            for j in range(scores.shape[1])])
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return point, float(lo), float(hi), len(y)


def figure_rq3():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
    for ax, dataset, title in ((axes[0], "hrd", "HRD"), (axes[1], "globem", "GLOBEM")):
        run = ROOT / "results" / dataset / RUN
        oof = pd.read_csv(run / "oof_predictions.csv", dtype={"participant": str})
        oof = oof[oof.probe == "logistic"]
        rows = []
        ys = np.arange(len(RQ3_METHODS))[::-1]
        for y, (key, label, colour) in zip(ys, RQ3_METHODS):
            point, lo, hi, n = auroc_interval(oof[oof.method == key])
            rows.append(dict(method=key, auroc=point, low=lo, high=hi, participants=n))
            ax.plot([lo, hi], [y, y], color=colour, linewidth=1.6, solid_capstyle="round")
            ax.plot(point, y, "o", color=colour, markersize=5.5 if key == "dssl" else 4.5,
                    zorder=3)
        pd.DataFrame(rows).to_csv(run / "rq3_auroc_ci.csv", index=False)
        ax.axvline(0.5, color=INK2, linewidth=0.8, linestyle=(0, (3, 2)))
        ax.text(0.505, ys[-1] - 0.75, "chance", color=INK2, fontsize=6.8, va="center")
        ax.set_ylim(ys[-1] - 1.1, ys[0] + 0.6)
        ax.set_yticks(ys)
        ax.set_yticklabels([label for _, label, _ in RQ3_METHODS])
        ax.set_xlim(0.3, 0.9)
        ax.set_xlabel("Participant AUROC")
        ax.set_title(f"{title}, n = {rows[0]['participants']}", loc="left")
        recessive_grid(ax)
        print(f"{title}:")
        for r in rows:
            print(f"  {r['method']:24s} {r['auroc']:.3f} [{r['low']:.3f}, {r['high']:.3f}]")
    handles = [plt.Line2D([], [], color=c, marker="o", linewidth=1.6, markersize=4.5)
               for c in (BLUE, ORANGE, NEUTRAL)]
    fig.legend(handles, ["Proposed", "Baselines the criterion requires DSSL to exceed",
                         "Further reference points"], frameon=False, loc="lower center",
               ncol=3, bbox_to_anchor=(0.55, -0.1))
    fig.savefig(OUT / "rq3_auroc.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    figure_rq1()
    figure_rq2()
    figure_rq3()
    print(f"wrote {sorted(p.name for p in OUT.glob('rq*.pdf'))}")
