"""Result figures of the manuscript (RQ1-RQ3), drawn only from the canonical run's saved outputs.

    python scripts/paper_figures.py            # writes SSL_Rhythmicity/figures/rq*.pdf

Every plotted number uses the same aggregation as the matching manuscript table:
  RQ1  rq1_errors.csv: per-participant error, averaged over channels, then over seeds. The
       differences between two methods' points are exactly Table 'tab:rq1'.
  RQ2  phase: rq2_by_level.csv (the stratified concordance restricted to one shift size).
       amplitude: sign agreement averaged within participant, then across participants, from
       rq2_personalized.csv -- the table's aggregation. The floor uses the same aggregation.
  RQ3  oof_predictions.csv (own AUROC, mean over seeds, with a label-stratified participant
       bootstrap) and rq3_paired_intervals.csv (paired DSSL-minus-method intervals), for HRD,
       GLOBEM 2018 (narval_v2) and GLOBEM leave-one-year-out (loyo_v1, predictions pooled over
       the four held-out years).
Nothing is retrained. Own-AUROC intervals are written to rq3_auroc_ci.csv.
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
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

# One colour and name per method, in every figure.
METHODS = {"dssl": ("DSSL", BLUE), "untrained": ("Untrained encoder", ORANGE),
           "raw": ("Raw window", YELLOW), "cost_reference_adapter": ("CoST", AQUA),
           "training_mean": ("Population mean", NEUTRAL),
           "random_projection": ("Random projection", NEUTRAL)}


def recessive_grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def rq2_rows(dataset="hrd"):
    frames = []
    for path in sorted(glob.glob(str(ROOT / "results" / dataset / RUN / "seed_*/fold_*/rq2_personalized.csv"))):
        f = pd.read_csv(path, dtype={"participant": str})
        frames.append(f.assign(seed=Path(path).parent.parent.name))
    return pd.concat(frames, ignore_index=True)


def amplitude_agreement(rows):
    """Sign agreement per method and level, averaged within participant, then across
    participants, then over seeds (the aggregation of the RQ2 table). Ties count one half."""
    a = rows[(rows.perturbation == "amplitude") & (rows.raw_delta != 0)].copy()
    same = np.sign(a.representation_delta) == np.sign(a.raw_delta)
    a["agree"] = np.where(a.representation_delta == 0, 0.5, same.astype(float))
    return (a.groupby(["method", "level", "seed", "participant"]).agree.mean()
             .groupby(["method", "level", "seed"]).mean().groupby(["method", "level"]).mean())


def phase_concordance(rows):
    """Phase-shift concordance per method and shift size with the table's aggregation: the
    stratified Mann-Whitney concordance of each participant at that shift, averaged over
    participants, then over seeds (evaluation_protocol.summarize computes the same per participant
    over all shifts)."""
    from tasks.rhythm import stratum_pairs, concordance
    g = rows[rows.perturbation == "phase"]
    out = []
    for (method, level, seed, pid), h in g.groupby(["method", "level", "seed", "participant"]):
        value = concordance(stratum_pairs(h.representation_delta.to_numpy(), h.raw_delta.to_numpy(),
                                          np.full(len(h), pid)))
        if np.isfinite(value):
            out.append((method, level, seed, value))
    f = pd.DataFrame(out, columns=["method", "level", "seed", "c"])
    return f.groupby(["method", "level", "seed"]).c.mean().groupby(["method", "level"]).mean()


def always_increase_floor(rows=None):
    """Sign agreement of the rule 'amplification always moves the week away', with the table's
    aggregation. The target sign does not depend on alpha, so the value is the same at every
    level."""
    rows = rq2_rows() if rows is None else rows
    a = rows[(rows.method == "dssl") & (rows.perturbation == "amplitude") & (rows.raw_delta != 0)]
    return float(a.assign(away=(a.raw_delta > 0).astype(float))
                 .groupby(["seed", "participant"]).away.mean().groupby("seed").mean().mean())


# --------------------------------------------------------------------------------------
# RQ1: held-out error of each method, per marker, with participant-bootstrap intervals
# --------------------------------------------------------------------------------------
RQ1_MARKERS = [("amplitude", "24 h amplitude"), ("IS", "Interdaily stability"),
               ("IV", "Intradaily variability"), ("RA", "Relative amplitude"), ("MESOR", "MESOR")]
RQ1_METHODS = ["training_mean", "untrained", "dssl", "raw", "cost_reference_adapter"]


def rq1_person_errors():
    e = pd.read_csv(ROOT / "results/hrd" / RUN / "rq1_errors.csv", dtype={"participant": str})
    return (e.groupby(["seed", "method", "participant", "marker"]).family_error.mean()
             .groupby(["method", "participant", "marker"]).mean())


def mean_ci(x, rng, draws=2000):
    x = np.asarray(x, float)
    boot = [rng.choice(x, len(x)).mean() for _ in range(draws)]
    return x.mean(), *np.percentile(boot, [2.5, 97.5])


def figure_rq1():
    person = rq1_person_errors()
    rng = np.random.default_rng(20260914)
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.7), gridspec_kw=dict(width_ratios=[2.6, 1], wspace=0.35))
    offsets = np.linspace(-0.3, 0.3, len(RQ1_METHODS))
    for ax, markers, unit in ((a, RQ1_MARKERS, "Error (training-SD units)"),
                              (b, [("phase_hours", "Acrophase")], "Error (h)")):
        ys = np.arange(len(markers))[::-1]
        for off, method in zip(offsets, RQ1_METHODS):
            label, colour = METHODS[method]
            for y, (key, _) in zip(ys, markers):
                m, lo, hi = mean_ci(person.loc[method, :, key].to_numpy(), rng)
                ax.plot([lo, hi], [y - off] * 2, color=colour, linewidth=1.4)
                ax.plot(m, y - off, "o", color=colour, markersize=4.5 if method == "dssl" else 3.5,
                        label=label if (key == markers[0][0]) else None, zorder=3)
        ax.set_yticks(ys)
        ax.set_yticklabels([name for _, name in markers])
        ax.set_xlim(left=0)
        ax.set_xlabel(unit + "\nlower is better")
        recessive_grid(ax)
    a.set_title("a  Markers in training-SD units", loc="left")
    b.set_title("b  Peak time", loc="left")
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=5, bbox_to_anchor=(0.5, -0.06))
    fig.savefig(OUT / "rq1_recovery.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# RQ2: the two perturbations (top) and detection by perturbation size (bottom)
# --------------------------------------------------------------------------------------
RQ2_METHODS = ["dssl", "cost_reference_adapter", "untrained", "random_projection"]


def figure_rq2():
    rows = rq2_rows()
    fig = plt.figure(figsize=(6.6, 4.1))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.35], hspace=0.62, wspace=0.28)
    # Illustration on a pure 24 h component over two days; the experiment perturbs real weeks.
    t = np.linspace(0, 48, 961)
    base = np.cos(2 * np.pi * (t - 14) / 24)
    a = fig.add_subplot(grid[0, 0])
    a.plot(t, base, color=INK2, linewidth=1.4, label="current week")
    a.plot(t, np.cos(2 * np.pi * (t - 17) / 24), color=BLUE, linewidth=1.8, label="shifted 3 h later")
    a.set_title("Phase shift: timing changes", loc="left")
    b = fig.add_subplot(grid[0, 1], sharey=a)
    b.plot(t, base, color=INK2, linewidth=1.4, label="current week")
    b.plot(t, 1.3 * base, color=BLUE, linewidth=1.8, label=r"$\times(1+\alpha)$")
    b.plot(t, 0.7 * base, color=BLUE, linewidth=1.2, linestyle=(0, (3, 2)), label=r"$\times(1-\alpha)$")
    b.set_title("Amplitude scaling: strength changes", loc="left")
    for ax in (a, b):
        ax.set_xticks([0, 12, 24, 36, 48])
        ax.set_xlabel("Hours")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_ylim(-1.45, 2.3)
        ax.legend(frameon=False, loc="upper left", ncol=3, handlelength=1.6, borderaxespad=0.0)

    phase = phase_concordance(rows)
    amp = amplitude_agreement(rows)
    floor = always_increase_floor(rows)
    for col, (series, xlabel, title, ylabel, ref, ref_label) in enumerate((
            (phase, "Shift (hours)", "Detection of timing changes", "Concordance", 0.5, "chance"),
            (amp, r"Strength change $\alpha$", "Detection of strength changes", "Sign agreement",
             floor, f"always-'away' rule ({floor:.3f})"))):
        ax = fig.add_subplot(grid[1, col])
        for key in RQ2_METHODS:
            label, colour = METHODS[key]
            s = series.loc[key]
            ax.plot(s.index, s.values, color=colour, marker="o", markersize=4, linewidth=1.6,
                    label=label, zorder=3 if key == "dssl" else 2)
        ax.axhline(ref, color=INK, linewidth=0.9, linestyle=(0, (1, 1.5)), zorder=1)
        ax.text(ax.get_xlim()[1], ref - 0.012, ref_label + " ", color=INK, fontsize=6.8, va="top", ha="right")
        ax.set_ylim(0.45, 0.9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        recessive_grid(ax, "y")
        if col == 0:
            handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.02))
    fig.savefig(OUT / "rq2_perturbation.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# RQ3: own AUROC and paired difference to DSSL, both cohorts
# --------------------------------------------------------------------------------------
RQ3_METHODS = [("dssl", "DSSL"), ("raw", "Raw window"), ("untrained", "Untrained encoder"),
               ("yan_cosinor", "Cosinor (Yan et al.)"), ("handcrafted", "Handcrafted rhythm"),
               ("nonparametric", "Nonparametric (IS, IV, RA)"), ("distribution", "Channel mean and SD"),
               ("random_projection", "Random projection"),
               ("cost_reference_adapter", "CoST"), ("pca", "PCA")]


def auroc_interval(frame, draws=2000, seed=0):
    """Point AUROC (mean over seeds) and a label-stratified participant-bootstrap 95% interval."""
    wide = frame.pivot(index="participant", columns="seed", values="probability")
    y = frame.groupby("participant").label.first().reindex(wide.index).to_numpy()
    scores = wide.to_numpy()
    point = float(np.mean([roc_auc_score(y, scores[:, j]) for j in range(scores.shape[1])]))
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    stats = np.empty(draws)
    for i in range(draws):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        stats[i] = np.mean([roc_auc_score(y[idx], scores[idx, j]) for j in range(scores.shape[1])])
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return point, float(lo), float(hi), len(y)


def figure_rq3():
    fig, axes = plt.subplots(3, 2, figsize=(6.8, 6.6), sharey="row",
                             gridspec_kw=dict(hspace=0.55, wspace=0.12))
    for row, dataset, run_name, name in ((0, "hrd", RUN, "HRD"),
                                         (1, "globem", RUN, "GLOBEM 2018"),
                                         (2, "globem", "loyo_v1/tcn_none", "GLOBEM, leave one year out")):
        run = ROOT / "results" / dataset / run_name
        oof = pd.read_csv(run / "oof_predictions.csv", dtype={"participant": str})
        oof = oof[oof.probe == "logistic"]
        paired = pd.read_csv(run / "rq3_paired_intervals.csv")
        paired = paired[paired.probe == "logistic"].set_index("control")
        ys = np.arange(len(RQ3_METHODS))[::-1]
        rows = []
        for y, (key, label) in zip(ys, RQ3_METHODS):
            point, lo, hi, n = auroc_interval(oof[oof.method == key])
            rows.append(dict(method=key, auroc=point, low=lo, high=hi, participants=n))
            colour = BLUE if key == "dssl" else (ORANGE if key in ("raw", "untrained") else NEUTRAL)
            axes[row, 0].plot([lo, hi], [y, y], color=colour, linewidth=1.5)
            axes[row, 0].plot(point, y, "o", color=colour, markersize=4.5)
            if key != "dssl":
                r = paired.loc[key]
                solid = r.low > 0 or r.high < 0
                axes[row, 1].plot([r.low, r.high], [y, y], color=colour, linewidth=1.5)
                axes[row, 1].plot(r.difference, y, "o", color=colour, markersize=4.5,
                                  markerfacecolor=colour if solid else "white")
        pd.DataFrame(rows).to_csv(run / "rq3_auroc_ci.csv", index=False)
        n = rows[0]["participants"]
        axes[row, 0].axvline(0.5, color=INK2, linewidth=0.7, linestyle=(0, (3, 2)))
        axes[row, 1].axvline(0, color=INK2, linewidth=0.7)
        axes[row, 0].set_yticks(ys)
        axes[row, 0].set_yticklabels([label for _, label in RQ3_METHODS])
        axes[row, 0].set_xlim(0.3, 0.9)
        axes[row, 1].set_xlim(-0.3, 0.3)
        axes[row, 0].set_title(f"{'ace'[row]}  {name} ({n}): AUROC", loc="left")
        axes[row, 1].set_title(f"{'bdf'[row]}  DSSL minus method (paired)", loc="left")
        for ax in axes[row]:
            recessive_grid(ax)
    axes[2, 0].set_xlabel("Participant AUROC (dashed: chance)")
    axes[2, 1].set_xlabel("AUROC difference (positive favours DSSL)")
    fig.savefig(OUT / "rq3_auroc.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    figure_rq1()
    figure_rq2()
    figure_rq3()
    print(f"wrote {sorted(p.name for p in OUT.glob('rq*.pdf'))}")
