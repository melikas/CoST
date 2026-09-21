"""Main-text figures of the manuscript, drawn only from saved outputs of the canonical run.

    python scripts/manuscript_figures.py     # writes SSL_Rhythmicity/figures/fig*.pdf

Figure 1 model overview; 2 representation anatomy; 3 RQ1 schematic on a real participant;
4 RQ2 design and result; 5 depressed vs non-depressed representations; 6 transitions;
7 group-level rhythm comparison; 8 RQ3 with paired uncertainty. Figures 5-7 read
results/hrd/narval_v2/tcn_none/group_analysis (scripts/group_analysis.py). Nothing is retrained.
The example participant is chosen by a fixed data-quality rule, never by a result, and is
shown without its identifier.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datautils import load_npz
from tasks.rhythm import amplitude_scale, cosinor_z, individual_markers, phase_shift
from scripts.paper_figures import (BLUE, ORANGE, AQUA, YELLOW, NEUTRAL, INK, INK2, GRID,
                                   recessive_grid, auroc_interval, always_increase_floor)

OUT = ROOT / "SSL_Rhythmicity" / "figures"
RUN = ROOT / "results/hrd/narval_v2/tcn_none"
GROUPS = RUN / "group_analysis"
CH_LABEL = {"HR": "Heart rate (bpm)", "Steps": "Steps / 15 min", "is_asleep": "Asleep",
            "screen": "Screen (min / 15 min)"}
CH_SHORT = {"HR": "Heart rate", "Steps": "Steps", "is_asleep": "Sleep", "screen": "Screen"}
# Group colours: fixed per group in every figure.
G_COL = {"stable_non_depressed": BLUE, "stable_depressed": ORANGE,
         "became_depressed": AQUA, "recovered": YELLOW}
G_LAB = {"stable_non_depressed": "Stable non-depressed", "stable_depressed": "Stable depressed",
         "became_depressed": "Became depressed", "recovered": "Recovered"}
G_ORDER = list(G_LAB)


def cohort():
    return load_npz(ROOT / "datasets/cache/hrd_rescue_v1.npz")


def example_participant(c):
    """Fixed rule: the labelled participant with >= 10 windows whose worst-channel observed
    coverage, averaged over windows, is highest; its best-covered window with >= 4 earlier
    contiguous weeks (so RQ2 can score it)."""
    ids, _ = c.participants()
    best, score = None, -1
    for p in ids:
        idx = np.flatnonzero(c.pids == p)
        if len(idx) >= 10:
            s = c.observed[idx].mean(1).min(1).mean()
            if s > score:
                best, score = p, s
    idx = np.flatnonzero(c.pids == best)
    dates = np.array([np.datetime64(str(w).rsplit("_", 1)[1]) for w in c.window_ids[idx]])
    ok = [j for j in range(4, len(idx))
          if np.all(np.diff(dates[j - 4:j + 1]) == np.timedelta64(7, "D"))]
    cov = c.observed[idx[ok]].mean(1).min(1)
    return best, idx, int(idx[ok][int(np.argmax(cov))])


def heldout_fold(pid, seed=1):
    for f in range(5):
        m = json.loads((RUN / f"seed_{seed}" / f"fold_{f}" / "manifest.json").read_text())
        if pid in m["test_ids"]:
            return f
    raise ValueError(pid)


def box(ax, x, y, w, h, text, face, edge=INK2, size=7.2, weight="normal", color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=face, edgecolor=edge, linewidth=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            weight=weight, color=color, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8,
                                 color=INK2, linewidth=0.9))


# ---------------------------------------------------------------------------- Figure 1
def figure_overview(c):
    pid, idx, w = example_participant(c)
    fig = plt.figure(figsize=(7.0, 3.3))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 4.7)
    ax.axis("off")
    ax.text(0.05, 4.5, "a  Sensing", weight="bold", fontsize=8)
    ax.text(2.05, 4.5, "b  Weekly windows", weight="bold", fontsize=8)
    ax.text(4.05, 4.5, "c  Self-supervised encoder", weight="bold", fontsize=8)
    ax.text(6.6, 4.5, "d  Representation", weight="bold", fontsize=8)
    ax.text(8.45, 4.5, "e  What we test", weight="bold", fontsize=8)
    # a: three real days of the example participant, one row per channel
    x = c.raw_X[w][: 3 * c.bins_per_day]
    t = np.arange(len(x)) / c.bins_per_day
    for k, name in enumerate(c.sensor_cols):
        sub = fig.add_axes([0.015, 0.73 - k * 0.18, 0.17, 0.13])
        sub.plot(t, x[:, k], color=INK, linewidth=0.6)
        sub.set_xticks([])
        sub.set_yticks([])
        for s in sub.spines.values():
            s.set_visible(False)
        sub.text(0, 1.0, CH_SHORT[name], transform=sub.transAxes, fontsize=6.5, color=INK2,
                 va="bottom")
    ax.text(0.95, 0.2, "3 of the 7 days shown", ha="center", fontsize=6.3, color=INK2)
    arrow(ax, 1.95, 2.35, 2.15, 2.35)
    box(ax, 2.15, 1.35, 1.6, 2.0, "15-min bins\n\n7-day window:\n672 time steps\n× 4 channels\n\n"
        "standardised within\neach participant", "#f4f3ef")
    arrow(ax, 3.8, 2.35, 4.0, 2.35)
    # c: encoder
    box(ax, 4.0, 3.3, 2.35, 0.75, "Shared temporal network\n(sees the whole week)", "#e9f1fb",
        edge=BLUE)
    arrow(ax, 4.6, 3.3, 4.6, 2.85)
    arrow(ax, 5.75, 3.3, 5.75, 2.85)
    box(ax, 4.0, 2.05, 1.12, 0.8, "Context\nbranch\n(slow change)", "#fdeee6", edge=ORANGE,
        size=6.6)
    box(ax, 5.23, 2.05, 1.12, 0.8, "Rhythm branch\n(24, 12, 8, 6 h\nfilters only)", "#e6f6ef",
        edge=AQUA, size=6.6)
    box(ax, 4.0, 0.55, 2.35, 1.15, "Training without labels:\ntwo noisy copies of the same\n"
        "week must look alike;\nrhythm strength is never\ntreated as noise", "#f4f3ef", size=6.4)
    # d: representation, three named parts
    arrow(ax, 6.4, 2.45, 6.6, 2.45)
    for k, (label, colour, face) in enumerate([("Context\n(160)", ORANGE, "#fdeee6"),
                                               ("Rhythm strength\n(amplitude)", AQUA, "#e6f6ef"),
                                               ("Rhythm timing\n(phase)", AQUA, "#e6f6ef")]):
        box(ax, 6.6, 3.2 - k * 0.8, 1.45, 0.65, label, face, edge=colour, size=6.6)
    ax.text(7.32, 0.75, "one vector per week;\naveraged over weeks\nfor one per person",
            ha="center", fontsize=6.3, color=INK2)
    # e: evaluations
    for k, text in enumerate(["RQ1  Is each person's\nrhythm recoverable?",
                              "RQ2  Does it move when the\nperson's own rhythm changes?",
                              "RQ3  Does it identify\ndepressive symptoms?",
                              "Groups  How does it differ\nby depression course?"]):
        arrow(ax, 8.1, 2.45, 8.45, 3.56 - k * 0.85)
        box(ax, 8.45, 3.25 - k * 0.85, 2.1, 0.62, text, "white", size=6.3)
    fig.savefig(OUT / "fig1_overview.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------- Figure 2
def rep_and_markers(c):
    """Seed-1 held-out representations of every labelled window, with measured markers."""
    ids, _ = c.participants()
    keep = np.isin(c.pids, ids)
    V = np.zeros((keep.sum(), 1760), np.float32)
    pids = c.pids[keep]
    for f in range(5):
        z = np.load(RUN / "seed_1" / f"fold_{f}" / "representations.npz", allow_pickle=True)
        test = set(json.loads((RUN / "seed_1" / f"fold_{f}" / "manifest.json").read_text())["test_ids"])
        rows = np.isin(pids, list(test))
        V[rows] = z["dssl"][keep][rows]
    m = individual_markers(c.raw_X[keep], c.observed[keep], c.bins_per_day, list(c.sensor_cols))["values"]
    return V, pids, m, keep


def figure_anatomy(c):
    pid, idx, w = example_participant(c)
    fold = heldout_fold(pid)
    z = np.load(RUN / "seed_1" / f"fold_{fold}" / "representations.npz", allow_pickle=True)
    v = z["dssl"][w]
    amp = v[160:960].reshape(5, 160)
    ph = v[960:].reshape(5, 160)
    fig = plt.figure(figsize=(7.0, 5.2))
    g = fig.add_gridspec(3, 4, height_ratios=[1.05, 1, 1], hspace=1.25, wspace=0.6)
    # a: the raw week
    a = fig.add_subplot(g[0, :])
    x = c.raw_X[w]
    t = np.arange(len(x)) / c.bins_per_day
    for k, (name, col) in enumerate(zip(c.sensor_cols, (INK, BLUE, NEUTRAL, ORANGE))):
        s = (x[:, k] - x[:, k].min()) / (np.ptp(x[:, k]) + 1e-9)
        a.plot(t, s * 0.8 + 3 - k, color=col, linewidth=0.55)
        a.text(-0.08, 3.35 - k, CH_SHORT[name], ha="right", va="center", fontsize=6.8, color=INK2)
    a.set_yticks([])
    a.spines["left"].set_visible(False)
    a.set_xlim(0, 7)
    a.set_xlabel("Day of the window")
    a.set_title("a  One participant-week (each channel scaled to its own range)", loc="left")
    # b: strength at each readout frequency
    b = fig.add_subplot(g[1, 0:2])
    labels = ["weekly", "24 h", "12 h", "8 h", "6 h"]
    support = [range(0, 40), range(0, 40), range(40, 80), range(80, 120), range(120, 160)]
    for k, rows in enumerate(support):
        vals = amp[k, list(rows)]
        b.scatter(np.full(len(vals), k) + np.linspace(-0.25, 0.25, len(vals)), vals, s=5,
                  color=AQUA if k == 1 else NEUTRAL, linewidth=0)
    b.set_yscale("log")
    b.set_xticks(range(5))
    b.set_xticklabels(labels)
    b.set_ylabel("Amplitude (log)")
    b.set_title("b  Rhythm strength: 40 coordinates per period", loc="left")
    recessive_grid(b, "y")
    # c: timing at 24 h, polar
    cpol = fig.add_subplot(g[1, 2], projection="polar")
    r = amp[1, :40]
    cpol.scatter(ph[1, :40], r / r.max(), s=9, color=AQUA, linewidth=0)
    cpol.set_yticks([])
    cpol.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    cpol.set_xticklabels(["0", "π/2", "π", "3π/2"], fontsize=6)
    cpol.set_title("c  24 h timing", loc="left", fontsize=7.5, pad=10)
    cpol.text(0.5, -0.42, "angle = phase; radius = strength", transform=cpol.transAxes,
              ha="center", fontsize=6, color=INK2)
    # d: context
    d = fig.add_subplot(g[1, 3])
    d.imshow(v[:160].reshape(8, 20), aspect="auto", cmap="RdBu_r",
             vmin=-np.abs(v[:160]).max(), vmax=np.abs(v[:160]).max())
    d.set_xticks([])
    d.set_yticks([])
    d.set_title("d  Context: 160 values", loc="left", fontsize=7.5)
    d.text(0.5, -0.2, "red / blue = above / below zero", transform=d.transAxes, ha="center",
           fontsize=6, color=INK2)
    # e, f: does the representation follow the measured rhythm week to week?
    V, pids, m, keep = rep_and_markers(c)
    w_amp = V[:, 160 + 160:160 + 160 + 40]
    w_ph = V[:, 960 + 160:960 + 160 + 40]
    meas_amp = np.log(np.maximum(m["amplitude"], 1e-9))
    meas_ph = np.arctan2(m["phase_sin"], m["phase_cos"])
    frame = pd.DataFrame({"p": pids})
    rep_strength = np.log(w_amp).mean(1)
    meas_strength = np.nanmean(meas_amp - pd.DataFrame(meas_amp).groupby(pids).transform("mean").to_numpy(), 1)
    rep_strength = rep_strength - pd.Series(rep_strength).groupby(pids).transform("mean").to_numpy()
    wt = w_amp / w_amp.sum(1, keepdims=True)
    mean_dir = pd.DataFrame(np.exp(1j * w_ph)).groupby(pids).transform("mean").to_numpy()
    rep_shift = -np.angle((wt * np.exp(1j * w_ph) / np.exp(1j * np.angle(mean_dir))).sum(1)) * 12 / np.pi
    mdir = pd.DataFrame(np.exp(1j * np.nan_to_num(meas_ph))).groupby(pids).transform("mean").to_numpy()
    dev = np.exp(1j * meas_ph) / np.exp(1j * np.angle(mdir))
    meas_shift = np.angle(np.nanmean(dev, 1)) * 12 / np.pi
    for col, (xv, yv, xl, yl, title) in enumerate((
            (meas_strength, rep_strength, "Measured 24 h amplitude\n(log, vs own mean)",
             "DSSL 24 h strength\n(log, vs own mean)", "e  Strength follows the week"),
            (meas_shift, rep_shift, "Measured acrophase (h, vs own mean)",
             "DSSL timing (h, vs own mean)", "f  Timing follows the week"))):
        ax = fig.add_subplot(g[2, col * 2:col * 2 + 2])
        ok = np.isfinite(xv) & np.isfinite(yv)
        ax.scatter(xv[ok], yv[ok], s=3, color=INK2, alpha=0.35, linewidth=0)
        rho = pd.Series(xv[ok]).corr(pd.Series(yv[ok]), method="spearman")
        ax.text(0.03, 0.92, f"Spearman ρ = {rho:.2f}\n{ok.sum():,} weeks, {len(set(pids[ok]))} people",
                transform=ax.transAxes, fontsize=6.5, va="top", color=INK)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title, loc="left")
        if col == 1:
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
        else:
            ax.set_xlim(-2.5, 2.5)
        recessive_grid(ax, "both")
    fig.text(0.0, -0.05, "b-d: the example week's representation, from an encoder that never saw "
             "this participant. e-f: every labelled week, each scored by an encoder that never saw its "
             "participant (seed 1).", fontsize=6.5, color=INK2)
    fig.savefig(OUT / "fig2_anatomy.pdf")
    plt.close(fig)
    return dict(strength_rho=float(pd.Series(meas_strength).corr(pd.Series(rep_strength), method="spearman")),
                timing_rho=float(pd.Series(meas_shift).corr(pd.Series(rep_shift), method="spearman")))


# ---------------------------------------------------------------------------- Figure 3
def cos_curve(t, mesor, amp, phi):
    return mesor + amp * np.cos(2 * np.pi * t / 24 - phi)


def figure_rq1(c):
    pid, idx, w = example_participant(c)
    rec = pd.concat([pd.read_csv(RUN / f"seed_1/fold_{f}/rq1_recovery.csv", dtype={"participant": str})
                     for f in range(5)])
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), gridspec_kw=dict(width_ratios=[1.35, 1, 1],
                                                                           wspace=0.45))
    ch = "Steps"
    r = rec[(rec.participant == pid) & (rec.channel == ch)]
    t = np.linspace(0, 24, 241)
    profile = c.raw_X[idx][:, :, list(c.sensor_cols).index(ch)].reshape(len(idx), -1, c.bins_per_day).mean((0, 1))
    ax = axes[0]
    ax.plot(np.arange(c.bins_per_day) * 24 / c.bins_per_day, profile, color=NEUTRAL, linewidth=0.8,
            label="measured average day")
    curves = {}
    for method, label, colour in (("raw", "true rhythm", INK), ("dssl", "recovered from DSSL", BLUE),
                                  ("training_mean", "population average", ORANGE)):
        q = r[r.method == method].set_index("marker")
        col = "truth" if method == "raw" else "prediction"
        phi = np.arctan2(q.loc["phase_sin", col], q.loc["phase_cos", col])
        curves[method] = (q.loc["MESOR", col], q.loc["amplitude", col], phi)
        ax.plot(t, cos_curve(t, *curves[method]), color=colour, linewidth=1.6,
                linestyle="-" if method != "training_mean" else (0, (3, 2)), label=label)
    true, est = curves["raw"], curves["dssl"]
    amp_err = abs(est[1] - true[1])
    ph_err = abs(np.angle(np.exp(1j * (est[2] - true[2])))) * 12 / np.pi
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(CH_LABEL[ch])
    ax.set_title("a  One held-out participant", loc="left")
    ax.text(0.03, 0.62, f"DSSL errors:\namplitude {amp_err:.1f} steps\ntiming {ph_err * 60:.0f} min",
            transform=ax.transAxes, fontsize=6.3, va="top", ha="left", color=BLUE)
    ax.legend(frameon=False, fontsize=6.2, loc="upper right", handlelength=1.6)
    recessive_grid(ax, "y")
    # b, c: every held-out participant, truth vs recovered
    d = rec[(rec.channel == ch)]
    for ax, marker, label, title in ((axes[1], "amplitude", "24 h amplitude (steps)", "b  Amplitude"),
                                     (axes[2], "phase", "Acrophase (h)", "c  Timing")):
        for method, colour in (("untrained", ORANGE), ("dssl", BLUE)):
            q = d[d.method == method]
            if marker == "amplitude":
                x = q[q.marker == "amplitude"].set_index("participant")
                xv, yv = x.truth, x.prediction
            else:
                cs = q[q.marker == "phase_cos"].set_index("participant")
                sn = q[q.marker == "phase_sin"].set_index("participant")
                xv = (np.arctan2(sn.truth, cs.truth) * 12 / np.pi) % 24
                yv = (np.arctan2(sn.prediction, cs.prediction) * 12 / np.pi) % 24
            ax.scatter(xv, yv, s=7, color=colour, alpha=0.8, linewidth=0,
                       label="DSSL" if method == "dssl" else "untrained encoder")
        lim = ax.get_xlim() if marker == "amplitude" else (10, 22)
        ax.plot(lim, lim, color=INK2, linewidth=0.7, linestyle=(0, (3, 2)))
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f"True {label.lower()}")
        ax.set_ylabel(f"Recovered")
        ax.set_title(title, loc="left")
        recessive_grid(ax, "both")
    axes[1].legend(frameon=False, fontsize=6.2, loc="upper left", handletextpad=0.2)
    fig.text(0.0, -0.12, "Steps channel; seed 1; each participant's markers are predicted by a ridge "
             "fitted on other participants, from an encoder that never saw them.", fontsize=6.5, color=INK2)
    fig.savefig(OUT / "fig3_rq1.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------- Figure 4
def figure_rq2(c):
    pid, idx, w = example_participant(c)
    ch = list(c.sensor_cols).index("Steps")
    bpd = c.bins_per_day
    j = list(idx).index(w)
    # average day, smoothed with a circular 1-hour moving average for display only
    day = lambda X: np.convolve(np.tile(X[:, ch].reshape(-1, bpd).mean(0), 3), np.ones(4) / 4,
                                mode="same")[bpd:2 * bpd]
    t = np.arange(bpd) * 24 / bpd
    fig = plt.figure(figsize=(7.0, 4.3))
    g = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.42)
    a = fig.add_subplot(g[0, 0])
    for k in range(1, 5):
        a.plot(t, day(c.raw_X[idx[j - k]]), color=NEUTRAL, linewidth=0.7,
               label="4 preceding weeks" if k == 1 else None)
    a.plot(t, day(c.raw_X[w]), color=INK, linewidth=1.3, label="current week")
    a.set_title("a  Personal baseline", loc="left")
    b = fig.add_subplot(g[0, 1], sharey=a)
    X = c.raw_X[w][None].astype(float)
    b.plot(t, day(X[0]), color=INK, linewidth=1.3, label="current week")
    b.plot(t, day(phase_shift(X, 2.0, X.shape[-1], c.bin_minutes)[0]), color=BLUE, linewidth=1.3,
           label="shifted 2 h later")
    b.set_title("b  Timing perturbation", loc="left")
    b2 = fig.add_subplot(g[0, 2], sharey=a)
    b2.plot(t, day(X[0]), color=INK, linewidth=1.3, label="current week")
    b2.plot(t, day(amplitude_scale(X, 1.3, X.shape[-1], bpd)[0]), color=AQUA, linewidth=1.3,
            label="24 h strength × 1.3")
    b2.plot(t, day(amplitude_scale(X, 0.7, X.shape[-1], bpd)[0]), color=AQUA, linewidth=1.0,
            linestyle=(0, (3, 2)), label="× 0.7")
    b2.set_title("c  Strength perturbation", loc="left")
    for ax in (a, b, b2):
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_xlabel("Hour of day")
        ax.legend(frameon=False, fontsize=6, loc="upper left", handlelength=1.4)
        recessive_grid(ax, "y")
    a.set_ylabel("Steps / 15 min (avg. day)")
    levels = pd.read_csv(RUN / "rq2_by_level.csv")
    floor = always_increase_floor()
    methods = [("dssl", "DSSL", BLUE, "o"), ("untrained", "Untrained encoder", ORANGE, "s"),
               ("random_projection", "Random projection", NEUTRAL, "^")]
    for col, (pert, xlabel, title, ylabel, ref, ref_label) in enumerate((
            ("phase", "Shift (hours)", "d  Timing change detected?", "Concordance", 0.5, "chance"),
            ("amplitude", r"Strength change $\alpha$", "e  Strength change detected?",
             "Sign agreement", floor, "always-'away' rule"))):
        ax = fig.add_subplot(g[1, col])
        block = levels[levels.perturbation == pert]
        for key, label, colour, marker in methods:
            s = block[block.method == key].groupby("level").concordance
            ax.errorbar(s.mean().index, s.mean().values, yerr=s.std(ddof=0).values, color=colour,
                        marker=marker, markersize=4, linewidth=1.5, elinewidth=0.9, label=label)
        ax.axhline(ref, color=INK2, linewidth=0.8, linestyle=(0, (3, 2)))
        ax.text(ax.get_xlim()[1], ref - 0.008, ref_label + " ", fontsize=6.3, color=INK2,
                ha="right", va="top")
        ax.set_ylim(0.45, 0.9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        recessive_grid(ax, "y")
        if col == 0:
            handles, labels = ax.get_legend_handles_labels()
    txt = fig.add_subplot(g[1, 2])
    txt.axis("off")
    txt.legend(handles, labels, frameon=False, loc="upper left", fontsize=7)
    txt.text(0, 0.45, "Each week is compared with the same\nperson's four preceding weeks. A method\n"
             "succeeds when its representation moves\nfurther from that baseline whenever the\n"
             "week's true rhythm moves further.\nPoints: mean of 3 seeds; bars: seed SD.",
             fontsize=6.5, color=INK2, va="top")
    fig.savefig(OUT / "fig4_rq2.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------- Figure 8
RQ3_ROWS = [("raw", "Raw window"), ("untrained", "Untrained encoder"),
            ("yan_cosinor", "Cosinor (Yan et al.)"), ("handcrafted", "Handcrafted rhythm"),
            ("random_projection", "Random projection"), ("cost_reference_adapter", "CoST"),
            ("pca", "PCA")]


def figure_rq3():
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.2), sharey="row",
                             gridspec_kw=dict(hspace=0.55, wspace=0.16))
    for row, dataset, name in ((0, "hrd", "HRD (113 participants)"), (1, "globem", "GLOBEM 2018 (142)")):
        run = ROOT / "results" / dataset / "narval_v2/tcn_none"
        oof = pd.read_csv(run / "oof_predictions.csv", dtype={"participant": str})
        oof = oof[oof.probe == "logistic"]
        paired = pd.read_csv(run / "rq3_paired_intervals.csv")
        paired = paired[paired.probe == "logistic"].set_index("control")
        rows = [("dssl", "DSSL (proposed)")] + RQ3_ROWS
        ys = np.arange(len(rows))[::-1]
        for y, (key, label) in zip(ys, rows):
            point, lo, hi, n = auroc_interval(oof[oof.method == key])
            colour = BLUE if key == "dssl" else (ORANGE if key in ("raw", "untrained") else NEUTRAL)
            axes[row, 0].plot([lo, hi], [y, y], color=colour, linewidth=1.5)
            axes[row, 0].plot(point, y, "o", color=colour, markersize=4.5)
            if key != "dssl":
                r = paired.loc[key]
                solid = r.low > 0 or r.high < 0
                axes[row, 1].plot([r.low, r.high], [y, y], color=colour, linewidth=1.5)
                axes[row, 1].plot(r.difference, y, "o", color=colour, markersize=4.5,
                                  markerfacecolor=colour if solid else "white")
        axes[row, 0].axvline(0.5, color=INK2, linewidth=0.7, linestyle=(0, (3, 2)))
        axes[row, 1].axvline(0, color=INK2, linewidth=0.7)
        axes[row, 0].set_yticks(ys)
        axes[row, 0].set_yticklabels([l for _, l in rows])
        axes[row, 0].set_xlim(0.3, 0.9)
        axes[row, 1].set_xlim(-0.2, 0.25)
        axes[row, 0].set_title(f"{'ab'[0] if row == 0 else 'c'}  {name}: AUROC and 95% CI", loc="left")
        axes[row, 1].set_title(f"{'b' if row == 0 else 'd'}  DSSL minus method (paired 95% CI)", loc="left")
        for ax in axes[row]:
            recessive_grid(ax)
    axes[0, 0].set_title("a  HRD (113 participants): AUROC and 95% CI", loc="left")
    axes[1, 0].set_xlabel("Participant AUROC")
    axes[1, 1].set_xlabel("Difference in AUROC (positive favours DSSL)")
    fig.text(0.0, -0.04, "Orange: the two baselines the pre-registered criterion requires DSSL to beat. "
             "Filled difference markers: the paired interval excludes zero.", fontsize=6.5, color=INK2)
    fig.savefig(OUT / "fig8_rq3.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    c = cohort()
    figure_overview(c)
    print("anatomy", figure_anatomy(c))
    figure_rq1(c)
    figure_rq2(c)
    figure_rq3()
    if (GROUPS / "participant_scores.csv").exists():
        from scripts.group_figures import figure_groups, figure_transitions, figure_markers
        figure_groups()
        figure_transitions()
        figure_markers()
    print("wrote", sorted(p.name for p in OUT.glob("fig*.pdf")))
