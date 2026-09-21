"""Figures 5-7 of the manuscript, from results/hrd/narval_v2/tcn_none/group_analysis.

    python scripts/manuscript_figures.py     # calls these after figures 1-4 and 8
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.paper_figures import BLUE, ORANGE, AQUA, YELLOW, NEUTRAL, INK, INK2, recessive_grid

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "SSL_Rhythmicity" / "figures"
G = ROOT / "results/hrd/narval_v2/tcn_none/group_analysis"
G_COL = {"stable_non_depressed": BLUE, "stable_depressed": ORANGE,
         "became_depressed": AQUA, "recovered": YELLOW}
G_LAB = {"stable_non_depressed": "Stable\nnon-depressed", "stable_depressed": "Stable\ndepressed",
         "became_depressed": "Became\ndepressed", "recovered": "Recovered"}
G_SHORT = {"stable_non_depressed": "SN", "stable_depressed": "SD", "became_depressed": "N→D",
           "recovered": "D→N"}
B_SHORT = {"Context (trend)": "Context", "24 h strength": "24 h\nstrength", "24 h timing": "24 h\ntiming",
           "Day-shape strength (12, 8, 6 h)": "Day-shape\nstrength", "Day-shape timing (12, 8, 6 h)": "Day-shape\ntiming",
           "Weekly strength": "Weekly\nstrength", "Weekly timing": "Weekly\ntiming"}
STABLE = "Stable depressed vs stable non-depressed"


def strip(ax, frame, value, rng):
    for k, grp in enumerate(G_LAB):
        x = frame.loc[frame.group == grp, value].to_numpy()
        jitter = rng.uniform(-0.18, 0.18, len(x))
        ax.scatter(k + jitter, x, s=9, color=G_COL[grp], alpha=0.75, linewidth=0)
        boot = [rng.choice(x, len(x)).mean() for _ in range(2000)]
        ax.plot([k - 0.3, k + 0.3], [x.mean()] * 2, color=INK, linewidth=1.4)
        ax.plot([k, k], np.percentile(boot, [2.5, 97.5]), color=INK, linewidth=2.2)
    ax.set_xticks(range(len(G_LAB)))
    ax.set_xticklabels([f"{G_SHORT[g]}\n{int((frame.group == g).sum())}" for g in G_LAB], fontsize=6.5)


def figure_groups():
    rng = np.random.default_rng(5)
    r2 = pd.read_csv(G / "block_r2.csv")
    person = pd.read_csv(G / "participant_scores.csv", dtype={"participant": str})
    var = pd.read_csv(G / "variability.csv")
    fig = plt.figure(figsize=(7.0, 5.0))
    g = fig.add_gridspec(2, 3, height_ratios=[1, 1.1], hspace=0.7, wspace=0.75)
    # a: share of between-person variance explained by stable-group membership, per block
    a = fig.add_subplot(g[0, :2])
    s = r2[r2.comparison == STABLE]
    blocks = list(dict.fromkeys(s.block))
    x = np.arange(len(blocks))
    for off, space, colour, label in ((-0.18, "dssl", BLUE, "DSSL (trained)"),
                                      (0.18, "untrained", ORANGE, "Untrained encoder")):
        q = s[s.space == space].set_index("block").loc[blocks]
        a.bar(x + off, q.r2 * 100, width=0.34, color=colour, label=label)
        for xi, (_, row) in zip(x + off, q.iterrows()):
            if row.p_holm < 0.05:
                a.text(xi, row.r2 * 100 + 0.3, "*", ha="center", fontsize=8, color=INK)
    null95 = s.groupby("block").null_95.max().loc[blocks] * 100
    a.step(np.r_[x - 0.5, x[-1] + 0.5], np.r_[null95.values, null95.values[-1]], where="post",
           color=INK2, linewidth=0.8, linestyle=(0, (3, 2)), label="95th pct. under random labels")
    a.set_xticks(x)
    a.set_xticklabels([B_SHORT[b] for b in blocks], fontsize=6.3)
    a.set_ylabel("Variance explained by group (%)")
    a.set_title("a  Where do the stable groups differ?", loc="left")
    a.legend(frameon=False, fontsize=6.3, loc="upper right")
    recessive_grid(a, "y")
    # b: variability
    b = fig.add_subplot(g[0, 2])
    v = var[var.space == "dssl"].set_index("block").loc[blocks]
    y = np.arange(len(blocks))[::-1]
    for yi, (_, row) in zip(y, v.iterrows()):
        solid = row.p_holm < 0.05
        b.plot([row.low, row.high], [yi, yi], color=BLUE, linewidth=1.4)
        b.plot(row.effect, yi, "o", color=BLUE, markersize=4.5, markerfacecolor=BLUE if solid else "white")
    b.axvline(0, color=INK2, linewidth=0.7)
    b.set_yticks(y)
    b.set_yticklabels([B_SHORT[bl].replace("\n", " ") for bl in blocks], fontsize=6.3)
    b.set_xlabel("Hedges' g, SD − SN")
    b.set_title("b  Week-to-week variability", loc="left")
    recessive_grid(b)
    # c, d: participant positions on the depression axis
    for col, (space, title) in enumerate((("dssl", "c  DSSL representation"),
                                          ("measured", "d  Measured rhythm markers"))):
        ax = fig.add_subplot(g[1, col])
        f = person[person.space == space]
        ax.set_ylim(-2.2, 3.2)
        strip(ax, f, "score", rng)
        ax.axhline(0, color=BLUE, linewidth=0.6, linestyle=(0, (3, 2)))
        ax.axhline(1, color=ORANGE, linewidth=0.6, linestyle=(0, (3, 2)))
        ax.set_ylabel("Position on depression axis\n(0 = SN centre, 1 = SD centre)" if col == 0 else "")
        ax.set_title(title, loc="left")
        recessive_grid(ax, "y")
    eff = pd.read_csv(G / "axis_effects.csv")
    e = fig.add_subplot(g[1, 2])
    rows = [("stable_depressed", "stable_non_depressed", "SD vs SN"),
            ("became_depressed", "stable_non_depressed", "N→D vs SN"),
            ("recovered", "stable_depressed", "D→N vs SD")]
    yy = np.arange(len(rows))[::-1]
    for off, space, colour in ((0.15, "dssl", BLUE), (0, "untrained", ORANGE), (-0.15, "measured", NEUTRAL)):
        q = eff[eff.space == space].set_index(["group_a", "group_b"])
        for yi, (ga, gb, _) in zip(yy, rows):
            r = q.loc[(ga, gb)]
            e.plot([r.low, r.high], [yi + off] * 2, color=colour, linewidth=1.3)
            e.plot(r.effect, yi + off, "o", color=colour, markersize=4,
                   markerfacecolor=colour if r.p_holm < 0.05 else "white")
    e.axvline(0, color=INK2, linewidth=0.7)
    e.set_yticks(yy)
    e.set_yticklabels([r[2] for r in rows], fontsize=6.3)
    e.set_xlabel("Hedges' g with 95% CI")
    e.set_title("e  Group separation", loc="left")
    handles = [plt.Line2D([], [], color=c, marker="o", linewidth=1.3, markersize=4)
               for c in (BLUE, ORANGE, NEUTRAL)]
    e.legend(handles, ["DSSL", "Untrained", "Measured"], frameon=False, fontsize=6, loc="upper left",
             bbox_to_anchor=(0.0, -0.22), ncol=3, handlelength=1.2, columnspacing=0.8)
    recessive_grid(e)
    fig.text(0.0, -0.06, "SN stable non-depressed, SD stable depressed, N→D became depressed, D→N recovered (n under "
             "each group). a: * Holm-corrected permutation p < 0.05, averaged over 15 encoders.\nb, e: filled "
             "markers Holm p < 0.05. c-d: one dot per participant, scored only by encoders that never saw them; "
             "bars: mean and 95% CI.", fontsize=6.3, color=INK2)
    fig.savefig(OUT / "fig5_groups.pdf")
    plt.close(fig)


def figure_transitions():
    rng = np.random.default_rng(6)
    W = pd.read_csv(G / "window_scores.csv", dtype={"participant": str})
    change = pd.read_csv(G / "trajectory_change.csv", dtype={"participant": str})
    prob = pd.read_csv(G / "classifier_by_group.csv")
    fig = plt.figure(figsize=(7.0, 4.6))
    g = fig.add_gridspec(2, 3, hspace=0.7, wspace=0.45)
    for col, (space, title) in enumerate((("dssl", "a  DSSL representation"),
                                          ("measured", "b  Measured rhythm markers"))):
        ax = fig.add_subplot(g[0, col])
        w = W[W.space == space].copy()
        w["bin"] = (w.week // 4) * 4
        w = w[w.bin <= 28]
        for grp in G_COL:
            per = w[w.group == grp].groupby(["participant", "bin"]).score.mean().unstack()
            m = per.mean(0)
            boots = np.array([per.iloc[rng.integers(0, len(per), len(per))].mean(0).to_numpy()
                              for _ in range(1000)])
            lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
            ax.fill_between(m.index + 2, lo, hi, color=G_COL[grp], alpha=0.15, linewidth=0)
            ax.plot(m.index + 2, m.values, color=G_COL[grp], linewidth=1.6, marker="o", markersize=3,
                    label=G_LAB[grp].replace("\n", " "))
        ax.set_xlabel("Weeks since first window")
        ax.set_ylabel("Position on depression axis" if col == 0 else "")
        ax.set_title(title, loc="left")
        recessive_grid(ax, "y")
        if col == 0:
            handles, labels = ax.get_legend_handles_labels()
    leg = fig.add_subplot(g[0, 2])
    leg.axis("off")
    leg.legend(handles, labels, frameon=False, fontsize=7, loc="upper left")
    leg.text(0, 0.35, "Each week is placed on the axis that\nseparates the stable groups (0 and 1),\n"
             "computed without the participant.\nLines: 4-week means; bands: 95% CI\nover participants.",
             fontsize=6.5, color=INK2, va="top")
    # c: change from the first to the last 4 weeks
    c = fig.add_subplot(g[1, :2])
    for off, space, colour in ((-0.15, "dssl", BLUE), (0.15, "measured", NEUTRAL)):
        q = change[change.space == space]
        for k, grp in enumerate(G_COL):
            x = q.loc[q.group == grp, "change"].to_numpy()
            boot = [rng.choice(x, len(x)).mean() for _ in range(2000)]
            c.plot([k + off] * 2, np.percentile(boot, [2.5, 97.5]), color=colour, linewidth=1.6)
            c.plot(k + off, x.mean(), "o", color=colour, markersize=4.5)
            c.scatter(k + off + rng.uniform(-0.05, 0.05, len(x)), x, s=4, color=colour, alpha=0.35,
                      linewidth=0)
    c.axhline(0, color=INK2, linewidth=0.7)
    c.set_xticks(range(4))
    counts = change[change.space == "dssl"].group.value_counts()
    c.set_xticklabels([f"{G_SHORT[g]} (n={counts.get(g, 0)})" for g in G_COL], fontsize=6.8)
    c.set_ylabel("Last 4 weeks − first 4 weeks")
    c.set_ylim(-3, 3)
    c.set_title("c  Change over the study (blue: DSSL, grey: measured)", loc="left")
    recessive_grid(c, "y")
    # d: classifier probability by group
    d = fig.add_subplot(g[1, 2])
    for off, method, colour in ((-0.15, "dssl", BLUE), (0.15, "raw", ORANGE)):
        q = prob[prob.method == method].set_index("group")
        for k, grp in enumerate(G_COL):
            r = q.loc[grp]
            d.plot([k + off] * 2, [r.low, r.high], color=colour, linewidth=1.6)
            d.plot(k + off, r.mean_probability, "o", color=colour, markersize=4.5)
    d.set_xticks(range(4))
    d.set_xticklabels([G_SHORT[g] for g in G_COL], fontsize=6.8)
    d.set_ylabel("Predicted probability (RQ3)")
    d.set_title("d  RQ3 classifier output", loc="left")
    d.legend([plt.Line2D([], [], color=BLUE, marker="o"), plt.Line2D([], [], color=ORANGE, marker="o")],
             ["DSSL", "Raw window"], frameon=False, fontsize=6.3, loc="upper left",
             bbox_to_anchor=(0.0, -0.18), ncol=2)
    recessive_grid(d, "y")
    fig.savefig(OUT / "fig6_transitions.pdf")
    plt.close(fig)


def figure_markers():
    m = pd.read_csv(G / "markers_by_group.csv")
    m = m[m.comparison == STABLE]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.5))
    names = {"MESOR": "Level", "amplitude": "24 h amplitude", "IS": "Day-to-day regularity (IS)",
             "IV": "Fragmentation (IV)", "RA": "Relative amplitude"}
    lin = m[m.marker != "acrophase"]
    keys = [(mk, ch) for mk in names for ch in ["Heart rate", "Steps", "Sleep", "Screen"]
            if ((lin.marker == mk) & (lin.channel == ch)).any()]
    y = np.arange(len(keys))[::-1]
    for off, source, colour, label in ((0.17, "measured", INK2, "Measured"),
                                       (-0.17, "dssl_decoded", BLUE, "Read out from DSSL")):
        q = lin[lin.source == source].set_index(["marker", "channel"])
        for yi, key in zip(y, keys):
            r = q.loc[key]
            axes[0].plot([r.low, r.high], [yi + off] * 2, color=colour, linewidth=1.2)
            axes[0].plot(r.effect, yi + off, "o", color=colour, markersize=3.8,
                         markerfacecolor=colour if r.p_holm < 0.05 else "white",
                         label=label if key == keys[0] else None)
    axes[0].axvline(0, color=INK2, linewidth=0.7)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{names[mk]} · {ch}" for mk, ch in keys], fontsize=6.3)
    axes[0].set_xlabel("Hedges' g, stable depressed − stable non-depressed")
    axes[0].set_title("a  Strength, level and regularity", loc="left")
    axes[0].legend(frameon=False, fontsize=6.3, loc="lower right")
    recessive_grid(axes[0])
    ac = m[m.marker == "acrophase"]
    chans = ["Heart rate", "Steps", "Sleep", "Screen"]
    y = np.arange(len(chans))[::-1]
    for off, source, colour in ((0.12, "measured", INK2), (-0.12, "dssl_decoded", BLUE)):
        q = ac[ac.source == source].set_index("channel")
        for yi, ch in zip(y, chans):
            r = q.loc[ch]
            axes[1].plot([r.low * 60, r.high * 60], [yi + off] * 2, color=colour, linewidth=1.2)
            axes[1].plot(r.effect * 60, yi + off, "o", color=colour, markersize=3.8,
                         markerfacecolor=colour if r.p_holm < 0.05 else "white")
    axes[1].axvline(0, color=INK2, linewidth=0.7)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(chans, fontsize=6.8)
    axes[1].set_xlabel("Later peak in depressed group (minutes)")
    axes[1].set_title("b  Timing of the daily peak", loc="left")
    recessive_grid(axes[1])
    n = lin[lin.source == "measured"].iloc[0]
    fig.text(0.0, -0.06, f"{int(n.n_a)} stable depressed vs {int(n.n_b)} stable non-depressed participants. "
             "95% stratified bootstrap intervals;\nfilled: Holm-corrected permutation p < 0.05 within each "
             "source. 'Read out from DSSL': RQ1's held-out predictions.", fontsize=6.3, color=INK2)
    fig.savefig(OUT / "fig7_markers.pdf")
    plt.close(fig)
