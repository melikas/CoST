"""What the trend (V^T) and seasonal (V^S) branches hold, and how they differ by depression.

    python scripts/representation_figures.py --run narval_v3 --globem-run loyo_v3 [--out DIR]

Reads the dynamics.npz files that `run_experiment.py --export-dynamics` writes into
results/<dataset>/<run>/tcn_none_selected/seed_<s>/fold_<f>/, the window caches and, for HRD,
datasets/cache/hrd_status_v1.csv (scripts/hrd_depression_status.py). Every participant is
represented only by encoders that never saw them, and each fold's scale comes from its training
participants. A participant's value is the mean over the seeds available. Groups:
  endpoint  elevated depressive symptoms at study end (the RQ3 label), both datasets;
  course    stable non-depressed / stable depressed / became depressed / recovered, HRD only.
Figures (each answers one question):
  rep_rhythm_shape   HRD   the 24 h shape of behaviour, of V^S and of V^T, by group
  rep_timescales     both  autocorrelation of V^S and V^T over lag: rhythmic versus slow
  rep_spectrum       both  where V^S puts its power across periods, by group
  rep_rq2_movement   HRD   how one person's representation moves under a timing or strength change
  rep_actogram       HRD   every participant's day, measured and represented, side by side
  rep_group_features both  group differences per branch, with raw-behaviour analogues
  rep_trajectories   HRD   V^S 24 h power and V^T level over the study, by symptom course
Statistics: Hedges' g (circular difference in minutes for peak times), 95% stratified participant
bootstrap intervals, two-sided label-permutation p, Holm-corrected within each dataset.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datautils import load_npz
from tasks.rhythm import individual_markers
from scripts.paper_figures import INK, INK2, GRID, BLUE, ORANGE, AQUA, YELLOW, recessive_grid

CACHE = {"hrd": "datasets/cache/hrd_rescue_v1.npz", "globem": "datasets/cache/globem_rescue_v1.npz"}
NAME = {"hrd": "HRD", "globem": "GLOBEM, leave one year out"}
SHORT = {"hrd": "HRD", "globem": "GLOBEM (year held out)"}
ENDPOINT = {0: ("Not elevated", BLUE), 1: ("Elevated", ORANGE)}
COURSE = {"stable_non_depressed": ("Stable non-depressed", BLUE, "o"),
          "stable_depressed": ("Stable depressed", ORANGE, "s"),
          "became_depressed": ("Became depressed", AQUA, "^"),
          "recovered": ("Recovered", YELLOW, "D")}
NEUTRAL_MID = "#f2f1ec"
DIVERGING = LinearSegmentedColormap.from_list("blue_orange", [BLUE, NEUTRAL_MID, ORANGE])
RNG = np.random.default_rng(20260922)
N_BOOT, N_PERM = 2000, 2000


# ----------------------------------------------------------------------------- loading
def load(dataset, run):
    """Participant-level arrays averaged over seeds, window-level tables, labels and raw data."""
    root = ROOT / "results" / dataset / run / "tcn_none_selected"
    files = sorted(root.glob("seed_*/fold_*/dynamics.npz"))
    if not files:
        raise FileNotFoundError(f"no dynamics.npz under {root}; run --export-dynamics first")
    per, windows, rq2 = {}, [], []
    for f in files:
        z = np.load(f, allow_pickle=False)
        seed = int(f.parent.parent.name.split("_")[1])
        for i, p in enumerate(z["participants"]):
            per.setdefault(p, []).append({k: z[k][i] for k in z.files
                                          if k.startswith(("profile_", "spectrum_", "acf_"))})
        windows.append(pd.DataFrame({k[len("window_"):]: z[k] for k in z.files if k.startswith("window_")})
                       .rename(columns={"id": "window_id"}).assign(seed=seed))
        if seed == 1 and len(z["rq2_participant"]):
            rq2.append({k: z[k] for k in z.files if k.startswith("rq2_")})
        meta = dict(lags_hours=z["lags_hours"], period_hours=z["period_hours"],
                    bins_per_day=int(z["bins_per_day"]), days=int(z["days"]))
    people = sorted(per)
    arrays = {k: np.stack([np.mean([r[k] for r in per[p]], axis=0) for p in people])
              for k in per[people[0]][0]}
    c = load_npz(ROOT / CACHE[dataset])
    ids, y = c.participants()
    label = pd.Series(y, index=ids.astype(str))
    groups = pd.DataFrame(index=pd.Index(people, name="participant"))
    groups["endpoint"] = label.reindex(people).to_numpy()
    if dataset == "hrd":
        status = pd.read_csv(ROOT / "datasets/cache/hrd_status_v1.csv", dtype={"participant": str})
        groups["course"] = status.set_index("participant").group.reindex(people).to_numpy()
    w = pd.concat(windows, ignore_index=True)
    w = w.groupby(["participant", "window_id"], as_index=False).agg(
        {k: "mean" for k in w.columns if k not in ("participant", "window_id", "seed", "peak_hour")}
        | {"peak_hour": lambda h: (np.angle(np.exp(1j * h * np.pi / 12).mean()) * 12 / np.pi) % 24})
    return dict(people=np.array(people), arrays=arrays, groups=groups, windows=w, rq2=rq2,
                cohort=c, meta=meta, n_seeds=len({int(f.parent.parent.name.split('_')[1]) for f in files}))


def raw_participant(c, people):
    """Measured per-participant daily step profile and rhythm markers (steps channel)."""
    step = next(i for i, n in enumerate(c.sensor_cols) if "step" in n.lower())
    b = c.bins_per_day
    pids = c.pids.astype(str)
    m = individual_markers(c.raw_X, c.observed, b, list(c.sensor_cols))["values"]
    rows, prof = [], []
    for p in people:
        i = pids == p
        prof.append(c.raw_X[i][:, :, step].reshape(i.sum(), -1, b).mean(axis=(0, 1)))
        phase = np.angle(np.nanmean(np.exp(1j * np.arctan2(m["phase_sin"][i, step], m["phase_cos"][i, step]))))
        rows.append(dict(participant=p, amplitude=np.nanmean(m["amplitude"][i, step]),
                         acrophase_hour=(phase * 12 / np.pi) % 24, IS=np.nanmean(m["IS"][i, step]),
                         MESOR=np.nanmean(m["MESOR"][i, step])))
    return pd.DataFrame(rows).set_index("participant"), np.array(prof)


# ----------------------------------------------------------------------------- statistics
def mean_band(x):
    """Mean over rows and a 95% participant-bootstrap band, column by column."""
    x = np.asarray(x, float)
    idx = RNG.integers(0, len(x), size=(N_BOOT, len(x)))
    boot = np.nanmean(x[idx], axis=1)
    return np.nanmean(x, 0), np.nanpercentile(boot, 2.5, 0), np.nanpercentile(boot, 97.5, 0)


def hedges(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return (1 - 3 / (4 * (na + nb) - 9)) * (np.mean(a) - np.mean(b)) / sp


def circ_minutes(a, b):
    return np.angle(np.exp(1j * a * np.pi / 12).mean() / np.exp(1j * b * np.pi / 12).mean()) * 12 / np.pi * 60


def compare(a, b, stat):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    est = stat(a, b)
    boot = np.array([stat(RNG.choice(a, len(a)), RNG.choice(b, len(b))) for _ in range(N_BOOT)])
    if stat is circ_minutes:
        boot = est + (boot - est + 720) % 1440 - 720
    pool = np.r_[a, b]
    null = np.array([abs(stat(*np.split(RNG.permutation(pool), [len(a)]))) for _ in range(N_PERM)])
    return dict(n_a=len(a), n_b=len(b), effect=est, low=np.percentile(boot, 2.5),
                high=np.percentile(boot, 97.5), p=(1 + (null >= abs(est)).sum()) / (1 + N_PERM))


def holm(p):
    p = np.asarray(p, float)
    out, run = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - rank) * p[i])
        out[i] = min(1.0, run)
    return out


def split(series, groups, col="endpoint"):
    g = groups[col].reindex(series.index)
    return {k: series[g == k].dropna().to_numpy() for k in pd.unique(g.dropna())}


# ----------------------------------------------------------------------------- figure helpers
def band_plot(ax, x, rows, colour, label, **kw):
    m, lo, hi = mean_band(rows)
    ax.fill_between(x, lo, hi, color=colour, alpha=0.18, linewidth=0)
    ax.plot(x, m, color=colour, linewidth=1.8, label=label, **kw)


def hour_axis(ax, label=True):
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xticklabels(["00", "06", "12", "18", "24"])
    if label:
        ax.set_xlabel("Hour of day")


def by_endpoint(groups):
    return [(k, *ENDPOINT[k], groups.index[groups.endpoint == k]) for k in (0, 1)]


def legend_counts(ax, groups, **kw):
    handles = [plt.Line2D([], [], color=ENDPOINT[k][1], linewidth=1.8,
                          label=f"{ENDPOINT[k][0]} (n={int((groups.endpoint == k).sum())})") for k in (0, 1)]
    ax.legend(handles=handles, frameon=False, fontsize=7, **kw)


# ----------------------------------------------------------------------------- figures
def rhythm_shape(d, out):
    A, g, b = d["arrays"], d["groups"], d["meta"]["bins_per_day"]
    people = pd.Index(d["people"])
    _, raw_prof = raw_participant(d["cohort"], d["people"])
    hours = (np.arange(b) + 0.5) * 24 / b
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.9), gridspec_kw=dict(hspace=0.55, wspace=0.3))
    panels = [(axes[0, 0], raw_prof, "Steps per 15 min", "a  Measured behaviour (steps)"),
              (axes[0, 1], A["profile_seasonal"][:, :, 0], "V$^S$ component 1 (SD units)",
               "b  Rhythm branch V$^S$"),
              (axes[1, 0], A["profile_trend"][:, :, 0], "V$^T$ component 1 (SD units)", "c  Trend branch V$^T$")]
    for ax, rows, ylabel, title in panels:
        for k, name, colour, members in by_endpoint(g):
            band_plot(ax, hours, rows[people.get_indexer(members)], colour, name)
        hour_axis(ax)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        recessive_grid(ax, "y")
    legend_counts(axes[0, 0], g, loc="upper left")
    ax = axes[1, 1]
    s = A["profile_seasonal"][:, :, 0]
    dep, non = s[people.get_indexer(g.index[g.endpoint == 1])], s[people.get_indexer(g.index[g.endpoint == 0])]
    diffs = np.array([RNG.choice(dep, len(dep)).mean(0) - RNG.choice(non, len(non)).mean(0) for _ in range(N_BOOT)])
    lo, hi = np.percentile(diffs, [2.5, 97.5], axis=0)
    ax.fill_between(hours, lo, hi, color=INK2, alpha=0.2, linewidth=0)
    ax.plot(hours, dep.mean(0) - non.mean(0), color=INK, linewidth=1.6)
    ax.axhline(0, color=INK2, linewidth=0.7)
    hour_axis(ax)
    ax.set_ylabel("Elevated minus not elevated (SD units)")
    ax.set_title("d  Where in the day V$^S$ differs", loc="left")
    recessive_grid(ax, "y")
    ev = A.get("explained_seasonal")
    fig.text(0.0, -0.04, "Lines: group mean of participants' average day; bands: 95% participant-bootstrap interval. "
             "Components are fitted on each fold's training participants;\neach participant is scored by encoders "
             f"that never saw them ({d['n_seeds']} seed(s)). Where the band in (d) excludes zero the groups differ at "
             "that hour, without correction across hours.", fontsize=6.3, color=INK2)
    fig.savefig(out / "rep_rhythm_shape.pdf")
    plt.close(fig)


def timescales(ds, out):
    fig, axes = plt.subplots(len(ds), 2, figsize=(7.0, 2.5 * len(ds)), squeeze=False,
                             gridspec_kw=dict(hspace=0.65, wspace=0.25))
    for row, (dataset, d) in enumerate(ds.items()):
        lags = d["meta"]["lags_hours"] / 24
        keep = lags <= 5 + 1e-9
        people = pd.Index(d["people"])
        for col, (branch, title) in enumerate((("seasonal", "Rhythm branch V$^S$"), ("trend", "Trend branch V$^T$"))):
            ax = axes[row, col]
            for k, name, colour, members in by_endpoint(d["groups"]):
                band_plot(ax, lags[keep], d["arrays"][f"acf_{branch}"][people.get_indexer(members)][:, keep], colour, name)
            for day in range(1, 6):
                ax.axvline(day, color=GRID, linewidth=0.8, zorder=0)
            ax.axhline(0, color=INK2, linewidth=0.6)
            ax.set_xlim(0, 5)
            ax.set_ylim(-0.9, 1.05)
            if col == 0 and row == 0:
                ax.annotate("one day apart", xy=(1, 0.5), xytext=(1.5, 0.72), fontsize=6.5, color=INK2,
                            arrowprops=dict(arrowstyle="->", color=INK2, linewidth=0.7))
            ax.set_xlabel("Lag (days)")
            ax.set_ylabel("Autocorrelation")
            ax.set_title(f"{'abcd'[2 * row + col]}  {SHORT[dataset]}: {title}", loc="left")
        legend_counts(axes[row, 1], d["groups"], loc="upper right")
    fig.text(0.0, -0.03, "A rhythmic sequence returns to high correlation at every whole day; a slow one decays "
             "smoothly. Pooled over channels; mean and 95% participant-bootstrap band.", fontsize=6.3, color=INK2)
    fig.savefig(out / "rep_timescales.pdf")
    plt.close(fig)


def spectrum(ds, out, bands):
    fig, axes = plt.subplots(len(ds), 1, figsize=(7.0, 2.4 * len(ds)), squeeze=False,
                             gridspec_kw=dict(hspace=0.7))
    for row, (dataset, d) in enumerate(ds.items()):
        ax = axes[row, 0]
        period = d["meta"]["period_hours"]
        spec = d["arrays"]["spectrum_seasonal"][:, 1:]
        share = spec / spec.sum(1, keepdims=True)
        x = period[1:]
        people = pd.Index(d["people"])
        top = min(max(bands[dataset])[1] + 5, len(x))
        for i, (lo, hi) in enumerate(bands[dataset]):
            ax.axvspan(x[lo - 1] * 1.02 if lo > 1 else x[0] * 1.2, x[hi - 2] * 0.98, color=GRID,
                       alpha=0.75 if i % 2 == 0 else 0.35, linewidth=0, zorder=0)
        for k, name, colour, members in by_endpoint(d["groups"]):
            band_plot(ax, x[:top], share[people.get_indexer(members)][:, :top], colour, name,
                      marker="o", markersize=2.5)
        for h in (24, 12, 8, 6):
            if h * 2 <= x[0] and h > x[top - 1]:
                ax.axvline(h, color=INK2, linewidth=0.7, linestyle=(0, (3, 2)))
                ax.text(h, ax.get_ylim()[1] if False else 1.0, f"{h} h", transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=6.5, color=INK2)
        ax.set_xscale("log")
        ax.set_xlim(x[0] * 1.05, x[top - 1] * 0.95)
        ax.set_xlabel("Period (hours, log scale)")
        ax.set_ylabel("Share of V$^S$ power")
        ax.set_title(f"{'ab'[row]}  {SHORT[dataset]}: where the rhythm branch puts its power", loc="left", pad=12)
        legend_counts(ax, d["groups"], loc="upper left")
        recessive_grid(ax, "y")
    fig.text(0.0, -0.03, "Grey bands: the branch's Fourier bands; it is zero outside them. Mean share per period "
             "with 95% participant-bootstrap bands.", fontsize=6.3, color=INK2)
    fig.savefig(out / "rep_spectrum.pdf")
    plt.close(fig)


def rq2_movement(d, out):
    rq2 = d["rq2"]
    parts = np.concatenate([r["rq2_participant"] for r in rq2])
    get = lambda k: np.concatenate([r[f"rq2_{k}"] for r in rq2])
    ref, cur, sh, up, dn = get("ref"), get("current"), get("shifted"), get("scaled_up"), get("scaled_down")
    hours, alpha = rq2[0]["rq2_shift_hours"], rq2[0]["rq2_alpha"]
    mu, sd = ref.mean(1), ref.std(1, ddof=1) + 1e-6
    dist = lambda v: np.sqrt((((v - mu[:, None]) / sd[:, None]) ** 2).mean(-1))
    base = np.sqrt((((cur - mu) / sd) ** 2).mean(-1))
    # Example: fixed rule, the participant with the most exported windows (ties: first by id).
    counts = d["windows"].groupby("participant").size().reindex(parts).fillna(0)
    ex = int(np.argmax(counts.to_numpy()))
    pts = np.vstack([ref[ex], cur[ex][None], sh[ex], up[ex], dn[ex]])
    z = (pts - mu[ex]) / sd[ex]
    z = z - z[4]                                    # everything relative to the unperturbed week
    # Axes: the direction the largest shift moves the week, and the part of the largest
    # amplification orthogonal to it. Both in units of the participant's own week-to-week SD.
    u = z[9] / np.linalg.norm(z[9])
    v = z[14] - (z[14] @ u) * u
    v = v / np.linalg.norm(v)
    xy = z @ np.stack([u, v], 1)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.1), gridspec_kw=dict(wspace=0.62, width_ratios=[1.45, 1, 1]))
    ax = axes[0]
    ax.scatter(*xy[:4].T, s=22, color=INK2, alpha=0.7, zorder=3, label="4 reference weeks")
    ax.scatter(*xy[4], s=55, color=INK, marker="*", zorder=4, label="current week (origin)")
    ax.plot(*np.vstack([xy[4], xy[5:10]]).T, color=BLUE, marker="o", markersize=3.5, linewidth=1.4,
            label="shifted 0.5–4 h later")
    ax.plot(*np.vstack([xy[4], xy[10:15]]).T, color=AQUA, marker="^", markersize=3.5, linewidth=1.4,
            label=r"24 h strength $\times(1+\alpha)$")
    ax.plot(*np.vstack([xy[4], xy[15:20]]).T, color=AQUA, marker="v", markersize=3.5, linewidth=1.2,
            linestyle=(0, (3, 2)), label=r"$\times(1-\alpha)$")
    ax.axhline(0, color=GRID, linewidth=0.8, zorder=0)
    ax.axvline(0, color=GRID, linewidth=0.8, zorder=0)
    for k in (0, len(hours) - 1):
        ax.annotate(f"{hours[k]:g} h", xy=xy[5 + k], xytext=(4, -6), textcoords="offset points",
                    fontsize=6, color=BLUE)
    for k in (0, len(alpha) - 1):
        ax.annotate(fr"$\alpha$={alpha[k]:g}", xy=xy[10 + k], xytext=(5, 0), textcoords="offset points",
                    fontsize=6, color=AQUA)
    ax.set_xlabel("Movement along the timing direction")
    ax.set_ylabel("Movement along the strength direction")
    ax.set_title("a  One participant", loc="left")
    ax.legend(frameon=False, fontsize=5.6, loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=2,
              handlelength=1.4, columnspacing=0.9)
    recessive_grid(ax, "both")
    for ax, (vals, xs, xlabel, title, colour) in zip(axes[1:], (
            (dist(sh), hours, "Shift (hours)", "b  Timing change", BLUE),
            (np.stack([dist(up), dist(dn)]), alpha, r"Strength change $\alpha$", "c  Strength change", AQUA))):
        if vals.ndim == 3:
            for v, style, lab in ((vals[0], "-", r"$1+\alpha$"), (vals[1], (0, (3, 2)), r"$1-\alpha$")):
                m, lo, hi = mean_band(v - base[:, None])
                ax.fill_between(xs, lo, hi, color=colour, alpha=0.18, linewidth=0)
                ax.plot(xs, m, color=colour, linestyle=style, marker="o", markersize=3.5, label=lab)
            ax.legend(frameon=False, fontsize=6.5)
        else:
            m, lo, hi = mean_band(vals - base[:, None])
            ax.fill_between(xs, lo, hi, color=colour, alpha=0.18, linewidth=0)
            ax.plot(xs, m, color=colour, marker="o", markersize=3.5)
        ax.axhline(0, color=INK2, linewidth=0.7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Change in distance from\nown reference")
        ax.set_title(title, loc="left")
        recessive_grid(ax, "y")
    fig.text(0.0, -0.42, f"(a) Participant chosen by a fixed rule (most weeks). Axes: the direction in which the "
             f"largest shift moves the week, and the part of the largest amplification orthogonal to it,\nin units of "
             f"that person's week-to-week SD. (b, c) All {len(parts)} participants with a four-week reference: change "
             "in the standardised distance of the perturbed week from its\nreference, relative to the unperturbed "
             "week; mean and 95% participant-bootstrap band. Seed 1.", fontsize=6.3, color=INK2)
    fig.savefig(out / "rep_rq2_movement.pdf")
    plt.close(fig)


def actogram(d, out):
    A, g, b = d["arrays"], d["groups"], d["meta"]["bins_per_day"]
    people = pd.Index(d["people"])
    raw_df, raw_prof = raw_participant(d["cohort"], d["people"])
    zrow = lambda x: (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-9)
    rep = zrow(A["profile_seasonal"][:, :, 0])
    raw = zrow(raw_prof)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.2), sharex=True,
                             gridspec_kw=dict(hspace=0.12, wspace=0.08))
    for col, k in enumerate((0, 1)):
        members = g.index[g.endpoint == k]
        order = raw_df.loc[members, "acrophase_hour"].sort_values().index      # same order in both rows
        idx = people.get_indexer(order)
        for row, (mat, title) in enumerate(((raw, "Measured steps"), (rep, "Rhythm branch V$^S$, component 1"))):
            ax = axes[row, col]
            im = ax.imshow(mat[idx], aspect="auto", cmap=DIVERGING, vmin=-2.5, vmax=2.5,
                           extent=[0, 24, len(idx), 0], interpolation="nearest")
            ax.set_yticks([])
            ax.set_title(f"{ENDPOINT[k][0]} (n={len(idx)}): {title}" if row == 0 else title, loc="left", fontsize=7.5)
            if col == 0:
                ax.set_ylabel("Participants, by measured\npeak time (earliest on top)")
            if row == 1:
                hour_axis(ax)
    cb = fig.colorbar(im, ax=axes, shrink=0.5, pad=0.02)
    cb.set_label("Above / below the person's own daily mean (SD)")
    fig.text(0.0, 0.03, "Each row is one participant's average day, standardised within the row. Rows keep the same "
             "order in both panels of a column, so a participant's measured day (top)\ncan be compared with how the "
             "rhythm branch represents it (bottom).", fontsize=6.3, color=INK2)
    fig.savefig(out / "rep_actogram.pdf")
    plt.close(fig)


def participant_features(d):
    w = d["windows"]
    per = w.groupby("participant").agg(
        p24=("p24", lambda v: np.log(np.mean(v))), p12=("p12", lambda v: np.log(np.mean(v))),
        day_corr_s=("day_corr_seasonal", "mean"), day_corr_t=("day_corr_trend", "mean"),
        level=("trend_level", "mean"), slope=("trend_slope", "mean"), level_sd=("trend_level", "std"),
        peak=("peak_hour", lambda h: (np.angle(np.exp(1j * h * np.pi / 12).mean()) * 12 / np.pi) % 24))
    return per


FEATURES = [("Rhythm branch V$^S$", [("p24", "24 h power (log)"), ("p12", "12 h power (log)"),
                                    ("day_corr_s", "Day-to-day similarity")]),
            ("Trend branch V$^T$", [("level", "Level"), ("slope", "Within-week slope"),
                                   ("level_sd", "Week-to-week variability"), ("day_corr_t", "Day-to-day similarity")]),
            ("Measured steps", [("amplitude", "24 h amplitude"), ("IS", "Day-to-day regularity (IS)"),
                                ("MESOR", "Level (MESOR)")])]


def group_features(ds, out):
    rows = []
    for dataset, d in ds.items():
        per = participant_features(d).join(raw_participant(d["cohort"], d["people"])[0])
        fam = []
        for branch, feats in FEATURES:
            for key, label in feats:
                if per[key].notna().sum() < 10:
                    continue
                s = split(per[key], d["groups"])
                fam.append(dict(dataset=dataset, branch=branch, feature=label, kind="g",
                                **compare(s[1], s[0], hedges)))
        for key, label in (("peak", "Rhythm branch V$^S$: peak time"), ("acrophase_hour", "Measured steps: peak time")):
            s = split(per[key], d["groups"])
            fam.append(dict(dataset=dataset, branch=label.split(":")[0], feature=label, kind="minutes",
                            **compare(s[1], s[0], circ_minutes)))
        f = pd.DataFrame(fam)
        f["p_holm"] = holm(f.p)
        rows.append(f)
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(out / "rep_group_features.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6), gridspec_kw=dict(width_ratios=[2.2, 1], wspace=0.55))
    offs = {"hrd": 0.16, "globem": -0.16}
    colours = {"hrd": INK, "globem": INK2}
    gpart = res[res.kind == "g"]
    labels = list(dict.fromkeys(zip(gpart.branch, gpart.feature)))
    y = np.arange(len(labels))[::-1]
    for dataset in ds:
        q = gpart[gpart.dataset == dataset].set_index(["branch", "feature"])
        for yi, key in zip(y, labels):
            if key in q.index:
                r = q.loc[key]
                axes[0].plot([r.low, r.high], [yi + offs[dataset]] * 2, color=colours[dataset], linewidth=1.3)
                axes[0].plot(r.effect, yi + offs[dataset], "o", color=colours[dataset], markersize=4,
                             markerfacecolor=colours[dataset] if r.p_holm < 0.05 else "white")
    axes[0].axvline(0, color=INK2, linewidth=0.7)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{b}: {f}" for b, f in labels], fontsize=6.3)
    axes[0].set_xlabel("Hedges' g, elevated minus not elevated")
    axes[0].set_title("a  Group differences per branch", loc="left")
    recessive_grid(axes[0])
    mpart = res[res.kind == "minutes"]
    ym = np.arange(mpart.feature.nunique())[::-1]
    for dataset in ds:
        q = mpart[mpart.dataset == dataset].set_index("feature")
        for yi, feat in zip(ym, dict.fromkeys(mpart.feature)):
            if feat in q.index:
                r = q.loc[feat]
                axes[1].plot([r.low, r.high], [yi + offs[dataset]] * 2, color=colours[dataset], linewidth=1.3)
                axes[1].plot(r.effect, yi + offs[dataset], "o", color=colours[dataset], markersize=4,
                             markerfacecolor=colours[dataset] if r.p_holm < 0.05 else "white")
    axes[1].axvline(0, color=INK2, linewidth=0.7)
    axes[1].set_yticks(ym)
    axes[1].set_yticklabels([f.split(": ")[0] for f in dict.fromkeys(mpart.feature)], fontsize=6.3)
    axes[1].set_xlabel("Later peak in elevated group (min)")
    axes[1].set_title("b  Peak time", loc="left")
    recessive_grid(axes[1])
    handles = [plt.Line2D([], [], color=colours[k], marker="o", linewidth=1.3, markersize=4, label=SHORT[k]) for k in ds]
    fig.legend(handles=handles, frameon=False, fontsize=6.8, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.06))
    counts = "; ".join(f"{NAME[k]}: {int((d['groups'].endpoint == 1).sum())} elevated, "
                       f"{int((d['groups'].endpoint == 0).sum())} not" for k, d in ds.items())
    fig.text(0.0, -0.05, f"{counts}. 95% stratified bootstrap intervals; filled: Holm-corrected permutation p < 0.05 "
             "within each dataset. 12 h power only on HRD (on GLOBEM it is the Nyquist bin).", fontsize=6.3, color=INK2)
    fig.savefig(out / "rep_group_features.pdf")
    plt.close(fig)
    return res


def trajectories(d, out):
    w = d["windows"].copy()
    start = pd.to_datetime(w.window_id.str.rsplit("_", n=1).str[1])
    w["week"] = ((start - start.groupby(w.participant).transform("min")).dt.days // 7)
    w["bin"] = (w.week // 4) * 4
    w = w[w.bin <= 28]
    w["course"] = d["groups"].course.reindex(w.participant).to_numpy()
    w["log_p24"] = np.log(w.p24)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw=dict(wspace=0.35))
    for ax, (col, ylabel, title) in zip(axes, (("log_p24", "V$^S$ 24 h power (log)", "a  Rhythm branch"),
                                              ("trend_level", "V$^T$ level (SD units)", "b  Trend branch"))):
        for grp, (name, colour, marker) in COURSE.items():
            per = w[w.course == grp].groupby(["participant", "bin"])[col].mean().unstack()
            if per.empty:
                continue
            m, lo, hi = mean_band(per.to_numpy())
            x = per.columns.to_numpy() + 2
            ax.fill_between(x, lo, hi, color=colour, alpha=0.13, linewidth=0)
            ax.plot(x, m, color=colour, marker=marker, markersize=3.5, linewidth=1.5,
                    label=f"{name} (n={per.shape[0]})")
        ax.set_xlabel("Weeks since first window (4-week bins)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        recessive_grid(ax, "y")
    axes[0].legend(frameon=False, fontsize=6.3, loc="best")
    fig.text(0.0, -0.06, "Symptom course from study-entry and study-end CES-D. Mean over participants of each "
             "4-week bin with 95% participant-bootstrap bands; groups of 9 and 13 participants are wide by design.",
             fontsize=6.3, color=INK2)
    fig.savefig(out / "rep_trajectories.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="narval_v3", help="HRD run name")
    ap.add_argument("--globem-run", default="loyo_v3", help="GLOBEM leave-one-year-out run name")
    ap.add_argument("--out", type=Path, default=ROOT / "SSL_Rhythmicity" / "figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.bbox": "tight", "savefig.pad_inches": 0.03})
    ds = {"hrd": load("hrd", args.run), "globem": load("globem", args.globem_run)}
    from models.encoder import rhythm_bands
    bands = {k: rhythm_bands(d["cohort"].seq_len, d["cohort"].bins_per_day) for k, d in ds.items()}
    rhythm_shape(ds["hrd"], args.out)
    timescales(ds, args.out)
    spectrum(ds, args.out, bands)
    rq2_movement(ds["hrd"], args.out)
    actogram(ds["hrd"], args.out)
    res = group_features(ds, args.out)
    trajectories(ds["hrd"], args.out)
    print(res.round(3).to_string())
    print("wrote", sorted(p.name for p in args.out.glob("rep_*.pdf")))


if __name__ == "__main__":
    main()
