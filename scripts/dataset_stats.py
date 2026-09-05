"""TASK -- describe the two input corpora (HRD and GLOBEM) side by side.

Runs BEFORE any model: measures each raw CSV directly, so the preprocessing and
modelling choices downstream can be justified from what the data actually holds.
Covers both datasets in one pass -- there is no per-dataset variant of this script.

Writes, and prints every number it plots:
    fig1_channel_coverage.png   observed fraction per model channel, both datasets
    fig2_label_density.png      how often a label arrives, both datasets
    fig3_distributions.png      recording length (both) + CES-D severity (HRD only)
    an HRD-vs-GLOBEM comparison table on stdout

The HRD pass streams a 3.2 GB file in chunks (a couple of minutes); nothing is
cached, so every number is re-measured on every run.

    python scripts/dataset_stats.py [--out-dir docs/figures]
"""
import argparse
import collections
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tasks.style import (BASE, C_GLB, C_HRD, CRIT, GRID, INK, INK2,   # noqa: E402
                    MUTED, SURFACE, save, strip)
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.patches import Rectangle                     # noqa: E402

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--hrd-csv", default="datasets/HRD_RAW_MinuteLevel.csv")
ap.add_argument("--globem-csv", default="datasets/GLOBEM_REDUCED.csv")
ap.add_argument("--out-dir", default=str(Path("docs") / "figures"))
args = ap.parse_args()
OUT = args.out_dir
S = {}

# ===================================================================== measure GLOBEM
g = pd.read_csv(args.globem_csv, low_memory=False)
feat = [c for c in g.columns if c.startswith("f_")]
g["date"] = pd.to_datetime(g["date"], errors="coerce")
gd = g.dropna(subset=["date"])
days = gd.groupby("pid")["date"].agg(lambda s: s.dt.date.nunique())
lab = gd.groupby("pid")["LABEL_ENDPOINT"].first()
lb = lab.astype(str).str.lower().isin(["true", "1", "1.0"])
wk = gd.dropna(subset=["LABEL_WEEKLY"])
wkb = wk["LABEL_WEEKLY"].astype(str).str.lower().isin(["true", "1", "1.0"])
S["globem"] = dict(
    rows=len(g), participants=int(g["pid"].nunique()),
    date_min=str(gd["date"].min().date()), date_max=str(gd["date"].max().date()),
    days_median=float(days.median()), days_hist=days.values.tolist(),
    n_labelled=int(lab.notna().sum()), n_pos=int(lb.sum()),
    n_neg=int((~lb & lab.notna()).sum()),
    # the weekly label is repeated on all 4 segments of its day, so the number of SURVEYS
    # is the distinct (pid, date) count -- not the row count
    weekly_surveys=int(wk.groupby(["pid", "date"]).ngroups),
    weekly_pos_frac=float(wkb.mean()),
    weekly_per_pid_median=float(wk.groupby("pid")["date"].nunique().median()),
    present={c: float(gd[c].notna().mean()) for c in feat},
)
del g, gd

# ===================================================================== measure HRD (streamed)
NUM = ["heart_rate", "floors", "fairly_active_minutes", "lightly_active_minutes",
       "very_active_minutes", "sedentary_minutes", "steps_minutes", "sleep_status",
       "screen", "call"]
LAB = ["depression_status_baseline", "depression_status_endpoint",
       "depression_trajectory", "ces_d_endpoint_score"]
n, notna, pid_days, tmin, tmax = 0, {c: 0 for c in NUM}, {}, None, None
labmap = {c: {} for c in LAB}
for ch in pd.read_csv(args.hrd_csv, usecols=["pid", "timestamp"] + NUM + LAB,
                      chunksize=3_000_000, low_memory=False):
    n += len(ch)
    for c in NUM:
        notna[c] += int(ch[c].notna().sum())
    t = pd.to_datetime(ch["timestamp"], errors="coerce")
    tmin = t.min() if tmin is None else min(tmin, t.min())
    tmax = t.max() if tmax is None else max(tmax, t.max())
    for pid, d in zip(ch["pid"], t.dt.date):
        pid_days.setdefault(pid, set()).add(d)
    for c in LAB:
        for k, v in ch.groupby("pid")[c].first().items():
            labmap[c].setdefault(k, v)

days_h = pd.Series({p: len(v) for p, v in pid_days.items()})
ep = pd.Series(labmap["depression_status_endpoint"]).dropna()
bl = pd.Series(labmap["depression_status_baseline"]).dropna()
tj = pd.Series(labmap["depression_trajectory"]).dropna()
ce = pd.Series(labmap["ces_d_endpoint_score"]).dropna().astype(float)
S["hrd"] = dict(
    rows=n, participants=len(pid_days),
    date_min=str(tmin.date()), date_max=str(tmax.date()),
    days_median=float(days_h.median()), days_max=int(days_h.max()),
    days_hist=days_h.values.tolist(), present={c: notna[c] / n for c in NUM},
    n_endpoint=int(len(ep)), endpoint_pos=int((ep == 1).sum()), endpoint_neg=int((ep == 0).sum()),
    n_baseline=int(len(bl)), trajectory={str(k): int(v) for k, v in tj.value_counts().items()},
    cesd_median=float(ce.median()), cesd_hist=ce.values.tolist(),
)

# ===================================================================== figures
HRD_LAB = {"heart_rate": ("Heart rate", 1), "sleep_status": ("Sleep state", 1),
           "steps_minutes": ("Steps", 0), "very_active_minutes": ("Very-active min", 0),
           "fairly_active_minutes": ("Fairly-active min", 0),
           "lightly_active_minutes": ("Lightly-active min", 0),
           "screen": ("Screen events", 0)}
G = "f_slp:fitbit_sleep_intraday_rapids_"
G_LAB = {G + "sumdurationasleepunifiedmain": ("Sleep: asleep", 1),
         G + "sumdurationawakeunifiedmain": ("Sleep: awake", 1),
         G + "ratiodurationasleepunifiedwithinmain": ("Sleep: efficiency", 1),
         "f_steps:fitbit_steps_intraday_rapids_sumsteps": ("Steps: total", 0),
         "f_steps:fitbit_steps_intraday_rapids_avgdurationactivebout": ("Steps: active bout", 0),
         "f_steps:fitbit_steps_intraday_rapids_countepisodesedentarybout": ("Steps: sedentary bouts", 0),
         "f_screen:phone_screen_rapids_countepisodeunlock": ("Screen: unlocks", 0),
         "f_screen:phone_screen_rapids_sumdurationunlock": ("Screen: duration", 0),
         "f_loc:phone_locations_doryab_timeathome": ("Location: time at home", 0),
         "f_loc:phone_locations_doryab_locationentropy": ("Location: entropy", 0),
         "f_loc:phone_locations_doryab_totaldistance": ("Location: distance", 0),
         "f_loc:phone_locations_doryab_numberlocationtransitions": ("Location: transitions", 0)}

# --- Fig 1: coverage -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
for ax, (key, mapping, colour, title) in zip(axes, [
        ("hrd", HRD_LAB, C_HRD, "HRD  ·  7 model channels  ·  1-minute grid"),
        ("globem", G_LAB, C_GLB, "GLOBEM  ·  12 model channels  ·  6-hour segments")]):
    items = [(mapping[c][0], mapping[c][1], S[key]["present"][c] * 100)
             for c in mapping if c in S[key]["present"]]
    items.sort(key=lambda t: (-t[1], -t[2]))          # physiological first, then by coverage
    y = np.arange(len(items))[::-1]
    ax.barh(y, [t[2] for t in items], height=0.62, color=colour, edgecolor=SURFACE, linewidth=2)
    for yy, (_, _k, v) in zip(y, items):
        ax.text(v + 1.5, yy, f"{v:.0f}%", va="center", ha="left", fontsize=8.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{t[0]}  ·  {'PHYS' if t[1] else 'BEHAV'}" for t in items],
                       fontsize=8.5, color=INK2)
    for lbl, (_, kind, _v) in zip(ax.get_yticklabels(), items):
        if kind:
            lbl.set_color(INK), lbl.set_fontweight("bold")
    ax.set_xlim(0, 112), ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("timestamps with an observed value (%)", fontsize=8.5)
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=10)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7), ax.set_axisbelow(True)
    strip(ax)
fig.suptitle("Channel coverage before imputation — bold rows are physiological",
             fontsize=11.5, color=INK, x=0.012, ha="left", y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.94])
save(fig, OUT, "fig1_channel_coverage.png")

# --- Fig 2: label density --------------------------------------------------------
gap = 97 / S["globem"]["weekly_per_pid_median"]
fig, axes = plt.subplots(2, 1, figsize=(11.5, 4.4), sharex=True)
for ax, (name, span, marks, colour, note) in zip(axes, [
        ("HRD", int(S["hrd"]["days_median"]), [S["hrd"]["days_median"]], C_HRD,
         "one endpoint label per participant, at the end of follow-up"),
        ("GLOBEM", 97, list(np.arange(gap * 0.8, 98, gap)), C_GLB,
         f"~{S['globem']['weekly_per_pid_median']:.0f} weekly surveys per participant, "
         f"one every ~{gap:.1f} days")]):
    ax.add_patch(Rectangle((0, 0.32), span, 0.36, facecolor=colour, alpha=0.16, edgecolor="none"))
    ax.plot([0, span], [0.5, 0.5], color=colour, linewidth=2, solid_capstyle="butt")
    ax.scatter(marks, [0.5] * len(marks), s=64, color=colour, zorder=3,
               edgecolor=SURFACE, linewidth=2)
    ax.text(0, 0.86, name, fontsize=11, color=INK, fontweight="bold", va="bottom")
    ax.text(0, 0.05, note, fontsize=8.5, color=MUTED, va="bottom")
    ax.text(span + 4, 0.5, f"{span} d", fontsize=8.5, color=INK2, va="center")
    ax.set_ylim(0, 1.05), ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7), ax.set_axisbelow(True)
axes[1].set_xlim(-3, S["hrd"]["days_median"] + 22)
axes[1].set_xlabel("days since a participant's first recording  (median-length participant)",
                   fontsize=8.5)
fig.suptitle("Label density is the axis the two datasets differ on most",
             fontsize=11.5, color=INK, x=0.012, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.92])
save(fig, OUT, "fig2_label_density.png")

# --- Fig 3: distributions --------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
ax = axes[0]
ax.hist(S["hrd"]["days_hist"], bins=28, color=C_HRD, edgecolor=SURFACE, linewidth=1.2)
ax.axvline(S["hrd"]["days_median"], color=INK, linewidth=1.4)
ax.text(S["hrd"]["days_median"] + 14, ax.get_ylim()[1] * 0.78,
        f"median {S['hrd']['days_median']:.0f} d", fontsize=8.5, color=INK)
ax.set_title("HRD · days recorded per participant", fontsize=9.5, color=INK, loc="left")
ax.set_xlabel("days", fontsize=8.5), ax.set_ylabel("participants", fontsize=8.5)

# GLOBEM takes only a handful of distinct recording lengths -- one per study cohort -- so a
# histogram would draw lonely spikes and imply a continuum that is not there. Show the values.
ax = axes[1]
cnt = collections.Counter(S["globem"]["days_hist"])
xs = sorted(cnt)
ax.bar(range(len(xs)), [cnt[x] for x in xs], width=0.5, color=C_GLB,
       edgecolor=SURFACE, linewidth=2)
for i, x in enumerate(xs):
    ax.text(i, cnt[x] + max(cnt.values()) * 0.025, f"{cnt[x]}", ha="center",
            fontsize=8.5, color=INK2)
ax.set_xticks(range(len(xs))), ax.set_xticklabels([f"{x} d" for x in xs])
ax.set_ylim(0, max(cnt.values()) * 1.18)
ax.set_title(f"GLOBEM · recording length ({len(xs)} cohorts)", fontsize=9.5,
             color=INK, loc="left")
ax.set_xlabel("days recorded — administratively fixed", fontsize=8.5)

ax = axes[2]
ax.hist(S["hrd"]["cesd_hist"], bins=22, color=C_HRD, edgecolor=SURFACE, linewidth=1.2)
ax.axvline(16, color=CRIT, linewidth=1.4)
ax.text(17, ax.get_ylim()[1] * 0.9, "CES-D ≥ 16\nclinical cut-off",
        fontsize=8.5, color=CRIT, va="top")
ax.set_title(f"HRD · CES-D endpoint score (n={len(S['hrd']['cesd_hist'])})",
             fontsize=9.5, color=INK, loc="left")
ax.set_xlabel("CES-D score", fontsize=8.5)

for ax in axes:
    ax.yaxis.grid(True, color=GRID, linewidth=0.7), ax.set_axisbelow(True)
    strip(ax)
fig.suptitle("GLOBEM's recording length is fixed by cohort; HRD's varies widely — "
             "and HRD alone carries a continuous severity score",
             fontsize=11, color=INK, x=0.012, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.90])
save(fig, OUT, "fig3_distributions.png")

# ===================================================================== printed summary
h, gl = S["hrd"], S["globem"]


def row(lab, a, b):
    print(f"  {lab:<24}{str(a):>28}{str(b):>30}")


print("\n" + "=" * 84)
print(f"  {'':<24}{'HRD':>28}{'GLOBEM':>30}")
print("=" * 84)
row("records", f"{h['rows']:,}", f"{gl['rows']:,}")
row("participants", h["participants"], gl["participants"])
row("period", f"{h['date_min']}..{h['date_max']}", f"{gl['date_min']}..{gl['date_max']}")
row("granularity", "1 minute", "4 segments/day (6 h)")
row("days per participant", f"median {h['days_median']:.0f} (max {h['days_max']})",
    "/".join(str(x) for x in sorted(set(gl["days_hist"]))))
row("model channels", 7, 12)
row("endpoint-labelled", f"{h['n_endpoint']} ({h['endpoint_pos']}+/{h['endpoint_neg']}-)",
    f"{gl['n_labelled']} ({gl['n_pos']}+/{gl['n_neg']}-)")
row("repeated labels", "none", f"{gl['weekly_surveys']:,} weekly surveys")
row("continuous severity", f"CES-D median {h['cesd_median']:.0f}", "none")
row("records per label", f"~{h['rows'] // max(h['n_endpoint'], 1):,}",
    f"~{gl['rows'] // max(gl['weekly_surveys'], 1):,}")
print("=" * 84)
