"""TASK -- render the result panels for one training sweep, for EITHER dataset.

Dataset-agnostic by design: point ``--results-dir`` at ``results_hrd`` or
``results_globem``. Run ids are always given explicitly on the command line, so a
figure can never be silently built from runs that are not on disk (the previous
version hard-coded four ids and, once they were gone, wrote four all-NaN figures
without a word).

Per ``<results-dir>/<run>/<variant>/`` it reads
    metrics.json                  required -- backbone, pe, seed, downstream scores
    probe_scores.json             panels A and B (per-subject probe outputs)
    decomposition_recovery.json   panels C and D (DIS / recovery / leak)

Writes, and prints every number it plots:
    figA_reduction_asymmetry.png     paired effect of PCA-20, per representation view
    figB_representation_ranking.png  subject AUC of every view under one probe
    figC_pe_dissociation.png         positional encoding: prediction vs disentanglement
    figD_clock_ablation.png          paired calendar-channel on/off (only with --clock-*)

    python scripts/results_figures.py --runs 19314126
    python scripts/results_figures.py --results-dir results_globem --runs <id> <id>
    python scripts/results_figures.py --runs <id> --clock-on <id> --clock-off <id>
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _results import census, in_cell, load_variants               # noqa: E402
from _style import (ACCENT, BASE, GRID, INK, INK2, MUTED,         # noqa: E402
                    NEG, POS, SURFACE, save, strip)
import matplotlib.pyplot as plt                                   # noqa: E402


# ------------------------------------------------------------------- statistics
def ci(per):
    """(mean, lo, hi) over per-seed values; NaN-safe and never raises on n < 2."""
    per = np.asarray(per, dtype=float)
    per = per[~np.isnan(per)]
    if per.size == 0:
        return np.nan, np.nan, np.nan
    if per.size < 2:
        return float(per[0]), float(per[0]), float(per[0])
    lo, hi = stats.t.interval(.95, per.size - 1, per.mean(), stats.sem(per))
    return float(per.mean()), float(lo), float(hi)


def auc(views, name):
    return roc_auc_score(views[name]["y"], views[name]["prob"])


def _per_cell(cur, cells, fn):
    """Mean of ``fn(rec)`` within each independent cell, one value per cell (NaN if absent)."""
    vals = []
    for cell in cells:
        v = [fn(r) for k, r in cur.items() if in_cell(k, cell)]
        v = [x for x in v if x is not None]
        vals.append(np.mean(v) if v else np.nan)
    return np.array(vals, dtype=float)


def paired(cur, cells, a, b):
    """Cell-level paired AUC contrast between two representation views."""
    def f(r):
        v = r["views"]
        if a in v and b in v and np.array_equal(v[a]["y"], v[b]["y"]):
            return auc(v, a) - auc(v, b)
        return None
    per = _per_cell(cur, cells, f)
    m, lo, hi = ci(per)
    keep = per[~np.isnan(per)]
    p = stats.ttest_1samp(keep, 0.).pvalue if keep.size > 1 else np.nan
    return m, lo, hi, p


def level(cur, cells, view):
    return ci(_per_cell(cur, cells, lambda r: auc(r["views"], view)
                        if view in r["views"] else None))


# ============================================================ FIG A
ROWS = [("V (encoder pre-decomp)", 320, "V  (pre-decomposition)"),
        ("Full [V^(T);V^(S)]", 320, "Full  [V^T;V^S]"),
        ("Trend V^(T)", 160, "Trend  V^T"), ("Season V^(S)", 160, "Season  V^S"),
        ("Seasonal amp", 2400, "Seasonal amplitude"),
        ("Seasonal phase", 2400, "Seasonal phase"),
        ("Seasonal Re/Im", 4800, "Seasonal Re/Im")]


def fig_a(cur, cells, out_dir):
    print("\n" + "=" * 78 + "\nFIG A - effect of PCA-20 by native dimensionality")
    print(f"{'representation':<24}{'dim':>6}{'delta':>9}{'95% CI':>19}{'p':>8}")
    A = []
    for view, dim, lab in ROWS:
        d, lo, hi, p = paired(cur, cells, f"{view} (PCA)", view)
        if np.isnan(d):
            print(f"{lab:<24}{dim:>6}      -- view pair absent, skipped")
            continue
        A.append((lab, dim, d, lo, hi, p))
        print(f"{lab:<24}{dim:>6}{d:>+9.4f}  [{lo:+.3f},{hi:+.3f}]{p:>8.4f}")
    if not A:
        print("  no view pairs present; figure A skipped")
        return

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    A.sort(key=lambda r: r[1])
    for i, (lab, dim, d, lo, hi, p) in enumerate(A):
        ax.barh(i, d, height=0.6, color=POS if d > 0 else NEG, edgecolor=SURFACE, linewidth=2)
        ax.plot([lo, hi], [i, i], color=INK, linewidth=1.3, zorder=3)
        ax.text(hi + 0.002 if d > 0 else lo - 0.002, i, f"{d:+.3f}", va="center",
                ha="left" if d > 0 else "right", fontsize=8.5, color=INK2)
    ax.axvline(0, color=BASE, linewidth=1.1)
    ax.set_yticks(range(len(A)))
    ax.set_yticklabels([f"{lab}\n{dim:,} dim" for lab, dim, *_ in A], fontsize=8.5, color=INK2)
    ax.set_xlabel("change in subject AUC when the view is reduced to 20 principal components",
                  fontsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7), ax.set_axisbelow(True)
    strip(ax)
    fig.suptitle("Effect of dimensionality reduction, by native dimensionality",
                 fontsize=12, color=INK, x=0.012, ha="left", y=0.98)
    fig.text(0.012, 0.905, f"bars are the paired change, black rules the 95 % CI; "
             f"{len(cells)} independent cells x {len({(k.backbone, k.pe) for k in cur})} encoder variants",
             fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    save(fig, out_dir, "figA_reduction_asymmetry.png")


# ============================================================ FIG B
RANK = [("Seasonal amp+phase (block PCA)", "amp+phase, per-block PCA", "spectral"),
        ("Seasonal amp (PCA)", "Seasonal amplitude (PCA-20)", "spectral"),
        ("V (encoder pre-decomp)", "V  (pre-decomposition)", "cost"),
        ("Seasonal Re/Im (PCA)", "Seasonal Re/Im (PCA-20)", "spectral"),
        ("Transformer-sin (supervised)", "Transformer (supervised)", "ref"),
        ("Seasonal phase (PCA)", "Seasonal phase (PCA-20)", "spectral"),
        ("Full [V^(T);V^(S)]", "Full  [V^T;V^S]", "cost"),
        ("Seasonal amp (PLS)", "Seasonal amplitude (PLS-20)", "spectral"),
        ("TCN (supervised)", "TCN (supervised)", "ref"),
        ("Trend V^(T)", "Trend  V^T", "cost"),
        ("Seasonal amp", "Seasonal amplitude (full)", "spectral"),
        ("Season V^(S)", "Season  V^S", "cost"),
        ("Cosinor (paper)", "Cosinor (classical)", "ref")]
COL = {"spectral": ACCENT, "cost": POS, "ref": MUTED}


def fig_b(cur, cells, out_dir):
    print("\n" + "=" * 78 + "\nFIG B - representation ranking (subject AUC)")
    B = []
    for view, lab, kind in RANK:
        m, lo, hi = level(cur, cells, view)
        if np.isnan(m):
            print(f"  {lab:<32}-- absent, skipped")
            continue
        B.append((lab, kind, m, lo, hi))
        print(f"  {lab:<32}{m:.4f}   [{lo:.4f}, {hi:.4f}]")
    if not B:
        print("  no views present; figure B skipped")
        return
    B.sort(key=lambda r: r[2])

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for i, (lab, kind, m, lo, hi) in enumerate(B):
        ax.plot([lo, hi], [i, i], color=COL[kind], linewidth=2, solid_capstyle="round", zorder=2)
        ax.scatter([m], [i], s=52, color=COL[kind], zorder=3, edgecolor=SURFACE, linewidth=1.6)
        ax.text(hi + 0.0018, i, f"{m:.3f}", va="center", fontsize=8.5, color=INK2)
    ax.axvline(0.5, color=BASE, linewidth=1.1)
    ax.text(0.5015, -0.55, "chance", fontsize=8, color=MUTED)
    ax.set_yticks(range(len(B))), ax.set_yticklabels([r[0] for r in B], fontsize=8.5, color=INK2)
    ax.set_xlabel(f"subject-level AUC  (mean over {len(cells)} cells, bar = 95 % CI)", fontsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7), ax.set_axisbelow(True)
    strip(ax)
    h = [plt.Line2D([], [], color=COL[k], marker="o", linestyle="-", markersize=7, linewidth=2)
         for k in ("spectral", "cost", "ref")]
    ax.legend(h, ["seasonal spectrum", "CoST representation", "reference model"],
              loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.suptitle("Representation ranking under one probe, split and evaluation unit",
                 fontsize=12, color=INK, x=0.012, ha="left", y=0.98)
    fig.text(0.012, 0.925, "every row uses the same probe, split and evaluation unit, so the rows "
             "are comparable to each other", fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, out_dir, "figB_representation_ranking.png")


# ============================================================ FIG C
def fig_c(cur, out_dir, backbone="transformer"):
    tr = defaultdict(lambda: {"auc": [], "dis": []})
    for k, r in cur.items():
        if k.backbone != backbone:
            continue
        tr[k.pe]["auc"].append(r["m"]["participant_level"]["auc_roc"])
        tr[k.pe]["dis"].append(r["d"].get("DIS", float("nan")))
    order = [p for p in sorted(tr, key=lambda p: -np.nanmean(tr[p]["dis"]))
             if len(tr[p]["auc"]) > 1]
    print("\n" + "=" * 78 + "\nFIG C - positional encoding: prediction vs disentanglement")
    if len(order) < 2:
        print(f"  fewer than 2 usable {backbone} PE variants; figure C skipped")
        return
    f_auc = stats.f_oneway(*[tr[p]["auc"] for p in order])
    f_dis = stats.f_oneway(*[tr[p]["dis"] for p in order])
    for p in order:
        print(f"  {p:<14}AUC={np.mean(tr[p]['auc']):.4f}   DIS={np.nanmean(tr[p]['dis']):.4f}")
    print(f"  ANOVA  AUC: F={f_auc[0]:.2f} p={f_auc[1]:.3f}    "
          f"DIS: F={f_dis[0]:.1f} p={f_dis[1]:.1e}")

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    for ax, (key, lab, ttl, fs) in zip(axes, [
            ("auc", "participant-level AUC", "Prediction", f_auc),
            ("dis", "disentanglement (DIS)", "Disentanglement", f_dis)]):
        for i, p in enumerate(order):
            m, lo, hi = ci(tr[p][key])
            c = POS if key == "auc" else MUTED
            ax.plot([lo, hi], [i, i], color=c, linewidth=2, zorder=2)
            ax.scatter([m], [i], s=48, color=c, zorder=3, edgecolor=SURFACE, linewidth=1.6)
        ax.set_yticks(range(len(order))), ax.set_yticklabels(order, fontsize=8.5, color=INK2)
        ax.set_xlabel(lab, fontsize=9)
        ax.set_title(f"{ttl}    F = {fs[0]:.2f}, p = "
                     f"{'%.2f' % fs[1] if fs[1] > 0.01 else '%.0e' % fs[1]}",
                     fontsize=9.5, color=INK, loc="left", pad=8)
        ax.xaxis.grid(True, color=GRID, linewidth=0.7), ax.set_axisbelow(True)
        strip(ax)
    axes[0].axvline(0.5, color=BASE, linewidth=1.1)
    axes[1].axvline(0, color=BASE, linewidth=1.1)
    fig.suptitle(f"Positional encoding ({backbone}): prediction vs how the representation "
                 "is organised", fontsize=12, color=INK, x=0.012, ha="left", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, out_dir, "figC_pe_dissociation.png")


# ============================================================ FIG D
# `better` = +1 when larger is better. Leak is the only metric where an increase is a
# DEGRADATION, so the colour must follow this and not the raw sign.
CLK = [("auc", "participant AUC", "prediction", +1),
       ("DIS", "disentanglement (DIS)", "structure", +1),
       ("rec_rhythm_branch", "rhythm capture, S branch", "structure", +1),
       ("leak_into_rhythm", "rhythm to trend leak", "structure", -1)]


def fig_d(on, off, out_dir):
    # Pair a clock-on cell with the clock-off cell matching on everything but the run id --
    # fold and target included, so a DS1 cell is never differenced against a DS2 one.
    def pair(k):
        return (k.backbone, k.pe, k.holdout, k.label, k.seed)

    on = {pair(k): r for k, r in on.items()}
    off = {pair(k): r for k, r in off.items()}
    both = sorted(set(on) & set(off))
    print("\n" + "=" * 78 + f"\nFIG D - removing the calendar channels (paired, {len(both)} cells)")
    if len(both) < 2:
        print("  fewer than 2 paired cells; figure D skipped")
        return

    def get(r, path):
        return (r["m"]["participant_level"]["auc_roc"] if path == "auc"
                else r["d"].get(path, float("nan")))

    seeds = sorted({k[-1] for k in both})                  # seed is last in the pair key
    D = []
    for path, lab, kind, better in CLK:
        diff = np.array([get(off[k], path) - get(on[k], path) for k in both], dtype=float)
        per = np.array([np.nanmean(diff[[i for i, k in enumerate(both) if k[-1] == s]])
                        for s in seeds])
        m, lo, hi = ci(per)
        keep = per[~np.isnan(per)]
        p = stats.ttest_1samp(keep, 0.).pvalue if keep.size > 1 else np.nan
        base = abs(np.nanmean([get(off[k], path) for k in both])) or 1e-9
        if np.isnan(m):
            print(f"  {lab:<26}-- absent, skipped")
            continue
        D.append((lab, kind, 100 * m / base, 100 * lo / base, 100 * hi / base, p, better))
        print(f"  {lab:<26}{m:>+9.4f}  [{lo:+.4f},{hi:+.4f}]  p={p:.4f}  "
              f"({100 * m / base:+.1f} % relative)")
    if not D:
        print("  no metrics present; figure D skipped")
        return

    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    D = D[::-1]
    for i, (lab, kind, rd, rlo, rhi, p, better) in enumerate(D):
        ax.barh(i, rd, height=0.58, color=POS if rd * better > 0 else NEG,
                edgecolor=SURFACE, linewidth=2, alpha=1.0 if p < 0.05 else 0.35)
        ax.plot([rlo, rhi], [i, i], color=INK, linewidth=1.3, zorder=3)
        ax.text(rhi + 0.4 if rd > 0 else rlo - 0.4, i,
                f"{rd:+.1f}%" + ("" if p < 0.05 else "  n.s."), va="center",
                ha="left" if rd > 0 else "right", fontsize=8.5, color=INK2)
    ax.axvline(0, color=BASE, linewidth=1.1)
    ax.set_yticks(range(len(D)))
    ax.set_yticklabels([f"{lab}   ·  {kind}" + ("  (lower is better)" if b < 0 else "")
                        for lab, kind, *_r, b in D], fontsize=8.5, color=INK2)
    ax.set_xlabel("relative change when the calendar channels are removed (%)", fontsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7), ax.set_axisbelow(True)
    strip(ax)
    fig.suptitle("Effect of the calendar covariates", fontsize=12, color=INK,
                 x=0.012, ha="left", y=0.99)
    fig.text(0.012, 0.885, f"blue = the property improved, red = it degraded; solid = p < 0.05, "
             f"faded = n.s.; paired over {len(seeds)} seeds", fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    save(fig, out_dir, "figD_clock_ablation.png")


# --------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results_hrd",
                    help="results_hrd | results_globem | any tree with <run>/<variant>/metrics.json")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run (job) ids to pool for panels A-C")
    ap.add_argument("--clock-on", help="run id WITH the calendar channels (enables panel D)")
    ap.add_argument("--clock-off", help="run id WITHOUT the calendar channels (enables panel D)")
    ap.add_argument("--backbone", default="transformer",
                    help="backbone whose PE variants panel C compares")
    # vit_plain ran at batch 4 against 64 for tcn/transformer, so at a fixed --iters budget it
    # saw ~16x less data -- its comparison was never like-for-like.
    ap.add_argument("--exclude", nargs="*", default=["vit_plain"],
                    help="backbones to drop from the study")
    ap.add_argument("--drop-seed", nargs="*", default=[], metavar="RUN:SEED",
                    help="drop one (run, seed) cell, e.g. a seed shared between two pooled runs")
    ap.add_argument("--out-dir", default=str(Path("docs") / "figures"))
    args = ap.parse_args()

    drop = set()
    for spec in args.drop_seed:
        run, _, seed = spec.partition(":")
        if not seed.isdigit():
            raise SystemExit(f"--drop-seed expects RUN:SEED, got {spec!r}")
        drop.add((run, int(seed)))

    cur = load_variants(args.results_dir, args.runs, set(args.exclude), drop)
    if not cur:
        raise SystemExit(
            f"no runs found: no {args.results_dir}/<run>/*/metrics.json for "
            f"runs {args.runs}. Available: "
            f"{sorted(p.name for p in Path(args.results_dir).glob('*') if p.is_dir()) or 'none'}")
    cells = census(cur, args.results_dir)
    print(f"  excluded backbones: {sorted(args.exclude) or 'none'}")

    fig_a(cur, cells, args.out_dir)
    fig_b(cur, cells, args.out_dir)
    fig_c(cur, args.out_dir, args.backbone)

    if args.clock_on and args.clock_off:
        on = load_variants(args.results_dir, [args.clock_on], set(args.exclude), drop)
        off = load_variants(args.results_dir, [args.clock_off], set(args.exclude), drop)
        fig_d(on, off, args.out_dir)
    else:
        print("\nFIG D - skipped (pass --clock-on and --clock-off to build it)")


if __name__ == "__main__":
    main()
