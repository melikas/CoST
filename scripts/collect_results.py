"""Collect every variant's results into one comparison table, and link
rhythmicity to prediction across variants.

For each results_hrd/<run>/<variant>/ it reads:
  * metrics.json      - downstream depression classification (AUC / F1 / Acc)
  * hrd_rhythm.json   - rhythm metrics (FFT-alignment, per-representation probe
                        AUCs incl. the classical cosinor baseline), if present.
  * probe_scores.json - subject-level (label, probe score) pairs, if present.

It prints/writes a table sorted by participant-level AUC and, when rhythm metrics
are available, saves a cross-variant scatter of rhythm capture (how well the
latent recovers the true circadian amplitude, R^2) vs participant AUC with its
Pearson correlation -> the central claim "do rhythm-capturing encodings predict
better?".

From the probe scores it also saves probe_score_report.png: the endpoint separation
on the probe's decision axis, with the ROC averaged over a variant's seeds and a
+/-1 sd band -- the multi-seed view no single run can produce.

Run:  python scripts/collect_results.py --results-dir results_hrd [--csv summary.csv]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _results import iter_metrics, read_json, variant_key   # noqa: E402  (shared results reader)


# EVERY representation the probe scores in hrd_rhythm.json, as (row key, separability names --
# first present wins). These are all subject-level AUCs from the SAME single train/val/test
# split, so they are directly comparable to EACH OTHER; they are NOT the headline `pid_auc`,
# which comes from metrics.json's k-fold model (see hrd_rhythm.separability_table's docstring).
# The cosinor row is "Cosinor (paper)" (exact CosinorPy clone) with the legacy "Cosinor (24h)"
# as a fallback so older runs still populate the column.
_PROBE_VIEWS = [
    ("probe_full_auc",     ("Full [V^(T);V^(S)]",)),
    ("probe_V_auc",        ("V (encoder pre-decomp)",)),
    ("probe_VT_auc",       ("Trend V^(T)",)),
    ("probe_VS_auc",       ("Season V^(S)",)),
    ("amp_auc",            ("Seasonal amp",)),
    ("phase_auc",          ("Seasonal phase",)),
    ("reim_auc",           ("Seasonal Re/Im",)),
    ("cosinor_auc",        ("Cosinor (paper)", "Cosinor (24h)")),
    # matched-dimension companions: the SAME views at one common PCA budget (see
    # hrd_rhythm.separability_table). Runs predating the matched block only have the
    # amp/phase two, so the rest stay NaN there -- read the two blocks as a pair.
    ("probe_fullPCA_auc",  ("Full [V^(T);V^(S)] (PCA)",)),
    ("probe_VPCA_auc",     ("V (encoder pre-decomp) (PCA)",)),
    ("probe_VTPCA_auc",    ("Trend V^(T) (PCA)",)),
    ("probe_VSPCA_auc",    ("Season V^(S) (PCA)",)),
    ("probe_ampPCA_auc",   ("Seasonal amp (PCA)",)),
    ("probe_phasePCA_auc", ("Seasonal phase (PCA)",)),
    ("probe_reimPCA_auc",  ("Seasonal Re/Im (PCA)",)),
    ("probe_blockPCA_auc", ("Seasonal amp+phase (block PCA)",)),
    # hybrid: encoder output at full width + the REDUCED spectrum, each block
    # treated the way section 1 says it should be.
    ("probe_VampPCA_auc",  ("V + Seasonal amp (PCA)",)),
    ("probe_VTampPCA_auc", ("Trend V^(T) + Seasonal amp (PCA)",)),
    ("cosinorPCA_auc",     ("Cosinor (paper) (PCA)",)),
    # SUPERVISED reduction at the same budget: the control that turns "reduction is
    # necessary" into a claim about WHICH reducer. Compare against ampPCA/phasePCA.
    ("probe_ampPLS_auc",   ("Seasonal amp (PLS)",)),
    ("probe_phasePLS_auc", ("Seasonal phase (PLS)",)),
]


def _bacc_key(auc_key):
    """Row key holding the balanced-accuracy twin of a probe-AUC key."""
    return (auc_key[:-4] if auc_key.endswith("_auc") else auc_key) + "_bacc"


def read_prevalence_calibration(d, pid):
    """Prevalence-transported + calibration columns from a run's participant_level block.

    The test cohort is class-balanced by construction (--test-per-class), so pid_f1 / pid_acc
    / pid_mcc in the table above are quoted at an implied 50% base rate and do NOT describe a
    deployment population -- only pid_auc / pid_sens / pid_spec / pid_bacc transfer. These
    columns carry the same model's numbers at the run's target prevalence (the observed cohort
    base rate unless --target-prevalence was given) plus how well the SCORES are calibrated.

    All NaN for runs written before this was recorded, so archived results still collect."""
    nan = float("nan")
    out = {"prevalence": nan, "ppv_at_prev": nan, "npv_at_prev": nan,
           "f1_at_prev": nan, "acc_at_prev": nan, "mcc_at_prev": nan,
           "brier": nan, "ece": nan}
    atp = (pid or {}).get("at_prevalence") or {}
    if atp:
        # the first entry is the run's primary target (cohort base rate by default)
        k = next(iter(atp))
        t = atp[k] or {}
        out.update(prevalence=t.get("prevalence", nan), ppv_at_prev=t.get("ppv", nan),
                   npv_at_prev=t.get("npv", nan), f1_at_prev=t.get("f1", nan),
                   acc_at_prev=t.get("accuracy", nan), mcc_at_prev=t.get("mcc", nan))
    cal = (pid or {}).get("calibration") or {}
    out.update(brier=cal.get("brier", nan), ece=cal.get("ece", nan))
    return out


def read_rhythm(metrics_fp):
    """Rhythm scalars for each variant: the subject-level probe AUC of EVERY representation
    (`_PROBE_VIEWS`) from hrd_rhythm.json, and the Full-representation recovery of the 24 h
    rhythm / trend from the reframed DRS (decomposition_recovery.json). All NaN-safe if a
    file is missing/partial."""
    nan = float("nan")
    out = {"rec_full_rhythm": nan, "rec_full_trend": nan,
           # extra rhythmicity scalars for summary_rhythmicity (branch recovery / leak / DIS
           # from the DRS report, 24 h weight-importance & representation-amplitude from the
           # seasonal-Fourier frequency spectrum)
           "rec_rhythm_branch": nan, "rhy_to_tr_leak": nan, "dis": nan,
           "rec_person_amp": nan, "rec_person_phase": nan, "rec_person_IS": nan, "rec_person_IV": nan, "rec_person_RA": nan,
           "wt_imp_24h": nan, "rep_amp_24h": nan}
    out.update({col: nan for col, _ in _PROBE_VIEWS})
    out.update({_bacc_key(col): nan for col, _ in _PROBE_VIEWS})
    rj = metrics_fp.parent / "hrd_rhythm.json"
    if rj.exists():
        try:
            sep = json.loads(rj.read_text(encoding="utf-8")).get("separability", {})
            for col, names in _PROBE_VIEWS:
                for k in names:                      # first present name wins (NaN-safe)
                    d = sep.get(k, {})
                    v = d.get("subj_auc", nan)
                    if v == v:                       # not NaN
                        out[col] = v
                        # balanced accuracy at the probe's val-tuned threshold. Unlike AUC it
                        # depends on that threshold, so it degenerates to exactly 0.500 when the
                        # probe puts every subject on one side -- read the pair, not BAcc alone.
                        out[_bacc_key(col)] = d.get("subj_bacc", nan)
                        break
        except (json.JSONDecodeError, OSError):
            pass
    dj = metrics_fp.parent / "decomposition_recovery.json"
    if dj.exists():
        try:
            d = json.loads(dj.read_text(encoding="utf-8"))
            # reframed DRS (Full recovery); fall back to the old per-branch DRS_S/DRS_T
            # for runs that predate it, so this figure also works on older runs
            out["rec_full_rhythm"] = d.get("rec_full_rhythm", d.get("DRS_S", nan))
            out["rec_full_trend"] = d.get("rec_full_trend", d.get("DRS_T", nan))
            out["rec_rhythm_branch"] = d.get("rec_rhythm_branch", d.get("DRS_S", nan))
            out["rhy_to_tr_leak"] = d.get("leak_into_rhythm", nan)   # rhythm leaking into trend
            out["dis"] = d.get("DIS", nan)
            out["rec_person_amp"] = d.get("rec_person_amp", nan)
            out["rec_person_phase"] = d.get("rec_person_phase", nan)
            out["rec_person_IS"] = d.get("rec_person_IS", nan)
            out["rec_person_IV"] = d.get("rec_person_IV", nan)
            out["rec_person_RA"] = d.get("rec_person_RA", nan)
        except (json.JSONDecodeError, OSError):
            pass
    fj = metrics_fp.parent / "frequency_spectrum.json"
    if fj.exists():
        try:
            circ = (json.loads(fj.read_text(encoding="utf-8"))
                    .get("key_periods", {}).get("24h_circadian") or {})
            out["wt_imp_24h"] = circ.get("weight_importance", nan)
            out["rep_amp_24h"] = circ.get("repr_amplitude", nan)
        except (json.JSONDecodeError, OSError):
            pass
    return out


# base/reference models live in each variant's hrd_rhythm.json separability table: the
# classical Cosinor clone and the supervised TCN / Transformer baselines. Matched by substring
# so "Cosinor (paper)", "Cosinor (24h)", "TCN (supervised)", ... are all picked up.
BASE_KEYS = ("cosinor", "supervised")


def read_base_models(metrics_fp):
    """[(model_name, {auc,f1,mcc,bacc,sensitivity,specificity})] for the base/reference models
    (cosinor + supervised baselines) in this folder's hrd_rhythm.json. Subject-level metrics,
    the same unit as the SSL models' participant_level. Empty if the file is missing/partial."""
    rj = metrics_fp.parent / "hrd_rhythm.json"
    if not rj.exists():
        return []
    try:
        sep = json.loads(rj.read_text(encoding="utf-8")).get("separability", {})
    except (json.JSONDecodeError, OSError):
        return []
    nan = float("nan")
    out = []
    for name, m in sep.items():
        if not any(k in name.lower() for k in BASE_KEYS):
            continue
        out.append((name, {
            "auc":  m.get("subj_auc", nan),
            "f1":   m.get("subj_f1", nan),
            "mcc":  m.get("subj_mcc", nan),
            "bacc": m.get("subj_bacc", nan),
            "sensitivity": m.get("subj_sensitivity", nan),
            "specificity": m.get("subj_specificity", nan),
        }))
    return out


def report_missing_cosinor(results_dir, rows):
    """Shout when the paper-cosinor baseline is missing, instead of letting it pass as 'n/a'.

    hrd_rhythm.py deliberately makes this view non-fatal so a dependency problem cannot kill a
    12-hour sweep. The cost is that its absence is invisible in the tables -- runs 66404249 and
    66440129 lost it in 130/130 variants and it only showed as an n/a column. So: count the
    variants without a cosinor AUC, and echo the reason each one recorded in
    paper_cosinor.FAILED.txt. Returns the number of affected variants."""
    missing = [r for r in rows if r.get("cosinor_auc", float("nan")) != r.get("cosinor_auc")]
    if not missing:
        return 0
    reasons = {}
    for fp in Path(results_dir).rglob("paper_cosinor.FAILED.txt"):
        try:                                  # first line holds 'ExcType: message'
            reasons.setdefault(fp.read_text(encoding="utf-8").splitlines()[0], 0)
            reasons[fp.read_text(encoding="utf-8").splitlines()[0]] += 1
        except (OSError, IndexError):
            continue
    print(f"\n!! Cosinor (paper) baseline MISSING in {len(missing)}/{len(rows)} variants "
          f"-- its columns are n/a, not 'no effect'.")
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"   {n:>4}x  {reason}")
    if not reasons:
        print("        (no paper_cosinor.FAILED.txt found -- these runs predate that report)")
    return len(missing)


def rhythm_vs_prediction(rows, out_png):
    """Two-panel scatter: does capturing the rhythm / trend predict depression? ONE point
    per (backbone, pe) variant (repeated seeds averaged, error bar = std over seeds). Left:
    24 h RHYTHM capture = Full-representation recovery R^2 (reframed DRS). Right: TREND
    capture = Full-representation recovery R^2. y = participant-level AUC; the encoder-
    independent cosinor baseline is a horizontal reference. Returns ``{'rhythm': r,
    'trend': r}`` (Pearson r of capture vs AUC over per-variant means)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    cmap = plt.get_cmap("tab20")
    markers = {"transformer": "o", "tcn": "^"}
    cidx = {k: i for i, k in enumerate(            # consistent colour per variant across panels
        sorted({(r_["backbone"], r_["pe"]) for r_ in rows}))}
    cos = np.array([r_["cosinor_auc"] for r_ in rows], dtype=float)
    cos = cos[~np.isnan(cos)]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), sharey=True)
    panels = [("rec_full_rhythm", "24h rhythm capture  (recovery R$^2$)", "rhythm", "Rhythm"),
              ("rec_full_trend", "trend capture  (recovery R$^2$)", "trend", "Trend")]
    out_r = {}
    for ax, (xkey, xlabel, tag, title) in zip(axes, panels):
        groups = {}
        for r_ in rows:
            x, y = r_[xkey], r_["pid_auc"]
            if x != x or y != y:                               # skip NaN
                continue
            groups.setdefault((r_["backbone"], r_["pe"]), []).append((x, y))
        if cos.size:                                           # cosinor reference
            cm = float(cos.mean())
            if cos.size > 1:
                ax.axhspan(cm - cos.std(), cm + cos.std(), color="0.85", zorder=0)
            ax.axhline(cm, color="0.45", ls="--", lw=1.3, zorder=1,
                       label=f"cosinor baseline ({cm:.2f})")
        if len(groups) >= 2:
            names = sorted(groups)
            mx = np.array([np.mean([p[0] for p in groups[k]]) for k in names])
            my = np.array([np.mean([p[1] for p in groups[k]]) for k in names])
            sy = np.array([np.std([p[1] for p in groups[k]]) for k in names])
            ns = [len(groups[k]) for k in names]
            out_r[tag] = float(np.corrcoef(mx, my)[0, 1])
            for k, x, y, s, n in zip(names, mx, my, sy, ns):
                ax.errorbar(x, y, yerr=(s if n > 1 else None),
                            fmt=markers.get(k[0], "o"), ms=8, color=cmap(cidx[k] % 20),
                            ecolor=cmap(cidx[k] % 20), elinewidth=1, capsize=2, zorder=3,
                            label=f"{k[0][:2]}/{k[1]}" + (f" (n={n})" if n > 1 else ""))
            rtxt = f"Pearson r = {out_r[tag]:.2f}"
        else:
            out_r[tag] = None
            rtxt = "Pearson r = n/a"
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_title(f"{title}  ({rtxt})", fontsize=11)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("participant-level AUC", fontsize=10)

    # one shared legend (union of both panels, deduplicated by label)
    hl = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            hl.setdefault(l, h)
    fig.legend(hl.values(), hl.keys(), fontsize=7, loc="center left",
               bbox_to_anchor=(1.0, 0.5), frameon=False,
               title="variant  (○ transformer, △ tcn)")
    fig.suptitle("Rhythm capture vs prediction across variants", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.99, 0.96])
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_r


HEADLINE_VIEW = "Full [V^(T);V^(S)]"             # matches hrd_rhythm.HEADLINE_VIEW
ROC_GRID = np.linspace(0, 1, 101)                # common FPR grid for vertical ROC averaging


def read_probe_scores(results_dir):
    """{(backbone, pe): {seed: {view: {y, prob, auc}}}} from every variant's probe_scores.json.

    These are the subject-level (label, score) pairs the per-variant figure plots for ONE seed;
    pooling them here is what makes the +/-1 sd ROC band possible."""
    out = {}
    for fp in sorted(Path(results_dir).rglob("probe_scores.json")):
        d = read_json(fp)                        # shared partial-file guard (scripts/_results.py)
        if d is None:
            continue
        # `label` is in the key for the same reason `holdout` is: weekly and endpoint are
        # different prediction targets, so their ROC bands must not be pooled. HRD writes
        # "-" for both, so this is a no-op there and only bites on GLOBEM sweeps.
        key = (d.get("backbone", "?"), d.get("pe", "?"),
               d.get("holdout") or "-", d.get("globem_label", "-"))
        out.setdefault(key, {})[str(d.get("seed", "?"))] = d.get("views", {})
    return out


def _mean_roc(samples):
    """(mean_tpr, sd_tpr, [auc per seed]) on ROC_GRID by vertical averaging over seeds.

    Vertical averaging (interpolate TPR onto a shared FPR grid, then average) is the standard
    way to combine ROC curves whose thresholds do not line up -- here the seeds are independent
    runs with different score scales, so per-threshold averaging would be meaningless."""
    from sklearn.metrics import roc_curve, roc_auc_score
    tprs, aucs = [], []
    for y, prob in samples:
        y, prob = np.asarray(y, dtype=int), np.asarray(prob, dtype=float)
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, prob)
        t = np.interp(ROC_GRID, fpr, tpr)
        t[0] = 0.0
        tprs.append(t)
        aucs.append(float(roc_auc_score(y, prob)))
    if not tprs:
        return None, None, []
    tprs = np.vstack(tprs)
    return tprs.mean(axis=0), tprs.std(axis=0), aucs


def probe_score_report(results_dir, out_png, view=HEADLINE_VIEW):
    """Cross-seed honest picture of the endpoint separation -- the run-level companion to each
    variant's hrd_probe_scores.png, and the replacement for the label-coloured t-SNE panels.

    Left   : subject-level probe scores by endpoint for the best variant, POOLED over its seeds
             (one point per participant per seed) -- the overlap IS the finding.
    Middle : that variant's ROC, mean over seeds with a +/-1 sd band (vertical averaging),
             against the chance diagonal.
    Right  : mean +/- sd AUC over seeds for EVERY model, sorted, with the 0.5 chance line --
             the cross-variant answer to 'which representation separates the groups?'.

    Returns the {(backbone, pe): (mean_auc, sd, n_seeds)} it plotted, or None if unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    scores = read_probe_scores(results_dir)
    if not scores:
        return None

    # per-variant seed samples for `view` (skip variants whose runs predate probe_scores.json)
    per_variant = {}
    for key, seeds in scores.items():
        s = [(v[view]["y"], v[view]["prob"]) for v in seeds.values() if view in v]
        if s:
            per_variant[key] = s
    if not per_variant:
        return None

    summary = {}
    for key, s in per_variant.items():
        _, _, aucs = _mean_roc(s)
        if aucs:
            summary[key] = (float(np.mean(aucs)), float(np.std(aucs)), len(aucs))
    if not summary:
        return None
    best = max(summary, key=lambda k: summary[k][0])

    fig, (ax_d, ax_r, ax_f) = plt.subplots(1, 3, figsize=(16.5, 4.9))
    class_colors = ["#0072B2", "#D55E00"]                    # 0 non-depressed, 1 depressed
    class_names = {0: "non-depressed (0)", 1: "depressed (1)"}

    # ---- left: pooled score distribution of the best variant ------------------
    y = np.concatenate([np.asarray(a, dtype=int) for a, _ in per_variant[best]])
    prob = np.concatenate([np.asarray(b, dtype=float) for _, b in per_variant[best]])
    bins = np.linspace(prob.min(), prob.max(), 25) if prob.max() > prob.min() else 10
    for c in np.unique(y):
        m = y == c
        col = class_colors[int(c) % len(class_colors)]
        ax_d.hist(prob[m], bins=bins, histtype="stepfilled", alpha=0.45, color=col,
                  label=f"{class_names.get(int(c), c)}  (n={int(m.sum())})")
        ax_d.hist(prob[m], bins=bins, histtype="step", lw=1.6, color=col)
        ax_d.axvline(prob[m].mean(), color=col, ls="--", lw=1.2)
    gap = prob[y == 1].mean() - prob[y == 0].mean()
    ax_d.set_title(f"probe score by endpoint  --  {best[0]}/{best[1]}\n"
                   f"{summary[best][2]} seeds pooled, {view}", fontsize=10)
    ax_d.set_xlabel("predicted P(depressed)", fontsize=9)
    ax_d.set_ylabel("participant-seed pairs", fontsize=9)
    ax_d.annotate(f"mean gap = {gap:+.3f}", xy=(0.02, 0.97), xycoords="axes fraction",
                  va="top", ha="left", fontsize=9,
                  bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.9))
    ax_d.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax_d.tick_params(labelsize=8); ax_d.grid(alpha=0.2)

    # ---- middle: mean ROC +/- 1 sd over seeds ---------------------------------
    mtpr, stpr, aucs = _mean_roc(per_variant[best])
    ax_r.plot([0, 1], [0, 1], color="0.6", ls="--", lw=1, label="chance (AUC 0.500)")
    ax_r.fill_between(ROC_GRID, np.clip(mtpr - stpr, 0, 1), np.clip(mtpr + stpr, 0, 1),
                      color=class_colors[1], alpha=0.20, lw=0, label="+/-1 sd over seeds")
    ax_r.plot(ROC_GRID, mtpr, color=class_colors[1], lw=2,
              label=f"mean ROC (AUC {np.mean(aucs):.3f} +/- {np.std(aucs):.3f})")
    from sklearn.metrics import roc_curve                    # lazy: keeps sklearn optional
    for (ys, ps) in per_variant[best]:                       # the individual seeds, faint
        if len(np.unique(np.asarray(ys, dtype=int))) < 2:
            continue
        f_, t_, _ = roc_curve(np.asarray(ys, dtype=int), np.asarray(ps, dtype=float))
        ax_r.plot(f_, t_, color=class_colors[1], lw=0.8, alpha=0.35)
    ax_r.set_xlim(0, 1); ax_r.set_ylim(0, 1)
    ax_r.set_title(f"ROC over {len(aucs)} seeds  --  {best[0]}/{best[1]}", fontsize=10)
    ax_r.set_xlabel("false-positive rate  (1 - specificity)", fontsize=9)
    ax_r.set_ylabel("true-positive rate  (sensitivity)", fontsize=9)
    ax_r.legend(loc="lower right", fontsize=8, framealpha=0.85)
    ax_r.tick_params(labelsize=8); ax_r.grid(alpha=0.2)

    # ---- right: every model's AUC, mean +/- sd over seeds ---------------------
    order = sorted(summary, key=lambda k: summary[k][0])
    ypos = np.arange(len(order))
    means = np.array([summary[k][0] for k in order])
    sds = np.array([summary[k][1] for k in order])
    ax_f.axvline(0.5, color="0.45", ls="--", lw=1.3, label="chance (0.5)")
    ax_f.errorbar(means, ypos, xerr=sds, fmt="o", ms=5, color="#0072B2",
                  ecolor="#0072B2", elinewidth=1.2, capsize=3, lw=0)
    ax_f.set_yticks(ypos)
    ax_f.set_yticklabels([f"{k[0][:2]}/{k[1]}  (n={summary[k][2]})" for k in order], fontsize=8)
    ax_f.set_xlabel("subject-level AUC  (mean +/- sd over seeds)", fontsize=9)
    ax_f.set_title(f"every variant, {view}", fontsize=10)
    ax_f.legend(loc="lower right", fontsize=8, framealpha=0.85)
    ax_f.tick_params(labelsize=8); ax_f.grid(alpha=0.2, axis="x")

    fig.suptitle("Depression-endpoint separation on the probe's decision axis "
                 "(held-out HRD test set)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return summary


def print_aggregated(rows):
    """Group per-run rows by (backbone, pe) and print participant-level mean +/- std across
    seeds for the headline metrics (AUC, MCC, balanced acc, F1). One line per variant;
    n = number of seeds. This is the multi-seed summary -- std is the run-to-run spread that
    a single-seed number hides. Sorted by mean AUC. std is population std (ddof=0), matching
    the error bars in rhythm_vs_prediction."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[(r["backbone"], r["pe"], r["holdout"], r["label"])].append(r)

    def ms(vals):                                            # NaN-safe (mean, std)
        a = np.array([v for v in vals if v == v], dtype=float)
        return (float(a.mean()), float(a.std())) if a.size else (float("nan"), float("nan"))

    agg = []
    for (bb, pe, hd, lb), rs in groups.items():
        agg.append({
            "backbone": bb, "pe": pe, "holdout": hd, "label": lb, "n": len(rs),
            "seeds": sorted(str(r["seed"]) for r in rs),
            "auc":  ms([r["pid_auc"]  for r in rs]),
            "cv":   ms([r["cv_auc"]   for r in rs]),
            "mcc":  ms([r["pid_mcc"]  for r in rs]),
            "bacc": ms([r["pid_bacc"] for r in rs]),
            "f1":   ms([r["pid_f1"]   for r in rs]),
        })
    agg.sort(key=lambda a: (a.get("label", "-"), a.get("holdout", "-"),
                            -(a["auc"][0] if a["auc"][0] == a["auc"][0] else -1.0)))

    cell = lambda t: (f"{t[0]:.3f}+/-{t[1]:.3f}" if t[0] == t[0] else "n/a")
    # cv_AUC = internal k-fold CV within the probe pool (test untouched); n/a when --cv-folds<2
    header = (f"{'backbone':<12} {'pe':<11} {'n':>2}  "
              f"{'pid_AUC':^15} {'cv_AUC':^15} {'pid_MCC':^15} {'pid_BAcc':^15} {'pid_F1':^15}")
    print("\n=== Aggregated over seeds (participant-level, mean +/- std) ===")
    print("  pid_* = held-out 36-participant test | cv_AUC = internal k-fold CV within probe pool")
    print(header)
    print("-" * len(header))
    for a in agg:
        print(f"{a['backbone']:<12} {a['pe']:<11} {a['n']:>2}  "
              f"{cell(a['auc']):^15} {cell(a['cv']):^15} {cell(a['mcc']):^15} "
              f"{cell(a['bacc']):^15} {cell(a['f1']):^15}")
    return agg


_SUMMARY_METRICS = ["auc", "sensitivity", "specificity", "f1", "mcc", "bacc"]

# Representation-quality scalars added to the model summary next to the prediction metrics:
# V^T trend-recovery R^2 and V^S rhythm-recovery R^2, averaged over seeds. (display name,
# per-row key from read_rhythm.) Base models (cosinor, supervised) have no such
# representation, so these cells are n/a there.
_REPR_METRICS = [
    ("vt_recovery", "rec_full_trend"),
    ("vs_recovery", "rec_full_rhythm"),
]

# Subject-level probe AUC of each representation the encoder produces, averaged over seeds.
# Printed as a second stdout block (the combined table is too wide for a terminal) and written
# to summary_models.csv alongside everything else. These share one probe and one split, so the
# columns rank the representations AGAINST EACH OTHER; the `auc` column is a different
# estimator (k-fold, metrics.json) and is not comparable cell-for-cell with them.
# Cosinor is omitted here -- it is identical across a seed's variants and has its own row.
# Split into two blocks that MIRROR each other: native width, then the same views at the common
# PCA budget. Only the second block is a like-for-like comparison across representations -- in the
# first, a view's width is free to explain its ranking. Two blocks also keep each line inside a
# terminal, which one 12-column block would not.
_PROBE_AUC_METRICS = [
    ("full_AUC",     "probe_full_auc"),
    ("V_AUC",        "probe_V_auc"),
    ("VT_AUC",       "probe_VT_auc"),
    ("VS_AUC",       "probe_VS_auc"),
    ("amp_AUC",      "amp_auc"),
    ("phase_AUC",    "phase_auc"),
    ("reim_AUC",     "reim_auc"),
]
_PROBE_AUC_PCA_METRICS = [
    ("fullPCA_AUC",  "probe_fullPCA_auc"),
    ("VPCA_AUC",     "probe_VPCA_auc"),
    ("VTPCA_AUC",    "probe_VTPCA_auc"),
    ("VSPCA_AUC",    "probe_VSPCA_auc"),
    ("ampPCA_AUC",   "probe_ampPCA_auc"),
    ("phasePCA_AUC", "probe_phasePCA_auc"),
    ("reimPCA_AUC",  "probe_reimPCA_auc"),
    ("blockPCA_AUC", "probe_blockPCA_auc"),
    ("V_ampPCA_AUC", "probe_VampPCA_auc"),
    ("VT_ampPCA_AUC", "probe_VTampPCA_auc"),
    ("ampPLS_AUC",   "probe_ampPLS_auc"),
    ("phasePLS_AUC", "probe_phasePLS_auc"),
]


# Balanced-accuracy twin of every probe-AUC column, derived so the two lists cannot drift.
# CSV only: the stdout blocks are already at terminal width.
_PROBE_BACC_METRICS = [(c.replace("_AUC", "_BAcc"), _bacc_key(k))
                       for c, k in _PROBE_AUC_METRICS + _PROBE_AUC_PCA_METRICS]


def _mean_sd(vals):                                          # NaN-safe (mean, std=ddof0)
    a = np.array([v for v in vals if v == v], dtype=float)
    return (float(a.mean()), float(a.std())) if a.size else (float("nan"), float("nan"))


def aggregate_all_models(rows, base_samples):
    """Unified mean+/-sd table over ALL models -- base (cosinor, supervised baselines) AND
    non-base (the SSL variants) -- for auc / sensitivity / specificity / f1 / mcc / bacc.

    SSL variants aggregate their held-out participant_level metrics over seeds (grouped by
    backbone/pe). Base models aggregate over seeds too, deduplicated by seed: each base model is
    recomputed inside every variant folder, so we keep one value per seed to avoid counting the
    same estimate ~17x (which would collapse the sd). Returns a list sorted by mean AUC."""
    from collections import defaultdict
    # non-base: SSL variants from metrics.json participant_level
    ssl = defaultdict(list)
    for r in rows:
        ssl[(r["backbone"], r["pe"], r["holdout"], r["label"])].append(r)
    keymap = {"auc": "pid_auc", "sensitivity": "pid_sens", "specificity": "pid_spec",
              "f1": "pid_f1", "mcc": "pid_mcc", "bacc": "pid_bacc"}
    agg = []
    for (bb, pe, hd, lb), rs in ssl.items():
        rec = {"model": f"{bb}/{pe}", "holdout": hd, "label": lb, "type": "ssl", "n": len(rs)}
        for m in _SUMMARY_METRICS:
            rec[m] = _mean_sd([r[keymap[m]] for r in rs])
        for col, key in (_REPR_METRICS + _PROBE_AUC_METRICS + _PROBE_AUC_PCA_METRICS
                         + _PROBE_BACC_METRICS):   # recovery R^2 + per-view AUC & BAcc
            rec[col] = _mean_sd([r.get(key, float("nan")) for r in rs])
        agg.append(rec)
    # base: one value per seed (dedup), then mean+/-sd over seeds
    for (name, hd, lb), per_seed in base_samples.items():
        vals = list(per_seed.values())
        rec = {"model": name, "holdout": hd, "label": lb, "type": "base", "n": len(vals)}
        for m in _SUMMARY_METRICS:
            rec[m] = _mean_sd([v[m] for v in vals])
        for col, _ in (_REPR_METRICS + _PROBE_AUC_METRICS + _PROBE_AUC_PCA_METRICS
                      + _PROBE_BACC_METRICS):      # base models have no encoder rep
            rec[col] = (float("nan"), float("nan"))
        agg.append(rec)
    agg.sort(key=lambda a: (a.get("label", "-"), a.get("holdout", "-"),
                            -(a["auc"][0] if a["auc"][0] == a["auc"][0] else -1.0)))
    return agg


def write_summary_table(agg, csv_path):
    """Write/print the unified mean+/-sd table. Each metric cell is 'mean+/-sd' (matching the
    aggregated stdout table's format), one row per model.

    stdout gets THREE blocks -- prediction metrics + recovery, then the per-representation probe
    AUCs at native width, then the same views at the matched PCA budget -- because all columns on
    one line do not fit a terminal. The CSV keeps them on a single row per model."""
    cell = lambda t: (f"{t[0]:.3f}+/-{t[1]:.3f}" if t[0] == t[0] else "n/a")
    repr_names = [c for c, _ in _REPR_METRICS]
    probe_names = [c for c, _ in _PROBE_AUC_METRICS]
    probe_pca_names = [c for c, _ in _PROBE_AUC_PCA_METRICS]
    all_cols = _SUMMARY_METRICS + repr_names + probe_names + probe_pca_names
    csv_cols = all_cols + [c for c, _ in _PROBE_BACC_METRICS]   # BAcc: CSV only

    def block(title, note, cols, width):
        print(f"\n=== {title} ===")
        print(f"  {note}")
        hdr = f"{'model':<30}{'type':<6}{'n':>3}  " + "".join(f"{c:^{width}}" for c in cols)
        print(hdr); print("-" * len(hdr))
        for a in agg:
            print(f"{a['model']:<30}{a['type']:<6}{a['n']:>3}  "
                  + "".join(f"{cell(a[c]):^{width}}" for c in cols))

    block("All models -- mean +/- sd over seeds (base + SSL, participant/subject level)",
          "vt/vs_recovery = V^T/V^S recovery R^2",
          _SUMMARY_METRICS + repr_names, 19)
    block("Probe AUC per representation, NATIVE width -- mean +/- sd over seeds (subject level, "
          "single split)",
          "one logistic probe, one split; the 'auc' column above is the k-fold headline and is a "
          "different estimator. Widths DIFFER per view (240/240/120/120/Ffreq*dS) -- for the "
          "like-for-like ranking read the matched-PCA block below",
          probe_names, 17)
    block("Probe AUC per representation, matched PCA -- mean +/- sd over seeds (subject level, "
          "single split)",
          "same views, same probe, same split, all reduced to ONE common component budget, PCA "
          "fit on train only -> these columns rank the representations against each other with "
          "dimensionality held constant (n/a for runs predating the matched block)",
          probe_pca_names, 17)

    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model", "holdout", "label", "type", "n_seeds"] + csv_cols)
            for a in agg:
                w.writerow([a["model"], a.get("holdout", "-"), a.get("label", "-"),
                            a["type"], a["n"]] + [cell(a[m]) for m in csv_cols])
        print(f"Wrote {csv_path}")


# rhythmicity table columns: (display name, row key). Mirrors the paper's rhythmicity table --
# rhythm capture (Full + seasonal branch), rhythm->trend leak, disentanglement DIS, the 24 h
# circadian weight-importance & representation-amplitude, and the seasonal amp/phase AUCs.
_RHY_METRICS = [
    ("rhythm_cap_full", "rec_full_rhythm"),
    # between-PERSON rhythm, per descriptor: the quantity the depression hypothesis
    # is about. rhythm_cap_full measures the shared population waveform instead.
    ("person_amp",      "rec_person_amp"),
    ("person_phase",    "rec_person_phase"),
    ("person_IS",       "rec_person_IS"),
    ("person_IV",       "rec_person_IV"),
    ("person_RA",       "rec_person_RA"),
    ("rhythm_cap_S",    "rec_rhythm_branch"),
    ("rhy_to_tr_leak",  "rhy_to_tr_leak"),
    ("DIS",             "dis"),
    ("wt_imp_24h",      "wt_imp_24h"),
    ("rep_amp_24h",     "rep_amp_24h"),
    ("amp_AUC",         "amp_auc"),
    ("phase_AUC",       "phase_auc"),
]


def _enc_kind(pe):
    """Encoder/positional-code family shown in the 'enc' column: None (TCN, no PE),
    Time (Time2Vec time encoding) or Pos (a positional encoding)."""
    p = str(pe).lower().replace("_plain", "")
    if p == "none":
        return "None"
    if "time2vec" in p:
        return "Time"
    return "Pos"


def aggregate_rhythmicity(rows):
    """Per-(backbone, pe) mean+/-sd rhythmicity table. Variants with no rhythm outputs at all
    (plain-SSL runs, or failed rhythm analysis) are skipped so the table stays to the disentangled
    CoST variants -- like the paper's rhythmicity table. Sorted by 24 h rhythm capture (Full)."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[(r["backbone"], r["pe"], r["holdout"], r["label"])].append(r)
    agg = []
    for (bb, pe, hd, lb), rs in groups.items():
        rec = {"model": f"{bb}/{pe}", "holdout": hd, "label": lb, "enc": _enc_kind(pe), "n": len(rs)}
        any_data = False
        for name, key in _RHY_METRICS:
            rec[name] = _mean_sd([r.get(key, float("nan")) for r in rs])
            any_data = any_data or (rec[name][0] == rec[name][0])
        if any_data:
            agg.append(rec)
    agg.sort(key=lambda a: a["rhythm_cap_full"][0] if a["rhythm_cap_full"][0] == a["rhythm_cap_full"][0]
             else -1.0, reverse=True)
    return agg


def write_rhythmicity_table(agg, csv_path):
    """Write/print the rhythmicity mean+/-sd table (one row per variant)."""
    cell = lambda t: (f"{t[0]:.3f}+/-{t[1]:.3f}" if t[0] == t[0] else "n/a")
    names = [n for n, _ in _RHY_METRICS]
    print("\n=== Rhythmicity per variant -- mean +/- sd over seeds ===")
    hdr = f"{'model':<24}{'enc':<6}{'n':>3}  " + "".join(f"{n:^17}" for n in names)
    print(hdr); print("-" * len(hdr))
    for a in agg:
        print(f"{a['model']:<24}{a['enc']:<6}{a['n']:>3}  "
              + "".join(f"{cell(a[n]):^17}" for n in names))
    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model", "holdout", "label", "enc", "n_seeds"] + names)
            for a in agg:
                w.writerow([a["model"], a.get("holdout", "-"), a.get("label", "-"),
                            a["enc"], a["n"]] + [cell(a[n]) for n in names])
        print(f"Wrote {csv_path}")


def main():
    p = argparse.ArgumentParser(description="Summarise CoST PE-variant results")
    p.add_argument("--results-dir", default="results_hrd")
    p.add_argument("--csv", default=None, help="Optional path to write the table as CSV")
    args = p.parse_args()

    rows = []
    base_samples = {}                       # model_name -> {seed -> metrics}, deduped by seed
    # One shared reader (scripts/_results.py) derives the variant identity for BOTH this
    # script and results_figures.py, so the two can never disagree about what a run is.
    for fp, d in iter_metrics(args.results_dir):
        k = variant_key(d, fp)
        win, pid = d.get("window_level", {}), d.get("participant_level", {})
        seed = k.seed
        for bname, bmetrics in read_base_models(fp):        # cosinor + supervised baselines
            # key on the fold too: several folds share a seed, and a (name, seed)
            # key would let setdefault keep only whichever fold was read first.
            base_samples.setdefault((bname, k.holdout, k.label), {}).setdefault(str(seed), bmetrics)
        row = {
            "run_id": k.run,
            "backbone": k.backbone,
            # `pe` carries a "_plain" suffix for the no-disentangler baseline (see variant_key)
            "pe": k.pe,
            # cross-dataset fold (--holdout): DS1..DS4 | pre | post, or "-" for the default
            # random split. Runs from different folds test DIFFERENT populations, so every
            # aggregation below keys on it -- averaging across folds would be meaningless.
            "holdout": k.holdout,
            # weekly (state) vs endpoint (trait) are DIFFERENT prediction targets
            "label": k.label,
            "seed": seed,
            # split/model seeds fall back to `seed` for runs written before they were split
            "split_seed": d.get("config", {}).get("split_seed", seed),
            "model_seed": d.get("config", {}).get("model_seed", seed),
            # The unit of the win_* columns. Under --globem-label weekly a "window" IS a week,
            # so these rows are week-level, NOT participant-level -- pid_* is the participant
            # collapse (mean probability vs the participant's majority weekly label).
            "eval_unit": d.get("eval_unit", "window"),
            "win_auc": win.get("auc_roc", float("nan")),
            "win_f1": win.get("f1", float("nan")),
            "win_acc": win.get("accuracy", float("nan")),
            "pid_auc": pid.get("auc_roc", float("nan")),
            "pid_f1": pid.get("f1", float("nan")),
            "pid_acc": pid.get("accuracy", float("nan")),
            "pid_bacc": pid.get("balanced_accuracy", float("nan")),
            "pid_mcc": pid.get("mcc", float("nan")),
            "pid_sens": pid.get("sensitivity", float("nan")),
            "pid_spec": pid.get("specificity", float("nan")),
            # internal k-fold CV within the probe pool (test untouched); nan when --cv-folds<2
            "cv_auc": (d.get("cv_internal") or {}).get("auc_roc", float("nan")),
            "cv_bacc": (d.get("cv_internal") or {}).get("balanced_accuracy", float("nan")),
            "win_bacc": win.get("balanced_accuracy", float("nan")),
            "win_mcc": win.get("mcc", float("nan")),
            "time_s": d.get("pretrain_seconds", float("nan")),
            "n_params": d.get("n_params", float("nan")),
            "n_params_trainable": d.get("n_params_trainable", float("nan")),
        }
        row.update(read_prevalence_calibration(d, pid))
        row.update(read_rhythm(fp))
        rows.append(row)

    if not rows:
        raise SystemExit(f"No metrics*.json found in {args.results_dir}")

    report_missing_cosinor(args.results_dir, rows)

    # group each variant's seeds adjacently (the per-variant ranking lives in the
    # aggregated table below, so the detailed table is organised for readability)
    rows.sort(key=lambda r: (r["label"], r["holdout"], r["backbone"], r["pe"], str(r["seed"])))

    def _secs(v):
        return f"{v:>8.0f}" if v == v else f"{'n/a':>8}"      # NaN-safe

    def _params(v):
        return f"{v / 1e6:>8.2f}M" if v == v else f"{'n/a':>9}"   # report in millions

    def _f3(v):
        return f"{v:>8.3f}" if v == v else f"{'n/a':>8}"      # NaN-safe 3-decimal

    print("=== Per-run detail (one row per seed) ===")
    header = (f"{'backbone':<12} {'pe':<11} {'seed':>5} {'pid_AUC':>8} {'cv_AUC':>8} {'pid_MCC':>8} "
              f"{'pid_F1':>7} {'win_AUC':>8} {'rhythmCap':>9} {'cosinor':>8} {'time_s':>8} {'params':>9}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['backbone']:<12} {r['pe']:<11} {str(r['seed']):>5} {r['pid_auc']:>8.3f} "
              f"{_f3(r['cv_auc'])} {r['pid_mcc']:>8.3f} {r['pid_f1']:>7.3f} {r['win_auc']:>8.3f} "
              f"{r['rec_full_rhythm']:>9.3f} {r['cosinor_auc']:>8.3f} "
              f"{_secs(r['time_s'])} {_params(r['n_params'])}")

    print_aggregated(rows)

    # unified mean+/-sd table over ALL models (base + SSL), all 6 requested metrics.
    # Written next to --csv as summary_models.csv (companion to the per-seed summary.csv).
    agg_all = aggregate_all_models(rows, base_samples)
    agg_csv = str(Path(args.csv).with_name("summary_models.csv")) if args.csv else None
    write_summary_table(agg_all, agg_csv)

    # rhythmicity table (rhythm capture / leak / DIS / 24h spectrum / seasonal AUCs),
    # written next to --csv as summary_rhythmicity.csv
    rhy = aggregate_rhythmicity(rows)
    rhy_csv = str(Path(args.csv).with_name("summary_rhythmicity.csv")) if args.csv else None
    write_rhythmicity_table(rhy, rhy_csv)

    # endpoint separation on the probe's decision axis, pooled over seeds (score
    # distribution + mean ROC with a +/-1 sd band + per-variant AUC forest)
    score_png = Path(args.results_dir) / "probe_score_report.png"
    ps = probe_score_report(args.results_dir, score_png)
    if ps:
        best = max(ps, key=lambda k: ps[k][0])
        print(f"\nBest subject-level AUC ({HEADLINE_VIEW}): {best[0]}/{best[1]} = "
              f"{ps[best][0]:.3f} +/- {ps[best][1]:.3f} over {ps[best][2]} seeds")
        print(f"Wrote {score_png}")
    else:
        print("\n[probe scores] no probe_scores.json found -- re-run the variants to "
              "generate probe_score_report.png")

    # cross-variant rhythm <-> prediction link
    out_png = Path(args.results_dir) / "rhythm_vs_prediction.png"
    rr = rhythm_vs_prediction(rows, out_png)
    if rr is not None:
        fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else "n/a"
        print(f"\nPearson r vs participant AUC  ->  rhythm capture: {fmt(rr.get('rhythm'))}   "
              f"trend capture: {fmt(rr.get('trend'))}")
        print(f"Wrote {out_png}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
