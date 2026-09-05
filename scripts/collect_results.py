"""Collect every variant's results into one comparison table, and link
rhythmicity to prediction across variants.

For each results_hrd/<run>/<variant>/ it reads:
  * metrics.json      - downstream depression classification (AUC / F1 / Acc)
  * hrd_rhythm.json   - rhythm metrics (FFT-alignment, per-representation probe
                        AUCs incl. the classical cosinor baseline), if present.

It prints/writes a table sorted by participant-level AUC and, when rhythm metrics
are available, saves a cross-variant scatter of rhythm capture (how well the
latent recovers the true circadian amplitude, R^2) vs participant AUC with its
Pearson correlation -> the central claim "do rhythm-capturing encodings predict
better?".

It also renders the two cross-variant panels (written to --fig-dir):
    figC_pe_dissociation.png   positional encoding: prediction vs disentanglement
    figD_clock_ablation.png    paired calendar-channel on/off (needs --clock-on/--clock-off)

Run:  python scripts/collect_results.py --results-dir results_hrd [--csv summary.csv]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, for `tasks`
from tasks.rq_paths import rq_path                                   # noqa: E402
from _results import (census, iter_metrics, load_variants,   # noqa: E402
                      read_json, variant_key)                # shared results reader


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
    rj = rq_path(metrics_fp.parent, "hrd_rhythm.json", create=False)
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
    dj = rq_path(metrics_fp.parent, "decomposition_recovery.json", create=False)
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
    fj = rq_path(metrics_fp.parent, "frequency_spectrum.json", create=False)
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
    rj = rq_path(metrics_fp.parent, "hrd_rhythm.json", create=False)
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


def _mean_sd(vals):
    """NaN-safe (mean, SAMPLE sd).

    ddof=1, not the numpy default of 0. These are the S seeds of a sweep -- a sample drawn to
    estimate run-to-run variability, not the whole population -- so the sample sd is the right
    estimator and the paper reports "mean +/- sd over seeds". At S=6 the population form is
    sqrt(5/6) = 0.91 of it, i.e. every interval in the summary table read ~9% tighter than it
    should. One value returns nan rather than 0: an sd of 0 from a single seed would claim
    perfect reproducibility.
    """
    a = np.array([v for v in vals if v == v], dtype=float)
    if a.size == 0:
        return (float("nan"), float("nan"))
    return (float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else float("nan"))


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


def _holm(pvals):
    """Holm-Bonferroni adjusted p-values, input order preserved. Hand-rolled because
    statsmodels is not in requirements.txt and would import fine here but fail on the cluster."""
    p = np.asarray(pvals, dtype=float)
    n, o = len(p), np.argsort(pvals)
    adj = np.empty(n)
    adj[o] = np.minimum(1.0, np.maximum.accumulate(p[o] * (n - np.arange(n))))
    return adj


def pe_contrast(results_dir, ref=("tcn", "none"), metric="sigma", n_boot=2000, seed=0):
    """E1.4 -- Delta_pi = R2_pi - R2_F0 on the RQ1 headline, with a real interval.

    Replaces the design's "paired Wilcoxon over the S seeds", which was never implemented and
    could not have worked: at S=6 the two-sided signed-rank test bottoms out at p=0.031, so
    after Holm over ~5 families nothing reaches 0.05 however large the effect.

    Two bootstrap stages instead, on PAIRED draws. Participants, because the claim is about a
    new person; and seeds with replacement, which carries run-to-run variance as a random
    effect rather than averaging it away (numpy, not a mixed model -- see _holm). Every variant
    is scored on the SAME drawn seeds and people, so Delta is paired and its interval can
    exclude zero even when the marginal ones overlap.

    Exact and cheap because experiment_q1.py stores per-participant sufficient statistics
    (tasks/decomposition.py::_probe_r2): a resample is a reweighting, not a refit. Variants of
    one seed share a test split, so a pid missing from a variant raises rather than quietly
    producing an unpaired contrast.
    """
    by = {}                                     # (backbone, pe) -> {seed: sufficient stats}
    for fp, d in iter_metrics(results_dir):
        side = rq_path(fp.parent, "rq1.json", create=False)
        j = read_json(side) if side.exists() else None
        if j and j.get("bootstrap"):
            k = variant_key(d, fp)
            by.setdefault((k.backbone, k.pe), {})[str(k.seed)] = j["bootstrap"]
    others = sorted(k for k in by if k != ref)
    if ref not in by or not others:
        print("\n[pe contrast] needs rq1/rq1.json with a 'bootstrap' block for "
              f"{'/'.join(ref)} and at least one other variant -- re-run experiment_q1.py")
        return None

    def r2(blk, pids):
        """Variance-weighted R2 over a participant list, from sufficient statistics:
        SStot = sum(syy) - sum(sy)^2/sum(n), identical to recomputing it from the rows."""
        ss = blk["per_participant"][metric]
        n = sum(ss[p]["n"] for p in pids)
        sy, syy, res = (sum(np.asarray(ss[p][f]) for p in pids)
                        for f in ("sy", "syy", "ssres"))
        tot = syy - sy ** 2 / max(n, 1)
        ok = tot > 0
        return float(np.asarray(blk["weights"][metric])
                     @ np.clip(np.where(ok, 1.0 - res / np.where(ok, tot, 1.0), 0.0), 0.0, 1.0))

    seeds = sorted(set(by[ref]) & {s for k in others for s in by[k]})
    pool = {s: sorted(by[ref][s]["per_participant"][metric]) for s in seeds}
    delta = lambda k, s, pids: r2(by[k][s], pids) - r2(by[ref][s], pids)

    rng = np.random.default_rng(seed)
    boot = {k: [] for k in others}
    for _ in range(n_boot):
        drawn = [(s, [pool[s][i] for i in rng.integers(0, len(pool[s]), len(pool[s]))])
                 for s in (seeds[i] for i in rng.integers(0, len(seeds), len(seeds)))]
        for k in others:                        # same seeds, same people, every variant
            v = [delta(k, s, pids) for s, pids in drawn if s in by[k]]
            if v:
                boot[k].append(float(np.mean(v)))

    rows = []
    for k in others:
        b = np.asarray(boot[k])
        rows.append({"variant": "/".join(k),
                     "delta": float(np.mean([delta(k, s, pool[s]) for s in seeds if s in by[k]])),
                     "ci_lo": float(np.percentile(b, 2.5)), "ci_hi": float(np.percentile(b, 97.5)),
                     # two-sided bootstrap p, floored at 1/B so it is never reported as 0
                     "p": float(max(2 * min((b <= 0).mean(), (b >= 0).mean()), 1.0 / len(b)))})
    for r, q in zip(rows, _holm([r["p"] for r in rows])):
        r["p_holm"] = float(q)
    rows.sort(key=lambda r: -r["delta"])

    print(f"\n=== Delta_pi on Full->{metric} vs {'/'.join(ref)} "
          f"({len(seeds)} seeds, {n_boot} paired seed x participant draws) ===")
    print(f"{'variant':<26}{'delta':>9}{'95% CI':>21}{'p':>9}{'p_holm':>9}")
    for r in rows:
        print(f"{r['variant']:<26}{r['delta']:>+9.4f}   [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]"
              f"{r['p']:>9.4f}{r['p_holm']:>9.4f}")
    print(f"survives Holm at 0.05: "
          f"{[r['variant'] for r in rows if r['p_holm'] < 0.05] or 'none'}")
    return rows


def circadian_landscape(results_dir, out_png):
    """Cross-variant view of the circadian analysis. BEYOND the paper: WavesFM compares two
    models on one figure and never varies a seed, so it needs no such summary.

    One point per (variant, seed, stage), the stages being the backbone output and the two
    branches the disentangler splits it into -- so a collapse can be attributed to the backbone
    or to the disentangler instead of merely observed.

    A representation is only useful to RQ1-RQ3 when it
    moves with the clock AND still separates people at the same hour, so the two axes are
    exactly those two shares of variance, and the reader wants the TOP-RIGHT corner:

        y low  -> the embedding is a clock and nothing else (or is collapsed outright);
        x low  -> the embedding ignores time of day;
        both   -> useful.

    The right panel is the same runs as a strip of diurnal swing, because that is the number
    the per-run figures print and it is the one that varies wildly between seeds -- seeing the
    spread is the point, so it is drawn per seed rather than averaged away.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = []
    for fp, d in iter_metrics(results_dir):
        k = variant_key(d, fp)
        rj = rq_path(fp.parent, "hrd_rhythm.json", create=False)
        if not rj.exists():
            continue
        try:
            circ = json.loads(rj.read_text(encoding="utf-8")).get("circadian_similarity") or {}
        except (json.JSONDecodeError, OSError):
            continue
        for br, st in circ.items():
            if br.startswith("Raw"):                 # the raw row is the input, not a result
                continue
            cv, pv = st.get("clock_var_frac"), st.get("participant_var_frac")
            if cv is None or pv is None:             # written by an older run
                continue
            # Three stages, in pipeline order: the backbone output, then the two branches the
            # disentangler splits it into. Keeping `backbone` here is what lets the reader see
            # WHERE a variant loses its structure instead of only that it did.
            stage = ("backbone" if br.startswith("Backbone") else
                     "season" if "V^(S)" in br else "trend")
            pts.append({"variant": f"{k.backbone}/{k.pe}", "seed": k.seed,
                        "branch": stage,
                        "clock": float(cv), "person": float(pv),
                        "swing": float(st.get("diurnal_amplitude", float("nan"))),
                        "bad": bool(st.get("degenerate"))})
    if not pts:
        return {}

    variants = sorted({p["variant"] for p in pts})
    cmap = plt.get_cmap("tab10")
    col = {v: cmap(i % 10) for i, v in enumerate(variants)}
    mark = {"backbone": "s", "trend": "^", "season": "o"}     # pipeline order

    jit = np.random.default_rng(0)                    # stable jitter, not hash-based
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(14.5, 5.2),
                                  gridspec_kw={"width_ratios": [1.15, 1.0]})

    ax.axhspan(0, 0.05, color="0.90", zorder=0)
    ax.text(0.99, 0.025, "no person information -- a clock, or collapsed", ha="right",
            va="center", fontsize=8, color="#7a1a1a")
    for p in pts:
        ax.scatter(p["clock"], p["person"], s=64 if not p["bad"] else 90,
                   marker=mark[p["branch"]], color="none" if p["bad"] else col[p["variant"]],
                   edgecolor="#b30000" if p["bad"] else col[p["variant"]],
                   linewidth=1.8 if p["bad"] else 0.8, alpha=0.95, zorder=3)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("variance explained by CLOCK TIME")
    ax.set_ylabel("variance explained by PARTICIPANT")
    ax.set_title("(a) is the representation rhythmic AND personal?", fontsize=10)
    ax.grid(alpha=0.25)
    hs = ([plt.Line2D([], [], ls="", marker="o", color=col[v], label=v) for v in variants]
          + [plt.Line2D([], [], ls="", marker=m, color="0.35", label=b)
             for b, m in mark.items()]
          + [plt.Line2D([], [], ls="", marker="o", mfc="none", mec="#b30000", mew=1.8,
                        color="none", label="degenerate")])
    ax.legend(handles=hs, fontsize=7.5, loc="upper left", ncol=2, framealpha=0.9)

    order = [(v, b) for v in variants for b in ("backbone", "trend", "season")
             if any(p["variant"] == v and p["branch"] == b for p in pts)]
    for i, (v, b) in enumerate(order):
        sel = [p for p in pts if p["variant"] == v and p["branch"] == b]
        for p in sel:
            axb.scatter(i + jit.normal(0, .05), p["swing"],
                        s=52, marker=mark[b], color="none" if p["bad"] else col[v],
                        edgecolor="#b30000" if p["bad"] else col[v],
                        linewidth=1.6 if p["bad"] else 0.8, zorder=3)
        if sel:
            m = float(np.median([p["swing"] for p in sel]))
            axb.plot([i - .28, i + .28], [m, m], color=col[v], lw=2.2, zorder=4)
    axb.set_xticks(range(len(order)))
    axb.set_xticklabels([f"{v}\n{b}" for v, b in order], fontsize=7.0)
    axb.set_ylabel("diurnal swing of cosine similarity")
    axb.set_ylim(-0.05, 2.05)                        # the full possible range, never autoscaled
    axb.set_title("(b) one point per seed -- how reproducible is it?", fontsize=10)
    axb.grid(alpha=0.25, axis="y")

    fig.suptitle("Circadian structure across variants and seeds  (supplementary to the "
                 "per-run figures)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {"n_points": len(pts), "n_degenerate": sum(p["bad"] for p in pts),
            "variants": variants}


def utility_across_tasks(results_dir, energy_dir):
    """RQ3 Part A as the design asks for it: one table, ladder x tasks.

    Depression comes from each variant's rq3/rq3_utility.csv; the two emotional-energy tasks
    come from the matching results_hrd_energy/<run>/<variant>_energy/metrics.json. Merging
    them is legitimate because train_hrd builds the energy test set as
    `set(test_pids) & set(pids_with_ee)` -- the SAME held-out participants -- so the columns
    are the same people scored on different targets.

    Rungs differ per task (the energy ladder has no cosinor or supervised rung), so a missing
    cell is printed as "-" rather than silently dropped: which comparisons exist is itself
    part of the reading.
    """
    from collections import defaultdict
    out = defaultdict(dict)                       # (variant, rung) -> {task: auc}
    for fp in sorted(Path(results_dir).glob("*/*/RQ3/rq3_utility.csv")):
        variant = fp.parent.parent.name
        with open(fp, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                # Part B rows share this file now; they are not ladder rungs, so they must not
                # appear as one in a cross-task comparison.
                if r.get("role", "ladder") != "ladder":
                    continue
                try:
                    out[(variant, r["representation"])]["depression"] = float(r["auc"])
                except (ValueError, KeyError):
                    pass
    for fp in sorted(Path(energy_dir).glob("*/*_energy/metrics.json")):
        variant = fp.parent.name[:-len("_energy")]
        m = read_json(fp) or {}
        for task, key in (("EE >= 4 (day)", "task1_high_energy"),
                          ("EE vs own median", "task3_within_person")):
            for rung, v in (m.get(key) or {}).items():
                if isinstance(v, dict) and v.get("auc_roc") is not None:
                    out[(variant, rung)][task] = float(v["auc_roc"])
    if not out:
        return []
    tasks = ["depression", "EE >= 4 (day)", "EE vs own median"]
    rows = [{"variant": v, "representation": r,
             **{t: o.get(t) for t in tasks}} for (v, r), o in sorted(out.items())]
    print("")
    print("=== RQ3 Part A: utility ladder x tasks "
          "(AUROC; '-' = rung absent for that task) ===")
    print(f"{'variant':<26}{'representation':<30}" + "".join(f"{t:>18}" for t in tasks))
    for r in rows:
        cells = "".join(f"{r[t]:>18.3f}" if r[t] is not None else f"{'-':>18}" for t in tasks)
        print(f"{r['variant']:<26}{r['representation']:<30}{cells}")
    return rows


# --------------------------------------------------------------------------- #
# Cross-variant panels. Both read the SAME tree the tables above do, through the
# shared reader, so a figure can never disagree with the table beside it.
# --------------------------------------------------------------------------- #
def _seed_ci(per):
    """(mean, lo, hi) over per-seed values; NaN-safe and never raises on n < 2."""
    from scipy import stats
    per = np.asarray(per, dtype=float)
    per = per[~np.isnan(per)]
    if per.size == 0:
        return np.nan, np.nan, np.nan
    if per.size < 2:
        return float(per[0]), float(per[0]), float(per[0])
    lo, hi = stats.t.interval(.95, per.size - 1, per.mean(), stats.sem(per))
    return float(per.mean()), float(lo), float(hi)


def fig_pe_dissociation(cur, out_dir, backbone="tcn"):
    """Positional encoding: does it move prediction, or only how the space is organised?"""
    from collections import defaultdict
    from scipy import stats
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tasks.style import BASE, GRID, INK, INK2, MUTED, POS, SURFACE, save, strip

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
            m, lo, hi = _seed_ci(tr[p][key])
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


# `better` = +1 when larger is better. Leak is the only metric where an increase is a
# DEGRADATION, so the colour must follow this and not the raw sign.
CLK = [("auc", "participant AUC", "prediction", +1),
       ("DIS", "disentanglement (DIS)", "structure", +1),
       ("rec_rhythm_branch", "rhythm capture, S branch", "structure", +1),
       ("leak_into_rhythm", "rhythm to trend leak", "structure", -1)]


def fig_clock_ablation(on, off, out_dir):
    """Paired calendar-channel on/off, over the cells the two runs share."""
    from scipy import stats
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tasks.style import BASE, GRID, INK, INK2, MUTED, NEG, POS, SURFACE, save, strip

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
        m, lo, hi = _seed_ci(per)
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
    ax.set_yticklabels([f"{lab}   \u00b7  {kind}" + ("  (lower is better)" if b < 0 else "")
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


def main():
    p = argparse.ArgumentParser(description="Summarise DSSL PE-variant results")
    p.add_argument("--results-dir", default="results_hrd")
    p.add_argument("--energy-dir", default="results_hrd_energy",
                   help="Where --energy-output-dir wrote the EE probes; merged into Part A.")
    p.add_argument("--csv", default=None, help="Optional path to write the table as CSV")
    p.add_argument("--fig-dir", default=str(Path("docs") / "figures"),
                   help="Where the cross-variant panels are written.")
    p.add_argument("--fig-backbone", default="tcn",
                   help="Backbone whose PE variants the dissociation panel compares.")
    p.add_argument("--clock-on", help="Run id WITH the calendar channels (enables the "
                                      "clock-ablation panel).")
    p.add_argument("--clock-off", help="Run id WITHOUT the calendar channels.")
    args = p.parse_args()

    rows = []
    base_samples = {}                       # model_name -> {seed -> metrics}, deduped by seed
    # One shared reader (scripts/_results.py) derives the variant identity for the tables
    # and the panels alike, so the two can never disagree about what a run is.
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

    # cross-variant rhythm <-> prediction link
    out_png = Path(args.results_dir) / "rhythm_vs_prediction.png"
    rr = rhythm_vs_prediction(rows, out_png)
    if rr is not None:
        fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else "n/a"
        print(f"\nPearson r vs participant AUC  ->  rhythm capture: {fmt(rr.get('rhythm'))}   "
              f"trend capture: {fmt(rr.get('trend'))}")
        print(f"Wrote {out_png}")

    # RQ3 Part A: the ladder against all three downstream tasks
    circ_png = Path(args.results_dir) / "circadian_landscape.png"
    cl = circadian_landscape(args.results_dir, circ_png)
    if cl:
        print(f"Wrote {circ_png}  ({cl['n_points']} branch-runs, "
              f"{cl['n_degenerate']} degenerate)")

    ut = utility_across_tasks(args.results_dir, args.energy_dir)
    if ut and args.csv:
        fp = Path(args.csv).with_name("summary_utility_tasks.csv")
        with open(fp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(ut[0]))
            w.writeheader(); w.writerows(ut)
        print(f"Wrote {fp}")

    # E1.4: effect of the temporal reference frame, the second half of RQ1
    pe = pe_contrast(args.results_dir)
    if pe and args.csv:
        fp = Path(args.csv).with_name("summary_pe_contrast.csv")
        with open(fp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(pe[0]))
            w.writeheader(); w.writerows(pe)
        print(f"Wrote {fp}")

    # cross-variant panels, from the same tree
    cur = load_variants(args.results_dir)
    census(cur, args.results_dir)
    fig_pe_dissociation(cur, args.fig_dir, args.fig_backbone)
    if args.clock_on and args.clock_off:
        fig_clock_ablation(load_variants(args.results_dir, [args.clock_on]),
                           load_variants(args.results_dir, [args.clock_off]), args.fig_dir)
    else:
        print("\nFIG D - skipped (pass --clock-on and --clock-off to build it)")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
