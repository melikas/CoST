"""Evaluation: RQ1, RQ2 and RQ3 over the folds `train.py` produced.

Two modes, so the expensive part shards across a SLURM array exactly like training does:

    python eval.py --run runs/oneshot --npz hrd_2224103.npz --only-fold r0f0   # one fold
    python eval.py --run runs/oneshot --npz hrd_2224103.npz --aggregate        # the report

Per-fold work writes `<arm>/<fold>/eval.json`; `--aggregate` reads all of them, applies the
Nadeau-Bengio correction across folds, and writes `results_summary.txt`.

WHAT EACH RQ CLAIMS, AND WHAT IT DOES NOT

  RQ2 (primary)   Can an unlabelled personal baseline detect a within-person rhythmic phase
                  deviation? Stratified Mann-Whitney concordance C, null exactly 0.5. This is
                  the flagship: it is the one level where the representation has ever
                  separated from its own architecture-matched untrained control.

  RQ3 (reported)  The downstream ladder, including RF-on-raw and a random projection of the
                  raw window. The expected finding is NEGATIVE -- untrained baselines top the
                  ladder -- and the protocol exists to make that finding provable rather than
                  merely unrejected. Nothing here is arranged to hide it.

  RQ1 (secondary) Does the representation carry cosinor structure better than a
                  dimension-matched PCA of the raw window? It is compared against PCA, NOT
                  against random-init, because random-init wins that comparison (0/24, 2/24,
                  0/24 on the three stability metrics) and the honest reading is that the
                  architecture's inductive bias carries the rhythm.

RQ2 re-encodes phase-perturbed windows, so it needs the ENCODER, not just the stored
representations. Run train.py with --save-encoder (scripts/oneshot.sh does).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score

from baselines.random_projection import raw_projection
from cost import CoST
from cv import Fold, nadeau_bengio, paired_test, required_margin
from data_loader import load_npz
from probe import make_probe

PHASE_LEVELS = (0.5, 1.0, 2.0, 3.0, 4.0)
BASELINE_R = 4                      # personal reference = 4 preceding windows = 28 days
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
PROBE_C = (0.01, 0.1, 1.0)


# =======================================================================================
# Shared rhythm machinery (ported verbatim from experiment_q2.py -- see its docstrings)
# =======================================================================================
def cosinor_z(Xs, bpd):
    """Per-window, per-channel 24 h cosinor coefficient as one complex number z = a + ib.

    Amplitude is |z| and acrophase is arg z, so a phase shift is exactly a rotation of z.
    Never pooled across channels: sleep runs roughly antiphase to activity and heart rate,
    so a pooled circular mean lands between two different constructs.
    """
    t = 2 * np.pi * np.arange(Xs.shape[1]) / bpd
    Z = Xs - Xs.mean(1)[:, None]
    a = 2 * (Z * np.cos(t)[None, :, None]).mean(1)
    b = 2 * (Z * np.sin(t)[None, :, None]).mean(1)
    return a + 1j * b


def window_start_days(window_ids):
    """Elapsed days of each window's start, from the "pid_<isotime>" window id."""
    t = pd.to_datetime([str(w).rsplit("_", 1)[1] for w in window_ids])
    return (t - t.min()).days.to_numpy().astype(float)


def personal_baseline(V, pids, R, tdays=None, max_span=None):
    """(mu, sd, ok): each window's reference is its R PRECEDING windows of the same person.

    Windows without a full reference are never scored. `max_span` additionally requires the
    reference to be contiguous in time -- the reference is selected by position, and a
    dropped window silently stretches it (8.6% of scored windows otherwise carry a reference
    spanning more than the nominal R-1 weeks, up to 77 days).
    """
    mu, sd, ok = np.zeros_like(V), np.ones_like(V), np.zeros(len(V), bool)
    for p in np.unique(pids):
        idx = np.flatnonzero(pids == p)
        for j in range(R, len(idx)):
            if max_span is not None and tdays[idx[j - 1]] - tdays[idx[j - R]] > max_span:
                continue
            ref = V[idx[j - R:j]]
            mu[idx[j]], sd[idx[j]], ok[idx[j]] = ref.mean(0), ref.std(0) + 1e-6, True
    return mu, sd, ok


def dscore(V, mu, sd):
    """Standardised Euclidean distance to the personal baseline: each dimension divided by
    its own within-person SD, then a plain Euclidean norm. Not Mahalanobis -- there is no
    inverse covariance, so correlated dimensions are counted once each."""
    return np.sqrt((((V - mu) / sd) ** 2).mean(1))


def raw_deviation(Zw, zbar):
    """The same functional form as `dscore`, in raw 24 h cosinor space. dd and dg are one
    object measured in two spaces, which is what makes their comparison interpretable."""
    return np.sqrt((np.abs(Zw - zbar) ** 2).mean(1))


def phase_shift(X, hours, n_sensors, bin_minutes):
    """Circular shift of the sensor channels only -- exactly a rotation of z. Clock channels
    are left alone; shifting them would move the reference frame with the signal."""
    Xp = X.copy()
    k = int(round(hours * 60 / bin_minutes))
    Xp[:, :, :n_sensors] = np.roll(X[:, :, :n_sensors], k, axis=1)
    return Xp


def resolve_phase_levels(levels, bin_minutes):
    """Keep only shifts the sampling grid can express, and that do not collide.

    A level below half a bin rolls by ZERO and perturbs nothing, and nothing downstream
    notices: dg is exactly 0 and the stratum contributes no pair. Run 2074344 hit this --
    at GLOBEM's 360-minute bins four of five levels were no-ops and RQ2 measured nothing
    while reporting success.
    """
    seen, keep, dropped = set(), [], []
    for lv in levels:
        k = int(round(float(lv) * 60 / bin_minutes))
        if k == 0 or k in seen:
            dropped.append(float(lv))
            continue
        seen.add(k)
        keep.append(float(lv))
    k_extra = 1
    while len(keep) < 3 and k_extra <= 12:
        if k_extra not in seen:
            seen.add(k_extra)
            keep.append(k_extra * bin_minutes / 60)
        k_extra += 1
    if dropped:
        print(f"    [rq2] dropped unrepresentable levels {dropped} h at {bin_minutes} min/bin")
    return sorted(set(keep))


def stratum_pairs(dd, dg, keys):
    """Per-stratum (U, n_pairs). U counts concordant (dg>0, dg<0) pairs, ties at 1/2, from
    ranks rather than an O(n^2) loop. A stratum with only one sign contributes nothing."""
    out = {}
    for s in np.unique(keys):
        m = (keys == s) & np.isfinite(dd) & np.isfinite(dg)
        pos, neg = dd[m & (dg > 0)], dd[m & (dg < 0)]
        if len(pos) == 0 or len(neg) == 0:
            continue
        r = rankdata(np.r_[pos, neg])
        out[s] = (float(r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2),
                  float(len(pos) * len(neg)))
    return out


def concordance(per_stratum):
    u = sum(v[0] for v in per_stratum.values())
    n = sum(v[1] for v in per_stratum.values())
    return (u / n) if n else float("nan")


# =======================================================================================
# Loading a fold's artefacts
# =======================================================================================
def load_fold(run, arm_tag, fold_tag):
    d = Path(run) / arm_tag / fold_tag
    rec = json.loads((d / "fold.json").read_text(encoding="utf-8"))
    z = np.load(d / "repr.npz", allow_pickle=True)
    return rec, {k: z[k] for k in z.files}


def load_encoder(run, weights_name, fold_tag, coh, plan, readout, device):
    """Rebuild the fold's encoder and load its weights. The geometry comes from plan.json,
    so the control and the model are guaranteed to be the architecture that was trained."""
    p = Path(run) / f"encoders_{weights_name}" / fold_tag / "encoder.pt"
    if not p.exists():
        raise SystemExit(
            f"missing {p}.\nRQ2 re-encodes phase-perturbed windows, so it needs the encoder, "
            "not just the stored representations.\nRe-run train.py with --save-encoder "
            "(scripts/oneshot.sh passes it), or use --skip-rq2.")
    m = _blank_model(coh, plan, readout, device, seed=0)
    return m.load(p)


def _blank_model(coh, plan, readout, device, seed):
    """An untrained CoST with the run's exact geometry -- the random-init control, and the
    shell RQ2 loads trained weights into."""
    return CoST(input_dims=coh.n_features, seq_len=coh.seq_len, bins_per_day=coh.bins_per_day,
                output_dims=plan["trend_dims"] + plan["seasonal_dims"],
                hidden_dims=plan.get("hidden_dims", 64), depth=plan["depth"],
                n_time_features=coh.n_features - coh.n_sensors,
                trend_kernel_cap=max(plan["trend_kernels"]) if plan["trend_kernels"] else None,
                seasonal_frac=plan["seasonal_dims"] / (plan["trend_dims"] + plan["seasonal_dims"]),
                phase_readout=readout, device=device, model_seed=seed)


# =======================================================================================
# RQ2 -- the flagship
# =======================================================================================
def rq2_arm(model, X, pids, tdays, coh, levels, test_mask):
    """Concordance C for one representation, over the fold's held-out participants.

    Scored on test participants only: the encoder never saw them, in pretraining or
    otherwise, so C measures a personal baseline built in a representation that is genuinely
    out of sample for that person.
    """
    bpd, ns, bm = coh.bins_per_day, coh.n_sensors, coh.bin_minutes
    V0 = model.encode(X, parts=False)
    max_span = float(7 * (BASELINE_R - 1))
    mu, sd, ok = personal_baseline(V0, pids, BASELINE_R, tdays, max_span)
    d0 = dscore(V0, mu, sd)

    # The raw-space reference is the COMPLEX mean of the preceding windows, not the mean of
    # magnitudes: a phase shift is a rotation of z, and averaging |z| would discard exactly
    # the quantity the perturbation moves. `personal_baseline` is reused for the mask only.
    Z0 = cosinor_z(X[:, :, :ns], bpd)
    _, _, zok = personal_baseline(np.abs(Z0), pids, BASELINE_R, tdays, max_span)
    zbar = np.zeros_like(Z0)
    for p in np.unique(pids):
        idx = np.flatnonzero(pids == p)
        for j in range(BASELINE_R, len(idx)):
            zbar[idx[j]] = Z0[idx[j - BASELINE_R:j]].mean(0)
    g0 = raw_deviation(Z0, zbar)

    scored = ok & zok & test_mask
    dd_all, dg_all, key_all, pid_all = [], [], [], []
    for lv in levels:
        Xp = phase_shift(X, lv, ns, bm)
        Vp = model.encode(Xp, parts=False)
        dp = dscore(Vp, mu, sd)                       # SAME frozen baseline
        gp = raw_deviation(cosinor_z(Xp[:, :, :ns], bpd), zbar)
        dd_all.append(dp - d0)
        dg_all.append(gp - g0)
        key_all.append(np.array([f"{p}|{lv}" for p in pids]))
        pid_all.append(pids)
    dd, dg = np.concatenate(dd_all), np.concatenate(dg_all)
    keys, pid_of_row = np.concatenate(key_all), np.concatenate(pid_all)
    m = np.tile(scored, len(levels))
    per = stratum_pairs(np.where(m, dd, np.nan), np.where(m, dg, np.nan), keys)
    return {"C": concordance(per), "n_strata": len(per),
            "n_pairs": float(sum(v[1] for v in per.values())),
            "n_scored_windows": int(scored.sum())}


def run_rq2(args, coh, plan, fold, recs, device):
    """Every arm at this fold, plus the architecture-matched random-init control."""
    tdays = window_start_days(coh.window_ids)
    levels = resolve_phase_levels(PHASE_LEVELS, coh.bin_minutes)
    out = {"levels": levels, "baseline_r": BASELINE_R, "arms": {}}
    for arm_tag, rec in recs.items():
        readout = rec["arm"]["phase_readout"]
        w = rec["arm"]["weights"]
        te = np.isin(coh.pids, list(rec["fold"]["test_pids"]))
        model = load_encoder(args.run, w, fold.tag, coh, plan, readout, device)
        out["arms"][arm_tag] = rq2_arm(model, coh.X, coh.pids, tdays, coh, levels, te)
        print(f"    [rq2] {arm_tag:22s} C={out['arms'][arm_tag]['C']:.4f} "
              f"({out['arms'][arm_tag]['n_pairs']:.0f} pairs)", flush=True)

    for readout in sorted({r["arm"]["phase_readout"] for r in recs.values()}):
        te = np.isin(coh.pids, list(fold.test_pids))
        ctrl = _blank_model(coh, plan, readout, device, seed=fold.model_seed)
        tag = f"random-init_{readout}"
        out["arms"][tag] = rq2_arm(ctrl, coh.X, coh.pids, tdays, coh, levels, te)
        print(f"    [rq2] {tag:22s} C={out['arms'][tag]['C']:.4f}", flush=True)
    return out


# =======================================================================================
# RQ3 -- the honest ladder
# =======================================================================================
def participant_scores(prob, pids_w, y_w):
    """Collapse window predictions to one score and one label per participant."""
    ps, ys = [], []
    for p in np.unique(pids_w):
        m = pids_w == p
        ps.append(float(prob[m].mean()))
        ys.append(int(round(float(y_w[m].mean()))))
    return np.array(ps), np.array(ys)


def probe_auc(Xtr, ytr, Xte, pids_te, y_te_w, seed, mode="supervised",
              pair_start=None, pair_width=0):
    """Fit a probe on the training participants, score the held-out ones.

    `C` is chosen on the TRAINING split only, by a small internal grid, and the winning
    value is applied identically to every arm -- so no arm is granted a probe another was
    denied. That was measured to matter: the probe family alone was worth nearly half the
    margin being competed for on the GLOBEM benchmark.
    """
    best, best_c = -np.inf, PROBE_C[0]
    n = len(Xtr)
    cut = max(1, int(n * 0.75))
    for c in PROBE_C:
        try:
            pr = make_probe(mode, c, seed, pair_start=pair_start, pair_width=pair_width)
            pr.fit(Xtr[:cut], ytr[:cut])
            s = roc_auc_score(ytr[cut:], pr.predict_proba(Xtr[cut:])[:, 1])
        except (ValueError, IndexError):
            continue
        if s > best:
            best, best_c = s, c
    pr = make_probe(mode, best_c, seed, pair_start=pair_start, pair_width=pair_width)
    pr.fit(Xtr, ytr)
    prob = pr.predict_proba(Xte)[:, 1]
    sc, ys = participant_scores(prob, pids_te, y_te_w)
    if len(np.unique(ys)) < 2:
        return float("nan"), best_c
    return float(roc_auc_score(ys, sc)), best_c


def run_rq3(args, coh, plan, fold, recs, device):
    """The ladder: untrained references first, then the arms, all on identical rows."""
    tr = split_idx(args.run, fold.tag, recs, "probe_train")
    te = split_idx(args.run, fold.tag, recs, "probe_test")
    y_tr = coh.y[tr]
    y_te_w = coh.y[te]
    pids_te = coh.pids[te]
    seed = fold.probe_seed
    ladder = {}

    flat = coh.X[:, :, :coh.n_sensors].reshape(len(coh.X), -1)
    flat = np.nan_to_num(flat, nan=0.0)
    ladder["RF on raw"] = probe_auc(flat[tr], y_tr, flat[te], pids_te, y_te_w,
                                    seed, mode="forest")
    proj = raw_projection(coh.X, coh.n_sensors, 512, seed)
    ladder["Random projection (512)"] = probe_auc(proj[tr], y_tr, proj[te], pids_te, y_te_w, seed)
    for readout in sorted({r["arm"]["phase_readout"] for r in recs.values()}):
        ctrl = _blank_model(coh, plan, readout, device, seed=fold.model_seed)
        V = ctrl.encode(coh.X, parts=False)
        ps, pw = _pair_of(recs, readout)
        ladder[f"Random-init ({readout})"] = probe_auc(V[tr], y_tr, V[te], pids_te, y_te_w, seed,
                                                       pair_start=ps, pair_width=pw)
    for arm_tag, rec in recs.items():
        z = np.load(Path(args.run) / arm_tag / fold.tag / "repr.npz")
        V = z["full"]
        ps, pw = rec["pair_block_full"]
        ladder[f"DSSL {arm_tag}"] = probe_auc(V[tr], y_tr, V[te], pids_te, y_te_w, seed,
                                              pair_start=ps, pair_width=pw or 0)
    for k, (auc, c) in ladder.items():
        print(f"    [rq3] {k:34s} AUROC={auc:.4f} (C={c})", flush=True)
    return {k: {"auc": a, "probe_C": c} for k, (a, c) in ladder.items()}


def split_idx(run, tag, recs, key):
    """The split indices for one fold, asserted identical across arms rather than assumed.

    Every arm of a fold is built from the same participants, so if these ever disagree the
    paired comparisons downstream are meaningless -- better to stop than to average them.
    """
    vals = {arm: tuple(np.load(Path(run) / arm / tag / "repr.npz")[key].tolist())
            for arm in recs}
    if len(set(vals.values())) != 1:
        raise AssertionError(f"arms of fold {tag} disagree on {key!r}: "
                             f"{ {a: len(v) for a, v in vals.items()} }")
    return np.asarray(next(iter(vals.values())))


def _pair_of(recs, readout):
    for rec in recs.values():
        if rec["arm"]["phase_readout"] == readout:
            ps, pw = rec["pair_block_full"]
            return ps, (pw or 0)
    return None, 0


# =======================================================================================
# RQ1 -- cosinor recovery against a dimension-matched PCA
# =======================================================================================
def cosinor_markers(X, n_sensors, bpd):
    """Targets read from the RAW signal: MESOR, amplitude and acrophase per channel.

    Acrophase is regressed as (cos, sin) and recovered by atan2, so the branch cut is never
    crossed by a linear model that cannot represent it.
    """
    Xs = X[:, :, :n_sensors]
    z = cosinor_z(Xs, bpd)
    return {"MESOR": Xs.mean(1), "amplitude": np.abs(z),
            "acrophase_cos": np.cos(np.angle(z)), "acrophase_sin": np.sin(np.angle(z)),
            "_angle": np.angle(z)}


def ridge_readout(Vtr, Ttr, Vte):
    m = RidgeCV(alphas=RIDGE_ALPHAS).fit(Vtr, Ttr)
    return m.predict(Vte)


def r2(t, p):
    ss = ((t - p) ** 2).sum()
    tot = ((t - t.mean(0)) ** 2).sum()
    return float(1 - ss / tot) if tot > 0 else float("nan")


def acrophase_hours(true_ang, pred_cos, pred_sin, bpd_hours=24.0):
    """Mean absolute acrophase error in hours, on the circle."""
    pred = np.arctan2(pred_sin, pred_cos)
    d = np.abs(np.angle(np.exp(1j * (pred - true_ang))))
    return float(d.mean() * bpd_hours / (2 * np.pi))


def run_rq1(args, coh, plan, fold, recs):
    """Ridge read-out of cosinor markers from the representation, against a PCA of the raw
    window at the representation's own width.

    PCA, not random-init, is the comparator here on purpose. Random-init WINS the stability
    comparison (0/24, 2/24, 0/24 across 24 seeds), and the honest reading of that is that
    the architecture's inductive bias -- not the training -- carries the rhythm. RQ1's
    defensible claim is against a linear compression of the same input, and it is stated
    that way rather than inflated.
    """
    tr = split_idx(args.run, fold.tag, recs, "probe_train")
    te = split_idx(args.run, fold.tag, recs, "probe_test")
    tgt = cosinor_markers(coh.X, coh.n_sensors, coh.bins_per_day)
    ang_te = tgt["_angle"][te]
    out = {}

    flat = np.nan_to_num(coh.X[:, :, :coh.n_sensors].reshape(len(coh.X), -1), nan=0.0)

    def score(V, name):
        res = {}
        for k in ("MESOR", "amplitude"):
            res[k] = r2(tgt[k][te], ridge_readout(V[tr], tgt[k][tr], V[te]))
        pc = ridge_readout(V[tr], tgt["acrophase_cos"][tr], V[te])
        ps = ridge_readout(V[tr], tgt["acrophase_sin"][tr], V[te])
        res["acrophase_mae_hours"] = acrophase_hours(ang_te, pc, ps)
        out[name] = res
        print(f"    [rq1] {name:34s} MESOR R2={res['MESOR']:.3f} "
              f"amp R2={res['amplitude']:.3f} acro MAE={res['acrophase_mae_hours']:.2f} h",
              flush=True)

    widths = set()
    for arm_tag in recs:
        V = np.load(Path(args.run) / arm_tag / fold.tag / "repr.npz")["full"]
        score(V, f"DSSL {arm_tag}")
        widths.add(V.shape[1])

    # One PCA comparator per DISTINCT achievable width -- "dimension-matched" is the whole
    # point of the comparison, and the two readouts differ in width. Dedup on the width PCA
    # can actually reach, not on the requested one: both readouts often clip to the same k
    # (bounded by the training rows), and fitting it twice would print one comparator twice.
    # NAMED BY THE WIDTH IT IS MATCHED TO, not by the k actually achieved. k is capped by
    # the training rows and so drifts fold to fold (151/159/167 on a 3-fold run); naming the
    # series after it would split one comparator into three, each present in a couple of
    # folds, and nothing would be comparable across the protocol.
    for w in sorted(widths):
        k = min(w, len(tr) - 1, flat.shape[1])
        pca = PCA(n_components=k, random_state=fold.probe_seed).fit(flat[tr])
        score(pca.transform(flat), f"PCA matched to {w}d")
        out[f"PCA matched to {w}d"]["k_achieved"] = int(k)
    return out


# =======================================================================================
# Driver
# =======================================================================================
def fold_from_plan(plan, tag):
    for f in plan["folds"]:
        if f"r{f['repeat']}f{f['fold']}" == tag:
            return Fold(repeat=f["repeat"], fold=f["fold"], split_seed=f["split_seed"],
                        model_seed=f["model_seed"], probe_seed=f["probe_seed"],
                        test_pids=tuple(f["test_pids"]), train_pids=tuple(f["train_pids"]))
    raise SystemExit(f"fold {tag!r} is not in plan.json")


def eval_fold(args, coh, plan, tag):
    run = Path(args.run)
    arms = [a["phase_readout"] + "_" + a["weights"] for a in plan["arms"]]
    recs = {}
    for a in arms:
        p = run / a / tag / "fold.json"
        if p.exists():
            recs[a] = json.loads(p.read_text(encoding="utf-8"))
    if not recs:
        raise SystemExit(f"no arm has results for fold {tag} under {run}")
    fold = fold_from_plan(plan, tag)
    print(f"[fold] {tag}: {len(recs)} arms, {len(fold.test_pids)} held-out participants")

    res = {"fold": tag, "arms_present": sorted(recs)}
    if not args.skip_rq1:
        res["rq1"] = run_rq1(args, coh, plan, fold, recs)
    if not args.skip_rq2:
        res["rq2"] = run_rq2(args, coh, plan, fold, recs, args.device)
    if not args.skip_rq3:
        res["rq3"] = run_rq3(args, coh, plan, fold, recs, args.device)
    for a in recs:
        (run / a / tag / "eval.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


# ---------------------------------------------------------------------------------------
def aggregate(args, plan):
    run = Path(args.run)
    per_fold = {}
    for p in sorted(run.glob("*/*/eval.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        per_fold[r["fold"]] = r
    if not per_fold:
        raise SystemExit(f"no eval.json under {run}; run the per-fold pass first")
    nf, nr = plan["protocol"]["n_folds"], plan["protocol"]["n_repeats"]
    txt = report(plan, per_fold, nf, nr)
    out = run / "results_summary.txt"
    out.write_text(txt, encoding="utf-8")
    print(txt)
    print(f"\n[out] {out}")
    return 0


def _collect(per_fold, rq, path, sub=None):
    """{series name: array over folds}, aligned so every series shares one fold order.

    `sub` names the nested dict holding the series -- RQ2 stores its arms under "arms",
    alongside the protocol fields (levels, baseline_r) that are not series at all.
    """
    tags = sorted(per_fold)

    def block(t):
        d = per_fold[t].get(rq, {})
        return d.get(sub, {}) if sub else d

    names = set()
    for t in tags:
        names |= set(block(t))
    series = {}
    for n in sorted(names):
        v = []
        for t in tags:
            d = block(t).get(n)
            if d is None:
                v.append(np.nan)
            elif isinstance(d, dict):
                v.append(d.get(path, np.nan))
            else:
                v.append(d)
        series[n] = np.array(v, dtype=float)
    return tags, series


CONTRAST_COLS = ["diff", "margin", "wins", "verdict"]


def _contrast(t):
    """The four cells every paired contrast prints: effect, margin, wins, verdict.

    The verdict carries the SIGN, which a bare significance flag does not. `paired_test`
    reports significance as |mean| > margin -- sign-blind by construction, because it asks
    whether two arms differ, not which is better. Printing "YES" for both therefore reads
    identically for an arm that beats its control and one that loses to it, and this study
    contains both: angle_contracted clears random-init by +0.0254 while circular_contracted
    falls below its own control by -0.0339, and both are significant. Direction is the
    finding, so it is stated rather than left to be inferred from the sign of `diff`.

    WIN and LOSS are from the perspective of the row's subject -- the first column -- against
    the comparator named beside it.
    """
    verdict = "ns" if not t["significant"] else ("WIN" if t["mean_diff"] > 0 else "LOSS")
    return [_fmt(t["mean_diff"]), _fmt(t["required_margin"]),
            f"{t['wins']}/{t['n']}", verdict]


def _table(rows, headers):
    w = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    line = "  ".join("-" * x for x in w)
    out = ["  ".join(str(h).ljust(w[i]) for i, h in enumerate(headers)).rstrip(), line]
    out += ["  ".join(str(r[i]).ljust(w[i]) for i in range(len(headers))).rstrip() for r in rows]
    return "\n".join(out)


def _fmt(v, n=4):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{n}f}"


def report(plan, per_fold, nf, nr):
    L = []
    A = L.append
    rule = "=" * 88
    A(rule)
    A("RESULTS -- pre-registered 4-arm run")
    A(rule)
    A(f"Cohort     {plan.get('n_labelled_participants','?')} labelled of "
      f"{plan.get('n_participants_total','?')} participants, prevalence "
      f"{plan.get('prevalence','?')}, {plan['windows'][0]} windows of "
      f"{plan['seq_len']} x {plan['windows'][2]}.")
    A(f"Protocol   {nf}-fold x {nr} repeats; {len(per_fold)} of {nf*nr} folds evaluated.")
    A(f"Geometry   depth {plan['depth']}, RF {plan['receptive_field']} "
      f"({plan['rf_over_window']}x window), bands {plan['bands']},")
    A(f"           trend kernels {plan['trend_kernels']}, V^T {plan['trend_dims']} / "
      f"V^S {plan['seasonal_dims']}, {plan['n_params']:,} params.")
    A(f"Inference  Nadeau-Bengio corrected: variance inflation 1/K + 1/(F-1) = "
      f"{nadeau_bengio(nf, nr):.4f}.")
    A("           A difference is significant only if it exceeds the margin printed beside")
    A(f"           it. Uncorrected, {nf*nr} correlated folds would be treated as {nf*nr}")
    A("           independent samples and nearly everything would look significant.")

    # ---- RQ2 -------------------------------------------------------------------------
    A("\n" + rule)
    A("RQ2 (PRIMARY) -- within-person rhythmic deviation, stratified Mann-Whitney C")
    A(rule)
    A("""
C is the probability that a window pushed FURTHER from its own rhythm scores higher than one
pushed closer, within a (participant, shift level) stratum. Null is exactly 0.5. Scored on
held-out participants only: the encoder never saw them, in pretraining or otherwise.
""")
    tags, series = _collect(per_fold, "rq2", "C", sub="arms")
    if not series:
        A("(no RQ2 results)")
    else:
        rows = [[n, _fmt(np.nanmean(v)), _fmt(np.nanstd(v, ddof=1)), int(np.isfinite(v).sum())]
                for n, v in sorted(series.items(), key=lambda kv: -np.nanmean(kv[1]))]
        A(_table(rows, ["arm", "mean C", "SD", "folds"]))
        A("\nPaired contrasts (same folds, Nadeau-Bengio corrected):")
        rows = []
        for arm in sorted(a for a in series if not a.startswith("random-init")):
            ro = "circular" if "circular" in arm else "angle"
            ctrl = f"random-init_{ro}"
            if ctrl not in series:
                continue
            m = np.isfinite(series[arm]) & np.isfinite(series[ctrl])
            if m.sum() < 2:
                continue
            t = paired_test(series[arm][m], series[ctrl][m], nf, nr)
            rows.append([arm, f"vs {ctrl}"] + _contrast(t))
        arms_only = sorted(a for a in series if not a.startswith("random-init"))
        for i in range(len(arms_only)):
            for j in range(i + 1, len(arms_only)):
                a1, a2 = arms_only[i], arms_only[j]
                m = np.isfinite(series[a1]) & np.isfinite(series[a2])
                if m.sum() < 2:
                    continue
                t = paired_test(series[a1][m], series[a2][m], nf, nr)
                rows.append([a1, f"vs {a2}"] + _contrast(t))
        A(_table(rows, ["arm", "against"] + CONTRAST_COLS)
          if rows else "(no comparable pairs)")

    # ---- RQ3 -------------------------------------------------------------------------
    A("\n" + rule)
    A("RQ3 (REPORTED) -- downstream ladder, participant-level AUROC")
    A(rule)
    A("""
Read this ladder expecting the untrained references to win. That is the finding, it is
supported by an architecture-matched control on every rung, and nothing here is arranged to
soften it. Prior measurement on the single-holdout protocol: RF-on-raw 0.731, random
projection 0.7198, random-init 0.6874, DSSL 0.679, supervised 0.6609.
""")
    tags, series = _collect(per_fold, "rq3", "auc")
    if not series:
        A("(no RQ3 results)")
    else:
        rows = [[n, _fmt(np.nanmean(v)), _fmt(np.nanstd(v, ddof=1)), int(np.isfinite(v).sum())]
                for n, v in sorted(series.items(), key=lambda kv: -np.nanmean(kv[1]))]
        A(_table(rows, ["rung", "mean AUROC", "SD", "folds"]))
        best_dssl = max((n for n in series if n.startswith("DSSL")),
                        key=lambda n: np.nanmean(series[n]), default=None)
        if best_dssl:
            A(f"\nEvery rung against the best DSSL arm ({best_dssl}):")
            rows = []
            for n in sorted(series):
                if n == best_dssl:
                    continue
                m = np.isfinite(series[n]) & np.isfinite(series[best_dssl])
                if m.sum() < 2:
                    continue
                t = paired_test(series[n][m], series[best_dssl][m], nf, nr)
                rows.append([n] + _contrast(t))
            A(_table(rows, ["rung"] + CONTRAST_COLS))

    # ---- RQ1 -------------------------------------------------------------------------
    A("\n" + rule)
    A("RQ1 (SECONDARY) -- cosinor recovery vs a dimension-matched PCA of the raw window")
    A(rule)
    A("""
Compared against PCA, NOT against random-init. Random-init wins the stability comparison
(0/24, 2/24, 0/24 across three metrics on 24 seeds), and the honest reading is that the
architecture's inductive bias, not the training, carries the rhythm. RQ1's defensible claim
is against a linear compression of the same input, and is stated that way.
""")
    for metric, better, unit in (("MESOR", "higher", "R2"),
                                 ("amplitude", "higher", "R2"),
                                 ("acrophase_mae_hours", "lower", "hours")):
        tags, series = _collect(per_fold, "rq1", metric)
        if not series:
            continue
        rev = better == "higher"
        rows = [[n, _fmt(np.nanmean(v), 3), _fmt(np.nanstd(v, ddof=1), 3),
                 int(np.isfinite(v).sum())]
                for n, v in sorted(series.items(),
                                   key=lambda kv: -np.nanmean(kv[1]) if rev else np.nanmean(kv[1]))]
        A(f"\n{metric} ({unit}, {better} is better)")
        A(_table(rows, ["representation", f"mean {unit}", "SD", "folds"]))

    A("\n" + rule)
    A("END")
    A(rule)
    return "\n".join(L)


# ---------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a train.py run: RQ1, RQ2, RQ3.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run", required=True, help="train.py --out directory")
    p.add_argument("--npz", help="the window cache the run used (required unless --aggregate)")
    p.add_argument("--only-fold", default=None, help="evaluate one fold, e.g. r0f3")
    p.add_argument("--aggregate", action="store_true", help="combine eval.json into the report")
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-rq1", action="store_true")
    p.add_argument("--skip-rq2", action="store_true", help="skip if encoders were not saved")
    p.add_argument("--skip-rq3", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    plan = json.loads((Path(args.run) / "plan.json").read_text(encoding="utf-8"))
    if args.aggregate:
        return aggregate(args, plan)
    if not args.npz:
        raise SystemExit("--npz is required for the per-fold pass")
    coh = load_npz(args.npz)
    if coh.window_ids is None and not args.skip_rq2:
        raise SystemExit("RQ2 needs window_ids (for contiguous personal baselines) and the "
                         "cache has none; pass --skip-rq2 or use a cache that carries them")
    tags = ([args.only_fold] if args.only_fold
            else [f"r{f['repeat']}f{f['fold']}" for f in plan["folds"]])
    for t in tags:
        eval_fold(args, coh, plan, t)
    print(f"[done] {len(tags)} fold(s) evaluated -> run with --aggregate for the report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
