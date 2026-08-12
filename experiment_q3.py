"""RQ3 -- Utility and limits of the frozen representation for depression detection.

Runs on a FINISHED train_hrd.py variant directory: loads the frozen encoder, never trains
the encoder. Only the linear probe and the majority rule are fitted, on train participants.

  Part A  utility     linear probe vs a baseline ladder (majority / handcrafted /
                      random-init / frozen CoST), Delta AUC with a participant bootstrap CI.
  Part B  "how?"      three inference-only ablations -- which BRANCH (V^T vs V^S), which
                      SENSOR (channel zeroing), which TIMESCALE (input minus its circadian
                      component) carries the signal.
  Part C  limits      test-time degradation grid (duration / granularity / missingness /
                      channels) summarised by the breakdown point c*, the worst setting still
                      within 0.05 AUC of the intact input.

The probe is fit ONCE on clean training participants; every ablation and degradation is
applied to the held-out input only, which is what makes Parts B and C nearly free.

  python experiment_q3.py --variant-dir results_hrd/<run>/<backbone>_<pe>_seed<S>
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from baselines.plain_ssl import encode_plain, plain_ssl_encoder
from tasks._eval_protocols import participant_aggregate, participant_bootstrap_auc
from tasks._experiment_common import (encode, load_context, out_dir, random_init_model,
                                      save, wants_plain_ssl, write_csv)
from tasks.decomposition import harmonic_reference
from train_hrd import _make_probe

DROP = 0.05                       # AUC loss that defines the breakdown point c*


def fit_probe(feat, ctx, a):
    """Logistic probe on one row per participant -- the label is participant-level."""
    tr = ctx.train_mask & ctx.last_mask
    clf = _make_probe("supervised", a.probe_c, ctx.seed)
    clf.fit(feat[tr], ctx.y[tr])
    return clf


def score(clf, feat, ctx):
    """Participant-level AUROC on the held-out cohort."""
    te = ctx.test_mask & ctx.last_mask
    prob = clf.predict_proba(feat[te])[:, 1]
    pp, pl = participant_aggregate(ctx.pids[te], prob, ctx.y[te])
    return (float(roc_auc_score(pl, pp)) if len(set(pl)) > 1 else np.nan), pp, pl


def degrade(X, kind, level, n_sensors, sig, bpd, bin_minutes, rng):
    """Test-time input degradation. Only sensor channels are touched, so clock channels (when
    enabled) keep the shape the encoder was pretrained with."""
    Xd = X.copy()
    S = Xd[:, :, :n_sensors]
    T = X.shape[1]
    if kind == "duration_days":
        keep = int(level * bpd)
        S[:, :max(0, T - keep)] = 0.0                       # mask everything before the tail
    elif kind == "granularity_min":
        w = max(1, int(level // bin_minutes))
        if w > 1:
            n = (T // w) * w
            S[:, :n] = np.repeat(S[:, :n].reshape(len(X), -1, w, n_sensors).mean(2), w, axis=1)
    elif kind == "missing_mcar":
        S[rng.random(S.shape) < level] = 0.0
    elif kind == "missing_block":
        blk = max(1, int(0.25 * bpd))                       # 6-h non-wear gaps
        for i in range(len(X)):
            for st in rng.integers(0, T - blk, int(round(level * T / blk))):
                S[i, st:st + blk] = 0.0
    elif kind == "channels":
        S[:, :, int(level):] = 0.0                          # keep the first `level` sensors
    elif kind == "no_circadian":
        S -= sig
    return Xd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant-dir", required=True)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--probe-c", type=float, default=0.1)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--no-plain-ssl", action="store_true",
                   help="Drop the non-disentangled SSL rung. It is the ONLY baseline that "
                        "costs a real pretraining (~1 extra per seed x variant).")
    a = p.parse_args()

    ctx = load_context(a.variant_dir, a.cache_dir, a.gpu)
    d = out_dir(ctx, "rq3")
    rng = np.random.default_rng(ctx.seed)
    Xs = ctx.X[:, :, :ctx.n_sensors]
    _, sig = harmonic_reference(Xs, 24 * 60.0 / ctx.bin_minutes)
    res = {"variant": ctx.tag, "seed": ctx.seed}

    V = encode(ctx.model, ctx.X, ctx.cfg)
    dT = V.shape[1] // 2                                    # [V^(T) ; V^(S)]
    te = ctx.test_mask & ctx.last_mask

    # --- Part A: baseline ladder ---------------------------------------------------------
    ladder = {"Handcrafted": np.concatenate([Xs.mean(1), Xs.std(1)], axis=1),
              "Random-init": encode(random_init_model(ctx), ctx.X, ctx.cfg)}
    if wants_plain_ssl(ctx) and not a.no_plain_ssl:
        # Same SSL, same data, disentangler OFF -- the control that isolates what the
        # trend/seasonal split buys. Costs a real pretraining; cached beside the encoder.
        plain = plain_ssl_encoder(ctx.X, ctx.pretrain_mask, ctx.cfg, ctx.n_sensors,
                                  ctx.device, seed=ctx.seed,
                                  cache_path=ctx.variant_dir / "plain_encoder.pt")
        ladder["CoST plain (no disentangle)"] = encode_plain(plain, ctx.X, ctx.cfg)
    ladder["CoST (frozen)"] = V
    rows, probs = [], {}
    maj = float(ctx.y[ctx.train_mask & ctx.last_mask].mean())
    rows.append(["Majority", round(0.5, 4), "", ""])
    for name, feat in ladder.items():
        auc, pp, pl = score(fit_probe(feat, ctx, a), feat, ctx)
        probs[name] = (pp, pl)
        ci = participant_bootstrap_auc(pl, pp, np.arange(len(pl)), n_boot=a.n_boot,
                                       seed=ctx.seed)                    # rows already = people
        rows.append([name, round(auc, 4), round(ci["lo"], 4), round(ci["hi"], 4)])
        print(f"[rq3] {name}: AUC={auc:.3f} CI=[{ci['lo']:.3f}, {ci['hi']:.3f}]")
    write_csv(d, "rq3_utility", ["representation", "auc", "ci_lo", "ci_hi"], rows)
    res["utility"] = {r[0]: {"auc": r[1], "ci": r[2:]} for r in rows}
    res["majority_rate"] = maj

    # Delta AUC vs each baseline, bootstrapped over the SAME participants (paired).
    pp_c, pl_c = probs["CoST (frozen)"]
    res["delta_auc"] = {}
    # vs plain SSL is the headline contrast: it is the only rung differing ONLY by the split.
    for name in [n for n in ladder if n != "CoST (frozen)"]:
        pp_b, _ = probs[name]
        boot = []
        for _ in range(a.n_boot):
            ix = rng.integers(0, len(pl_c), len(pl_c))
            if len(set(pl_c[ix])) < 2:
                continue
            boot.append(roc_auc_score(pl_c[ix], pp_c[ix]) - roc_auc_score(pl_c[ix], pp_b[ix]))
        res["delta_auc"][name] = {
            "delta": float(np.mean(boot)) if boot else None,
            "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if boot else None}

    # --- Part B: how? branch / channel / timescale ---------------------------------------
    abl = [["branch", "V^T (trend) only", score(fit_probe(V[:, :dT], ctx, a), V[:, :dT], ctx)[0]],
           ["branch", "V^S (seasonal) only", score(fit_probe(V[:, dT:], ctx, a), V[:, dT:], ctx)[0]],
           ["branch", "full", res["utility"]["CoST (frozen)"]["auc"]]]
    clf = fit_probe(V, ctx, a)                    # probe trained on the intact input, reused
    for c, cname in enumerate(ctx.sensor_cols):
        Xz = ctx.X.copy(); Xz[:, :, c] = 0.0
        abl.append(["channel", f"drop {cname}", score(clf, encode(ctx.model, Xz, ctx.cfg), ctx)[0]])
    Xnc = degrade(ctx.X, "no_circadian", 0, ctx.n_sensors, sig, ctx.bins_per_day,
                  ctx.bin_minutes, rng)
    abl.append(["timescale", "input minus circadian", score(clf, encode(ctx.model, Xnc, ctx.cfg), ctx)[0]])
    write_csv(d, "rq3_ablation", ["level", "setting", "auc"],
              [[l, s, round(v, 4) if np.isfinite(v) else ""] for l, s, v in abl])
    res["ablation"] = [{"level": l, "setting": s, "auc": v} for l, s, v in abl]

    # --- Part C: degradation grid --------------------------------------------------------
    # Every factor must reach a level that actually BREAKS the probe, or c* just reports the
    # grid edge: in run 19606825 all five did exactly that. Extended to the point where the
    # construct is destroyed rather than merely degraded --
    #   duration    -- below 1 day there is less than one circadian cycle left;
    #   granularity -- 720 min = 12 h bins is the Nyquist limit for a 24 h rhythm, so the
    #                  rhythm is unrecoverable in principle at that point;
    #   missingness -- up to 95% MCAR / 80% contiguous non-wear.
    grid = {"duration_days": [0.25, 0.5, 1, 2, 3, 5, 7],
            "granularity_min": [15, 30, 60, 120, 240, 480, 720],
            "missing_mcar": [0.1, 0.2, 0.4, 0.6, 0.8, 0.95],
            "missing_block": [0.1, 0.2, 0.4, 0.6, 0.8],
            "channels": list(range(1, ctx.n_sensors))}
    full = res["utility"]["CoST (frozen)"]["auc"]
    drows, curves = [], {}
    for kind, levels in grid.items():
        for lv in levels:
            Xd = degrade(ctx.X, kind, lv, ctx.n_sensors, sig, ctx.bins_per_day,
                         ctx.bin_minutes, rng)
            auc = score(clf, encode(ctx.model, Xd, ctx.cfg), ctx)[0]
            drows.append([kind, lv, round(auc, 4) if np.isfinite(auc) else ""])
            curves.setdefault(kind, []).append((lv, auc))
    write_csv(d, "rq3_degradation", ["factor", "level", "auc"], drows)

    # c* = the most degraded level still within DROP of the intact AUC. "More degraded" runs
    # downward for duration/channels and upward for granularity/missingness, hence the sign.
    worse_is_larger = {"granularity_min", "missing_mcar", "missing_block"}
    res["auc_full"] = full
    res["n_test_participants"] = int(len(np.unique(ctx.pids[te])))
    # c* only means something if there is performance to lose: at chance level every
    # degradation trivially sits "within 0.05", which would read as perfect robustness.
    res["breakdown_valid"] = bool(np.isfinite(full) and full >= 0.55 + DROP)
    res["breakdown_point"] = {}
    for kind, pts in curves.items():
        okl = [lv for lv, auc in pts if np.isfinite(auc) and auc >= full - DROP]
        res["breakdown_point"][kind] = (
            (max(okl) if kind in worse_is_larger else min(okl))
            if okl and res["breakdown_valid"] else None)
    print(f"[rq3] breakdown points (within {DROP} AUC of {full:.3f}): {res['breakdown_point']}")

    fig, axes = plt.subplots(1, len(curves), figsize=(3.1 * len(curves), 3.4), squeeze=False)
    for ax, (kind, pts) in zip(axes[0], curves.items()):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "o-", color="#0072B2")
        ax.axhline(full, ls="-", c="grey", lw=0.8)
        ax.axhline(full - DROP, ls="--", c="#D55E00", lw=0.9)
        c = res["breakdown_point"][kind]
        if c is not None:
            ax.axvline(c, ls=":", c="#009E73")
        ax.set_xlabel(kind); ax.grid(alpha=0.25); ax.set_ylim(0.3, 1.02)
    axes[0][0].set_ylabel("participant AUROC")
    fig.suptitle(f"RQ3 degradation -- {ctx.tag} (dashed = full $-$ {DROP})", fontsize=11)
    fig.tight_layout(); fig.savefig(d / "rq3_degradation.png", dpi=200); plt.close(fig)

    save(d, "rq3", res)


if __name__ == "__main__":
    main()
