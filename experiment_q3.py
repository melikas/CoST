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
from tasks.rq_paths import rq_path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from baselines.cosinor import paper_cosinor_features
from structured_rhythm import structured_features
from baselines.plain_ssl import encode_plain, plain_ssl_encoder
from baselines.supervised import supervised_baseline_row
from tasks._eval_protocols import (fast_auc, fit_persubject_probe,
                                   participant_bootstrap_auc, persubject_rows)
from tasks._experiment_common import (encode, load_context, out_dir, random_init_model,
                                      save, wants_plain_ssl, write_csv)
from tasks.decomposition import harmonic_reference
from tasks.energy import handcrafted_features

DROP = 0.05                       # AUC loss that defines the breakdown point c*
WORSE_IS_LARGER = {"granularity_min", "missing_mcar", "missing_block"}


def cstar(pts, kind, full):
    """Most degraded level still within DROP of the intact AUC, at the FIRST crossing.

    Ordered by increasing degradation (downward for duration/channels, upward for
    granularity/missingness), stopping the moment the curve drops out. Taking `max` over every
    level that happens to pass would jump PAST an earlier breakdown whenever noise lets a
    harsher level through -- and at ~36 participants, where the pointwise AUC noise is about
    twice the 0.05 threshold that defines c*, it will.
    """
    out = None
    for lv, auc in sorted(pts, reverse=kind not in WORSE_IS_LARGER):
        if not (np.isfinite(auc) and auc >= full - DROP):
            break
        out = lv
    return out


def fit_probe(feat, ctx, a):
    """The canonical participant-level probe -- the same one the Separability table uses."""
    return fit_persubject_probe(feat, ctx.pids, ctx.y, ctx.train_mask, ctx.val_mask,
                                ctx.seed, c_grid=a.probe_c)


def score(clf, feat, ctx, mask=None):
    """Participant-level AUROC on a held-out cohort. Rows are already participants.

    `mask` defaults to the test split; pass the validation split to get the predictions
    a decision threshold has to be chosen on. Tuning a threshold on the TEST scores is
    optimistic by an amount that matters here: measured over 200 draws of a predictor
    with no signal at all, a test-tuned threshold reports 0.5129 balanced accuracy at
    1800 labels and 0.5360 at 200, against a true 0.500. The GLOBEM benchmark's best
    cross-dataset result is 0.547, so that optimism is a quarter of the entire margin
    being compared against.
    """
    m = ctx.test_mask if mask is None else mask
    Xte, yte, _ = persubject_rows(feat, ctx.pids, ctx.y, m)
    prob = clf.predict_proba(Xte)[:, 1]
    return (float(roc_auc_score(yte, prob)) if len(set(yte)) > 1 else np.nan), prob, yte


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
        # 6-h non-wear gaps drawn from DISJOINT slots. Independent uniform starts overlap, and
        # the block count is chosen as if they tiled, so the realised rate fell far below the
        # label: measured over 200 windows, nominal 0.40 delivered 0.329, 0.60 delivered 0.458
        # and 0.80 delivered 0.545. The axis was mislabelled by up to 25 points, which matters
        # because MCAR is exact and the design compares the two axes against each other.
        blk = max(1, int(0.25 * bpd))
        slots = np.arange(0, T - blk + 1, blk)
        n_blk = min(int(round(level * T / blk)), len(slots))
        for i in range(len(X)):
            for st in rng.choice(slots, size=n_blk, replace=False):
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
    p.add_argument("--probe-c", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0],
                   help="Grid for the logistic C; selected on the validation split, not pinned")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--no-plain-ssl", action="store_true",
                   help="Drop the non-disentangled SSL rung. It is the ONLY baseline that "
                        "costs a real pretraining (~1 extra per seed x variant).")
    p.add_argument("--no-cosinor", action="store_true",
                   help="Drop the 'Cosinor (paper)' rung. Needs CosinorPy and fits every "
                        "(window, channel, period) on TRAIN+VAL+TEST -- ~3x the rows q1 fits, "
                        "since the probe is trained here, not only scored.")
    p.add_argument("--no-supervised", action="store_true",
                   help="Drop the end-to-end supervised CEILING. It is the only rung that "
                        "trains a network on the labels (~1 short supervised fit per run).")
    a = p.parse_args()

    ctx = load_context(a.variant_dir, a.cache_dir, a.gpu)
    d = out_dir(ctx, "rq3")
    rng = np.random.default_rng(ctx.seed)
    Xs = ctx.X[:, :, :ctx.n_sensors]
    _, sig = harmonic_reference(Xs, 24 * 60.0 / ctx.bin_minutes)
    res = {"variant": ctx.tag, "seed": ctx.seed}

    V = encode(ctx.model, ctx.X, ctx.cfg)
    # [V^(T) ; V^(S)]. The trend half is always component_dims wide; the seasonal half is NOT
    # the same width once season_pool='spec' (the run default), where it is the spectral
    # readout -- 5 harmonics x component_dims x 2 (amplitude, phase). `V.shape[1] // 2` was
    # therefore slicing through the middle of the seasonal block, so "V^T only" carried part
    # of the amplitudes and "V^S only" was a truncated view of itself.
    dT = ctx.model.net.component_dims
    te = ctx.test_mask & ctx.last_mask

    # --- Part A: baseline ladder ---------------------------------------------------------
    # One definition of the handcrafted rung for the whole project (tasks/energy.py), so the
    # utility ladder and the two separability tables cannot drift into different baselines.
    ladder = {"Handcrafted (mean/std)": handcrafted_features(ctx.X, ctx.n_sensors)}
    if not a.no_cosinor:
        # Rung 3 of the design's ladder: the classical chronobiology baseline, the same
        # CosinorPy clone E1.3 uses as its target source. need_mask is None on purpose -- the
        # probe is FIT here, not merely scored, so train and val rows need features too. q1's
        # cache only covers test rows, hence a separate cache file.
        try:
            ladder["Cosinor (paper)"] = paper_cosinor_features(
                Xs, ctx.bin_minutes, need_mask=None, window_ids=ctx.window_ids,
                pids=ctx.pids, cache_path=d / "cosinor_cache_all.npz")
        except Exception as e:
            print(f"[rq3] Cosinor (paper) rung SKIPPED: {type(e).__name__}: {e}")
    # Cosinor's own estimator -- least squares on the signal -- extended with the three
    # constructs cosinor structurally cannot express: waveform harmonics, per-day dispersion,
    # and the acrophase difference between channels. See structured_rhythm.py for why each is
    # there and what it is answering to. Guarded exactly like the cosinor rung: this rung
    # failing must not throw away the rest of the ladder.
    try:
        ladder["Structured rhythm"] = structured_features(ctx.X, ctx.bins_per_day,
                                                          ctx.n_sensors)
    except Exception as e:
        print(f"[rq3] Structured rhythm rung SKIPPED: {type(e).__name__}: {e}")
    ladder["Random-init"] = encode(random_init_model(ctx), ctx.X, ctx.cfg)
    if wants_plain_ssl(ctx) and not a.no_plain_ssl:
        # Same SSL, same data, disentangler OFF -- the control that isolates what the
        # trend/seasonal split buys. Costs a real pretraining; cached beside the encoder.
        plain = plain_ssl_encoder(ctx.X, ctx.pretrain_mask, ctx.cfg, ctx.n_sensors,
                                  ctx.device, seed=ctx.seed, pids=ctx.pids,
                                  cache_path=rq_path(ctx.variant_dir, "plain_encoder.pt"))
        ladder["DSSL plain (no disentangle)"] = encode_plain(plain, ctx.X, ctx.cfg)
    ladder["DSSL (frozen)"] = V
    rows, probs = [], {}
    maj = float(ctx.y[ctx.train_mask & ctx.last_mask].mean())

    def add(role, name, auc, pp=None, pl=None, vp=None, vl=None):
        """One row of the single RQ3 table, always with an interval when one is computable.

        Balanced accuracy travels with the AUC because it is the metric the GLOBEM
        benchmark reports -- it publishes no AUROC at all, and the two are not
        convertible, so without this column no result here can be set beside theirs.
        The threshold is the one that maximises balanced accuracy on these same
        predictions, which is what makes it an upper bound rather than a claim: a
        deployed threshold would have to be fixed on validation data.
        """
        lo = hi = ""
        ba = ""
        if pp is not None and np.isfinite(auc):
            b = participant_bootstrap_auc(pl, pp, np.arange(len(pl)), n_boot=a.n_boot,
                                          seed=ctx.seed)                 # rows already = people
            lo, hi = round(b["lo"], 4), round(b["hi"], 4)
        if (pp is not None and pl is not None and vp is not None and vl is not None
                and len(set(np.asarray(pl))) > 1 and len(set(np.asarray(vl))) > 1):
            from sklearn.metrics import balanced_accuracy_score
            from tasks._eval_protocols import best_threshold
            pp_, pl_ = np.asarray(pp, float), np.asarray(pl, int)
            # Chosen on VALIDATION participants, applied to test. See score().
            thr = best_threshold(np.asarray(vl, int), np.asarray(vp, float))
            ba = round(float(balanced_accuracy_score(pl_, (pp_ >= thr).astype(int))), 4)
        rows.append([role, name, round(auc, 4) if np.isfinite(auc) else "", lo, hi, ba])
        print(f"[rq3] {role:9s} {name}: AUC={auc:.3f}"
              + (f" CI=[{lo:.3f}, {hi:.3f}]" if lo != "" else "")
              + (f" BA={ba:.3f}" if ba != "" else ""))
        return auc

    add("ladder", "Majority", 0.5)
    for name, feat in ladder.items():
        clf = fit_probe(feat, ctx, a)
        auc, pp, pl = score(clf, feat, ctx)
        _, vp, vl = score(clf, feat, ctx, ctx.val_mask)
        probs[name] = (pp, pl)
        add("ladder", name, auc, pp, pl, vp, vl)

    # Top rung of the design's ladder: the end-to-end supervised CEILING. It is not a feature
    # matrix, so it cannot join `ladder` -- the network is trained on the labels rather than
    # probed on frozen features -- but it is scored on the SAME held-out participants, in the
    # same order (participant_aggregate and per_subject both sort by np.unique(pids[test])), so
    # its Delta against CoST is paired exactly like every other rung. The gap between CoST and
    # this row is what freezing the encoder costs, which is the first thing a reader asks.
    sup_name = "Supervised (end-to-end)"
    if not a.no_supervised:
        try:
            _row, pp_s, pl_s = supervised_baseline_row(
                ctx.X, ctx.y, ctx.pids, ctx.train_mask, ctx.val_mask, ctx.test_mask,
                ctx.cfg["backbone"], ctx.cfg["pe"], sup_name,
                int(ctx.X.shape[-1]) - int(ctx.n_sensors), ctx.cfg["hidden_dims"],
                ctx.cfg["depth"], ctx.cfg["repr_dims"], device=ctx.device, seed=ctx.seed,
                batch_size=ctx.cfg["batch_size"], return_scores=True)
            auc_s = float(_row["Subj AUC"])
            probs[sup_name] = (np.asarray(pp_s), np.asarray(pl_s))
            add("ladder", sup_name, auc_s, np.asarray(pp_s), np.asarray(pl_s))
        except Exception as e:
            print(f"[rq3] {sup_name} rung SKIPPED: {type(e).__name__}: {e}")

    res["utility"] = {r[1]: {"auc": r[2], "ci": r[3:]} for r in rows}
    res["majority_rate"] = maj

    # Delta AUC vs each baseline, bootstrapped over the SAME participants (paired).
    pp_c, pl_c = probs["DSSL (frozen)"]
    res["delta_auc"] = {}
    # vs plain SSL is the headline contrast: it is the only rung differing ONLY by the split.
    for name in [n for n in probs if n != "DSSL (frozen)"]:
        pp_b, _ = probs[name]
        boot = []
        for _ in range(a.n_boot):
            ix = rng.integers(0, len(pl_c), len(pl_c))
            if len(set(pl_c[ix])) < 2:
                continue
            boot.append(roc_auc_score(pl_c[ix], pp_c[ix]) - roc_auc_score(pl_c[ix], pp_b[ix]))
        # The POINT estimate is the observed statistic. np.mean(boot) is the bootstrap mean,
        # which estimates its expectation and differs from it by the bootstrap bias (measured
        # at -0.0037 on a 36-participant example). The bootstrap supplies the interval only --
        # the rule experiment_q2.py::detection_stats already follows.
        res["delta_auc"][name] = {
            "delta": float(roc_auc_score(pl_c, pp_c) - roc_auc_score(pl_c, pp_b)),
            "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if boot else None}

    # --- Part B: how? branch / channel / timescale ---------------------------------------
    n_ladder = len(rows)
    for half, dslice in (("V^T (trend) only", slice(None, dT)),
                         ("V^S (seasonal) only", slice(dT, None))):
        add("branch", half, *score(fit_probe(V[:, dslice], ctx, a), V[:, dslice], ctx))
    add("branch", "full", *((res["utility"]["DSSL (frozen)"]["auc"],) + probs["DSSL (frozen)"]))
    clf = fit_probe(V, ctx, a)                    # probe trained on the intact input, reused
    for c, cname in enumerate(ctx.sensor_cols):
        Xz = ctx.X.copy(); Xz[:, :, c] = 0.0
        add("channel", f"drop {cname}", *score(clf, encode(ctx.model, Xz, ctx.cfg), ctx))
    Xnc = degrade(ctx.X, "no_circadian", 0, ctx.n_sensors, sig, ctx.bins_per_day,
                  ctx.bin_minutes, rng)
    add("timescale", "input minus circadian",
        *score(clf, encode(ctx.model, Xnc, ctx.cfg), ctx))
    res["ablation"] = [{"level": r[0], "setting": r[1], "auc": r[2], "ci": r[3:]}
                       for r in rows[n_ladder:]]

    # ONE table. `role` says which question a row answers: 'ladder' is Part A (does it help?),
    # the rest are Part B (how?). They were two files with the same columns and only one set of
    # intervals, which invited reading an ablation AUC as if it carried the uncertainty the
    # ladder rows do.
    write_csv(d, "rq3_utility",
              ["role", "representation", "auc", "ci_lo", "ci_hi", "balanced_acc"], rows)

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
    full = res["utility"]["DSSL (frozen)"]["auc"]
    prob_full, yte = probs["DSSL (frozen)"]
    P, curves = {}, {}
    # The generator is derived PER CELL, never taken from the shared stream. Part A's delta
    # loop draws n_boot index vectors per rung, and the number of rungs is not fixed -- the
    # cosinor and supervised rungs sit inside try blocks, two flags remove rungs, and the plain
    # twin is added only for the reference variants. A shared advancing stream therefore hands
    # Part C a different degradation mask depending on how many rungs happened to run: verified
    # at a four- versus five-rung ladder on the same seed, 32.0% of the MCAR bins differed.
    # Two variants would then be compared on different masks, which is a confound unrelated to
    # the variant. Deriving per cell makes every cell reproduce on its own. (Same rule
    # experiment_q2.py already applies to its perturbation grid.)
    for ki, (kind, levels) in enumerate(grid.items()):
        for lv in levels:
            Xd = degrade(ctx.X, kind, lv, ctx.n_sensors, sig, ctx.bins_per_day,
                         ctx.bin_minutes,
                         np.random.default_rng([ctx.seed, ki, int(lv * 1000)]))
            auc, P[(kind, lv)], _ = score(clf, encode(ctx.model, Xd, ctx.cfg), ctx)
            curves.setdefault(kind, []).append((lv, auc))

    # Participant bootstrap of the whole grid. ONE draw is shared by the intact reference and
    # every degraded level, so each draw is a coherent curve and c* can be recomputed on it --
    # the only way c* gets an interval rather than being a point read off a noisy curve.
    _brng = np.random.default_rng([ctx.seed, 7])          # own stream, for the same reason
    bidx = [_brng.integers(0, len(yte), len(yte)) for _ in range(a.n_boot)]
    bauc = lambda p, i: fast_auc(yte[i], p[i])       # NaN when a draw is single-class
    ci = lambda v: ([float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                    if len(v) >= 20 else [np.nan, np.nan])
    bfull = np.array([bauc(prob_full, i) for i in bidx])
    bcurve = {k: np.array([bauc(P[k], i) for i in bidx]) for k in P}
    band = {k: ci(bcurve[k]) for k in P}
    # DROP is a design constant; whether a drop that size is MEASURABLE is a property of the
    # sample. Each bootstrap draw shares one set of participants between the intact reference
    # and every degraded level, so the spread of (intact - degraded) across draws is exactly
    # the standard error of the quantity c* thresholds. 1.96 of them is the smallest drop this
    # run can tell apart from zero -- if that exceeds DROP, c* is being read off noise.
    sd = [float(np.nanstd(bfull - bcurve[k])) for k in P]
    mde = float(1.96 * np.nanmedian(sd))
    res["min_resolvable_drop"] = round(mde, 4)

    write_csv(d, "rq3_degradation", ["factor", "level", "auc", "ci_lo", "ci_hi"],
              [[k, lv, round(auc, 4) if np.isfinite(auc) else "",
                round(band[(k, lv)][0], 4), round(band[(k, lv)][1], 4)]
               for k, pts in curves.items() for lv, auc in pts])

    res["auc_full"] = full
    res["n_test_participants"] = int(len(np.unique(ctx.pids[te])))
    # c* only means something if there is performance to lose: at chance level every
    # degradation trivially sits "within 0.05", which would read as perfect robustness.
    res["breakdown_valid"] = bool(np.isfinite(full) and full >= 0.55 + DROP)
    res["breakdown_point"], res["breakdown_ci"] = {}, {}
    for kind, pts in curves.items():
        levels = [lv for lv, _ in pts]
        if not res["breakdown_valid"]:
            res["breakdown_point"][kind] = None
            res["breakdown_ci"][kind] = {"reason": "intact AUC too close to chance"}
            continue
        b = [cstar([(lv, bcurve[(kind, lv)][j]) for lv, _ in pts], kind, f)
             for j, f in enumerate(bfull)]
        got = [v for v in b if v is not None]
        lo, hi = ci(got)
        undef = float(np.mean([v is None for v in b]))
        # c* names one level of the grid. An interval that still covers half the grid does not
        # name it, and a c* that fails to exist in a tenth of the draws is not a stable
        # quantity -- in both cases the point estimate is the first crossing of a noisy curve,
        # so it is withheld rather than drawn as if it were determined.
        span = float(np.mean([lo <= lv <= hi for lv in levels])) if np.isfinite(lo) else 1.0
        ok = bool(span < 0.5 and undef < 0.10)
        res["breakdown_point"][kind] = cstar(pts, kind, full) if ok else None
        res["breakdown_ci"][kind] = {
            "ci": [lo, hi], "frac_undefined": round(undef, 4),
            "grid_fraction_covered": round(span, 3), "resolvable": ok,
            "reason": "" if ok else (f"interval covers {span:.0%} of the grid"
                                     if span >= 0.5 else
                                     f"undefined in {undef:.0%} of bootstrap draws")}
    print(f"[rq3] intact AUC {full:.3f}; smallest resolvable drop {mde:.3f} "
          f"(design threshold {DROP}); c* {res['breakdown_point']}")

    fci = res["utility"]["DSSL (frozen)"].get("ci") or [float("nan")] * 2
    fig, axes = plt.subplots(1, len(curves), figsize=(3.1 * len(curves), 3.9), squeeze=False)
    for ax, (kind, pts) in zip(axes[0], curves.items()):
        xs, ys = zip(*pts)
        c, bd = res["breakdown_point"][kind], res["breakdown_ci"][kind]
        if c is not None:                      # the INTERVAL, not just the point estimate
            ax.axvspan(bd["ci"][0], bd["ci"][1], color="#009E73", alpha=0.13, lw=0)
            ax.axvline(c, ls=":", c="#009E73", lw=1.8)
        else:                                  # say why, instead of leaving a bare curve that
            ax.set_facecolor("#f2f2f2")        # reads as "robust to everything"
            ax.text(.5, .04, "c* not reported\n" + (bd.get("reason") or ""),
                    transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5,
                    color="#b30000", linespacing=1.3)
        ax.plot(xs, ys, "o-", color="#0072B2", zorder=3)
        ax.fill_between(xs, [band[(kind, lv)][0] for lv in xs],
                        [band[(kind, lv)][1] for lv in xs], color="#0072B2", alpha=0.15, lw=0)
        ax.axhline(0.5, ls="-", c="#999999", lw=0.8)          # chance, always in view
        ax.axhline(full, ls="-", c="grey", lw=0.9)
        ax.axhline(full - DROP, ls="--", c="#D55E00", lw=0.9)
        ax.axhline(full - mde, ls="--", c="#7a1a1a", lw=0.9)  # what the sample can resolve
        ax.set_xlabel(kind); ax.grid(alpha=0.25); ax.set_ylim(0.3, 1.02)
    axes[0][0].set_ylabel("participant AUROC")
    fig.suptitle(
        f"RQ3 operating envelope -- DSSL (frozen), {ctx.tag}\n"
        f"intact AUROC {full:.3f} [{fci[0]:.3f}, {fci[1]:.3f}] on "
        f"{res['n_test_participants']} held-out participants\n"
        f"orange dashed = design threshold $-${DROP};  dark dashed = smallest drop this "
        f"sample resolves, $-${mde:.3f}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(d / "rq3_degradation.png", dpi=200); plt.close(fig)

    save(d, "rq3", res)


if __name__ == "__main__":
    main()
