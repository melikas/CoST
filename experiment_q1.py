"""RQ1 -- Representation validity of SSL, and the effect of the temporal reference frame.

Runs on a FINISHED train_hrd.py variant directory: loads the frozen encoder, never trains.

  E1.3  HEADLINE    per-CHANNEL cosinor amplitude / acrophase / IS read out of the latent, each
                    against the same probe on a raw-window PCA (`gain_over_raw`)
                    -> tasks/rhythm.py::rhythm_axis_probe
  E1.2  diagnostic  Ridge R2( [V^(T);V^(S)] -> tau / sigma ), penalty selected on validation
                    -> tasks/decomposition.py::run_decomposition_recovery (full DRS report)

E1.3 is the headline and E1.2 is not, because tau/sigma are a FIXED LINEAR projection of the
input (one design matrix for every window, decomposition.py::harmonic_reference). Recovering
them therefore tests whether the representation preserves a linear subspace of the input --
informative, but not "does it encode a biobehavioral construct". Amplitude, acrophase and
interdaily stability are NONLINEAR functionals of the signal and are the established
chronobiological constructs RQ1 actually asks about.
  E1.5  controls    random-init encoder, time-permuted input scored against the ORIGINAL
                    reference, and the permutation-invariant floor -- without these the R2 has
                    no scale.

E1.4 (Delta_pi across PE families) is a CROSS-variant contrast and belongs to the collector;
this script emits the per-variant row it needs.

  python experiment_q1.py --variant-dir results_hrd/<run>/<backbone>_<pe>_seed<S>
"""
import argparse
import json
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from baselines.plain_ssl import plain_ssl_encoder
from tasks._experiment_common import (encode, load_context, out_dir, random_init_model, save,
                                      wants_plain_ssl)
from tasks.decomposition import (RIDGE_ALPHAS, _probe_r2, _selection_split, _var_weights,
                                 extract_components, harmonic_reference,
                                 run_decomposition_recovery)


def _plain_components(model, X, stride, batch_size=128):
    """Plain SSL emits ONE sequence (cost.py:536 returns season=None), so V^(F) is that
    sequence. There are no branches, hence no own/leak terms and no DIS -- headline only."""
    idx = np.arange(0, X.shape[1], stride)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            z, _ = model.net(torch.from_numpy(X[i:i + batch_size]).float().to(model.device))
            out.append(z[:, idx].cpu().numpy())
    return np.concatenate(out), idx


def time_shuffled_input(X, n_sensors, rng, per_channel=False):
    """Destroy each window's time ordering, with an INDEPENDENT permutation per window.

    A single permutation shared by every window is not a null. Writing p for that permutation
    and x^shuf_i(s) = x_i(p[s]), the reference's harmonic coefficients obey

        b_ik = (2/T) sum_t x_i(t) cos(k w t) = (2/T) sum_s x^shuf_i(s) cos(k w p[s]),

    and cos(k w p[s]) is one fixed vector for ALL windows -- so sigma stays an exact fixed
    LINEAR functional of the shuffled input and a ridge probe recovers it at R2 ~ 1. The
    control would then measure only how far off-distribution the encoder was pushed. Drawing
    p_i per window makes that vector window-dependent, and no single linear map can undo it.

    The permutation is shared across sensor CHANNELS by default: that destroys the temporal
    ordering while leaving each channel's marginal distribution and the instantaneous
    cross-channel coupling intact, so the control isolates time ordering alone.
    `per_channel=True` also breaks the coupling -- a stronger null that confounds the two.

    Clock/calendar channels are never permuted: they carry the temporal reference frame, and
    shuffling them would turn this control into a PE ablation.
    """
    N, T = X.shape[0], X.shape[1]
    out = X.copy()
    if per_channel:
        for c in range(n_sensors):
            p = np.argsort(rng.random((N, T)), axis=1)
            out[:, :, c] = np.take_along_axis(X[:, :, c], p, axis=1)
    else:
        p = np.argsort(rng.random((N, T)), axis=1)           # (N, T) independent permutations
        out[:, :, :n_sensors] = np.take_along_axis(X[:, :, :n_sensors], p[:, :, None], axis=1)
    return out


def perm_invariant_floor(Xs, tau, sig, fit, sel, test, stride=4):
    """The R2 a CORRECT shuffle control cannot fall below, measured rather than assumed.

    Some of the reference survives ANY permutation: the window spans 7 whole 24 h periods, so
    the harmonic columns of the design matrix are orthogonal to the constant one and
    mean_t(tau) == mean_t(x) exactly. The window mean is permutation-invariant, hence Full->tau
    can never reach 0 and "R2 -> 0" is the wrong expectation for the trend half.

    Features are per-channel window mean and std, held constant across timesteps -- low-order
    moments of the multiset a permutation leaves untouched. That makes this a CONSERVATIVE
    floor (the shuffled multiset carries more than two moments), which is the useful direction:
    a time_shuffled score at or below it is consistent with the encoder reading nothing but
    permutation-invariant statistics.
    """
    idx = np.arange(0, Xs.shape[1], stride)
    F = np.concatenate([Xs.mean(1), Xs.std(1)], axis=1).astype(np.float32)   # (N, 2C)
    F = np.repeat(F[:, None, :], len(idx), axis=1)                           # (N, T', 2C)
    t, s = tau[:, idx], sig[:, idx]
    rT, _ = _probe_r2(F, t, fit, sel, test, RIDGE_ALPHAS)
    rS, _ = _probe_r2(F, s, fit, sel, test, RIDGE_ALPHAS)
    return float(_var_weights(t, test) @ rT), float(_var_weights(s, test) @ rS)


def headline(model, X_feat, tau, sig, fit, sel, test, stride=4, plain=False, groups=None):
    """The two headline numbers only: Full->tau, Full->sigma. Features come from X_feat, the
    reference from tau/sig -- which lets the shuffled control keep the ORIGINAL reference.

    With `groups` (pids) it also returns per-participant sufficient statistics and the fixed
    channel weights, which is what lets collect_results.py::pe_contrast bootstrap Delta_pi
    over PARTICIPANTS. The weights are held fixed at their full-test-set value on purpose:
    they are a declared weighting scheme, not a quantity being estimated.
    """
    if plain:
        VF, idx = _plain_components(model, X_feat, stride)
    else:
        VT, VS, idx = extract_components(model, X_feat, time_stride=stride)
        VF = np.concatenate([VT, VS], axis=-1)
    t, s = tau[:, idx], sig[:, idx]
    rT = _probe_r2(VF, t, fit, sel, test, RIDGE_ALPHAS, groups)
    rS = _probe_r2(VF, s, fit, sel, test, RIDGE_ALPHAS, groups)
    wT, wS = _var_weights(t, test), _var_weights(s, test)
    out = float(wT @ rT[0]), float(wS @ rS[0])
    if groups is None:
        return out
    return out, {"weights": {"tau": wT.tolist(), "sigma": wS.tolist()},
                 "per_participant": {"tau": rT[2], "sigma": rS[2]}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant-dir", required=True)
    p.add_argument("--cache-dir", default=None, help="Scratch dir for the windowed-dataset cache")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--skip-controls", action="store_true")
    # There is deliberately NO --skip-axis-probe. E1.3 is the only part of RQ1 that touches an
    # established chronobiological construct (see the E1.3 block below); a run without it does
    # not answer RQ1, so it must not be skippable for convenience.
    p.add_argument("--allow-missing-cosinor", action="store_true",
                   help="Only for an environment that genuinely cannot install CosinorPy: "
                        "downgrade the missing-dependency ABORT to a recorded, clearly flagged "
                        "omission in rq1.json. Never hides any other failure.")
    p.add_argument("--force-drs", action="store_true",
                   help="Recompute the DRS even when train_hrd.py already wrote one")
    p.add_argument("--n-shuffle", type=int, default=3,
                   help="Draws of the time-shuffled control to average. One draw is a single "
                        "sample of a RANDOM control and carries no error bar; each extra draw "
                        "costs one encode + two ridge fits. Pass 1 for the old cost.")
    p.add_argument("--shuffle-per-channel", action="store_true",
                   help="Permute every sensor channel independently. The default shares one "
                        "permutation across channels within a window (see time_shuffled_input).")
    p.add_argument("--no-plain-ssl", action="store_true",
                   help="Drop the Ctrl plain-SSL row. It is the ONLY control that costs a "
                        "real pretraining (~1 extra per seed x variant, cached for q3).")
    a = p.parse_args()

    ctx = load_context(a.variant_dir, a.cache_dir, a.gpu)
    d = out_dir(ctx, "rq1")
    res = {"variant": ctx.tag, "seed": ctx.seed}

    # --- E1.2 headline + full DRS report (own-branch / leak / DIS) -----------------------
    # train_hrd.py already runs this with the SAME masks and the same validation-selected
    # penalty unless --no-rhythm-viz was passed, so reuse it rather than pay for it twice.
    prev = ctx.variant_dir / "decomposition_recovery.json"
    if prev.exists() and not a.force_drs:
        agg = json.loads(prev.read_text(encoding="utf-8"))
        print(f"[rq1] reusing the DRS train_hrd.py already wrote -> {prev.name}")
    else:
        agg = run_decomposition_recovery(
            ctx.model, ctx.X, ctx.train_mask, ctx.test_mask, d,
            seq_len=ctx.seq_len, bin_minutes=ctx.bin_minutes, sensor_cols=ctx.sensor_cols,
            seed=ctx.seed, val_mask=ctx.val_mask, pids=ctx.pids)
    res["decomposition"] = agg
    print(f"[rq1] Full->tau={agg['rec_full_trend']:.3f} Full->sigma={agg['rec_full_rhythm']:.3f} "
          f"DIS={agg['DIS']:.3f}")

    # --- E1.5 negative controls ----------------------------------------------------------
    if not a.skip_controls:
        tau, sig = harmonic_reference(ctx.X[:, :, :ctx.n_sensors],
                                      24 * 60.0 / ctx.bin_minutes)
        fit, sel, src = _selection_split(ctx.train_mask, ctx.val_mask, ctx.pids, ctx.seed)

        # Only the trained row carries per-participant statistics: it is the one the PE
        # contrast (E1.4) resamples. The controls need a point estimate only.
        trained, boot = headline(ctx.model, ctx.X, tau, sig, fit, sel, ctx.test_mask,
                                 groups=ctx.pids)
        res["bootstrap"] = boot
        res["controls"] = {
            "trained": dict(zip(("Full->tau", "Full->sigma"), trained)),
            "random_init": dict(zip(("Full->tau", "Full->sigma"),
                                    headline(random_init_model(ctx), ctx.X, tau, sig,
                                             fit, sel, ctx.test_mask))),
            "selection_split": src,
        }

        # Own RNG stream, so changing --n-shuffle never moves any other control's randomness.
        rng = np.random.default_rng(ctx.seed + 9973)
        draws = np.asarray([
            headline(ctx.model,
                     time_shuffled_input(ctx.X, ctx.n_sensors, rng, a.shuffle_per_channel),
                     tau, sig, fit, sel, ctx.test_mask)
            for _ in range(max(1, a.n_shuffle))])                    # (n_draws, 2)
        res["controls"]["time_shuffled"] = {
            "Full->tau": float(draws[:, 0].mean()), "Full->sigma": float(draws[:, 1].mean()),
            "sd": {"Full->tau": float(draws[:, 0].std()),
                   "Full->sigma": float(draws[:, 1].std())},
            "n_draws": int(len(draws)),
            "scheme": "per-channel" if a.shuffle_per_channel else "per-window"}
        # Read time_shuffled against THIS, never against 0 (see perm_invariant_floor).
        res["controls"]["perm_invariant_floor"] = dict(zip(
            ("Full->tau", "Full->sigma"),
            perm_invariant_floor(ctx.X[:, :, :ctx.n_sensors], tau, sig,
                                 fit, sel, ctx.test_mask)))
        if wants_plain_ssl(ctx) and not a.no_plain_ssl:
            # Ctrl family: same SSL, disentangler OFF. Headline recovery only -- with one
            # branch there is nothing to leak between, so DIS is undefined, not zero.
            plain = plain_ssl_encoder(ctx.X, ctx.pretrain_mask, ctx.cfg, ctx.n_sensors,
                                      ctx.device, seed=ctx.seed,
                                      cache_path=ctx.variant_dir / "plain_encoder.pt")
            res["controls"]["plain_ssl"] = dict(zip(
                ("Full->tau", "Full->sigma"),
                headline(plain, ctx.X, tau, sig, fit, sel, ctx.test_mask, plain=True)))
            res["controls"]["plain_ssl"]["DIS"] = None
        c = res["controls"]
        print("[rq1] controls " + " | ".join(
            f"{k}: tau={v['Full->tau']:.3f} sig={v['Full->sigma']:.3f}"
            for k, v in c.items() if isinstance(v, dict)))

        names = [n for n in ("trained", "plain_ssl", "random_init", "time_shuffled") if n in c]
        x, w = np.arange(len(names)), 0.38
        fig, ax = plt.subplots(figsize=(1.5 * len(names) + 2, 3.8))
        for i, (key, col) in enumerate((("Full->tau", "#0072B2"), ("Full->sigma", "#009E73"))):
            ax.bar(x + (i - 0.5) * w, [c[n][key] for n in names], w,
                   yerr=[c[n].get("sd", {}).get(key, 0.0) for n in names], capsize=3,
                   color=col, label=key)
            # The floor a permutation cannot breach -- the honest zero for the bar beside it.
            ax.axhline(c["perm_invariant_floor"][key], ls="--", lw=1.0, color=col, alpha=0.85,
                       label="perm-invariant floor" if i == 0 else None)
        ax.set_xticks(x); ax.set_xticklabels([n.replace("_", " ") for n in names])
        ax.set_ylabel("held-out $R^2$"); ax.grid(axis="y", alpha=0.25)
        # R^2 is unclipped (tasks/decomposition.py::_probe_r2), so allow negative bars.
        _lo = min([0.0] + [c[n][k] for n in names for k in ("Full->tau", "Full->sigma")])
        ax.set_ylim(min(-0.05, _lo * 1.1), 1)
        ax.set_title(f"RQ1 recovery vs negative controls\n{ctx.tag}", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(d / "rq1_controls.png", dpi=200); plt.close(fig)

    # --- E1.3 person-level chronobiology tether -------------------------------------------
    # Runs LAST on purpose: it is the slow part and the only one with an external dependency,
    # so E1.2 and E1.5 are already computed and can be banked before anything here can fail.
    #
    # E1.2's targets tau/sigma are a FIXED linear projection of the input, so Full->tau/sigma
    # measures linear information preservation. Amplitude / acrophase / interdaily stability
    # are NONLINEAR functionals of the signal and are the established chronobiological
    # constructs RQ1 actually asks about -- which makes this block the construct-valid half of
    # RQ1, not an optional extra.
    #
    # Nothing here is swallowed. NARVAL_globem.md documents three sweeps (runs 66404249 /
    # 66440129 / 66465766, 325 variants of GPU time) that trained to completion and reported
    # success while a caught ModuleNotFoundError silently dropped the cosinor baseline; the
    # loss surfaced only as `n/a` in the summary, long after the compute was spent. So every
    # outcome below carries an explicit `status`, and an unexpected failure is recorded, saved
    # and re-raised rather than turned into an empty dict.
    try:
        from baselines.cosinor import paper_cosinor_features
        from tasks.rhythm import rhythm_axis_probe
    except ImportError as e:
        res["axis_probe"] = {"status": "SKIPPED_NO_COSINORPY", "error": f"{type(e).__name__}: {e}"}
        save(d, "rq1", res)                      # E1.2 + E1.5 are real; keep them either way
        if not a.allow_missing_cosinor:
            raise SystemExit(
                f"[rq1] FATAL: E1.3 needs CosinorPy and it is not importable ({e}).\n"
                f"       RQ1 has no construct-valid result without it, so this is an abort,\n"
                f"       not a skip. Fix the environment (`pip install CosinorPy seaborn`, and\n"
                f"       check baselines/cosinor.py was uploaded), or pass\n"
                f"       --allow-missing-cosinor to record the omission and exit 0.")
        print("[rq1] !! E1.3 SKIPPED: CosinorPy not importable -- RQ1 is INCOMPLETE for this "
              "variant (status SKIPPED_NO_COSINORPY in rq1.json).")
    else:
        Xs = ctx.X[:, :, :ctx.n_sensors]
        try:
            cf = paper_cosinor_features(Xs, ctx.bin_minutes, need_mask=ctx.test_mask,
                                        window_ids=ctx.window_ids, pids=ctx.pids,
                                        cache_path=d / "cosinor_cache.npz")
            out = rhythm_axis_probe(
                encode(ctx.model, ctx.X, ctx.cfg), Xs, ctx.test_mask, ctx.pids,
                ctx.bin_minutes, d, ctx.seed, cf, ctx.n_sensors, table_tag="rq1",
                sensor_cols=ctx.sensor_cols)
            # E1.3 against the E1.5 control, not only against raw PCA. `gain_over_raw` asks
            # "does the encoder beat the raw window", which cannot separate what PRETRAINING
            # adds from what the untrained architecture already provides. On E1.2 the
            # random-init encoder MATCHES or beats the trained one, so that separation is the
            # whole question: if random-init also recovers IS and acrophase, E1.3 is a fact
            # about a wide frozen convolutional map, not about the SSL objective.
            # Same markers, same participant-grouped folds, same cosinor fit -- `cf` is reused,
            # so the control costs one encode and no cosinor time.
            out_ri = rhythm_axis_probe(
                encode(random_init_model(ctx), ctx.X, ctx.cfg), Xs, ctx.test_mask, ctx.pids,
                ctx.bin_minutes, d, ctx.seed, cf, ctx.n_sensors, table_tag="rq1_randinit",
                sensor_cols=ctx.sensor_cols)
        except Exception:
            res["axis_probe"] = {"status": "FAILED", "traceback": traceback.format_exc()}
            save(d, "rq1", res)
            raise
        # rhythm_axis_probe returns {} when there are <3 participant groups or <20 windows
        # (rhythm.py:1295). "could not be estimated" is not "estimated and found nothing", so
        # the two are named apart instead of both arriving as an empty dict.
        res["axis_probe"] = (dict(out, status="OK") if out else
                             {"status": "EMPTY_TOO_FEW_PARTICIPANTS",
                              "n_test_participants": int(len(np.unique(ctx.pids[ctx.test_mask]))),
                              "n_test_windows": int(ctx.test_mask.sum())})
        res["axis_probe_random_init"] = (dict(out_ri, status="OK") if out_ri else
                                         {"status": "EMPTY_TOO_FEW_PARTICIPANTS"})
        ap = res["axis_probe"]
        g = [v["gain_over_raw"] for v in ap.values()
             if isinstance(v, dict) and "gain_over_raw" in v]
        # Paired per marker, so it answers "what did PRETRAINING add" rather than
        # "is the latent better than raw". Same marker set as `g` by construction.
        gr = [ap[k]["value"] - out_ri[k]["value"] for k, v in ap.items()
              if isinstance(v, dict) and "gain_over_raw" in v and k in out_ri]
        res["headline"] = {
            "acrophase_median_abs_err_hours":
                ap.get("acrophase | median over channels", {}).get("value"),
            "median_gain_over_raw_pca": float(np.median(g)) if g else None,
            "frac_markers_beating_raw": float(np.mean(np.asarray(g) > 0)) if g else None,
            "median_gain_over_random_init": float(np.median(gr)) if gr else None,
            "frac_markers_beating_random_init":
                float(np.mean(np.asarray(gr) > 0)) if gr else None,
            "n_markers_vs_random_init": len(gr)}
        print(f"[rq1] {ap['status']} | HEADLINE E1.3 acrophase err "
              f"{res['headline']['acrophase_median_abs_err_hours']} h, median gain over raw "
              f"PCA {res['headline']['median_gain_over_raw_pca']}")
        print(f"[rq1] E1.3 vs RANDOM-INIT (what pretraining adds): median gain "
              f"{res['headline']['median_gain_over_random_init']}, beating on "
              f"{res['headline']['frac_markers_beating_random_init']} of "
              f"{res['headline']['n_markers_vs_random_init']} markers")

    save(d, "rq1", res)


if __name__ == "__main__":
    main()
