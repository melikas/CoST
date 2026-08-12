"""RQ1 -- Representation validity of SSL, and the effect of the temporal reference frame.

Runs on a FINISHED train_hrd.py variant directory: loads the frozen encoder, never trains.

  E1.2  headline    Ridge R2( [V^(T);V^(S)] -> tau / sigma ), penalty selected on validation
                    -> tasks/decomposition.py::run_decomposition_recovery (full DRS report)
  E1.3  constructs  per-participant cosinor amplitude / IS / acrophase read out of the latent
                    -> tasks/rhythm.py::rhythm_axis_probe
  E1.5  controls    random-init encoder, and time-permuted input scored against the ORIGINAL
                    reference -- without these the R2 has no scale.

E1.4 (Delta_pi across PE families) is a CROSS-variant contrast and belongs to the collector;
this script emits the per-variant row it needs.

  python experiment_q1.py --variant-dir results_hrd/<run>/<backbone>_<pe>_seed<S>
"""
import argparse
import json

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


def headline(model, X_feat, tau, sig, fit, sel, test, stride=4, plain=False):
    """The two headline numbers only: Full->tau, Full->sigma. Features come from X_feat, the
    reference from tau/sig -- which lets the shuffled control keep the ORIGINAL reference."""
    if plain:
        VF, idx = _plain_components(model, X_feat, stride)
    else:
        VT, VS, idx = extract_components(model, X_feat, time_stride=stride)
        VF = np.concatenate([VT, VS], axis=-1)
    t, s = tau[:, idx], sig[:, idx]
    rT, _ = _probe_r2(VF, t, fit, sel, test, RIDGE_ALPHAS)
    rS, _ = _probe_r2(VF, s, fit, sel, test, RIDGE_ALPHAS)
    return float(_var_weights(t, test) @ rT), float(_var_weights(s, test) @ rS)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant-dir", required=True)
    p.add_argument("--cache-dir", default=None, help="Scratch dir for the windowed-dataset cache")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--skip-controls", action="store_true")
    p.add_argument("--skip-axis-probe", action="store_true",
                   help="Skip E1.3 (needs CosinorPy and is the slow part)")
    p.add_argument("--force-drs", action="store_true",
                   help="Recompute the DRS even when train_hrd.py already wrote one")
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

    # --- E1.3 person-level chronobiology tether -----------------------------------------
    if not a.skip_axis_probe:
        try:
            from baselines.cosinor import paper_cosinor_features
            from tasks.rhythm import rhythm_axis_probe
            Xs = ctx.X[:, :, :ctx.n_sensors]
            cf = paper_cosinor_features(Xs, ctx.bin_minutes, need_mask=ctx.test_mask,
                                        window_ids=ctx.window_ids, pids=ctx.pids,
                                        cache_path=d / "cosinor_cache.npz")
            res["axis_probe"] = rhythm_axis_probe(
                encode(ctx.model, ctx.X, ctx.cfg), Xs, ctx.test_mask, ctx.pids,
                ctx.bin_minutes, d, ctx.seed, cf, ctx.n_sensors, table_tag="rq1")
        except Exception as e:                       # CosinorPy is optional in the wheelhouse
            print(f"[rq1] axis probe skipped ({type(e).__name__}: {e})")
            res["axis_probe"] = {}

    # --- E1.5 negative controls ----------------------------------------------------------
    if not a.skip_controls:
        tau, sig = harmonic_reference(ctx.X[:, :, :ctx.n_sensors],
                                      24 * 60.0 / ctx.bin_minutes)
        fit, sel, src = _selection_split(ctx.train_mask, ctx.val_mask, ctx.pids, ctx.seed)
        rng = np.random.default_rng(ctx.seed)
        X_shuf = ctx.X.copy()
        X_shuf[:, :, :ctx.n_sensors] = ctx.X[:, rng.permutation(ctx.seq_len), :ctx.n_sensors]

        res["controls"] = {
            "trained": dict(zip(("Full->tau", "Full->sigma"),
                                headline(ctx.model, ctx.X, tau, sig, fit, sel, ctx.test_mask))),
            "random_init": dict(zip(("Full->tau", "Full->sigma"),
                                    headline(random_init_model(ctx), ctx.X, tau, sig,
                                             fit, sel, ctx.test_mask))),
            "time_shuffled": dict(zip(("Full->tau", "Full->sigma"),
                                      headline(ctx.model, X_shuf, tau, sig,
                                               fit, sel, ctx.test_mask))),
            "selection_split": src,
        }
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
            ax.bar(x + (i - 0.5) * w, [c[n][key] for n in names], w, color=col, label=key)
        ax.set_xticks(x); ax.set_xticklabels([n.replace("_", " ") for n in names])
        ax.set_ylabel("held-out $R^2$"); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.25)
        ax.set_title(f"RQ1 recovery vs negative controls\n{ctx.tag}", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(d / "rq1_controls.png", dpi=200); plt.close(fig)

    save(d, "rq1", res)


if __name__ == "__main__":
    main()
