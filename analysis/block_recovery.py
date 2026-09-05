"""RQ1, split into the two halves of the seasonal component that behave oppositely.

RQ1's headline is that pretraining makes the representation recover the decomposition WORSE
(trend recovery 0.6830 -> 0.5294, seasonal 0.9314 -> 0.8959). Those are measured on the
composed representation, and three other measurements now say that number is the average of
two effects pulling in opposite directions:

  RQ2, 24 seeds       V^S phase C = 0.8802     V^S amp C = 0.5661
  block ablation      with the phase block +0.0089, with it deleted -0.0545
  pretext difficulty  the amplitude task sits at 1.4x chance, the phase task at 17x

A quantity that averages an improving block with a degrading one is not a fact about either.

The seasonal component's amplitude and phase are per-window scalars, so they are the level
at which the split is well posed -- V^S itself is a per-timestep sequence and has no
amplitude or phase until something takes an FFT of it. So: fit the same harmonic reference
RQ1 already uses, take the true seasonal component's amplitude and phase at each circadian
harmonic, and ask how well a ridge on the representation recovers each.

Phase is recovered as (cos, sin), never as the angle. 23.5 h and 0.5 h are one hour apart
and average to 12.0, the opposite time of day, so an R2 on raw angles measures the branch cut
as much as the phase.

The prediction, made before running: DSSL beats its untrained control on PHASE recovery and
loses to it on AMPLITUDE recovery. If that holds, RQ1's claim changes from "pretraining
degrades the representation" to "pretraining builds phase and discards amplitude, because
the augmentation preserves one and destroys the other" -- which is a mechanism, and points
at the positive pair rather than at the loss.

    SCRIPT=analysis/block_recovery.py NEED_ENC=1 sbatch --array=0-23%24 \\
      scripts/stability_gate.sh results_hrd/<run>
    python analysis/block_recovery.py --aggregate results_hrd/<run>
"""
import sys
from pathlib import Path

# Run as `python analysis/<name>.py` from the repository root: the interpreter puts
# this file's own directory on sys.path, not the project root, so the shared modules
# would not import. scripts/ already does this; the pattern is the same.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import glob
import json

import numpy as np

NAME = "block_recovery"


def seasonal_targets(Xs, bins_per_day, harmonics=(1, 2, 3, 4)):
    """(amplitude, phase) of the TRUE seasonal component, per window, per channel.

    The reference is the same closed-form fit RQ1 uses, so the target here is the same
    seasonal component that report calls sigma -- not a second definition of it.
    """
    from tasks.decomposition import harmonic_reference
    _, sig = harmonic_reference(np.asarray(Xs, dtype=np.float32), int(bins_per_day))
    Z = np.fft.rfft(sig, axis=1)                       # (N, F, C)
    D = max(1, sig.shape[1] // int(bins_per_day))      # bin of the 24 h cycle
    f = [h * D for h in harmonics if h * D < Z.shape[1]]
    z = Z[:, f]                                        # (N, |f|, C)
    A = np.abs(z)
    # A harmonic with no amplitude has no phase -- its angle is uniform noise, and every
    # such column dilutes the phase target while looking like a legitimate one. Measured on
    # a planted signal with energy only at the fundamental, keeping all four harmonics put
    # a perfect phase feature at R2 0.44 instead of ~1. Columns are kept only where the
    # median amplitude is a tenth of the strongest harmonic's, which is a property of the
    # DATA and not of any representation, so it cannot favour an arm.
    med = np.median(A, axis=0)                         # (|f|, C)
    keep = med >= 0.1 * med.max()
    ang = np.angle(z)
    # (cos, sin), never the angle: an R2 on raw angles scores the branch cut, where 23.5 h
    # and 0.5 h are one hour apart and average to the opposite time of day.
    pha = np.concatenate([np.cos(ang)[:, keep], np.sin(ang)[:, keep]], axis=1)
    return A.reshape(len(A), -1).astype(np.float32), pha.astype(np.float32)


def ridge_r2(feat, target, tr, te, alphas=(0.1, 1, 10, 100, 1000, 10000)):
    """Held-out R2, penalty chosen on a split of TRAIN -- never on test.

    Multi-output, so the score is the variance-weighted R2 over the target columns, which
    keeps a channel with almost no seasonal amplitude from dominating the average.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler
    tr = np.asarray(tr)
    idx = np.flatnonzero(tr)
    if len(idx) < 20 or te.sum() < 5:
        return float("nan")
    cut = int(0.75 * len(idx))
    fit_i, sel_i = idx[:cut], idx[cut:]
    sc = StandardScaler().fit(feat[fit_i])
    F, Fs, Ft = sc.transform(feat[fit_i]), sc.transform(feat[sel_i]), sc.transform(feat[te])
    best, best_s = None, -np.inf
    for a in alphas:
        m = Ridge(alpha=a).fit(F, target[fit_i])
        s = r2_score(target[sel_i], m.predict(Fs), multioutput="variance_weighted")
        if s > best_s:
            best, best_s = m, s
    return float(r2_score(target[te], best.predict(Ft), multioutput="variance_weighted"))


def readout_blocks(model, X, cfg):
    """{block: (n, d)} -- the trend half, and the amplitude and phase halves of the
    spectral seasonal readout, exactly as the shipped readout builds them."""
    import torch
    net = model.net
    net.eval()
    Xf = np.asarray(X, dtype=np.float32)
    t, amp, pha = [], [], []
    with torch.no_grad():
        for i in range(0, len(Xf), 64):
            xb = torch.from_numpy(Xf[i:i + 64])
            out_t, out_s = net(xb)
            t.append(out_t.mean(dim=1).numpy())
            amp.append(model._seasonal_spectral(out_s, "spec_amp").numpy())
            pha.append(model._seasonal_spectral(out_s, "spec_phase").numpy())
    cat = lambda v: np.concatenate(v).astype(np.float32)
    T, A, P = cat(t), cat(amp), cat(pha)
    return {"V^T": T, "V^S amp": A, "V^S phase": P,
            "full readout": np.concatenate([T, A, P], axis=1)}


def aggregate(run_dir):
    files = sorted(glob.glob(str(Path(run_dir) / "*" / "**" / f"{NAME}.json"), recursive=True))
    rows = [json.loads(Path(f).read_text()) for f in files]
    if not rows:
        raise SystemExit(f"no {NAME}.json under {run_dir}")
    from tasks.sign_test import sign_summary
    blocks = list(rows[0]["DSSL"])
    print()
    print(f"  {len(rows)} seeds, HRD -- held-out R2 of recovering the TRUE seasonal")
    print("  component's amplitude and phase from the representation")
    for tgt in ("amp", "phase"):
        print()
        print(f"  --- target: seasonal {tgt} ---")
        print(f"  {'feature block':16s} {'DSSL':>8s} {'Rand-init':>10s} {'diff':>9s}"
              f" {'wins':>8s} {'p':>8s}")
        for b in blocks:
            a = np.array([r["DSSL"][b][tgt] for r in rows], float)
            c = np.array([r["Random-init"][b][tgt] for r in rows], float)
            k, m, p = sign_summary(a - c)
            print(f"  {b:16s} {np.nanmean(a):8.4f} {np.nanmean(c):10.4f}"
                  f" {np.nanmean(a - c):+9.4f} {k:4d}/{m} {p:8.4f}")
    print()
    print("  The prediction under test: DSSL above its control on PHASE, below it on")
    print("  AMPLITUDE. RQ1's single number is the average of those two.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant-dir")
    ap.add_argument("--aggregate", metavar="RUN_DIR")
    ap.add_argument("--cache-dir", default=None)
    a = ap.parse_args()
    if a.aggregate:
        aggregate(a.aggregate)
        return
    if not a.variant_dir:
        ap.error("one of --variant-dir or --aggregate is required")

    from tasks._experiment_common import load_context, out_dir, random_init_model, save
    ctx = load_context(a.variant_dir, a.cache_dir, gpu=-1)
    if not getattr(ctx, "trained", True):
        raise SystemExit(f"{a.variant_dir} has no trained encoder -- this is a contrast "
                         "between a trained encoder and its untrained control, and without "
                         "the first it would compare two random ones")
    amp, pha = seasonal_targets(ctx.X[:, :, :ctx.n_sensors], ctx.bins_per_day)
    tr, te = ctx.train_mask, ctx.test_mask
    print(f"[block_rec] {ctx.tag} seed={ctx.seed} | targets amp {amp.shape} phase {pha.shape}"
          f" | {int(tr.sum())} train / {int(te.sum())} test windows", flush=True)

    res = {"variant": ctx.tag, "seed": ctx.seed}
    for tag, model in (("DSSL", ctx.model), ("Random-init", random_init_model(ctx))):
        res[tag] = {}
        for name, F in readout_blocks(model, ctx.X, ctx.cfg).items():
            res[tag][name] = {"amp": ridge_r2(F, amp, tr, te),
                              "phase": ridge_r2(F, pha, tr, te)}
        print(f"  {tag}: " + "  ".join(
            f"{k}(amp={v['amp']:.3f},pha={v['phase']:.3f})" for k, v in res[tag].items()),
            flush=True)
    save(out_dir(ctx, NAME), NAME, res)


if __name__ == "__main__":
    main()
