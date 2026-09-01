"""Why does an UNTRAINED encoder top the RQ3 ladder? Three explanations, one experiment.

On the day-disjoint run the ladder came out with Random-init first (AUC 0.6691), ahead of
cosinor (0.6492) and of the end-to-end supervised ceiling (0.6215). That is surprising enough
that it has to be attacked rather than reported. Three things could produce it, and they make
different predictions:

  1  A LEAK. If anything about the held-out participants reaches the probe, every
     high-dimensional arm floats up. Prediction: permute the participant-to-label map and the
     AUC stays high. A clean pipeline returns 0.5.

  2  GENERIC RANDOM FEATURES. Random projections of a structured signal preserve a great deal,
     and a penalised linear probe on 1760 random features is a strong learner on its own -- the
     random-features / kernel equivalence. Prediction: a plain Gaussian projection of the RAW
     window, with no encoder at all, scores the same.

  3  THE BANDING. With `seasonal_bands=harmonics` the Fourier layer is a hand-built band-pass
     filter sitting exactly on the circadian harmonics, so the random encoder is a random
     mixture of chronobiologically selected frequencies rather than of everything. Prediction:
     a random projection of the raw spectrum RESTRICTED TO THOSE BANDS scores like Random-init,
     while the unrestricted projection of (2) does not.

Explanation 3 would mean the number is real and the inductive bias is doing the work -- which
is this project's central finding, stated more sharply. Explanation 1 would invalidate every
result in the project. They are worth separating exactly.

Every arm goes through the SAME probe, the SAME split and the SAME participants, so nothing
but the feature matrix differs.

Run (GPU helps but is not required):
    SCRIPT=random_init_audit.py sbatch --array=0-23%12 scripts/stability_gate.sh results_hrd/<run>
    python random_init_audit.py --aggregate results_hrd/<run>
"""
import argparse
import json
from pathlib import Path

import numpy as np


def _probe_auc(feat, ctx, y=None):
    """Participant AUC on the run's own held-out participants, with the canonical probe."""
    from sklearn.metrics import roc_auc_score
    from tasks._eval_protocols import fit_persubject_probe, persubject_rows
    y = ctx.y if y is None else y
    clf = fit_persubject_probe(feat, ctx.pids, y, ctx.train_mask, ctx.val_mask, ctx.seed)
    Xs, ys, _ = persubject_rows(feat, ctx.pids, y, ctx.test_mask)
    if len(set(ys)) < 2:
        return float("nan")
    return float(roc_auc_score(ys, clf.predict_proba(Xs)[:, 1]))


def raw_projection(X, n_sensors, width, seed, bands=None, length=None):
    """A Gaussian random projection of the raw window -- no encoder anywhere.

    With `bands`, the window is taken to the frequency domain first and only the listed rFFT
    bin ranges are kept before projecting. That is the only difference between arm 2 and arm 3:
    whether the random mixture is over the whole signal or over the circadian bands alone.
    """
    S = np.nan_to_num(np.asarray(X[:, :, :n_sensors], dtype=float), nan=0.0)
    if bands is None:
        F = S.reshape(len(S), -1)
    else:
        Z = np.fft.rfft(S, axis=1)
        keep = np.concatenate([np.arange(lo, min(hi, Z.shape[1])) for lo, hi in bands])
        z = Z[:, keep]
        F = np.concatenate([z.real.reshape(len(S), -1), z.imag.reshape(len(S), -1)], axis=1)
    F = (F - F.mean(0)) / (F.std(0) + 1e-8)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1.0 / np.sqrt(F.shape[1]), (F.shape[1], int(width)))
    return (F @ W).astype(np.float32)


def audit(ctx, n_perm=20):
    """Every arm, one split, one probe. Returns a flat dict of AUCs."""
    import torch
    from model_build import random_init_model as build_random
    from models.encoder import seasonal_band_edges
    from tasks._experiment_common import encode, random_init_model

    res = {"variant": ctx.tag, "seed": ctx.seed}

    # A split that is not participant-disjoint would explain everything on its own, so it is
    # checked here rather than assumed from the loader.
    tr = set(np.unique(ctx.pids[ctx.train_mask]))
    va = set(np.unique(ctx.pids[ctx.val_mask]))
    te = set(np.unique(ctx.pids[ctx.test_mask]))
    res["overlap_train_test"] = len(tr & te)
    res["overlap_val_test"] = len(va & te)
    res["n_test_participants"] = len(te)
    print(f"[audit] {ctx.tag} seed={ctx.seed} | train {len(tr)} val {len(va)} test {len(te)} "
          f"participants | train-test overlap {len(tr & te)}, val-test {len(va & te)}")

    with torch.no_grad():
        V = encode(ctx.model, ctx.X, ctx.cfg)
        ctrl = random_init_model(ctx)
        R = encode(ctrl, ctx.X, ctx.cfg)
    width = V.shape[1]

    # Two further random draws of the SAME architecture. If the headline number rides on one
    # lucky initialisation these will not reproduce it.
    cfg = dict(ctx.cfg)
    sfd = getattr(ctx.model.net, "sfd", None)
    if sfd is not None:
        cfg["seasonal_bands"] = "single" if len(sfd) == 1 else "harmonics"
    alt = []
    for s in (ctx.seed + 1, ctx.seed + 2):
        with torch.no_grad():
            alt.append(encode(build_random(cfg, ctx.X, ctx.n_sensors, ctx.device, s),
                              ctx.X, ctx.cfg))

    bands = (ctx.model.net.seasonal_bands if hasattr(ctx.model.net, "seasonal_bands")
             else seasonal_band_edges(ctx.seq_len, ctx.bins_per_day))
    res["bands"] = [list(map(int, b)) for b in bands]
    res["readout_width"] = int(width)

    arms = {
        "DSSL": V,
        "Random-init": R,
        "Random-init seed+1": alt[0],
        "Random-init seed+2": alt[1],
        "Raw random projection": raw_projection(ctx.X, ctx.n_sensors, width, ctx.seed),
        "Banded random projection": raw_projection(ctx.X, ctx.n_sensors, width, ctx.seed,
                                                   bands=bands),
    }
    print(f"[audit] readout width {width} | bands {res['bands']}")
    for name, feat in arms.items():
        res[f"auc/{name}"] = _probe_auc(feat, ctx)
        print(f"  {name:28s} AUC {res[f'auc/{name}']:.4f}")

    # ---- the leak test -------------------------------------------------------------------
    # The participant -> label map is permuted globally, so train and test stay consistent with
    # each other and only the association with the SIGNAL is destroyed. A pipeline with no leak
    # returns chance; anything well above it means information about the held-out participants
    # is reaching the probe through a route other than their features.
    rng = np.random.default_rng(ctx.seed + 7919)
    pid_list = np.unique(ctx.pids)
    lab = {p: int(ctx.y[ctx.pids == p][0]) for p in pid_list}
    perms = []
    for _ in range(int(n_perm)):
        shuffled = dict(zip(pid_list, rng.permutation([lab[p] for p in pid_list])))
        yp = np.array([shuffled[p] for p in ctx.pids])
        perms.append(_probe_auc(R, ctx, y=yp))
    res["auc/Random-init permuted labels"] = float(np.nanmean(perms))
    res["auc/Random-init permuted labels max"] = float(np.nanmax(perms))
    res["n_perm"] = int(n_perm)
    print(f"  {'Random-init, labels permuted':28s} AUC {np.nanmean(perms):.4f} "
          f"(max {np.nanmax(perms):.4f} over {n_perm} draws) -- chance is 0.5")
    return res


def aggregate(run_dir):
    """Pool the per-seed audits and test each explanation against Random-init."""
    from tasks._stats import paired
    rows = [json.loads(f.read_text(encoding="utf-8"))
            for f in sorted(Path(run_dir).glob("*_seed*/RQ3/random_init_audit.json"))]
    if not rows:
        raise SystemExit(f"no random_init_audit.json under {run_dir}")
    ntest = [r["n_test_participants"] for r in rows]
    ntr = [r.get("n_train_participants", 0) for r in rows]
    n_splits = 3.17 if not any(ntr) else 1.0 + float(np.mean(ntr)) / float(np.mean(ntest))
    keys = sorted({k[4:] for r in rows for k in r if k.startswith("auc/")})
    col = lambda k: np.array([r.get(f"auc/{k}", np.nan) for r in rows], float)

    print(f"[agg] {len(rows)} seeds | corrected t with n_splits={n_splits:.2f}")
    bad = sum(r["overlap_train_test"] + r["overlap_val_test"] for r in rows)
    print(f"  participant-disjoint split: {'OK' if bad == 0 else f'*** {bad} OVERLAPS ***'}")
    print("")
    for k in sorted(keys, key=lambda x: -np.nanmean(col(x))):
        print(f"  {k:34s} {np.nanmean(col(k)):.4f}")
    print("")
    base = col("Random-init")
    for k in keys:
        if k == "Random-init":
            continue
        r = paired(col(k), base, n_splits)
        print(f"  {k[:32]:32s} - Random-init  {r['diff']:+.4f}  p={r['p']:.4f}  "
              f"wins {r['wins']}/{r['n']}")
    out = {"n_seeds": len(rows), "mean": {k: float(np.nanmean(col(k))) for k in keys},
           "overlaps": bad}
    (Path(run_dir) / "random_init_audit_summary.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"[saved] {Path(run_dir) / 'random_init_audit_summary.json'}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant-dir")
    ap.add_argument("--aggregate", metavar="RUN_DIR")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n-perm", type=int, default=20)
    a = ap.parse_args()
    if a.aggregate:
        aggregate(a.aggregate)
        return
    if not a.variant_dir:
        ap.error("one of --variant-dir or --aggregate is required")

    from tasks._experiment_common import load_context, out_dir, save
    ctx = load_context(a.variant_dir, a.cache_dir, a.gpu)
    res = audit(ctx, a.n_perm)
    res["n_train_participants"] = int(len(np.unique(ctx.pids[ctx.train_mask])))
    save(out_dir(ctx, "rq3"), "random_init_audit", res)


if __name__ == "__main__":
    main()
