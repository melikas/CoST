"""The readout collapses time. How much does that cost, and does keeping time recover it?

architecture_loss.py measured that the deficit is architectural, not learned:

    raw random projection -> random-init encoder   -0.0493
    random-init encoder   -> trained DSSL          -0.0004

So no change to the contrastive objective can recover it; it is gone before the first
gradient step. What the projection does that the encoder does not is keep TIME. Each of its
1760 outputs is a different random weighting of all 2016 inputs, so the ensemble preserves
when things happened. The encoder's readout has four settings -- last, mean, max, meanmax --
and every one of them is an aggregate over the 672 timesteps. The trend half mean-pools and
the seasonal half keeps five frequency lines, and in that measurement both halves lost about
equally, so this is not one branch's bug but the property they share.

This sweeps the one axis that distinguishes them: how much time resolution survives the
readout. `seg S` mean-pools the encoder's feature sequence within S equal segments and
concatenates, so S=1 IS the current readout and larger S keeps proportionally more. If the
score climbs with S, the encoder was computing something useful all along and the readout
was throwing it away -- and the fix is a readout, which costs no retraining. If it stays
flat, the encoder itself destroys the information and the TCN has to change.

Encoding is the expensive part, so every readout is computed from ONE forward pass, per
batch, without ever materialising the (3890, 672, 320) feature tensor.

    python readout_sweep.py --npz hrd_2224103.npz --encoder-seeds 3
"""
import sys
from pathlib import Path

# Run as `python analysis/<name>.py` from the repository root: the interpreter puts
# this file's own directory on sys.path, not the project root, so the shared modules
# would not import. scripts/ already does this; the pattern is the same.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

SEGS = [1, 2, 4, 7, 14, 28]
WIDTH = 1760            # the raw-projection reference, at the production readout's width


def readout_parts(model, X, season_pool="spec", batch=64, segs=SEGS):
    """Every candidate readout of one encoder, from ONE forward pass.

    Returns {"trend seg S", "season seg S", "season spec"} -> (n, d). The trend and seasonal
    halves are kept APART because the shipped readout does not treat them alike: it mean-pools
    the trend and reads the seasonal branch spectrally, and cost.py notes that time-domain
    pooling on that branch provably discards the rhythm. Pooling both is a readout nobody
    runs, and an early version of this file called that the production reference and
    understated it by 0.018.

    The forward pass is the expensive part and the readouts are not, so they are all
    accumulated per batch and the (n, T, d) feature tensor is never materialised.
    """
    import torch
    model.net.eval()
    X = np.asarray(X, dtype=np.float32)

    def cut(z):
        T, d = z.shape[1], z.shape[2]
        return {k: z[:, :k * (T // k)].reshape(z.shape[0], k, T // k, d)
                .mean(dim=2).reshape(z.shape[0], -1).numpy() for k in segs}

    acc = {"t": [], "s": [], "spec": []}
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out_t, out_s = model.net(torch.from_numpy(X[i:i + batch]))
            acc["t"].append(cut(out_t))
            acc["s"].append(cut(out_s))
            acc["spec"].append(model._seasonal_spectral(out_s, season_pool).numpy())
    cat = lambda k, j: np.concatenate([b[j] for b in acc[k]]).astype(np.float32)
    parts = {f"trend seg {j:2d}": cat("t", j) for j in segs}
    parts.update({f"season seg {j:2d}": cat("s", j) for j in segs})
    parts["season spec"] = np.concatenate(acc["spec"]).astype(np.float32)
    return parts


def production_readout(parts):
    """The readout that actually ships: trend mean-pooled, seasonal read spectrally.

    Asserted elsewhere to reproduce model_build.encode_repr exactly (max abs diff 0.0), which
    is what makes it usable as the reference row rather than an approximation of one.
    """
    return np.concatenate([parts["trend seg  1"], parts["season spec"]], axis=1)


def segment_readouts(npz, seed, cache_dir, batch=64, threads=None):
    """`readout_parts` for one RANDOM-INIT draw over an npz dump, cached on disk."""
    import torch
    from analysis.local_context import local_context
    from model_build import random_init_model
    ctx = local_context(npz, seed)
    f = Path(cache_dir) / f"readout_{Path(npz).stem}_{seed}.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k] for k in z.files}, ctx

    torch.set_num_threads(threads or os.cpu_count() or 4)
    t0 = time.time()
    m = random_init_model(ctx.cfg, ctx.X, ctx.n_sensors, "cpu", seed)
    feats = readout_parts(m, ctx.X, ctx.cfg.get("season_pool") or "spec", batch)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **feats)
    print(f"  encoded seed {seed} in {(time.time() - t0) / 60:.1f} min "
          + " ".join(f"{k}={v.shape[1]}d" for k, v in feats.items()), flush=True)
    return feats, ctx


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--encoder-seeds", type=int, default=3)
    ap.add_argument("--cache-dir", default="_randinit_cache")
    ap.add_argument("--threads", type=int, default=None,
                    help="CPU threads for the encoder pass"
                         " (default: every core)")
    ap.add_argument("--out", default="readout_sweep.json")
    a = ap.parse_args()

    from analysis.local_context import local_context
    from analysis.random_init_audit import _probe_auc, raw_projection

    seeds = [int(s) for s in np.load(a.npz, allow_pickle=True)["seeds"]]
    rows, dims = [], {}
    for es in seeds[:a.encoder_seeds]:
        parts, ctx0 = segment_readouts(a.npz, es, a.cache_dir, threads=a.threads)
        # Two families. A varies the time resolution of BOTH halves; B is the minimal
        # change to what actually ships -- the seasonal branch keeps its spectral readout
        # and only the trend's resolution moves. B is the one a fix would adopt.
        feats = {"PRODUCTION (trend mean + season spec)":
                 np.concatenate([parts["trend seg  1"], parts["season spec"]], axis=1)}
        for s in SEGS:
            k = f"seg {s:2d}"
            feats[f"A both {k}"] = np.concatenate(
                [parts[f"trend {k}"], parts[f"season {k}"]], axis=1)
            feats[f"B trend {k} + spec"] = np.concatenate(
                [parts[f"trend {k}"], parts["season spec"]], axis=1)
        # NOT forced to a common width. Doing that looked like the fair move and is not:
        # taking the 1760-dim production readout down to 320 costs it 0.066 (0.6779 ->
        # 0.6118), so the projection is a treatment of its own rather than a neutral
        # rescaling, and every row would be scored through a step no candidate design has.
        # Each readout is measured at the width it would ship at, printed beside it. Width
        # alone does not drive these numbers: a sweep of the raw projection over 16..1760
        # dims spans 0.6865 to 0.7198 with no trend in between.
        dims.update({k: v.shape[1] for k, v in feats.items()})
        feats["raw projection"] = raw_projection(ctx0.X, ctx0.n_sensors, WIDTH, es)
        dims["raw projection"] = WIDTH
        for ss in seeds:
            ctx = local_context(a.npz, ss)
            r = {"encoder_seed": es, "split_seed": ss}
            for name, F in feats.items():
                r[name] = _probe_auc(F, ctx)
            rows.append(r)
            json.dump(rows, open(a.out, "w"), indent=1)
        d = [r for r in rows if r["encoder_seed"] == es]
        print(f"  seed {es}: "
              + "  ".join(f"{k}={np.nanmean([x[k] for x in d]):.4f}"
                          for k in ("PRODUCTION (trend mean + season spec)",
                                    "raw projection")), flush=True)

    report(rows, dims)


def report(rows, dims):
    ref = "PRODUCTION (trend mean + season spec)"
    names = [k for k in rows[0] if not k.endswith("_seed")]
    base = np.array([r[ref] for r in rows], float)
    print()
    print(f"  {len(rows)} measurements, HRD, same probe and splits,"
          f" each readout at the width it would ship at")
    print()
    print(f"  {'readout':34s} {'dim':>6s} {'AUC':>8s} {'vs production':>14s} {'wins':>8s}")
    for n in names:
        v = np.array([r[n] for r in rows], float)
        print(f"  {n:34s} {dims.get(n, 0):6d} {np.nanmean(v):8.4f}"
              f" {np.nanmean(v - base):+14.4f} {int(np.nansum(v - base > 0)):4d}/{len(rows)}")


if __name__ == "__main__":
    main()
