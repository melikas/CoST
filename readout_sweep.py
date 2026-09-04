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
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

SEGS = [1, 2, 4, 7, 14, 28]
WIDTH = 1760            # every arm is compared at the production readout width


def _project(F, seed, width=WIDTH):
    """Down to a common width, so nothing in this table is won by dimension count alone."""
    if F.shape[1] <= width:
        return F
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1.0 / np.sqrt(width), (F.shape[1], width))
    return (F @ W).astype(np.float32)


def segment_readouts(npz, seed, cache_dir, batch=64, threads=None):
    """{name: (n, d)} for every segment count, from one pass over the encoder."""
    import torch
    from local_context import local_context
    from model_build import random_init_model
    ctx = local_context(npz, seed)
    f = Path(cache_dir) / f"readout_{Path(npz).stem}_{seed}.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k] for k in z.files}, ctx

    torch.set_num_threads(threads or os.cpu_count() or 4)
    t0 = time.time()
    m = random_init_model(ctx.cfg, ctx.X, ctx.n_sensors, "cpu", seed)
    m.net.eval()
    acc = {s: [] for s in SEGS}
    X = np.asarray(ctx.X, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out_t, out_s = m.net(torch.from_numpy(X[i:i + batch]))
            # trend and seasonal are read the SAME way here: the question is time
            # resolution, so the two halves must not differ in anything else.
            z = out_t if out_s is None else torch.cat([out_t, out_s], dim=-1)
            T = z.shape[1]
            for s in SEGS:
                w = T // s
                acc[s].append(z[:, :s * w].reshape(z.shape[0], s, w, z.shape[-1])
                              .mean(dim=2).reshape(z.shape[0], -1).numpy())
    feats = {f"seg {s:2d}": np.concatenate(acc[s]).astype(np.float32) for s in SEGS}
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

    from local_context import local_context
    from random_init_audit import _probe_auc, raw_projection

    seeds = [int(s) for s in np.load(a.npz, allow_pickle=True)["seeds"]]
    rows = []
    for es in seeds[:a.encoder_seeds]:
        feats, ctx0 = segment_readouts(a.npz, es, a.cache_dir, threads=a.threads)
        feats = {k: _project(v, es) for k, v in feats.items()}
        feats["raw projection"] = raw_projection(ctx0.X, ctx0.n_sensors, WIDTH, es)
        for ss in seeds:
            ctx = local_context(a.npz, ss)
            r = {"encoder_seed": es, "split_seed": ss}
            for name, F in feats.items():
                r[name] = _probe_auc(F, ctx)
            rows.append(r)
            json.dump(rows, open(a.out, "w"), indent=1)
        d = [r for r in rows if r["encoder_seed"] == es]
        print("  seed %d: " % es
              + "  ".join(f"{k}={np.nanmean([x[k] for x in d]):.4f}"
                          for k in d[0] if not k.endswith("_seed")), flush=True)

    names = [k for k in rows[0] if not k.endswith("_seed")]
    print(f"\n  {len(rows)} measurements, HRD, same probe and splits."
          f"  'seg 1' IS the production readout.\n")
    print(f"  {'readout':20s} {'AUC':>7s} {'vs seg 1':>10s} {'wins':>8s}")
    base = np.array([r["seg  1"] for r in rows], float)
    for n in names:
        v = np.array([r[n] for r in rows], float)
        print(f"  {n:20s} {np.nanmean(v):7.4f} {np.nanmean(v - base):+10.4f} "
              f"{int(np.nansum(v - base > 0)):4d}/{len(rows)}")


if __name__ == "__main__":
    main()
