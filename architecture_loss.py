"""Does the ARCHITECTURE lose information, before any training happens?

The gap that needs explaining is 0.035: a Gaussian random projection of the raw window
scores 0.712-0.720 on HRD while the best DSSL scores 0.6790. Two very different things
could produce it, and they need different fixes:

  A  The CONTRASTIVE OBJECTIVE discards it. Then an untrained encoder of the same shape
     should score like the random projection, and only training pulls it down. Fix: change
     the objective.

  B  The ARCHITECTURE discards it, before a single gradient step. A TCN is a bank of linear
     filters, BandedFourierLayer keeps only selected bands, and the readout MEAN-pools over
     time -- each of those is a projection that throws something away. Fix: change the
     information path, and no objective can rescue it until that is done.

They are separated by one measurement: random-init encoder vs a random linear map of the
same input at the SAME output width. Same probe, same splits, same held-out participants.

The trend and seasonal halves are scored separately as well, because if one of them is
carrying the loss then the fix is local to that branch rather than to the whole encoder.

Encoding is the expensive part on CPU (~14 min for 3890 windows) and probing is not, so
this draws a few encoders and evaluates each against all 24 splits rather than pairing one
encoder to one split. Vectors are cached, so an interrupted run resumes.

    python architecture_loss.py --npz hrd_2224103.npz --encoder-seeds 4
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def encode_once(npz, seed, cache_dir):
    """The random-init readout for one encoder draw, cached on disk."""
    import torch
    from local_context import local_context
    from model_build import random_init_model, encode_repr
    f = Path(cache_dir) / f"randinit_{Path(npz).stem}_{seed}.npy"
    ctx = local_context(npz, seed)
    if f.exists():
        return np.load(f), ctx
    torch.set_num_threads(os.cpu_count() or 4)
    t = time.time()
    m = random_init_model(ctx.cfg, ctx.X, ctx.n_sensors, "cpu", seed)
    V = encode_repr(m, ctx.X, ctx.cfg, batch=64)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, V)
    print(f"  encoded seed {seed} -> {V.shape} in {(time.time() - t) / 60:.1f} min", flush=True)
    return V, ctx


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--encoder-seeds", type=int, default=4,
                    help="how many encoder draws (default: %(default)s)")
    ap.add_argument("--cache-dir", default="_randinit_cache")
    ap.add_argument("--out", default="architecture_loss.json")
    a = ap.parse_args()

    from local_context import local_context
    from random_init_audit import _probe_auc, raw_projection

    seeds = [int(s) for s in np.load(a.npz, allow_pickle=True)["seeds"]]
    enc_seeds = seeds[:a.encoder_seeds]
    rows = []
    for es in enc_seeds:
        V, ctx0 = encode_once(a.npz, es, a.cache_dir)
        dT = int(ctx0.cfg["repr_dims"])
        proj = raw_projection(ctx0.X, ctx0.n_sensors, V.shape[1], es)
        feats = {"random-init encoder": V, "raw projection (same width)": proj,
                 "  encoder trend half": V[:, :dT], "  encoder seasonal half": V[:, dT:]}
        for ss in seeds:
            ctx = local_context(a.npz, ss)
            r = {"encoder_seed": es, "split_seed": ss}
            for name, F in feats.items():
                r[name] = _probe_auc(F, ctx)
            rows.append(r)
            json.dump(rows, open(a.out, "w"), indent=1)
        done = [r for r in rows if r["encoder_seed"] == es]
        print(f"  seed {es}: encoder {np.nanmean([r['random-init encoder'] for r in done]):.4f}"
              f"  projection {np.nanmean([r['raw projection (same width)'] for r in done]):.4f}"
              f"  ({len(done)} splits)", flush=True)

    names = [k for k in rows[0] if not k.endswith("_seed")]
    ref = "raw projection (same width)"
    print(f"\n  {len(enc_seeds)} encoder draws x {len(seeds)} splits = {len(rows)}"
          f" measurements, HRD\n")
    print(f"  {'arm':30s} {'AUC':>7s} {'vs projection':>14s} {'wins':>8s}")
    for n in sorted(names, key=lambda n: -float(np.nanmean([r[n] for r in rows]))):
        v = np.array([r[n] for r in rows], float)
        d = v - np.array([r[ref] for r in rows], float)
        print(f"  {n:30s} {np.nanmean(v):7.4f} {np.nanmean(d):+14.4f} "
              f"{int(np.nansum(d > 0)):4d}/{len(rows)}")


if __name__ == "__main__":
    main()
