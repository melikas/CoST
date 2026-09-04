"""Does the trend head's causal read cost anything on the task we have?

The backbone is SamePadConv with padding = receptive_field // 2 -- centred, and hundreds of
steps wide at depth 10 -- so the representation at step t already depends on the future
before the trend head sees it. The trend head is then read causally, keeping the leading T
outputs of a k-1 padded convolution. That read cannot make the representation causal; what
it does is discard the right-hand context of every step, and at kernel 256 the start of the
window is left with almost none on either side.

It is not needed for forecasting either. A forecast at time t is made by encoding a window
that ENDS at t, which contains no future data whatever the padding does -- which is how
TS2Vec and CoST are used for forecasting in the first place. So the causal read is inherited
from CoST's forecasting origin rather than required by it.

The constraint on any change here is that the CURRENT task must not get worse. This measures
exactly that, and nothing is adopted unless the answer is neutral or better.

Both arms share their weights bit for bit -- `trend_causal` selects where a padded
convolution is READ, not what it computes -- and the backbone runs once per draw and feeds
both, so the comparison is paired at the level of the encoder and not just the split.

    python trend_causality.py --npz hrd_2224103.npz --draws 3
"""
import argparse
import json
import os
import time

import numpy as np


def readouts(cfg, X, n_sensors, seed, batch=64, threads=None):
    """{'causal': (n, d), 'centred': (n, d)} from ONE backbone pass per batch."""
    import torch
    from model_build import random_init_model
    torch.set_num_threads(threads or os.cpu_count() or 4)
    m = random_init_model(cfg, X, n_sensors, "cpu", seed)
    net = m.net
    net.eval()
    sp = cfg.get("season_pool") or "spec"
    Xf = np.asarray(X, dtype=np.float32)
    acc = {"causal": [], "centred": []}
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(Xf), batch):
            xb = torch.from_numpy(Xf[i:i + batch])
            # the seasonal half is untouched by this flag, so it is computed once
            net.trend_causal = True
            out_t_c, out_s = net(xb)
            net.trend_causal = False
            out_t_n, _ = net(xb)
            season = m._seasonal_spectral(out_s, sp)
            for key, tr in (("causal", out_t_c), ("centred", out_t_n)):
                acc[key].append(torch.cat([tr.mean(dim=1), season], dim=-1).numpy())
    net.trend_causal = True
    print(f"  encoded draw {seed} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return {k: np.concatenate(v).astype(np.float32) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--out", default="trend_causality.json")
    a = ap.parse_args()

    from local_context import local_context
    from random_init_audit import _probe_auc
    from tasks.sign_test import sign_summary

    seeds = [int(s) for s in np.load(a.npz, allow_pickle=True)["seeds"]]
    rows = []
    for es in seeds[:a.draws]:
        ctx0 = local_context(a.npz, es)
        F = readouts(ctx0.cfg, ctx0.X, ctx0.n_sensors, es)
        for ss in seeds:
            ctx = local_context(a.npz, ss)
            r = {"encoder_seed": es, "split_seed": ss}
            for name, feat in F.items():
                r[name] = _probe_auc(feat, ctx)
            rows.append(r)
            json.dump(rows, open(a.out, "w"), indent=1)
        d = [x for x in rows if x["encoder_seed"] == es]
        print(f"  seed {es}: causal={np.nanmean([x['causal'] for x in d]):.4f}"
              f"  centred={np.nanmean([x['centred'] for x in d]):.4f}", flush=True)

    c = np.array([r["causal"] for r in rows], float)
    n = np.array([r["centred"] for r in rows], float)
    k, m, p = sign_summary(n - c)
    print()
    print(f"  {len(rows)} measurements, {a.draws} encoder draws x {len(seeds)} splits, HRD")
    print()
    print(f"  causal trend read (shipped)   {np.nanmean(c):.4f}")
    print(f"  centred trend read            {np.nanmean(n):.4f}")
    print(f"  difference                    {np.nanmean(n - c):+.4f}   {k}/{m}   p={p:.4f}")
    print()
    print("  ADOPT -- the centred read does not cost the current task"
          if np.nanmean(n - c) >= 0 else
          "  KEEP THE CAUSAL READ -- the centred one is worse on the task we have")


if __name__ == "__main__":
    main()
