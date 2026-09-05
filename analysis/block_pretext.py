"""Which of the model's contrastive tasks is actually hard before training?

The block breakdown of RQ2 says exactly one part of the representation works:

    V^S phase   C = 0.8802        V^S amp   C = 0.5661        V^T   C = 0.6054

and cost.py says why it might: the phase is mapped to the unit circle before the contrastive
term (`circular_phase`), so its dot product is a similarity between angles, while the
amplitude is neither normalised nor given a temperature. On a non-negative vector |F| a bare
dot product is dominated by magnitude, so that term can be satisfied by "which window is
larger" -- a person fingerprint, constant across a person's windows, and nothing to learn.

If that is the mechanism it makes a prediction nobody has tested: at initialisation the PHASE
task should be HARD and the AMPLITUDE task EASY. A task already solved has no gradient, which
is the same reason the trend branch learns nothing -- its MoCo top-1 is 0.8223, 6737x chance.

Every block is scored through the SAME in-batch retrieval the seasonal loss itself uses:
for each timestep, a 2B x 2B similarity with the diagonal removed, the positive being the
other augmented view of the same window. Chance is 1/(2B-1).

The fourth row is the reason this exists now. V^N -- the residual branch, built and never
trained -- has the highest ceiling measured in this project (0.7202) and the residual is
where the depression label actually lives (0.6862 against 0.6228 for trend and seasonal
together). But it is trained with the window positive pair, and that pair is nearly solved
at init FOR THE TREND. If it is also solved for the residual, V^N will learn nothing and
that ceiling is unreachable. The residual is high-frequency and carries no level, so the
task may be genuinely harder -- which is a claim, and this measures it.

    python analysis/block_pretext.py --npz hrd_2224103.npz
"""
import sys
from pathlib import Path

# Run as `python analysis/<name>.py` from the repository root: the interpreter puts
# this file's own directory on sys.path, not the project root, so the shared modules
# would not import. scripts/ already does this; the pattern is the same.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

import numpy as np


def top1(z1, z2):
    """Fraction of queries whose nearest neighbour is their own other view.

    The mask and the pairing follow CoSTModel.instance_contrastive_loss exactly: the score
    is a bare dot product over the last axis, every timestep votes, and an item is never its
    own neighbour. Reproducing the loss's own geometry is the point -- a cosine version here
    would measure a task the model is not being trained on.
    """
    import torch
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0).transpose(0, 1)          # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))                # T x 2B x 2B
    eye = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, float("-inf"))
    partner = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return float((sim.argmax(dim=-1) == partner).float().mean())


def blocks(model, xq, xk, cfg):
    """{block: (view1, view2)} -- the tensors each contrastive term actually compares."""
    import torch
    from torch import fft
    net, cost = model.net, model.cost
    out = {}
    with torch.no_grad():
        qt, qs = net(xq)
        kt, ks = net(xk)
        # the trend term contrasts ONE timestep through head_q; mean-pooling instead would
        # describe a readout rather than the objective
        idx = int(np.random.randint(0, xq.shape[1]))
        out["V^T (trend view)"] = (cost._trend_view(qt, idx).unsqueeze(1),
                                   cost._trend_view(kt, idx).unsqueeze(1))
        qa, qp = cost.convert_coeff(fft.rfft(qs.float(), dim=1))
        ka, kp = cost.convert_coeff(fft.rfft(ks.float(), dim=1))
        out["V^S amplitude"] = (qa, ka)
        if cost.phase_mode != "raw":
            w_q = qa if cost.phase_mode == "circular_amp" else None
            w_k = ka if cost.phase_mode == "circular_amp" else None
            qp, kp = cost.circular_phase(qp, w_q), cost.circular_phase(kp, w_k)
        out["V^S phase"] = (qp, kp)
        if getattr(net, "noise_branch", False):
            out["V^N (residual)"] = (net.encode_noise(xq), net.encode_noise(xk))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--batches", type=int, default=24)
    ap.add_argument("--noise-depth", type=int, default=3)
    ap.add_argument("--out", default="results/block_pretext.json")
    a = ap.parse_args()

    import torch
    from cost import PretrainDataset
    from analysis.local_context import local_context
    from model_build import build_model

    seeds = [int(s) for s in np.load(a.npz, allow_pickle=True)["seeds"]]
    ctx = local_context(a.npz, seeds[0])
    Xp = np.asarray(ctx.X[ctx.pretrain_mask], dtype=np.float32)
    cfg = dict(ctx.cfg)
    # The branch has to EXIST to be measured; the weight stays 0 so nothing about the
    # comparison depends on an objective that has never been run.
    cfg["noise_branch"], cfg["noise_depth"] = True, a.noise_depth
    torch.manual_seed(ctx.seed)
    model = build_model(cfg, Xp, ctx.n_sensors, "cpu")
    model.net.eval()
    ds = PretrainDataset(torch.from_numpy(Xp),
                         jitter_sigma=cfg.get("jitter_sigma", 0.1),
                         shift_sigma=cfg.get("shift_sigma", 0.5),
                         n_sensors=ctx.n_sensors, bins_per_day=ctx.bins_per_day,
                         smooth_bins=cfg.get("smooth_bins", 0))
    chance = 1.0 / (2 * a.batch - 1)
    print(f"[block] {len(Xp):,} pretrain windows | batch {a.batch} -> chance {chance:.4f}",
          flush=True)

    rng = np.random.default_rng(0)
    acc = {}
    for b in range(a.batches):
        i = rng.choice(len(Xp), a.batch, replace=False)
        xq = torch.stack([ds.transform(torch.from_numpy(Xp[j])) for j in i])
        xk = torch.stack([ds.transform(torch.from_numpy(Xp[j])) for j in i])
        for name, (z1, z2) in blocks(model, xq, xk, cfg).items():
            acc.setdefault(name, []).append(top1(z1, z2))
        if (b + 1) % 6 == 0:
            print(f"  [{b + 1:2d}/{a.batches}] "
                  + "  ".join(f"{k.split()[0]}={np.mean(v):.3f}" for k, v in acc.items()),
                  flush=True)

    json.dump({k: {"top1": float(np.mean(v)), "sd": float(np.std(v)), "n": len(v)}
               for k, v in acc.items()} | {"chance": chance}, open(a.out, "w"), indent=2)
    print()
    print(f"  {a.batches} batches of {a.batch}, untrained encoder, HRD")
    print()
    print(f"  {'contrastive task':22s} {'top-1':>8s} {'sd':>7s} {'x chance':>9s}")
    for k, v in acc.items():
        m = float(np.mean(v))
        print(f"  {k:22s} {m:8.4f} {float(np.std(v)):7.4f} {m / chance:9.1f}")
    print()
    print("  A task already solved at initialisation has no gradient left to give. The")
    print("  prediction under test: phase HARD, amplitude EASY -- which would explain why")
    print("  exactly one block of this representation works.")


if __name__ == "__main__":
    main()
