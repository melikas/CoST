"""Where does the amplitude collapse happen: inside the TCN backbone, or after it?

    python scripts/backbone_rank.py --run results_eq_h --fold r0f0 --device cuda

Loads the fold's trained encoder (encoders_<weights>/<fold>/encoder.pt, as RQ2 does) and its
untrained control (eval._blank_model with the fold's model seed), feeds both the same random
sample of the fold's NON-test windows (labels unused), and reports the effective rank --
exp(entropy of the normalised eigenvalues of the standardised correlation matrix), Roy &
Vetterli 2007 -- at every stage between the input and the probe:

  TCN output            320 channels, over (window, timestep) samples
  seasonal z_s          160 channels, the banded Fourier layers' output, same samples
  seasonal, L2 per step 160 channels, after the readout's per-timestep normalisation
  amp readout           800 per window, the vector the probe reads (5 bins x 160)
  amp at 24 h           160 per window

A sanity line checks that the loaded encoder reproduces the run's stored amp readout.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval as E                                    # noqa: E402
from data_loader import load_npz                    # noqa: E402
from probe import spectral_freqs                    # noqa: E402


def erank(M):
    M = np.asarray(M, np.float64)
    sd = M.std(0)
    M = (M[:, sd > 1e-9] - M.mean(0)[sd > 1e-9]) / sd[sd > 1e-9]
    lam = np.clip(np.linalg.eigvalsh(np.cov(M, rowvar=False)), 0, None)
    p = lam / lam.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


@torch.no_grad()
def stages(model, X, device, stride, i24):
    net = model.net.eval()
    tcn, zs, zn = [], [], []
    for i in range(0, len(X), 64):
        xb = torch.as_tensor(X[i:i + 64], dtype=torch.float32, device=device)
        h = net(xb, tcn_output=True)[:, ::stride]
        s = net(xb)[1][:, ::stride]
        tcn.append(h.reshape(-1, h.size(-1)).cpu().numpy())
        zs.append(s.reshape(-1, s.size(-1)).cpu().numpy())
        zn.append(F.normalize(s, dim=-1).reshape(-1, s.size(-1)).cpu().numpy())
    amp = model.encode(X, batch_size=64, parts=True)["amp"]
    d = net.seasonal_dims
    return {"TCN output": np.concatenate(tcn), "seasonal z_s": np.concatenate(zs),
            "seasonal, L2 per step": np.concatenate(zn), "amp readout": amp,
            "amp at 24 h": amp.reshape(len(X), -1, d)[:, i24]}


def cplx(Z):
    return erank(np.concatenate([Z.real, Z.imag], axis=1))


@torch.no_grad()
def chain(model, X, device, bins, amp):
    """Per readout bin f, the seasonal head step by step, each across weeks:
    X_f = rFFT(TCN output) at f (320 complex) -> W_f, the owning band layer's weight at f
    -> Y_f = X_f W_f + b_f (40 complex) -> |Y_f| -> the readout's amplitude at f after the
    per-timestep L2 normalisation (own band's 40 columns, and all 160)."""
    net = model.net.eval()
    xs = []
    for i in range(0, len(X), 64):
        xb = torch.as_tensor(X[i:i + 64], dtype=torch.float32, device=device)
        xs.append(torch.fft.rfft(net(xb, tcn_output=True).float(), dim=1)[:, bins].cpu())
    XF = torch.cat(xs)
    starts = np.cumsum([0] + [layer.out_channels for layer in net.sfd])
    A = amp.reshape(len(X), len(bins), net.seasonal_dims)
    rows = []
    for j, f in enumerate(bins):
        k = next(i for i, layer in enumerate(net.sfd) if layer.start <= f < layer.end)
        layer = net.sfd[k]
        W, b = layer.weight[f - layer.start].detach().cpu(), layer.bias[f - layer.start].detach().cpu()
        Y = XF[:, j] @ W + b
        sv2 = torch.linalg.svdvals(W).numpy() ** 2
        p = sv2 / sv2.sum()
        rows.append([cplx(XF[:, j].numpy()), float(np.exp(-(p * np.log(p)).sum())),
                     cplx(Y.numpy()), erank(Y.abs().numpy()),
                     erank(A[:, j, starts[k]:starts[k + 1]]), erank(A[:, j])])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--fold", default="r0f0")
    ap.add_argument("--npz", default="hrd_2224103.npz")
    ap.add_argument("--weights", default="contracted")
    ap.add_argument("--n-windows", type=int, default=600)
    ap.add_argument("--stride", type=int, default=4, help="timestep subsampling for T x C stages")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    run = Path(a.run)
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    coh = load_npz(a.npz)
    fold = E.fold_from_plan(plan, a.fold)
    pre = np.flatnonzero(~np.isin(coh.pids, list(fold.test_pids)))
    idx = np.sort(np.random.default_rng(0).choice(pre, min(a.n_windows, len(pre)), replace=False))
    X = coh.X[idx]
    T, bpd = X.shape[1], coh.bins_per_day
    i24 = list(spectral_freqs(T, bpd, plan.get("harmonics", 4))).index(T // bpd)

    trained = E.load_encoder(run, a.weights, a.fold, coh, plan, "angle", a.device)
    untrained = E._blank_model(coh, plan, "angle", a.device, seed=fold.model_seed)
    st = stages(trained, X, a.device, a.stride, i24)
    su = stages(untrained, X, a.device, a.stride, i24)

    stored = np.load(run / f"angle_{a.weights}" / a.fold / "repr.npz")["amp"][idx]
    print(f"{a.run} {a.fold}: {len(idx)} non-test windows, timestep stride {a.stride}; "
          f"trained encoder reproduces the stored amp readout: max |diff| = "
          f"{np.abs(st['amp readout'] - stored).max():.2e}\n")
    print(f"{'stage':22s} {'dims':>5s} {'erank trained':>14s} {'erank untrained':>16s} "
          f"{'trained / untrained':>20s}")
    for k in st:
        rt, ru = erank(st[k]), erank(su[k])
        print(f"{k:22s} {st[k].shape[1]:5d} {rt:14.1f} {ru:16.1f} {rt / ru:20.3f}")

    bins = [int(f) for f in spectral_freqs(T, bpd, plan.get("harmonics", 4))]
    ct = chain(trained, X, a.device, bins, st["amp readout"])
    cu = chain(untrained, X, a.device, bins, su["amp readout"])
    cols = ["X_f TCN (640)", "W_f weight (40)", "Y_f band (80)", "|Y_f| (40)",
            "readout own (40)", "readout all (160)"]
    print("\nSeasonal head per readout bin, effective rank: trained / untrained (ratio)")
    print(f"{'bin (period)':14s} " + " ".join(f"{c:>21s}" for c in cols))
    for f, rt, ru in zip(bins, ct, cu):
        per = T / bpd / f
        lab = f"{f} ({per:g} d)" if per >= 1 else f"{f} ({per * 24:g} h)"
        print(f"{lab:14s} " + " ".join(f"{t:6.1f}/{u:6.1f} ({t / u:4.2f})" for t, u in zip(rt, ru)))


if __name__ == "__main__":
    main()
