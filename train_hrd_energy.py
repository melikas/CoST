"""Pretrain CoST from scratch, then probe the emotional-energy (EE) target.

Sibling of ``train_hrd.py``: same encoder, same pretraining, a different downstream. The
windows are SLIDING trailing weeks -- one 7-day window per labelled day ([D-6, D] -> EE(D)),
stride 1 day -- so the probe answers a day-resolution question the endpoint task cannot.

The probes themselves live in ``tasks/energy.py`` and are shared with
``train_hrd.py --energy-probe``, which reuses an already-trained encoder instead of fitting
one; this script is the standalone path.

Run:  python train_hrd_energy.py --sensor-csv datasets/HRD_RAW_MinuteLevel.csv \
          --output-dir results_hrd_energy --run-id local
"""
import argparse
import time

import numpy as np

from model_build import paper_kernels
from tasks.energy import run_energy_tasks, split_pids


def parse_args():
    p = argparse.ArgumentParser(description="Emotional-energy probe on CoST representations")
    p.add_argument("--sensor-csv", required=True)
    p.add_argument("--backbone", default="transformer",
                   choices=["tcn", "transformer"])
    p.add_argument("--pe", default=None)
    p.add_argument("--pool", choices=["last", "mean", "max", "meanmax"], default="last",
                   help="How the 7-day representation is collapsed before the probe. "
                        "'last' (default) = final timestep, closest to the labelled day.")
    p.add_argument("--season-pool", choices=["spec", "spec_amp", "spec_phase", "same"],
                   default="spec",
                   help="Readout of the SEASONAL half only; see train_hrd.py --season-pool. "
                        "'same' = use --pool for it too (the DC/MESOR-only ablation).")
    p.add_argument("--energy-threshold", type=float, default=4.0,
                   help="EE >= this = 'high-energy day' (default 4 -> ~47%% positive, balanced).")
    p.add_argument("--window-hours", type=int, default=168)
    p.add_argument("--bin-minutes", type=int, default=15)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--repr-dims", type=int, default=320)
    p.add_argument("--hidden-dims", type=int, default=64)
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output-dir", default="./results_hrd_energy")
    p.add_argument("--run-id", default="local")
    return p.parse_args()


def main():
    from pathlib import Path

    from cost import CoST
    from data_processing.data_preprocessing import prepare_hrd_energy_sliding
    from utils import init_dl_program

    args = parse_args()
    if args.pe is None:
        args.pe = "sinusoidal" if args.backbone == "transformer" else "none"
    device = init_dl_program(args.gpu if args.gpu >= 0 else "cpu", seed=args.seed)
    t0 = time.time()

    # 1. data: SLIDING trailing windows -- one 7-day window per labelled day ([D-6, D] ->
    # EE(D)), stride 1 day (Day1-7->EE7, Day2-8->EE8, ...). X / ee / pids = PROBE windows;
    # X_pre / pids_pre = the label-free non-overlapping PRETRAIN windows (same CSV read).
    data = prepare_hrd_energy_sliding(args.sensor_csv, window_hours=args.window_hours,
                                      bin_minutes=args.bin_minutes)
    X, ee, pids = data["X"], data["ee"], data["pids"]
    X_pre, pids_pre = data["X_pretrain"], data["pids_pretrain"]
    n_sensors = data["n_sensors"]
    lab = np.isfinite(ee)                                   # windows with a last-day EE label
    pids_with_label = set(pids[lab])

    # 2. participant-level split -----------------------------------------------------
    tr_pids, va_pids, te_pids = split_pids(pids_with_label, args.seed,
                                           args.val_frac, args.test_frac)
    tr = np.isin(pids, list(tr_pids)) & lab
    va = np.isin(pids, list(va_pids)) & lab
    te = np.isin(pids, list(te_pids)) & lab
    pretrain_mask = ~np.isin(pids_pre, list(te_pids))      # label-free; excludes test pids only
    print(f"[split] participants: {len(tr_pids)} train / {len(va_pids)} val / {len(te_pids)} test")
    print(f"[split] labelled probe windows: {int(tr.sum())} train / {int(va.sum())} val / "
          f"{int(te.sum())} test | pretrain windows: {int(pretrain_mask.sum())}")

    # 3. CoST self-supervised pretraining (unchanged; label-free) --------------------
    seq_len = X.shape[1]
    model = CoST(input_dims=X.shape[-1], n_time_features=X.shape[-1] - n_sensors,
                 kernels=paper_kernels(seq_len), alpha=0.0005, max_train_length=seq_len,
                 output_dims=args.repr_dims, hidden_dims=args.hidden_dims, depth=args.depth,
                 backbone=args.backbone, pe=args.pe,
                 bins_per_day=(24 * 60 // args.bin_minutes),
                 device=device, lr=args.lr, batch_size=args.batch_size)
    print(f"[pretrain] backbone={args.backbone} pe={args.pe} on {int(pretrain_mask.sum())} windows ...")
    model.fit(X_pre[pretrain_mask], n_epochs=args.epochs, n_iters=args.iters, verbose=True)

    # 4. freeze -> encode -> 3 energy tasks -> report (shared with train_hrd --energy-probe)
    mode_desc = ("**Sliding trailing windows**: one 7-day window per labelled day "
                 "([D-6, D] -> EE(D)), stride 1 day; windows overlap by 6 days, split stays "
                 "participant-level so test is out of pretraining.")
    out = Path(args.output_dir) / args.run_id / f"{args.backbone}_{args.pe}_seed{args.seed}"
    run_energy_tasks(model, X, ee, pids, n_sensors, tr, va, te,
                     args.pool, args.energy_threshold, args.seed, out, mode_desc, vars(args),
                     season_pool=None if args.season_pool == "same" else args.season_pool,
                     ee_win=data.get("ee_win"))
    print(f"total time = {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
