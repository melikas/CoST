"""Train CoST on the HRD wearable data for depression-endpoint classification.

Pipeline (leakage-safe, participant-level throughout):
  1. data_preprocessing.prepare_hrd_dataset -> windows X (N, T, C), labels y, pids
  2. hold out 30% of the CONSISTENT (baseline==endpoint) participants as the test set
  3. self-supervised pretraining of the CoST encoder on ALL non-test windows
  4. encode windows -> representations; fit a logistic-regression classifier on the
     remaining 70% consistent cohort (a participant-level val split picks the
     decision threshold); evaluate on the held-out test set
  5. report window- and participant-level AUC / F1 / accuracy

Run:  python train_hrd.py --sensor-csv datasets/HRD_RAW_MinuteLevel.csv --backbone transformer
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cost import CoST
from data_preprocessing import prepare_hrd_dataset
from utils import init_dl_program


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def stratified_pid_holdout(unique_pids, pid_label, frac, seed):
    """Split participant ids into (rest, held) at the participant level."""
    pids = sorted(unique_pids)
    if len(pids) < 2 or frac <= 0:
        return set(pids), set()
    y = [pid_label.get(p, 0) for p in pids]
    n_held = min(max(1, int(round(len(pids) * frac))), len(pids) - 1)
    try:
        rest, held = train_test_split(pids, test_size=n_held, stratify=y, random_state=seed)
    except ValueError:
        rng = np.random.default_rng(seed)
        perm = list(rng.permutation(np.array(pids)))
        held, rest = perm[:n_held], perm[n_held:]
    return set(rest), set(held)


def participant_aggregate(pids, probs, labels):
    """Mean window probability per participant + the participant label."""
    uniq = np.unique(pids)
    pid_prob = np.array([probs[pids == p].mean() for p in uniq], dtype=np.float64)
    pid_lbl = np.array([int(labels[pids == p][0]) for p in uniq], dtype=int)
    return pid_prob, pid_lbl


def best_threshold(y_true, y_prob):
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 37):
        score = f1_score(y_true, (y_prob >= thr).astype(int), zero_division=0)
        if score > best_f1:
            best_f1, best_thr = score, float(thr)
    return best_thr


def binary_metrics(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    return {
        "auc_roc": float(auroc),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(thr),
    }


def paper_kernels(seq_len):
    """CoST mixture-of-AR-experts kernels: powers of 2 up to floor(log2(T/2))."""
    L = max(0, int(math.floor(math.log2(max(seq_len // 2, 1)))))
    return [2 ** i for i in range(L + 1)]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="CoST on HRD: depression-endpoint classification")
    p.add_argument("--sensor-csv", required=True, help="Path to HRD_RAW_MinuteLevel.csv")
    p.add_argument("--label-col", default="depression_status_endpoint")
    p.add_argument("--window-hours", type=int, default=168)
    p.add_argument("--bin-minutes", type=int, default=15)
    p.add_argument("--max-missing", type=float, default=0.30,
                   help="Drop participants with more than this fraction of wear-channel missingness")
    p.add_argument("--max-window-missing", type=float, default=0.30,
                   help="Drop windows with more than this fraction of empty time-bins")
    p.add_argument("--no-zscore", action="store_true", help="Disable per-participant z-scoring")
    p.add_argument("--test-frac", type=float, default=0.30,
                   help="Fraction of CONSISTENT participants held out for the test set")
    p.add_argument("--val-frac", type=float, default=0.25,
                   help="Fraction of the fine-tune cohort used to pick the decision threshold")
    # CoST encoder / pretraining
    p.add_argument("--backbone", default="tcn", choices=["tcn", "transformer"])
    p.add_argument("--repr-dims", type=int, default=320)
    p.add_argument("--hidden-dims", type=int, default=64)
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--kernels", type=int, nargs="+", default=None,
                   help="AR-expert kernel sizes (default: CoST powers-of-2 from window length)")
    p.add_argument("--alpha", type=float, default=0.0005)
    p.add_argument("--max-train-length", type=int, default=None,
                   help="Crop length for pretraining; defaults to the window length T "
                        "(the CoST Fourier layer is sized to this, so it must equal T)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--iters", type=int, default=None, help="Pretraining iterations")
    p.add_argument("--epochs", type=int, default=None, help="Pretraining epochs")
    # misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0, help="GPU index, or a negative value to force CPU")
    p.add_argument("--max-threads", type=int, default=None)
    p.add_argument("--output-dir", default="./results_hrd")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = "cpu" if args.gpu < 0 else args.gpu
    device = init_dl_program(dev, seed=args.seed, max_threads=args.max_threads)
    t_start = time.time()

    # 1. data -------------------------------------------------------------
    data = prepare_hrd_dataset(
        args.sensor_csv,
        window_hours=args.window_hours,
        bin_minutes=args.bin_minutes,
        label_col=args.label_col,
        max_missing=args.max_missing,
        max_window_missing=args.max_window_missing,
        z_score=not args.no_zscore,
    )
    X, y, pids = data["X"], data["y"], data["pids"]
    consistent_pids = data["consistent_pids"]
    pid_label = {p: int(y[pids == p][0]) for p in np.unique(pids)}

    # 2. leakage-safe split ----------------------------------------------
    rest_cons, test_pids = stratified_pid_holdout(
        consistent_pids, pid_label, args.test_frac, args.seed
    )
    if not test_pids:
        raise RuntimeError("Test holdout is empty; check --test-frac and the consistent cohort.")
    test_mask = np.isin(pids, list(test_pids))
    pretrain_mask = ~test_mask                                  # ALL non-test windows
    finetune_mask = np.isin(pids, list(rest_cons))              # 70% consistent cohort
    print(f"[split] pretrain windows={int(pretrain_mask.sum())} | "
          f"fine-tune windows={int(finetune_mask.sum())} ({len(rest_cons)} pids) | "
          f"test windows={int(test_mask.sum())} ({len(test_pids)} pids)")

    # 3. CoST self-supervised pretraining --------------------------------
    seq_len = X.shape[1]
    kernels = args.kernels if args.kernels is not None else paper_kernels(seq_len)
    # The CoST seasonal (Fourier) layer is sized to max_train_length, so it must
    # equal the window length T; clamp any larger request down to T.
    max_train_length = seq_len if args.max_train_length is None else min(args.max_train_length, seq_len)
    model = CoST(
        input_dims=X.shape[-1],
        kernels=kernels,
        alpha=args.alpha,
        max_train_length=max_train_length,
        output_dims=args.repr_dims,
        hidden_dims=args.hidden_dims,
        depth=args.depth,
        backbone=args.backbone,
        device=device,
        lr=args.lr,
        batch_size=args.batch_size,
    )
    print(f"[pretrain] CoST backbone={args.backbone} on {int(pretrain_mask.sum())} windows ...")
    loss_log = model.fit(X[pretrain_mask], n_epochs=args.epochs, n_iters=args.iters, verbose=True)
    pretrain_seconds = time.time() - t_start

    # 4. representations + classifier ------------------------------------
    reprs = model.encode(X, mode="forecasting").squeeze(1)      # (N, repr_dims)

    ft_pids = sorted(rest_cons)
    rem_pids, val_pids = stratified_pid_holdout(ft_pids, pid_label, args.val_frac, args.seed)
    train_mask = np.isin(pids, list(rem_pids))
    val_mask = np.isin(pids, list(val_pids)) if val_pids else train_mask

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=args.seed),
    )
    clf.fit(reprs[train_mask], y[train_mask])

    def predict(mask):
        return clf.predict_proba(reprs[mask])[:, 1]

    # threshold chosen on the participant-aggregated validation split
    val_pid_prob, val_pid_lbl = participant_aggregate(pids[val_mask], predict(val_mask), y[val_mask])
    thr = best_threshold(val_pid_lbl, val_pid_prob)

    # 5. evaluate on the held-out test set -------------------------------
    test_prob = predict(test_mask)
    win = binary_metrics(y[test_mask], test_prob, thr)
    pid_prob, pid_lbl = participant_aggregate(pids[test_mask], test_prob, y[test_mask])
    pid = binary_metrics(pid_lbl, pid_prob, thr)

    result = {
        "backbone": args.backbone,
        "window_level": win,
        "participant_level": pid,
        "n_test_windows": int(test_mask.sum()),
        "n_test_participants": int(len(np.unique(pids[test_mask]))),
        "n_pretrain_windows": int(pretrain_mask.sum()),
        "n_finetune_windows": int(finetune_mask.sum()),
        "pretrain_seconds": pretrain_seconds,
        "seq_len": int(seq_len),
        "n_features": int(X.shape[-1]),
        "config": vars(args),
    }
    (out_dir / f"metrics_{args.backbone}_seed{args.seed}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    np.save(out_dir / f"pretrain_loss_{args.backbone}_seed{args.seed}.npy", np.asarray(loss_log))

    print("\n========== HELD-OUT TEST RESULTS ==========")
    print(f"backbone = {args.backbone}")
    print(f"window-level       AUC={win['auc_roc']:.3f}  F1={win['f1']:.3f}  Acc={win['accuracy']:.3f}")
    print(f"participant-level  AUC={pid['auc_roc']:.3f}  F1={pid['f1']:.3f}  Acc={pid['accuracy']:.3f}")
    print(f"decision threshold = {thr:.3f}  (tuned on val)")
    print(f"saved -> {out_dir / f'metrics_{args.backbone}_seed{args.seed}.json'}")
    print(f"total time = {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
