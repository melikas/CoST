"""Supervised end-to-end baselines: plain TCN / Transformer, no SSL.

Follows the ``cost.py`` pattern -- the architecture is an ``nn.Module`` that owns
no optimiser, and the surrounding function owns the loop. ``models/`` therefore
stays free of training code.

    from baselines.supervised import supervised_baseline_row
    row = supervised_baseline_row(X, y, pids, tr, va, te,
                                  backbone="tcn", pe="none", name="Supervised TCN", ...)
"""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score)

from models.dilated_conv import DilatedConvEncoder
from models.encoder import TransformerFeatureExtractor
from tasks._eval_protocols import best_threshold, participant_aggregate


def _attn_budget(device, fraction=0.6, floor=1024 ** 3):
    """Bytes this rung may spend on attention: a fraction of what the card has FREE right now.

    A fixed constant would be wrong on both ends -- too small on a 40 GB A100 (needlessly slow)
    and still too large on a 20 GB MIG slice once the rest of the pipeline is resident. Reading
    the free memory adapts to whichever the job landed on, and to whatever the preceding stages
    are still holding. The fraction leaves room for parameters, gradients, optimiser state and
    allocator fragmentation, none of which this figure covers.
    """
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return 3 * 1024 ** 3
    free, _ = torch.cuda.mem_get_info(torch.device(device).index or 0)
    return max(floor, int(free * fraction))


def _attn_micro_batch(backbone, depth, output_dims, T, training, budget=3 * 1024 ** 3):
    """Largest batch whose self-attention matrices fit in `budget`, or None for a conv net.

    A transformer layer materialises a (B, heads, T, T) score matrix and its softmax. At
    T=672, heads=8, repr_dims=32, depth=10 that is 8*672*672*4*2 = 28.9 MB per sample per
    layer; training keeps every layer for the backward pass, so a batch of 64 needs ~18.5 GB
    -- which is what killed every task of run 1608369/1612926 on a 20 GB MIG slice, at the
    exact number the CUDA error reported (18.92 GiB live).

    The dilated-conv backbone is linear in T and needs no cap, so it returns None and the
    original single-backward path is used unchanged.
    """
    if backbone != "transformer":
        return None
    n_heads = 8 if output_dims % 8 == 0 else 1     # mirrors TransformerFeatureExtractor
    per_sample = n_heads * T * T * 4 * 2           # scores + softmax, fp32
    if training:
        per_sample *= max(1, int(depth))           # every layer is kept for the backward
    return max(1, int(budget // per_sample))


# --------------------------------------------------------------------------- #
# Supervised end-to-end baselines (plain TCN / Transformer, no SSL) for the
# separability table: the existing backbone -> masked mean-pool -> linear head,
# trained directly on the depression label.
# --------------------------------------------------------------------------- #
class _SupervisedNet(nn.Module):
    """input_fc (sensors) [+ time-feature PE] -> backbone -> masked mean-pool -> head.

    Reuses the existing backbones (DilatedConvEncoder / TransformerFeatureExtractor);
    drops the TFD/SFD and the SSL -- a plain supervised classifier over the same backbone."""

    def __init__(self, input_dims, n_time_features, backbone, pe,
                 hidden_dims, depth, output_dims):
        super().__init__()
        self.n_time = int(n_time_features)
        self.n_sensor = input_dims - self.n_time
        self.input_fc = nn.Linear(self.n_sensor, hidden_dims)
        self.time_fc = nn.Linear(self.n_time, hidden_dims) if self.n_time > 0 else None
        if backbone == "transformer":
            self.backbone = TransformerFeatureExtractor(
                hidden_dims, output_dims, depth=depth, pe=pe)
        else:
            self.backbone = DilatedConvEncoder(
                hidden_dims, [hidden_dims] * depth + [output_dims], kernel_size=3)
        self.head = nn.Linear(output_dims, 1)

    def forward(self, x):                              # x: (B, T, input_dims)
        if self.n_time > 0:
            xt = torch.nan_to_num(x[..., self.n_sensor:], nan=0.0)
            x = x[..., :self.n_sensor]
        nan_mask = ~torch.isnan(x).any(dim=-1)         # (B, T) True where the watch was worn
        h = self.input_fc(torch.nan_to_num(x, nan=0.0))
        if self.time_fc is not None:
            h = h + self.time_fc(xt)
        h = h * nan_mask.unsqueeze(-1)                 # zero non-wear timesteps
        feats = self.backbone(h.transpose(1, 2))       # (B, output_dims, T)
        m = nan_mask.unsqueeze(1).float()
        pooled = (feats * m).sum(dim=2) / m.sum(dim=2).clamp(min=1.0)   # masked mean over time
        return self.head(pooled).squeeze(-1)           # (B,) logits


def supervised_baseline_row(X, y, pids, train_mask, val_mask, test_mask, backbone, pe, name,
                            n_time_features, hidden_dims, depth, output_dims, device="cuda",
                            max_epochs=60, patience=12, lr=1e-3, batch_size=64, seed=42,
                            return_scores=False, return_window_scores=False):
    """Train the supervised backbone end-to-end and return a separability-table row dict
    (same keys as `separability_table`). Best epoch chosen by participant-level validation
    AUC (early stopping); F1/Acc use a val-tuned threshold; AUC is threshold-free."""
    torch.manual_seed(seed); np.random.seed(seed)
    X = np.asarray(X, dtype=np.float32); y = np.asarray(y)
    Xt = torch.from_numpy(X); yt = torch.from_numpy(y.astype(np.float32))
    net = _SupervisedNet(X.shape[-1], n_time_features, backbone, pe,
                         hidden_dims, depth, output_dims).to(device)
    # Measured AFTER the net is resident, so the budget reflects what is actually left.
    _budget = _attn_budget(device)

    ytr = y[train_mask]
    n_pos = max(1, int((ytr == 1).sum())); n_neg = max(1, int((ytr == 0).sum()))
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / n_pos], device=device))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    tr_idx = np.where(train_mask)[0]
    loader = DataLoader(TensorDataset(Xt[tr_idx], yt[tr_idx]),
                        batch_size=batch_size, shuffle=True, drop_last=False)

    # Inference frees each layer as it goes, so only ONE score matrix is live at a time --
    # hence training=False. Still capped: at batch 256 that single matrix is 3.7 GB.
    eval_chunk = min(256, _attn_micro_batch(backbone, depth, output_dims, X.shape[1],
                                            training=False, budget=_budget) or 256)

    def predict(mask):
        net.eval(); idx = np.where(mask)[0]; out = []
        with torch.no_grad():
            for i in range(0, len(idx), eval_chunk):
                out.append(torch.sigmoid(
                    net(Xt[idx[i:i + eval_chunk]].to(device))).cpu().numpy())
        return np.concatenate(out) if out else np.array([])

    # Micro-batching, NOT a smaller batch: the optimiser still steps once per `batch_size`
    # examples, and the accumulated gradient is exactly the full-batch one (verified to fp32
    # noise, 3e-8). What it does NOT reproduce bit-for-bit is the training TRAJECTORY --
    # dropout samples a fresh mask per forward call, so a batch split into chunks draws
    # different masks than one whole-batch forward. The objective, the effective batch and the
    # step count are unchanged; only the dropout noise differs, as it would between any two
    # seeds. Do not describe these numbers as identical to a pre-cap run, only as comparable.
    micro = _attn_micro_batch(backbone, depth, output_dims, X.shape[1], training=True,
                              budget=_budget)
    if micro is not None and micro < batch_size:
        print(f"[baseline] {name}: attention needs micro-batches of {micro} "
              f"({_budget / 2 ** 30:.1f} GiB free-memory budget); the optimiser still steps "
              f"once per {batch_size} examples and the accumulated gradient is the full-batch "
              f"one, but dropout masks are redrawn per chunk.")

    has_val = int(np.sum(val_mask)) > 0 and not np.array_equal(val_mask, train_mask)
    best_auc, best_state, bad = -1.0, None, 0
    for _ in range(max_epochs):
        net.train()
        for xb, yb in loader:
            opt.zero_grad()
            nb = len(xb)
            step = nb if micro is None else micro
            for i in range(0, nb, step):
                xs, ys = xb[i:i + step], yb[i:i + step]
                # BCEWithLogitsLoss reduces by 'mean' over elements (pos_weight scales the
                # summands, it does not renormalise), so weighting each chunk by its share of
                # the batch makes the accumulated gradient EXACTLY the full-batch gradient.
                (crit(net(xs.to(device)), ys.to(device)) * (len(xs) / nb)).backward()
            opt.step()
        vmask = val_mask if has_val else train_mask
        vp, vl = participant_aggregate(pids[vmask], predict(vmask), y[vmask])
        vauc = roc_auc_score(vl, vp) if len(np.unique(vl)) > 1 else 0.5
        if vauc > best_auc:
            best_auc, bad = vauc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)

    thr_mask = val_mask if has_val else train_mask
    vp, vl = participant_aggregate(pids[thr_mask], predict(thr_mask), y[thr_mask])
    thr = best_threshold(vl, vp)
    tprob = predict(test_mask); yte = y[test_mask]
    w_auc = roc_auc_score(yte, tprob) if len(np.unique(yte)) > 1 else float("nan")
    w_pred = (tprob >= thr).astype(int)
    pp, pl = participant_aggregate(pids[test_mask], tprob, yte)
    p_auc = roc_auc_score(pl, pp) if len(np.unique(pl)) > 1 else float("nan")
    p_pred = (pp >= thr).astype(int)
    row = {
        "Representation": name, "Dim": "e2e", "Thr": float(thr),
        "Win AUC": float(w_auc), "Win F1": float(f1_score(yte, w_pred, zero_division=0)),
        "Win Acc": float(accuracy_score(yte, w_pred)),
        "Win BAcc": float(balanced_accuracy_score(yte, w_pred)),
        "Win MCC": float(matthews_corrcoef(yte, w_pred)),
        "Subj AUC": float(p_auc), "Subj F1": float(f1_score(pl, p_pred, zero_division=0)),
        "Subj Acc": float(accuracy_score(pl, p_pred)),
        "Subj BAcc": float(balanced_accuracy_score(pl, p_pred)),
        "Subj MCC": float(matthews_corrcoef(pl, p_pred)),
    }
    # `return_scores` hands back the participant-level probabilities and labels as well.
    # RQ3's Delta AUC is bootstrapped on SHARED participant draws, so it needs this rung's
    # per-participant scores, not just its summary AUC -- and `participant_aggregate` orders
    # by np.unique(pids[test_mask]), the same order experiment_q3.per_subject uses, so the
    # two vectors are row-aligned and the contrast is genuinely paired.
    # Hand the GPU back before returning. Without this the net, its optimiser state and the
    # allocator's cached blocks stay live for the whole of the caller's remaining work, which
    # is why the SECOND rung in the ladder was the one that ran out of memory.
    def _release(value):
        nonlocal net, opt, crit
        del net, opt, crit
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        return value

    if return_scores:
        return _release((row, pp, pl))
    # WINDOW-level scores, for a downstream whose unit is the window rather than the
    # participant (the emotional-energy tasks: one row per labelled day). `thr` travels with
    # them because it was tuned on the validation split, and re-tuning it on test would be
    # the one thing this rung must not do.
    if return_window_scores:
        return _release((row, tprob, float(thr)))
    return _release(row)
