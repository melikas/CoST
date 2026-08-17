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
                            return_scores=False):
    """Train the supervised backbone end-to-end and return a separability-table row dict
    (same keys as `separability_table`). Best epoch chosen by participant-level validation
    AUC (early stopping); F1/Acc use a val-tuned threshold; AUC is threshold-free."""
    torch.manual_seed(seed); np.random.seed(seed)
    X = np.asarray(X, dtype=np.float32); y = np.asarray(y)
    Xt = torch.from_numpy(X); yt = torch.from_numpy(y.astype(np.float32))
    net = _SupervisedNet(X.shape[-1], n_time_features, backbone, pe,
                         hidden_dims, depth, output_dims).to(device)

    ytr = y[train_mask]
    n_pos = max(1, int((ytr == 1).sum())); n_neg = max(1, int((ytr == 0).sum()))
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / n_pos], device=device))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    tr_idx = np.where(train_mask)[0]
    loader = DataLoader(TensorDataset(Xt[tr_idx], yt[tr_idx]),
                        batch_size=batch_size, shuffle=True, drop_last=False)

    def predict(mask):
        net.eval(); idx = np.where(mask)[0]; out = []
        with torch.no_grad():
            for i in range(0, len(idx), 256):
                out.append(torch.sigmoid(net(Xt[idx[i:i + 256]].to(device))).cpu().numpy())
        return np.concatenate(out) if out else np.array([])

    has_val = int(np.sum(val_mask)) > 0 and not np.array_equal(val_mask, train_mask)
    best_auc, best_state, bad = -1.0, None, 0
    for _ in range(max_epochs):
        net.train()
        for xb, yb in loader:
            opt.zero_grad(); crit(net(xb.to(device)), yb.to(device)).backward(); opt.step()
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
    if return_scores:
        return row, pp, pl
    return row
