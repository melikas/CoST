"""Plain-SSL baseline: the SAME CoST, pretrained with the trend/seasonal disentangler OFF.

This is the control that isolates what the TFD/SFD split buys. Everything else -- backbone,
PE, dims, augmentation, loss, iteration count, pretrain windows -- is held fixed; only
``disentangle=False`` changes, so the encoder emits ONE representation of ``repr_dims``
instead of the concatenated [V^(T); V^(S)] (cost.py:246 for the loss, cost.py:710 for the
readout).

    from baselines.plain_ssl import plain_ssl_encoder, plain_ssl_baseline_row

WARNING -- unlike every other baseline in this project, this one is NOT free: it is a second
self-supervised pretraining per (seed, variant). Budget for it explicitly. The trained weights
are cached to `cache_path`, so the first caller pays and later ones reload.

Deliberately NOT importing train_hrd: train_hrd imports baselines.*, so the reverse edge
would be a cycle. The encoder comes from model_build instead, which both sides share.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model_build import build_model
from tasks._eval_protocols import best_threshold, participant_aggregate




def plain_ssl_encoder(X, pretrain_mask, cfg, n_sensors, device, seed=42, cache_path=None,
                      verbose=True, pids=None):
    """Pretrain (or reload) the non-disentangled twin of this run's encoder, frozen on return.

    `pretrain_mask` must be the SAME window set the disentangled model saw -- i.e. every
    non-test window -- or the comparison is confounded by data rather than by the split.
    """
    # `disentangle=False` IS the control; every other argument must match the
    # disentangled twin, which is why this goes through the shared builder.
    model = build_model(cfg, X, n_sensors, device, disentangle=False)

    # The cache is keyed by the PRETRAINING CONFIG, not by the file path alone. This control
    # only means anything if it saw exactly what the disentangled model saw, and a cached file
    # carries no record of that: a plain_encoder.pt written before --mask-mode changed would be
    # reloaded silently and the "only the split differs" claim would be false -- the control
    # would be the one without masking and the model the one with it. A sidecar .json holds the
    # settings the twin was trained under; any mismatch retrains instead of reloading.
    cache_path = Path(cache_path) if cache_path else None
    key = {k: cfg.get(k) for k in ("backbone", "pe", "repr_dims", "hidden_dims", "depth",
                                   "alpha", "lr", "batch_size", "iters", "epochs",
                                   "jitter_sigma", "mask_mode", "mask_keep_prob",
                                   "phase_encoding", "loss_balance", "time2vec_dim",
                                   "shift_sigma", "moco_k", "trend_pool")}
    key["n_pretrain_windows"] = int(pretrain_mask.sum())
    key["seed"] = int(seed)
    key_path = cache_path.with_suffix(".key.json") if cache_path else None

    cached = (cache_path is not None and cache_path.exists()
              and key_path.exists()
              and json.loads(key_path.read_text(encoding="utf-8")) == key)
    if cache_path is not None and cache_path.exists() and not cached and verbose:
        print(f"[plain-ssl] cache {cache_path.name} was trained under a DIFFERENT config "
              f"(or has no key file) -- retraining rather than reusing a mismatched control.")
    if cached:
        model.load(cache_path)
        if verbose:
            print(f"[plain-ssl] reloaded cached plain encoder -> {cache_path.name}")
    else:
        if verbose:
            print(f"[plain-ssl] pretraining the non-disentangled twin on "
                  f"{int(pretrain_mask.sum())} windows (this is a REAL pretraining) ...")
        np.random.seed(seed)
        # the control must see the SAME pretext task, or it stops being a control for it
        model.fit(X[pretrain_mask],
                  pids=None if pids is None else np.asarray(pids)[pretrain_mask],
                  n_iters=cfg.get("iters"), n_epochs=cfg.get("epochs"),
                  verbose=verbose)
        if cache_path is not None:
            model.save(cache_path)
            key_path.write_text(json.dumps(key, indent=2, sort_keys=True), encoding="utf-8")
    model.net.eval()
    return model


def encode_plain(model, X, cfg, batch=256):
    """Plain mode returns a single representation, so season_pool never applies."""
    return model.encode(X, mode="forecasting", pool=cfg["pool"], batch_size=batch).squeeze(1)


def plain_ssl_baseline_row(X, y, pids, train_mask, val_mask, test_mask, pretrain_mask, cfg,
                           n_sensors, name="DSSL plain (no disentangle)", device="cuda",
                           seed=42, probe_c=0.1, cache_path=None):
    """Separability-table row dict, same keys as `supervised_baseline_row`.

    The probe, the threshold rule and the participant aggregation are the ones the
    disentangled representation is scored with, so only the representation differs.
    """
    model = plain_ssl_encoder(X, pretrain_mask, cfg, n_sensors, device, seed, cache_path,
                              pids=pids)
    R = encode_plain(model, X, cfg)

    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=probe_c, max_iter=3000,
                                           class_weight="balanced", random_state=seed))
    clf.fit(R[train_mask], y[train_mask])

    thr_mask = val_mask if (val_mask.any() and not np.array_equal(val_mask, train_mask)) \
        else train_mask
    vp, vl = participant_aggregate(pids[thr_mask], clf.predict_proba(R[thr_mask])[:, 1],
                                   y[thr_mask])
    thr = best_threshold(vl, vp)

    tprob, yte = clf.predict_proba(R[test_mask])[:, 1], y[test_mask]
    pp, pl = participant_aggregate(pids[test_mask], tprob, yte)

    def block(prefix, yt, prob):
        pred = (prob >= thr).astype(int)
        return {f"{prefix} AUC": float(roc_auc_score(yt, prob)) if len(np.unique(yt)) > 1
                else float("nan"),
                f"{prefix} F1": float(f1_score(yt, pred, zero_division=0)),
                f"{prefix} Acc": float(accuracy_score(yt, pred)),
                f"{prefix} BAcc": float(balanced_accuracy_score(yt, pred)),
                f"{prefix} MCC": float(matthews_corrcoef(yt, pred))}

    return {"Representation": name, "Dim": int(R.shape[1]), "Thr": float(thr),
            **block("Win", yte, tprob), **block("Subj", pl, pp)}
