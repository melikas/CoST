"""Constructing a CoST encoder from a run config -- the one place that does it.

Before this module the same construction existed four times: `train_hrd.build` (the real
training run), `tasks/_experiment_common._build_model` (the experiment scripts),
`train_hrd._random_init_repr` (the negative control) and `baselines/plain_ssl` (the
non-disentangled twin). They drifted: the negative control was built WITHOUT `loss_balance`,
`mask_mode` or `phase_mode`, so "same architecture, weights never trained" was true only by
accident of those arguments adding no parameters.

It also sits below everything so the import graph is a tree. `tasks/_experiment_common`
imported `train_hrd`, which imported it back; the cycle was worked around three separate
times -- a copied `_paper_kernels` in `baselines/plain_ssl`, a function-scope import in
`tasks/energy`, and another in `train_hrd` for `PLAIN_REF`. This module imports only `cost`,
so those workarounds are gone.
"""
import math

import torch

from cost import CoST


def paper_kernels(seq_len):
    """CoST mixture-of-AR-experts kernels: powers of 2 up to floor(log2(T/2))."""
    L = max(0, int(math.floor(math.log2(max(seq_len // 2, 1)))))
    return [2 ** i for i in range(L + 1)]


def build_model(cfg, X, n_sensors, device, **override):
    """The encoder `cfg` describes, over windows shaped like `X`.

    `cfg` is a run's `metrics.json["config"]` (equivalently `vars(args)`). `override` is for
    the one caller that legitimately differs: the plain-SSL control passes
    `disentangle=False`, which IS the control.
    """
    seq_len = X.shape[1]
    mtl = cfg.get("max_train_length")
    kw = dict(
        input_dims=X.shape[-1], n_time_features=int(X.shape[-1]) - int(n_sensors),
        kernels=cfg.get("kernels") or paper_kernels(seq_len), alpha=cfg["alpha"],
        max_train_length=seq_len if mtl is None else min(mtl, seq_len),
        output_dims=cfg["repr_dims"], hidden_dims=cfg["hidden_dims"], depth=cfg["depth"],
        backbone=cfg["backbone"], pe=cfg["pe"], time2vec_dim=cfg["time2vec_dim"],
        loss_balance=cfg["loss_balance"], bins_per_day=24 * 60 // cfg["bin_minutes"],
        disentangle=cfg["disentangle"], jitter_sigma=cfg["jitter_sigma"],
        # Configs written before this key existed cannot say which layout they used, and the
        # answer differs between them, so no default is right for all of them. Loading is safe
        # either way -- CoST.load rebuilds `sfd` from the checkpoint. What is NOT safe is the
        # negative control, which is built from the config with no checkpoint to correct it;
        # tasks/_experiment_common.random_init_model therefore mirrors the layout of the
        # encoder it is the control FOR, rather than trusting this default.
        seasonal_bands=cfg.get("seasonal_bands", "harmonics"),
        shift_sigma=cfg.get("shift_sigma", 0.5), moco_k=cfg.get("moco_k", 4096),
        trend_pool=cfg.get("trend_pool", "random"),
        negatives=cfg.get("negatives", "global"),
        # 0 = the whole queue, i.e. the shipped denominator. The fallback MUST match
        # cost.py's own default: a config written before this key existed (every run in
        # results_hrd/) has no "n_negatives", and a non-zero fallback would silently
        # rebuild those models against 32 sampled negatives instead of all 4096.
        n_negatives=cfg.get("n_negatives", 0),
        positive_pair=cfg.get("positive_pair", "window"),
        # 0 = no smoothing augmentation, matching cost.py's own default, so every config
        # written before this key existed rebuilds exactly the model it was.
        smooth_bins=cfg.get("smooth_bins", 0),
        # "angle" is the fallback ON PURPOSE: a config written before this key existed
        # ran the raw-atan2 readout, and rebuilding it any other way would reproduce a
        # different model than the one whose numbers are archived beside it.
        phase_readout=cfg.get("phase_readout", "angle"),
        # 0.0 = no V^N branch at all, matching cost.py's own default, so a config written
        # before this key existed rebuilds exactly the model it was -- including the
        # random-init control, which is built from the config with no checkpoint to
        # correct it and would otherwise be a different architecture than the encoder it
        # is the control FOR.
        noise_weight=cfg.get("noise_weight", 0.0),
        noise_branch=cfg.get("noise_branch", False),
        noise_depth=cfg.get("noise_depth", None),
        noise_mask_frac=cfg.get("noise_mask_frac", 0.3),
        noise_span=cfg.get("noise_span", 8),
        mask_mode=cfg["mask_mode"], mask_prob=cfg["mask_keep_prob"],
        phase_mode=cfg["phase_encoding"], device=device, lr=cfg["lr"],
        batch_size=cfg["batch_size"])
    kw.update(override)
    return CoST(**kw)


def random_init_model(cfg, X, n_sensors, device, seed):
    """The negative control of every ladder in the project: same architecture, weights never
    updated. Seeded explicitly -- an unseeded control redraws its weights on every invocation,
    so the same variant would report a different control score each run and the comparison it
    exists to support would not be reproducible."""
    torch.manual_seed(seed)
    m = build_model(cfg, X, n_sensors, device)
    m.net.eval()
    return m


def encode_repr(model, X, cfg, pool=None, batch=256):
    """Window-pooled [V^(T); V^(S)] -- the readout the probes actually read.

    `pool` overrides `cfg["pool"]` for the energy path, which pools differently; that is the
    only difference between the depression and energy readouts.
    """
    sp = None if cfg.get("season_pool") == "same" else cfg.get("season_pool")
    return model.encode(X, mode="forecasting", pool=pool or cfg["pool"], season_pool=sp,
                        batch_size=batch).squeeze(1)


def random_init_repr(cfg, X, n_sensors, device, seed, pool=None):
    """`random_init_model` already encoded, for callers that want the vectors not the model."""
    return encode_repr(random_init_model(cfg, X, n_sensors, device, seed), X, cfg, pool=pool)
