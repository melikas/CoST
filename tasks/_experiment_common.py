"""Shared context loader for experiment_q1/q2/q3.py.

Rebuilds the EXACT frozen encoder, dataset and participant split of a finished train_hrd.py
run from its variant directory. It never trains: the whole point of the three experiment
scripts is that RQ1-RQ3 reuse the single pretraining that already happened, so the only
`model.fit()` in the project stays the one inside train_hrd.py.

The split is READ from metrics.json ("test_pids"), not re-derived, so the held-out
participants are bit-identical to the depression run no matter what changes upstream.
"""
import hashlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cost import CoST
from data_processing.data_preprocessing import prepare_hrd_dataset
from models.positional_encoding import CALENDAR_PES
from train_hrd import paper_kernels, stratified_pid_holdout

# The plain-SSL control is the ONLY baseline that costs a real pretraining, so it is trained
# just for these reference configurations -- the same two the supervised baselines already use
# (train_hrd.py:1191), so the paper states one convention for both controls. Each is compared
# against its OWN disentangled twin, which is what leaves `disentangle` as the single
# difference; it is deliberately NOT a common yardstick under the other variants, where
# backbone and PE would differ too.
PLAIN_REF = {("tcn", "none"), ("tcn", "time2vec"), ("transformer", "sinusoidal")}

# Preprocessing-relevant config keys: two runs agreeing on these produce the same windows,
# so one cached dataset serves every variant of an array task (the CSV is 53.5M rows).
_DATA_KEYS = ("sensor_csv", "window_hours", "bin_minutes", "label_col", "max_missing",
              "max_window_missing", "no_zscore", "with_clock_features", "pe")
# Bumped whenever prepare_hrd_dataset gains a KEY, not just when a config value changes.
# None of _DATA_KEYS moves when a new field is added, so without this a pickle written by
# the previous schema is a cache HIT and the new field silently arrives as None.
#   2 -> added "ee_win" (window-matched emotional energy) for RQ2 layer 3.
_SCHEMA_VERSION = 2


def _dataset(cfg, cache_dir):
    key = hashlib.md5(json.dumps({**{k: cfg.get(k) for k in _DATA_KEYS},
                                  "_schema": _SCHEMA_VERSION},
                                 sort_keys=True, default=str).encode()).hexdigest()[:12]
    fp = Path(cache_dir) / f"hrd_windows_{key}.pkl" if cache_dir else None
    if fp is not None and fp.exists():
        print(f"[ctx] dataset cache hit -> {fp}")
        return pickle.loads(fp.read_bytes())
    data = prepare_hrd_dataset(
        cfg["sensor_csv"], window_hours=cfg["window_hours"], bin_minutes=cfg["bin_minutes"],
        label_col=cfg["label_col"], max_missing=cfg["max_missing"],
        max_window_missing=cfg["max_window_missing"], z_score=not cfg["no_zscore"],
        clock_features=cfg["with_clock_features"], calendar_index=cfg["pe"] in CALENDAR_PES)
    if fp is not None:
        fp.write_bytes(pickle.dumps(data, protocol=4))
    return data


def _build_model(cfg, X, n_sensors, device):
    """CoST constructed with the same arguments train_hrd.py used (train_hrd.py:782)."""
    seq_len = X.shape[1]
    mtl = cfg.get("max_train_length")
    return CoST(
        input_dims=X.shape[-1], n_time_features=int(X.shape[-1]) - int(n_sensors),
        kernels=cfg.get("kernels") or paper_kernels(seq_len), alpha=cfg["alpha"],
        max_train_length=seq_len if mtl is None else min(mtl, seq_len),
        output_dims=cfg["repr_dims"], hidden_dims=cfg["hidden_dims"], depth=cfg["depth"],
        backbone=cfg["backbone"], pe=cfg["pe"], time2vec_dim=cfg["time2vec_dim"],
        loss_balance=cfg["loss_balance"], bins_per_day=24 * 60 // cfg["bin_minutes"],
        disentangle=cfg["disentangle"], jitter_sigma=cfg["jitter_sigma"],
        mask_mode=cfg["mask_mode"], mask_prob=cfg["mask_keep_prob"],
        phase_mode=cfg["phase_encoding"], device=device, lr=cfg["lr"],
        batch_size=cfg["batch_size"])


def load_context(variant_dir, cache_dir=None, gpu=0):
    """Everything the experiment scripts need, with the trained encoder frozen in eval mode."""
    variant_dir = Path(variant_dir)
    meta = json.loads((variant_dir / "metrics.json").read_text(encoding="utf-8"))
    cfg = meta["config"]
    data = _dataset(cfg, cache_dir)
    X, y = data["X"], data["y"]
    pids = np.asarray(data["pids"]).astype(str)
    device = torch.device(f"cuda:{gpu}" if gpu >= 0 and torch.cuda.is_available() else "cpu")

    model = _build_model(cfg, X, data["n_sensors"], device)
    model.load(variant_dir / "encoder.pt")
    model.net.eval()

    # Split recovered from the run itself, never recomputed.
    test_pids = set(map(str, meta["test_pids"]))
    test_mask = np.isin(pids, list(test_pids))
    pid_label = {p: int(y[pids == p][0]) for p in np.unique(pids)}
    cohort = data["labeled_pids"] if cfg["cohort"] == "labeled" else data["consistent_pids"]
    pool = sorted({str(p) for p in cohort if str(p) in pid_label} - test_pids)
    rem, val = stratified_pid_holdout(pool, pid_label, cfg["val_frac"], cfg["split_seed"])
    train_mask, val_mask = np.isin(pids, list(rem)), np.isin(pids, list(val))
    if not val_mask.any():
        val_mask = train_mask

    # One row per participant (its most recent window) for participant-level labels; RQ1's
    # window-level targets deliberately ignore this (see docs/RQ_Minimal_Experiment_Design.md).
    last = np.zeros(len(pids), bool)
    for p in np.unique(pids):
        last[np.where(pids == p)[0][-1]] = True

    return SimpleNamespace(
        model=model, cfg=cfg, device=device, variant_dir=variant_dir,
        X=X, y=y, pids=pids, window_ids=data.get("window_ids"),
        ee_win=data.get("ee_win"),
        sensor_cols=list(data["sensor_cols"]), n_sensors=int(data["n_sensors"]),
        bin_minutes=cfg["bin_minutes"], seq_len=X.shape[1],
        bins_per_day=24 * 60 // cfg["bin_minutes"],
        trajectory_by_pid=data.get("trajectory_by_pid", {}),
        pid_label=pid_label, test_mask=test_mask, train_mask=train_mask, val_mask=val_mask,
        # every non-test window -- the exact set the disentangled encoder was pretrained on
        # (train_hrd.py:759), so the plain-SSL twin sees identical data.
        pretrain_mask=~test_mask,
        last_mask=last, seed=cfg["model_seed"], tag=f"{cfg['backbone']}/{cfg['pe']}",
        _n_sensors_data=data["n_sensors"])


def wants_plain_ssl(ctx):
    """True only for the reference configurations that carry the plain-SSL control."""
    return (ctx.cfg["backbone"], ctx.cfg["pe"]) in PLAIN_REF


def random_init_model(ctx):
    """The RQ1/RQ3 negative control: same architecture, weights never trained.

    Seeded explicitly -- an unseeded control redraws its weights on every invocation, so the
    same variant would report a different control AUC each run and the comparison it exists
    to support would not be reproducible.
    """
    torch.manual_seed(ctx.seed)
    m = _build_model(ctx.cfg, ctx.X, ctx.n_sensors, ctx.device)
    m.net.eval()
    return m


def encode(model, X, cfg, batch=256):
    """Window-pooled [V^(T); V^(S)], exactly the readout train_hrd.py probes."""
    sp = None if cfg["season_pool"] == "same" else cfg["season_pool"]
    return model.encode(X, mode="forecasting", pool=cfg["pool"], season_pool=sp,
                        batch_size=batch).squeeze(1)


def out_dir(ctx, name):
    d = ctx.variant_dir / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(d, stem, obj):
    (d / f"{stem}.json").write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
    print(f"[saved] {d / (stem + '.json')}")


def write_csv(d, stem, header, rows):
    (d / f"{stem}.csv").write_text(
        ",".join(header) + "\n" + "\n".join(",".join(map(str, r)) for r in rows) + "\n",
        encoding="utf-8")
