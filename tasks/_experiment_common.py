"""Shared context loader for experiment_q1/q2/q3.py.

Rebuilds the EXACT frozen encoder, dataset and participant split of a finished train_hrd.py
run from its variant directory. It never trains: the whole point of the three experiment
scripts is that RQ1-RQ3 reuse the single pretraining that already happened, so the only
`model.fit()` in the project stays the one inside train_hrd.py.

The split is READ from metrics.json ("test_pids"), not re-derived, so the held-out
participants are bit-identical to the depression run no matter what changes upstream.
"""
import hashlib
from tasks.rq_paths import rq_path
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from data_processing.data_preprocessing import prepare_hrd_dataset
from models.positional_encoding import CALENDAR_PES
# `encode` is re-exported: experiment_q1/q2/q3 import it from here so every RQ script encodes
# through the one builder without each reaching into model_build separately.
from model_build import build_model, encode_repr as encode   # noqa: F401
from model_build import random_init_model as _random_init_model
from utils import stratified_pid_holdout

# The plain-SSL control is the ONLY baseline that costs a real pretraining, so it is trained
# just for these reference configurations -- the same two the supervised baselines already use
# (train_hrd.py:1191), so the paper states one convention for both controls. Each is compared
# against its OWN disentangled twin, which is what leaves `disentangle` as the single
# difference; it is deliberately NOT a common yardstick under the other variants, where
# backbone and PE would differ too.
PLAIN_REF = {("tcn", "none"), ("tcn", "circular"), ("tcn", "time2vec"),
             ("transformer", "sinusoidal")}
# ("tcn","circular") is here because the RQ3 ladder has a "DSSL plain (no disentangle)" RUNG
# (experiment_q3.py:181). A variant outside PLAIN_REF silently gets a ladder one rung SHORTER
# than the others, so its Delta AUC column is not comparable to theirs -- which defeats the
# point of running the variant at all when the question is "what does the time reference
# frame change". Membership costs one extra SSL pretraining per (seed, variant).

# Preprocessing-relevant config keys: two runs agreeing on these produce the same windows,
# so one cached dataset serves every variant of an array task (the CSV is 53.5M rows).
_DATA_KEYS = ("dataset", "sensor_csv", "window_hours", "bin_minutes", "label_col",
              "max_missing", "max_window_missing", "no_zscore", "with_clock_features", "pe",
              # GLOBEM-only, and they MUST be in the key: two GLOBEM runs differing only in
              # window_days or the labelling mode would otherwise hash to the same cache entry
              # and the second would silently be handed the first one's windows.
              "window_days", "stride_days", "globem_label", "globem_anchor_weekday")
# Bumped whenever prepare_hrd_dataset gains a KEY, not just when a config value changes.
# None of _DATA_KEYS moves when a new field is added, so without this a pickle written by
# the previous schema is a cache HIT and the new field silently arrives as None.
#   2 -> added "ee_win" (window-matched emotional energy) for RQ2 layer 3.
#   3 -> _dataset dispatches on cfg["dataset"], and the GLOBEM window arguments joined
#        _DATA_KEYS. Any cache written before this cannot know which dataset it holds.
_SCHEMA_VERSION = 3


def _dataset(cfg, cache_dir):
    key = hashlib.md5(json.dumps({**{k: cfg.get(k) for k in _DATA_KEYS},
                                  "_schema": _SCHEMA_VERSION},
                                 sort_keys=True, default=str).encode()).hexdigest()[:12]
    fp = Path(cache_dir) / f"hrd_windows_{key}.pkl" if cache_dir else None
    if fp is not None and fp.exists():
        print(f"[ctx] dataset cache hit -> {fp}")
        return pickle.loads(fp.read_bytes())
    # Mirrors train_hrd.py's own dispatch (train_hrd.py:693). Without it every RQ script
    # rebuilt the HRD windows whatever the run actually trained on, so a GLOBEM variant would
    # be analysed against a different cohort entirely -- silently, since the shapes still fit.
    if cfg.get("dataset") == "globem":
        from data_processing.globem_preprocessing import prepare_globem_dataset
        data = prepare_globem_dataset(
            cfg["sensor_csv"],
            window_days=cfg.get("window_days", 28),
            stride_days=cfg.get("stride_days", 7),
            # train_hrd now writes the rewritten name back onto args, so a current config
            # names a real GLOBEM column. Configs written before that fix still carry the HRD
            # default, and the same rewrite is applied here so they remain analysable.
            label_col=("LABEL_ENDPOINT" if cfg["label_col"] == "depression_status_endpoint"
                       else cfg["label_col"]),
            z_score=not cfg["no_zscore"],
            clock_features=cfg["with_clock_features"],
            weekly_labels=(cfg.get("globem_label") == "weekly"),
            anchor_weekday=cfg.get("globem_anchor_weekday", 0))
    else:
        data = prepare_hrd_dataset(
            cfg["sensor_csv"], window_hours=cfg["window_hours"],
            bin_minutes=cfg["bin_minutes"],
            label_col=cfg["label_col"], max_missing=cfg["max_missing"],
            max_window_missing=cfg["max_window_missing"], z_score=not cfg["no_zscore"],
            clock_features=cfg["with_clock_features"],
            calendar_index=cfg["pe"] in CALENDAR_PES)
    if fp is not None:
        fp.write_bytes(pickle.dumps(data, protocol=4))
    return data




def load_context(variant_dir, cache_dir=None, gpu=0, require_encoder=True):
    """Everything the experiment scripts need, with the trained encoder frozen in eval mode.

    `require_encoder=False` returns the same context with encoder.pt left unloaded, for the
    questions that need only the data, the split and the architecture -- a run.sh sweep
    deletes every encoder except one unless KEEP_ENC_ALL=1, and re-training 24 seeds just to
    ask what a RANDOM encoder does would be absurd. `ctx.trained` says which one you have;
    a caller that reads ctx.model without checking it would silently score noise.
    """
    variant_dir = Path(variant_dir)
    meta = json.loads((rq_path(variant_dir, "metrics.json")).read_text(encoding="utf-8"))
    cfg = meta["config"]
    data = _dataset(cfg, cache_dir)
    X, y = data["X"], data["y"]
    pids = np.asarray(data["pids"]).astype(str)
    device = torch.device(f"cuda:{gpu}" if gpu >= 0 and torch.cuda.is_available() else "cpu")

    model = build_model(cfg, X, data["n_sensors"], device)
    enc = rq_path(variant_dir, "encoder.pt")
    trained = enc.exists()
    if require_encoder or trained:
        model.load(enc)
    model.net.eval()

    # Split recovered from the run itself, never recomputed.
    test_pids = set(map(str, meta["test_pids"]))
    test_mask = np.isin(pids, list(test_pids)) & (np.asarray(y) >= 0)
    # One label per participant. Taking any window's y is right for HRD, where the endpoint
    # label is repeated on all of a person's windows -- but GLOBEM's weekly mode marks a window
    # with no survey in its span as y=-1, and the first window is often one of those, which
    # would enter the probe as a third class. `pid_summary_label` is the majority of that
    # person's LABELLED windows and is what the split itself is stratified on, so it is the
    # right source whenever the dataset provides it.
    summary = data.get("pid_summary_label") or {}
    pid_label = {p: int(summary.get(p, y[pids == p][0])) for p in np.unique(pids)}
    pid_label = {p: v for p, v in pid_label.items() if v >= 0}
    cohort = data["labeled_pids"] if cfg["cohort"] == "labeled" else data["consistent_pids"]
    pool = sorted({str(p) for p in cohort if str(p) in pid_label} - test_pids)
    rem, val = stratified_pid_holdout(pool, pid_label, cfg["val_frac"], cfg["split_seed"])
    # Every SCORED mask excludes windows with no label, exactly as train_hrd.py does. Without
    # it GLOBEM's weekly mode puts y=-1 windows into the probe and the supervised rung, whose
    # roc_auc_score then sees three classes -- which is how RQ3's supervised ceiling came back
    # as "SKIPPED: multi_class must be in ('ovo', 'ovr')" on the first GLOBEM run.
    labelled = np.asarray(y) >= 0
    train_mask = np.isin(pids, list(rem)) & labelled
    val_mask = np.isin(pids, list(val)) & labelled
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
        trained=trained,
        _n_sensors_data=data["n_sensors"])


def wants_plain_ssl(ctx):
    """True only for the reference configurations that carry the plain-SSL control."""
    return (ctx.cfg["backbone"], ctx.cfg["pe"]) in PLAIN_REF


def random_init_model(ctx):
    """ctx-shaped binding of model_build.random_init_model -- the ONE negative control.

    The seasonal layout is copied from the TRAINED encoder rather than from the config. Runs
    predating the `seasonal_bands` key cannot state their own layout and the answer differs
    between them -- pre-banding runs are single-band, post-banding ones are not -- so a config
    default would give some archived runs a control of a different architecture. "Same
    architecture, weights never trained" has to be true by construction, not by luck.
    """
    sfd = getattr(ctx.model.net, "sfd", None)
    cfg = dict(ctx.cfg)
    if sfd is not None:
        cfg["seasonal_bands"] = "single" if len(sfd) == 1 else "harmonics"
    return _random_init_model(cfg, ctx.X, ctx.n_sensors, ctx.device, ctx.seed)





def out_dir(ctx, name):
    """The folder rq_paths assigns to this experiment's artifacts. Resolved from a filename
    rather than hard-coded, so the RQ->folder table stays the single source of truth."""
    return rq_path(ctx.variant_dir, f"{name}.json").parent


def save(d, stem, obj):
    (d / f"{stem}.json").write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
    print(f"[saved] {d / (stem + '.json')}")


def write_csv(d, stem, header, rows):
    (d / f"{stem}.csv").write_text(
        ",".join(header) + "\n" + "\n".join(",".join(map(str, r)) for r in rows) + "\n",
        encoding="utf-8")
