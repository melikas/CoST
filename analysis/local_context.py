"""Rebuild an experiment context from a dumped npz -- no CSV, no cluster, no trained weights.

`dump_context.py` writes the windows and every seed's masks once. This turns one seed of that
file back into the object the experiment scripts already take, so anything that does not train
a network or read trained weights runs on a laptop against exactly the participants the cluster
scored.

`ctx.model` is None and `ctx.trained` is False on purpose: a context with no encoder must not
be mistakable for one that has an untrained encoder sitting in it, which would be scored as if
it meant something.
"""
import sys
from pathlib import Path

# Run as `python analysis/<name>.py` from the repository root: the interpreter puts
# this file's own directory on sys.path, not the project root, so the shared modules
# would not import. scripts/ already does this; the pattern is the same.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from types import SimpleNamespace

import numpy as np

_CACHE = {}


def load_npz(path):
    """The dump, read once per process -- 24 seeds share one 40 MB window tensor."""
    if path not in _CACHE:
        _CACHE[path] = np.load(path, allow_pickle=False)
    return _CACHE[path]


def seeds(path):
    return [int(s) for s in load_npz(path)["seeds"]]


def local_context(path, seed, device="cpu"):
    """One seed's context: the shared windows plus that seed's own masks and config."""
    z = load_npz(path)
    seed = int(seed)
    if seed not in [int(s) for s in z["seeds"]]:
        raise KeyError(f"seed {seed} is not in {path}; it holds {list(z['seeds'])}")
    cfg = json.loads(str(z["configs_json"]))[str(seed)]
    pids = z["pids"].astype(str)
    y = z["y"]
    m = lambda k: z[f"{k}/{seed}"]

    pid_label = {p: int(y[pids == p][0]) for p in np.unique(pids)}
    return SimpleNamespace(
        model=None, trained=False,
        cfg=cfg, device=device, variant_dir=None,
        X=z["X"], y=y, pids=pids,
        window_ids=z["window_ids"].astype(str) if "window_ids" in z.files else None,
        sensor_cols=list(z["sensor_cols"].astype(str)), n_sensors=int(z["n_sensors"]),
        bin_minutes=int(z["bin_minutes"]), bins_per_day=int(z["bins_per_day"]),
        seq_len=int(z["X"].shape[1]),
        pid_label={p: v for p, v in pid_label.items() if v >= 0},
        train_mask=m("train_mask"), val_mask=m("val_mask"), test_mask=m("test_mask"),
        last_mask=m("last_mask"), pretrain_mask=m("pretrain_mask"),
        seed=seed, tag=f"{cfg['backbone']}/{cfg['pe']}",
    )
