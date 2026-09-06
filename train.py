"""Runner for the pre-registered 4-arm experiment.

    phase_readout in {angle, circular}  x  objective weights in {paper, contracted}

evaluated on repeated grouped K-fold over every labelled participant (default 10 x 3).

WHAT EACH AXIS ISOLATES

  phase_readout  The confound that made run 2224103 (angle, C=0.8743, 21/24, p=3e-4) and
                 run 2412728 (circular, C=0.8289, 4/14, p=0.18) disagree. It was the only
                 live difference between them out of 73 config fields. Under `angle` the
                 seasonal readout is [amp | phase]; under `circular` it is [amp | cos | sin]
                 and probe.py scales the pair isotropically so the comparison is fair.

  weights        `paper` reproduces 2224103's objective bit-for-bit (trend 1.0, amp/phase
                 0.5/0.5). `contracted` re-allocates onto the phase term by measured RQ2
                 concordance. This is the experiment's one deliberate bet.

Everything else is held fixed across all four arms, including the two architectural
corrections (window-sized depth, repaired seasonal bands) and the trend-head kernel cap,
because those are defect fixes rather than hypotheses and applying them to only some arms
would confound the axes above.

    python train.py --dataset hrd    --npz hrd_2224103.npz --out runs/oneshot
    python train.py --dataset globem --sensor-csv datasets/GLOBEM_REDUCED.csv --out runs/g
    python train.py --arms angle:paper --folds 3 --repeats 1 --iters 50 --dry-run   # smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import objective as O
from cv import make_folds, nadeau_bengio, required_margin
from model import CoSTEncoder, depth_for_window, receptive_field, rhythm_bands
from probe import phase_block_layout

WEIGHTS = {"paper": O.PAPER, "contracted": O.CONTRACTED}
READOUTS = ("angle", "circular")


@dataclass(frozen=True)
class Arm:
    readout: str
    weights: str

    @property
    def tag(self) -> str:
        return f"{self.readout}_{self.weights}"

    def as_dict(self):
        w = WEIGHTS[self.weights]
        return {"phase_readout": self.readout, "weights": self.weights,
                "w_trend": w.trend, "w_amp": w.amp, "w_phase": w.phase}


def parse_arms(spec):
    """'all' -> the full 2x2; otherwise a comma-separated list of readout:weights."""
    if spec == "all":
        return [Arm(r, w) for r in READOUTS for w in WEIGHTS]
    arms = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise argparse.ArgumentTypeError(f"arm {item!r} must look like readout:weights")
        r, w = item.split(":", 1)
        if r not in READOUTS:
            raise argparse.ArgumentTypeError(f"readout {r!r} not in {READOUTS}")
        if w not in WEIGHTS:
            raise argparse.ArgumentTypeError(f"weights {w!r} not in {tuple(WEIGHTS)}")
        arms.append(Arm(r, w))
    if not arms:
        raise argparse.ArgumentTypeError("no arms given")
    return arms


# --------------------------------------------------------------------------------------
def load_windows(args):
    """Windows, participant ids and labels.

    An `--npz` cache is preferred: it is the exact window set a previous run used, so a
    result stays comparable to it. Without one the cohort is rebuilt from the raw CSV,
    which takes minutes on the 3.4 GB HRD file.
    """
    if args.npz:
        z = np.load(args.npz, allow_pickle=True)
        X, y, pids = z["X"], z["y"], z["pids"]
        n_sensors = int(z["n_sensors"])
        bins_per_day = int(z["bins_per_day"])
        return X, y, pids, n_sensors, bins_per_day, labelled_cohort(z, pids)
    raise SystemExit(
            "--npz is required in this build. Rebuilding the cohort from CSV goes through\n"
            "data_processing/, which is still the pre-cleanup loader and has not been wired\n"
            "to this runner yet. Point --npz at a window cache, or wire data_loader.py first.")


def labelled_cohort(z, pids):
    """The participants that actually carry a depression label.

    `y` in the cache is 0/1 for EVERY participant, including the 38 of 152 who are
    unlabelled and are present only to pretrain on. Folding over `y` directly would put
    those 38 into the negative class and corrupt every fold. The labelled cohort is the
    union of the train/val/test masks -- verified identical across all 24 cached seeds --
    and on HRD it is 114 participants at prevalence 0.456, matching
    `n_labeled_participants` in the run's metrics.json.
    """
    keys = [k for k in z.files if k.split("/")[0] in ("train_mask", "val_mask", "test_mask")]
    if not keys:
        raise SystemExit(f"{z} has no train/val/test masks; cannot identify the labelled "
                         "cohort, and folding over y would include unlabelled participants")
    cohorts = {}
    for k in keys:
        seed = k.split("/")[1]
        cohorts.setdefault(seed, set()).update(np.asarray(pids)[z[k]].tolist())
    uniq = {frozenset(v) for v in cohorts.values()}
    if len(uniq) != 1:
        raise SystemExit(f"the cache's labelled cohort differs across seeds ({len(uniq)} "
                         "variants); a fold set built from it would not be comparable")
    return next(iter(uniq))


def participant_table(y, pids, cohort=None):
    """One row per LABELLED participant, with that participant's (consistent) label."""
    pids = np.asarray(pids)
    y = np.asarray(y)
    keep = sorted(set(pids.tolist()) if cohort is None else cohort)
    uniq, lab = [], []
    for p in keep:
        v = np.unique(y[pids == p])
        if len(v) != 1:
            raise ValueError(f"participant {p} carries {len(v)} distinct labels")
        uniq.append(p)
        lab.append(int(v[0]))
    return np.array(uniq), np.array(lab)


# --------------------------------------------------------------------------------------
def build_encoder(args, X, n_sensors, bins_per_day, seed):
    import torch
    torch.manual_seed(seed)
    return CoSTEncoder(
        input_dims=X.shape[-1], output_dims=args.repr_dims, seq_len=X.shape[1],
        bins_per_day=bins_per_day, hidden_dims=args.hidden_dims, depth=args.depth,
        n_time_features=X.shape[-1] - n_sensors,
        seasonal_bands=args.seasonal_bands, disentangle=not args.plain,
        mask_mode=args.mask_mode, trend_kernel_cap=args.trend_kernel_cap,
        seasonal_frac=args.seasonal_frac)


def plan(args, arms, folds, X, n_sensors, bins_per_day):
    """The run matrix and its geometry, resolved before a single GPU-hour is spent."""
    T = X.shape[1]
    depth = args.depth if args.depth is not None else depth_for_window(T)
    enc = build_encoder(args, X, n_sensors, bins_per_day, seed=0)
    n_par = sum(p.numel() for p in enc.parameters())
    pair_start, pair_width = phase_block_layout("circular", enc.seasonal_dims)
    return {
        "dataset": args.dataset,
        "windows": list(X.shape),
        "n_sensors": n_sensors,
        "bins_per_day": bins_per_day,
        "seq_len": T,
        "depth": depth,
        "receptive_field": receptive_field(depth),
        "rf_over_window": round(receptive_field(depth) / T, 3),
        "bands": [list(b) for b in enc.bands],
        "trend_kernels": enc.kernels,
        "trend_dims": enc.trend_dims,
        "seasonal_dims": enc.seasonal_dims,
        "n_params": n_par,
        "circular_pair_block": [pair_start, pair_width],
        "arms": [a.as_dict() for a in arms],
        "protocol": {
            "n_folds": args.folds, "n_repeats": args.repeats,
            "n_models": len(arms) * len(folds),
            "nadeau_bengio_factor": round(nadeau_bengio(args.folds, args.repeats), 4),
            "required_margin_at_sd_0.042": round(
                required_margin(0.042, args.folds, args.repeats), 4),
        },
    }


# --------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Pre-registered 4-arm run: phase_readout x objective weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("data")
    g.add_argument("--dataset", choices=["hrd", "globem"], default="hrd")
    g.add_argument("--npz", help="window cache (X, y, pids, n_sensors, bins_per_day)")
    g.add_argument("--sensor-csv", help="raw CSV; requires the loader to be wired")

    g = p.add_argument_group("experiment")
    g.add_argument("--arms", type=parse_arms, default="all",
                   help="'all' for the 2x2, or e.g. 'angle:paper,circular:contracted'")
    g.add_argument("--folds", type=int, default=10)
    g.add_argument("--repeats", type=int, default=3)
    g.add_argument("--master-seed", type=int, default=20260906,
                   help="the ONE seed; split/model/probe seeds are spawned from it and are "
                        "guaranteed distinct")

    g = p.add_argument_group("architecture (held fixed across arms)")
    g.add_argument("--repr-dims", type=int, default=320)
    g.add_argument("--hidden-dims", type=int, default=64)
    g.add_argument("--depth", type=int, default=None,
                   help="default: smallest depth whose receptive field covers the window")
    g.add_argument("--trend-kernel-cap", type=int, default=None,
                   help="largest AR-expert kernel; default seq_len//8")
    g.add_argument("--seasonal-frac", type=float, default=0.5,
                   help="share of repr-dims given to V^S; 0.5 is upstream. Raising it is a "
                        "FIFTH arm, not a default -- see model.py")
    g.add_argument("--seasonal-bands", choices=["harmonics", "single"], default="harmonics")
    g.add_argument("--mask-mode", choices=["none", "binomial"], default="none")
    g.add_argument("--plain", action="store_true",
                   help="plain-SSL control: no trend/seasonal split")

    g = p.add_argument_group("optimisation")
    g.add_argument("--iters", type=int, default=6000)
    g.add_argument("--batch-size", type=int, default=64)
    g.add_argument("--lr", type=float, default=5e-4)
    g.add_argument("--alpha", type=float, default=0.005, help="seasonal-vs-trend scale")

    g = p.add_argument_group("output")
    g.add_argument("--out", default="runs/oneshot")
    g.add_argument("--dry-run", action="store_true",
                   help="resolve and write the plan, train nothing")

    a = p.parse_args(argv)
    if isinstance(a.arms, str):
        a.arms = parse_arms(a.arms)
    if not 0.1 <= a.seasonal_frac <= 0.9:
        p.error("--seasonal-frac must be in [0.1, 0.9]")
    return a


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    X, y, pids, n_sensors, bins_per_day, cohort = load_windows(args)
    upids, ulab = participant_table(y, pids, cohort)
    folds = make_folds(upids, ulab, args.folds, args.repeats, args.master_seed)
    spec = plan(args, args.arms, folds, X, n_sensors, bins_per_day)
    spec["n_participants_total"] = len(set(pids.tolist()))
    spec["n_labelled_participants"] = len(upids)
    spec["prevalence"] = round(float(ulab.mean()), 4)
    spec["folds"] = [f.as_dict() for f in folds]

    (out / "plan.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

    pr = spec["protocol"]
    print(f"[plan] {args.dataset}: {spec['windows']} windows, "
          f"{spec['n_labelled_participants']} labelled of "
          f"{spec['n_participants_total']} participants, "
          f"prevalence {spec['prevalence']}")
    print(f"[arch] depth {spec['depth']}  RF {spec['receptive_field']} "
          f"({spec['rf_over_window']}x window)  bands {spec['bands']}")
    print(f"[arch] trend kernels {spec['trend_kernels']} -> "
          f"{spec['n_params']:,} params  (V^T {spec['trend_dims']} / V^S {spec['seasonal_dims']})")
    print(f"[prot] {pr['n_folds']}-fold x {pr['n_repeats']}, {len(args.arms)} arms "
          f"-> {pr['n_models']} models; NB factor {pr['nadeau_bengio_factor']}, "
          f"margin ~{pr['required_margin_at_sd_0.042']}")
    for a in args.arms:
        d = a.as_dict()
        print(f"[arm ] {a.tag:22s} w_trend={d['w_trend']:.3f} "
              f"w_amp={d['w_amp']:.3f} w_phase={d['w_phase']:.3f}")
    print(f"[out ] {out / 'plan.json'}")

    if args.dry_run:
        return 0

    raise SystemExit(
        "\nTraining is not wired in this build, and the plan above is the deliverable.\n"
        "What remains, in order:\n"
        "  1. data_loader.py  -- port cohort building from data_processing/ so --sensor-csv works\n"
        "  2. cost.py         -- point PretrainDataset/CoSTModel at model.CoSTEncoder and\n"
        "                        objective.total_loss; delete the V^N branch and the PE imports\n"
        "  3. eval.py         -- RQ1/RQ2/RQ3 over the folds in plan.json, probes from probe.py\n"
        "Run with --dry-run to produce the plan without this message.")


if __name__ == "__main__":
    sys.exit(main())
