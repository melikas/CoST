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
from cost import CoST
from cv import make_folds, make_lodo_folds, nadeau_bengio, required_margin
from data_loader import load_npz
from model import CoSTEncoder, depth_for_window, receptive_field
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


def plan(args, arms, folds, X, n_sensors, bins_per_day, n_folds_eff=None):
    """The run matrix and its geometry, resolved before a single GPU-hour is spent."""
    T = X.shape[1]
    depth = args.depth if args.depth is not None else depth_for_window(T)
    enc = build_encoder(args, X, n_sensors, bins_per_day, seed=0)
    n_par = sum(p.numel() for p in enc.parameters())
    pair_start, pair_width = phase_block_layout(
        "circular", enc.seasonal_dims, T, bins_per_day)
    nf = n_folds_eff if n_folds_eff is not None else args.folds
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
            "name": args.protocol,
            "n_folds": nf, "n_repeats": args.repeats,
            "n_arm_fold_results": len(arms) * len(folds),
            "n_encoder_fits": len({a.weights for a in arms}) * len(folds),
            "nadeau_bengio_factor": round(nadeau_bengio(nf, args.repeats), 4),
            "required_margin_at_sd_0.042": round(
                required_margin(0.042, nf, args.repeats), 4),
        },
    }



# --------------------------------------------------------------------------------------
def run_fold(args, weights_name, readouts, coh, fold, out_dir):
    """Pretrain ONE encoder for this (weights, fold), then read it out under each readout.

    The test participants are excluded from PRETRAINING as well as from the probe, so the
    representation a held-out participant is scored under has never seen that participant.
    Nothing here reads a label: the split is by participant, and `y` is only carried through
    so eval.py can read it off the same arrays.

    WHY ONE ENCODER SERVES BOTH READOUTS. `phase_readout` never enters the training path --
    the objective compares phases through `phase_mode` inside objective.seasonal_loss, while
    `phase_readout` only chooses what CoST._spectral emits at encode time. Verified: with a
    fixed seed, angle and circular give bit-identical training losses. So the 2x2 needs 2
    encoder fits per fold, not 4.

    That is not merely cheaper. It makes the readout contrast EXACTLY PAIRED -- the same
    weights read two ways -- so encoder-training variance cancels out of it completely
    instead of being one more thing the 30 folds have to average over.
    """
    fd = coh.fold_data(fold)
    Xp = coh.X[fd.pretrain]
    rng = np.random.default_rng(fold.model_seed)
    perm = rng.permutation(len(Xp))
    n_val = int(len(Xp) * args.val_frac)
    val, tr = Xp[perm[:n_val]], Xp[perm[n_val:]]

    model = CoST(
        input_dims=coh.n_features, seq_len=coh.seq_len, bins_per_day=coh.bins_per_day,
        output_dims=args.repr_dims, hidden_dims=args.hidden_dims, depth=args.depth,
        n_time_features=coh.n_features - coh.n_sensors, seasonal_bands=args.seasonal_bands,
        disentangle=not args.plain, mask_mode=args.mask_mode,
        trend_kernel_cap=args.trend_kernel_cap, seasonal_frac=args.seasonal_frac,
        phase_readout=readouts[0], weights=WEIGHTS[weights_name], alpha=args.alpha,
        moco_k=args.moco_k, jitter_sigma=args.jitter_sigma, shift_sigma=args.shift_sigma,
        smooth_bins=args.smooth_bins, lr=args.lr, batch_size=args.batch_size,
        device=args.device, model_seed=fold.model_seed)

    hist = model.fit(tr, n_iters=args.iters, val_data=val if len(val) else None,
                     log_every=args.log_every, verbose=args.verbose)

    if args.save_encoder:
        # Saved BEFORE the readout loop, and under the weighting rather than the arm: these
        # weights are readout-agnostic, and writing the checkpoint after the loop would stamp
        # it with whichever readout happened to run last.
        enc_dir = out_dir / f"encoders_{weights_name}" / fold.tag
        enc_dir.mkdir(parents=True, exist_ok=True)
        model.save(enc_dir / "encoder.pt")

    recs = []
    for readout in readouts:
        model.phase_readout = readout            # inference-time only; see the docstring
        reps = model.encode(coh.X, batch_size=args.encode_batch, pool=args.pool, parts=True)
        arm = Arm(readout, weights_name)
        d = out_dir / arm.tag / fold.tag
        d.mkdir(parents=True, exist_ok=True)
        if args.save_repr:
            np.savez_compressed(d / "repr.npz", **reps, y=coh.y, pids=coh.pids,
                                pretrain=fd.pretrain, probe_train=fd.probe_train,
                                probe_test=fd.probe_test)
        trend_w = reps["trend"].shape[1] if "trend" in reps else 0
        rec = {"arm": arm.as_dict(), "fold": fold.as_dict(), "split": fd.summary(),
               "n_pretrain_train": int(len(tr)), "n_pretrain_val": int(len(val)),
               "iters": model.n_iters,
               "final_top1": hist["top1"][-1] if hist["top1"] else None,
               "loss": hist,
               "repr_dims": {k: list(v.shape) for k, v in reps.items()},
               # Where eval.py must put IsotropicPairScaler when probing `full`. Recorded
               # here rather than recomputed downstream, so the scaler can never be pointed
               # at the wrong columns.
               "pair_block_full": list(phase_block_layout(
                   readout, model.component_dims, coh.seq_len, coh.bins_per_day, trend_w)),
               "pair_block_seasonal": list(phase_block_layout(
                   readout, model.component_dims, coh.seq_len, coh.bins_per_day, 0))}
        (d / "fold.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        recs.append(rec)
    return recs


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
    g.add_argument("--protocol", choices=["kfold", "lodo"], default="kfold",
                   help="kfold = repeated grouped K-fold over all labelled participants. "
                        "lodo = leave-one-study-year-out, the GLOBEM benchmark's own split, "
                        "which is what makes our numbers comparable to their 0.547")
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
    g.add_argument("--moco-k", type=int, default=4096, help="MoCo queue size")
    g.add_argument("--jitter-sigma", type=float, default=0.1)
    g.add_argument("--shift-sigma", type=float, default=0.5)
    g.add_argument("--smooth-bins", type=int, default=5,
                   help="widest box filter for the smoothing augmentation; 0 disables it")
    g.add_argument("--val-frac", type=float, default=0.10,
                   help="share of pretrain windows held out to monitor the pretext loss")
    g.add_argument("--pool", choices=["mean", "last", "max"], default="mean")
    g.add_argument("--device", default="cuda")
    g.add_argument("--encode-batch", type=int, default=256)
    g.add_argument("--log-every", type=int, default=200)

    g = p.add_argument_group("output")
    g.add_argument("--out", default="runs/oneshot")
    g.add_argument("--dry-run", action="store_true",
                   help="resolve and write the plan, train nothing")
    g.add_argument("--save-repr", action="store_true", default=True,
                   help="write each fold's frozen representations for eval.py")
    g.add_argument("--no-save-repr", dest="save_repr", action="store_false")
    g.add_argument("--save-encoder", action="store_true",
                   help="also keep each fold's weights")
    g.add_argument("--verbose", action="store_true", default=True)
    g.add_argument("--quiet", dest="verbose", action="store_false")
    g.add_argument("--only-fold", default=None,
                   help="run a single fold, e.g. r0f3 -- for SLURM array sharding")

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

    if not args.npz:
        raise SystemExit("--npz is required: the raw-CSV path still goes through the "
                         "pre-cleanup data_processing/ loader and is not wired here.")
    coh = load_npz(args.npz)
    X, n_sensors, bins_per_day = coh.X, coh.n_sensors, coh.bins_per_day
    upids, ulab = coh.participants()
    if args.protocol == "lodo":
        yr = coh.participant_years()
        folds = make_lodo_folds(upids, ulab, [yr[p] for p in upids],
                                args.repeats, args.master_seed)
        n_folds_eff = len({f.fold for f in folds})
    else:
        folds = make_folds(upids, ulab, args.folds, args.repeats, args.master_seed)
        n_folds_eff = args.folds

    # plan.json's "arms" must be the UNION of every arm ever requested against this --out
    # directory, not just this invocation's args.arms. oneshot.sh shards by weighting, so
    # each of the 60 array tasks calls main() with only 2 of the 4 arms; writing args.arms
    # straight to plan.json meant every task OVERWROTE the file with its own slice, and
    # whichever task finished last left plan.json holding only that slice's 2 arms
    # permanently -- eval.py trusts plan.json to know which arm directories exist, so it
    # silently stopped evaluating the other 2 even though their fold.json/repr.npz were
    # sitting right there on disk. Merging with whatever is already recorded means the
    # field can only grow across concurrent invocations, never shrink.
    prior = out / "plan.json"
    seen = {(a.readout, a.weights) for a in args.arms}
    if prior.exists():
        try:
            for a in json.loads(prior.read_text(encoding="utf-8"))["arms"]:
                seen.add((a["phase_readout"], a["weights"]))
        except (json.JSONDecodeError, KeyError, OSError):
            pass          # a torn concurrent write; this invocation's own arms still count
    plan_arms = [Arm(r, w) for r, w in sorted(seen)]

    spec = plan(args, plan_arms, folds, X, n_sensors, bins_per_day, n_folds_eff)
    spec["n_participants_total"] = len(set(coh.pids.tolist()))
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
    print(f"[prot] {args.protocol}: {pr['n_folds']}-fold x {pr['n_repeats']}, "
          f"{len(args.arms)} arms "
          f"-> {pr['n_encoder_fits']} encoder fits / "
          f"{pr['n_arm_fold_results']} results; NB factor {pr['nadeau_bengio_factor']}, "
          f"margin ~{pr['required_margin_at_sd_0.042']}")
    for a in args.arms:
        d = a.as_dict()
        print(f"[arm ] {a.tag:22s} w_trend={d['w_trend']:.3f} "
              f"w_amp={d['w_amp']:.3f} w_phase={d['w_phase']:.3f}")
    print(f"[out ] {out / 'plan.json'}")

    if args.dry_run:
        return 0

    # The training axis is `weights`; `phase_readout` is applied at encode time, so one
    # encoder per (weights, fold) covers every readout requested for that weighting.
    by_weights = {}
    for a in args.arms:
        by_weights.setdefault(a.weights, []).append(a.readout)
    todo = [(w, ros, f) for w, ros in by_weights.items() for f in folds
            if args.only_fold is None or f.tag == args.only_fold]
    if not todo:
        raise SystemExit(f"--only-fold {args.only_fold!r} matched no fold")
    print(f"[run ] {len(todo)} encoder fits on {args.device} "
          f"-> {sum(len(r) for _, r, _ in todo)} (arm, fold) results")

    done = []
    for i, (wname, ros, fold) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] weights={wname} readouts={','.join(ros)} {fold.tag}", flush=True)
        done.extend(run_fold(args, wname, ros, coh, fold, out))
    name = "folds.json" if args.only_fold is None else f"folds_{args.only_fold}.json"
    (out / name).write_text(json.dumps(done, indent=2), encoding="utf-8")
    print(f"[done] {len(done)} models -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
