"""Is the pretext task solvable before any training? The gate that decides whether a sweep runs.

CoST's trend branch is a MoCo InfoNCE: a query is matched to its own augmented view against a
queue of negatives. On this dataset, with the shipped global queue, top-1 retrieval on an
UNTRAINED encoder is 1.000 -- the task is already solved at initialisation, the loss starts at
zero, and pretraining therefore teaches nothing. That single number explains why the frozen
DSSL representation has never separated from its own random-init control.

`--negatives subject` draws the denominator from the SAME participant instead, so participant
identity is constant across it and cannot discriminate anything. This script measures whether
that actually makes the task hard, on the real pretraining windows, on CPU, before 144 GPU
hours are committed.

    top-1 ~ 1.000            the task is still solved at init -- DO NOT submit the sweep
    top-1 -> 1/(1+M)         the task is now learnable, and the sweep is worth running

Nothing here re-implements the objective: the model's own `forward` is called, and `last_top1`
is recorded inside `compute_loss` beside the loss it describes. The encoder is random-init and
stays that way -- no optimiser is constructed and no backward pass is taken, so the numbers
describe initialisation, which is the whole point.

Run (no GPU needed):
    python pretext_difficulty.py --variant-dir results_hrd/<run>/tcn_none_seed42
"""
import sys
from pathlib import Path

# Run as `python analysis/<name>.py` from the repository root: the interpreter puts
# this file's own directory on sys.path, not the project root, so the shared modules
# would not import. scripts/ already does this; the pattern is the same.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from cost import PretrainDataset
from model_build import random_init_model
from tasks._experiment_common import load_context, out_dir, save


def measure(model, X, pids, n_neg, mode, positive, batch_size, warm_batches,
            measure_batches, seed):
    """Top-1 over `measure_batches`, after `warm_batches` have filled the queue.

    The queue starts full of random vectors that belong to nobody, and `select_negatives`
    excludes those, so the first batches would contrast against an unrepresentatively small
    pool. Warming first is what makes the measurement describe the steady state the sweep
    would actually train in.
    """
    torch.manual_seed(seed)
    m = model.cost
    m.negatives, m.n_negatives = mode, n_neg
    m.queue_pid.fill_(-1)                       # forget any warm-up from a previous mode
    m.queue_ptr.zero_()

    # Mirrors CoST.fit's own construction (cost.py:755-757) field for field, so the views
    # the gate scores are the views the sweep would train on. multiplier=1 because one pass
    # per window is enough to fill and then measure the queue.
    ds = PretrainDataset(torch.from_numpy(X).to(torch.float),
                         jitter_sigma=model.jitter_sigma, shift_sigma=model.shift_sigma,
                         multiplier=1, n_exact_tail=model._n_exact_tail,
                         pids=pids, positive=positive,
                         decomp=None, n_sensors=model.n_sensors)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    top1, seen = [], 0
    with torch.no_grad():
        for _ in range(2):                      # a couple of passes if the set is small
            for batch in dl:
                x_q, x_k, x_q_s, x_k_s, pid = (t.to(model.device) for t in batch)
                # update=True enqueues, which is what warms the queue. No optimiser exists
                # and no backward is taken, so the WEIGHTS cannot move.
                model.cost(x_q, x_k, x_q_s, x_k_s, update=True, pid=pid)
                seen += 1
                if seen == warm_batches:
                    # Zero the shortfall counters only once the queue is warm. Counting the
                    # warm-up would report a shortfall rate that says more about the first
                    # few batches than about the regime the sweep would train in -- it read
                    # 68% that way on a cohort whose steady-state rate is near zero.
                    m.neg_short = m.neg_calls = 0
                if seen > warm_batches:
                    top1.append(m.last_top1)
                if len(top1) >= measure_batches:
                    break
            if len(top1) >= measure_batches:
                break
    return np.array(top1, float), (m.neg_short, m.neg_calls)


def _ceilings(path):
    """{positive pair: ceiling AUROC} from positive_pair_ceiling.py, or {} if not measured.

    Read rather than hard-coded so the two halves of the gate cannot drift: if the ceilings
    are re-measured, this picks up the new numbers without an edit here.
    """
    import json
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return {}
    rows = json.load(open(p))
    keys = [k for k in rows[0] if k != "seed"]
    return {k: float(np.nanmean([r[k] for r in rows])) for k in keys}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default=None,
                    help="Run from a dump_context npz instead of a variant directory, so the "
                         "gate can be answered on a laptop while the sweep it is meant to "
                         "gate is still queued -- which is the only time the answer is worth "
                         "anything.")
    ap.add_argument("--variant-dir",
                    help="any completed variant -- only its config and dataset are used; the "
                         "encoder is re-initialised at random")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--n-negatives", type=int, default=0,
                   help="0 (default) = the whole queue, i.e. the shipped denominator.")
    ap.add_argument("--negatives", default="global", choices=["global", "subject"],
                   help="Kept so the closed result stays reproducible: restricting the "
                        "denominator to the same participant was measured on this dataset to "
                        "leave difficulty untouched (top-1 0.8285 global vs 0.8324 subject). "
                        "The positive pair, not the negatives, sets the difficulty.")
    ap.add_argument("--moco-k", type=int, default=8192,
                    help="queue size the sweep will use. It matters here: the "
                         "same-participant pool is K/n_pretrain_pids, and if that is below "
                         "--n-negatives the sampler must draw with replacement.")
    ap.add_argument("--ceilings",
                    default=str(Path(__file__).resolve().parent.parent
                                / "results" / "positive_pair_ceiling.json"),
                    help="Ceilings measured by analysis/positive_pair_ceiling.py. Difficulty "
                         "alone cannot pass this gate: a pair that is hard can also be one "
                         "that discards the signal (default: %(default)s).")
    ap.add_argument("--baseline", type=float, default=0.7198,
                    help="What an UNTRAINED baseline already reaches, which the hard pair's "
                         "ceiling has to clear for pretraining on it to be worth anything. "
                         "Default is the raw random projection on HRD over 24 seeds.")
    ap.add_argument("--warm-batches", type=int, default=40)
    ap.add_argument("--measure-batches", type=int, default=40)
    a = ap.parse_args()

    if a.npz:
        from analysis.local_context import local_context
        import numpy as _np
        ctx = local_context(a.npz, [int(x) for x in _np.load(a.npz, allow_pickle=True)["seeds"]][0])
    elif a.variant_dir:
        ctx = load_context(a.variant_dir, a.cache_dir, gpu=-1)
    else:
        ap.error("one of --npz or --variant-dir is required")
    # The sweep pretrains on every non-test window; measuring on anything else would describe
    # a task the sweep never sees.
    Xp, pp = ctx.X[ctx.pretrain_mask], ctx.pids[ctx.pretrain_mask]
    n_pid = len(np.unique(pp))
    print(f"[gate] {ctx.tag} | {len(Xp):,} pretrain windows from {n_pid} participants")
    print(f"[gate] K={a.moco_k} -> ~{a.moco_k / n_pid:.0f} queue slots per participant, "
          f"against n_negatives={a.n_negatives}")

    cfg = dict(ctx.cfg)
    cfg["moco_k"] = a.moco_k
    # Same architecture as the sweep, weights never trained -- the negative control's own
    # constructor, so 'untrained' means exactly what it means everywhere else in the project.
    model = random_init_model(cfg, Xp, ctx.n_sensors, ctx.device, ctx.seed)

    chance = 1.0 / (1 + (a.n_negatives if a.n_negatives > 0 else a.moco_k))
    print("")
    print(f"  {'positive pair':14s} {'top-1':>8} {'SD':>8} {'chance':>8} {'x chance':>9}  shortfall")
    res = {}
    for mode in ("window", "participant"):
        t1, (short, calls) = measure(model, Xp, pp, a.n_negatives, a.negatives, mode,
                                     cfg["batch_size"], a.warm_batches, a.measure_batches,
                                     ctx.seed)
        res[mode] = dict(top1=float(t1.mean()), sd=float(t1.std(ddof=1)), n_batches=int(len(t1)),
                         shortfall=(short / calls) if calls else 0.0)
        print(f"  {mode:14s} {t1.mean():8.4f} {t1.std(ddof=1):8.4f} {chance:8.4f} "
              f"{t1.mean() / chance:9.1f}  {res[mode]['shortfall']:.1%}")

    g, s = res["window"]["top1"], res["participant"]["top1"]
    print("")
    print(f"  window {g:.4f} -> participant {s:.4f}   (gate: participant <= 0.30)")

    # DIFFICULTY IS NECESSARY, NOT SUFFICIENT. This gate used to pass on the left column
    # alone, and that verdict is wrong: a positive pair fixes two things at once, and the
    # one it makes hard it can also make useless. Measured on HRD, 24 seeds:
    #
    #     window pair        top-1 0.8223  (6737x chance)   ceiling 0.7151
    #     participant pair   top-1 0.0312  ( 256x chance)   ceiling 0.6658
    #
    # The only pair that is HARD caps the representation at 0.6658, below the 0.7198 a
    # random projection of the raw window already reaches with no training at all. Passing
    # that pair on difficulty alone would send a sweep whose best possible outcome is worse
    # than doing nothing -- which is exactly what this file exists to prevent.
    ceil = _ceilings(a.ceilings)
    hard = s <= 0.30
    c = ceil.get("participant pair")
    print("  ceiling of the hard pair: "
          + (f"{c:.4f}   (baseline to beat: {a.baseline:.4f})" if c is not None
             else "UNKNOWN -- run analysis/positive_pair_ceiling.py"))
    if not hard:
        verdict = "DO NOT SUBMIT -- still solved at init; the gradient would teach nothing"
    elif c is None:
        verdict = ("MEASURE THE CEILING FIRST -- the task is learnable, but a pair that is "
                   "hard can also be one that discards the signal")
    elif c <= a.baseline:
        verdict = (f"DO NOT SUBMIT -- the task is learnable but its ceiling {c:.4f} is at or "
                   f"below the {a.baseline:.4f} an untrained baseline already reaches, so the "
                   f"best possible outcome is worse than not pretraining")
    else:
        verdict = "SUBMIT      -- learnable at init AND its ceiling clears the baseline"
    print(f"  VERDICT: {verdict}")
    res["verdict"] = verdict
    if a.negatives == "subject" and res["participant"]["shortfall"] > 0.10:
        # Only meaningful when the SUBJECT-conditional denominator is in use; it is not the
        # default, because restricting the negatives was measured to leave difficulty
        # untouched (0.8285 -> 0.8324) while the positive pair moves it by 200x.
        print(f"  WARNING: {res['participant']['shortfall']:.1%} of queries had fewer than "
              f"{a.n_negatives} same-participant slots and drew with replacement. Raise "
              f"--moco-k (or lower --n-negatives).")

    res["chance"] = chance
    res["verdict"] = verdict
    res["n_pretrain_participants"] = int(n_pid)
    # --npz mode has no variant directory to write into, and the gate's whole value is the
    # verdict printed above -- crashing after producing it would throw the answer away.
    if getattr(ctx, "variant_dir", None):
        save(out_dir(ctx, "rq1"), "pretext_difficulty", res)
    else:
        print("[gate] no variant directory (--npz mode); result not saved")


if __name__ == "__main__":
    main()
