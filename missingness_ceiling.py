"""Does the observation mask carry signal the encoder is never shown?

The quality gate admits up to 30%% of a channel's bins missing, fills them by linear
interpolation, and then discards the record of which bins those were (the mask is built in
_fill_bin_grid, read by the gate, and dropped). So the encoder sees a window in which as much
as a third of a channel may be manufactured, with no way to tell a measured value from a
manufactured one -- and linear interpolation manufactures exactly the smooth low-frequency
shape the trend branch and the Fourier layer are built to fit.

That is a mechanism for the one thing DSSL most needs explained: training moves the score by
0.0011 while DEGRADING the rhythm structure it is supposed to learn (trend stability
0.6830 -> 0.5294 over 24 seeds, phase concentration 0.5636 -> 0.4766, p=1.2e-5). If part of
the signal is synthetic smoothness shared across windows, the contrastive objective has a
shortcut: fit the interpolator instead of the participant.

Two fixes follow from it, and this measurement chooses between them:

  give the encoder the mask as input channels   -- only worth it if the mask carries signal
  mask the fabricated bins out of the views     -- worth it either way

So: does the mask predict the label, and does it add anything to the raw window? Same probe,
same splits and the same held-out participants as every other number in the project -- the
windows are aligned to the run's own npz by window_id, which is asserted rather than assumed.

Needs the raw CSV and ~16 GB (the file is 3.3 GB and pandas needs several times that), so
this runs on a CPU node rather than a laptop:

    python missingness_ceiling.py --csv datasets/HRD_RAW_MinuteLevel.csv --npz hrd_2224103.npz
"""
import argparse
import json

import numpy as np


def mask_features(M, bins_per_day):
    """Everything the observation mask can say about one window, per channel.

    M is (N, T, C) boolean, True where a raw sample landed in that bin. The blocks answer
    different questions and are kept separate so a result can name which one carried it:
    how much was worn, when in the day it was worn, whether that changed across days, and
    how the absences were shaped -- one long removal or many short ones.
    """
    N, T, C = M.shape
    F = np.asarray(M, dtype=np.float32)
    d, w = T // bins_per_day, bins_per_day
    day = F[:, :d * w].reshape(N, d, w, C)

    q = w // 4                                             # six-hour quarters of the day
    tod = np.stack([day[:, :, i * q:(i + 1) * q].mean(axis=(1, 2)) for i in range(4)], axis=1)

    # A gap STARTS where an observed bin is followed by a missing one; the window is padded
    # with observed on both sides so a gap at the very start counts once and never twice.
    pad = np.pad(F, ((0, 0), (1, 1), (0, 0)), constant_values=1.0)
    gaps_n = ((pad[:, 1:-1] == 0) & (pad[:, :-2] == 1)).sum(axis=1).astype(np.float32)
    # Longest run of missing bins, as a running count that resets on every observed bin.
    # The loop is over T (672), not over N x C x T (10M), which is the difference between
    # seconds and minutes here.
    run = np.zeros((N, C), np.float32)
    gaps_max = np.zeros((N, C), np.float32)
    miss = (F == 0)
    for t in range(T):
        run = (run + 1.0) * miss[:, t]
        gaps_max = np.maximum(gaps_max, run)
    gaps_max = gaps_max / T

    return np.concatenate([
        F.mean(axis=1),                                    # C   coverage
        tod.reshape(N, -1),                                # 4C  coverage by time of day
        day.mean(axis=2).mean(axis=1),                     # C   mean daily coverage
        day.mean(axis=2).std(axis=1),                      # C   day-to-day variability of it
        gaps_n / T, gaps_max,                              # 2C  shape of the absences
    ], axis=1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None,
                    help="raw CSV; defaults to the path the run itself used")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--width", type=int, default=512,
                    help="raw projection width; swept over 16..1760 on HRD and flat, with"
                         " 512 the best measured value (default: %(default)s)")
    ap.add_argument("--out", default="missingness_ceiling.json")
    a = ap.parse_args()

    from data_processing.data_preprocessing import prepare_hrd_dataset
    from local_context import local_context
    from random_init_audit import _probe_auc, raw_projection
    from train_hrd import CALENDAR_PES

    z = np.load(a.npz, allow_pickle=True)
    seeds = [int(s) for s in z["seeds"]]
    # The RUN's preprocessing arguments, not this script's defaults. Every one of these
    # changes which windows survive the gate, so guessing them would silently compare a mask
    # from one windowing against splits from another. Mirrors train_hrd.py's own call.
    cfg = json.loads(str(z["configs_json"]))[str(seeds[0])]
    data = prepare_hrd_dataset(
        a.csv or cfg["sensor_csv"],
        window_hours=cfg["window_hours"], bin_minutes=cfg["bin_minutes"],
        label_col=cfg["label_col"], max_missing=cfg["max_missing"],
        max_window_missing=cfg["max_window_missing"],
        z_score=not cfg["no_zscore"], clock_features=cfg["with_clock_features"],
        calendar_index=cfg["pe"] in CALENDAR_PES,
        keep_observed=True)
    by_id = {w: i for i, w in enumerate(np.asarray(data["window_ids"]).astype(str))}
    want = np.asarray(z["window_ids"]).astype(str)
    # The run's own windows, in the run's own order. Steps is dropped AFTER the quality gate,
    # so a default preprocessing pass produces the same window set; if it does not, the two
    # are not the same experiment and averaging across them would be meaningless.
    absent = [w for w in want if w not in by_id]
    if absent:
        raise SystemExit(f"{len(absent)} of {len(want)} windows in {a.npz} are not in a fresh"
                         f" preprocessing pass (first: {absent[0]}) -- the mask cannot be"
                         f" aligned to this run")
    idx = np.array([by_id[w] for w in want])
    M = np.asarray(data["observed"])[idx]
    print(f"[missingness] aligned {len(idx)} windows, mask {M.shape}, "
          f"{data['n_sensors']} sensor channels")

    miss = 1.0 - M.mean(axis=1)
    print(f"[missingness] windows with any interpolated bin: {(miss.sum(1) > 0).mean():.1%}"
          f" | mean missing fraction per channel {miss.mean():.4f}")

    feats_shared = {"missingness (all)": mask_features(M, int(z["bins_per_day"])),
                    "coverage only": M.mean(axis=1).astype(np.float32)}
    rows = []
    for i, sd in enumerate(seeds):
        ctx = local_context(a.npz, sd)
        P = raw_projection(ctx.X, ctx.n_sensors, a.width, sd)
        feats = dict(feats_shared)
        feats["raw projection"] = P
        feats["raw + missingness"] = np.concatenate([P, feats_shared["missingness (all)"]], 1)
        r = {"seed": sd}
        for name, F in feats.items():
            r[name] = _probe_auc(F, ctx)
        rows.append(r)
        json.dump(rows, open(a.out, "w"), indent=1)
        print(f"[{i + 1:2d}/{len(seeds)}] seed {sd:3d}  "
              + "  ".join(f"{k}={v:.3f}" for k, v in r.items() if k != "seed"), flush=True)

    names = [k for k in rows[0] if k != "seed"]
    base = np.array([r["raw projection"] for r in rows], float)
    print()
    print(f"  {len(rows)} seeds, HRD, the run's own probe, splits and participants")
    print()
    print(f"  {'arm':22s} {'dim':>5s} {'AUC':>8s} {'vs raw':>9s} {'wins':>8s}")
    for n in names:
        v = np.array([r[n] for r in rows], float)
        d = v - base
        dim = feats[n].shape[1] if n in feats else 0
        print(f"  {n:22s} {dim:5d} {np.nanmean(v):8.4f} {np.nanmean(d):+9.4f}"
              f" {int(np.nansum(d > 0)):4d}/{len(rows)}")


if __name__ == "__main__":
    main()
