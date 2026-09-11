"""Per-participant sensor scale of an HRD window cache, so eval.py's handcrafted blocks can
read physical units (bpm, fraction asleep, screen use) instead of within-person z-scores.

The cache is z-scored WITHIN participant, per channel, on the raw samples and before binning
(data_preprocessing.zscore_within_participant). Bin means and linear interpolation commute
with that affine map, so a window in physical units is exactly X * sd_p + mu_p. The z-score
erases every between-person level difference -- how high someone's heart rate runs, how much
they sleep, how much they use their phone -- and no rung of the ladder has ever seen one.

This script does not trust the argument above. It rebuilds the windows UNstandardised from
the raw CSV, requires the rebuild to reproduce the cache's window ids exactly, matches the
channels BY NAME, recovers (mu_p, sd_p) per participant and channel by least squares, and
refuses to write if any window disagrees with the cache beyond float32 tolerance.

    python scripts/build_raw_scale.py --npz hrd_2224103.npz \\
        --csv datasets/HRD_RAW_MinuteLevel.csv --out hrd_2224103_scale.npz

No label is read. The sidecar holds two numbers per participant and channel.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.data_preprocessing import prepare_hrd_dataset  # noqa: E402

TOL = 1e-3          # max |X*sd+mu - X_raw| relative to the channel's range in that person


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--npz", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    z = np.load(a.npz, allow_pickle=True)
    X, pids, wid = z["X"], z["pids"].astype(str), z["window_ids"].astype(str)
    ns = int(z["n_sensors"])
    names = list(np.asarray(z["sensor_cols"]).astype(str)[:ns])
    raw = prepare_hrd_dataset(a.csv, z_score=False)
    if not np.array_equal(np.asarray(raw["window_ids"]).astype(str), wid):
        raise SystemExit("the unstandardised rebuild does not reproduce the cache's window ids; "
                         "the cache was built with other settings -- refusing to write")
    # By NAME, never by position: the cache's channels need not be today's default list in
    # today's order, and a positional slice would silently pair one channel with another.
    have = list(np.asarray(raw["sensor_cols"]).astype(str))
    if not set(names) <= set(have):
        raise SystemExit(f"the rebuild has channels {have}; the cache needs {names}")
    print(f"[scale] rebuild channels {have} -> cache channels {names}")
    Xr = np.asarray(raw["X"])[:, :, [have.index(c) for c in names]].astype(np.float64)
    Xz = X[:, :, :ns].astype(np.float64)

    pu = np.unique(pids)
    mu, sd = np.zeros((len(pu), ns)), np.ones((len(pu), ns))
    worst, at = 0.0, ""
    for i, q in enumerate(pu):
        m = pids == q
        for c in range(ns):
            xz, xr = Xz[m, :, c].ravel(), Xr[m, :, c].ravel()
            if np.ptp(xz) == 0:
                # Constant over every window of this person (in hrd_2224103: i48's screen at
                # z = -0.1436, x50's at 0) -- constant in the windows, not necessarily in the
                # whole record the z-score was computed on. The scale is not identifiable
                # from these windows and not needed to reconstruct them: sd = 1 and the
                # offset alone is exact. mu = mean(raw) reconstructed z + raw instead.
                mu[i, c] = (xr - xz).mean()
            else:
                sd[i, c], mu[i, c] = np.polyfit(xz, xr, 1)
            # Relative to the channel's range -- or to its magnitude when it has no range.
            err = float(np.abs(xz * sd[i, c] + mu[i, c] - xr).max()
                        / max(np.ptp(xr), np.abs(xr).max(), 1e-12))
            if err > worst:
                worst, at = err, f"{q}/{names[c]}"
    print(f"[scale] {len(pu)} participants x {ns} channels; worst relative residual "
          f"{worst:.2e} ({at})")
    if worst > TOL:
        raise SystemExit(f"residual {worst:.2e} exceeds {TOL:g}: the cache is not an affine "
                         "per-participant map of the raw windows -- refusing to write")
    np.savez(a.out, pids=pu, mu=mu, sd=sd, window_ids=wid, sensor_cols=np.asarray(names),
             source=str(a.npz))
    for c, n in enumerate(names):
        print(f"[scale] {n:9s} participant mean: median {np.median(mu[:, c]):.3f}, "
              f"IQR [{np.percentile(mu[:, c], 25):.3f}, {np.percentile(mu[:, c], 75):.3f}]")
    print(f"[out] {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
