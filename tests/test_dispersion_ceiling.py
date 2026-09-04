"""Does the instrument measure what it claims? Plant one thing at a time; only it may move."""
import numpy as np
from tasks.signal_decomposition import decompose
from dispersion_ceiling import day_features

W, D, C = 96, 7, 3
T = W * D
t = np.arange(T)
base = 2.0 * np.sin(2 * np.pi * t / W) + 0.5 * np.cos(4 * np.pi * t / W)

# 1. x = trend + seasonal + residual must be exact, and a pure rhythm on a drift must
#    leave NO residual at all -- including at the ends of the window.
X = np.tile((base + np.linspace(0, 3, T))[None, :, None], (4, 1, C)).astype(np.float32)
tr, se, re = decompose(X, W, C)
print(f"  rhythm + drift -> residual rms {np.sqrt((re ** 2).mean()):.5f}"
      f"   reconstruction err {np.abs(tr + se + re - X).max():.2e}")
assert np.sqrt((re ** 2).mean()) < 0.01, "the rhythm leaked into the residual"
assert np.abs(tr + se + re - X).max() < 1e-4, "x != trend + seasonal + residual"

rng = np.random.default_rng(0)
n = 400
lab = np.repeat([0, 1], n // 2)
def sep(F):
    """Cohen's d on the most separated column. Standardising by the TOTAL std instead
    caps at 2.0 for a 50/50 split with perfect separation, so a threshold above 2 could
    never be met however good the feature was."""
    a, b = F[lab == 0], F[lab == 1]
    pooled = np.sqrt((a.var(0) + b.var(0)) / 2) + 1e-9
    return float(np.abs((b.mean(0) - a.mean(0)) / pooled).max())

def report(tag, Xg, expect_fire, expect_quiet):
    _, _, r = decompose(Xg, W, C)
    d = {k: sep(day_features(r if "resid" in k else Xg, W, kinds=(k.split()[0],)))
         for k in ("level", "level_var", "within")}
    print(f"  {tag:26s} " + "  ".join(f"{k}={v:5.2f}" for k, v in d.items()))
    for k in expect_fire:
        assert d[k] > 3.0, f"{tag}: {k} missed a signal planted in it (d={d[k]:.2f})"
    for k in expect_quiet:
        assert d[k] < 0.5, f"{tag}: {k} fired on a signal not in it (d={d[k]:.2f})"

# 2. signal ONLY in within-day dispersion. `level` must stay silent: the extra noise is
#    zero-mean, so it moves the spread of a day and not its total.
Xa = np.tile(base[None, :, None], (n, 1, C)) + rng.normal(0, 1.0, (n, 1, C))
Xa = (Xa + rng.normal(0, 0.3 + 0.9 * lab[:, None, None], (n, T, C))).astype(np.float32)
report("within-day dispersion", Xa, ["within"], ["level"])

# 3. signal ONLY in the level. Neither dispersion arm may see it.
Xb = np.tile(base[None, :, None], (n, 1, C)) + rng.normal(0, 0.3, (n, T, C))
Xb = (Xb + 1.5 * lab[:, None, None]).astype(np.float32)
report("level shift", Xb, ["level"], ["level_var", "within"])

# 4. signal ONLY in day-to-day irregularity of the level: same mean, same within-day
#    shape, but the daily totals bounce around more in one group.
Xc = np.tile(base[None, :, None], (n, 1, C)) + rng.normal(0, 0.3, (n, T, C))
bump = rng.normal(0, 0.15 + 1.2 * lab[:, None, None], (n, D, C))
Xc = (Xc + np.repeat(bump, W, axis=1)).astype(np.float32)
report("day-to-day level spread", Xc, ["level_var"], ["level"])
print("PASS")
