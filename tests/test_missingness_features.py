"""Each mask-feature block must respond to what it names, and to nothing else."""
import numpy as np

from missingness_ceiling import mask_features

W, D, C = 96, 7, 3
T = W * D


def feat(M):
    return mask_features(M, W)


def blocks(F):
    """(coverage, time-of-day, daily mean, daily sd, n gaps, longest gap)."""
    return (F[:, :C], F[:, C:5 * C], F[:, 5 * C:6 * C], F[:, 6 * C:7 * C],
            F[:, 7 * C:8 * C], F[:, 8 * C:9 * C])


def test_fully_observed_window():
    F = feat(np.ones((1, T, C), bool))
    cov, tod, dmean, dsd, n, longest = blocks(F)
    assert F.shape == (1, 9 * C)
    assert np.allclose(cov, 1) and np.allclose(tod, 1) and np.allclose(dmean, 1)
    assert np.allclose(dsd, 0) and np.allclose(n, 0) and np.allclose(longest, 0)


def test_one_contiguous_gap():
    M = np.ones((1, T, C), bool)
    M[0, 100:160, 1] = False                      # 60 bins missing on channel 1 only
    cov, tod, dmean, dsd, n, longest = blocks(feat(M))
    assert np.isclose(cov[0, 1], 1 - 60 / T, atol=1e-6)
    assert np.isclose(longest[0, 1], 60 / T, atol=1e-6)
    assert np.isclose(n[0, 1], 1 / T, atol=1e-6)
    assert np.allclose(cov[0, [0, 2]], 1), "a gap on one channel moved another"


def test_many_short_gaps_differ_from_one_long_gap():
    """Same coverage, different shape -- the pair the gap block exists to separate."""
    a = np.ones((1, T, C), bool); a[0, 0:60, 0] = False
    b = np.ones((1, T, C), bool); b[0, 0:120:2, 0] = False
    ca, _, _, _, na, la = blocks(feat(a))
    cb, _, _, _, nb, lb = blocks(feat(b))
    assert np.isclose(ca[0, 0], cb[0, 0]), "the two cases should have equal coverage"
    assert nb[0, 0] > na[0, 0] and lb[0, 0] < la[0, 0]


def test_time_of_day_block_localises():
    """Missing every night, present every day: coverage falls in one quarter only."""
    M = np.ones((1, T, C), bool)
    M = M.reshape(1, D, W, C)
    M[0, :, :W // 4, 0] = False                   # the first six hours of every day
    _, tod, _, _, _, _ = blocks(feat(M.reshape(1, T, C)))
    q = tod.reshape(1, 4, C)
    assert np.isclose(q[0, 0, 0], 0.0), "the missing quarter is not empty"
    assert np.allclose(q[0, 1:, 0], 1.0), "a neighbouring quarter moved"


def test_day_to_day_variability_block():
    """One day removed entirely: mean daily coverage drops AND its spread rises."""
    even = np.ones((1, D, W, C), bool)
    uneven = np.ones((1, D, W, C), bool); uneven[0, 3, :, 0] = False
    _, _, dm_e, sd_e, _, _ = blocks(feat(even.reshape(1, T, C)))
    _, _, dm_u, sd_u, _, _ = blocks(feat(uneven.reshape(1, T, C)))
    assert np.isclose(sd_e[0, 0], 0.0) and sd_u[0, 0] > 0.3
    assert np.isclose(dm_u[0, 0], 6 / 7, atol=1e-6)
