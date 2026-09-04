"""segN pooling: a strict generalisation of the default, on the real encoder."""
import numpy as np
import pytest
import torch

from local_context import local_context
from model_build import random_init_model

NPZ = "hrd_2224103.npz"
pytestmark = pytest.mark.skipif(not __import__("pathlib").Path(NPZ).exists(),
                                reason=f"{NPZ} not present")


@pytest.fixture(scope="module")
def enc():
    ctx = local_context(NPZ, 7)
    torch.set_num_threads(2)
    m = random_init_model(ctx.cfg, ctx.X, ctx.n_sensors, "cpu", 7)
    m.net.eval()
    return m, np.asarray(ctx.X[:4], dtype=np.float32), ctx


def take(m, X, pool, season_pool):
    return m.encode(X, mode="forecasting", pool=pool, season_pool=season_pool,
                    batch_size=4).squeeze(1)


def test_seg1_is_exactly_mean(enc):
    """If these differ at all, segN is not a generalisation of the default and every
    archived result would be a different model under the new code path."""
    m, X, _ = enc
    for sp in (None, "spec"):
        a = take(m, X, "mean", sp)
        b = take(m, X, "seg1", sp)
        assert a.shape == b.shape
        assert np.abs(a - b).max() == 0.0, f"seg1 != mean with season_pool={sp}"


def test_segn_width_scales_and_stays_narrower_than_the_spectral_readout(enc):
    m, X, _ = enc
    w1 = take(m, X, "seg1", None).shape[1]
    for n in (2, 4, 7):
        assert take(m, X, f"seg{n}", None).shape[1] == n * w1
    # the claim that motivated this: the winning readout is not winning on dimensions
    assert take(m, X, "seg2", None).shape[1] < take(m, X, "mean", "spec").shape[1]


def test_first_segment_is_the_mean_of_the_first_half(enc):
    m, X, _ = enc
    with torch.no_grad():
        out_t, _ = m.net(torch.from_numpy(X))
    half = out_t.shape[1] // 2
    got = take(m, X, "seg2", None)[:, :out_t.shape[2]]
    assert np.allclose(got, out_t[:, :half].mean(dim=1).numpy(), atol=1e-5)


def test_a_bad_segment_count_is_refused(enc):
    m, X, _ = enc
    with pytest.raises(ValueError, match="segments of a"):
        take(m, X, "seg99999", None)
