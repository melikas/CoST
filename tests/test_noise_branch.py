"""V^N: off by default, bit-identical when off, and real when on."""
import numpy as np
import pytest
import torch

from model_build import build_model, paper_kernels

CFG = dict(alpha=0.005, repr_dims=320, hidden_dims=64, depth=10, backbone="tcn", pe="none",
           time2vec_dim=65, loss_balance="fixed", bin_minutes=15, disentangle=True,
           jitter_sigma=0.1, mask_mode="none", mask_keep_prob=0.5, phase_encoding="circular",
           lr=1e-3, batch_size=32, max_train_length=336, kernels=None)
X = np.random.default_rng(0).normal(size=(8, 336, 3)).astype(np.float32)


def model(**over):
    cfg = dict(CFG, **over)
    torch.manual_seed(0)
    return build_model(cfg, X, 3, "cpu")


def test_off_by_default_and_the_readout_is_unchanged():
    """A config written before V^N existed must rebuild the model it was."""
    a, b = model(), model()
    assert a.net.noise_branch is False
    assert a.net.encode_noise(torch.from_numpy(X)) is None
    va, vb = a.encode(X, mode="forecasting", pool="mean", season_pool="spec"), \
        b.encode(X, mode="forecasting", pool="mean", season_pool="spec")
    assert np.abs(va - vb).max() == 0.0
    assert a.cost.noise_weight == 0.0


def test_on_adds_a_branch_and_widens_the_readout_by_exactly_one_component():
    off = model()
    on = model(noise_weight=0.1, noise_depth=3)
    assert on.net.noise_branch is True
    w_off = off.encode(X, mode="forecasting", pool="mean", season_pool="spec").shape[-1]
    w_on = on.encode(X, mode="forecasting", pool="mean", season_pool="spec").shape[-1]
    assert w_on - w_off == on.net.component_dims, (w_off, w_on)


def test_the_branch_reads_the_residual_not_the_signal():
    """A window that is pure daily rhythm plus drift has no residual, so V^N must be the
    same for it whatever its amplitude -- which a branch reading the signal could not be."""
    m = model(noise_weight=0.1, noise_depth=3).net.eval()
    t = np.arange(336)
    base = (np.sin(2 * np.pi * t / 96) + np.linspace(0, 2, 336))[None, :, None]
    a = np.tile(base, (1, 1, 3)).astype(np.float32)
    b = (a * 3.0).astype(np.float32)
    with torch.no_grad():
        na, nb = m.encode_noise(torch.from_numpy(a)), m.encode_noise(torch.from_numpy(b))
    assert torch.allclose(na, nb, atol=1e-4), "V^N moved with a signal that has no residual"


def test_the_loss_gains_a_term_only_when_the_weight_is_nonzero():
    """Two things make a naive version of this pass for the wrong reason: repr_dropout is
    live in train mode, and the trend view draws a random timestep even in eval. So the
    model goes to eval AND both RNGs are reseeded before each call, leaving the weight as
    the only thing that differs."""
    xq, xk = torch.from_numpy(X[:4]), torch.from_numpy(X[4:])

    def loss_of(m):
        np.random.seed(0)
        torch.manual_seed(0)
        m.cost.eval()
        with torch.no_grad():
            return float(m.cost(xq, xk, update=False))

    m = model(noise_weight=0.5, noise_depth=3)
    l_on = loss_of(m)
    m.cost.noise_weight = 0.0                          # the ONLY thing that changes
    l_off = loss_of(m)
    assert np.isfinite(l_on) and np.isfinite(l_off)
    assert l_on != l_off, "the V^N term did not reach the total"

    # and with the branch absent entirely, the weight cannot do anything
    p = model()
    base = loss_of(p)
    p.cost.noise_weight = 0.5
    assert loss_of(p) == base, "a weight moved the loss with no branch to weigh"


def test_a_random_init_control_gets_the_same_architecture():
    """The control is built from the config with no checkpoint to correct it, so a missing
    key here would compare a two-branch control against a three-branch encoder."""
    from model_build import random_init_model
    c = random_init_model(dict(CFG, noise_weight=0.1, noise_depth=3, model_seed=0),
                          X, 3, "cpu", 0)
    assert c.net.noise_branch is True
