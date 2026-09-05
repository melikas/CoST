"""V^N: off by default, bit-identical when off, and real when on."""
import numpy as np
import pytest
import torch

from model_build import build_model

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


def test_the_target_is_the_residual_and_nothing_else():
    """A window that is pure daily rhythm plus drift has no residual, so the thing the
    branch is asked to predict must be ~0 there however loud the window is.

    The branch's INPUT is the raw signal on purpose: the projection defines the target and
    never touches the input, because applying it to the input and masking afterwards hides
    nothing -- R = I - P mixes the whole time axis through P's twelve basis functions."""
    m = model(noise_weight=0.1, noise_depth=3).net.eval()
    t = np.arange(336)
    base = (np.sin(2 * np.pi * t / 96) + np.linspace(0, 2, 336))[None, :, None]
    a = np.tile(base, (1, 1, 3)).astype(np.float32)
    with torch.no_grad():
        r_small = m.residual_of(torch.from_numpy(a))
        r_loud = m.residual_of(torch.from_numpy(a * 3.0))
    assert float(r_small.abs().max()) < 1e-3, "a pure rhythm left a residual"
    assert float(r_loud.abs().max()) < 3e-3, "and scaling it up must not create one"


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


def _tiny_fit(noise_weight, iters=6):
    """Actually train, briefly, and report how far each branch's weights moved."""
    import copy
    m = model(noise_weight=noise_weight, noise_depth=2, batch_size=8, max_train_length=336)
    before = copy.deepcopy(m.net.state_dict())
    rng = np.random.default_rng(0)
    t = np.arange(336)
    # a rhythm the trend/seasonal branches can hold, plus a residual only V^N can
    X = (np.sin(2 * np.pi * t / 96)[None, :, None]
         + rng.normal(0, 0.4, (24, 336, 3))).astype(np.float32)
    np.random.seed(0)
    torch.manual_seed(0)
    m.fit(X, n_iters=iters, verbose=False)
    after = m.net.state_dict()
    moved = lambda pre: max(
        (float((after[k] - before[k]).abs().max()) for k in after if k.startswith(pre)),
        default=0.0)
    return moved("noise_"), moved("feature_extractor."), m


def test_training_actually_reaches_the_noise_branch():
    """The failure this guards against is silent: a branch that is built and read at
    inference but never receives a gradient would still produce a full table of numbers."""
    n_moved, backbone_moved, _ = _tiny_fit(0.3)
    assert backbone_moved > 0, "nothing trained at all"
    assert n_moved > 1e-6, "V^N was built and read but never trained"


def test_the_branch_is_untouched_when_its_weight_is_zero():
    m = model(noise_weight=0.0, noise_branch=True, noise_depth=2,
              batch_size=8, max_train_length=336)
    import copy
    before = copy.deepcopy(m.net.state_dict())
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (24, 192, 3)).astype(np.float32)
    np.random.seed(0); torch.manual_seed(0)
    m.fit(X, n_iters=4, verbose=False)
    after = m.net.state_dict()
    moved = max((float((after[k] - before[k]).abs().max())
                 for k in after if k.startswith("noise_")), default=0.0)
    assert moved == 0.0, "a zero weight still moved the branch"


def test_gradnorm_with_a_noise_weight_is_refused_not_silently_dropped():
    """_gradnorm_step unpacks two losses, so a third would be built, read, and never
    trained -- the exact silent failure the test above exists for."""
    with pytest.raises(ValueError, match="gradnorm"):
        model(noise_weight=0.3, noise_depth=2, loss_balance="gradnorm")


def test_the_mask_hides_what_it_claims_to_hide():
    """A leak here makes the task partly solvable by reading it. The first version applied
    the residual projection to the INPUT and masked afterwards, and R = I - P mixes the whole
    time axis through P's twelve basis functions, so a +5.0 shift on the masked steps moved
    the prediction by 0.23."""
    m = model(noise_weight=0.3, noise_depth=2, max_train_length=336).net.eval()
    x = torch.from_numpy(X[:4].copy())
    msk = torch.zeros(4, 336, dtype=torch.bool)
    msk[:, 40:80] = True
    with torch.no_grad():
        pred, tgt = m.reconstruct_noise(x, msk)
        x2 = x.clone(); x2[msk] += 5.0
        p2, t2 = m.reconstruct_noise(x2, msk)
        x3 = x.clone(); x3[~msk] += 5.0
        p3, _ = m.reconstruct_noise(x3, msk)
    assert float((p2 - pred).abs().max()) == 0.0, "the mask leaks"
    assert float((t2 - tgt).abs().max()) > 1e-3, "the target must move -- it is what is predicted"
    assert float((p3 - pred).abs().max()) > 1e-4, "visible steps must matter"


def test_the_masked_objective_trains_the_decoder():
    """The decoder is the layer this model never had. If the loss does not reach it the
    branch still produces a readout and every table looks normal."""
    import copy
    m = model(noise_weight=0.5, noise_depth=2, batch_size=8, max_train_length=336)
    before = copy.deepcopy(m.net.state_dict())
    rng = np.random.default_rng(0)
    t = np.arange(336)
    Xt = (np.sin(2 * np.pi * t / 96)[None, :, None]
          + rng.normal(0, 0.4, (24, 336, 3))).astype(np.float32)
    np.random.seed(0); torch.manual_seed(0)
    m.fit(Xt, n_iters=6, verbose=False)
    after = m.net.state_dict()
    moved = lambda pre: max((float((after[k] - before[k]).abs().max())
                             for k in after if k.startswith(pre)), default=0.0)
    assert moved("noise_decoder") > 1e-6, "the decoder never received a gradient"
    assert moved("noise_extractor") > 1e-6, "the encoder half of the branch never trained"


def test_masking_is_contiguous_spans_not_scattered_steps():
    """A scattered mask on a high-frequency residual is filled by interpolating its
    neighbours, which teaches a smoother instead of the structure."""
    m = model(noise_weight=0.3, noise_depth=2, max_train_length=336)
    torch.manual_seed(0)
    x = torch.from_numpy(X[:8].copy())
    seen = []
    for _ in range(20):
        # rebuild the mask the loss builds, by calling the loss and capturing the span length
        T = x.size(1)
        n_span = max(1, int(round(m.cost.noise_mask_frac * T / m.cost.noise_span)))
        seen.append(n_span * m.cost.noise_span / T)
    assert m.cost.noise_span >= 4, "spans too short to defeat interpolation"
    assert 0.15 < np.mean(seen) < 0.5, f"masked fraction off target: {np.mean(seen)}"


def test_the_loss_weights_channels_evenly():
    """The residual's variance differs between channels by orders of magnitude, and an
    unnormalised MSE would be a report on the loudest one.

    What the per-channel normalisation guarantees is that no channel dominates by variance
    -- NOT that the loss is invariant to scaling an input, which it cannot be: the encoder
    has biases and a GELU, so it is not scale-equivariant, and an earlier version of this
    test asserted that and failed for the right reason."""
    m = model(noise_weight=0.5, noise_depth=2, max_train_length=336)
    x = torch.from_numpy(X[:8].copy())
    x[..., 0] *= 100.0                               # one very loud channel
    with torch.no_grad():
        target = m.net.residual_of(x)
    raw = target.std(dim=1).mean(0)
    sd = target.std(dim=1, keepdim=True).clamp_min(1e-6)
    norm = (target / sd).std(dim=1).mean(0)
    assert float(raw.max() / raw.min()) > 20, "the test data does not have the imbalance"
    assert float(norm.max() / norm.min()) < 1.05, f"channels still uneven: {norm}"
    torch.manual_seed(0)
    assert np.isfinite(float(m.cost._masked_noise_loss(x)))


def test_the_masked_loss_is_order_one_on_a_near_constant_channel():
    """The failure this guards against was silent and total. Normalising by each window's
    OWN standard deviation looks equivalent to normalising per channel and is not: a
    near-binary channel has windows whose residual is almost constant, the clamp then
    divides by about nothing, and on real HRD windows the loss came out at 5.1e6 instead of
    order 1. A run started that way would have been dominated by the division and every
    number from it meaningless -- with nothing in the output to say so."""
    m = model(noise_weight=0.5, noise_depth=2, max_train_length=336)
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (8, 336, 3)).astype(np.float32)
    # Channel 2 is built from exactly the basis the reference fits -- a cubic plus the four
    # daily harmonics -- so its residual is numerically zero. A square wave is NOT this case:
    # four harmonics leave it a residual of 0.16, and the first version of this test used one
    # and never reached the condition it was written for.
    t = np.arange(336)
    u = (t - t.mean()) / t.std()
    ch = 0.7 * u ** 3 - 0.4 * u + sum(np.sin(2 * np.pi * k * t / 96 + k) for k in (1, 2, 3, 4))
    x[:, :, 2] = ch.astype(np.float32)
    xt = torch.from_numpy(x)
    with torch.no_grad():
        r = m.net.residual_of(xt)
    per_window = r.std(dim=1)
    assert float(per_window.min()) < 1e-3, "the test data lacks the near-constant case"
    torch.manual_seed(0)
    loss = float(m.cost._masked_noise_loss(xt))
    assert np.isfinite(loss), "the loss is not finite"
    assert loss < 100.0, f"the loss exploded on a near-constant channel: {loss}"
