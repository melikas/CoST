"""The instantiated model against the paper's DSSL specification, and the CoST reference
adapter against upstream CoST. Expected values are written out, never recomputed from the
code under test.

Run: python -m unittest discover -s tests -t . -v
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cost import DSSL, REFERENCE_SHARED, spectral_freqs
from models import losses as O
from models.backbones import BACKBONES, build_backbone, register

ROOT = Path(__file__).resolve().parents[1]
HRD = dict(seq_len=672, bins_per_day=96)       # 7 days of 15-min bins
GLOBEM = dict(seq_len=112, bins_per_day=4)     # 28 days of 6-h bins


def config(dataset):
    return json.loads((ROOT / "configs" / f"{dataset}.json").read_text())


class PaperModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.hrd = DSSL(3, **HRD, device="cpu", model_seed=1, **config("hrd")["model"])

    def test_config_files_are_the_paper_defaults(self):
        for dataset, geometry, channels in (("hrd", HRD, 4), ("globem", GLOBEM, 14)):
            with self.subTest(dataset=dataset):
                cfg = config(dataset)
                self.assertEqual(cfg["normalization"], "within_person")
                self.assertEqual((cfg["steps"], cfg["val_frac"]), (6000, 0.1))
                self.assertEqual(DSSL(channels, **geometry, device="cpu", **cfg["model"]).config,
                                 DSSL(channels, **geometry, device="cpu").config)

    def test_hrd_architecture(self):
        m, c = self.hrd, self.hrd.config
        self.assertEqual((c["method"], c["backbone"], c["temporal_encoding"]), ("dssl", "tcn", "none"))
        self.assertEqual((c["tcn_depth"], c["receptive_field"]), (7, 1021))
        self.assertEqual(len(m.net.feature_extractor.net), 8)          # blocks at dilations 1..128
        self.assertEqual(c["bands"], [[1, 10], [10, 17], [17, 24], [24, 31]])
        self.assertEqual([layer.out_channels for layer in m.net.sfd], [40, 40, 40, 40])
        self.assertEqual(c["trend_kernels"], [1, 2, 4, 8, 16, 32, 64])   # up to T/8 = 84
        self.assertEqual((m.net.trend_dims, m.net.seasonal_dims), (160, 160))
        # Complex band weights counted once per complex number, as torch's numel does.
        self.assertEqual(sum(p.numel() for p in m.net.parameters()), 7_451_984)

    def test_hrd_objective(self):
        c, cost = self.hrd.config, self.hrd.cost
        self.assertEqual(c["weights"], "contracted")
        self.assertEqual(cost.weights, O.CONTRACTED)
        self.assertAlmostEqual(cost.weights.trend, 0.277)
        self.assertAlmostEqual(cost.weights.amp, 0.174 / 1.174)
        self.assertAlmostEqual(cost.weights.phase, 1.0 / 1.174)
        self.assertEqual(cost.phase_mode, "circular_amp")
        self.assertEqual((cost.alpha, cost.K, cost.m, cost.T), (0.005, 4096, 0.999, 0.07))
        # Working default since the w_eq dose-response (HRD, 5 folds): 1.0 / 0.05 / 0 gave RQ1
        # gain -0.0343 / -0.0136 / +0.0096, ordered that way in every fold.
        self.assertEqual(cost.w_eq, 0.0)
        self.assertIsNone(cost.head_eq)
        self.assertEqual(cost.w_ac, 0.0)
        self.assertEqual(c["smooth_bins"], 5)                             # 75 min at 15-min bins
        self.assertEqual((c["jitter_sigma"], c["shift_sigma"], c["scale_sigma"]), (0.1, 0.5, 0.0))
        self.assertEqual((c["lr"], c["batch_size"], c["mask_mode"]), (5e-4, 64, "none"))

    def test_hrd_readout(self):
        self.assertEqual(spectral_freqs(672, 96), [1, 7, 14, 21, 28])
        x = np.random.default_rng(0).normal(size=(2, 672, 3)).astype(np.float32)
        self.assertEqual(self.hrd.config["phase_readout"], "angle")
        self.assertEqual(self.hrd.encode(x).shape, (2, 160 + 800 + 800))
        self.assertIsNone(self.hrd.pair_block())
        self.assertEqual(self.hrd.blocks(), {"trend": (0, 160), "amplitude": (160, 960), "phase": (960, 1760)})
        circular = DSSL(3, **HRD, device="cpu", phase_readout="circular")
        circular.net.load_state_dict(self.hrd.net.state_dict())
        self.assertEqual(circular.pair_block(), (960, 800))
        self.assertEqual(circular.blocks()["phase"], (960, 2560))
        self.assertEqual(circular.encode(x).shape, (2, 160 + 800 + 2 * 800))

    def test_readout_norm_decides_whether_amplitude_survives(self):
        """The measured cause of RQ2's inverted amplitude arm: per-timestep normalisation
        removes the amplitude of the seasonal sequence entirely. "split" keeps it."""
        from cost import spectral_readout
        z = torch.randn(2, 672, 8, generator=torch.Generator().manual_seed(0))
        for norm, factor in (("none", 2.0), ("split", 2.0), ("timestep", 1.0)):
            one, _ = spectral_readout(z, 96, "angle", readout_norm=norm)
            two, _ = spectral_readout(2 * z, 96, "angle", readout_norm=norm)
            torch.testing.assert_close(two, factor * one, rtol=2e-4, atol=1e-5)
        with self.assertRaisesRegex(ValueError, "readout_norm"):
            spectral_readout(z, 96, "angle", readout_norm="per_window")

    def test_split_readout_phase_tracks_a_known_shift(self):
        """Rolling the sequence by k bins must rotate bin f's phase by -2*pi*f*k/T. Rolling
        commutes with per-timestep normalisation, so the relation is exact in every mode."""
        from cost import spectral_readout, spectral_freqs
        z = torch.randn(3, 672, 6, generator=torch.Generator().manual_seed(2))
        bins = spectral_freqs(672, 96)
        k = 13
        for norm in ("none", "timestep", "split"):
            _, before = spectral_readout(z, 96, "angle", readout_norm=norm)
            _, after = spectral_readout(torch.roll(z, k, dims=1), 96, "angle", readout_norm=norm)
            expected = -2 * np.pi * torch.tensor(bins, dtype=torch.float) * k / 672
            moved = (after - before).reshape(3, len(bins), 6)
            wrapped = torch.atan2(torch.sin(moved - expected[None, :, None]),
                                  torch.cos(moved - expected[None, :, None]))
            self.assertLess(wrapped.abs().max().item(), 1e-2, norm)   # eps=1e-3 biases the angle

    def test_split_readout_is_amplitude_of_none_and_phase_of_timestep(self):
        """The mode's whole contract, and the reason it is a clean test of the trade-off: it
        changes nothing except which sequence each block is read from.

        Note the two sequences are NOT related by a positive scalar -- per-timestep normalisation
        divides by the time-varying norm ||z(t)||, which reshapes the spectrum -- so "none" and
        "timestep" give genuinely different angles, not merely differently conditioned ones.
        """
        from cost import spectral_readout
        z = torch.randn(3, 672, 7, generator=torch.Generator().manual_seed(4))
        amp_split, phase_split = spectral_readout(z, 96, "angle", readout_norm="split")
        amp_plain, phase_plain = spectral_readout(z, 96, "angle", readout_norm="none")
        amp_norm, phase_norm = spectral_readout(z, 96, "angle", readout_norm="timestep")
        torch.testing.assert_close(amp_split, amp_plain, rtol=0, atol=0)
        torch.testing.assert_close(phase_split, phase_norm, rtol=0, atol=0)
        self.assertGreater((phase_plain - phase_norm).abs().max().item(), 0.1)
        self.assertGreater((amp_plain - amp_norm).abs().max().item(), 0.1)

    def test_eps_already_conditions_near_zero_amplitude_in_every_mode(self):
        """Guards the eps=1e-3 fix: at amplitudes near the floor, the measured batch-to-batch
        jitter (7.5e-08) must not move any mode's angle. It moved 'none' by 0.65 rad at the old
        eps=1e-6, which is what that fix addressed -- so conditioning is NOT what separates the
        modes now.
        The realistic regime is the one the encoder produces: ||z(t)|| is O(1) and individual BINS
        are silent. (With a uniformly tiny z, normalisation instead divides by a tiny norm and
        amplifies the noise -- 3.03 rad -- so neither mode is unconditionally safer.)
        """
        from cost import spectral_readout, spectral_freqs
        g = torch.Generator().manual_seed(5)
        z = torch.randn(4, 672, 5, generator=g)
        silent = spectral_freqs(672, 96)[2]                     # empty this bin exactly
        Z = torch.fft.rfft(z, dim=1)
        Z[:, silent] = 0
        z = torch.fft.irfft(Z, n=672, dim=1)
        noise = 7.5e-8 * torch.randn(4, 672, 5, generator=g)    # the measured batch-to-batch jitter
        for norm in ("none", "timestep", "split"):
            _, a = spectral_readout(z, 96, "angle", readout_norm=norm)
            _, b = spectral_readout(z + noise, 96, "angle", readout_norm=norm)
            swing = torch.atan2(torch.sin(b - a), torch.cos(b - a)).abs().max().item()
            self.assertLess(swing, 0.05, norm)

    def test_globem_architecture(self):
        c = DSSL(14, **GLOBEM, device="cpu", **config("globem")["model"]).config
        self.assertEqual((c["tcn_depth"], c["receptive_field"]), (4, 125))
        self.assertEqual(c["bands"], [[1, 42], [42, 57]])
        self.assertEqual(c["trend_kernels"], [1, 2, 4, 8])                # up to T/8 = 14
        self.assertEqual(c["smooth_bins"], 0)                              # no odd box fits 6-h bins
        self.assertEqual(spectral_freqs(112, 4), [4, 28])


class CoSTReference(unittest.TestCase):
    def test_reference_is_upstream_cost_behind_the_shared_readout(self):
        model = config("hrd")["model"]
        shared = {k: model[k] for k in REFERENCE_SHARED if k in model}
        ref = DSSL(4, **HRD, method="cost_reference", device="cpu", **shared)
        c = ref.config
        self.assertEqual(c["bands"], [[0, 337]])                           # one full-spectrum band
        self.assertEqual(c["trend_kernels"], [2 ** i for i in range(9)])   # up to T/2 = 336
        self.assertEqual((c["phase_mode"], c["weights"], c["w_eq"], c["w_ac"]),
                         ("raw", "paper", 0.0, 0.0))
        self.assertEqual(ref.cost.weights, O.PAPER)
        self.assertIsNone(ref.cost.head_eq)
        self.assertEqual((c["jitter_sigma"], c["shift_sigma"], c["scale_sigma"], c["smooth_bins"]),
                         (0.5, 0.5, 0.5, 0))
        self.assertEqual((c["tcn_depth"], c["phase_readout"], c["alpha"], c["moco_k"]),
                         (7, "angle", 0.005, 4096))

    def test_reference_ignores_dssl_only_settings(self):
        tiny = dict(output_dims=8, hidden_dims=8, tcn_depth=0, batch_size=4, moco_k=8, device="cpu")
        plain = DSSL(2, 28, 4, method="cost_reference", **tiny).config
        dssl_settings = DSSL(2, 28, 4, method="cost_reference", w_eq=1.0, weights="contracted",
                             phase_mode="circular_amp", seasonal_bands="harmonics",
                             trend_kernel_cap=2, smooth_minutes=75.0, jitter_sigma=0.1,
                             **tiny).config
        self.assertEqual(plain, dssl_settings)


class PreEncoderDecomposition(unittest.TestCase):
    """The 24-h filter and the two-encoder split that restricts each branch's INPUT."""

    def test_daily_filter_is_zero_phase_and_nulls_the_daily_harmonics(self):
        from models.encoder import daily_average_kernel
        for bins_per_day, seq_len in ((96, 672), (4, 112)):
            k = daily_average_kernel(bins_per_day)
            self.assertEqual(k.numel(), bins_per_day + 1)          # odd: centred, zero phase
            self.assertAlmostEqual(float(k.sum()), 1.0, places=6)
            torch.testing.assert_close(k, k.flip(0))
            padded = torch.nn.functional.pad(k, (0, seq_len - k.numel()))
            H = torch.fft.rfft(torch.roll(padded, -(k.numel() // 2)))
            self.assertLess(H.imag.abs().max().item(), 1e-6)       # real => zero phase
            self.assertAlmostEqual(H.abs()[0].item(), 1.0, places=5)          # keeps the level
            days = seq_len // bins_per_day
            for harmonic in range(1, bins_per_day // 2 + 1):     # 24 h and every faster harmonic
                self.assertLess(H.abs()[days * harmonic].item(), 1e-6, f'{bins_per_day}:{harmonic}')

    def test_decomposition_reconstructs_and_keeps_missingness(self):
        from models.encoder import daily_average_kernel, decompose_daily
        x = torch.randn(4, 672, 3, generator=torch.Generator().manual_seed(0))
        x[0, 5:9, 1] = float('nan')
        trend, seasonal = decompose_daily(x, daily_average_kernel(96))
        finite = torch.isfinite(x)
        torch.testing.assert_close((trend + seasonal)[finite], x[finite], rtol=0, atol=1e-5)
        self.assertTrue(torch.isnan(trend[~finite]).all() and torch.isnan(seasonal[~finite]).all())
        self.assertTrue(torch.isfinite(trend[finite]).all())

    def test_trend_view_cannot_see_the_daily_rhythm(self):
        """The architectural claim, stated as an invariant: adding a 24-h component changes the
        seasonal view and leaves the trend view alone."""
        from models.encoder import daily_average_kernel, decompose_daily
        t = torch.arange(672, dtype=torch.float)
        base = torch.randn(1, 672, 1, generator=torch.Generator().manual_seed(1))
        rhythm = torch.sin(2 * np.pi * t / 96 + 0.7).view(1, 672, 1)
        kernel = daily_average_kernel(96)
        trend_a, seasonal_a = decompose_daily(base, kernel)
        trend_b, seasonal_b = decompose_daily(base + rhythm, kernel)
        self.assertLess((trend_b - trend_a).abs().max().item(), 1e-5)
        torch.testing.assert_close(seasonal_b - seasonal_a, rhythm, rtol=0, atol=1e-5)

    def test_decomposed_encoder_matches_the_shared_readout_and_splits_its_heads(self):
        from models.encoder import DecomposedEncoder
        shared = DSSL(4, **HRD, device='cpu', model_seed=1)
        split = DSSL(4, **HRD, device='cpu', model_seed=1, decompose=True)
        self.assertIsInstance(split.net, DecomposedEncoder)
        self.assertEqual(split.blocks(), shared.blocks())               # same readout geometry
        self.assertEqual((split.net.trend_dims, split.net.seasonal_dims), (160, 160))
        self.assertEqual(len(split.net.trend_net.sfd), 0)               # trend encoder: no seasonal head
        self.assertEqual(len(split.net.seasonal_net.tfd), 0)            # seasonal encoder: no trend head
        self.assertTrue(split.config['decompose'])
        shared_params = {id(p) for p in split.net.trend_net.parameters()}
        self.assertFalse(shared_params & {id(p) for p in split.net.seasonal_net.parameters()})

    def test_decomposed_encoder_trains_and_reloads(self):
        x = np.random.default_rng(0).normal(size=(8, 672, 4)).astype(np.float32)
        model = DSSL(4, **HRD, device='cpu', model_seed=2, decompose=True, output_dims=16,
                     hidden_dims=8, tcn_depth=0, batch_size=4, moco_k=8)
        history = model.fit(x, n_iters=2, log_every=2, verbose=False)
        self.assertTrue(np.isfinite(history['train']).all())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'enc.pt'
            model.save(path)
            reloaded = DSSL(4, **HRD, device='cpu', model_seed=3, decompose=True, output_dims=16,
                            hidden_dims=8, tcn_depth=0, batch_size=4, moco_k=8).load(path)
            np.testing.assert_allclose(reloaded.encode(x), model.encode(x), rtol=1e-5, atol=2e-3)


class TrendViews(unittest.TestCase):
    TINY = dict(output_dims=8, hidden_dims=8, tcn_depth=0, batch_size=4, moco_k=8, device="cpu")

    def test_default_is_same_and_the_reference_ignores_it(self):
        self.assertEqual(DSSL(2, 28, 4, **self.TINY).config["trend_views"], "same")
        self.assertFalse(hasattr(DSSL(2, 28, 4, **self.TINY).cost, "queue_ids"))   # checkpoints unchanged
        ref = DSSL(2, 28, 4, method="cost_reference", trend_views="disjoint_days", **self.TINY)
        self.assertEqual(ref.config["trend_views"], "same")
        with self.assertRaisesRegex(ValueError, "trend_views"):
            DSSL(2, 28, 4, trend_views="crop", **self.TINY)

    def test_disjoint_day_views_share_no_timestep(self):
        cost = DSSL(3, 28, 4, trend_views="disjoint_days", **self.TINY).cost    # 7 days of 4 bins
        x = torch.randn(64, 28, 3, generator=torch.Generator().manual_seed(0))
        (xa, xb), (va, vb) = cost._disjoint_days(x, x + 1)
        self.assertFalse((va & vb).any())
        self.assertTrue((va.sum(1) == 3 * 4).all() and (vb.sum(1) == 3 * 4).all())
        days_a = va.view(64, 7, 4)
        self.assertTrue((days_a.all(2) | ~days_a.any(2)).all())          # whole days only
        self.assertTrue(torch.isnan(xa[~va]).all() and torch.isnan(xb[~vb]).all())
        torch.testing.assert_close(xa[va], x[va])
        torch.testing.assert_close(xb[vb], (x + 1)[vb])

    def test_a_weeks_own_keys_are_not_its_negatives(self):
        q = torch.nn.functional.normalize(torch.randn(2, 4, generator=torch.Generator().manual_seed(1)), dim=1)
        queue = torch.cat([q, torch.nn.functional.normalize(-q, dim=1)]).T       # 4 x 4: copies, then opposites
        own = torch.tensor([[True, False, False, False], [False, True, False, False]])
        _, unmasked = O.moco_ce_loss(q, q, queue)
        _, masked = O.moco_ce_loss(q, q, queue, neg_mask=own)
        self.assertEqual((unmasked, masked), (0.0, 1.0))

    def test_disjoint_days_trains_and_records_window_ids(self):
        model = DSSL(2, 28, 4, trend_views="disjoint_days", model_seed=3, **self.TINY)
        x = np.random.default_rng(0).normal(size=(6, 28, 2)).astype(np.float32)
        history = model.fit(x, n_iters=4, log_every=2, verbose=False)
        self.assertTrue(np.isfinite(history["train"]).all())
        ids = model.cost.queue_ids
        self.assertTrue(((ids >= 0) & (ids < 6)).all())                 # 4 steps x 4 = two full queues


class BackboneRegistry(unittest.TestCase):
    def test_registry_and_decoupled_depth(self):
        self.assertEqual(sorted(BACKBONES), ["mamba", "tcn", "transformer"])
        with self.assertRaisesRegex(ValueError, "unknown backbone"):
            build_backbone("lstm", width=8, output_dims=8, seq_len=28)
        with self.assertRaisesRegex(ValueError, "already registered"):
            register("tcn")(lambda **kw: None)
        x = torch.randn(2, 28, 8)
        # tcn_depth counts dilation levels and is the only option the TCN reads.
        tcn = build_backbone("tcn", width=8, output_dims=16, seq_len=28, tcn_depth=2, n_layers=9)
        self.assertEqual((tcn.depth, len(tcn.net), tcn.receptive_field), (2, 3, 29))
        self.assertEqual(tuple(tcn(x).shape), (2, 28, 16))
        auto = build_backbone("tcn", width=8, output_dims=16, seq_len=672)
        self.assertEqual((auto.depth, auto.receptive_field), (7, 1021))
        # n_layers is the Transformer's layer count; tcn_depth does not reach it.
        transformer = build_backbone("transformer", width=8, output_dims=16, seq_len=28,
                                     tcn_depth=7, n_layers=3, n_heads=2)
        self.assertEqual(len(transformer.layers), 3)
        self.assertEqual(transformer.layers[0].self_attn.num_heads, 2)
        self.assertEqual(tuple(transformer(x).shape), (2, 28, 16))

    def test_mamba_has_no_substitute(self):
        try:
            import mamba_ssm  # noqa: F401
        except ImportError:
            with self.assertRaisesRegex(ImportError, "mamba-ssm"):
                build_backbone("mamba", width=8, output_dims=16, seq_len=28, n_layers=2)
        else:
            self.skipTest("mamba-ssm is installed; its CUDA path is checked on the cluster")


if __name__ == "__main__":
    unittest.main()
