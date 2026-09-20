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
from sklearn.preprocessing import StandardScaler

from cost import (DSSL, REFERENCE_SHARED, spectral_freqs, spectral_readout,
                  band_keep, band_support)
from models import losses as O
from models.backbones import BACKBONES, build_backbone, register

ROOT = Path(__file__).resolve().parents[1]
HRD = dict(seq_len=672, bins_per_day=96)       # 7 days of 15-min bins
GLOBEM = dict(seq_len=112, bins_per_day=4)     # 28 days of 6-h bins
TINY = dict(output_dims=8, hidden_dims=8, tcn_depth=0, n_layers=1, trend_kernel_cap=2,
            batch_size=4, moco_k=8, device="cpu")


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
        self.assertEqual((c["phase_mode"], c["weights"]), ("raw", "paper"))
        self.assertEqual(ref.cost.weights, O.PAPER)
        self.assertEqual((c["jitter_sigma"], c["shift_sigma"], c["scale_sigma"], c["smooth_bins"]),
                         (0.5, 0.5, 0.5, 0))
        self.assertEqual((c["tcn_depth"], c["phase_readout"], c["alpha"], c["moco_k"]),
                         (7, "angle", 0.005, 4096))

    def test_reference_ignores_dssl_only_settings(self):
        tiny = dict(output_dims=8, hidden_dims=8, tcn_depth=0, batch_size=4, moco_k=8, device="cpu")
        plain = DSSL(2, 28, 4, method="cost_reference", **tiny).config
        dssl_settings = DSSL(2, 28, 4, method="cost_reference", weights="contracted",
                             phase_mode="circular_amp", seasonal_bands="harmonics",
                             trend_kernel_cap=2, smooth_minutes=75.0, jitter_sigma=0.1,
                             **tiny).config
        self.assertEqual(plain, dssl_settings)


class BackboneRegistry(unittest.TestCase):
    def test_registry_and_decoupled_depth(self):
        self.assertEqual(sorted(BACKBONES), ["lstm", "mamba", "mlp", "tcn", "transformer"])
        with self.assertRaisesRegex(ValueError, "unknown backbone"):
            build_backbone("gru", width=8, output_dims=8, seq_len=28)
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
        # RQ4 backbones obey the same contract: (B, T, H) -> (B, T, D), T preserved.
        lstm = build_backbone("lstm", width=8, output_dims=16, seq_len=28, n_layers=2)
        self.assertEqual((lstm.rnn.num_layers, lstm.rnn.bidirectional), (2, True))
        self.assertEqual(tuple(lstm(x).shape), (2, 28, 16))
        mlp = build_backbone("mlp", width=8, output_dims=16, seq_len=28, n_layers=2)
        self.assertEqual(tuple(mlp(x).shape), (2, 28, 16))
        # The MLP mixes nothing across time: perturbing one timestep must leave the others
        # bit-identical. This is what makes it the receptive-field-1 control.
        mlp.eval()
        before = mlp(x)
        perturbed = x.clone()
        perturbed[:, 0] += 10.0
        after = mlp(perturbed)
        self.assertTrue(torch.equal(before[:, 1:], after[:, 1:]))
        self.assertFalse(torch.equal(before[:, 0], after[:, 0]))

    def test_mamba_has_no_substitute(self):
        try:
            import mamba_ssm  # noqa: F401
        except ImportError:
            with self.assertRaisesRegex(ImportError, "mamba-ssm"):
                build_backbone("mamba", width=8, output_dims=16, seq_len=28, n_layers=2)
        else:
            self.skipTest("mamba-ssm is installed; its CUDA path is checked on the cluster")


class ReadoutContract(unittest.TestCase):
    """The three levels of the architecture contract, and the band/readout consistency.

    (a) backbone      -> (B, T, output_dims)
    (b) two branches  -> V^T, V^S each (B, T, output_dims/2)
    (c) frozen vector -> one readout of those sequences

    The FFT acts on the time axis only; the channel count is set by the learned complex map
    inside each band, not by the transform. Amplitude and phase are read from the same V^S,
    so they are two views of one branch, not two encoders.
    """

    def geometry(self, dataset, channels, seq_len, bins_per_day, **over):
        cfg = {**config(dataset)["model"], "model_seed": 1, **over}
        return DSSL(channels, seq_len, bins_per_day, device="cpu", **cfg)

    def test_three_levels_have_the_dimensions_the_method_section_claims(self):
        for dataset, ch, T, bpd in (("hrd", 4, 672, 96), ("globem", 14, 112, 4)):
            with self.subTest(dataset=dataset):
                m = self.geometry(dataset, ch, T, bpd)
                n = m.net
                x = torch.randn(2, T, ch)
                z = n.feature_extractor(n.temporal_encoding(n.input_fc(x)))
                self.assertEqual(tuple(z.shape), (2, T, 320))
                trend, season = n(x)
                self.assertEqual(tuple(trend.shape), (2, T, 160))
                self.assertEqual(tuple(season.shape), (2, T, 160))
                self.assertEqual(n.trend_dims + n.seasonal_dims, 320)

    def test_band_support_is_geometry_only_and_flags_unreadable_bands(self):
        # HRD: every band is read at some readout bin; GLOBEM's second band is not, because
        # its harmonic is the Nyquist bin, which spectral_freqs excludes by design.
        hrd = self.geometry("hrd", 4, 672, 96)
        mask, report = band_support(hrd.net)
        self.assertEqual((report["supported"], report["total"]), (200, 800))
        self.assertEqual(report["bands_never_read"], [])
        self.assertEqual(mask.shape, (len(spectral_freqs(672, 96, 4)), 160))

        globem = self.geometry("globem", 14, 112, 4)
        _, report = band_support(globem.net)
        self.assertEqual((report["supported"], report["total"]), (160, 320))
        self.assertEqual(report["bands_never_read"], [[42, 57]])
        self.assertNotIn(56, spectral_freqs(112, 4, 4))     # Nyquist for T=112

    def test_banded_readout_keeps_exactly_the_supported_coordinates(self):
        for dataset, ch, T, bpd, expected in (("hrd", 4, 672, 96, 560),
                                              ("globem", 14, 112, 4, 480)):
            with self.subTest(dataset=dataset):
                full = self.geometry(dataset, ch, T, bpd, band_readout=False)
                band = self.geometry(dataset, ch, T, bpd, band_readout=True)
                band.net.load_state_dict(full.net.state_dict())
                mask, _ = band_support(full.net)
                keep = band_keep(band.net).numpy()
                np.testing.assert_array_equal(keep, np.flatnonzero(mask.reshape(-1)))
                self.assertTrue((np.diff(keep) > 0).all())      # bin-major, strictly sorted

                x = np.random.default_rng(0).normal(size=(4, T, ch)).astype(np.float32)
                a, b = full.encode(x, parts=True), band.encode(x, parts=True)
                self.assertEqual(b["full"].shape[1], expected)
                # the banded vector is the full one restricted to those columns, exactly
                np.testing.assert_array_equal(a["amp"][:, keep], b["amp"])
                np.testing.assert_array_equal(a["phase"][:, keep], b["phase"])
                np.testing.assert_array_equal(a["trend"], b["trend"])
                # blocks() must tile the vector with no gap or overlap
                edges = sorted(band.blocks().values())
                self.assertEqual(edges[0][0], 0)
                self.assertEqual(edges[-1][1], b["full"].shape[1])
                self.assertTrue(all(edges[i][1] == edges[i + 1][0] for i in range(len(edges) - 1)))

    def test_column_order_does_not_depend_on_seed_and_pairs_stay_aligned(self):
        a = self.geometry("hrd", 4, 672, 96, band_readout=True, model_seed=1)
        b = self.geometry("hrd", 4, 672, 96, band_readout=True, model_seed=99)
        self.assertTrue(torch.equal(band_keep(a.net), band_keep(b.net)))
        circular = self.geometry("hrd", 4, 672, 96, band_readout=True, phase_readout="circular")
        start, width = circular.pair_block()
        amp = circular.blocks()["amplitude"]
        self.assertEqual(width, amp[1] - amp[0])            # one (cos, sin) pair per amplitude
        self.assertEqual(start, amp[1])
        self.assertEqual(circular.blocks()["phase"], (start, start + 2 * width))

    def test_unsupported_coordinates_are_numerical_zero_not_signal(self):
        """The defect this audit reproduces: coordinates with no band support are at the
        float32 floor, and per-column standardisation rescales that floor to unit variance."""
        m = self.geometry("hrd", 4, 672, 96)
        m.net.eval()                                          # encode() uses eval; dropout off
        mask, _ = band_support(m.net)
        x = np.random.default_rng(0).normal(size=(32, 672, 4)).astype(np.float32)
        parts = m.encode(x, parts=True)
        flat = mask.reshape(-1)
        amp_sd = parts["amp"].std(0)
        self.assertGreater(np.median(amp_sd[flat]), 1e-2)      # supported columns carry signal
        self.assertLess(np.median(amp_sd[~flat]), 1e-5)        # unsupported ones do not
        # phase of a ~zero coefficient is not constant, it is atan2 of noise: it varies, but
        # orders of magnitude below a real phase, so it survives as an apparently live feature.
        pha_sd = parts["phase"].std(0)
        self.assertGreater(np.median(pha_sd[flat]), 1.0)
        self.assertLess(np.median(pha_sd[~flat]), 1e-3)
        scaled = StandardScaler().fit_transform(parts["amp"])
        self.assertAlmostEqual(float(np.median(scaled[:, ~flat].std(0))), 1.0, places=3)

    def test_seasonal_loss_is_scale_invariant_while_the_readout_is_not(self):
        """The objective L2-normalises V^S, so it cannot constrain absolute amplitude; the
        readout does not normalise, so amplitude survives there. Both are true at once."""
        m = self.geometry("hrd", 4, 672, 96)
        m.net.eval()
        with torch.no_grad():
            _, season = m.net(torch.randn(8, 672, 4))
        losses, amps = [], []
        for c in (1.0, 4.0):
            scaled = season * c
            amp_term, phase_term = O.seasonal_loss(scaled, scaled.roll(1, 0),
                                                   O.CONTRACTED, "circular_amp")
            losses.append(float(amp_term + phase_term))
            a, _ = spectral_readout(scaled, 96, "angle", 4)
            amps.append(float(a.mean()))
        self.assertAlmostEqual(losses[0], losses[1], places=5)   # invariant
        self.assertGreater(amps[1], 3 * amps[0])                 # equivariant

    def test_both_views_carry_gradient_but_only_through_the_query_encoder(self):
        m = DSSL(2, 28, 4, model_seed=1, **TINY)
        xq = torch.randn(4, 28, 2, requires_grad=True)
        xk = torch.randn(4, 28, 2, requires_grad=True)
        m.cost.train()
        m.cost(xq, xk, update=False).backward()
        self.assertGreater(float(xq.grad.norm()), 0)
        # the key view reaches the loss through the seasonal term, which uses encoder_q
        self.assertGreater(float(xk.grad.norm()), 0)
        self.assertFalse(any(p.requires_grad for p in m.cost.encoder_k.parameters()))


if __name__ == "__main__":
    unittest.main()
