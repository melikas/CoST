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
