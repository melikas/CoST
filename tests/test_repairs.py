"""Regression checks for the scientific invariants repaired after the audit.

Run: python -m unittest discover -s tests -t . -v
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cost import DSSL, spectral_freqs
from datautils import load_npz, make_folds
from data_processing.hrd_clean import _interpolate_short_gaps, _minute_grid, iter_hrd
from data_processing.hrd_config import CHANNELS
from data_processing.hrd_windows import _window_participant
from data_processing.globem_dataset import _to_binary_label, _prep_participant, FEATURE_COLS
from evaluation_protocol import logistic_probe, participant_mean
from tasks.projection import RawProjection
from tasks.rhythm import (personal_baseline, phase_shift, resolve_phase_levels,
                          window_start_days, amplitude_scale, cosinor_z, individual_markers)
from utils import paired_auc_interval

ROOT = Path(__file__).resolve().parents[1]
# Execution-sized model: every code path of the paper model, none of its size.
TINY = dict(output_dims=8, hidden_dims=8, tcn_depth=0, n_layers=1, trend_kernel_cap=2,
            batch_size=4, moco_k=8, device="cpu")


class DataRepairs(unittest.TestCase):
    def test_streaming_preserves_participant_boundaries_and_rejects_interleaving(self):
        rows = []
        for pid in ["a", "b"]:
            for minute in range(4):
                rows.append({"pid": pid, "timestamp": f"2022-01-01 00:0{minute}:00",
                             "heart_rate": 80, "steps_minutes": 2, "sleep_status": "awake",
                             "screen": 0, "depression_status_baseline": 0,
                             "depression_status_endpoint": 0, "depression_trajectory": "stable",
                             "ces_d_baseline_score": 2, "ces_d_endpoint_score": 2,
                             "emotional_energy": 3})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            summary = {}
            parts = list(iter_hrd(path, chunksize=3, summary=summary))
            self.assertEqual([len(p[0]) for p in parts], [4, 4])
            self.assertEqual(summary["raw_rows"], 8)
            pd.DataFrame(rows + rows[:1]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "noncontiguous"):
                list(iter_hrd(path, chunksize=3))

    def test_long_gaps_are_not_partly_filled(self):
        for gap, expected in [(30, 30), (60, 0)]:
            values = np.r_[0., np.full(gap, np.nan), float(gap + 1)]
            frame = pd.DataFrame({c: values for c in CHANNELS.wear_dependent})
            frame["pid"] = "p"
            filled = _interpolate_short_gaps(frame, 30)
            self.assertEqual(int(filled.HR.iloc[1:-1].notna().sum()), expected)

    def test_absent_minutes_and_original_mask(self):
        frame = pd.DataFrame({c: [1., 2., 3.] for c in CHANNELS.sensors})
        frame["pid"] = "p"
        frame["timestamp"] = pd.to_datetime(["2022-01-01 00:00", "2022-01-01 00:01", "2022-01-01 01:02"])
        grid = _minute_grid(frame)
        self.assertEqual(len(grid), 63)
        self.assertEqual(int(grid["observed/Steps"].sum()), 3)
        grid = _interpolate_short_gaps(grid, 30)
        self.assertEqual(int(grid.Steps.notna().sum()), 3)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _minute_grid(pd.concat([frame, frame.iloc[:1]]))

    def test_complete_final_week_and_mask_survive(self):
        t = pd.date_range("2022-01-01", periods=7 * 1440, freq="min")
        frame = pd.DataFrame({c: 2 + np.cos(np.arange(len(t)) * 2 * np.pi / 1440)
                              for c in CHANNELS.sensors})
        frame["pid"], frame["timestamp"] = "p", t
        frame.loc[150:179, "Steps"] = np.nan
        frame = _interpolate_short_gaps(_minute_grid(frame), 30)
        masks = []
        windows = _window_participant(frame, 7 * 1440, 15, CHANNELS.sensors,
                                      .3, False, observed_out=masks)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0][0].shape, (672, 4))
        self.assertFalse(masks[0][10:12, 1].any())
        self.assertTrue(np.isfinite(windows[0][0]).all())

    def test_globem_all_features_masks_and_strict_labels(self):
        self.assertEqual(len(FEATURE_COLS), 14)
        frame = pd.DataFrame({c: [1., 2., 3., 4.] for c in FEATURE_COLS})
        frame["date"], frame["seg_order"] = pd.Timestamp("2022-01-01"), np.arange(4)
        frame.loc[1, FEATURE_COLS[0]] = np.nan
        filled, observed, _ = _prep_participant(frame, True)
        self.assertEqual(observed.shape, (4, 14))
        self.assertFalse(observed[1, 0])
        self.assertTrue(np.isfinite(filled).all())
        for bad in ["unknown", "yes", 2, np.nan]:
            with self.assertRaises(ValueError):
                _to_binary_label(bad)


class EvaluationRepairs(unittest.TestCase):
    def test_complete_training_resume_matches_uninterrupted_training(self):
        torch.set_num_threads(2)
        x = np.random.default_rng(3).normal(size=(8, 28, 2)).astype(np.float32)
        kwargs = dict(input_dims=2, seq_len=28, bins_per_day=4, model_seed=7, **TINY)
        full = DSSL(**kwargs)
        full.fit(x, n_iters=4, log_every=1, verbose=False)
        partial = DSSL(**kwargs)
        partial.fit(x, n_iters=4, log_every=1, verbose=False, stop_after=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training.pt"
            partial.save_training(path)
            restored = DSSL(**kwargs).load_training(path)
            restored.fit(x, n_iters=4, log_every=1, verbose=False)
            self.assertEqual(restored.history, full.history)
            for key, value in full.cost.state_dict().items():
                self.assertTrue(torch.equal(value, restored.cost.state_dict()[key]), key)
            np.testing.assert_array_equal(full.encode(x), restored.encode(x))
            with self.assertRaises(ValueError):
                restored.fit(x, n_iters=5, verbose=False)

    def test_training_uses_tf32_convolutions_and_encoding_stays_float32(self):
        torch.backends.cudnn.allow_tf32 = False                       # as exact_numerics() leaves it
        model = DSSL(2, 28, 4, model_seed=3, **TINY)
        seen, loss = [], model._loss
        model._loss = lambda batch, update=True: (seen.append(torch.backends.cudnn.allow_tf32),
                                                  loss(batch, update))[1]
        model.fit(np.random.default_rng(3).normal(size=(8, 28, 2)).astype(np.float32),
                  n_iters=2, verbose=False)
        self.assertEqual(seen, [True, True])
        self.assertFalse(torch.backends.cudnn.allow_tf32)

    def test_backbone_encoding_contract_and_reference_preset(self):
        x = np.random.default_rng(9).normal(size=(4, 28, 2)).astype(np.float32)
        for backbone in ["tcn", "transformer"]:
            for encoding in ["none", "sinusoidal", "time2vec"]:
                model = DSSL(2, 28, 4, backbone=backbone, temporal_encoding=encoding,
                             model_seed=9, **TINY)
                model.fit(x, n_iters=1, log_every=1, verbose=False)
                a, b = model.encode(x), model.encode(x)
                np.testing.assert_array_equal(a, b)
                self.assertTrue(np.isfinite(a).all())
        reference = DSSL(2, 28, 4, method="cost_reference", **TINY)
        self.assertEqual(reference.cost.phase_mode, "raw")
        self.assertEqual(reference.scale_sigma, .5)
        self.assertEqual(reference.jitter_sigma, .5)
        self.assertEqual(spectral_freqs(112, 4), [4, 28])

    def test_paired_bootstrap_of_identical_scores_is_degenerate(self):
        scores = np.array([[.1, .2, .8, .9], [.2, .1, .9, .8]])
        interval = paired_auc_interval([0, 0, 1, 1], scores, scores, n_boot=100)
        self.assertEqual(interval["ci95"], [0., 0.])
        with self.assertRaises(ValueError):
            paired_auc_interval([0, 0, 1, 1], scores, scores[:, :3], n_boot=100)

    def test_folds_need_enough_participants_per_class(self):
        with self.assertRaises(ValueError):
            make_folds(np.array(["a", "b", "c", "d"]), np.array([0, 1, 0, 1]), n_folds=3)

    def test_personal_reference_includes_current_gap(self):
        values, pids = np.arange(5.)[:, None], np.array(["p"] * 5)
        self.assertFalse(personal_baseline(values, pids, 4, [0, 7, 14, 21, 70], 21)[2][-1])
        self.assertTrue(personal_baseline(values, pids, 4, [0, 7, 14, 21, 28], 21)[2][-1])
        with self.assertRaises(ValueError):
            personal_baseline(values, pids, 4, [7, 0, 14, 21, 28], 21)
        elapsed = window_start_days(["p_2022-01-01T00:00:00", "p_2022-01-01T06:00:00"])
        np.testing.assert_array_equal(elapsed, [0, .25])

    def test_perturbation_semantics(self):
        self.assertEqual(resolve_phase_levels([.5, 1, 2, 3, 4], 15), [.5, 1, 2, 3, 4])
        with self.assertRaises(ValueError):
            resolve_phase_levels([.5, 1, 2, 3, 4], 360)
        t = np.arange(672)
        x = (3 + 2 * np.cos(2 * np.pi * t / 96))[None, :, None]
        z = cosinor_z(x, 96)
        np.testing.assert_allclose(cosinor_z(amplitude_scale(x, .5, 1, 96), 96), z * .5, atol=1e-10)
        np.testing.assert_allclose(cosinor_z(phase_shift(x, 3, 1, 15), 96), z * np.exp(1j * np.pi / 4), atol=1e-10)
        with self.assertRaises(ValueError):
            phase_shift(x, 4, 1, 360)

    def test_original_individual_markers_and_undefined_cases(self):
        t = np.arange(7 * 96)
        x = (3 + 2 * np.cos(2 * np.pi * t / 96))[None, :, None]
        result = individual_markers(x, np.ones_like(x, dtype=bool), 96, ["Steps"])
        v = result["values"]
        np.testing.assert_allclose(v["IS"], 1, atol=1e-12)
        np.testing.assert_allclose(v["amplitude"], 2, atol=1e-12)
        self.assertTrue(0 < v["RA"][0, 0] < 1)
        self.assertTrue(0 < v["IV"][0, 0] < .1)
        flat = individual_markers(np.ones_like(x), np.ones_like(x, dtype=bool), 96, ["Steps"])
        self.assertTrue(np.isnan(flat["values"]["IS"]).all())
        self.assertTrue(np.isnan(flat["values"]["phase_cos"]).all())
        coarse = x[:, ::24]
        out = individual_markers(coarse, np.ones_like(coarse, dtype=bool), 4, ["Steps"])
        self.assertFalse(out["definition"]["RA_grid_supported"])
        self.assertTrue(np.isnan(out["values"]["RA"]).all())

    def test_projection_is_frozen_and_independent_of_other_test_rows(self):
        x = np.random.default_rng(4).normal(size=(12, 20, 3))
        with self.assertRaises(ValueError):
            RawProjection(3, 1, 8).encode(x)
        model = RawProjection(3, 1, 8).fit(x[:8])
        original = model.encode(x)
        x[-1] *= 100
        np.testing.assert_array_equal(model.encode(x)[:-1], original[:-1])

    def test_real_cache_tiny_model_and_participant_probe(self):
        torch.set_num_threads(2)
        for name in ["datasets/cache/hrd_rescue_v1.npz", "datasets/cache/globem_rescue_v1.npz",
                     "hrd_2224103.npz", "globem_windows.npz"]:
            with self.subTest(dataset=name):
                if not (ROOT / name).exists():
                    self.skipTest(f"{name} is not present in this checkout")
                coh = load_npz(ROOT / name)
                ids, labels = coh.participants()
                train, test = [], []
                for cls in (0, 1):
                    selected = ids[labels == cls][:6]
                    train.extend(selected[:4]); test.extend(selected[4:])
                idx = np.concatenate([np.flatnonzero(coh.pids == p)[:2] for p in train + test])
                x, pids, y = coh.X[idx], coh.pids[idx], coh.y[idx]
                tr = np.isin(pids, train)
                kwargs = dict(input_dims=coh.n_features, seq_len=coh.seq_len,
                              bins_per_day=coh.bins_per_day, **{**TINY, "tcn_depth": 1,
                                                                "output_dims": 16})
                model = DSSL(**kwargs, model_seed=2)
                history = model.fit(x[tr], n_iters=1, log_every=1, verbose=False)
                self.assertTrue(np.isfinite(history["train"]).all())
                reps = model.encode(x, parts=False)
                self.assertTrue(np.isfinite(reps).all())
                grads = [p.grad for p in model.net.parameters() if p.grad is not None]
                self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "encoder.pt"
                    model.save(path)
                    clone = DSSL(**kwargs, model_seed=99).load(path)
                    np.testing.assert_array_equal(reps, clone.encode(x, parts=False))
                # The probe's unit is the participant: one row per person, as in evaluate().
                z_train = participant_mean(reps, pids, np.array(train))
                z_test = participant_mean(reps, pids, np.array(test))
                y_train = np.array([y[pids == p][0] for p in train])
                probe = logistic_probe(1.0, 3).fit(z_train, y_train)
                self.assertEqual(probe.predict_proba(z_test).shape, (4, 2))


if __name__ == "__main__":
    unittest.main()
