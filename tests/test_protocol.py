"""Outcome isolation in the participant protocol, and the circular phase scaler.

Run: python -m unittest discover -s tests -t . -v
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from evaluation_protocol import (IsotropicPairScaler, disentanglement, disentanglement_intervals,
                                 evaluate, feature_scaler, logistic_probe)
from tasks.personalized import personalized_records
from tasks.rhythm import window_rhythm


class EvaluationProtocolTests(unittest.TestCase):
    def test_rq2_is_label_free_with_contiguous_personal_history(self):
        class Encoder:
            def encode(self, x, parts=False, batch_size=16):
                return np.concatenate([x.mean(1), x.std(1), x[:, ::24].reshape(len(x), -1)], axis=1)
        t = np.arange(336) * 2 * np.pi / 48
        x = np.stack([np.sin(t + phase)[:, None] for phase in np.linspace(-.6, .6, 7)]).astype('float32')
        ids = np.array([f'p_2022-01-{1 + 7 * i:02d}T00:00:00' for i in range(5)] +
                       ['p_2022-02-05T00:00:00', 'p_2022-02-12T00:00:00'])
        rows, status = personalized_records(Encoder(), 'dssl', x, x, np.array(['p'] * 7), ids, ['p'], 48, 30)
        self.assertEqual(status['uses_endpoint_labels'], False)
        self.assertEqual(status['baseline_weeks'], 4)
        self.assertTrue(any(r['perturbation'] == 'phase' for r in rows))
        self.assertTrue(any(r['perturbation'] == 'amplitude' for r in rows))
        rows, status = personalized_records(Encoder(), 'dssl', x[:, :112], x[:, :112], np.array(['p'] * 7),
                                            ids, ['p'], 4, 360)
        self.assertEqual(rows, [])
        self.assertEqual(status['status'], 'not_applicable')

    def test_test_labels_do_not_select_or_fit_models(self):
        rng = np.random.default_rng(43)
        pids = np.repeat([f'p{i:02}' for i in range(12)], 2)
        labels = np.repeat(np.arange(12) % 2, 2)
        features = {k: rng.normal(size=(24, 5)) for k in ['raw', 'untrained', 'dssl', 'cost_reference_adapter',
                                                          'distribution', 'random_projection', 'yan_cosinor']}
        markers = {k: rng.normal(size=(24, 1)) for k in ['MESOR', 'amplitude', 'phase_cos', 'phase_sin',
                                                         'IS', 'IV', 'RA']}
        ids = np.unique(pids)
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a', Path(directory) / 'b'
            # Columns 1-2 / 3-4 stand in for a circular readout's (cos, sin) pairs.
            pairs = {'dssl': (1, 2), 'untrained': (1, 2)}
            evaluate(features, markers, pids, labels, ids[:8], ids[8:], ['Steps'], a, 17, pair_blocks=pairs)
            changed = labels.copy()
            changed[np.isin(pids, ids[8:])] = 1 - changed[np.isin(pids, ids[8:])]
            evaluate(features, markers, pids, changed, ids[:8], ids[8:], ['Steps'], b, 17, pair_blocks=pairs)
            pd.testing.assert_frame_equal(pd.read_csv(a / 'probe_selection.csv'), pd.read_csv(b / 'probe_selection.csv'))
            aa, bb = pd.read_csv(a / 'rq3_predictions.csv'), pd.read_csv(b / 'rq3_predictions.csv')
            np.testing.assert_array_equal(aa.probability, bb.probability)
            self.assertFalse(aa.duplicated(['probe', 'method', 'participant']).any())
            with self.assertRaisesRegex(ValueError, 'disjoint'):
                evaluate(features, markers, pids, labels, ids[:8], ids[7:], ['Steps'], a, 17)


class CircularPhaseScaler(unittest.TestCase):
    def test_pairs_share_one_scale_so_circles_stay_circles(self):
        rng = np.random.default_rng(0)
        theta = rng.uniform(-.8, .8, 200)       # a narrow arc: cos and sin get unequal SDs
        other = rng.normal(3, 5, (200, 2))
        X = np.column_stack([other[:, 0], np.cos(theta), np.cos(2 * theta),
                             np.sin(theta), np.sin(2 * theta), other[:, 1]])
        scaler = IsotropicPairScaler(1, 2).fit(X)
        np.testing.assert_allclose(scaler.scale_[1:3], scaler.scale_[3:5])
        self.assertLess(np.std(np.cos(theta)), .5 * np.std(np.sin(theta)))
        Z = scaler.transform(X)
        # One uniform scale per pair: the scaled chord is the unit-circle chord, for every row.
        chord = np.hypot(Z[:, 1] - Z[0, 1], Z[:, 3] - Z[0, 3]) * scaler.scale_[1]
        np.testing.assert_allclose(chord, 2 * np.abs(np.sin((theta - theta[0]) / 2)), atol=1e-10)
        # A per-column StandardScaler shears the same pair into an ellipse.
        S = StandardScaler().fit_transform(X)
        sheared = np.hypot(S[:, 1] - S[0, 1], S[:, 3] - S[0, 3])
        ratio = sheared[1:] / np.maximum(chord[1:], 1e-12)
        self.assertGreater(ratio.max() / ratio.min(), 1.5)
        np.testing.assert_allclose(Z[:, [0, 5]], S[:, [0, 5]])
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            IsotropicPairScaler(4, 2).fit(X)

    def test_probe_uses_the_pair_scaler_only_when_given_a_block(self):
        self.assertIsInstance(feature_scaler(None), StandardScaler)
        self.assertIsInstance(logistic_probe(1.0, 0, (1, 2)).steps[1][1], IsotropicPairScaler)
        self.assertIsInstance(logistic_probe(1.0, 0).steps[1][1], StandardScaler)


class Disentanglement(unittest.TestCase):
    def test_window_targets_follow_the_cosinor_convention(self):
        t = np.arange(7 * 96)
        x = (3 + 2 * np.cos(2 * np.pi * (t - 24) / 96))[None, :, None]    # peak 6 h after the start
        w = window_rhythm(x, 96)
        np.testing.assert_allclose([w['MESOR'][0, 0], w['amplitude'][0, 0]], [3, 2], atol=1e-12)
        np.testing.assert_allclose([w['phase_cos'][0, 0], w['phase_sin'][0, 0]], [0, 1], atol=1e-12)

    def test_own_branch_beats_leakage_only_when_branches_separate(self):
        rng = np.random.default_rng(0)
        n = 240
        pids = np.repeat([f'p{i:02}' for i in range(24)], 10)
        angle = rng.uniform(-np.pi, np.pi, (n, 1))
        window = dict(MESOR=rng.normal(size=(n, 1)), amplitude=rng.gamma(2., size=(n, 1)),
                      phase_cos=np.cos(angle), phase_sin=np.sin(angle))
        noise = lambda k: rng.normal(size=(n, k))
        separated = np.hstack([window['MESOR'] + .1 * noise(1), noise(3),
                               window['amplitude'] + .1 * noise(1), noise(3),
                               window['phase_cos'], window['phase_sin'], noise(2)])
        blocks = {'trend': (0, 4), 'amplitude': (4, 8), 'phase': (8, 12)}
        features = {'dssl': separated.astype(np.float32), 'untrained': noise(12).astype(np.float32)}  # as encode()
        with tempfile.TemporaryDirectory() as directory:
            disentanglement(features, {m: (blocks, None) for m in features}, window, pids, pids,
                            np.unique(pids)[:8], ['Steps'], directory)
            frame = pd.read_csv(Path(directory) / 'rq1_disentanglement.csv').assign(seed=1)
        intervals, _ = disentanglement_intervals(frame, np.random.default_rng(1), 100)
        q = intervals.set_index(['method', 'target', 'quantity'])
        for target in ('MESOR', 'amplitude', 'acrophase'):
            self.assertGreater(q.loc[('dssl', target, 'own'), 'estimate'], .9)
            self.assertLess(q.loc[('dssl', target, 'leakage'), 'estimate'], .1)
            self.assertGreater(q.loc[('untrained', target, 'dssl_minus_method'), 'low'], 0)
        # The bootstrap's point estimate is the ordinary held-out R².
        g = frame[(frame.method == 'dssl') & (frame.target == 'MESOR') & (frame.role == 'own')]
        self.assertAlmostEqual(q.loc[('dssl', 'MESOR', 'own'), 'estimate'], r2_score(g.truth, g.prediction))


if __name__ == '__main__':
    unittest.main()
