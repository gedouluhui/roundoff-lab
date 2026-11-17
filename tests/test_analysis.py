import json
import unittest
import numpy as np
from roundofflab.analysis import analyze
from roundofflab.demo import make_demo
from roundofflab.quantize import Config, quantize


class AnalysisTests(unittest.TestCase):
    def test_metrics_from_independent_scalar_output(self):
        w = np.array([[7., .5, 1.5], [-7., -2.5, 2.5]])
        x = np.array([[1., 2., 3.], [-2., 0., 1.]])
        # Known rounded weights, not a call to the production quantizer.
        qw = [[7., 0., 2.], [-7., -2., 2.]]
        expected_y = np.array([[sum(float(x[s, i]) * float(w[o, i]) for i in range(3))
                                for o in range(2)] for s in range(2)])
        quant_y = np.array([[sum(float(x[s, i]) * qw[o][i] for i in range(3))
                             for o in range(2)] for s in range(2)])
        report = analyze(w, x, [Config(mode='tensor')])
        entry = report['configurations'][0]
        self.assertAlmostEqual(entry['metrics']['output_mse'], float(np.mean((quant_y - expected_y) ** 2)))
        # Four half-unit errors and two exact endpoints: (4 * .5**2) / 6.
        self.assertAlmostEqual(entry['metrics']['weight_mse'], 1 / 6)
        self.assertEqual(report['shapes']['output'], [2, 2])
        self.assertEqual(entry['metrics']['scale_count'], 1)
        self.assertEqual(report['schema_version'], 1)
        json.dumps(report, allow_nan=False)

    def test_ranking_reversal_and_isotropic_control(self):
        w = np.array([[9., 1., 2., 0.]])
        configs = [Config(), Config(percentile=75)]
        report = analyze(w, np.array([[0., 1., 0., 0.]]), configs)
        self.assertEqual(report['ranking_reversals'], [
            {'lower_weight_error': 'int4-channel', 'lower_output_error': 'int4-channel-p75'}])
        report = analyze(w, np.eye(4), configs)
        self.assertEqual(report['ranking_reversals'], [])
        for entry in report['configurations']:
            self.assertAlmostEqual(entry['metrics']['weight_mse'], entry['metrics']['output_mse'])

    def test_null_relative_error_and_zero_errors(self):
        for x in (np.zeros((2, 3)), np.eye(3)):
            r = analyze(np.zeros((1, 3)), x, [Config()])['configurations'][0]
            self.assertEqual(r['metrics']['output_mse'], 0)
            self.assertIsNone(r['metrics']['relative_output_l2'])

    def test_array_hashes_and_input_immutability(self):
        w, x = make_demo()
        before = (w.copy(), x.copy())
        first = analyze(w, x)
        second = analyze(np.asfortranarray(w), x.copy())
        self.assertEqual(first['provenance'], second['provenance'])
        np.testing.assert_array_equal(w, before[0])
        np.testing.assert_array_equal(x, before[1])
        self.assertNotIn('weights', first['configurations'][0])
        x[0, 0] += 1
        self.assertNotEqual(first['provenance']['input_sha256'], analyze(w, x)['provenance']['input_sha256'])

    def test_actual_demo_scenarios_and_reversals(self):
        w, x = make_demo()
        w2, x2 = make_demo(scenario='active-outlier')
        np.testing.assert_array_equal(w, w2)
        self.assertFalse(np.array_equal(x, x2))
        quiet, active = analyze(w, x), analyze(w2, x2)
        self.assertTrue(quiet['ranking_reversals'])
        self.assertNotEqual(quiet['configurations'], active['configurations'])
        with self.assertRaises(ValueError):
            make_demo(scenario='typo')

    def test_aggregated_heatmap_covers_every_element(self):
        w = np.random.default_rng(9).normal(size=(25, 65))
        result = analyze(w, np.eye(65), [Config()])['configurations'][0]
        heat = result['heatmap']
        self.assertLessEqual(len(heat['values']), 24)
        self.assertLessEqual(len(heat['values'][0]), 64)
        error = np.abs(quantize(w, Config()).weights - w)
        covered = np.zeros(w.shape, dtype=int)
        for ri, (a, b) in enumerate(heat['row_bounds']):
            for ci, (c, d) in enumerate(heat['col_bounds']):
                self.assertAlmostEqual(heat['values'][ri][ci], float(error[a:b, c:d].mean()))
                covered[a:b, c:d] += 1
        np.testing.assert_array_equal(covered, np.ones(w.shape))

    def test_nonzero_subnormal_reference_is_not_called_zero(self):
        w = np.array([[7., .5]])
        x = np.array([[0., 1e-323], [0., 0.], [0., 0.], [0., 0.]])
        with self.assertRaisesRegex(ValueError, 'underflow'):
            analyze(w, x, [Config()])

    def test_invalid_shapes_budgets_and_configurations(self):
        w, x = np.ones((2, 3)), np.ones((4, 3))
        for kwargs in ({'max_elements': 1}, {'max_output_elements': 1},
                       {'max_elements': True}, {'max_elements': -1},
                       {'configs': []}, {'configs': [Config(), Config()]},
                       {'configs': ['not a config']}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                analyze(w, x, **kwargs)
        with self.assertRaises(ValueError):
            analyze(w, np.ones((4, 5)))
        with self.assertRaises(ValueError):
            analyze(np.full((2, 3), 1e308), np.full((4, 3), 1e308))


if __name__ == '__main__':
    unittest.main()
