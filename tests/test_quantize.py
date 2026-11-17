import unittest
import numpy as np
from roundofflab.quantize import Config, quantize


class QuantizationTests(unittest.TestCase):
    def test_narrow_range_ties_to_even(self):
        w = np.array([[-7., -2.5, -1.5, -.5, .5, 1.5, 2.5, 7.]])
        q = quantize(w, Config(mode='tensor'))
        np.testing.assert_array_equal(q.weights, [[-7, -2, -2, 0, 0, 2, 2, 7]])
        self.assertEqual(q.clipped_count, 0)
        self.assertEqual(q.scale_count, 1)

    def test_scalar_oracle_all_modes(self):
        w = np.random.default_rng(3).normal(size=(3, 7))
        for mode in ('tensor', 'channel', 'group'):
            cfg = Config(bits=3, mode=mode, group_size=3)
            expected = np.zeros_like(w)
            units = [(range(3), range(7))] if mode == 'tensor' else [
                ([r], range(start, min(7, start + (3 if mode == 'group' else 7))))
                for r in range(3) for start in range(0, 7, 3 if mode == 'group' else 7)]
            for rows, cols in units:
                coords = [(r, c) for r in rows for c in cols]
                maximum = max(abs(float(w[r, c])) for r, c in coords)
                scale = maximum / 3 if maximum else 1
                for r, c in coords:
                    expected[r, c] = max(-3, min(3, round(float(w[r, c]) / scale))) * scale
            np.testing.assert_allclose(quantize(w, cfg).weights, expected, rtol=1e-14, atol=1e-14)

    def test_zero_units_and_zero_threshold(self):
        q = quantize(np.zeros((2, 5)), Config(mode='group', group_size=3))
        np.testing.assert_array_equal(q.weights, np.zeros((2, 5)))
        np.testing.assert_array_equal(q.scales, np.ones(4))
        q = quantize(np.array([[0., 0., 0., 8.]]), Config(percentile=50))
        np.testing.assert_array_equal(q.weights, [[0, 0, 0, 0]])
        self.assertEqual(q.clipped_count, 1)

    def test_tail_group_and_no_mutation(self):
        w = np.array([[7., 2., 5.]], dtype=np.float32)
        before = w.copy()
        q = quantize(w, Config(mode='group', group_size=2))
        self.assertEqual(q.scale_count, 2)
        self.assertEqual(q.weights[0, 2], 5)
        np.testing.assert_array_equal(w, before)
        self.assertFalse(np.shares_memory(q.weights, w))

    def test_clip_percentile_is_linear(self):
        q = quantize(np.array([[0., 1., 2., 9.]]), Config(mode='tensor', percentile=75))
        self.assertAlmostEqual(q.scales[0], 3.75 / 7)
        self.assertEqual(q.clipped_count, 1)
        self.assertEqual(q.weights[0, -1], 3.75)

    def test_invalid_configs_and_inputs(self):
        for kw in ({'bits': True}, {'bits': 4.5}, {'bits': 1}, {'bits': 9},
                   {'mode': 'rows'}, {'group_size': 0}, {'group_size': True},
                   {'percentile': float('nan')}, {'percentile': 0}, {'percentile': 101}, {'percentile': True}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                Config(**kw)
        for w in (np.array([1.]), np.empty((0, 2)), np.array([[1]]),
                  np.array([[True]]), np.array([[complex(1, 2)]]),
                  np.array([[float('nan')]]), np.array([[float('inf')]])):
            with self.subTest(dtype=w.dtype), self.assertRaises(ValueError):
                quantize(w, Config())

    def test_scale_underflow_rejected(self):
        with self.assertRaisesRegex(ValueError, 'underflow'):
            quantize(np.array([[np.nextafter(0., 1.)]]), Config())

    def test_config_identity(self):
        self.assertEqual(Config().id, 'int4-channel')
        self.assertEqual(Config(mode='group', group_size=3, percentile=95).id, 'int4-group-g3-p95')


if __name__ == '__main__':
    unittest.main()
