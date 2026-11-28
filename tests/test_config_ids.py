import math
from fractions import Fraction
import unittest

import numpy as np

from roundofflab.analysis import analyze
from roundofflab.quantize import Config, default_configs


class ConfigIdTests(unittest.TestCase):
    def test_percentile_ids_do_not_collide(self):
        a = Config(bits=4, mode='group', group_size=64, percentile=99.1234567)
        b = Config(bits=4, mode='group', group_size=64, percentile=99.1234568)
        self.assertNotEqual(a.id, b.id)

    def test_released_ids_remain_stable(self):
        self.assertEqual([config.id for config in default_configs()], [
            'int4-tensor', 'int4-channel', 'int4-group-g32', 'int8-channel',
            'int4-channel-p99', 'int4-channel-p95',
        ])
        self.assertEqual(Config(mode='group', group_size=3, percentile=95).id,
                         'int4-group-g3-p95')
        self.assertEqual(Config(percentile=75).id, 'int4-channel-p75')

    def test_roundtrippable_percentiles_keep_legacy_ids(self):
        self.assertEqual(Config(percentile=99.1234).id, 'int4-channel-p99.1234')
        self.assertEqual(Config(percentile=1e-7).id, 'int4-channel-p1e-07')

    def test_lossless_percentiles_use_hexadecimal_ids(self):
        self.assertEqual(Config(percentile=99.1234567).id,
                         'int4-channel-p0x1.8c7e6b6ee3e37p+6')
        self.assertEqual(Config(percentile=99.1234568).id,
                         'int4-channel-p0x1.8c7e6b7599e01p+6')

    def test_percentile_rounded_to_100_does_not_collide_with_default(self):
        self.assertEqual(Config(percentile=99.9999999).id,
                         'int4-channel-p0x1.8ffffff94a036p+6')
        self.assertNotEqual(Config(percentile=99.9999999).id, Config().id)

    def test_adjacent_float_percentiles_have_distinct_ids(self):
        for value in (1e-7, 1.0, 50.0, 99.1234567, 100.0):
            adjacent = math.nextafter(value, 0.0)
            with self.subTest(value=value, adjacent=adjacent):
                self.assertNotEqual(Config(percentile=value).id,
                                    Config(percentile=adjacent).id)

    def test_analysis_preserves_close_percentile_configurations(self):
        configs = [Config(percentile=99.1234567), Config(percentile=99.1234568)]
        report = analyze(np.array([[1., 2., 3., 4.]]), np.array([[1., 1., 1., 1.]]), configs)
        self.assertEqual([entry['id'] for entry in report['configurations']], [
            'int4-channel-p0x1.8c7e6b6ee3e37p+6',
            'int4-channel-p0x1.8c7e6b7599e01p+6',
        ])
        self.assertEqual([entry['percentile'] for entry in report['configurations']],
                         [99.1234567, 99.1234568])

    def test_real_percentiles_normalize_to_float64(self):
        exact = Fraction(99, 1)
        rounded = Fraction(2 ** 53 + 1, 2 ** 53) * 50
        self.assertNotEqual(rounded, float(rounded))
        self.assertEqual(Config(percentile=rounded), Config(percentile=float(rounded)))
        self.assertIs(type(Config(percentile=exact).percentile), float)
        self.assertIs(type(Config(percentile=np.float32(95)).percentile), float)

    def test_analysis_still_rejects_duplicate_configurations(self):
        with self.assertRaisesRegex(ValueError, 'configuration IDs must be unique'):
            analyze(np.ones((1, 2)), np.ones((1, 2)), [Config(), Config()])


if __name__ == '__main__':
    unittest.main()
