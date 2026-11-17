import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def run_cli(self, *args):
        return subprocess.run([sys.executable, '-m', 'roundofflab', *map(str, args)],
                              capture_output=True, text=True, encoding='utf-8', timeout=45)

    def test_demo_and_analyze_round_trip(self):
        out = self.root / 'demo'
        r = self.run_cli('demo', '--out', out)
        self.assertEqual(r.returncode, 0, r.stderr)
        data = json.loads((out / 'report.json').read_text(encoding='utf-8'))
        self.assertTrue(data['ranking_reversals'])
        self.assertIn('synthetic', data['sample_label'])
        w, x = np.load(out / 'weights.npy'), np.load(out / 'inputs.npy')
        cfg = data['configurations'][0]
        scale = float(np.abs(w).max()) / 7
        q = np.clip(np.rint(w / scale), -7, 7) * scale
        self.assertAlmostEqual(cfg['metrics']['output_mse'], float(np.mean((x @ q.T - x @ w.T) ** 2)))
        self.assertTrue((out / 'report.html').read_text(encoding='utf-8').startswith('<!doctype html>'))
        out2 = self.root / 'user'
        r = self.run_cli('analyze', '--weights', out / 'weights.npy', '--inputs', out / 'inputs.npy', '--out', out2)
        self.assertEqual(r.returncode, 0, r.stderr)
        data2 = json.loads((out2 / 'report.json').read_text(encoding='utf-8'))
        self.assertEqual(data['configurations'], data2['configurations'])
        self.assertEqual(data2['sample_label'], 'user-provided')

    def test_refuse_overwrite_then_force(self):
        out = self.root / 'demo'
        self.assertEqual(self.run_cli('demo', '--out', out).returncode, 0)
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
        r = self.run_cli('demo', '--out', out, '--scenario', 'active-outlier')
        self.assertEqual(r.returncode, 2)
        self.assertIn('--force', r.stderr)
        self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()})
        r = self.run_cli('demo', '--out', out, '--scenario', 'active-outlier', '--force')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('active-outlier', json.loads((out / 'report.json').read_text())['sample_label'])

    def test_invalid_input_creates_no_report(self):
        w, x = self.root / 'w.npy', self.root / 'x.npy'
        np.save(x, np.ones((2, 3)))
        for value in (np.array([['private']], dtype=object), np.ones((2, 4)),
                      np.ones((2, 3), dtype=int), np.full((2, 3), np.inf)):
            np.save(w, value)
            r = self.run_cli('analyze', '--weights', w, '--inputs', x, '--out', self.root / 'bad')
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertNotIn('Traceback', r.stderr)
            self.assertFalse((self.root / 'bad' / 'report.json').exists())

    def test_config_validation_and_no_overwrite_of_config_input(self):
        w, x = self.root / 'w.npy', self.root / 'x.npy'
        np.save(w, np.ones((2, 3))); np.save(x, np.ones((2, 3)))
        config = self.root / 'report.json'
        for text in ('[{"unknown": 4}]', '[{"bits":true}]', '[{"bits":4,"bits":8}]', '[{"percentile":NaN}]'):
            config.write_text(text)
            r = self.run_cli('analyze', '--weights', w, '--inputs', x, '--configs', config, '--out', self.root / 'bad')
            self.assertEqual(r.returncode, 2, r.stderr)
        config.write_text('[{"bits":4}]')
        r = self.run_cli('analyze', '--weights', w, '--inputs', x, '--configs', config, '--out', self.root, '--force')
        self.assertEqual(r.returncode, 2)
        self.assertEqual(config.read_text(), '[{"bits":4}]')

    def test_empty_npy_has_actionable_error_without_traceback(self):
        w, x = self.root / 'empty.npy', self.root / 'x.npy'
        w.write_bytes(b''); np.save(x, np.ones((2, 3)))
        r = self.run_cli('analyze', '--weights', w, '--inputs', x, '--out', self.root / 'bad')
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertNotIn('Traceback', r.stderr)
        self.assertIn('empty or truncated', r.stderr)

    def test_missing_files_and_budget_fail_cleanly(self):
        r = self.run_cli('analyze', '--weights', self.root / 'none.npy', '--inputs', self.root / 'none2.npy', '--out', self.root / 'bad')
        self.assertEqual(r.returncode, 2)
        self.assertNotIn('Traceback', r.stderr)
        r = self.run_cli('demo', '--out', self.root / 'bad', '--max-elements', '1')
        self.assertEqual(r.returncode, 2)
        self.assertIn('limit', r.stderr)
        self.assertFalse((self.root / 'bad').exists())


if __name__ == '__main__':
    unittest.main()
