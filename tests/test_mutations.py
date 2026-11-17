"""Prove selected regressions are detected. Mutate only disposable source copies."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class MutationTests(unittest.TestCase):
    def check_mutation(self, filename, old, new, module, test_name):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='roundoff-mutation-') as tmp:
            source = Path(tmp) / 'src'
            shutil.copytree(root / 'src', source, ignore=shutil.ignore_patterns('__pycache__', '*.egg-info'))
            env = dict(os.environ, PYTHONPATH=str(source))
            cmd = [sys.executable, '-m', 'unittest', 'discover', '-s', str(root / 'tests'),
                   '-p', module, '-k', test_name, '-v']
            def run():
                return subprocess.run(cmd, cwd=tmp, env=env, text=True, capture_output=True, timeout=60)
            baseline = run()
            self.assertEqual(baseline.returncode, 0, baseline.stderr)
            target = source / 'roundofflab' / filename
            content = target.read_text(encoding='utf-8')
            self.assertEqual(content.count(old), 1, 'mutation anchor must be unique')
            changed = content.replace(old, new, 1)
            self.assertNotEqual(content, changed, 'no-op mutation is not evidence')
            target.write_text(changed, encoding='utf-8')
            # Do not let timestamp-based pyc caching disguise a same-size mutation.
            for cache in source.rglob('__pycache__'):
                shutil.rmtree(cache)
            result = run()
            self.assertNotEqual(result.returncode, 0, 'known-bad behavior escaped its test')
            self.assertIn('FAIL: ' + test_name, result.stderr,
                          'a crash/import error is not a named assertion catching the mutation')
            self.assertNotIn('FAILED (errors=', result.stderr)

    def test_truncation_instead_of_even_rounding_is_caught(self):
        self.check_mutation('quantize.py', 'codes = np.rint(', 'codes = np.trunc(',
                            'test_quantize.py', 'test_narrow_range_ties_to_even')

    def test_wrong_error_ranking_direction_is_caught(self):
        self.check_mutation('analysis.py',
                            'a["metrics"]["output_mse"] > b["metrics"]["output_mse"]',
                            'a["metrics"]["output_mse"] < b["metrics"]["output_mse"]',
                            'test_analysis.py', 'test_ranking_reversal_and_isotropic_control')

    def test_disabled_subnormal_guard_is_caught(self):
        self.check_mutation('analysis.py', 'if np.any((peak != 0) & (result == 0)):',
                            'if False:', 'test_analysis.py', 'test_nonzero_subnormal_reference_is_not_called_zero')

    def test_constant_fingerprint_is_caught(self):
        self.check_mutation('analysis.py', 'return digest.hexdigest()', "return '0' * 64",
                            'test_analysis.py', 'test_array_hashes_and_input_immutability')

    def test_missing_empty_file_boundary_is_caught(self):
        self.check_mutation('io.py', 'except (EOFError, ValueError) as exc:', 'except IndexError as exc:',
                            'test_cli.py', 'test_empty_npy_has_actionable_error_without_traceback')

    def test_ignored_tail_group_is_caught(self):
        self.check_mutation('quantize.py',
                            'result[unit] = np.clip(codes, -qmax, qmax) * scale',
                            'result[unit] = 0 if config.mode == "group" and block.shape[1] < config.group_size else np.clip(codes, -qmax, qmax) * scale',
                            'test_quantize.py', 'test_tail_group_and_no_mutation')


if __name__ == '__main__':
    unittest.main()
