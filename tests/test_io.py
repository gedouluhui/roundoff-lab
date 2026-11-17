import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from roundofflab.io import load_npy, write_artifacts


class OutputTests(unittest.TestCase):
    def test_empty_npy_normalizes_numpy_124_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / 'empty.npy'; empty.write_bytes(b'')
            # Actual NumPy 1.24 behavior observed in the minimum-version CI job.
            with patch('roundofflab.io.np.load', side_effect=ValueError('Cannot load file containing pickled data when allow_pickle=False')):
                with self.assertRaisesRegex(ValueError, 'empty or truncated'):
                    load_npy(empty)

    def test_failed_force_restores_entire_previous_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            old = {'report.json': b'old-json', 'report.html': b'old-html', 'inputs.npy': b'old-input'}
            new = {'report.json': b'new-json', 'report.html': b'new-html', 'inputs.npy': b'new-input'}
            for name, content in old.items():
                (out / name).write_bytes(content)
            replace = os.replace
            def locked(source, destination):
                if Path(source).name == 'report.html' or Path(destination).name == 'report.html':
                    raise PermissionError('simulated external reader lock')
                return replace(source, destination)
            with patch('roundofflab.io.os.replace', side_effect=locked):
                with self.assertRaises(OSError):
                    write_artifacts(out, new, force=True)
            self.assertEqual({p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}, old)

    def test_publication_failure_rolls_back_all_new_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            old = {'report.json': b'old-json', 'report.html': b'old-html'}
            for name, content in old.items():
                (out / name).write_bytes(content)
            link = os.link
            def fail_second(source, target):
                if Path(target).name == 'report.html':
                    raise OSError('simulated publication failure')
                return link(source, target)
            with patch('roundofflab.io.os.link', side_effect=fail_second):
                with self.assertRaises(OSError):
                    write_artifacts(out, {k: b'new' for k in old}, force=True)
            self.assertEqual({p.name: p.read_bytes() for p in out.iterdir()}, old)

    def test_failed_rollback_preserves_backup_for_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / 'report.json').write_bytes(b'previous report')
            replace = os.replace
            def fail_restore(source, target):
                if Path(source).parent.name == 'old':
                    raise PermissionError('simulated restore lock')
                return replace(source, target)
            with patch('roundofflab.io.os.link', side_effect=OSError('disk error')):
                with patch('roundofflab.io.os.replace', side_effect=fail_restore):
                    with self.assertRaisesRegex(OSError, 'rollback incomplete'):
                        write_artifacts(out, {'report.json': b'new'}, force=True)
            backups = list(out.glob('.roundoff-stage-*/old/report.json'))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), b'previous report')
            self.assertFalse((out / 'report.json').exists())

    def test_cleanup_failure_does_not_reclassify_committed_publication(self):
        import warnings
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with patch('roundofflab.io.shutil.rmtree', side_effect=PermissionError('temporary file still open')):
                with warnings.catch_warnings(record=True) as notices:
                    warnings.simplefilter('always')
                    write_artifacts(out, {'report.json': b'committed'})
            self.assertEqual((out / 'report.json').read_bytes(), b'committed')
            self.assertTrue(any('cleanup' in str(w.message) for w in notices))
            self.assertEqual(len(list(out.glob('.roundoff-stage-*'))), 1)

    def test_alias_input_remains_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            source = out / 'input.npy'; source.write_bytes(b'protected')
            os.link(source, out / 'report.json')
            with self.assertRaisesRegex(ValueError, 'aliases an input'):
                write_artifacts(out, {'report.json': b'overwrite'}, force=True, inputs=(source,))
            self.assertEqual(source.read_bytes(), b'protected')


if __name__ == '__main__':
    unittest.main()
