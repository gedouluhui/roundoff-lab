import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class OptionalIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('torch'), 'optional actual PyTorch export requires torch; no install is performed')
    def test_real_quantized_tensor_export_and_missing_zero_point(self):
        script=Path(__file__).resolve().parents[1]/'examples/pytorch_export.py'
        with tempfile.TemporaryDirectory() as tmp:
            result=subprocess.run([sys.executable,str(script),'--out',tmp],capture_output=True,text=True,timeout=90)
            self.assertEqual(result.returncode,0,result.stderr)
            root=Path(tmp);data=json.loads((root/'report.json').read_text())
            for view in data['cohorts'].values():
                entries={c['id']:c for c in view['configurations']}
                self.assertEqual(entries['external:dequantized']['metrics']['propagated_mse'],0)
                self.assertEqual(entries['external:omitted-zero-point']['metrics']['propagated_mse'],65536)
            metadata=json.loads((root/'integration.json').read_text())
            self.assertEqual(metadata['integer_codes'],[[0,3]])
            self.assertEqual(metadata['dequantized'],[[-256.0,-250.0]])
            self.assertTrue(metadata['torch_version'])
            self.assertEqual(metadata['device'],'cpu')


if __name__=='__main__':unittest.main()
