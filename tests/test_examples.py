import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET


class SiteExampleTests(unittest.TestCase):
    def test_site_is_generated_from_real_analysis(self):
        script = Path(__file__).resolve().parents[1] / 'examples' / 'build_site.py'
        with tempfile.TemporaryDirectory() as tmp:
            run = subprocess.run([sys.executable, str(script), '--out', tmp],
                                 text=True, capture_output=True, timeout=45)
            self.assertEqual(run.returncode, 0, run.stderr)
            root = Path(tmp)
            quiet = json.loads((root / 'quiet.json').read_text())
            active = json.loads((root / 'active.json').read_text())
            self.assertEqual(quiet['provenance']['weight_sha256'], active['provenance']['weight_sha256'])
            self.assertNotEqual(quiet['provenance']['input_sha256'], active['provenance']['input_sha256'])
            self.assertIn('active.html', (root / 'index.html').read_text(encoding='utf-8'))
            svg = ET.parse(root / 'preview.svg').getroot()
            text = ' '.join(svg.itertext())
            cfg = next(c for c in quiet['configurations'] if c['id'] == 'int4-channel-p95')
            self.assertIn(format(cfg['metrics']['output_mse'], '.4g'), text)
            self.assertIn('synthetic', text.lower())


if __name__ == '__main__':
    unittest.main()
