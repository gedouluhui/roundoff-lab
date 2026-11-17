import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET


class SiteExampleTests(unittest.TestCase):
    def test_study_example_is_reproducible_from_manifest(self):
        script = Path(__file__).resolve().parents[1] / 'examples' / 'build_study.py'
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, str(script), '--out', tmp],text=True,capture_output=True,timeout=45)
            self.assertEqual(result.returncode,0,result.stderr)
            root=Path(tmp);data=json.loads((root/'report.json').read_text(encoding='utf-8'))
            self.assertEqual(data['selection']['candidate_id'],'external:p95-export')
            self.assertEqual(data['selection']['validation_best_ids'],['external:absmax-export'])
            self.assertIn('synthetic',data['cohorts']['discovery']['sample_label'])
            cli=subprocess.run([sys.executable,'-m','roundofflab','study','--weights',str(root/'weights.npy'),
                '--discovery',str(root/'discovery.npy'),'--validation',str(root/'validation.npy'),
                '--candidates',str(root/'candidates.json'),'--out',str(root/'rerun')],text=True,capture_output=True,timeout=45)
            self.assertEqual(cli.returncode,0,cli.stderr)
            rerun=json.loads((root/'rerun/report.json').read_text())
            self.assertEqual(data['selection'],rerun['selection'])
            self.assertTrue((root/'index.html').exists())

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
