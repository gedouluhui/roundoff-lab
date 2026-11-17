import json
import re
import shutil
import subprocess
import unittest
from roundofflab.analysis import analyze
from roundofflab.demo import make_demo
from roundofflab.report import render_report


class ReportTests(unittest.TestCase):
    def test_real_report_round_trip_and_static_table(self):
        report = analyze(*make_demo())
        html = render_report(report)
        match = re.search(r'<script id="report-data" type="application/json">(.*?)</script>', html, re.S)
        self.assertIsNotNone(match)
        self.assertEqual(json.loads(match.group(1)), report)
        for cfg in report['configurations']:
            self.assertIn(cfg['id'], html.split('<script id="report-data"')[0])
        self.assertIn('Weight MSE', html)
        self.assertIn('Output MSE', html)
        self.assertIn('id="config-select"', html)
        self.assertIn('id="download-json"', html)
        self.assertNotRegex(html, r'<script[^>]+src=|<link[^>]+href=|@import|fetch\(')

    def test_script_injection_is_inert_in_text_and_json(self):
        title = '</script><img src=x onerror=alert(1)> & \u2028'
        report = analyze(*make_demo(), title=title, sample_label=title)
        html = render_report(report)
        self.assertNotIn('</script><img', html)
        self.assertNotIn('<img src=x', html)
        match = re.search(r'<script id="report-data" type="application/json">(.*?)</script>', html, re.S)
        self.assertEqual(json.loads(match.group(1))['title'], title)
        self.assertNotIn('<', match.group(1))

    def test_dense_heatmap_has_large_controls_and_one_keyboard_entry(self):
        html = render_report(analyze(*make_demo()))
        self.assertIn('id="row-select"', html)
        self.assertIn('id="col-select"', html)
        self.assertIn("b.tabIndex", html)
        self.assertIn('ArrowRight', html)

    @unittest.skipUnless(shutil.which('node'), 'optional JavaScript arithmetic check requires Node.js')
    def test_large_finite_channel_ratios_in_actual_javascript_expression(self):
        from roundofflab.report import SCRIPT
        expression = re.search(r'bar\.style\.width=`\$\{([^}]+)\}%', SCRIPT)
        self.assertIsNotNone(expression, 'update this test when the production width expression moves')
        code = 'const largest=2.5e307; console.log(JSON.stringify([2.5e307,6.25e306].map(n=>{const v={output_mse:n};return (' + expression.group(1) + ')})))'
        result = subprocess.run([shutil.which('node'), '-e', code], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [100, 25])

    def test_no_javascript_retains_comparison_and_scope(self):
        html = render_report(analyze(*make_demo()))
        self.assertIn('<noscript>', html)
        self.assertIn('<table', html)
        self.assertIn('not a model-accuracy', html)
        self.assertIn('prefers-reduced-motion', html)
        self.assertIn(':focus-visible', html)


if __name__ == '__main__':
    unittest.main()
