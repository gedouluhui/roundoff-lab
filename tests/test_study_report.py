import json
import re
import unittest
import numpy as np
from roundofflab.study import study
from roundofflab.report import render_report


class StudyReportTests(unittest.TestCase):
    def make_study(self,**kw):
        w=np.array([[1.,1.],[2.,2.]])
        return study(w,np.array([[0.,1.]]),np.array([[1.,0.]]),
                     candidates={'a':w+[[.1,0.],[.2,0.]],'b':w+[[0.,.3],[0.,.6]]},**kw)

    def test_complete_payload_and_static_selection_comparison(self):
        report=self.make_study();page=render_report(report)
        payload=re.search(r'<script id="report-data" type="application/json">(.*?)</script>',page,re.S)
        self.assertEqual(json.loads(payload.group(1)),report)
        static=page.split('<script id="report-data"')[0]
        for text in ('Discovery','Validation','External','external:a','external:b','id="cohort-select"',
                     'id="fixed-selection"','propagated_mse','unknown','worst-samples'):
            self.assertTrue(text in static,text)
        self.assertIn('independence',static)
        self.assertIn('forward',static)

    def test_fingerprint_has_explicit_word_break_rule(self):
        page=render_report(self.make_study())
        self.assertTrue('#candidate-origin{overflow-wrap:anywhere}' in page)

    def test_injection_is_inert_in_bundle_and_static_heading(self):
        title='</script><img src=x onerror=alert(1)>'
        page=render_report(self.make_study(title=title))
        self.assertNotIn('<img src=x',page)
        payload=re.search(r'<script id="report-data" type="application/json">(.*?)</script>',page,re.S)
        self.assertEqual(json.loads(payload.group(1))['title'],title)


if __name__=='__main__':unittest.main()
