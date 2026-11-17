import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np


class StudyCliTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        data=self.root/'data';data.mkdir()
        w=np.array([[1.,1.],[2.,2.]])
        for name,a in {'w':w,'a':w+[[.1,0.],[.2,0.]],'b':w+[[0.,.3],[0.,.6]],
                       'd':np.array([[0.,1.],[0.,2.]]),'v':np.array([[1.,0.],[2.,0.]])}.items():
            np.save(data/(name+'.npy'),a)
        self.manifest=data/'candidates.json';self.manifest.write_text('{"a":"a.npy","b":"b.npy"}')

    def tearDown(self):self.temp.cleanup()

    def run_cli(self,*extras):
        cmd=[sys.executable,'-m','roundofflab','study','--weights',str(self.root/'data/w.npy'),
             '--discovery',str(self.root/'data/d.npy'),'--validation',str(self.root/'data/v.npy'),
             '--candidates',str(self.manifest),'--out',str(self.root/'result'),*map(str,extras)]
        return subprocess.run(cmd,text=True,capture_output=True,encoding='utf-8',timeout=45)

    def test_actual_manifest_relative_study(self):
        r=self.run_cli('--batch-rows','1');self.assertEqual(r.returncode,0,r.stderr)
        report=json.loads((self.root/'result/report.json').read_text(encoding='utf-8'))
        self.assertEqual(report['selection']['candidate_id'],'external:a')
        self.assertEqual(report['selection']['validation_best_ids'],['external:b'])
        self.assertIn('External', (self.root/'result/report.html').read_text(encoding='utf-8'))
        self.assertNotIn(str(self.root),json.dumps(report))

    def test_selected_id_is_honored(self):
        r=self.run_cli('--selected','external:b','--selection-metric','output_mse')
        self.assertEqual(r.returncode,0,r.stderr)
        report=json.loads((self.root/'result/report.json').read_text())
        self.assertEqual(report['selection']['candidate_id'],'external:b')
        self.assertEqual(report['selection']['metric'],'output_mse')

    def test_bad_manifest_never_creates_output(self):
        for text in ('{}','[]','{"a":1}','{"a":"a.npy","a":"b.npy"}', '{"a":"missing.npy"}'):
            self.manifest.write_text(text)
            r=self.run_cli();self.assertEqual(r.returncode,2,r.stderr)
            self.assertNotIn('Traceback',r.stderr)
            self.assertFalse((self.root/'result/report.json').exists())

    def test_candidate_manifest_protected_from_output_alias(self):
        result=self.root/'result';result.mkdir()
        self.manifest=result/'report.json'
        content=json.dumps({'a':str(self.root/'data/a.npy')})
        self.manifest.write_text(content)
        r=self.run_cli('--force');self.assertEqual(r.returncode,2,r.stderr)
        self.assertIn('aliases an input',r.stderr)
        self.assertEqual(self.manifest.read_text(),content)


if __name__=='__main__':unittest.main()
