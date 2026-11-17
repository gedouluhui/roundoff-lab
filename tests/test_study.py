import json
import unittest
from unittest.mock import patch
from importlib import import_module

study_module = import_module('roundofflab.study')
import numpy as np
from roundofflab.study import study
from roundofflab.quantize import Config


def fixture():
    w=np.array([[1.,1.],[2.,2.]])
    a=w+np.array([[.1,0.],[.2,0.]])
    b=w+np.array([[0.,.3],[0.,.6]])
    return w,np.array([[0.,1.],[0.,2.]]),np.array([[1.,0.],[2.,0.]]),{'a':a,'b':b}


class StudyTests(unittest.TestCase):
    def test_discovery_selects_once_validation_cannot_reselect(self):
        w,d,v,c=fixture();r=study(w,d,v,candidates=c)
        self.assertEqual(r['schema_version'],2)
        self.assertEqual(r['selection']['candidate_id'],'external:a')
        self.assertEqual(r['selection']['validation_best_ids'],['external:b'])
        self.assertGreater(r['selection']['excess_validation_metric'],0)
        r2=study(w,d,v*9,candidates=c)
        self.assertEqual(r2['selection']['candidate_id'],r['selection']['candidate_id'])
        self.assertNotEqual(r2['selection']['validation_value'],r['selection']['validation_value'])
        self.assertEqual(r['selection']['metric'],'propagated_mse')
        json.dumps(r,allow_nan=False)

    def test_external_candidate_is_not_requantized_and_unknown_counts_are_null(self):
        w,d,v,c=fixture()
        with patch.object(study_module, 'quantize',side_effect=AssertionError('external candidate must not be quantized')):
            r=study(w,d,v,candidates=c)
        entry=r['cohorts']['validation']['configurations'][0]
        expected=(v@(c['a']-w).T)
        self.assertAlmostEqual(entry['metrics']['propagated_mse'],float(np.mean(expected**2)))
        self.assertEqual(entry['origin'],'external_dequantized')
        for key in ('clipped_count','scale_count'):
            self.assertIsNone(entry['metrics'][key])
        self.assertIsNone(entry['bits'])
        self.assertEqual(len(entry['candidate_sha256']),64)

    def test_explicit_selection_and_tie_rule(self):
        w,d,v,c=fixture()
        r=study(w,d,v,candidates=c,selected='external:b')
        self.assertEqual(r['selection']['method'],'user_declared')
        self.assertEqual(r['selection']['candidate_id'],'external:b')
        r=study(w,d,v,candidates={'z':w.copy(),'a':w.copy()})
        self.assertEqual(r['selection']['candidate_id'],'external:a')
        self.assertEqual(r['selection']['validation_best_ids'],['external:a','external:z'])
        with self.assertRaises(ValueError):study(w,d,v,candidates=c,selected='absent')

    def test_each_simulation_quantized_once_not_once_per_cohort(self):
        module=study_module
        w,d,v,c=fixture()
        real=module.quantize
        with patch.object(module,'quantize',wraps=real) as wrapped:
            r=study(w,d,v,configs=[Config(),Config(bits=8)],candidates=c)
            self.assertEqual(wrapped.call_count,2)
        self.assertEqual(len(r['cohorts']['discovery']['configurations']),4)

    def test_numeric_cohort_identity_warning_and_independence_disclaimer(self):
        w,d,v,c=fixture()
        r=study(w,d.astype(np.float32),d.astype(np.float64),candidates=c)
        self.assertTrue(r['cohort_checks']['identical_values'])
        self.assertFalse(r['cohort_checks']['independence_verified'])
        self.assertTrue(any('identical' in m.lower() for m in r['warnings']))
        other=study(w,d,v,candidates=c)
        self.assertFalse(other['cohort_checks']['identical_values'])
        self.assertFalse(other['cohort_checks']['independence_verified'])

    def test_input_immutability_and_budget_before_quantization(self):
        w,d,v,c=fixture();old=[a.copy() for a in [w,d,v,*c.values()]]
        study(w,d,v,candidates=c)
        for actual,previous in zip([w,d,v,*c.values()],old):np.testing.assert_array_equal(actual,previous)
        with patch.object(study_module, 'quantize',side_effect=AssertionError('budget must run first')):
            with self.assertRaisesRegex(ValueError,'element'):
                study(w,d,v,candidates=c,configs=[Config()],max_elements=1)
        with self.assertRaisesRegex(ValueError,'output'):
            study(w,d,v,candidates=c,max_output_elements=1)

    def test_batch_size_cannot_change_selection_under_cancellation(self):
        w=np.zeros((1,4));d=np.ones((2,4));v=np.ones((1,4))
        candidates={'a':np.array([[1e16,1.,-1e16,1.]]),'b':np.array([[1.5,0.,0.,0.]])}
        reports=[study(w,np.array(d,dtype=dtype,order=order),v,candidates=candidates,batch_rows=b)
                 for dtype in (np.float32,np.float64) for order in ('C','F') for b in (1,2,256)]
        self.assertEqual(len({r['selection']['candidate_id'] for r in reports}),1)
        for result in reports:
            self.assertEqual(result['selection'],reports[0]['selection'])

    def test_external_provenance_does_not_claim_unknown_quantization(self):
        w,d,v,c=fixture();r=study(w,d,v,candidates=c)
        policy=r['cohorts']['discovery']['provenance']['numerical_policy']
        self.assertNotIn('ties-to-even',policy)
        self.assertNotIn('percentile',policy)
        self.assertIn('float64',policy)
        self.assertIsNone(r['cohorts']['discovery']['configurations'][0]['quantization_policy'])

    def test_bad_candidates_configs_and_data(self):
        w,d,v,c=fixture()
        for external in ({'bad/name':w},{'a':np.ones((3,2))},{'a':w.astype(int)},
                         {'a':np.full_like(w,np.nan)},{str(i):w for i in range(25)}):
            with self.assertRaises(ValueError):study(w,d,v,candidates=external)
        for kw in ({'configs':[]},{'configs':[Config(),Config()]},{'batch_rows':True},
                   {'selection_metric':'accuracy'},{'max_elements':True}):
            with self.assertRaises(ValueError):study(w,d,v,**kw)
        with self.assertRaises(ValueError):study(w,d,np.ones((2,3)),candidates=c)


if __name__=='__main__':unittest.main()
