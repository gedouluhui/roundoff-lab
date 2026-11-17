import unittest
import numpy as np
from roundofflab.study_metrics import evaluate


class BatchedMetricsTests(unittest.TestCase):
    def test_scalar_oracle_and_batch_equivalence(self):
        rng = np.random.default_rng(19)
        w = rng.normal(size=(3, 7)); q = w + rng.normal(size=w.shape) * .2
        x = rng.normal(size=(11, 7))
        ref = np.array([[sum(float(a)*float(b) for a,b in zip(row,wr)) for wr in w] for row in x])
        out = np.array([[sum(float(a)*float(b) for a,b in zip(row,wr)) for wr in q] for row in x])
        propagation = np.array([[sum(float(row[i])*(float(q[o,i])-float(w[o,i])) for i in range(7)) for o in range(3)] for row in x])
        before = [a.copy() for a in (w,q,x)]
        for batch in (1,3,256):
            r = evaluate(w,q,x,batch_rows=batch)
            self.assertAlmostEqual(r['metrics']['output_mse'], float(np.mean((out-ref)**2)))
            self.assertAlmostEqual(r['metrics']['propagated_mse'], float(np.mean(propagation**2)))
            self.assertAlmostEqual(r['metrics']['relative_output_l2'], float(np.linalg.norm(out-ref)/np.linalg.norm(ref)))
            self.assertEqual(r['execution']['maximum_batch_rows'], min(batch,len(x)))
            for i,c in enumerate(r['channels']):
                self.assertAlmostEqual(c['output_mse'], float(np.mean((out-ref)[:,i]**2)))
        for actual,old in zip((w,q,x),before):
            np.testing.assert_array_equal(actual,old)

    def test_forward_rounding_mask_does_not_hide_propagation(self):
        r = evaluate(np.array([[1.,.1]]),np.array([[1.,0.]]),np.array([[1e16,1.]]))
        self.assertEqual(r['metrics']['output_mse'],0)
        self.assertAlmostEqual(r['metrics']['propagated_mse'],.01)
        self.assertTrue(r['rounding_mask_detected'])
        self.assertEqual(r['worst_samples'][0]['sample_index'],0)
        self.assertEqual(r['worst_samples'][0]['output_channel'],0)
        self.assertAlmostEqual(r['worst_samples'][0]['max_abs_error'],.1)

    def test_exact_candidate_zero_reference_and_known_worst(self):
        w=np.zeros((2,2));x=np.array([[1.,0.],[0.,8.],[0.,2.]])
        r=evaluate(w,w.copy(),x,batch_rows=1)
        self.assertEqual(r['metrics']['propagated_mse'],0)
        self.assertIsNone(r['metrics']['relative_output_l2'])
        self.assertEqual(r['worst_samples'],[])
        q=np.array([[1.,0.],[0.,1.]])
        r=evaluate(w,q,x,batch_rows=2)
        self.assertEqual([(v['sample_index'],v['output_channel']) for v in r['worst_samples']],[(1,1),(2,1),(0,0)])
        self.assertEqual(r['worst_samples'][0]['max_abs_error'],8)

    def test_large_finite_and_underflow(self):
        r=evaluate(np.zeros((1,2)),np.array([[1e153,0.]]),np.array([[1.,0.],[.5,0.]]))
        self.assertAlmostEqual(r['metrics']['propagated_mse']/1e306,.625)
        with self.assertRaisesRegex(ValueError,'underflow'):
            evaluate(np.array([[7.,.5]]),np.array([[7.,0.]]),np.array([[0.,1e-323],[0.,0.],[0.,0.],[0.,0.]]))
        with self.assertRaises(ValueError):
            evaluate(np.ones((1,1))*1e308,np.zeros((1,1)),np.ones((1,1))*1e308)

    def test_row_batches_are_observed_at_real_contraction_boundary(self):
        from importlib import import_module
        from unittest.mock import patch
        module=import_module('roundofflab.study_metrics')
        real=module._row_products;seen=[]
        def tracked(x,w):
            seen.append(len(x));return real(x,w)
        with patch.object(module,'_row_products',side_effect=tracked):
            evaluate(np.ones((3,7)),np.ones((3,7))*.8,np.ones((11,7)),batch_rows=3)
        self.assertEqual(seen,[3,3,3,3,3,3,3,3,3,2,2,2])

    def test_same_values_in_strided_layouts_have_identical_metrics(self):
        w=np.arange(21,dtype=np.float64).reshape(3,7)/8
        q=w+np.array([.125,-.25,.5])[:,None]
        x=np.arange(77,dtype=np.float64).reshape(11,7)-30
        def views(a):
            backing=np.zeros((a.shape[0]*2,a.shape[1]*2));backing[::2,::2]=a
            return [a.copy(order='C'),a.copy(order='F'),backing[::2,::2],a[:,::-1].copy()[:,::-1]]
        reference=evaluate(w,q,x,batch_rows=1)
        for ww,qq in zip(views(w),views(q)):
            for xx in views(x):
                for batch in (1,3,256):
                    actual=evaluate(ww,qq,xx,batch_rows=batch)
                    self.assertEqual(actual['metrics'],reference['metrics'])
                    self.assertEqual(actual['channels'],reference['channels'])
                    self.assertEqual(actual['worst_samples'],reference['worst_samples'])

    def test_invalid_batch_and_shapes(self):
        w=np.ones((2,3));x=np.ones((2,3))
        for n in (0,-1,True,2.5,4097):
            with self.assertRaises(ValueError):evaluate(w,w,x,batch_rows=n)
        with self.assertRaises(ValueError):evaluate(w,np.ones((3,2)),x)


if __name__=='__main__':unittest.main()
