import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import mlx.core as mx
import numpy as np
from runtime import reverse_kl, logsoftmax
from task import cases, TRAIN_WORDS, DEV_WORDS, oracle

class MathTask(unittest.TestCase):
    def test_kl_value_and_gradient(self):
        # Independent analytic derivative: p_j*(log(p_j/q_j) - KL).
        p=np.array([.2,.3,.5]);q=np.array([.1,.7,.2]);z=mx.array(np.log(p).astype(np.float32))
        teacher=mx.array(np.log(q).astype(np.float32))
        f=lambda v:reverse_kl(v,teacher)
        got=f(z).item();expected=np.sum(p*np.log(p/q))
        np.testing.assert_allclose(got,expected,rtol=2e-6)
        np.testing.assert_allclose(np.array(mx.grad(f)(z)),p*(np.log(p/q)-expected),rtol=3e-6)
        self.assertAlmostEqual(reverse_kl(z,logsoftmax(z)).item(),0,places=6)
        # Verify teacher stop-gradient independently.
        np.testing.assert_array_equal(np.array(mx.grad(lambda t:reverse_kl(z,t))(teacher)),np.zeros(3))
    def test_contract(self):
        expected=['kestrel(text="AX")','kestrel(text="AX-Q")','marten(text="AX")','marten(text="AX-Q")']
        self.assertEqual([oracle(c) for c in cases(['ax'])],expected)
        self.assertEqual(len(cases(TRAIN_WORDS)),16)
        self.assertFalse(set(TRAIN_WORDS)&set(DEV_WORDS))

if __name__=='__main__':unittest.main()
