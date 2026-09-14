"""Check the exact source-loaded diagnostic objective against analytic gradients."""
import ast,sys,unittest
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from runtime import logsoftmax,reverse_kl

class DiagnosticMath(unittest.TestCase):
    def test_forward_and_reverse(self):
        path=Path(__file__).resolve().parents[1]/'scripts/diagnosis_v2.py'
        node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='loss')
        env=dict(mx=mx,nn=nn,logsoftmax=logsoftmax,reverse_kl=reverse_kl)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env)
        p=np.array([.2,.3,.5]);q=np.array([.1,.7,.2]);z=mx.array(np.log(p).astype('float32'));t=mx.array(np.log(q).astype('float32'))
        for kind,value,gradient in [('forward',sum(q*np.log(q/p)),p-q),('reverse',sum(p*np.log(p/q)),p*(np.log(p/q)-sum(p*np.log(p/q))))]:
            f=lambda x:env['loss'](x,t,kind,mx.array(0))
            np.testing.assert_allclose(f(z).item(),value,rtol=3e-6)
            np.testing.assert_allclose(np.array(mx.grad(f)(z)),gradient,rtol=3e-6)

if __name__=='__main__':unittest.main()
