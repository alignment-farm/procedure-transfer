"""Controlled loss-direction rescue from identical collapsed weights."""
import argparse,json,random,time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from runtime import Runtime,logsoftmax,reverse_kl,digest,sha
from task import cases,SCHEMA,query,oracle
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();out=Path(str(a.run)+'-rescue');out.mkdir(exist_ok=False)
(out/'diagnostic_rescue.py').write_bytes(Path(__file__).read_bytes());(out/'protocol.md').write_bytes(Path('protocol/diagnosis-v2-rescue.md').read_bytes())
start=time.monotonic();ev=(out/'events.jsonl').open('x');resp=(out/'responses.jsonl').open('x')
def emit(kind,**kw):ev.write(json.dumps(dict(kind=kind,elapsed=time.monotonic()-start,**kw))+'\n');ev.flush()
status='failed'
try:
    rt=Runtime();rt.reinitialize(41);data=json.loads((a.run/'tokens.json').read_text());order=json.loads((a.run/'order.json').read_text())[:128]
    for r in data:r['t'],sec,n=rt.teacher(r['tp'],r['y']);emit('teacher_cache',seconds=sec,input_tokens=n)
    for name,file,kind in [('collapsed_reverse','reverse-0.0005-256','reverse'),('collapsed_forward','reverse-0.0005-256','forward'),('acquired_reverse','ce-0.0005-128','reverse')]:
        state=list(mx.load(str(a.run/f'{file}.safetensors')).items());rt.restore(state);opt=optim.AdamW(learning_rate=.0005,weight_decay=0.)
        emit('start',arm=name,source_checkpoint=file,initial_hash=digest(rt.snapshot()));tick=time.monotonic()
        for step,index in enumerate(order,1):
            assert time.monotonic()-start<1200 and mx.get_peak_memory()<40e9
            r=data[index];ids=mx.array(r['p']+r['y'])[None,:];pos=len(r['p']);rt.model.train()
            def objective(m):
                z=m(ids[:,:-1])[:,pos-1:,:]
                if kind=='reverse':return reverse_kl(z,r['t'])
                lp=logsoftmax(z);return mx.mean(mx.sum(mx.exp(r['t'])*(r['t']-lp),axis=-1))
            value,grad=nn.value_and_grad(rt.model,objective)(rt.model);opt.update(rt.model,grad);mx.eval(rt.model.parameters(),opt.state,value)
            assert mx.isfinite(value).item();emit('update',arm=name,step=step,index=index,loss=value.item())
        mx.save_safetensors(str(out/f'{name}.safetensors'),dict(rt.snapshot()));emit('trained',arm=name,seconds=time.monotonic()-tick,hash=digest(rt.snapshot()))
        for suite,cs in [('recall',[r['case'] for r in data]),('development',cases(['halmek','przuna','veskolit','dunfepazor']))]:
            correct=0
            for c in cs:
                prefix=rt.encode(SCHEMA+'\n\n'+query(c));r=rt.generate(prefix);ok=r['action']==oracle(c);correct+=ok
                resp.write(json.dumps(dict(arm=name,suite=suite,case=c,prefix=prefix,expected=oracle(c),correct=ok,**r))+'\n');resp.flush()
            emit('evaluation',arm=name,suite=suite,correct=correct,total=len(cs));print(name,suite,correct,flush=True)
        emit('invariants',arm=name,**rt.invariants())
    status='complete'
except BaseException as e:emit('failure',error=repr(e));raise
finally:
    emit('complete',status=status,seconds=time.monotonic()-start,peak_mlx_bytes=mx.get_peak_memory());ev.close();resp.close()
    (out/'SHA256SUMS').write_text('\n'.join(sha(p)+'  '+p.name for p in sorted(out.iterdir()) if p.is_file() and p.name!='SHA256SUMS'))
