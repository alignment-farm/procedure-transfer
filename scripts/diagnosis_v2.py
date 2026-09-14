"""Diagnostic loss-direction x learning-rate matrix; all outputs are development."""
import json,time,random,argparse,subprocess
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from runtime import Runtime,logsoftmax,reverse_kl,sha,digest,resource
from task import cases,query,oracle,SCHEMA,REMINDER
import re

def components(raw,c):
    m=re.fullmatch(r'(kestrel|marten)\(text="([^"\n]*)"\)',raw.strip())
    if not m:return dict(format=False,route=False,identifier=False,suffix=False)
    tool,arg=m.groups();body=arg[:-2] if arg.endswith("-Q") else arg
    return dict(format=True,route=tool=={"copper":"kestrel","violet":"marten"}[c["channel"]],identifier=body==c["identifier"].upper(),suffix=arg.endswith("-Q")== (c["priority"]=="fast"))
TRAIN=['pelk','druvan','snebit','korvaz']
DEV=['halmek','przuna','veskolit','dunfepazor']

def prompt(c,teacher=False):
    ctx=(REMINDER+'Checked successful calls:\n'+'\n'.join(query(x)+' -> '+oracle(x) for x in cases(TRAIN))) if teacher else ''
    return SCHEMA+'\n'+ctx+'\n'+query(c)

def loss(logits,t,kind,y):
    lp=logsoftmax(logits)
    if kind=='ce':return nn.losses.cross_entropy(logits.astype(mx.float32),y,reduction='mean')
    if kind=='reverse':return reverse_kl(logits,t)
    return mx.mean(mx.sum(mx.exp(t)*(t-lp),axis=-1))

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
out=a.output;out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
for f in ['diagnosis_v2.py','runtime.py','task.py']:(out/f).write_bytes(Path('scripts',f).read_bytes())
(out/'protocol.md').write_bytes(Path('protocol/diagnosis-v2.md').read_bytes())
ev=(out/'events.jsonl').open('x');resp=(out/'responses.jsonl').open('x')
def emit(kind,**kw):ev.write(json.dumps(dict(kind=kind,elapsed=time.monotonic()-start,**kw))+'\n');ev.flush()
def save(n,v):(out/n).write_text(json.dumps(v,indent=2))
def check():
    assert time.monotonic()-start<2700,'45-minute budget'
    assert mx.get_peak_memory()<40e9,'40 GB budget'
status='failed'
try:
    save('resource.json',resource());emit('revision',git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    rt=Runtime();rt.reinitialize(41);initial=rt.snapshot();data=[];audit=[]
    for c in cases(TRAIN):
        p=rt.encode(prompt(c));tp=rt.encode(prompt(c,True));y=rt.target(p,oracle(c))
        t,seconds,n=rt.teacher(tp,y)
        gen=rt.generate(tp)
        probs=mx.exp(t);gold=mx.take_along_axis(probs,mx.array(y)[None,:,None],axis=-1).reshape(-1)
        mx.eval(gold)
        audit.append(dict(case=c,expected=oracle(c),correct=gen['action']==oracle(c),teacher=gen,
            canonical_y=y,gold_probabilities=gold.tolist(),teacher_argmax=t.argmax(-1).reshape(-1).tolist(),
            first_argmax_text=rt.tokenizer.decode([int(t[0,0].argmax().item())]),
            canonical_equals_generated=gen['ids']==y,teacher_seconds=seconds,teacher_input_tokens=n))
        data.append(dict(case=c,p=p,tp=tp,y=y,t=t))
    save('teacher-audit.json',audit);save('tokens.json',[{k:v for k,v in r.items() if k!='t'} for r in data])
    print('teacher recall',sum(r['correct'] for r in audit),'/16',flush=True)
    def evaluate(name,step):
        scores={}
        for suite,cs in [('recall',cases(TRAIN)),('development',cases(DEV))]:
            correct=0
            for i,c in enumerate(cs):
                check();prefix=rt.encode(prompt(c));r=rt.generate(prefix);ok=r['action']==oracle(c);correct+=ok
                resp.write(json.dumps(dict(arm=name,step=step,suite=suite,index=i,case=c,prefix=prefix,expected=oracle(c),correct=ok,components=components(r['raw'],c),**r))+'\n');resp.flush()
            scores[suite]=correct
        emit('evaluation',arm=name,step=step,**scores);print(name,step,scores,flush=True)
    rng=random.Random(41);order=[]
    for _ in range(16):
        x=list(range(16));rng.shuffle(x);order+=x
    save('order.json',order)
    evaluate('base',0)
    for kind,lr in [('ce',.0005),('reverse',.0005),('reverse',.00005),('forward',.0005),('forward',.00005)]:
        name=f'{kind}-{lr:g}';rt.restore(initial);opt=optim.AdamW(learning_rate=lr,weight_decay=0.)
        tick=time.monotonic();emit('arm_start',arm=name,initial_hash=digest(rt.snapshot()))
        for step,index in enumerate(order,1):
            check();r=data[index];ids=mx.array(r['p']+r['y'])[None,:];pos=len(r['p']);rt.model.train()
            def objective(m):return loss(m(ids[:,:-1])[:,pos-1:,:],r['t'],kind,ids[:,pos:])
            before=time.monotonic();value,grad=nn.value_and_grad(rt.model,objective)(rt.model)
            norm=mx.sqrt(sum(mx.sum(g*g) for _,g in tree_flatten(grad)))
            opt.update(rt.model,grad);mx.eval(rt.model.parameters(),opt.state,value,norm)
            assert mx.isfinite(value).item() and mx.isfinite(norm).item()
            emit('update',arm=name,step=step,index=index,loss=value.item(),gradient_norm=norm.item(),seconds=time.monotonic()-before)
            if step in [32,128,256]:
                mx.save_safetensors(str(out/f'{name}-{step}.safetensors'),dict(rt.snapshot()));evaluate(name,step)
        emit('arm_complete',arm=name,seconds=time.monotonic()-tick,**rt.invariants())
    status='complete'
except BaseException as e:emit('failure',error=repr(e));raise
finally:
    emit('complete',status=status,seconds=time.monotonic()-start,peak_mlx_bytes=mx.get_peak_memory());ev.close();resp.close()
    (out/'SHA256SUMS').write_text('\n'.join(sha(p)+'  '+p.name for p in sorted(out.iterdir()) if p.is_file() and p.name!='SHA256SUMS'))
