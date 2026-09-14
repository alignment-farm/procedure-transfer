"""Development-only teacher and native-gradient feasibility; no final test calls."""
import json,time
from pathlib import Path
import mlx.core as mx
from runtime import Runtime,resource,sha,digest
from task import cases,TRAIN_WORDS,DEV_WORDS,content,oracle

out=Path('evidence/pilot-01');out.mkdir(parents=True,exist_ok=False)
for name in ['pilot.py','runtime.py','task.py']:(out/name).write_bytes(Path('scripts',name).read_bytes())
started=time.monotonic()
log=(out/'events.jsonl').open('x')
def emit(kind,**kw):
    log.write(json.dumps(dict(kind=kind,elapsed=time.monotonic()-started,**kw))+'\n');log.flush()
try:
    info=resource();(out/'resource.json').write_text(json.dumps(info,indent=2))
    rt=Runtime();emit('loaded',base_hash=rt.base_hash,adapter_hash=digest(rt.initial))
    rows=[]
    for branch in ['none','examples','rule']:
        for c in cases(DEV_WORDS):
            p=rt.encode(content(c,branch));r=rt.generate(p)
            row=dict(branch=branch,case=c,expected=oracle(c),correct=r['action']==oracle(c),prompt=content(c,branch),**r)
            emit('reference',**row);rows.append(row)
        print(branch,sum(r['correct'] for r in rows if r['branch']==branch),'/16',flush=True)
    # Same checked input across objectives, two updates each; no performance selection.
    for method in ['sft','rkl_checked','rkl_onpolicy']:
        rt.restore(rt.initial);opt=rt.optimizer()
        for c in cases(TRAIN_WORDS)[:2]:
            p=rt.encode(content(c));tp=rt.encode(content(c,'examples'))
            rollout=rt.generate(p,temp=1.) if method=='rkl_onpolicy' else None
            y=rollout['ids'] if rollout else rt.target(p,oracle(c))
            t=None;teacher_s=0;teacher_n=0
            if method!='sft':
                t,teacher_s,teacher_n=rt.teacher(tp,y)
                assert t.shape==(1,len(y),rt.model.args.vocab_size)
                current=rt.snapshot();rt.restore(rt.initial)
                first=rt.logits(tp)
                from runtime import logsoftmax
                # Teacher first distribution is independent of response suffix.
                delta=mx.max(mx.abs(t[:,0,:]-logsoftmax(first))).item()
                assert delta<.3
                rt.restore(current)
                emit("teacher_alignment",max_delta=delta,shape=list(t.shape))
            result=rt.step(p,y,opt,t)
            emit('update',method=method,y=y,prefix=p,teacher_prefix=tp,rollout=rollout,
                 teacher_seconds=teacher_s,teacher_input_tokens=teacher_n,**result)
            assert result['gradient_norm']>0
        emit('invariants',method=method,**rt.invariants())
    emit('complete',seconds=time.monotonic()-started,peak_mlx_bytes=mx.get_peak_memory())
except BaseException as e:
    emit('failure',error=repr(e));raise
finally:
    log.close()
    (out/'SHA256SUMS').write_text('\n'.join(sha(p)+'  '+p.name for p in sorted(out.iterdir()) if p.name!='SHA256SUMS'))
