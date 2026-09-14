"""Prospective comparison; freeze protocol and source before invoking."""
import argparse,json,random,subprocess,time
from pathlib import Path
import mlx.core as mx
from runtime import Runtime,resource,sha,digest
from task import cases,TRAIN_WORDS,DEV_WORDS,content,oracle,evaluation

parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=False);out=a.output
started=time.monotonic();events=(out/'events.jsonl').open('x');responses=(out/'responses.jsonl').open('x')
def emit(kind,**kw):
    events.write(json.dumps(dict(kind=kind,elapsed=time.monotonic()-started,**kw))+'\n');events.flush()
def check():
    if time.monotonic()-started>5400: raise TimeoutError('90-minute main-run limit')
    if mx.get_peak_memory()>40_000_000_000: raise MemoryError('40 GB MLX peak limit')
def save(name,data): (out/name).write_text(json.dumps(data,indent=2))
for name in ['experiment.py','runtime.py','task.py']:(out/name).write_bytes(Path('scripts',name).read_bytes())
(out/'protocol.md').write_bytes(Path('protocol/transfer-v1.md').read_bytes())
status='failed'
try:
    save('resource.json',resource());emit('revision',git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),status=subprocess.check_output(['git','status','--short'],text=True))
    rt=Runtime();emit('loaded',base_hash=rt.base_hash)
    train=cases(TRAIN_WORDS);save('acquisition.json',[dict(case=c,expected=oracle(c)) for c in train])
    encoded=[dict(case=c,prefix=rt.encode(content(c)),teacher_prefix=rt.encode(content(c,'examples')),
                  answer=rt.target(rt.encode(content(c)),oracle(c))) for c in train]
    save('training_tokens.json',encoded)
    checkpoints={}
    for seed in [17,29]:
        rt.reinitialize(seed);initial=rt.snapshot();initial_hash=digest(initial)
        save(f'initial-{seed}.json',dict(hash=initial_hash,parameters=sum(v.size for _,v in initial)))
        rng=random.Random(seed);order=[]
        for epoch in range(8):
            inds=list(range(16));rng.shuffle(inds);order+=inds
        save(f'order-{seed}.json',order)
        for method in ['sft','rkl_checked','rkl_onpolicy']:
            rt.restore(initial);mx.random.seed(seed+1000);opt=rt.optimizer();tick=time.monotonic()
            emit('training_start',seed=seed,method=method,initial_hash=digest(rt.snapshot()))
            for step,index in enumerate(order,1):
                check();row=encoded[index];p=row['prefix'];tp=row['teacher_prefix'];y=row['answer'];rollout=None
                if method=='rkl_onpolicy':
                    rollout=rt.generate(p,temp=1.,limit=48);y=rollout['ids']
                teacher_lp=None;teacher_s=0.;teacher_n=0
                if method!='sft':teacher_lp,teacher_s,teacher_n=rt.teacher(tp,y)
                result=rt.step(p,y,opt,teacher_lp)
                emit('train_step',seed=seed,method=method,step=step,index=index,y=y,rollout=rollout,
                     teacher_seconds=teacher_s,teacher_input_tokens=teacher_n,**result)
                if step%16==0: print(seed,method,step,result['loss'],flush=True)
            state=rt.snapshot();checkpoints[(seed,method)]=state
            mx.save_safetensors(str(out/f'{method}-{seed}.safetensors'),dict(state))
            emit('training_complete',seed=seed,method=method,seconds=time.monotonic()-tick,hash=digest(state),**rt.invariants())
    emit('all_checkpoints_fixed')
    # First construction of prospective identifiers occurs after every checkpoint is fixed.
    tests=evaluation();assert not {c['identifier'] for c in tests}&set(TRAIN_WORDS+DEV_WORDS)
    save('evaluation.json',tests)
    def evaluate(branch,seed,suite,data):
        for index,c in enumerate(data):
            check();ctx=branch if branch in ['none','examples','rule'] else 'none'
            p=rt.encode(content(c,ctx));r=rt.generate(p,limit=48)
            row=dict(branch=branch,seed=seed,suite=suite,index=index,case=c,prefix=p,
                     expected=oracle(c),correct=r['action']==oracle(c),**r)
            responses.write(json.dumps(row)+'\n');responses.flush()
        print('evaluated',branch,seed,suite,len(data),flush=True)
    rt.restore(rt.initial)
    for branch in ['none','examples','rule']:
        evaluate(branch,None,'transfer',tests)
        if branch=='examples':evaluate(branch,None,'recall',train)
    for (seed,method),state in checkpoints.items():
        rt.restore(state);evaluate(method,seed,'recall',train);evaluate(method,seed,'transfer',tests)
    emit('final_invariants',**rt.invariants());status='complete'
except BaseException as e:
    emit('failure',error=repr(e));raise
finally:
    emit('complete',status=status,seconds=time.monotonic()-started,peak_mlx_bytes=mx.get_peak_memory())
    events.close();responses.close()
    (out/'SHA256SUMS').write_text('\n'.join(sha(p)+'  '+p.name for p in sorted(out.iterdir()) if p.is_file() and p.name!='SHA256SUMS'))
