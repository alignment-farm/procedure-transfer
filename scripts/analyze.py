"""Audit and summarize immutable completed run; no model calls."""
import argparse,collections,json
from pathlib import Path
import numpy as np
from task import oracle
from runtime import sha

p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args();run=args.run
rows=[json.loads(s) for s in (run/'responses.jsonl').read_text().splitlines()]
events=[json.loads(s) for s in (run/'events.jsonl').read_text().splitlines()]
assert events[-1]['kind']=='complete' and events[-1]['status']=='complete'
for line in (run/'SHA256SUMS').read_text().splitlines():
    h,name=line.split('  ',1);assert sha(run/name)==h,name
for r in rows:
    assert r['expected']==oracle(r['case'])
    assert r['correct']==(r['raw'].strip()==oracle(r['case']))
    assert len(r['ids'])==r['completion_tokens'] and len(r['prefix'])==r['prompt_tokens']
keys={(r['branch'],r['seed'],r['suite']) for r in rows}
metrics=[]
for branch,seed,suite in sorted(keys,key=str):
    rs=[r for r in rows if (r['branch'],r['seed'],r['suite'])==(branch,seed,suite)]
    assert len(rs)==(96 if suite=='transfer' else 16)
    assert len({r['index'] for r in rs})==len(rs)
    strata={}
    for length in sorted({len(r['case']['identifier']) for r in rs}):
        z=[r for r in rs if len(r['case']['identifier'])==length]
        strata[str(length)]=dict(correct=sum(r['correct'] for r in z),n=len(z))
    metrics.append(dict(branch=branch,seed=seed,suite=suite,correct=sum(r['correct'] for r in rs),n=len(rs),
        prompt_tokens=sum(r['prompt_tokens'] for r in rs),completion_tokens=sum(r['completion_tokens'] for r in rs),
        seconds=sum(r['seconds'] for r in rs),truncated=sum(r['at_limit'] for r in rs),lengths=strata))
train=[]
for seed in [17,29]:
    for method in ['sft','rkl_checked','rkl_onpolicy']:
        es=[e for e in events if e['kind']=='train_step' and e['seed']==seed and e['method']==method]
        assert len(es)==128 and [e['step'] for e in es]==list(range(1,129))
        assert sorted(collections.Counter(e['index'] for e in es).values())==[8]*16
        assert all(np.isfinite(e['loss']) and np.isfinite(e['gradient_norm']) for e in es)
        done=next(e for e in events if e['kind']=='training_complete' and e['seed']==seed and e['method']==method)
        assert done['base_unchanged'] and done['reset_max_logit_delta']==0
        train.append(dict(seed=seed,method=method,steps=len(es),total_seconds=done['seconds'],
            update_seconds=sum(e['seconds'] for e in es),teacher_seconds=sum(e['teacher_seconds'] for e in es),
            teacher_input_tokens=sum(e['teacher_input_tokens'] for e in es),
            rollout_seconds=sum(e['rollout']['seconds'] for e in es if e['rollout']),
            rollout_prompt_tokens=sum(e['rollout']['prompt_tokens'] for e in es if e['rollout']),
            rollout_completion_tokens=sum(e['rollout']['completion_tokens'] for e in es if e['rollout']),
            rollout_truncated=sum(e['rollout']['at_limit'] for e in es if e['rollout']),
            update_input_tokens=sum(e['input_tokens'] for e in es),loss_tokens=sum(e['loss_tokens'] for e in es),
            first_epoch_loss=float(np.mean([e['loss'] for e in es[:16]])),last_epoch_loss=float(np.mean([e['loss'] for e in es[-16:]]))))
words=sorted({r['case']['identifier'] for r in rows if r['suite']=='transfer'});assert len(words)==24
contrasts=[]
for baseline in ['sft','rkl_checked']:
    seed_diffs=[];diff=[]
    for seed in [17,29]:
        def scores(method):
            d={w:[] for w in words}
            for r in rows:
                if r['suite']=='transfer' and r['seed']==seed and r['branch']==method:d[r['case']['identifier']].append(r['correct'])
            assert all(len(v)==4 for v in d.values())
            return np.array([np.mean(d[w]) for w in words])
        delta=scores('rkl_onpolicy')-scores(baseline);seed_diffs.append(float(delta.mean()));diff.append(delta)
    clusters=np.mean(diff,axis=0);rng=np.random.default_rng(20260914)
    boots=clusters[rng.integers(0,24,size=(10000,24))].mean(axis=1)
    lo,hi=np.quantile(boots,[.025,.975])
    contrasts.append(dict(contrast='rkl_onpolicy - '+baseline,mean_difference=float(clusters.mean()),
        seed_differences=seed_diffs,identifier_cluster_95_interval=[float(lo),float(hi)],
        positive_local_advantage=all(d>0 for d in seed_diffs) and lo>0))
report=dict(audit='passed',metrics=metrics,training=train,contrasts=contrasts,run=events[-1])
# Analysis does not mutate the run whose hashes it validates.
out=Path(str(run)+'-analysis');out.mkdir(exist_ok=False)
(out/'analysis.json').write_text(json.dumps(report,indent=2))
(out/'analyze.py').write_bytes(Path(__file__).read_bytes())
lines=['# Audited results','', '| Branch | Seed | Recall | Transfer | Length 4 | 6 | 8 | 10 |','|---|---:|---:|---:|---:|---:|---:|---:|']
for m in metrics:
    if m['suite']!='transfer':continue
    rec=next((x for x in metrics if x['suite']=='recall' and x['branch']==m['branch'] and x['seed']==m['seed']),None)
    cells=[m['branch'],str(m['seed'] or 'base'),str(rec['correct'])+'/16' if rec else '—',str(m['correct'])+'/96']
    cells += [str(m['lengths'][str(n)]['correct'])+'/24' for n in [4,6,8,10]]
    lines.append('| '+' | '.join(cells)+' |')
lines+=['','| Method | Seed | Training seconds | Teacher seconds | Rollout seconds | Update seconds | Loss tokens |','|---|---:|---:|---:|---:|---:|---:|']
for t in train:lines.append('| '+ ' | '.join([t['method'],str(t['seed'])]+[f'{t[k]:.2f}' for k in ['total_seconds','teacher_seconds','rollout_seconds','update_seconds']]+[str(t['loss_tokens'])])+' |')
lines+=['','Paired contrasts (percentage points; descriptive identifier-cluster intervals):','']
for c in contrasts:lines.append(f'- {c["contrast"]}: {100*c["mean_difference"]:.2f}; interval {100*c["identifier_cluster_95_interval"][0]:.2f} to {100*c["identifier_cluster_95_interval"][1]:.2f}; seeds {[round(100*d,2) for d in c["seed_differences"]]}. Positive-advantage rule: {c["positive_local_advantage"]}.')
(out/'README.md').write_text('\n'.join(lines)+'\n')
(out/'SHA256SUMS').write_text('\n'.join(sha(p)+'  '+p.name for p in sorted(out.iterdir()) if p.name!='SHA256SUMS'))
print('\n'.join(lines))
