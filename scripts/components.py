"""Post-hoc descriptive error audit; not checkpoint selection or causal diagnosis."""
import argparse,collections,json,re
from pathlib import Path
from task import oracle
from runtime import sha

def components(raw,c):
    m=re.fullmatch(r'(kestrel|marten)\(text="([^"\n]*)"\)',raw.strip())
    if not m:return dict(format=False,route=False,identifier=False,suffix=False)
    tool,arg=m.groups();body=arg[:-2] if arg.endswith('-Q') else arg
    return dict(format=True,route=tool=={'copper':'kestrel','violet':'marten'}[c['channel']],
                identifier=body==c['identifier'].upper(),suffix=arg.endswith('-Q')==(c['priority']=='fast'))

# Independent hand-written parser checks, including a misleading embedded call.
c=dict(channel='violet',priority='fast',identifier='ax')
assert components('marten(text="AX-Q")',c)==dict(format=True,route=True,identifier=True,suffix=True)
assert components('kestrel(text="AX")',c)==dict(format=True,route=False,identifier=True,suffix=False)
assert not components('Answer: marten(text="AX-Q")',c)['format']
assert not components('marten(text="A"")',c)['format']

p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();run=a.run
out=Path(str(run)+'-components');out.mkdir(exist_ok=False)
rows=[json.loads(s) for s in (run/'responses.jsonl').read_text().splitlines()]
events=[json.loads(s) for s in (run/'events.jsonl').read_text().splitlines()]
train=json.loads((run/'acquisition.json').read_text());result=[]
for branch,seed in sorted({(r['branch'],r['seed']) for r in rows},key=str):
    rs=[r for r in rows if r['branch']==branch and r['seed']==seed and r['suite']=='transfer']
    byword=collections.defaultdict(list)
    for r in rs:byword[r['case']['identifier']].append(r['raw'])
    scores=[components(r['raw'],r['case']) for r in rs]
    result.append(dict(branch=branch,seed=seed,n=len(rs),correct=sum(r['correct'] for r in rs),
        component_counts={k:sum(s[k] for s in scores) for k in ['format','route','identifier','suffix']},
        identical_response_across_four_conditions=sum(len(set(v))==1 for v in byword.values()),
        identifiers=len(byword),emitted_tools=dict(collections.Counter(re.match(r'^(\w+)\(',r['raw']).group(1) if re.match(r'^(\w+)\(',r['raw']) else 'no-leading-tool' for r in rs))))
rollout=[]
for seed in [17,29]:
    es=[e for e in events if e['kind']=='train_step' and e['seed']==seed and e['method']=='rkl_onpolicy']
    epochs=[]
    for epoch in range(8):
        group=es[epoch*16:(epoch+1)*16]
        epochs.append(dict(epoch=epoch+1,n=16,correct=sum(e['rollout']['raw'].strip()==oracle(train[e['index']]['case']) for e in group),
                           truncated=sum(e['rollout']['at_limit'] for e in group)))
    rollout.append(dict(seed=seed,epochs=epochs))
report=dict(scope='Post-hoc descriptive parsing. Components require exact call syntax and are not independent successes. No mechanism intervention or new model calls.',transfer=result,onpolicy_training_rollouts=rollout)
(out/'components.json').write_text(json.dumps(report,indent=2));(out/'components.py').write_bytes(Path(__file__).read_bytes())
(out/'SHA256SUMS').write_text('\n'.join(sha(p)+'  '+p.name for p in sorted(out.iterdir()) if p.name!='SHA256SUMS'))
print(json.dumps(report,indent=2))
