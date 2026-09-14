"""One predeclared generic induction reminder; development only, no training."""
import json,time
from pathlib import Path
import mlx.core as mx
from runtime import Runtime,sha
from task import cases,TRAIN_WORDS,DEV_WORDS,content,context,oracle,SCHEMA,query

REMINDER=('Infer a single consistent protocol from all checked calls below. '
          'Account for both input fields and all changes to the text argument '
          'when applying it to the new identifier.\n')
out=Path('evidence/pilot-02-teacher');out.mkdir(parents=True,exist_ok=False)
for f in ['teacher_pilot.py','task.py','runtime.py']:(out/f).write_bytes(Path('scripts',f).read_bytes())
(out/'design.json').write_text(json.dumps(dict(candidate=REMINDER,
    selection='Use candidate only if strictly more than 9/16 development calls correct; otherwise retain original. No further variants.',
    development=cases(DEV_WORDS),recall=cases(TRAIN_WORDS)),indent=2))
tick=time.monotonic();rt=Runtime();log=(out/'responses.jsonl').open('x');rows=[]
try:
    for suite,cs in [('development',cases(DEV_WORDS)),('recall',cases(TRAIN_WORDS))]:
        for c in cs:
            text=SCHEMA+'\n'+REMINDER+context()+'\n'+query(c)
            r=rt.generate(rt.encode(text));row=dict(suite=suite,case=c,expected=oracle(c),correct=r['action']==oracle(c),prompt=text,**r)
            log.write(json.dumps(row)+'\n');log.flush();rows.append(row)
        print(suite,sum(r['correct'] for r in rows if r['suite']==suite),'/16',flush=True)
    summary=dict(seconds=time.monotonic()-tick,peak_mlx_bytes=mx.get_peak_memory(),**rt.invariants(),
        development_correct=sum(r['correct'] for r in rows if r['suite']=='development'),
        recall_correct=sum(r['correct'] for r in rows if r['suite']=='recall'))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
finally:
    log.close();(out/'SHA256SUMS').write_text('\n'.join(sha(p)+'  '+p.name for p in sorted(out.iterdir()) if p.name!='SHA256SUMS'))
