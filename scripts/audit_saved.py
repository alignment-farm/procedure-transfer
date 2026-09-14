"""Reload artifacts; repeat index-zero cases only as verification, not new evidence."""
import argparse,json,time
from pathlib import Path
import mlx.core as mx
from runtime import Runtime,digest,sha
from task import content,evaluation,cases,TRAIN_WORDS
p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args();run=args.run
out=Path(str(run)+'-reload-audit');out.mkdir(exist_ok=False)
(out/'audit_saved.py').write_bytes(Path(__file__).read_bytes())
rows=[json.loads(s) for s in (run/'responses.jsonl').read_text().splitlines()]
events=[json.loads(s) for s in (run/'events.jsonl').read_text().splitlines()]
assert events[-1]['kind']=='complete' and events[-1]['status']=='complete'
assert json.loads((run/'evaluation.json').read_text())==evaluation()
started=time.monotonic();rt=Runtime();records=[]
for r in rows:
    branch=r['branch'] if r['branch'] in ['none','examples','rule'] else 'none'
    assert rt.encode(content(r['case'],branch))==r['prefix']
    eos=rt.tokenizer.eos_token_ids
    assert (r['ids'][-1] in eos)==r['ended']
    assert not any(t in eos for t in r['ids'][:-1])
    assert rt.tokenizer.decode(r['ids'][:-1] if r['ended'] else r['ids'])==r['raw']
    expected_cases=evaluation() if r['suite']=='transfer' else cases(TRAIN_WORDS)
    assert r['case']==expected_cases[r['index']]
for r in [r for r in rows if r['suite']=='transfer' and r['index']==0]:
    if r['seed'] is None:rt.restore(rt.initial)
    else:
        path=run/f'{r["branch"]}-{r["seed"]}.safetensors'
        state=list(mx.load(str(path)).items())
        expected=next(e['hash'] for e in events if e['kind']=='training_complete' and e['method']==r['branch'] and e['seed']==r['seed'])
        assert digest(state)==expected
        rt.restore(state)
    repeat=rt.generate(r['prefix'])
    assert repeat['ids']==r['ids'], (r['branch'],r['seed'])
    records.append(dict(branch=r['branch'],seed=r['seed'],index=0,exact_token_match=True,**repeat))
    print('reload verified',r['branch'],r['seed'],flush=True)
invariants=rt.invariants()
summary=dict(status='passed',all_prompt_and_output_token_records_checked=len(rows),
             repeated_cases_are_not_new_test_evidence=True,records=records,
             seconds=time.monotonic()-started,peak_mlx_bytes=mx.get_peak_memory(),**invariants)
(out/'audit.json').write_text(json.dumps(summary,indent=2))
(out/'SHA256SUMS').write_text('\n'.join(sha(p)+'  '+p.name for p in sorted(out.iterdir()) if p.name!='SHA256SUMS'))
