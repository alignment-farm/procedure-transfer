"""Privileged-prefix intervention to separate routing, copying and suffix use."""
import argparse,json,random,string,time
from pathlib import Path
import mlx.core as mx
from runtime import Runtime,sha
from task import SCHEMA,REMINDER,cases,query,oracle
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();out=Path(str(a.run)+'-components');out.mkdir(exist_ok=False)
(out/'diagnostic_components.py').write_bytes(Path(__file__).read_bytes());(out/'protocol.md').write_bytes(Path('protocol/diagnosis-v2-signal.md').read_bytes())
data=json.loads((a.run/'tokens.json').read_text());rng=random.Random(2026091402);words=[]
known=set(['nup','zelk','bimav','fosted','wexi','jupnal','votkeris','zunpelavik','halmek','przuna','veskolit','dunfepazor']+[r['case']['identifier'] for r in data])
known.update(c['identifier'] for c in json.loads(Path('evidence/transfer-v1/evaluation.json').read_text()))
for length in [4,6,8,10]:
    for _ in range(3):
        while True:
            word=''.join(rng.choice(string.ascii_lowercase) for _ in range(length))
            if word not in known:break
        words.append(word);known.add(word)
cs=cases(words)
(out/'design.json').write_text(json.dumps(dict(seed=2026091402,cases=cs,checkpoints=['ce-0.0005-128','forward-0.0005-128'],selection='Earliest >=15/16 acquisition recall; both first succeed at 128. Lower-rate forward never qualifies.',conditions=['free','route_supplied','route_and_identifier_supplied'],warning='Oracle-supplied response prefixes are privileged diagnostics, not deployment results.'),indent=2))
start=time.monotonic();rt=Runtime();initial=rt.snapshot();log=(out/'responses.jsonl').open('x');summary=[]
try:
    for state in ['teacher','ce-0.0005-128','forward-0.0005-128']:
        rt.restore(initial if state=='teacher' else list(mx.load(str(a.run/f'{state}.safetensors')).items()))
        for mode in (['free'] if state=='teacher' else ['free','route_supplied','route_and_identifier_supplied']):
            correct=0
            for c in cs:
                tool={'copper':'kestrel','violet':'marten'}[c['channel']]
                supplied='' if mode=='free' else f'{tool}(text="'+(c['identifier'].upper() if mode=='route_and_identifier_supplied' else '')
                extra=(REMINDER+'Checked successful calls:\n'+'\n'.join(query(r['case'])+' -> '+oracle(r['case']) for r in data)) if state=='teacher' else ''
                base=rt.encode(SCHEMA+'\n'+extra+'\n'+query(c));forced=rt.tokenizer.encode(supplied,add_special_tokens=False) if supplied else []
                assert rt.tokenizer.decode(forced)==supplied
                r=rt.generate(base+forced);continuation=r['raw'];raw=supplied+continuation;ok=raw.strip()==oracle(c);correct+=ok
                record=dict(state=state,condition=mode,case=c,base_prefix=base,supplied=supplied,supplied_ids=forced,expected=oracle(c),correct=ok,**r)
                record.update(raw=raw,action=raw.strip(),continuation=continuation)
                log.write(json.dumps(record)+'\n');log.flush()
            summary.append(dict(state=state,condition=mode,correct=correct,total=len(cs)));print(summary[-1],flush=True)
    invariants=rt.invariants();(out/'summary.json').write_text(json.dumps(dict(results=summary,seconds=time.monotonic()-start,peak_mlx_bytes=mx.get_peak_memory(),**invariants),indent=2))
finally:
    log.close();(out/'SHA256SUMS').write_text('\n'.join(sha(p)+'  '+p.name for p in sorted(out.iterdir()) if p.name!='SHA256SUMS'))
