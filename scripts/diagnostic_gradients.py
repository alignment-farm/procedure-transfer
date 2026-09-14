"""First-response logit-space signal audit, without training or fresh test claims."""
import argparse,json,time
from pathlib import Path
import mlx.core as mx
from runtime import Runtime,logsoftmax,sha
p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--legacy',action='store_true');a=p.parse_args();out=Path(str(a.run)+'-gradients');out.mkdir(exist_ok=False)
(out/'diagnostic_gradients.py').write_bytes(Path(__file__).read_bytes())
data=json.loads((a.run/('training_tokens.json' if a.legacy else 'tokens.json')).read_text())
if a.legacy:data=[dict(case=r['case'],p=r['prefix'],tp=r['teacher_prefix'],y=r['answer']) for r in data]
rt=Runtime();rt.reinitialize(41);initial=rt.snapshot();start=time.monotonic();teacher=[];alignment=[]
for r in data:
    t,sec,n=rt.teacher(r['tp'],r['y']);teacher.append(t[0,0])
    standalone=logsoftmax(rt.logits(r['tp']))[0]
    delta=mx.max(mx.abs(standalone-t[0,0])).item()
    alignment.append(dict(case=r['case'],max_abs_logprob_delta=delta,full_argmax=int(t[0,0].argmax().item()),prefix_argmax=int(standalone.argmax().item())))
records=[]
states=['base']+sorted(p.stem for p in a.run.glob('*.safetensors')) if a.legacy else ['base','ce-0.0005-128','forward-0.0005-128','reverse-0.0005-256','reverse-5e-05-256','forward-0.0005-256','forward-5e-05-256']
for state in states:
    rt.restore(initial if state=='base' else list(mx.load(str(a.run/f'{state}.safetensors')).items()))
    for r,t in zip(data,teacher):
        lp=logsoftmax(rt.logits(r['p']))[0];prob=mx.exp(lp);q=mx.exp(t);kl=mx.sum(prob*(lp-t))
        rg=prob*(lp-t-kl);fg=prob-q;y=r['y'][0];token=int(lp.argmax().item())
        mx.eval(kl,rg,fg)
        records.append(dict(state=state,case=r['case'],correct_first_id=y,predicted_first_id=token,
            predicted_first_text=rt.tokenizer.decode([token]),p_correct=prob[y].item(),q_correct=q[y].item(),
            reverse_kl=kl.item(),reverse_correct_logit_gradient=rg[y].item(),forward_correct_logit_gradient=fg[y].item(),
            reverse_logit_gradient_norm=mx.linalg.norm(rg).item(),forward_logit_gradient_norm=mx.linalg.norm(fg).item()))
summary={}
for state in dict.fromkeys(r['state'] for r in records):
    rs=[r for r in records if r['state']==state];clear=[r for r in rs if r['q_correct']>.9];wrong=[r for r in clear if r['predicted_first_id']!=r['correct_first_id']]
    summary[state]=dict(teacher_clear=len(clear),student_wrong_on_teacher_clear=len(wrong),
        weak_reverse_strong_forward=sum(r['p_correct']<1e-4 and abs(r['reverse_correct_logit_gradient'])<1e-3 and abs(r['forward_correct_logit_gradient'])>.5 for r in wrong))
report=dict(scope='Raw first-position logit gradients, not parameter-gradient norms; computed analytically at saved model distributions. Thresholds specified before inspection in diagnosis-v2-signal.md.',summary=summary,records=records,teacher_alignment=alignment,seconds=time.monotonic()-start,**rt.invariants())
(out/'gradients.json').write_text(json.dumps(report,indent=2));(out/'SHA256SUMS').write_text('\n'.join(sha(p)+'  '+p.name for p in sorted(out.iterdir()) if p.name!='SHA256SUMS'))
print(json.dumps(summary,indent=2))
