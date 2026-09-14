"""Native MLX full-vocabulary distillation, adapting prior study's LoRA route."""
import hashlib
import importlib.metadata as md
import json
import time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.utils import linear_to_lora_layers

MODEL=Path('models/qwen3-4b-instruct')
REVISION='cdbee75f17c01a7cc42f958dc650907174af0554'

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def digest(items):
    h=hashlib.sha256()
    for k,v in sorted(items):
        h.update(k.encode());h.update(str((v.shape,v.dtype)).encode())
        h.update(np.asarray(v.astype(mx.float32)).tobytes())
    return h.hexdigest()

def logsoftmax(x):
    x=x.astype(mx.float32)
    return x-mx.logsumexp(x,axis=-1,keepdims=True)

def reverse_kl(logits, teacher_logp):
    lp=logsoftmax(logits)
    return mx.mean(mx.sum(mx.exp(lp)*(lp-mx.stop_gradient(teacher_logp)),axis=-1))

class Runtime:
    def __init__(self, seed=17):
        mx.random.seed(seed)
        self.model,self.tokenizer=load(str(MODEL))
        self.model.freeze()
        linear_to_lora_layers(self.model,8,dict(rank=8,scale=2.,dropout=0.,keys=['self_attn.q_proj','self_attn.v_proj']))
        mx.eval(self.model.parameters())
        self.initial=self.snapshot()
        self.base_hash=digest(self.base())
        self.probe=self.encode('Return exactly OK.')
        self.probe_logits=self.logits(self.probe)
    def base(self):
        return [(k,v) for k,v in tree_flatten(self.model.parameters()) if 'lora_' not in k]
    def snapshot(self):
        return [(k,mx.array(v)) for k,v in tree_flatten(self.model.trainable_parameters())]
    def restore(self, state):
        self.model.update(tree_unflatten(state));mx.eval(self.model.parameters());self.model.eval()
    def reinitialize(self, seed):
        mx.random.seed(seed)
        state=[]
        for k,v in self.initial:
            if k.endswith('lora_a'):
                scale=1/(v.shape[0]**.5)
                a=mx.random.uniform(low=-scale,high=scale,shape=v.shape)
            else: a=mx.zeros_like(v)
            state.append((k,a))
        self.restore(state); self.initial=self.snapshot()
    def encode(self, content):
        return self.tokenizer.apply_chat_template([dict(role='user',content=content)],tokenize=True,add_generation_prompt=True,enable_thinking=False)
    def target(self,prefix,text):
        # Tokenize prefix and continuation separately, as generated tokens are appended.
        y=self.tokenizer.encode(text,add_special_tokens=False)+[self.tokenizer.eos_token_id]
        assert self.tokenizer.decode(y[:-1])==text
        return y
    def logits(self,ids):
        z=self.model(mx.array(ids)[None,:])[:,-1,:].astype(mx.float32)
        mx.eval(z);return z
    def generate(self, prefix, temp=0, limit=48):
        self.model.eval();tick=time.monotonic();ys=[];ended=False
        eos=self.tokenizer.eos_token_ids
        for token,lp in generate_step(mx.array(prefix),self.model,max_tokens=limit,sampler=make_sampler(temp=temp)):
            ys.append(int(token))
            if int(token) in eos:
                ended=True;break
        raw=self.tokenizer.decode(ys[:-1] if ended else ys)
        return dict(ids=ys,raw=raw,action=raw.strip(),ended=ended,at_limit=not ended,
                    seconds=time.monotonic()-tick,prompt_tokens=len(prefix),completion_tokens=len(ys))
    def teacher(self,prefix,y):
        current=self.snapshot();tick=time.monotonic();self.restore(self.initial)
        ids=prefix+y[:-1]
        logits=self.model(mx.array(ids)[None,:])[:,len(prefix)-1:,:]
        lp=mx.stop_gradient(logsoftmax(logits));mx.eval(lp)
        self.restore(current)
        return lp,time.monotonic()-tick,len(ids)
    def optimizer(self):
        return optim.AdamW(learning_rate=0.0005,weight_decay=0.)
    def step(self,prefix,y,optimizer,teacher_logp=None):
        self.model.train();tick=time.monotonic()
        ids=mx.array(prefix+y)[None,:];start=len(prefix)
        def loss_fn(m):
            z=m(ids[:,:-1])[:,start-1:,:]
            if teacher_logp is None:
                return nn.losses.cross_entropy(z.astype(mx.float32),ids[:,start:],reduction='mean')
            return reverse_kl(z,teacher_logp)
        loss,grad=nn.value_and_grad(self.model,loss_fn)(self.model)
        norm=mx.sqrt(sum(mx.sum(g*g) for _,g in tree_flatten(grad)))
        optimizer.update(self.model,grad);mx.eval(self.model.parameters(),optimizer.state,loss,norm)
        assert np.isfinite(loss.item()) and np.isfinite(norm.item())
        return dict(loss=loss.item(),gradient_norm=norm.item(),seconds=time.monotonic()-tick,
                    input_tokens=len(prefix)+len(y)-1,loss_tokens=len(y))
    def invariants(self):
        current=self.snapshot();self.restore(self.initial)
        delta=mx.max(mx.abs(self.logits(self.probe)-self.probe_logits)).item()
        unchanged=digest(self.base())==self.base_hash
        self.restore(current)
        assert delta==0 and unchanged
        return dict(base_unchanged=unchanged,reset_max_logit_delta=delta)

def resource():
    files={p.name:sha(p) for p in sorted(MODEL.glob('*')) if p.is_file()}
    previous=json.loads(Path('sources/model-reference.json').read_text())
    assert all(files.get(k)==v for k,v in previous['files'].items()), 'Checkpoint differs from prior native reference'
    return dict(model='Qwen/Qwen3-4B-Instruct-2507',revision=REVISION,files=files,prior_hashes_verified=True,
                packages={n:md.version(n) for n in ['mlx','mlx-lm','transformers','numpy']},
                mlx_lm_source=md.distribution('mlx-lm').read_text('direct_url.json'),
                reused_study_revision='08d8ec1b0b757e8a537a14d708bd895e8829ac68')
