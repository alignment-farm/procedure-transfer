"""Audit diagnostic artifacts and replay one recall case per saved adapter.

Replays verify persistence; they are not additional evaluation observations.
"""
import argparse
import collections
import json
import re
import time
from pathlib import Path

import mlx.core as mx
from runtime import Runtime, digest, sha
from task import SCHEMA, REMINDER, cases, query, oracle


def readlines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def parts(raw, case):
    match = re.fullmatch(r'(kestrel|marten)\(text="([^"\n]*)"\)', raw.strip())
    if not match:
        return dict(format=False, route=False, identifier=False, suffix=False)
    tool, arg = match.groups()
    body = arg[:-2] if arg.endswith('-Q') else arg
    return dict(format=True, route=tool == {'copper': 'kestrel', 'violet': 'marten'}[case['channel']],
                identifier=body == case['identifier'].upper(),
                suffix=arg.endswith('-Q') == (case['priority'] == 'fast'))


p = argparse.ArgumentParser()
p.add_argument('run', type=Path)
a = p.parse_args()
run = a.run
out = Path(str(run) + '-audit')
out.mkdir(exist_ok=False)
(out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
started = time.monotonic()
bundles = [run, Path(str(run) + '-rescue'), Path(str(run) + '-components'),
           Path(str(run) + '-gradients'), Path('evidence/transfer-v1-gradients')]
hash_count = 0
for bundle in bundles:
    listed = set()
    for line in (bundle / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        assert sha(bundle / name) == expected, (bundle, name)
        listed.add(name)
        hash_count += 1
    assert listed == {f.name for f in bundle.iterdir() if f.name != 'SHA256SUMS'}

rt = Runtime()
data = json.loads((run / 'tokens.json').read_text())
train = cases(['pelk', 'druvan', 'snebit', 'korvaz'])
dev = cases(['halmek', 'przuna', 'veskolit', 'dunfepazor'])
assert [r['case'] for r in data] == train
context = REMINDER + 'Checked successful calls:\n' + '\n'.join(query(c) + ' -> ' + oracle(c) for c in train)
for r in data:
    assert r['p'] == rt.encode(SCHEMA + '\n\n' + query(r['case']))
    assert r['tp'] == rt.encode(SCHEMA + '\n' + context + '\n' + query(r['case']))
    assert r['y'] == rt.target(r['p'], oracle(r['case']))

component_design = json.loads((bundles[2] / 'design.json').read_text())
new_cases = component_design['cases']
old_cases = json.loads(Path('evidence/transfer-v1/evaluation.json').read_text())
new_words = {c['identifier'] for c in new_cases}
assert len(new_words) == 12 and len(new_cases) == 48
assert collections.Counter(map(len, new_words)) == {4: 3, 6: 3, 8: 3, 10: 3}
assert new_cases == cases(list(dict.fromkeys(c['identifier'] for c in new_cases)))
assert not new_words.intersection(c['identifier'] for c in train + dev + old_cases)
assert not new_words.intersection(['nup', 'zelk', 'bimav', 'fosted', 'wexi', 'jupnal', 'votkeris', 'zunpelavik'])

summaries = []
replays = []
costs = []
record_count = 0
order = json.loads((run / 'order.json').read_text())
for offset in range(0, 256, 16):
    assert sorted(order[offset:offset + 16]) == list(range(16))
for bundle in bundles[:3]:
    rows = readlines(bundle / 'responses.jsonl')
    is_component = bundle == bundles[2]
    groups = collections.defaultdict(list)
    for r in rows:
        c = r['case']
        assert r['expected'] == oracle(c)
        assert r['correct'] == (r['raw'].strip() == oracle(c))
        assert r['action'] == r['raw'].strip()
        assert bool(r['ids'][-1] in rt.tokenizer.eos_token_ids) == r['ended']
        assert not any(t in rt.tokenizer.eos_token_ids for t in r['ids'][:-1])
        decoded = rt.tokenizer.decode(r['ids'][:-1] if r['ended'] else r['ids'])
        if is_component:
            expected_prefix = rt.encode(SCHEMA + '\n' + (context if r['state'] == 'teacher' else '') + '\n' + query(c))
            assert r['base_prefix'] == expected_prefix
            assert rt.tokenizer.decode(r['supplied_ids']) == r['supplied']
            assert r['raw'] == r['supplied'] + decoded
            assert r['continuation'] == decoded
            assert r['prompt_tokens'] == len(expected_prefix) + len(r['supplied_ids'])
            assert oracle(c).startswith(r['supplied'])
            key = (r['state'], r['condition'])
        else:
            assert r['prefix'] == rt.encode(SCHEMA + '\n\n' + query(c))
            assert r['prompt_tokens'] == len(r['prefix']) and decoded == r['raw']
            key = (r['arm'], r.get('step'), r['suite'])
        assert r['completion_tokens'] == len(r['ids'])
        if 'components' in r:
            assert r['components'] == parts(r['raw'], c)
        groups[key].append(r)
        record_count += 1
    for key, group in groups.items():
        expected_cases = new_cases if is_component else train if key[-1] == 'recall' else dev
        assert [r['case'] for r in group] == expected_cases
        scores = [parts(r['raw'], r['case']) for r in group]
        summaries.append(dict(bundle=bundle.name, group=key, total=len(group),
                              correct=sum(r['correct'] for r in group),
                              components={k: sum(s[k] for s in scores) for k in scores[0]}))
    costs.append(dict(bundle=bundle.name, generation_calls=len(rows),
                      prompt_tokens=sum(r['prompt_tokens'] for r in rows),
                      completion_tokens=sum(r['completion_tokens'] for r in rows),
                      generation_seconds=sum(r['seconds'] for r in rows)))
    if is_component:
        assert len(rows) == 336
        continue
    events = readlines(bundle / 'events.jsonl')
    assert events[-1]['kind'] == 'complete' and events[-1]['status'] == 'complete'
    arms = sorted({e['arm'] for e in events if e['kind'] == 'update'})
    assert len(arms) == (5 if bundle == run else 3)
    for arm in arms:
        updates = [e for e in events if e['kind'] == 'update' and e['arm'] == arm]
        n = 256 if bundle == run else 128
        assert [e['step'] for e in updates] == list(range(1, n + 1))
        assert [e['index'] for e in updates] == order[:n]
    assert len(rows) == (512 if bundle == run else 96)
    for e in events:
        if e['kind'] == 'evaluation':
            for suite in ['recall', 'development']:
                if 'suite' in e and e['suite'] != suite:
                    continue
                group = groups[(e['arm'], e.get('step'), suite)]
                assert sum(r['correct'] for r in group) == e.get(suite, e.get('correct'))
    starts = [e for e in events if e['kind'] in ['arm_start', 'start']]
    if bundle == run:
        assert len({e['initial_hash'] for e in starts}) == 1
    else:
        assert starts[0]['initial_hash'] == starts[1]['initial_hash']
        for e in starts:
            assert digest(list(mx.load(str(run / (e['source_checkpoint'] + '.safetensors'))).items())) == e['initial_hash']
    for checkpoint in sorted(bundle.glob('*.safetensors')):
        if bundle == run:
            arm, step = checkpoint.stem.rsplit('-', 1)
            original = groups[(arm, int(step), 'recall')][0]
        else:
            original = groups[(checkpoint.stem, None, 'recall')][0]
        state = list(mx.load(str(checkpoint)).items())
        if bundle != run:
            assert digest(state) == next(e['hash'] for e in events if e['kind'] == 'trained' and e['arm'] == checkpoint.stem)
        rt.restore(state)
        repeat = rt.generate(original['prefix'])
        assert repeat['ids'] == original['ids'], checkpoint
        replays.append(dict(checkpoint=str(checkpoint), exact_tokens=True, ids=repeat['ids']))
    costs[-1].update(updates=sum(e['kind'] == 'update' for e in events), seconds=events[-1]['seconds'],
                     peak_mlx_bytes=events[-1]['peak_mlx_bytes'])

teacher_audit = json.loads((run / 'teacher-audit.json').read_text())
assert len(teacher_audit) == 16
for r, d in zip(teacher_audit, data):
    assert r['case'] == d['case'] and r['canonical_y'] == d['y']
    assert r['correct'] == (r['teacher']['raw'].strip() == oracle(r['case']))
    assert r['canonical_equals_generated'] == (r['teacher']['ids'] == d['y'])
report = dict(status='passed', files_hashed=hash_count, response_records_checked=record_count,
              teacher_records_checked=len(teacher_audit), summaries=summaries, costs=costs,
              replays=replays, repeated_cases_are_not_new_evaluation_evidence=True,
              seconds=time.monotonic() - started, **rt.invariants())
(out / 'audit.json').write_text(json.dumps(report, indent=2))
(out / 'SHA256SUMS').write_text('\n'.join(sha(f) + '  ' + f.name for f in sorted(out.iterdir()) if f.name != 'SHA256SUMS'))
print(json.dumps({k: v for k, v in report.items() if k not in ['summaries', 'replays']}, indent=2))
