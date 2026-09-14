"""Fresh synthetic routing contract; every condition combination is observed."""
import itertools
import random
import string

SCHEMA = ('Return exactly one tool call, with no explanation or Markdown. '
          'Available calls: kestrel(text="...") and marten(text="..."). '
          'The input has channel, priority, and a lowercase ASCII identifier. '
          'Apply the established routing and argument protocol.')
RULE = ('Channel copper selects kestrel; channel violet selects marten. '
        'Uppercase the identifier. If priority is fast, append -Q after uppercasing; '
        'if priority is slow, append nothing. These operations apply independently '
        'for every identifier and combination.')
TRAIN_WORDS = ['nup', 'zelk', 'bimav', 'fosted']
DEV_WORDS = ['wexi', 'jupnal', 'votkeris', 'zunpelavik']

def cases(words):
    return [dict(channel=c, priority=p, identifier=w) for w in words
            for c,p in itertools.product(['copper','violet'], ['slow','fast'])]

def query(c):
    return f'channel={c["channel"]}; priority={c["priority"]}; identifier={c["identifier"]}'

def oracle(c):
    tool = {'copper':'kestrel', 'violet':'marten'}[c['channel']]
    arg = c['identifier'].upper() + ('-Q' if c['priority']=='fast' else '')
    return f'{tool}(text="{arg}")'

def context():
    return 'Checked successful calls:\n' + '\n'.join(query(c)+' -> '+oracle(c) for c in cases(TRAIN_WORDS))

def content(c, branch='none'):
    extra = {'none':'', 'examples':context(), 'rule':RULE}[branch]
    return SCHEMA+'\n'+extra+'\n'+query(c)

def evaluation(seed=2026091401):
    rng=random.Random(seed); used=set(TRAIN_WORDS+DEV_WORDS); words=[]
    for length in [4,6,8,10]:
        for _ in range(6):
            while True:
                w=''.join(rng.choice(string.ascii_lowercase) for _ in range(length))
                if w not in used: break
            words.append(w);used.add(w)
    return cases(words)
