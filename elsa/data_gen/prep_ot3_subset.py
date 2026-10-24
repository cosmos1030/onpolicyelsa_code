"""Matched-prompt subset of the SCOUT NTP corpus (OT3 80% + FineWeb 20%) for the boosting comparison.

Picks N OT3 chat docs that still contain their final answer (after </think>; 20k of the 32k were
truncated at 8192 tokens before it) plus N/4 FineWeb docs, with a fixed seed, and writes:
  ot3/prompts.parquet   problem (user turn), solution (OT3 final answer = boosting hint), ot_idx
  ot3/fw.jsonl          the FineWeb docs, untouched (shared by every arm)
  ot3/orig.jsonl        baseline arm: the same OT3 docs verbatim + fw.jsonl
"""
import json, random
import pandas as pd
SRC = '/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl'
N = 8000
ot, fw = [], []
with open(SRC) as f:
    for i, l in enumerate(f):
        t = json.loads(l)['text']
        if '<|im_start|>assistant\n' in t:
            u, a = t.split('<|im_start|>assistant\n', 1)
            if '</think>' in a and u.startswith('<|im_start|>user\n'):
                q = u[len('<|im_start|>user\n'):].rsplit('<|im_end|>', 1)[0]
                fin = a.split('</think>', 1)[1].replace('<|im_end|>', '').strip()
                if q.strip() and fin:
                    ot.append((i, q, fin, t))
        else:
            fw.append(t)
rng = random.Random(0)
sel = rng.sample(ot, N)
fws = rng.sample(fw, N // 4)
pd.DataFrame({'ot_idx': [s[0] for s in sel], 'problem': [s[1] for s in sel],
              'solution': [s[2] for s in sel]}).to_parquet('ot3/prompts.parquet')
with open('ot3/fw.jsonl', 'w') as f:
    for t in fws:
        f.write(json.dumps({'text': t}) + '\n')
with open('ot3/orig.jsonl', 'w') as f:
    for s in sel:
        f.write(json.dumps({'text': s[3]}) + '\n')
    for t in fws:
        f.write(json.dumps({'text': t}) + '\n')
print(f'OT with final answer: {len(ot)}, FW: {len(fw)} -> selected {N} OT + {len(fws)} FW')
