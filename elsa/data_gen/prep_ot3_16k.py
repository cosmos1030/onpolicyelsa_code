"""Extend the boosting comparison set to 16,384 docs = exactly what 2048 steps x 8 samples consume (1 epoch).

13,108 OT + 3,276 FineWeb (80/20, as SCOUT's corpus). Keeps the 8,000 OT + 2,000 FW of prep_ot3_subset.py.
OT docs need a final answer (text after </think>) to serve as the boosting hint:
  - first, the remaining SCOUT-pool docs that have one (11,969 of the pool's 32,000 do; the other
    20,031 end inside <think> in raw OpenThoughts3 itself),
  - then the shortfall from OpenThoughts3 rows right after the pool in the same seed-42 shuffle
    (the order build_ot3_fineweb_dataset.py drew the pool from), rendered the same way.
Writes:
  ot3/prompts_extra.parquet  the new OT problems to generate (ot_idx = pool line, -1 = from beyond the pool)
  ot3/fw16k.jsonl            3,276 FineWeb docs (the 2,000 of fw.jsonl first)
  ot3/orig16k.jsonl          baseline arm: all 13,108 OT docs verbatim + fw16k.jsonl
"""
import json, random, sys
import pandas as pd
sys.path.insert(0, '/home1/doyoonkim/projects/elsa/scripts')
from build_ot3_fineweb_dataset import _manual_render, _ROLE_MAP

SRC = '/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl'
N_OT, N_FW = 13108, 3276
POOL_DRAW = 96000  # build_ot3_fineweb_dataset.py: shuffle(seed=42).select(range(n_ot * 3)) for n_ot=32000


def split_doc(t):
    if '<|im_start|>assistant\n' not in t:
        return None
    u, a = t.split('<|im_start|>assistant\n', 1)
    if '</think>' not in a or not u.startswith('<|im_start|>user\n'):
        return None
    q = u[len('<|im_start|>user\n'):].rsplit('<|im_end|>', 1)[0]
    fin = a.split('</think>', 1)[1].replace('<|im_end|>', '').strip()
    return (q, fin) if q.strip() and fin else None


ot, fw, pool_q = [], [], set()
with open(SRC) as f:
    for i, l in enumerate(f):
        t = json.loads(l)['text']
        if '<|im_start|>assistant\n' in t:
            pool_q.add(t.split('<|im_start|>assistant\n', 1)[0])
            s = split_doc(t)
            if s:
                ot.append((i, s[0], s[1], t))
        else:
            fw.append(t)
base = pd.read_parquet('ot3/prompts.parquet')
have = set(base.ot_idx)
rest = [o for o in ot if o[0] not in have]
print(f'pool OT with answer {len(ot)}, already selected {len(have)}, remaining {len(rest)}')
extra = rest[:]
need = N_OT - len(have) - len(extra)
if need > 0:
    from datasets import load_dataset
    ds = load_dataset('open-thoughts/OpenThoughts3-1.2M', split='train').shuffle(seed=42)
    j = POOL_DRAW
    while need > 0:
        convs = ds[j]['conversations']
        j += 1
        if isinstance(convs, str):
            convs = json.loads(convs)
        try:
            t = _manual_render([{'role': _ROLE_MAP.get(c['from'], c['from']), 'content': c['value']} for c in convs])
        except Exception:
            continue
        if t.split('<|im_start|>assistant\n', 1)[0] in pool_q:
            continue
        s = split_doc(t)
        if s:
            extra.append((-1, s[0], s[1], t))
            need -= 1
    print(f'drew {sum(e[0] == -1 for e in extra)} from beyond the pool (scanned rows {POOL_DRAW}..{j})')
assert len(have) + len(extra) == N_OT
pd.DataFrame({'ot_idx': [e[0] for e in extra], 'problem': [e[1] for e in extra],
              'solution': [e[2] for e in extra]}).to_parquet('ot3/prompts_extra.parquet')

fw_old = [json.loads(l)['text'] for l in open('ot3/fw.jsonl')]
old = set(fw_old)
fw_new = random.Random(1).sample([t for t in fw if t not in old], N_FW - len(fw_old))
fw16 = fw_old + fw_new
with open('ot3/fw16k.jsonl', 'w') as f:
    for t in fw16:
        f.write(json.dumps({'text': t}) + '\n')

src_lines = open(SRC).readlines()
with open('ot3/orig16k.jsonl', 'w') as f:
    for i in base.ot_idx:
        f.write(json.dumps({'text': json.loads(src_lines[i])['text']}) + '\n')
    for e in extra:
        f.write(json.dumps({'text': e[3]}) + '\n')
    for t in fw16:
        f.write(json.dumps({'text': t}) + '\n')
print(f'wrote {len(extra)} extra prompts; orig16k = {N_OT} OT + {len(fw16)} FW')
