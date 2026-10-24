"""Verify against raw OpenThoughts3-1.2M (not our 40k subset): how many of the SCOUT pool's OT rows
have a closed </think> + final answer in the raw assistant turn, and the same rate over the whole 1.2M."""
import json, random
from datasets import load_dataset
ds = load_dataset('open-thoughts/OpenThoughts3-1.2M', split='train')
def has_ans(convs):
    if isinstance(convs, str):
        convs = json.loads(convs)
    a = [c['value'] for c in convs if c['from'] in ('gpt', 'assistant')]
    return bool(a) and '</think>' in a[-1] and a[-1].rsplit('</think>', 1)[1].strip() != ''
pool = ds.shuffle(seed=42).select(range(96000))
n = sum(has_ans(c) for c in pool.select(range(32000))['conversations'])
print(f'raw OT3, first 32000 of the seed-42 shuffle (SCOUT pool): with final answer {n}/32000')
idx = random.Random(0).sample(range(len(ds)), 20000)
m = sum(has_ans(c) for c in ds.select(idx)['conversations'])
print(f'raw OT3, random 20000 of all {len(ds)}: with final answer {m}/20000 -> est. {m / 20000 * len(ds):.0f} total')
