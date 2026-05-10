import sys
sys.path.insert(0, '/home1/doyoonkim/projects/elsa')
import statistics
from transformers import AutoTokenizer
from lib.gkd_admm_trainer import MathCotKDDataset

MODEL = '/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca'
DATA  = '/home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl'

tokenizer = AutoTokenizer.from_pretrained(MODEL)
ds = MathCotKDDataset(DATA, tokenizer, max_prompt_len=512, max_len=2048)

eos_id      = tokenizer.eos_token_id
think_close = tokenizer.encode('</think>', add_special_tokens=False)
im_end      = tokenizer.encode('<|im_end|>', add_special_tokens=False)

print(f'eos_token_id={eos_id}', flush=True)
print(f'</think> tokens={think_close}', flush=True)
print(f'<|im_end|> tokens={im_end}', flush=True)
print(f'Total samples: {len(ds)}', flush=True)

n = min(5000, len(ds))
stats = dict(total=0, truncated=0, has_eos=0, has_think_close=0, has_im_end=0, has_final_ans=0)
sup_lens = []

for i in range(n):
    item = ds[i]
    ids    = item['input_ids']
    labels = item['labels']
    sup = ids[labels != -100]
    sup_list = sup.tolist()
    stats['total'] += 1
    sup_lens.append(len(sup_list))

    if len(ids) >= 2047:
        stats['truncated'] += 1
    if eos_id in sup_list:
        stats['has_eos'] += 1
    for j in range(len(sup_list) - len(think_close) + 1):
        if sup_list[j:j+len(think_close)] == think_close:
            stats['has_think_close'] += 1
            break
    for j in range(len(sup_list) - len(im_end) + 1):
        if sup_list[j:j+len(im_end)] == im_end:
            stats['has_im_end'] += 1
            break
    decoded = tokenizer.decode(sup_list, skip_special_tokens=False)
    if r'\boxed' in decoded:
        stats['has_final_ans'] += 1

print(f'\nsup_len  median={statistics.median(sup_lens):.0f}  mean={sum(sup_lens)/len(sup_lens):.0f}  max={max(sup_lens)}', flush=True)
print()
for k, v in stats.items():
    pct = 100*v/stats['total'] if stats['total'] > 0 else 0
    print(f'  {k}: {v}/{stats["total"]} ({pct:.1f}%)', flush=True)
