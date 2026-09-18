import sys, glob, os, json
sys.path.insert(0, '/NHNHOME/log-postech/doyoonkim/onpolicyelsa_code/elsa')
from lib.lighteval_bench import _compute_token_stats, _LONG_BENCHMARKS
SPEC = {b[0]: (b[1], b[2], b[3], b[4]) for b in _LONG_BENCHMARKS}
M = '/NHNHOME/log-postech/doyoonkim/models'
HDR_B = ['math500', 'GPQA', 'IFEval', 'LCB', 'GSM8K']
COLS = ['accuracy', 'truncation rate', 'correct truncation rate', 'avg tokens']
RUNS = [
    ('Qwen 3 4B s70 Ours w/o NTP (0/0.5/0.5)  [중단 -- LCB/GSM8K 미완]',
     f'{M}/gmp_s70pct_lr0.0001_onpol_lmda0.5_20260916_204154_p39636'),
    ('Qwen 3 8B s80 ALPS+retrain OPD-only (0/0/1.0)  [중단 -- LCB/GSM8K 미완]',
     f'{M}/gmp_8b_s80pct_lr0.0001_onpol_lmda1.0_20260918_052004_p1231854'),
]
def acc_of(out_dir, task, keys):
    js = sorted(glob.glob(os.path.join(out_dir, '**', 'results_*.json'), recursive=True))
    if not js: return None
    r = json.load(open(js[-1])).get('results', {})
    t = r.get(task) or r.get('all') or {}
    for k in keys + ['pass@k:k=1&n=1', 'extractive_match', 'acc']:
        if k in t: return t[k]
    return None
def pc(v): return round(100*v, 2) if isinstance(v, float) and v == v else '-'
out_rows = []
for label, d in RUNS:
    row = ['42', '']
    for b in ['math500', 'gpqa', 'ifeval', 'lcb', 'gsm8k']:
        task, mnt, mml, keys = SPEC[b]
        od = os.path.join(d, 'lighteval_bench', b)
        if b in ('lcb', 'gsm8k') or not os.path.isdir(od):
            row += ['-'] * 4; continue
        s = _compute_token_stats(od, b, mnt, keys, mml)
        a = acc_of(od, task, keys)
        row += [pc(a) if a is not None else '-', pc(s.get(b+'_truncation_rate')),
                pc(s.get(b+'_correct_truncation_rate')),
                round(s[b+'_avg_output_tokens']) if b+'_avg_output_tokens' in s else '-']
    out_rows += [[label]+['']*21,
                 ['', 'avg5'] + sum(([x]+['']*3 for x in HDR_B), []),
                 ['seed', 'accuracy'] + COLS*5, row, ['']*22]
p = '/NHNHOME/log-postech/doyoonkim/logs/RESULTS_SHEET_20260918.tsv'
with open(p, 'a') as f:
    for r in out_rows: f.write('\t'.join(str(c) for c in r) + '\n')
print('appended', len(out_rows), 'rows ->', p)
