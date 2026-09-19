#!/usr/bin/env python3
"""Long-profile eval results -> one spreadsheet-layout TSV per model size.

    python long_tsv_results/harvest_long_tsv.py      # needs WANDB_API_KEY

Writes long_tsv_results/{1.7b,4b,8b}/long_results_<size>.tsv. Re-runnable from
any server: the only source is wandb, nothing is retyped by hand.

ONLY the long profile goes in. wandb config does not record the profile, so a
(run, seed) is admitted per benchmark by the generation cap it actually ran
with, which is logged next to every score (lighteval_bench._compute_token_stats):

    math500 / gpqa / ifeval = 16384,  lcb = 32768,  gsm8k = 8192

quick (8192-prompt / 2048) and official (32768-prompt) can never produce these.
A seed with ANY benchmark at another cap is mixed-profile and dropped whole;
a benchmark that is merely missing (crashed / still running) stays '-'. The
pre-e9b30c5 long runs (long_dense_*) used gpqa 32768 and fall out here on
purpose -- that gpqa budget is not what the rest of the table uses.

Layout (same as notes/harvest_sheet.py): per block, seed rows then mean/std;
seed | avg5 | {math500,GPQA,IFEval,LCB,GSM8K} x {accuracy, truncation rate,
correct truncation rate, avg tokens}.
"""
import os, re, statistics as st, sys
import wandb

ENT = 'dyk6208-gwangju-institute-of-science-and-technology'
PROJ = {'1.7b': 'reasoning_qwen3_1.7b_nostrip8192',
        '4b': 'reasoning_qwen3_4b_nostrip8192',
        '8b': 'reasoning_qwen3_8b_nostrip8192'}
HERE = os.path.dirname(os.path.abspath(__file__))

BENCH = [('math500', 'lighteval/math500', 16384),
         ('gpqa', 'lighteval/gpqa_diamond', 16384),
         ('ifeval', 'lighteval/ifeval_prompt', 16384),
         ('lcb', 'lighteval/lcb', 32768),
         ('gsm8k', 'lighteval/gsm8k', 8192)]
HDR_B = ['math500', 'GPQA', 'IFEval', 'LCB', 'GSM8K']
COLS = ['accuracy', 'truncation rate', 'correct truncation rate', 'avg tokens']
SUFFIXES = ['_seed0', '_seed1', '_seed42', '']

# Training runs whose long eval was resumed into the training run itself: the
# run name says nothing readable, so they are labelled here. s3_* eval runs are
# labelled from their names (auto_label); a label here overrides that.
LABELS = {
    # 4B
    'r8eomj61': 'Ours w/o KD (0.5/0/0.5)',
    'q6vjas4j': 'ALPS+retrain',
    'crdshw6m': 'ALPS+retrain w/o OPD',
    'n9omc089': 'ALPS+retrain OPD-only (0/0/1.0)',
    '9s01x120': 'Ours + DPO (lr1e-5, ep0.4)',
    'vv5x9gx3': 'Ours + DPO (lr5e-6, ep1.0)',
    # 8B
    'g5htdr2q': 'Ours w/o OPD (0.5/0.5/0)',
    'keuegrrb': 'Ours (delta=0.03)',
    '9yyo7mmo': 'ALPS+retrain (0.33/0.33/0.33)',
    '4yuo2b36': 'ALPS+retrain w/o OPD (0.5/0.5/0)',
}
METHOD = {'dense': 'dense', 'sparsegpt': 'SparseGPT', 'sgpt_selfgen': 'SparseGPT selfgen',
          'alps': 'ALPS', 'alps_selfgen': 'ALPS selfgen', 'alpsretrain': 'ALPS+retrain',
          'ours': 'Ours'}
ORDER = ['dense', 'SparseGPT', 'SparseGPT selfgen', 'ALPS', 'ALPS selfgen',
         'ALPS+retrain', 'Ours']


def sparsity(run):
    m = re.search(r'(?:^|_)s(\d\d)(?:pct|_|$)', run.name)
    return int(m.group(1)) if m else 0


def auto_label(run):
    if run.id in LABELS:
        return LABELS[run.id]
    n = re.sub(r'_s42$', '', run.name)
    n = re.sub(r'^s3_[\d.]+b_', '', n)
    n = re.sub(r'(^|_)s\d\d(?=_|$)', '', n).strip('_')
    for k in sorted(METHOD, key=len, reverse=True):
        if n == k:
            return METHOD[k]
    return run.name


def pct(v):
    return 100 * v if isinstance(v, (int, float)) and v == v else '-'


def tok(v):
    return float(v) if isinstance(v, (int, float)) and v == v else '-'


def fmt(i, v):
    """Round only when writing, so mean/std come from unrounded values."""
    if not isinstance(v, float):
        return str(v)
    return str(round(v)) if i in TOKCOLS else str(round(v, 2))


def is_long(s, b, cap, suf):
    c = s.get(f'{b}_avg_gen_cap{suf}')
    return isinstance(c, (int, float)) and round(c) == cap


def mixed(s, suf):
    return any(isinstance(s.get(f'{b}_avg_gen_cap{suf}'), (int, float))
               and not is_long(s, b, cap, suf) for b, _, cap in BENCH)


def cells(s, suf):
    """One seed's 21 cells; a bench not run at its long cap stays '-'."""
    accs, out = [], []
    for b, akey, cap in BENCH:
        if not is_long(s, b, cap, suf):
            accs.append(None)
            out += ['-'] * 4
            continue
        a = s.get(akey + suf)
        accs.append(a)
        out += [pct(a), pct(s.get(f'{b}_truncation_rate{suf}')),
                pct(s.get(f'{b}_correct_truncation_rate{suf}')),
                tok(s.get(f'{b}_avg_output_tokens{suf}'))]
    avg5 = sum(accs) * 100 / 5 if all(isinstance(a, (int, float)) for a in accs) else '-'
    return [avg5] + out


TOKCOLS = {2 + 4 * j + 3 for j in range(5)}   # 'avg tokens' columns (avg5 at 1)


def agg(rows, fn):
    out = []
    for i in range(1, len(rows[0])):
        vals = [r[i] for r in rows if isinstance(r[i], (int, float))]
        if len(vals) < 2:
            out.append('-')
            continue
        out.append(float(fn(vals)))
    return out


def header(label, rid):
    r1 = [label] + [''] * 20 + [f'wandb {rid}']
    r2 = ['', 'avg5'] + sum(([b] + [''] * 3 for b in HDR_B), [])
    r3 = ['seed', 'accuracy'] + COLS * 5
    return [r1, r2, r3]


def expected_seeds(run):
    try:
        a = (run.metadata or {}).get('args', [])
    except Exception:
        a = []
    if '--seeds' in a:
        return len(a[a.index('--seeds') + 1].split(','))
    return 1


def harvest(api, size, proj):
    f = {'$or': [{f'summary_metrics.{b}_avg_gen_cap{x}': {'$exists': True}}
                 for b in ('math500', 'gsm8k') for x in SUFFIXES]}
    blocks = []
    for r in api.runs(f'{ENT}/{proj}', filters=f, per_page=200):
        s = r.summary._json_dict
        rows = []
        for suf in SUFFIXES:
            if suf == '' and rows:
                break
            if mixed(s, suf) or not any(is_long(s, b, cap, suf) for b, _, cap in BENCH):
                continue
            seed = suf.replace('_seed', '') or '42'
            rows.append([seed] + cells(s, suf))
        if not rows:
            continue
        label = f'Qwen 3 {size.upper()} ' + (f's{sparsity(r)} ' if sparsity(r) else '') + auto_label(r)
        full = [x for x in rows if x[1] != '-']
        want = expected_seeds(r)
        if len(full) < want or len(full) < len(rows):
            label += f'  [{len(full)}/{want} seeds complete -- {r.state}]'
        elif r.state != 'finished':
            label += f'  [{r.state}]'
        if len(rows) >= 2:
            rows = rows + [['mean'] + agg(rows, st.mean), ['std'] + agg(rows, st.stdev)]
        meth = auto_label(r)
        rank = next((i for i, m in enumerate(ORDER) if meth.startswith(m) and
                     not any(meth.startswith(m2) and len(m2) > len(m) for m2 in ORDER)), len(ORDER))
        blocks.append(((sparsity(r), rank, meth, r.created_at),
                       header(label, r.id) + rows + [[''] * 22]))
        print(f'  {size:>4} {r.id} {label}', file=sys.stderr)
    blocks.sort(key=lambda x: x[0])
    return [row for _, b in blocks for row in b]


def main():
    api = wandb.Api(timeout=180)
    for size, proj in PROJ.items():
        sheet = harvest(api, size, proj)
        os.makedirs(f'{HERE}/{size}', exist_ok=True)
        out = f'{HERE}/{size}/long_results_{size}.tsv'
        with open(out, 'w') as fh:
            for row in sheet:
                fh.write('\t'.join(fmt(i, c) for i, c in enumerate(row)) + '\n')
        print(f'wrote {out}  ({len(sheet)} rows)', file=sys.stderr)


if __name__ == '__main__':
    main()
