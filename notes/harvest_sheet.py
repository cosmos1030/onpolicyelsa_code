#!/usr/bin/env python3
"""Emit eval results in the spreadsheet layout:
seed | avg5 | {math500,GPQA,IFEval,LCB,GSM8K} x {accuracy, truncation rate,
correct truncation rate, avg tokens}.

Sources, in the order they are trusted:
  1. wandb run summary  (3-seed s3_* runs -> *_seed{0,1,42}; training runs -> bare keys)
  2. models/<dir>/eval_summary_resumed.json  (long evals launched with wandb id "-")
Nothing is retyped by hand.
"""
import json, os, statistics as st, sys
import wandb

os.environ['WANDB_API_KEY'] = open('/NHNHOME/log-postech/doyoonkim/secrets/wandb_api_key').read().strip()
ENT = 'dyk6208-gwangju-institute-of-science-and-technology'
P4, P8 = 'reasoning_qwen3_4b_nostrip8192', 'reasoning_qwen3_8b_nostrip8192'
MODELS = '/NHNHOME/log-postech/doyoonkim/models'

BENCH = [('math500', 'lighteval/math500'), ('gpqa', 'lighteval/gpqa_diamond'),
         ('ifeval', 'lighteval/ifeval_prompt'), ('lcb', 'lighteval/lcb'),
         ('gsm8k', 'lighteval/gsm8k')]
HDR_B = ['math500', 'GPQA', 'IFEval', 'LCB', 'GSM8K']
COLS = ['accuracy', 'truncation rate', 'correct truncation rate', 'avg tokens']

# label, source-kind, locator
WANDB_3SEED = [
    ('Qwen 3 8B dense',                      P8, 'xp6jch5n'),
    ('Qwen 3 4B s60 SparseGPT',              P4, 'rp0bi9rd'),
    ('Qwen 3 4B s60 ALPS',                   P4, 'lesvnnsg'),
    ('Qwen 3 4B s50 SparseGPT selfgen',      P4, 'pho3cqwf'),
]
WANDB_1SEED = [
    # vbchq36q (long_dense_8b) is the SAME Qwen3-8B weights as s3_8b_dense and is
    # deliberately not listed: it ran before e9b30c5 cut the long profile's gpqa
    # budget 32768 -> 16384, so its gpqa (58.59) is on a budget nothing else uses.
    # Its other four benches are the byte-identical cache s3_8b_dense seed42 reused.
    ('Qwen 3 8B s70 Ours w/o OPD (0.5/0.5/0)',       P8, 'g5htdr2q'),
    ('Qwen 3 8B s80 Ours (delta=0.03)',              P8, 'keuegrrb'),
    ('Qwen 3 8B s80 Ours (delta=0.02)',              P8, '3j2qqve1'),
    ('Qwen 3 8B s80 ALPS+retrain (0.33/0.33/0.33)',  P8, '9yyo7mmo'),
    ('Qwen 3 8B s80 ALPS+retrain w/o OPD (0.5/0.5/0)', P8, '4yuo2b36'),
    ('Qwen 3 4B s70 Ours w/o KD (0.5/0/0.5)',        P4, 'r8eomj61'),
    ('Qwen 3 4B s80 ALPS+retrain OPD-only (0/0/1.0)', P4, 'n9omc089'),
]
LOCAL = [
    ('Qwen 3 4B s70 Ours (long eval, control)', 'gmp_s70pct_lr0.0001_onpol_lmda0.33_20260901_064301'),
    ('Qwen 3 8B s80 Ours (long eval)',          'gmp_s80pct_lr0.0001_onpol_lmda0.33_20260916_200307_p43191'),
    ('Qwen 3 4B s80 Ours (long eval)',          'gmp_s80pct_lr0.0001_onpol_lmda0.33_20260916_204422_p39506'),
]

api = wandb.Api(timeout=120)


def cells(s, suf=''):
    """One seed's 21 cells: avg5 + 5 benches x 4 metrics. None where absent."""
    accs, out = [], []
    for b, akey in BENCH:
        a = s.get(akey + suf)
        tr = s.get(f'{b}_truncation_rate{suf}')
        ct = s.get(f'{b}_correct_truncation_rate{suf}')
        tk = s.get(f'{b}_avg_output_tokens{suf}')
        accs.append(a)
        out += [pct(a), pct(tr), pct(ct), tok(tk)]
    avg5 = round(sum(accs) * 100 / 5, 2) if all(isinstance(a, (int, float)) for a in accs) else ''
    return [avg5] + out


def pct(v):
    return round(100 * v, 2) if isinstance(v, (int, float)) else '-'


def tok(v):
    return round(v) if isinstance(v, (int, float)) else '-'


def header(label):
    r1 = [label] + [''] * 21
    r2 = ['', 'avg5'] + sum(([b] + [''] * 3 for b in HDR_B), [])
    r3 = ['seed', 'accuracy'] + COLS * 5
    return [r1, r2, r3]


def block(label, rows):
    out = header(label) + rows + [[''] * 22]
    return out


def agg(rows, fn):
    """Column-wise mean/std over seed rows, skipping '-' cells."""
    out = []
    for i in range(1, len(rows[0])):
        vals = [r[i] for r in rows if isinstance(r[i], (int, float))]
        if len(vals) < 2:
            out.append('-' if not vals else (round(vals[0], 2) if fn is st.mean else '-'))
            continue
        v = fn(vals)
        out.append(round(v) if i in TOKCOLS else round(v, 2))
    return out


TOKCOLS = {1 + 4 * j + 4 for j in range(5)}   # 'avg tokens' column index per bench
TOKCOLS = {2 + 4 * j + 3 for j in range(5)}   # avg5 at 1 -> bench j cols 2+4j .. 5+4j


def main():
    sheet = []
    for label, proj, rid in WANDB_3SEED:
        r = api.run(f'{ENT}/{proj}/{rid}')
        s = dict(r.summary)
        rows = []
        for sd in ['0', '1', '42']:
            if not any(k.endswith(f'_seed{sd}') for k in s):
                continue
            rows.append([sd] + cells(s, f'_seed{sd}'))
        note = '' if len(rows) == 3 else f'  [{len(rows)}/3 seeds -- {r.state}]'
        if len(rows) >= 2:
            rows.append(['mean'] + agg(rows, st.mean))
            rows.append(['std'] + agg(rows, lambda v: st.stdev(v)))
        sheet += block(label + note, rows)
        print(f'  ok  {label}{note}', file=sys.stderr)

    for label, proj, rid in WANDB_1SEED:
        r = api.run(f'{ENT}/{proj}/{rid}')
        sheet += block(label, [['42'] + cells(dict(r.summary))])
        print(f'  ok  {label}', file=sys.stderr)

    for label, d in LOCAL:
        p = f'{MODELS}/{d}/eval_summary_resumed.json'
        s = json.load(open(p))
        sheet += block(label, [['42'] + cells(s)])
        print(f'  ok  {label}  (local {d})', file=sys.stderr)

    out = '/NHNHOME/log-postech/doyoonkim/logs/RESULTS_SHEET_20260918.tsv'
    with open(out, 'w') as f:
        for row in sheet:
            f.write('\t'.join(str(c) for c in row) + '\n')
    print(f'\nwrote {out}  ({len(sheet)} rows)', file=sys.stderr)


main()
