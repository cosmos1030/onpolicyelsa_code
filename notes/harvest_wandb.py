#!/usr/bin/env python3
"""Harvest baselines that exist ONLY in wandb, not in this container's logs.

Why this exists: many baselines were run on the log_cluster (H200, host n92)
side, so /NHNHOME/.../logs has no trace of them.  Reading only local logs led to
the false conclusion that "there is no 4B ALPS+retrain baseline"; there are several,
including the 2:4 one.  harvest_results.py covers local logs; this covers wandb.

Writes RESULTS_WANDB_BASELINES.tsv next to it.  Re-runnable.
"""
import os, sys
import wandb

ENT = 'dyk6208-gwangju-institute-of-science-and-technology'
OUT = '/NHNHOME/log-postech/doyoonkim/logs/RESULTS_WANDB_BASELINES.tsv'
BENCH = [('math500', 'lighteval/math500'), ('gpqa', 'lighteval/gpqa_diamond'),
         ('ifeval', 'lighteval/ifeval_prompt'), ('lcb', 'lighteval/lcb'),
         ('gsm8k', 'lighteval/gsm8k')]

# (project, display-name regex) pairs worth pulling as baselines
QUERIES = [
    ('reasoning_qwen3_4b_nostrip8192', '(?i)alpssft'),
    ('reasoning_qwen3_8b_nostrip8192', '(?i)alpssft'),
    ('reasoning_qwen3_1.7b_nostrip8192', '(?i)alpssft'),
]


def main():
    api = wandb.Api(timeout=90)
    rows = []
    for proj, rx in QUERIES:
        try:
            runs = list(api.runs(f'{ENT}/{proj}',
                                 filters={'display_name': {'$regex': rx}}))
        except Exception as e:
            print(f'!! {proj}: {e}', file=sys.stderr)
            continue
        for r in runs:
            if r.state != 'finished':
                continue
            s = dict(r.summary)
            sc = {k: s.get(mk) for k, mk in BENCH}
            if any(not isinstance(v, (int, float)) for v in sc.values()):
                continue                      # incomplete eval -> no Avg, skip
            c = r.config
            rows.append({
                'proj': proj, 'id': r.id, 'name': r.name,
                'model': ('4B' if '_4b' in proj else
                          '8B' if '_8b' in proj else '1.7B'),
                'type': str(c.get('sparsity_type', '?')),
                'sp': c.get('sparsity_ratio', ''),
                'lr': c.get('lr', ''),
                'tok': c.get('gmp_onpolicy_max_new_tokens', ''),
                'ro': c.get('gmp_onpolicy_kd_interval', ''),
                'fsdp': c.get('gmp_use_fsdp', ''),
                'created': r.created_at,
                'avg': 100 * sum(sc.values()) / 5,
                **{k: 100 * v for k, v in sc.items()},
            })
    cols = ['model', 'type', 'sp', 'lr', 'tok', 'ro', 'fsdp', 'avg',
            'math500', 'gpqa', 'ifeval', 'lcb', 'gsm8k', 'created', 'id',
            'proj', 'name']
    rows.sort(key=lambda x: (x['model'], x['type'], str(x['sp']), -x['avg']))
    with open(OUT, 'w') as f:
        f.write('\t'.join(cols) + '\n')
        for r in rows:
            SCORE = {'avg', 'math500', 'gpqa', 'ifeval', 'lcb', 'gsm8k'}
            f.write('\t'.join(
                ('%.2f' % r[c]) if c in SCORE else
                ('%g' % r[c] if isinstance(r[c], float) else str(r[c]))
                for c in cols) + '\n')
    print('wrote %s : %d finished baseline runs' % (OUT, len(rows)))


if __name__ == '__main__':
    main()
