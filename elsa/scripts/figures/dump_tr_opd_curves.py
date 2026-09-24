#!/usr/bin/env python3
"""TR x OPD 그림용 학습 곡선을 wandb 에서 CSV 로 덤프한다.

출력: elsa/figures/plotdata/tr_opd/{run}_{wandb_id}.csv, all_runs_tidy.csv, RUNS.md

키를 하나씩 따로 받는 것이 핵심이다. wandb 의 scan_history(keys=[...]) 는 지정한
키가 **모두** 있는 행만 돌려주므로, 여러 키를 한 번에 요청하면 늦게 시작하는
pgd/* 때문에 앞 구간이 통째로 잘린다. 그것 때문에 4B jump 런이 "점프를 하지
않았다"고 잘못 읽을 뻔했다 (실제로는 resume 된 런이라 로그가 step 513 부터였다).

    python elsa/scripts/figures/dump_tr_opd_curves.py
"""
import csv
import os

import wandb

E = 'dyk6208-gwangju-institute-of-science-and-technology'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, '..', '..', 'figures', 'plotdata', 'tr_opd'))
P4 = 'reasoning_qwen3_4b_nostrip8192'
P8 = 'reasoning_qwen3_8b_nostrip8192'

RUNS = [
    ('4b_scout',      P4, 'r5j1uw8d', 'TR + grow2target, klb=0.02, step 1-2048'),
    ('4b_jump',       P4, 'hkshqtev', 'one-step jump, klb=0.02, step 1-748 (crashed; 3olz82te 가 513- 이어받음)'),
    ('4b_jump_tail',  P4, '3olz82te', 'one-step jump 이어받은 런, step 513-2048'),
    ('8b_scout',      P8, '0z5sc32y', 'TR + grow2target, klb=0.03, step 1-520 (crashed; 4ebzii1b -> ajm5l60w)'),
    ('8b_scout_mid',  P8, '4ebzii1b', 'SCOUT 체인 2번째, step 513-1469'),
    ('8b_scout_tail', P8, 'ajm5l60w', 'SCOUT 체인 3번째, step 1281-2048 (최종 체크포인트)'),
    ('8b_jump',       P8, 'iuwf08pf', 'one-step jump, klb=0.03, step 1-2048 (끊김 없음)'),
]
KEYS = [
    'train/sparsity', 'train/target_sparsity', 'train/grad_norm', 'train/loss',
    'train/ntp_loss', 'train/onpolicy_kd_loss', 'train/aux_loss', 'train/lr',
    'onpolicy/kl_loss', 'onpolicy/overlap_ratio_top100', 'onpolicy/entropy_gap',
    'onpolicy/student_entropy', 'onpolicy/teacher_entropy', 'onpolicy/gen_tokens',
    'pgd/kl_at_full_pgd', 'pgd/kl_at_k_actual', 'pgd/kl_budget', 'pgd/k_actual',
    'pgd/post_sparsity', 'pgd/prunings', 'pgd/revivals', 'pgd/turnover',
    'pgd/net_growth', 'pgd/n_prune_cand', 'pgd/n_revive_cand',
]


def main():
    os.makedirs(OUT, exist_ok=True)
    api = wandb.Api(timeout=600)
    tidy = []
    for label, proj, rid, note in RUNS:
        r = api.run(f'{E}/{proj}/{rid}')
        cols = {}
        for k in KEYS:
            s = {x['step']: x[k] for x in r.scan_history(keys=['step', k])
                 if isinstance(x.get(k), (int, float))}
            if s:
                cols[k] = s
        steps = sorted({st for s in cols.values() for st in s})
        path = os.path.join(OUT, f'{label}_{rid}.csv')
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['step'] + list(cols))
            for st in steps:
                w.writerow([st] + [cols[k].get(st, '') for k in cols])
        print(f'  {label:14s} {rid}  {len(steps):5d} steps, {len(cols):2d} keys')
        for st in steps:
            for k, s in cols.items():
                if st in s:
                    tidy.append((label, rid, st, k, s[st]))

    with open(os.path.join(OUT, 'all_runs_tidy.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'wandb_id', 'step', 'metric', 'value'])
        w.writerows(tidy)
    with open(os.path.join(OUT, 'RUNS.md'), 'w') as f:
        f.write('# TR x OPD 곡선 — 런 출처\n\n| 파일 | wandb id | 설명 |\n|---|---|---|\n')
        for label, proj, rid, note in RUNS:
            f.write(f'| `{label}_{rid}.csv` | [{rid}](https://wandb.ai/{E}/{proj}/runs/{rid}) | {note} |\n')
        f.write('\n`train/grad_norm` 은 **전체 손실**(NTP+KD+OPD) 기준이다. OPD 단독이 아니다.\n')
        f.write('jump 런도 step 1-7 은 dense 이고 step 8 에 한 번에 목표 희소도로 간다.\n')
        f.write('`pgd/*` 는 PGD 가 도는 step 에만 값이 있는 희소 열이다 — 결측을 이어 그리지 말 것.\n')
    print(f'\ntidy rows: {len(tidy)}')


if __name__ == '__main__':
    main()
