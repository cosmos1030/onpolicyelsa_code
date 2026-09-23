#!/usr/bin/env python3
"""제출 전 중복 확인: 이 체크포인트를 이미 누가 돌리고 있나?

왜: 평가 잡은 두 서버(이 클러스터, B200)에서 각각 제출된다. 로컬 squeue만 보면
상대 서버의 잡이 안 보이고, 그래서 같은 arm이 두 번 돌아간 사고가 반복됐다
(2026-09-23 4B s80 ALPS+retrain / w/o OPD 두 건, 9/19 8B s70 jump는 같은 서버
안에서 --seeds 42로 두 번). wandb는 양쪽이 공유하므로 여기가 유일한 공통 뷰다.

    python long_tsv_results/check_before_submit.py 4b cosmos1030/gmp-kd3e-1-4b-s80pct-lr1e-4_20260917_112952
    python long_tsv_results/check_before_submit.py 8b --name s3_8b_a3jump       # 이름으로도 검색

출력: 같은 체크포인트(또는 이름)를 쓴 최근 평가런의 상태와 시드. running이 있으면
제출하지 말고, finished가 있으면 그 시드는 빼고 제출할 것.
"""
import argparse
import re
import sys

import wandb

ENT = 'dyk6208-gwangju-institute-of-science-and-technology'
PROJ = {'1.7b': 'reasoning_qwen3_1.7b_nostrip8192',
        '4b': 'reasoning_qwen3_4b_nostrip8192',
        '8b': 'reasoning_qwen3_8b_nostrip8192'}


def seeds_of(run):
    s = run.summary._json_dict
    got = sorted({k.split('_seed')[1] for k in s
                  if '_seed' in k and k.split('_seed')[1].isdigit()})
    if got:
        return got
    args = (run.metadata or {}).get('args', [])
    if '--seeds' in args:
        return args[args.index('--seeds') + 1].split(',')
    return ['?']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('size', choices=list(PROJ))
    ap.add_argument('ckpt', nargs='?', default='',
                    help='체크포인트 경로/레포 id (일부만 줘도 됨, 예: 타임스탬프)')
    ap.add_argument('--name', default='', help='런 이름 부분 문자열로도 검색')
    ap.add_argument('--days', type=int, default=10)
    a = ap.parse_args()
    if not a.ckpt and not a.name:
        sys.exit('ckpt 또는 --name 중 하나는 필요')

    import datetime as dt
    since = (dt.datetime.utcnow() - dt.timedelta(days=a.days)).strftime('%Y-%m-%dT%H:%M:%S')
    api = wandb.Api(timeout=120)
    key = re.search(r'(\d{8}_\d{6})', a.ckpt)
    key = key.group(1) if key else a.ckpt

    hits = []
    for r in api.runs(f'{ENT}/{PROJ[a.size]}',
                      filters={'createdAt': {'$gte': since}}, per_page=200):
        mp = str(r.config.get('model_path') or '')
        blob = mp + ' ' + r.name
        if (key and key in blob) or (a.name and a.name in r.name):
            hits.append(r)

    if not hits:
        print(f'최근 {a.days}일 내 같은 체크포인트/이름의 평가런 없음 -- 제출해도 됨')
        return
    print(f'{len(hits)}개 발견 (running이 있으면 중복 제출 금지):')
    for r in sorted(hits, key=lambda x: str(x.created_at)):
        mark = '  <-- 실행 중' if r.state == 'running' else ''
        print(f'  {r.id} {str(r.created_at)[:16]} {r.state:9s} seeds={seeds_of(r)} '
              f'{r.name[:44]}{mark}')
        print(f'      {r.config.get("model_path")}')


if __name__ == '__main__':
    main()
