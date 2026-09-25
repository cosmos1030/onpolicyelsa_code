#!/usr/bin/env python3
"""두 생성문이 처음 갈라지는 토큰 위치를 재고, dense-dense 대조군과 비교한다.

왜 대조군이 필요한가: 평가가 temperature 0.6 / top-p 0.95 샘플링이라 **dense 끼리도**
갈라진다. "같은 프롬프트인데 dense 와 sparse 의 경로가 달라졌다" 만으로는 pruning
때문이라는 증거가 되지 않는다. dense(seed0) vs dense(seed1) 분기 위치 분포를 기준선
으로 두고, dense(seed0) vs sparse(seed0) 가 체계적으로 더 이른지 본다.

  python divergence_point.py --tok <모델경로> \
      --dense0 <dir> --dense1 <dir> --sparse <dir> [--out div.csv]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd


def _text(r):
    t = r.get('text')
    if isinstance(t, (list, np.ndarray)):
        t = t[0] if len(t) else ''
    return str(t)


def _find_parquet(d, seed):
    """glob 대신 os.walk 를 쓴다. dense 모델의 details 경로에는
    `.cache` 처럼 점으로 시작하는 디렉터리가 들어가는데 glob 의 `**` 는 그런
    디렉터리를 건너뛰어 파일을 못 찾는다."""
    want = f'seed{seed}' if seed is not None else None
    hits = []
    for root, _, files in os.walk(d):
        for f in files:
            if f.endswith('.parquet') and 'math500' in root:
                hits.append(os.path.join(root, f))
    if want:
        pref = [p for p in hits if f'/{want}/' in p]
        if pref:
            return pref
    return [p for p in hits if '/seed' not in p] or hits


def load_gen(d, seed):
    f = _find_parquet(d, seed)
    x = pd.read_parquet(f[0])
    return pd.DataFrame([
        dict(qid=str(r['doc'].get('id', '')) or str(r['doc'].get('query', ''))[:80],
             gen=_text(r['model_response']))
        for _, r in x.iterrows()])


def first_diff(tok, a, b):
    ta = tok(a, add_special_tokens=False).input_ids
    tb = tok(b, add_special_tokens=False).input_ids
    n = min(len(ta), len(tb))
    for i in range(n):
        if ta[i] != tb[i]:
            return i, len(ta), len(tb)
    return n, len(ta), len(tb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tok', required=True)
    ap.add_argument('--dense0', required=True)
    ap.add_argument('--dense1', required=True, help='대조군: 같은 dense 의 다른 시드')
    ap.add_argument('--sparse', required=True)
    ap.add_argument('--seed_sparse', default='0')
    ap.add_argument('--out', default='divergence.csv')
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tok, trust_remote_code=True)
    D0 = load_gen(a.dense0, '0')
    D1 = load_gen(a.dense1, '1')
    S = load_gen(a.sparse, a.seed_sparse)
    M = D0.merge(D1, on='qid', suffixes=('_d0', '_d1')).merge(
        S.rename(columns={'gen': 'gen_s'}), on='qid')
    print(f'공통 문제 {len(M)}개')

    rows = []
    for _, r in M.iterrows():
        i_ctrl, la, lb = first_diff(tok, r.gen_d0, r.gen_d1)
        i_test, _, lc = first_diff(tok, r.gen_d0, r.gen_s)
        rows.append(dict(qid=r.qid, div_dense_dense=i_ctrl, div_dense_sparse=i_test,
                         len_d0=la, len_d1=lb, len_s=lc))
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)

    from scipy.stats import wilcoxon
    c, t = df.div_dense_dense, df.div_dense_sparse
    p = wilcoxon(c, t).pvalue
    nz = (t - c)[(t - c) != 0]
    rbc = (np.sum(nz > 0) - np.sum(nz < 0)) / len(nz) if len(nz) else 0.0
    print('\n=== 분기 위치 (토큰 인덱스, 작을수록 일찍 갈라짐) ===')
    for nm, v in [('dense-dense (대조군)', c), ('dense-sparse', t)]:
        print(f'  {nm:22s} 중앙값 {v.median():7.1f}  평균 {v.mean():8.1f}  '
              f'25% {v.quantile(.25):6.1f}  75% {v.quantile(.75):7.1f}')
    print(f'  Wilcoxon p={p:.3e}   rank-biserial={rbc:+.3f} '
          f'(음수면 sparse 가 더 일찍 갈라짐)')
    print('wrote', a.out)


if __name__ == '__main__':
    main()
