#!/usr/bin/env python3
"""두 arm 의 math500 생성문을 같은 문제 기준으로 짝지어 비교한다.

왜 이렇게까지 하는가: "w/o OPD 가 'hmm' 을 더 많이 쓴다" 같은 관찰은 지표를 사후에
고르면 얼마든지 만들 수 있다. 그래서 (1) 지표 집합을 아래에 미리 고정하고,
(2) 500 문제를 짝지어 비교하며, (3) Holm 보정으로 다중비교를 통제하고,
(4) 정답/오답이 길이를 좌우하는 교란을 없애려고 **양쪽 다 맞힌 문제**에서도 따로 본다.

  python compare_generations.py <arm_a_dir> <arm_b_dir> [--seeds 0,1,42] [--out x.csv]
"""
import argparse
import glob
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

# 사전 고정 지표. 결과를 보고 추가하지 않는다.
MARKERS = {
    'wait': r'\bwait\b',
    'hmm': r'\bhmm+\b',
    'alternatively': r'\balternatively\b',
    'but': r'\bbut\b',
    'actually': r'\bactually\b',
    'let_me': r'\blet me\b',
    'recheck': r'\b(re-?check|double-?check|verify)\b',
    'mistake': r'\b(mistake|wrong|error|incorrect)\b',
    'so_the_answer': r'\bso the answer\b',
}
STRUCT = ['n_tokens', 'n_chars', 'closed_think', 'rep4', 'distinct4']


def _text(r):
    t = r.get('text')
    if isinstance(t, (list, np.ndarray)):
        t = t[0] if len(t) else ''
    return str(t)


def _ntok(r):
    o = r.get('output_tokens')
    if o is None:
        return np.nan
    o = np.asarray(o, dtype=object)
    if o.size == 1 and hasattr(o.flat[0], '__len__'):
        return len(o.flat[0])
    return int(o.size)


def rep_stats(s):
    """반복 퇴화 정도. distinct-4 는 생성 퇴화 문헌의 표준 지표다."""
    w = s.split()
    if len(w) < 8:
        return 0.0, 1.0
    g = [' '.join(w[i:i + 4]) for i in range(len(w) - 3)]
    c = Counter(g)
    return c.most_common(1)[0][1] / len(g), len(c) / len(g)


def load(d, seed):
    f = glob.glob(f'{d}/lighteval/seed{seed}/math500/**/*.parquet', recursive=True)
    if not f:
        f = glob.glob(f'{d}/lighteval/math500/**/*.parquet', recursive=True)
    x = pd.read_parquet(f[0])
    rows = []
    for _, row in x.iterrows():
        g = _text(row['model_response'])
        m = row['metric']
        ok = float(list(m.values())[0]) if isinstance(m, dict) and m else np.nan
        rep, dis = rep_stats(g)
        rec = {'qid': str(row['doc'].get('id', '')) or str(row['doc'].get('query', ''))[:80],
               'ok': ok, 'n_tokens': _ntok(row['model_response']), 'n_chars': len(g),
               'closed_think': float('</think>' in g), 'rep4': rep, 'distinct4': dis}
        low = g.lower()
        for k, p in MARKERS.items():
            rec[k] = len(re.findall(p, low))
        rows.append(rec)
    return pd.DataFrame(rows)


def holm(pvals):
    order = np.argsort(pvals)
    n = len(pvals)
    adj = np.empty(n)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (n - i) * pvals[j])
        adj[j] = min(1.0, run)
    return adj


def boot_ci(d, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    m = np.median(d[idx], axis=1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def compare(A, B, label, names):
    from scipy.stats import wilcoxon
    cols = STRUCT + list(MARKERS)
    out, ps = [], []
    for c in cols:
        a, b = A[c].to_numpy(float), B[c].to_numpy(float)
        ok = ~(np.isnan(a) | np.isnan(b))
        a, b = a[ok], b[ok]
        d = b - a
        if np.allclose(d, 0):
            p, r = 1.0, 0.0
        else:
            st = wilcoxon(a, b, zero_method='wilcox')
            p = st.pvalue
            # rank-biserial: 양수면 B 가 큼
            nz = d[d != 0]
            r = (np.sum(nz > 0) - np.sum(nz < 0)) / len(nz)
        lo, hi = boot_ci(d)
        out.append(dict(metric=c, a_mean=a.mean(), b_mean=b.mean(),
                        med_diff=np.median(d), ci_lo=lo, ci_hi=hi, rbc=r, p=p, n=len(a)))
        ps.append(p)
    df = pd.DataFrame(out)
    df['p_holm'] = holm(np.array(ps))
    print(f'\n=== {label}  (n={df["n"].iloc[0]} 문제, {names[0]} vs {names[1]}) ===')
    print(f'{"metric":14s}{names[0][:9]:>11s}{names[1][:9]:>11s}{"중앙차":>9s}'
          f'{"95% CI":>19s}{"rbc":>7s}{"p(Holm)":>10s}')
    for _, r in df.iterrows():
        star = '*' if r.p_holm < 0.05 else ' '
        print(f'{r.metric:14s}{r.a_mean:11.3f}{r.b_mean:11.3f}{r.med_diff:9.2f}'
              f'  [{r.ci_lo:7.2f},{r.ci_hi:7.2f}]{r.rbc:7.2f}{r.p_holm:9.2e}{star}')
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir_a')
    ap.add_argument('dir_b')
    ap.add_argument('--names', default='A,B')
    ap.add_argument('--seeds', default='0,1,42')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    names = a.names.split(',')
    seeds = [s for s in a.seeds.split(',')]
    fa, fb = [], []
    for s in seeds:
        try:
            fa.append(load(a.dir_a, s).assign(seed=s))
            fb.append(load(a.dir_b, s).assign(seed=s))
        except IndexError:
            print(f'  seed {s}: 한쪽에 parquet 없음, 건너뜀')
    A = pd.concat(fa).groupby('qid').mean(numeric_only=True)
    B = pd.concat(fb).groupby('qid').mean(numeric_only=True)
    q = A.index.intersection(B.index)
    A, B = A.loc[q], B.loc[q]
    print(f'시드 {seeds} 평균, 공통 문제 {len(q)}개')
    print(f'  정확도: {names[0]} {A.ok.mean()*100:.1f}%   {names[1]} {B.ok.mean()*100:.1f}%')
    d1 = compare(A, B, '전체 문제', names)
    both = (A.ok == 1) & (B.ok == 1)
    d2 = compare(A[both], B[both], '양쪽 다 정답인 문제만 (길이 교란 제거)', names)
    if a.out:
        d1.assign(subset='all').append if False else None
        pd.concat([d1.assign(subset='all'), d2.assign(subset='both_correct')]).to_csv(a.out, index=False)
        print('\nwrote', a.out)


if __name__ == '__main__':
    main()
