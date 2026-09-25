#!/usr/bin/env python3
"""rambling 을 표면 단어가 아니라 생성 구조로 측정한다.

'hmm' 을 몇 번 썼는지는 사후에 고른 지표라 약하다. 대신 답을 내고도 계속 쓰는지,
답이 오락가락하는지, 뒤로 갈수록 새 내용이 나오는지를 잰다. 전부 생성문만으로
계산되며 추가 forward pass 가 필요 없다.

  answer_churn    : 추론 중 제시한 서로 다른 \\boxed 답의 개수 (1 이면 흔들림 없음)
  answer_switches : 연속한 \\boxed 답이 바뀐 횟수
  frac_after_first: 첫 \\boxed 이후에 쓴 토큰 비율 — 답을 내고도 멈추지 않는 정도
  new4_late       : 뒤쪽 절반에서 처음 등장하는 4-gram 비율 — 낮을수록 같은 말 반복
  new4_slope      : 10분위 구간별 신규 4-gram 비율의 기울기 (음수면 갈수록 새 내용 없음)
  gzip_ratio      : 압축률. 낮을수록 중복이 많다
  think_frac      : </think> 이전(추론부)이 전체에서 차지하는 비율

  python rambling_metrics.py <dir_a> <dir_b> --names A,B [--seeds 0,1,42]
"""
import argparse
import glob
import gzip
import re

import numpy as np
import pandas as pd

BOXED = re.compile(r'\\boxed\{([^{}]{0,60})\}')
COLS = ['answer_churn', 'answer_switches', 'frac_after_first', 'new4_late',
        'new4_slope', 'gzip_ratio', 'think_frac', 'n_boxed']


def _text(r):
    t = r.get('text')
    if isinstance(t, (list, np.ndarray)):
        t = t[0] if len(t) else ''
    return str(t)


def metrics(g):
    n = max(len(g), 1)
    hits = [(m.start(), m.group(1).strip()) for m in BOXED.finditer(g)]
    vals = [v for _, v in hits]
    churn = len(set(vals)) if vals else 0
    switches = sum(1 for i in range(1, len(vals)) if vals[i] != vals[i - 1])
    frac_after = (n - hits[0][0]) / n if hits else 0.0

    w = g.split()
    grams = [' '.join(w[i:i + 4]) for i in range(max(len(w) - 3, 0))]
    if len(grams) >= 20:
        half = len(grams) // 2
        seen = set(grams[:half])
        new_late = sum(1 for x in grams[half:] if x not in seen) / max(len(grams) - half, 1)
        bins = np.array_split(np.arange(len(grams)), 10)
        seen, rates = set(), []
        for b in bins:
            new = sum(1 for i in b if grams[i] not in seen)
            rates.append(new / max(len(b), 1))
            seen.update(grams[i] for i in b)
        slope = np.polyfit(np.arange(10), rates, 1)[0]
    else:
        new_late, slope = np.nan, np.nan

    raw = g.encode()
    ratio = len(gzip.compress(raw)) / max(len(raw), 1)
    i = g.find('</think>')
    think = (i / n) if i >= 0 else 1.0
    return dict(answer_churn=churn, answer_switches=switches, frac_after_first=frac_after,
                new4_late=new_late, new4_slope=slope, gzip_ratio=ratio,
                think_frac=think, n_boxed=len(hits))


def load(d, seed):
    f = glob.glob(f'{d}/lighteval/seed{seed}/math500/**/*.parquet', recursive=True)
    if not f:
        f = glob.glob(f'{d}/lighteval/math500/**/*.parquet', recursive=True)
    x = pd.read_parquet(f[0])
    rows = []
    for _, row in x.iterrows():
        g = _text(row['model_response'])
        m = row['metric']
        rec = metrics(g)
        rec['ok'] = float(list(m.values())[0]) if isinstance(m, dict) and m else np.nan
        rec['qid'] = str(row['doc'].get('id', '')) or str(row['doc'].get('query', ''))[:80]
        rows.append(rec)
    return pd.DataFrame(rows)


def holm(p):
    o = np.argsort(p)
    n, adj, run = len(p), np.empty(len(p)), 0.0
    for i, j in enumerate(o):
        run = max(run, (n - i) * p[j])
        adj[j] = min(1.0, run)
    return adj


def compare(A, B, label, names):
    from scipy.stats import wilcoxon
    out, ps = [], []
    for c in COLS:
        a, b = A[c].to_numpy(float), B[c].to_numpy(float)
        ok = ~(np.isnan(a) | np.isnan(b))
        a, b = a[ok], b[ok]
        d = b - a
        if np.allclose(d, 0):
            p, r = 1.0, 0.0
        else:
            p = wilcoxon(a, b, zero_method='wilcox').pvalue
            nz = d[d != 0]
            r = (np.sum(nz > 0) - np.sum(nz < 0)) / len(nz)
        out.append(dict(metric=c, a=a.mean(), b=b.mean(), rbc=r, p=p, n=len(a)))
        ps.append(p)
    df = pd.DataFrame(out)
    df['p_holm'] = holm(np.array(ps))
    print(f'\n=== {label} (n={df.n.iloc[0]}) ===')
    print(f'{"metric":18s}{names[0][:9]:>11s}{names[1][:9]:>11s}{"rbc":>8s}{"p(Holm)":>11s}')
    for _, r in df.iterrows():
        print(f'{r.metric:18s}{r.a:11.4f}{r.b:11.4f}{r.rbc:8.2f}{r.p_holm:10.2e}'
              + ('*' if r.p_holm < 0.05 else ' '))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir_a')
    ap.add_argument('dir_b')
    ap.add_argument('--names', default='A,B')
    ap.add_argument('--seeds', default='0,1,42')
    a = ap.parse_args()
    names = a.names.split(',')
    fa, fb = [], []
    for s in a.seeds.split(','):
        try:
            fa.append(load(a.dir_a, s))
            fb.append(load(a.dir_b, s))
        except IndexError:
            print(f'  seed {s}: parquet 없음, 건너뜀')
    A = pd.concat(fa).groupby('qid').mean(numeric_only=True)
    B = pd.concat(fb).groupby('qid').mean(numeric_only=True)
    q = A.index.intersection(B.index)
    A, B = A.loc[q], B.loc[q]
    print(f'공통 문제 {len(q)}개   정확도 {names[0]} {A.ok.mean()*100:.1f}% / {names[1]} {B.ok.mean()*100:.1f}%')
    compare(A, B, '전체', names)
    both = (A.ok == 1) & (B.ok == 1)
    compare(A[both], B[both], '양쪽 다 정답', names)


if __name__ == '__main__':
    main()
