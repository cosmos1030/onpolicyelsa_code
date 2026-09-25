#!/usr/bin/env python3
"""같은 문제에서 dense prefix 와 sparse prefix 각각에 대해 KL(dense || student) 을 잰다.

논문 Figure 4 의 "fixed text 에서는 차이가 없는데 own rollout 에서는 벌어진다" 를
실제 MATH-500 응답 하나 단위로 재현한다. 분기 지점만 보여주는 방식은
temperature 0.6 샘플링이라 dense 끼리도 갈라지므로 증거가 되지 못한다. 반면 이
지표는 **같은 prefix** 에 대한 두 모델의 next-token 분포 비교라 샘플링 노이즈에
영향받지 않는다.

조건:
  on_dense  : dense 가 생성한 prefix 를 teacher forcing (= 고정 calibration 상황)
  on_self   : student 자신이 생성한 prefix 를 teacher forcing (= on-policy 상황)

각 위치에서 KL(p_dense || p_student) 와 top-100 겹침(학습 로그의
onpolicy/overlap_ratio_top100 과 같은 정의)을 구해 응답 단위로 평균한다.

  python prefix_kl.py --dense <path> --student <path> --dense_gen <parquet dir> \
      --student_gen <parquet dir> [--n 100] [--max_new 2048] --out kl.csv
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def _text(r):
    t = r.get('text')
    if isinstance(t, (list, np.ndarray)):
        t = t[0] if len(t) else ''
    return str(t)


def load_gen(d, seed):
    f = glob.glob(f'{d}/lighteval/seed{seed}/math500/**/*.parquet', recursive=True)
    if not f:
        f = glob.glob(f'{d}/lighteval/math500/**/*.parquet', recursive=True)
    x = pd.read_parquet(f[0])
    out = []
    for _, row in x.iterrows():
        m = row['metric']
        out.append(dict(qid=str(row['doc'].get('id', '')) or str(row['doc'].get('query', ''))[:80],
                        query=str(row['doc'].get('query', '')),
                        gen=_text(row['model_response']),
                        ok=float(list(m.values())[0]) if isinstance(m, dict) and m else np.nan))
    return pd.DataFrame(out)


@torch.no_grad()
def logits_of(model, ids):
    return model(ids).logits.float()


@torch.no_grad()
def compare_on_prefix(md, ms, ids, n_prompt):
    """prefix 를 teacher forcing 해 생성 구간의 KL 과 top-100 겹침을 낸다."""
    ld = logits_of(md, ids)[0, n_prompt - 1:-1]
    ls = logits_of(ms, ids)[0, n_prompt - 1:-1]
    pd_ = F.log_softmax(ld, -1)
    ps_ = F.log_softmax(ls, -1)
    kl = (pd_.exp() * (pd_ - ps_)).sum(-1)          # KL(dense || student), 위치별
    kd = pd_.topk(100, -1).indices
    ks = ps_.topk(100, -1).indices
    ov = torch.stack([torch.isin(kd[i], ks[i]).float().mean() for i in range(len(kd))])
    return kl.mean().item(), kl.median().item(), ov.mean().item(), len(kl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dense', required=True)
    ap.add_argument('--student', required=True)
    ap.add_argument('--dense_gen', required=True)
    ap.add_argument('--student_gen', required=True)
    ap.add_argument('--seed', default='0')
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--max_new', type=int, default=2048,
                    help='생성 구간을 이 토큰 수로 자른다 (비용/메모리 상한)')
    ap.add_argument('--out', default='prefix_kl.csv')
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.dense, trust_remote_code=True)
    md = AutoModelForCausalLM.from_pretrained(a.dense, torch_dtype=torch.bfloat16,
                                              device_map='cuda:0', trust_remote_code=True).eval()
    ms = AutoModelForCausalLM.from_pretrained(a.student, torch_dtype=torch.bfloat16,
                                              device_map='cuda:0', trust_remote_code=True).eval()
    D = load_gen(a.dense_gen, a.seed)
    S = load_gen(a.student_gen, a.seed)
    M = D.merge(S, on='qid', suffixes=('_d', '_s'))
    print(f'공통 문제 {len(M)}개 중 앞 {a.n}개 사용', flush=True)

    rows = []
    for i, r in M.head(a.n).iterrows():
        prompt = r['query_d']
        p_ids = tok(prompt, return_tensors='pt').input_ids
        n_p = p_ids.shape[1]
        rec = dict(qid=r['qid'], ok_dense=r['ok_d'], ok_student=r['ok_s'])
        for cond, gen in (('on_dense', r['gen_d']), ('on_self', r['gen_s'])):
            g_ids = tok(gen, return_tensors='pt', add_special_tokens=False).input_ids[:, :a.max_new]
            ids = torch.cat([p_ids, g_ids], 1).to('cuda:0')
            try:
                klm, klmed, ov, n = compare_on_prefix(md, ms, ids, n_p)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                klm = klmed = ov = float('nan'); n = 0
            rec[f'kl_mean_{cond}'] = klm
            rec[f'kl_med_{cond}'] = klmed
            rec[f'overlap100_{cond}'] = ov
            rec[f'ntok_{cond}'] = n
        rows.append(rec)
        if (len(rows)) % 10 == 0:
            df = pd.DataFrame(rows)
            print(f'  {len(rows):4d}개  KL on_dense {df.kl_mean_on_dense.mean():.4f}  '
                  f'on_self {df.kl_mean_on_self.mean():.4f}  '
                  f'overlap {df.overlap100_on_dense.mean():.3f} / {df.overlap100_on_self.mean():.3f}',
                  flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print('\n=== 요약 ===')
    for c in ['kl_mean', 'kl_med', 'overlap100']:
        d, s = df[f'{c}_on_dense'], df[f'{c}_on_self']
        from scipy.stats import wilcoxon
        ok = ~(d.isna() | s.isna())
        p = wilcoxon(d[ok], s[ok]).pvalue if ok.sum() > 5 else float('nan')
        print(f'  {c:12s} on_dense {d.mean():.4f}   on_self {s.mean():.4f}   '
              f'차이 {s.mean()-d.mean():+.4f}   p={p:.2e}')
    print('wrote', a.out)


if __name__ == '__main__':
    main()
