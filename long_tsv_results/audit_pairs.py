#!/usr/bin/env python3
"""논문에 들어가는 모든 비교 쌍의 학습 config를 기계적으로 대조한다.

왜: 같은 종류의 오염이 이미 두 번 나왔다.
  (1) 4B "w/o refresh 재현런"인 줄 알았던 izc2311t가 실은 jump arm이었고,
  (2) 8B SCOUT(δ=0.03, resweep2_opkdfix)과 frozen arm(δ=0.02)의 trust-region
      반경이 달랐다.
하나씩 발견하는 대신 전부 한 번에 훑는다.

출력 한 줄 = 한 비교 쌍:
  기대한 키(비교 종류별로 달라야 하는 키) 외의 config 차이, 양쪽 resume 여부,
  생성 날짜, wandb가 기록한 git commit.

체크포인트 -> 학습런 매핑은 hub_model_id / output_dir 안의 타임스탬프
(20260907_073958 같은)로 잡는다. 평가런(s3_*)의 config에는 model_path만 있고
학습 설정이 없기 때문이다. 학습런이 자체 평가한 블록은 그 런 자신이 학습런이다.

  python long_tsv_results/audit_pairs.py            # 전체
  python long_tsv_results/audit_pairs.py 4b         # 한 크기만
"""
import json
import os
import re
import sys

import wandb

ENT = 'dyk6208-gwangju-institute-of-science-and-technology'
PROJ = {'1.7b': 'reasoning_qwen3_1.7b_nostrip8192',
        '4b': 'reasoning_qwen3_4b_nostrip8192',
        '8b': 'reasoning_qwen3_8b_nostrip8192'}
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'audit_run_cache.json')

# 실행 환경/체크포인트 관리에만 영향을 주는 키. 결과를 바꾸지 않는다.
IGNORE = {
    'gmp_ckpt_dir', 'gmp_ckpt_every_steps', 'gmp_ckpt_keep', 'gmp_resume_from',
    'run_name_suffix', 'gmp_opkd_vllm_gpu_mem', 'gmp_opkd_vllm_sidecar',
    'gmp_opkd_vllm_tp_size', 'gmp_opkd_vllm_enforce_eager', 'gmp_use_fsdp',
    'gmp_save_path', 'wandb_project', 'wandb_run_id', 'output_dir', 'data_path',
    'gmp_prompt_path', 'gmp_gradient_checkpointing', 'gmp_kl_chunk_size',
    'gmp_recovery_diag', 'gmp_pgd_recovery_diag', 'gmp_pgd_debug_importance_hist',
    'gmp_pgd_debug_repeat_swap', 'gmp_pgd_debug_repeat_window',
    'gmp_saliency_diag_step', 'gmp_saliency_diag_k', 'gmp_saliency_diag_mc_nsamples',
    'gmp_saliency_corr_step', 'gmp_saliency_corr_groups', 'gmp_saliency_corr_group_size',
    'gmp_saliency_corr_seed', 'gmp_cubic_log_kl', 'eval_profile', 'eval_full_bench',
    'eval_zero_shot', 'eval_math500', 'push_to_hub', 'save_model', 'seed',
}
# 비교 종류별로 "달라도 되는" 키
EXPECTED = {
    'wo_opd':     {'gmp_onpolicy_kd_lambda', 'gmp_kd_lambda', 'gmp_ntp_lambda', 'gmp_kd_only'},
    'opd_only':   {'gmp_onpolicy_kd_lambda', 'gmp_kd_lambda', 'gmp_ntp_lambda', 'gmp_kd_only'},
    'wo_kd':      {'gmp_onpolicy_kd_lambda', 'gmp_kd_lambda', 'gmp_ntp_lambda', 'gmp_kd_only'},
    'wo_refresh': {'gmp_onpolicy_kd_interval'},
    'jump':       {'gmp_pgd_jump_to_target'},
    'cubic':      {'gmp_pgd', 'gmp_pgd_grow_to_target', 'gmp_growth_schedule',
                   'gmp_pruning_end_ratio', 'gmp_sparse_train_steps',
                   'gmp_pgd_grow_rule', 'gmp_pgd_grow_rule_end_ratio'},
    'alps_sft':   {'gmp_fixed_mask', 'gmp_pgd', 'gmp_pgd_grow_to_target',
                   'gmp_pgd_kl_budget', 'gmp_pgd_interval', 'gmp_pgd_kl_calib_size',
                   'gmp_pruning_end_ratio', 'model', 'gmp_pgd_jump_to_target',
                   'gmp_pgd_kl_bisect_iters', 'gmp_pgd_trust_ratio'},
    'dpo':        None,   # 후처리 arm: diff 의미 없음
    'oneshot':    None,   # 학습 없음
}
KIND_BY_LABEL = [
    (r'w/o OPD|w/o on-policy', 'wo_opd'),
    (r'OPD-only', 'opd_only'),
    (r'w/o KD', 'wo_kd'),
    (r'w/o rollout refresh|norefresh', 'wo_refresh'),
    (r'A3 jump|jump', 'jump'),
    (r'cubic', 'cubic'),
    (r'ALPS\+retrain|alpsretrain|ALPS\+PGD', 'alps_sft'),
    (r'DPO', 'dpo'),
    (r'SparseGPT|SparseLLM|^ALPS$|ALPS selfgen|ALPS$', 'oneshot'),
]


def kind_of(label):
    for pat, k in KIND_BY_LABEL:
        if re.search(pat, label):
            return k
    return None


def ts(s):
    m = re.search(r'(\d{8}_\d{6})', str(s) or '')
    return m.group(1) if m else None


def block_ids(size):
    """long_results TSV에서 블록 라벨 -> wandb 평가런 id 목록."""
    path = os.path.join(HERE, size, f'long_results_{size}.tsv')
    out = []
    for line in open(path):
        if not line.startswith('Qwen 3'):
            continue
        label = line.split('\t')[0].split('  [')[0]
        ids = re.findall(r'wandb ([a-z0-9 ]+)', line)
        ids = ids[0].split() if ids else []
        if ids:
            out.append((label, ids))
    return out


def main():
    sizes = sys.argv[1:] or list(PROJ)
    api = wandb.Api(timeout=180)
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

    for size in sizes:
        proj = PROJ[size]
        print(f'\n{"="*100}\n{size}\n{"="*100}', flush=True)
        # 1) 블록 -> 체크포인트
        blocks = []
        for label, ids in block_ids(size):
            ck = None
            for rid in ids:
                try:
                    r = api.run(f'{ENT}/{proj}/{rid}')
                except Exception:
                    continue
                ck = r.config.get('model_path') or r.summary._json_dict.get('hub_model_id')
                if ck:
                    break
            blocks.append((label, ids, str(ck) if ck else None))

        want = {ts(ck) for _, _, ck in blocks if ts(ck)}
        want -= set(cache)
        # 2) 프로젝트 1회 스캔으로 학습런 찾기
        if want:
            print(f'  [scan] 학습런 {len(want)}개 조회 ...', flush=True)
            for r in api.runs(f'{ENT}/{proj}', per_page=500):
                c = r.config
                if 'gmp_kd_lambda' not in c and 'gmp_onpolicy_kd_lambda' not in c:
                    continue
                blob = str(r.summary._json_dict.get('hub_model_id', '')) + str(c.get('output_dir', '')) + r.name
                t = ts(blob)
                if t and t in want:
                    cache[t] = {'run': r.id, 'name': r.name[:90], 'created': str(r.created_at)[:16],
                                'commit': (r.commit or '')[:8], 'state': r.state,
                                'resume': bool(c.get('gmp_resume_from')),
                                'config': {k: v for k, v in c.items()}}
                    want.discard(t)
            json.dump(cache, open(CACHE, 'w'), default=str)

        # 3) 희소도 그룹별로 SCOUT을 기준으로 diff
        groups = {}
        for label, ids, ck in blocks:
            m = re.search(r's(\d+)', label)
            sp = m.group(1) if m else 'dense'
            groups.setdefault(sp, []).append((label, ck))
        for sp in sorted(groups):
            rows = groups[sp]
            ref = next((x for x in rows if re.search(r'\bOurs\b', x[0]) and
                        not re.search(r'w/o|jump|cubic|OPD-only|DPO|norefresh', x[0])), None)
            print(f'\n--- s{sp}  기준: {ref[0] if ref else "(SCOUT 블록 없음)"}', flush=True)
            rc = cache.get(ts(ref[1])) if ref and ts(ref[1]) else None
            if rc:
                print(f'    기준 학습런 {rc["run"]}  {rc["created"]}  commit={rc["commit"] or "?"}'
                      f'  resume={rc["resume"]}', flush=True)
            for label, ck in rows:
                if ref and label == ref[0]:
                    continue
                t = ts(ck)
                info = cache.get(t)
                if not info:
                    print(f'    {label[:56]:<58} 학습런 못 찾음 (ck={str(ck)[:46]})', flush=True)
                    continue
                kind = kind_of(label)
                head = (f'    {label[:56]:<58} {info["run"]} {info["created"]} '
                        f'commit={info["commit"] or "?":8s} resume={str(info["resume"]):5s}')
                if not rc or kind is None or EXPECTED.get(kind) is None:
                    print(head + ('  (기준 없음/대조 제외)' if not rc or kind is None else '  (후처리·one-shot)'),
                          flush=True)
                    continue
                ca, cb = rc['config'], info['config']
                bad = sorted(k for k in set(ca) | set(cb)
                             if ca.get(k) != cb.get(k) and k not in IGNORE | EXPECTED[kind])
                if bad:
                    print(head + f'  MISMATCH({kind}): ' +
                          ', '.join(f'{k}[{ca.get(k)}→{cb.get(k)}]' for k in bad[:6]) +
                          (f' +{len(bad)-6}개' if len(bad) > 6 else ''), flush=True)
                else:
                    print(head + f'  OK({kind})', flush=True)


if __name__ == '__main__':
    main()
