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
    # The launcher's built-in eval never set --run_name, so this landed as
    # 'pending'; it is the seed-42 half of the delta=0.03 jump retrain whose
    # seeds 0/1 arrive as s3_8b_a3jump003_s70_seed{0,1}.
    'iuwf08pf': 'Ours w/o TR (one-step, delta=0.03)',}
METHOD = {'dense': 'dense', 'sparsegpt': 'SparseGPT', 'sgpt_selfgen': 'SparseGPT selfgen',
          'alps': 'ALPS', 'alps_selfgen': 'ALPS selfgen', 'alpsretrain': 'ALPS+retrain',
          'ours': 'Ours',
          # Extra-seed batches run as s3_8b_<arm>_seed{0,1} / _seeds01; the suffix
          # is stripped before this lookup so they land on the same label as the
          # seed-42 run and merge into one block.
          'ours_noopd': 'Ours w/o OPD (0.5/0.5/0)',
          # The s80 DPO arms were measured in two batches: seed 42 first
          # (s3_4b_s80_dpo_lr1e5_ep04_s42, labelled by run id above) and seeds
          # 0/1 afterwards (s3_4b_s80_dpo_lr1e5_s01). Same checkpoint, so they
          # have to land on the same label or the TSV shows one arm twice with
          # one seed each. The label text must match LABELS exactly.
          # SparseLLM s80 baseline; without this the block is titled with the
          # raw run name.
          'sparsellm': 'SparseLLM',
          # s80 ALPS+retrain / Ours: seed 42 came from the training run's own
          # eval and is labelled by run id in LABELS above, so the seed-0/1
          # batches have to resolve to that EXACT text or the arm shows up
          # twice with one seed each. The arm names in eval_8b_long.sh are
          # picked to land on these keys.
          'alpsretrain033': 'ALPS+retrain (0.33/0.33/0.33)',
          'alpsretrainnoopd': 'ALPS+retrain w/o OPD (0.5/0.5/0)',
          # 4B s80 쪽 런 이름만 밑줄이 하나 더 있다(s3_4b_s80_alpsretrain_noopd).
          # 키가 없어서 라벨이 런 이름 그대로 잡혔고, 시드 42(crdshw6m)와 다른
          # 블록으로 갈라져 시드 0 결과가 표에 안 들어갔다. crdshw6m 의 LABELS
          # 문자열과 정확히 같아야 합쳐진다.
          'alpsretrain_noopd': 'ALPS+retrain w/o OPD',
          'oursd003': 'Ours (delta=0.03)',
          # The alpssft4b re-runs (Sep 13) -- a separate training from the
          # August ALPS+retrain checkpoint, so they must not collapse onto the
          # 'ALPS+retrain' label or two different trainings would merge.
          'alpsretrain_3term': 'ALPS+retrain (0.33/0.33/0.33)',
          'alpsretrain_2term': 'ALPS+retrain w/o OPD (0.5/0.5/0)',
          'opdonly': 'Ours OPD-only (0/0/1.0)',
          'norefresh': 'Ours w/o rollout refresh',
          # seed 42 is wandb r8eomj61, labelled by run id above; these extra
          # seeds must resolve to the same text or the arm splits in two.
          'wokd': 'Ours w/o KD (0.5/0/0.5)',
          'cubic_matched': 'Ours, cubic schedule (pace-matched)',
          'dpo_lr1e5': 'Ours + DPO (lr1e-5, ep0.4)',
          'dpo_lr5e6': 'Ours + DPO (lr5e-6, ep1.0)',
          'noopd': 'Ours w/o OPD (0.5/0.5/0)',
          'jump': 'Ours (A3 jump)',
          # The s70 ladder that separates mask source / PGD / trust region:
          # ALPS mask held fixed, PGD on, trust region on (klb 0.02) vs off
          # (klb 99999). Trained on the B200 box 2026-09-21; their TRAINING
          # runs went into the 8B project by mistake (launcher default, fixed
          # in b9cfaee) but carry no eval, so only these eval runs count.
          'alpspgdtr': 'ALPS + PGD (trust region)',
          'alpspgdnotr': 'ALPS + PGD (no trust region)',
          # Cubic ramp 0->0.7 by step 232, then frozen-mask sparse training,
          # PGD OFF. Deliberately NOT 'cubic_matched': that label is the A2sched
          # control, which keeps PGD's KL-gated swaps running and so landed
          # within 0.03 of SCOUT. Merging the two would hide exactly the
          # difference this arm exists to measure.
          'cubicnopgd': 'Cubic schedule, no PGD (pace-matched)',
          # KL-gate jump rule, no frozen pool. There is no s50 arm -- only s60,
          # s70 and 2:4 were ever trained.
          'a3jump': 'Ours (A3 jump)',
          # Retrained 2026-09-23 at delta=0.03 so it matches the 8B s70 SCOUT
          # reference (ajm5l60w) that Figure 4(a) plots it against; the published
          # a3jump runs are delta=0.02. Different trainings, so NOT the same label
          # -- sharing one would present six seeds of a single arm.
          'a3jump003': 'Ours w/o TR (one-step, delta=0.03)',
          # izc2311t has jump_to_target=true AND onpolicy_kd_interval=4096: both
          # the trust region and the rollout refresh are off, and it resumes from
          # an A3B1jump checkpoint. The '_rep' in its run name made its 34.3 read
          # as a failed reproduction of the no-refresh arm's 49.2; the 15-point
          # gap is the jump.
          'norefresh_rep': 'Ours w/o TR (jump) + w/o rollout refresh',
          # The fourth cell of the TR x OPD 2x2 in Figure 4(a): jump_to_target=true
          # AND lambda_OPD=0, trained on the B200 box 2026-09-23/24. Additivity of
          # the two single-factor arms predicts 4B 40.6 and 8B 49.6 on the five-task
          # scale; a large miss either way is the result worth reading.
          'jumpnoopd': 'Ours w/o TR (one-step) + w/o OPD'}
ORDER = ['dense', 'SparseGPT', 'SparseGPT selfgen', 'SparseLLM', 'ALPS', 'ALPS selfgen',
         'ALPS+retrain', 'ALPS + PGD', 'Ours', 'Ours (A3 jump)',
         'Ours w/o TR (one-step, delta=0.03)']


def sparsity(run):
    m = re.search(r'(?:^|_)s(\d\d)(?:pct|_|$)', run.name)
    return int(m.group(1)) if m else 0


def auto_label(run):
    if run.id in LABELS:
        return LABELS[run.id]
    n = re.sub(r'_s42$', '', run.name)
    n = re.sub(r'^s3_[\d.]+b_', '', n)
    n = re.sub(r'_seeds?\d+$', '', n)   # _seed0 / _seed1 / _seeds01
    n = re.sub(r'_ep\d+$', '', n)        # _ep04 on the first DPO batch
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


# 잘렸는데 정답으로 채점된 응답을 오답으로 치는 모드. correct_truncation_rate는
# lighteval_bench._compute_token_stats에서 mean(truncated | correct)이므로
# acc x (1 - ct)가 "예산 안에 끝맺으면서 정답"인 비율이다. 원본 float에서 계산한다
# -- TSV의 반올림된 값을 다시 곱하면 오차가 누적된다.
STRICT = os.environ.get('HARVEST_STRICT') == '1'


def cells(s, suf):
    """One seed's 21 cells; a bench not run at its long cap stays '-'."""
    accs, out = [], []
    for b, akey, cap in BENCH:
        if not is_long(s, b, cap, suf):
            accs.append(None)
            out += ['-'] * 4
            continue
        a = s.get(akey + suf)
        if STRICT and isinstance(a, (int, float)):
            ct = s.get(f'{b}_correct_truncation_rate{suf}')
            # 정답이 0개면 ct가 nan -- 곱할 대상이 없으니 그대로 둔다.
            if isinstance(ct, (int, float)) and ct == ct:
                a = a * (1.0 - ct)
        accs.append(a)
        out += [pct(a), pct(s.get(f'{b}_truncation_rate{suf}')),
                pct(s.get(f'{b}_correct_truncation_rate{suf}')),
                tok(s.get(f'{b}_avg_output_tokens{suf}'))]
    avg5 = sum(accs) * 100 / 5 if all(isinstance(a, (int, float)) for a in accs) else '-'
    # GPQA(인덱스 1)를 뺀 4태스크 평균. 2026-09-23부터 새 평가 잡은 GPQA를 아예
    # 돌리지 않는다(정답의 절반이 잘린 응답이라 능력 지표로 못 씀) -- 그러면 avg5가
    # 통째로 '-'가 되므로 avg4를 별도 열로 둔다. **맨 끝에** 붙이는 이유는 기존 열
    # 위치를 읽는 스크립트(strict 계산, 플롯, 노트)를 깨지 않기 위해서다.
    four = [a for j, a in enumerate(accs) if j != 1]
    avg4 = sum(four) * 100 / 4 if all(isinstance(a, (int, float)) for a in four) else '-'
    return [avg5] + out + [avg4]


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


def header(label, rid, ckpt='', urls=''):
    # 블록 헤더에 체크포인트와 wandb URL을 같이 박는다. id만 있으면 어느 모델의
    # 숫자인지 알려고 runs_<size>.tsv나 wandb를 따로 열어야 했고, 같은 이름의
    # 블록이 둘일 때(4B s70 ALPS+retrain 8월/9월) 특히 헷갈렸다.
    r1 = ([label] + [''] * 21 + [f'wandb {rid}'] +
          ([ckpt] if ckpt else []) + ([urls] if urls else []))
    r2 = ['', 'avg5'] + sum(([b] + [''] * 3 for b in HDR_B), []) + ['avg4 (no GPQA)']
    r3 = ['seed', 'accuracy'] + COLS * 5 + ['accuracy']
    return [r1, r2, r3]


def expected_seeds(run):
    try:
        a = (run.metadata or {}).get('args', [])
    except Exception:
        a = []
    if '--seeds' in a:
        return len(a[a.index('--seeds') + 1].split(','))
    return 1


def run_seed(run):
    """Seed label for a run whose metrics carry bare keys.

    eval_full.py only suffixes metric keys when it runs MORE THAN ONE seed, so
    a `--seeds 0` job logs lighteval/math500, not lighteval/math500_seed0.
    Reading the arg back is the only way to tell seed 0 from seed 42; defaulting
    to '42' silently collapses every single-seed extra-seed run onto the
    original run, and the merge below then keeps just one of them.
    """
    try:
        a = (run.metadata or {}).get('args', [])
    except Exception:
        a = []
    if '--seeds' in a:
        v = a[a.index('--seeds') + 1].split(',')
        if len(v) == 1:
            return v[0].strip()
    return '42'


def harvest(api, size, proj):
    f = {'$or': [{f'summary_metrics.{b}_avg_gen_cap{x}': {'$exists': True}}
                 for b in ('math500', 'gsm8k') for x in SUFFIXES]}
    blocks, excl_blocks, merged = [], [], {}
    for r in api.runs(f'{ENT}/{proj}', filters=f, per_page=200):
        s = r.summary._json_dict
        rows = []
        for suf in SUFFIXES:
            if suf == '' and rows:
                break
            if mixed(s, suf) or not any(is_long(s, b, cap, suf) for b, _, cap in BENCH):
                continue
            seed = suf.replace('_seed', '') or run_seed(r)
            rows.append([seed] + cells(s, suf))
        if not rows:
            continue
        meth = auto_label(r)
        if r.id in EXCLUDE_RUNS:
            meth = meth + ' [EXCLUDED]'
        key = (sparsity(r), meth)
        # Merge runs that share an arm. Extra seeds are run as separate wandb
        # runs (s3_8b_<arm>_seed0, ...), and a re-run after a crashed benchmark
        # leaves two runs for the same seed -- both would otherwise become
        # half-empty blocks under one label. Group here, pick the best row per
        # seed below, then compute mean/std once over the merged seeds.
        merged.setdefault(key, {'rows': {}, 'ids': [], 'runs': [], 'states': [], 'want': 0})
        g = merged[key]
        g['ids'].append(r.id)
        g['runs'].append(r)
        g['states'].append(r.state)
        g['want'] = max(g['want'], expected_seeds(r))
        for row in rows:
            prev = g['rows'].get(row[0])
            # More numeric cells wins: that is the complete pass, not the
            # crashed one.
            score = sum(isinstance(c, (int, float)) for c in row)
            if prev is None or score > sum(isinstance(c, (int, float)) for c in prev):
                g['rows'][row[0]] = row
        print(f'  {size:>4} {r.id} -> s{sparsity(r)} {meth} seeds={sorted(g["rows"])}', file=sys.stderr)

    for (sp, meth), g in merged.items():
        rows = [g['rows'][s] for s in sorted(g['rows'], key=lambda x: (x != '0', x != '1', x))]
        label = f'Qwen 3 {size.upper()} ' + (f's{sp} ' if sp else '') + meth
        full = [x for x in rows if x[1] != '-']
        want = max(g['want'], len(rows))
        if len(full) < 3:
            label += f'  [{len(full)}/3 seeds complete]'
        # 런 수가 시드 수보다 많으면 같은 시드를 두 번 이상 돌린 것이다. 2026-09-19에
        # s3_8b_a3jump_s70이 --seeds 42로 두 번 제출됐는데, 블록에 wandb 런이 2개
        # 붙어 있어 3시드처럼 보였다(실제로는 1시드). 라벨에 박아서 눈에 띄게 한다.
        if len(g['ids']) > len(g['rows']):
            label += f"  [중복시드: 런 {len(g['ids'])}개 / 시드 {len(g['rows'])}개]"
        if any(st_ != 'finished' for st_ in g['states']):
            label += f'  [{"/".join(sorted(set(g["states"])))}]'
        if len(full) >= 2:
            rows = rows + [['mean'] + agg(rows, st.mean), ['std'] + agg(rows, st.stdev)]
        rank = next((i for i, m in enumerate(ORDER) if meth.startswith(m) and
                     not any(meth.startswith(m2) and len(m2) > len(m) for m2 in ORDER)), len(ORDER))
        # 체크포인트: 평가런의 model_path, 없으면 그 런 자신이 올린 hub_model_id
        # (학습런이 자체 평가한 블록이 그렇다).
        ckpt = ''
        for r_ in g['runs']:
            ckpt = r_.config.get('model_path') or r_.summary._json_dict.get('hub_model_id') or ''
            if ckpt:
                break
        urls = ' '.join(f'https://wandb.ai/{ENT}/{proj}/runs/{i}' for i in g['ids'])
        tgt = excl_blocks if '[EXCLUDED]' in meth else blocks
        tgt.append(((sp, rank, meth),
                    header(label, ' '.join(g['ids']), str(ckpt), urls) + rows + [[''] * 23]))
    blocks.sort(key=lambda x: x[0])
    excl_blocks.sort(key=lambda x: x[0])
    # provenance: 블록 -> (wandb id, run 객체) 목록. write_runs_meta()가 쓴다.
    prov = []
    for (sp, rank, meth), _ in blocks:
        g = merged[(sp, meth)]
        label = f'Qwen 3 {size.upper()} ' + (f's{sp} ' if sp else '') + meth
        prov.append((label, g['ids'], g['runs'], sorted(g['rows'])))
    return ([row for _, b in blocks for row in b],
            [row for _, b in excl_blocks for row in b], prov)



# ---------------------------------------------------------------- provenance
# 결과 표만 보면 같은 이름의 블록 둘이 같은 체크포인트를 두 번 돌린 것인지, 설정이
# 한 줄 다른 별개 arm인지 알 수 없다. 실제로 4B s70에 "ALPS+retrain"(44.94)과
# "ALPS+retrain (0.33/0.33/0.33)"(43.45)이 나란히 있었고, 손실 가중치·롤아웃 길이·
# 스텝이 전부 같고 gmp_onpolicy_kd_interval만 1 vs 32로 달랐다. 그 한 줄을 확인하려면
# 매번 wandb를 뒤져야 했다. 이 파일이 그 수고를 없앤다.
#
# 평가 런(s3_*)의 config에는 model_path만 있고 학습 설정이 없으므로, 체크포인트를
# 올린 학습 런을 찾아 그 config를 붙인다. 프로젝트 전체 스캔은 비싸므로 결과를
# train_config_cache.json에 캐시하고, 캐시에 없는 체크포인트가 생겼을 때만 스캔한다.
TRAIN_KEYS = [('gmp_ntp_lambda', 'ntp'), ('gmp_kd_lambda', 'kd'),
              ('gmp_onpolicy_kd_lambda', 'opd'),
              ('gmp_onpolicy_kd_interval', 'rollout_interval'),
              ('gmp_onpolicy_max_new_tokens', 'rollout_len'),
              ('gmp_fixed_mask', 'fixed_mask'), ('gmp_pgd', 'pgd'),
              ('gmp_pgd_kl_budget', 'kl_budget'), ('gmp_tr_enabled', 'tr'),
              ('gmp_growth_schedule', 'growth'), ('gmp_mask_interval', 'mask_interval'),
              ('steps', 'steps'), ('learning_rate', 'lr'), ('seqlen', 'seqlen')]
CACHE = os.path.join(HERE, 'train_config_cache.json')

# 주 결과에서 빼둘 평가런. 삭제하지 않고 EXCL_DIR로 따로 내보낸다 -- harvest는
# wandb에서 매번 새로 읽으므로 TSV에서 줄만 지우면 다음 주기에 되살아난다.
#
# 0eqkqmbs / qzhxc4h9: 4B s70 ALPS+retrain 3-term을 9월에 다시 돌린 런들.
#   본표·Figure 3(a)가 쓰는 것은 8월 런(5x4prktp, 평가 vfjcx821, avg4 49.36)이고
#   9월 런은 avg4 47.96이다. 설정 차이는 없다(로그상 갱신 주기도 둘 다 32스텝) --
#   같은 레시피의 독립 학습 런이 1.4점 차이 난 것이므로 학습 재현 분산의 유일한
#   데이터점으로 보관하되, 같은 이름의 블록 둘이 주 표에 나란히 있으면 어느 쪽이
#   baseline인지 알 수 없어 혼동을 부른다.
EXCLUDE_RUNS = {
    '0eqkqmbs': '4B s70 ALPS+retrain 3-term, Sep repeat (주 baseline은 8월 vfjcx821)',
    'qzhxc4h9': '4B s70 ALPS+retrain 3-term, Sep repeat (단일 시드, B200 로컬 체크포인트)',
    # 1.7B s60 SCOUT 은 이 칸만 rollout interval 8 로 학습한 체크포인트
    # (20260901_121104) 로 평가돼 있었다. 다른 칸은 전부 ro=32 다. ro=32 로 학습한
    # 94pwg46m 의 체크포인트(20260902_173413)로 다시 돌린 값이 같은 런 이름
    # s3_1.7b_s60_ours 로 들어오므로, 섞이지 않게 옛 런을 빼둔다.
    'padwnv7g': '1.7B s60 Ours, rollout interval 8 로 학습한 체크포인트 (표준은 ro=32)',
}
EXCL_DIR = os.path.join(os.path.dirname(HERE), 'tsv_excluded')


def _ckpt_of(run):
    """평가 런이 가리키는 체크포인트. 학습 런이 자체 평가한 경우엔 자기 자신."""
    mp = run.config.get('model_path')
    if mp:
        return str(mp)
    hub = run.summary._json_dict.get('hub_model_id')
    return str(hub) if hub else ''


def _norm(p):
    """허브 repo id와 로컬 출력 디렉터리를 같은 키로 묶기 위한 말단 타임스탬프."""
    m = re.search(r'(\d{8}_\d{6})', p or '')
    return m.group(1) if m else (p or '')


def resolve_train_configs(api, proj, ckpts):
    """체크포인트 -> 학습 config. 캐시에 없는 것이 있을 때만 프로젝트를 훑는다."""
    import json
    cache = {}
    if os.path.exists(CACHE):
        try:
            cache = json.load(open(CACHE))
        except Exception:
            cache = {}
    missing = [c for c in ckpts if c and _norm(c) not in cache]
    if missing:
        print(f'  [meta] {len(missing)}개 체크포인트의 학습 config 조회 중 ...', file=sys.stderr)
        for r in api.runs(f'{ENT}/{proj}', per_page=500):
            c = r.config
            if 'gmp_onpolicy_kd_lambda' not in c and 'gmp_kd_lambda' not in c:
                continue          # 평가 런은 건너뛴다
            blob = str(r.summary._json_dict.get('hub_model_id', '')) + str(c.get('output_dir', '')) + r.name
            key = _norm(blob)
            for cand in missing:
                if _norm(cand) and _norm(cand) in blob:
                    cache[_norm(cand)] = {'train_run': r.id, 'train_name': r.name[:80],
                                          **{short: c.get(k) for k, short in TRAIN_KEYS}}
        json.dump(cache, open(CACHE, 'w'), indent=1, default=str)
    return cache


def write_runs_meta(api, size, proj, prov):
    """블록별 런 출처 + 학습 설정을 runs_<size>.tsv로."""
    ckpts = {_ckpt_of(r) for _, _, runs, _ in prov for r in runs}
    cache = resolve_train_configs(api, proj, ckpts)
    cols = ['block', 'wandb_id', 'run_name', 'state', 'seeds', 'checkpoint',
            'train_run'] + [short for _, short in TRAIN_KEYS]
    out = os.path.join(HERE, size, f'runs_{size}.tsv')
    with open(out, 'w') as fh:
        fh.write('\t'.join(cols) + '\n')
        for label, ids, runs, seeds in prov:
            for r in runs:
                ck = _ckpt_of(r)
                tc = cache.get(_norm(ck), {})
                row = [label, r.id, r.name[:70], r.state, ','.join(seeds), ck,
                       str(tc.get('train_run', ''))]
                row += [str(tc.get(short, '')) for _, short in TRAIN_KEYS]
                fh.write('\t'.join(row) + '\n')
    print(f'wrote {out}', file=sys.stderr)


def main():
    api = wandb.Api(timeout=180)
    for size, proj in PROJ.items():
        sheet, excl, prov = harvest(api, size, proj)
        sub = 'strict' if STRICT else ''
        base = os.path.join(HERE, sub, size) if sub else f'{HERE}/{size}'
        os.makedirs(base, exist_ok=True)
        out = os.path.join(base, ('strict_results_%s.tsv' % size) if STRICT
                           else 'long_results_%s.tsv' % size)
        with open(out, 'w') as fh:
            for row in sheet:
                fh.write('\t'.join(fmt(i, c) for i, c in enumerate(row)) + '\n')
        print(f'wrote {out}  ({len(sheet)} rows)', file=sys.stderr)
        if not STRICT:
            write_runs_meta(api, size, proj, prov)
        if excl:
            ed = os.path.join(EXCL_DIR, size)
            os.makedirs(ed, exist_ok=True)
            # strict 모드가 같은 파일을 덮어쓰지 않게 이름을 분리한다.
            ep = os.path.join(ed, ('excluded_strict_%s.tsv' if STRICT else 'excluded_%s.tsv') % size)
            with open(ep, 'w') as fh:
                for row in excl:
                    fh.write('\t'.join(fmt(i, c) for i, c in enumerate(row)) + '\n')
            print(f'wrote {ep}  ({len(excl)} rows, 주 결과에서 제외)', file=sys.stderr)


if __name__ == '__main__':
    main()
