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
}
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
          'oursd003': 'Ours (delta=0.03)',
          # The alpssft4b re-runs (Sep 13) -- a separate training from the
          # August ALPS+retrain checkpoint, so they must not collapse onto the
          # 'ALPS+retrain' label or two different trainings would merge.
          'alpsretrain_3term': 'ALPS+retrain (0.33/0.33/0.33)',
          'alpsretrain_2term': 'ALPS+retrain w/o OPD (0.5/0.5/0)',
          'opdonly': 'Ours OPD-only (0/0/1.0)',
          'norefresh': 'Ours w/o rollout refresh',
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
          # KL-gate jump rule, no frozen pool. There is no s50 arm -- only s60,
          # s70 and 2:4 were ever trained.
          'a3jump': 'Ours (A3 jump)'}
ORDER = ['dense', 'SparseGPT', 'SparseGPT selfgen', 'SparseLLM', 'ALPS', 'ALPS selfgen',
         'ALPS+retrain', 'ALPS + PGD', 'Ours', 'Ours (A3 jump)']


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


def cells(s, suf):
    """One seed's 21 cells; a bench not run at its long cap stays '-'."""
    accs, out = [], []
    for b, akey, cap in BENCH:
        if not is_long(s, b, cap, suf):
            accs.append(None)
            out += ['-'] * 4
            continue
        a = s.get(akey + suf)
        accs.append(a)
        out += [pct(a), pct(s.get(f'{b}_truncation_rate{suf}')),
                pct(s.get(f'{b}_correct_truncation_rate{suf}')),
                tok(s.get(f'{b}_avg_output_tokens{suf}'))]
    avg5 = sum(accs) * 100 / 5 if all(isinstance(a, (int, float)) for a in accs) else '-'
    return [avg5] + out


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


def header(label, rid):
    r1 = [label] + [''] * 20 + [f'wandb {rid}']
    r2 = ['', 'avg5'] + sum(([b] + [''] * 3 for b in HDR_B), [])
    r3 = ['seed', 'accuracy'] + COLS * 5
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
    blocks, merged = [], {}
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
        key = (sparsity(r), meth)
        # Merge runs that share an arm. Extra seeds are run as separate wandb
        # runs (s3_8b_<arm>_seed0, ...), and a re-run after a crashed benchmark
        # leaves two runs for the same seed -- both would otherwise become
        # half-empty blocks under one label. Group here, pick the best row per
        # seed below, then compute mean/std once over the merged seeds.
        merged.setdefault(key, {'rows': {}, 'ids': [], 'states': [], 'want': 0})
        g = merged[key]
        g['ids'].append(r.id)
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
        if any(st_ != 'finished' for st_ in g['states']):
            label += f'  [{"/".join(sorted(set(g["states"])))}]'
        if len(full) >= 2:
            rows = rows + [['mean'] + agg(rows, st.mean), ['std'] + agg(rows, st.stdev)]
        rank = next((i for i, m in enumerate(ORDER) if meth.startswith(m) and
                     not any(meth.startswith(m2) and len(m2) > len(m) for m2 in ORDER)), len(ORDER))
        blocks.append(((sp, rank, meth), header(label, ' '.join(g['ids'])) + rows + [[''] * 22]))
    blocks.sort(key=lambda x: x[0])
    return [row for _, b in blocks for row in b]


def main():
    api = wandb.Api(timeout=180)
    for size, proj in PROJ.items():
        sheet = harvest(api, size, proj)
        os.makedirs(f'{HERE}/{size}', exist_ok=True)
        out = f'{HERE}/{size}/long_results_{size}.tsv'
        with open(out, 'w') as fh:
            for row in sheet:
                fh.write('\t'.join(fmt(i, c) for i, c in enumerate(row)) + '\n')
        print(f'wrote {out}  ({len(sheet)} rows)', file=sys.stderr)


if __name__ == '__main__':
    main()
