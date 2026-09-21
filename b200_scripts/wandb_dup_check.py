#!/usr/bin/env python3
"""Is this eval already done, or running somewhere else? Ask wandb before
spending 6 GPU-hours on it.

    wandb_dup_check.py <project> <run_name> <seed> [model_path]
    exit 0 -> duplicate found, SKIP      exit 1 -> none, RUN
    exit 2 -> could not check (network) -- caller decides

Two boxes and a SLURM cluster have been evaluating the same checkpoints
without seeing each other: on 2026-09-21 this box spent 4.8 h on a 4B s70
ALPS+retrain eval that n84 was already running. The unit that matters is a
cell of the long TSV, so a run counts as the same eval when
harvest_long_tsv.py would put it in the same block and seed row: same label
(its own auto_label rules, parsed out of that file so they cannot drift),
same sparsity, same seed. A run whose --model_path is the same checkpoint
counts too, whatever it was named.

It is a duplicate only if it can still produce the numbers:
  finished -> all five benchmarks at the long-profile cap for that seed
              (a run that lost LCB/GSM8K to OOM is NOT a reason to skip)
  running  -> heartbeat within 30 min (a dead container leaves runs
              'running' forever) and its --seeds include ours
"""
import ast, datetime, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
HARVEST = os.path.join(HERE, '..', 'long_tsv_results', 'harvest_long_tsv.py')
ENT = 'dyk6208-gwangju-institute-of-science-and-technology'
CAP = {'math500': 16384, 'gpqa': 16384, 'ifeval': 16384, 'lcb': 32768, 'gsm8k': 8192}

def harvest_tables():
    t = ast.parse(open(HARVEST).read())
    d = {}
    for n in t.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') in ('LABELS', 'METHOD'):
            d[n.targets[0].id] = ast.literal_eval(n.value)
    return d['LABELS'], d['METHOD']

LABELS, METHOD = harvest_tables()

def label(rid, name):
    if rid in LABELS:
        return LABELS[rid]
    n = re.sub(r'_s42$', '', name)
    n = re.sub(r'^s3_[\d.]+b_', '', n)
    n = re.sub(r'_seeds?\d+$', '', n)
    n = re.sub(r'_ep\d+$', '', n)
    n = re.sub(r'(^|_)s\d\d(?=_|$)', '', n).strip('_')
    return METHOD.get(n, name)

def sparsity(name):
    m = re.search(r'(?:^|_)s(\d\d)(?:pct|_|$)', name)
    return int(m.group(1)) if m else 0

def ckpt_key(p):
    return os.path.basename(str(p).rstrip('/')).replace('-', '_').lower() if p else ''

def argval(args, flag):
    return args[args.index(flag) + 1] if flag in args and args.index(flag) + 1 < len(args) else None

def main():
    proj, run_name, seed = sys.argv[1], sys.argv[2], str(sys.argv[3])
    mkey = ckpt_key(sys.argv[4]) if len(sys.argv) > 4 else ''
    want_label, want_sp = label('', run_name), sparsity(run_name)
    import wandb
    api = wandb.Api(timeout=120)
    now = datetime.datetime.now(datetime.timezone.utc)
    # Server-side filter, or a 4B project scan takes minutes per check. A run
    # can land in this cell three ways: its name resolves to the same METHOD
    # key, it is a LABELS-by-id run with that label, or its model_path is the
    # same checkpoint (hub id and local dir differ only by '-' vs '_').
    keys = [k for k, v in METHOD.items() if v == want_label]
    ors = [{'name': {'$in': [i for i, v in LABELS.items() if v == want_label] or ['-']}}]
    if keys:
        ors.append({'display_name': {'$regex': '(^|_)(' + '|'.join(map(re.escape, keys)) + ')(_|$)'}})
    if len(sys.argv) > 4:
        # wandb only exact-matches config values ($regex on config fields
        # silently returns nothing), so enumerate the spellings one
        # checkpoint goes by: the path as given, its Hub id as push_ckpts.sh
        # names it (dir name, '_' -> '-'), and the local dir that id came from.
        # Older Hub ids (gmp-kd3e-1-...) do not map back to a dir name; those
        # are caught by the label match instead, as long as runs are named
        # s3_<size>_<arm>_s<NN>_seed<N>.
        given = sys.argv[4].rstrip('/')
        base = os.path.basename(given)
        variants = {given,
                    'cosmos1030/' + base.replace('_', '-'),
                    '/NHNHOME/log-postech/doyoonkim/models/' + base.replace('-', '_')}
        ors.append({'config.model_path': {'$in': sorted(variants)}})
    for r in api.runs(f'{ENT}/{proj}', filters={'$or': ors}, per_page=200):
        args = (r.metadata or {}).get('args') or []
        same_ckpt = bool(mkey) and ckpt_key(argval(args, '--model_path')) == mkey
        same_cell = label(r.id, r.name) == want_label and sparsity(r.name) == want_sp
        if not (same_ckpt or same_cell):
            continue
        s = r.summary._json_dict if hasattr(r.summary, '_json_dict') else dict(r.summary)
        seeds = (argval(args, '--seeds') or argval(args, '--seed') or '42').split(',')
        if r.state == 'running':
            hb = getattr(r, 'heartbeatAt', None) or getattr(r, 'heartbeat_at', None)
            try:
                age = (now - datetime.datetime.fromisoformat(str(hb).replace('Z', '+00:00'))).total_seconds()
            except Exception:
                age = 1e9
            if age < 1800 and seed in seeds:
                print(f'DUP running {r.id} {r.name} host={(r.metadata or {}).get("host")} seeds={seeds}')
                return 0
            continue
        sufs = [f'_seed{seed}'] + ([''] if seeds == [seed] or (seed == '42' and len(seeds) == 1) else [])
        for suf in sufs:
            ok = sum(1 for b, c in CAP.items()
                     if isinstance(s.get(f'{b}_avg_gen_cap{suf}'), (int, float))
                     and round(s[f'{b}_avg_gen_cap{suf}']) == c)
            if ok == 5:
                print(f'DUP finished {r.id} {r.name} seed={seed} (5/5 long)')
                return 0
    print(f'no dup for {run_name} seed={seed} label={want_label!r} s{want_sp}')
    return 1

if __name__ == '__main__':
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:
        print(f'CHECK FAILED: {type(e).__name__}: {e}')
        sys.exit(2)
