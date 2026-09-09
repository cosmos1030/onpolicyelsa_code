#!/usr/bin/env python3
"""Harvest every SCOUT/pgd2growth run result out of the raw training logs into
one durable table.  Replaces the ad-hoc pull.py that was lost with a container.

Idempotent and re-runnable: it never edits logs, only reads them, so it can be
re-run after any run finishes.  Two outputs next to this file:
  RESULTS_ALL.tsv  -- one row per run, machine readable
  RESULTS_ALL.md   -- the same grouped into paper-shaped tables

The 5 quick-profile benchmarks are the paper's Avg set (math500, gpqa, ifeval,
lcb, gsm8k) -- ALL FIVE, GPQA included.  A run missing any of them gets Avg=None
rather than a mean over whatever happened to finish, so a partial eval can never
masquerade as a complete one.
"""
import re, os, sys, json, glob

LOGROOT = '/NHNHOME/log-postech/doyoonkim/logs'
BENCH = ['math500', 'gpqa_diamond', 'ifeval_prompt', 'lcb', 'gsm8k']
SHORT = {'math500': 'MATH', 'gpqa_diamond': 'GPQA', 'ifeval_prompt': 'IFEval',
         'lcb': 'LCB', 'gsm8k': 'GSM8K'}

RE_BENCH = re.compile(r'\[lighteval_bench\] lighteval/([a-z0-9_:]+): ([0-9.]+)')
RE_STEP  = re.compile(r'Step (\d+)/(\d+)')
RE_TRUNC = re.compile(r'\[lighteval_bench\] ([a-z0-9]+)_truncation_rate: ([0-9.]+)')
RE_COMP  = re.compile(r'\[pgd_nm_compensate\] step=(\d+) adjusted ([\d,]+) survivor '
                      r'weights, max \|delta\|=([0-9.e+-]+)')
RE_NOOP  = re.compile(r'\[pgd_nm_compensate\] step=\d+ adjusted NOTHING')
RE_SAVE  = re.compile(r'(gmp_s\d+pct_lr[0-9.e-]+_onpol_lmda[0-9.]+_\d{8}_\d{6})')

# Config is NOT read off the command line -- an expanded command line only
# appears in a log when the process crashed and bash dumped it, so parsing it
# silently left every clean run unlabelled.  Two sources that are present in
# EVERY log instead: the run directory name wandb prints (which the launcher
# builds out of the actual resolved knobs) and the trainer's own "ENABLED"
# banner lines for the optional features.
RE_RUNDIR = re.compile(r'logs/(gmp_[a-z0-9_.=-]+?)/wandb')
RE_HEADER = re.compile(r'^=== .*?Qwen3-(\S+?)\s', re.M)
RE_TARGET = re.compile(r'Target sparsity = ([0-9.]+), mask_interval = (\d+)')
RE_JUMP   = re.compile(r'PGD one-shot jump ENABLED')
RE_COMPON = re.compile(r'\[pgd_nm_compensate\] ENABLED')
RE_RULE   = re.compile(r'grow_rule=(\w+)|gmp_pgd_grow_rule=(\w+)')
RE_RO     = re.compile(r'On-policy KD: lambda=[0-9.]+, interval=(\d+)')
RE_FROZENM= re.compile(r'frozen-pool mode \(onpolicy_interval=(\d+) >= steps')

RUNDIR_BITS = {
    'model':    re.compile(r'gmp_[a-z_]*?_(1\.7b|4b|8b)_'),
    'nm':       re.compile(r'_(24)_'),
    'sparsity': re.compile(r'_s(\d+)(?:pct)?_'),
    'kl':       re.compile(r'_klb([0-9.]+)'),
    'lr':       re.compile(r'_lr([0-9.e-]+?)_'),
}

def model_of(path):
    for tag, name in (('_4b', '4B'), ('_8b', '8B'), ('4b_', '4B'), ('8b_', '8B'),
                      ('_1.7b', '1.7B')):
        if tag in path.lower():
            return name
    return '?'


def scan(path):
    r = {'log': os.path.relpath(path, LOGROOT), 'scores': {}, 'trunc': {},
         'comp_events': [], 'comp_noop': False, 'step': None, 'total': None,
         'ckpt': None, 'cfg': {}}
    try:
        txt = open(path, errors='replace').read()
    except OSError:
        return None
    for b, v in RE_BENCH.findall(txt):
        r['scores'][b] = float(v)          # last write wins = final eval
    for b, v in RE_TRUNC.findall(txt):
        r['trunc'][b] = float(v)
    for st, n, mx in RE_COMP.findall(txt):
        r['comp_events'].append((int(st), int(n.replace(',', '')), float(mx)))
    r['comp_noop'] = bool(RE_NOOP.search(txt))
    steps = RE_STEP.findall(txt)
    if steps:
        r['step'], r['total'] = int(steps[-1][0]), int(steps[-1][1])
    saves = RE_SAVE.findall(txt)
    if saves:
        r['ckpt'] = saves[-1]
    rd = RE_RUNDIR.findall(txt)
    r['rundir'] = rd[-1] if rd else None
    if r['rundir']:
        for k, rx in RUNDIR_BITS.items():
            m = rx.search(r['rundir'])
            if m:
                r['cfg'][k] = m.group(1)
    m = RE_TARGET.search(txt)
    if m:
        r['cfg']['sparsity_exact'] = m.group(1)
    if RE_JUMP.search(txt):
        r['cfg']['jump'] = 'true'
    if RE_COMPON.search(txt):
        r['cfg']['comp'] = 'true'
    m = RE_RULE.search(txt)
    if m:
        r['cfg']['grow_rule'] = m.group(1) or m.group(2)
    m = RE_RO.findall(txt)
    if m:
        r['cfg']['ro'] = m[-1]
    if RE_FROZENM.search(txt):
        r['cfg']['frozen'] = 'true'
    # The wandb run dir is NOT unique per arm -- the jump and jump+frozen 4B
    # arms share it exactly -- so the log's own basename, which is how the queue
    # named each arm, is the authoritative arm tag.
    base = os.path.basename(path)[:-len('.log')]
    r['arm'] = re.sub(r'(_resume|_retry|_v\d+)$', '', base)
    r['is_resume'] = base.endswith('_resume')
    r['mtime'] = os.path.getmtime(path)
    # The model column used to come from the PATH while the label came from the
    # run-dir config, so a log whose directory name lacks "8b" (e.g.
    # resweep2_opkdfix/n24_klb0.01_resume.log) was labelled "8B ..." but filed
    # under model "?", and any query filtering on the column silently dropped it.
    # Config wins; the path is only the fallback.
    r['model'] = (r['cfg'].get('model') or model_of(path)).upper()
    have = [b for b in BENCH if b in r['scores']]
    r['avg'] = (100 * sum(r['scores'][b] for b in BENCH) / len(BENCH)
                if len(have) == len(BENCH) else None)
    r['n_bench'] = len(have)
    return r


def label(r):
    c = r['cfg']
    if c.get('nm') == '24':
        sp = '2:4'
    elif c.get('sparsity'):
        sp = 'S%s' % c['sparsity']
    elif c.get('sparsity_exact'):
        sp = 'S%d' % round(100 * float(c['sparsity_exact']))
    else:
        sp = '?'
    bits = [c.get('model', r['model']).upper().replace('1.7B', '1.7B'), sp]
    if c.get('kl'):
        bits.append('d=%s' % c['kl'])
    if c.get('lr'):
        bits.append('lr=%s' % c['lr'])
    if c.get('grow_rule') and c['grow_rule'] != 'kl':
        bits.append('rule=%s' % c['grow_rule'])
    if c.get('jump') == 'true':
        bits.append('jump')
    if c.get('comp') == 'true':
        bits.append('+comp*' if (r['comp_noop'] or not r['comp_events']) else '+comp')
    if c.get('frozen') == 'true':
        bits.append('B1frozen(ro=%s)' % c.get('ro', '?'))
    elif c.get('ro') and c['ro'] not in ('32',):
        bits.append('ro=%s' % c['ro'])
    return ' '.join(bits)


def main():
    logs = sorted(set(glob.glob(LOGROOT + '/*/*.log') + glob.glob(LOGROOT + '/*.log')))
    rows = []
    for p in logs:
        if '/wandb/' in p or 'watchdog' in p:
            continue
        r = scan(p)
        if r and (r['scores'] or r['step']):
            r['label'] = label(r)
            rows.append(r)

    # A run that was killed mid-flight and finished under a *_resume log must not
    # be listed as if it were still going: the resume carries the real result.
    resumed_arms = {r['arm'] for r in rows if r.get('is_resume')}
    for r in rows:
        r['superseded'] = (not r.get('is_resume') and r['arm'] in resumed_arms
                           and r['avg'] is None)
    import time
    now = time.time()
    for r in rows:
        r['state'] = ('DONE' if r['avg'] is not None else
                      'SUPERSEDED' if r['superseded'] else
                      'LIVE' if now - r['mtime'] < 900 else 'DEAD')
    with open(LOGROOT + '/RESULTS_ALL.tsv', 'w') as f:
        f.write('\t'.join(['label', 'arm', 'state', 'model', 'avg5']
                          + [SHORT[b] for b in BENCH]
                          + ['n_bench', 'step', 'total', 'comp_events',
                             'comp_maxdelta', 'ckpt', 'log']) + '\n')
        for r in sorted(rows, key=lambda x: (x['avg'] is None, -(x['avg'] or 0))):
            ce = len(r['comp_events'])
            md = max((e[2] for e in r['comp_events']), default='')
            f.write('\t'.join([
                r['label'], r['arm'], r['state'], r['model'],
                '%.2f' % r['avg'] if r['avg'] is not None else '',
                *['%.2f' % (100 * r['scores'][b]) if b in r['scores'] else ''
                  for b in BENCH],
                str(r['n_bench']), str(r['step'] or ''), str(r['total'] or ''),
                str(ce), str(md), r['ckpt'] or '', r['log']]) + '\n')

    done = [r for r in rows if r['state'] == 'DONE']
    live = [r for r in rows if r['state'] in ('LIVE', 'DEAD')]
    with open(LOGROOT + '/RESULTS_ALL.md', 'w') as f:
        f.write('# All harvested results (auto-generated by harvest_results.py)\n\n')
        f.write('Avg is the mean of ALL FIVE quick-profile benchmarks '
                '(MATH, GPQA, IFEval, LCB, GSM8K). Blank Avg = eval incomplete.\n')
        f.write('`+comp*` marks a run launched with compensation that adjusted '
                'NOTHING (FSDP flat-shard no-op) -- it is the plain baseline.\n\n')
        f.write('## Completed (%d)\n\n' % len(done))
        f.write('| Run | Avg | %s | log |\n' % ' | '.join(SHORT[b] for b in BENCH))
        f.write('|---|---|%s|---|\n' % ('---|' * len(BENCH)))
        for r in sorted(done, key=lambda x: -x['avg']):
            f.write('| %s | **%.2f** | %s | `%s` |\n' % (
                r['label'], r['avg'],
                ' | '.join('%.2f' % (100 * r['scores'][b]) for b in BENCH), r['log']))
        f.write('\n## In flight / incomplete (%d)\n\n' % len(live))
        f.write('| Run | state | progress | benches | comp events | max delta | log |\n'
                '|---|---|---|---|---|---|---|\n')
        for r in sorted(live, key=lambda x: (x['state'] != 'LIVE', -(x['step'] or 0))):
            md = max((e[2] for e in r['comp_events']), default=None)
            f.write('| %s | %s | %s/%s | %d/5 | %d | %s | `%s` |\n' % (
                r['label'], r['state'], r['step'], r['total'], r['n_bench'],
                len(r['comp_events']), ('%.2e' % md) if md else '-', r['log']))
        sup = [r for r in rows if r['state'] == 'SUPERSEDED']
        f.write('\n## Superseded by a resume (%d) -- result lives in the *_resume row\n\n'
                % len(sup))
        for r in sorted(sup, key=lambda x: x['arm']):
            f.write('- `%s` (died at %s/%s)\n' % (r['log'], r['step'], r['total']))
    print('wrote RESULTS_ALL.tsv / RESULTS_ALL.md : %d completed, %d in flight'
          % (len(done), len(live)))


if __name__ == '__main__':
    main()
