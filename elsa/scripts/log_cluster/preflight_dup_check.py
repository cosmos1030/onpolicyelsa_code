#!/usr/bin/env python3
"""Exit 1 if this exact eval is already running somewhere else.

Every launcher calls this before starting vLLM. Several servers pull from the
same wandb project and the same hub, and twice now two boxes spent hours on the
identical (checkpoint, seed) pair -- once on 1.7B SparseLLM s70, where the two
runs differed only in the word order of their names (s3_1.7b_sparsellm_s70 vs
s3_1.7b_s70_sparsellm), so name matching alone would not have caught it.

Identity is therefore the CHECKPOINT, not the run name: the basename of
--model_path (a hub repo id and a local copy of the same weights share it) plus
the seed set. A run counts as live only if wandb heard from it in the last 15
minutes; a crashed run holding a stale heartbeat must not block a rerun.
"""
import argparse, datetime, os, re, subprocess, sys


def ident(path):
    """A checkpoint's identity across servers.

    The same weights appear as a hub repo id on one box and a local directory on
    another (cosmos1030/sparsellm-qwen3-1.7b-s70pct vs models/qwen3_1.7b_sparsellm
    _s70pct), so a basename comparison misses the match that matters. For the
    one-shot baselines, (method, size, sparsity) pins the checkpoint exactly --
    there is only one SparseLLM 1.7B at 70%. Trained checkpoints are not
    interchangeable that way (many gmp runs share a sparsity), so those fall back
    to the basename.
    """
    b = os.path.basename(path.rstrip('/')).lower().replace('_', '-')
    meth = next((m for m in ('sparsellm', 'sparsegpt', 'sgpt', 'alps', 'safe') if m in b), None)
    if meth in ('sgpt',):
        meth = 'sparsegpt'
    if meth and 'gmp' not in b:
        size = next((z for z in ('1.7b', '4b', '8b') if z in b), '?')
        sp = re.search(r's(\d\d)pct|-s(\d\d)\b|s(\d\d)-', b)
        sp = next((g for g in (sp.groups() if sp else ()) if g), '?')
        selfgen = 'selfgen' if 'selfgen' in b else ''
        return f'{meth}|{size}|s{sp}|{selfgen}'
    return b


def live_here(name_hint):
    """A wandb run on THIS cluster can sit in state 'running' for minutes after
    its SLURM job was cancelled. Trust squeue over the heartbeat for our own."""
    try:
        out = subprocess.run(['squeue', '-u', os.environ.get('USER', ''), '-h', '-t', 'RUNNING', '-o', '%j'],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return True
    return bool(out.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--seeds', required=True)
    ap.add_argument('--run', default='')
    ap.add_argument('--project', required=True)
    ap.add_argument('--stale-min', type=float, default=15.0)
    a = ap.parse_args()

    key = ident(a.model)
    want = {s.strip() for s in a.seeds.split(',') if s.strip()}
    try:
        import wandb
        api = wandb.Api(timeout=90)
        runs = list(api.runs(f"dyk6208-gwangju-institute-of-science-and-technology/{a.project}",
                             filters={'state': 'running'}))
    except Exception as e:                      # never block a job on a flaky API
        print(f"[preflight] wandb unreachable ({type(e).__name__}) -- proceeding", flush=True)
        return 0

    now = datetime.datetime.now(datetime.timezone.utc)
    me = os.uname().nodename
    for r in runs:
        try:
            m = r.metadata or {}
        except Exception:
            continue
        if (m.get('program') or '').split('/')[-1] != 'eval_full.py':
            continue
        host = str(m.get('host') or '')
        if host == me:
            continue
        args = m.get('args', [])
        def g(f):
            return args[args.index(f) + 1] if f in args else None
        other = g('--model_path')
        if not other or ident(other) != key:
            continue
        theirs = {s.strip() for s in (g('--seeds') or g('--seed') or '').split(',') if s.strip()}
        if theirs and want and not (theirs & want):
            continue                            # different seeds -- not a duplicate
        hb = r.heartbeatAt
        if isinstance(hb, str):
            hb = datetime.datetime.fromisoformat(hb.replace('Z', '+00:00'))
        if hb.tzinfo is None:
            hb = hb.replace(tzinfo=datetime.timezone.utc)
        age = (now - hb).total_seconds() / 60
        # our own cluster: a cancelled job keeps a fresh-looking heartbeat for a
        # while, so require that SLURM still has something running.
        if host.startswith('log-') and not live_here(r.name):
            print(f"[preflight] {r.id} on {host} has no live SLURM job -- ignoring", flush=True)
            continue
        if age > a.stale_min:
            print(f"[preflight] {r.id} on {m.get('host')} matches but is {age:.0f}min stale -- ignoring", flush=True)
            continue
        print(f"[preflight] ALREADY RUNNING elsewhere: {r.id} '{r.name}' on {m.get('host')} "
              f"(seeds {sorted(theirs) or '?'}, heartbeat {age:.1f}min ago)", flush=True)
        print(f"[preflight] refusing to duplicate {key}", flush=True)
        return 1
    print(f"[preflight] no live duplicate of {key} seeds {sorted(want)}", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
