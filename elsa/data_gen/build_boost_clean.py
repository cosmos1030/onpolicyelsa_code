"""Leak-free boosted arm: v2 boosted rows whose response never mentions the hint, plus the leak-rejecting
regeneration rows (out/v2fix) that came out clean so far. FineWeb added at the 80/20 OT:FW ratio from fw16k.
Also writes the dense-self-generation (plain v2) arm on exactly the same problems, for a matched control."""
import json, glob, re
LEAK = re.compile(r'(provided|given|example|reference|sample) solution|the solution (says|mentions|shows|uses|given|provided)'
                  r'|solution (provided|given) (above|earlier)', re.I)
def resp(t): return t.split('<|im_start|>assistant\n', 1)[1]
def load(pat, off):
    R = {}
    for f in glob.glob(pat):
        for l in open(f):
            r = json.loads(l); R[r['idx'] + off] = r['text']
    return R
boost = {**load('out/v2/ot3_boost_*.jsonl', 0), **load('out/v2/ot3x_boost_*.jsonl', 8000)}
plain = {**load('out/v2/ot3_plain.jsonl', 0), **load('out/v2/ot3x_plain.jsonl', 8000)}
fix = {}
for f in glob.glob('out/v2fix/*.jsonl'):
    off = 8000 if '/ot3x_' in f else 0
    for l in open(f):
        r = json.loads(l); fix[r['idx'] + off] = r['text']
clean = {i: t for i, t in boost.items() if not LEAK.search(resp(t))}
nfix = 0
for i, t in fix.items():
    if not LEAK.search(resp(t)):
        clean[i] = t; nfix += 1
keys = sorted(clean)
fw = list(open('ot3/fw16k.jsonl'))[:len(keys) // 4]
for name, src in [('boostclean', clean), ('plainmatch', plain)]:
    with open(f'ot3/{name}.jsonl', 'w') as o:
        for i in keys:
            o.write(json.dumps({'text': src[i]}) + '\n')
        o.writelines(fw)
print(f'clean OT {len(keys)} (v2 clean {len(keys) - nfix}, from regeneration {nfix}) + FW {len(fw)} = {len(keys) + len(fw)} docs '
      f'-> {16384 / (len(keys) + len(fw)):.2f} epochs at 16,384 samples')
