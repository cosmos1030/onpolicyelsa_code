"""v2 (8192-token budget, no end-of-turn token on unfinished responses) 16,384-doc arms, every row kept
(including the ~33% boosted rows whose thinking mentions the hint). Same OT order as orig16k.jsonl
(prompts.parquet then prompts_extra.parquet) and the same 3,276 FineWeb docs.
Also writes 8k (+2,000 FW) versions for the ALPS calibration arms, matching orig.jsonl."""
import json, glob
def load(pattern, n):
    rows = {}
    for f in glob.glob(pattern):
        for l in open(f):
            r = json.loads(l)
            rows[r['idx']] = r['text']
    assert len(rows) == n, (pattern, len(rows))
    return [rows[i] for i in sorted(rows)]
fw16 = list(open('ot3/fw16k.jsonl')); fw8 = list(open('ot3/fw.jsonl'))
for arm, pat in [('boost', 'out/v2/{}_boost_*.jsonl'), ('plain', 'out/v2/{}_plain.jsonl')]:
    a = load(pat.format('ot3'), 8000); b = load(pat.format('ot3x'), 5108)
    for name, ot, fw in [(f'{arm}16k_v2', a + b, fw16), (f'{arm}8k_v2', a, fw8)]:
        with open(f'ot3/{name}.jsonl', 'w') as o:
            for t in ot:
                o.write(json.dumps({'text': t}) + '\n')
            o.writelines(fw)
        print(name, len(ot), '+', len(fw))
