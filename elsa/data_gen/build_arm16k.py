"""16,384-doc NTP arms (13,108 OT3 problems + the same 3,276 FineWeb), matching ot3/orig16k.jsonl:
ot3/boost16k.jsonl  boosted solutions: 8,000 (ot3_boost_shard*) then 5,108 (ot3x_boost_shard*), every row kept
ot3/plain16k.jsonl  one hint-free dense sample per problem (ot3_plain + ot3x_plain)
OT order = prompts.parquet then prompts_extra.parquet, the order orig16k.jsonl was written in."""
import json, glob
fw = list(open('ot3/fw16k.jsonl'))
def load(files, n):
    rows = {}
    for f in files:
        for l in open(f):
            r = json.loads(l)
            rows[r['idx']] = r['text']
    assert len(rows) == n, (files[:1], len(rows))
    return [rows[i] for i in sorted(rows)]
arms = {
    'boost16k': load(sorted(glob.glob('out/ot3_boost_shard*.jsonl')), 8000) + load(sorted(glob.glob('out/ot3x_boost_shard*.jsonl')), 5108),
    'plain16k': load(['out/ot3_plain.jsonl'], 8000) + load(['out/ot3x_plain.jsonl'], 5108),
}
for name, ot in arms.items():
    with open(f'ot3/{name}.jsonl', 'w') as o:
        for t in ot:
            o.write(json.dumps({'text': t}) + '\n')
        o.writelines(fw)
    print(name, len(ot), '+', len(fw), 'FW =', len(ot) + len(fw))
