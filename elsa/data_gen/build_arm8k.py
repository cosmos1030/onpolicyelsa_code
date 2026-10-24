"""Calibration/training arm files over the 8,000 OT3 prompts of prep_ot3_subset.py (+ the same 2,000 FineWeb):
ot3/boost8k.jsonl  boosted solutions (all 32 shards, every row kept -- unfinished / hint-phrased ones too)
ot3/plain8k.jsonl  one hint-free dense sample per prompt
ot3/orig.jsonl (from prep_ot3_subset.py) is the matching original-solution arm."""
import json, glob
fw = [l for l in open('ot3/fw.jsonl')]
for name, files in [('boost8k', sorted(glob.glob('out/ot3_boost_shard*.jsonl'))), ('plain8k', ['out/ot3_plain.jsonl'])]:
    rows = {}
    for f in files:
        for l in open(f):
            r = json.loads(l)
            rows[r['idx']] = r['text']
    assert len(rows) == 8000, (name, len(rows))
    with open(f'ot3/{name}.jsonl', 'w') as o:
        for i in sorted(rows):
            o.write(json.dumps({'text': rows[i]}) + '\n')
        o.writelines(fw)
    print(name, len(rows), '+', len(fw), 'FW')
