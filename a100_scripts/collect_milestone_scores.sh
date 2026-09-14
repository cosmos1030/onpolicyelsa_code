#!/bin/bash
# Print the milestone trajectory from whatever has finished so far.
# avg5 is the mean of the same five tasks resume_eval_lighteval.py averages.
W=${WORK:-/home1/doyoonkim/projects/elsa/logs/milestone_eval}
/home1/doyoonkim/miniconda3/envs/rac/bin/python - "$W" <<'PY'
import json, os, sys
W = sys.argv[1]
KEYS = ["lighteval/math500", "lighteval/lcb", "lighteval/gpqa_diamond",
        "lighteval/gsm8k", "lighteval/ifeval_prompt"]
SHORT = ["math500", "lcb", "gpqa", "gsm8k", "ifeval"]
print(f"{'checkpoint':<34}{'avg5':>7}" + "".join(f"{s:>9}" for s in SHORT))
print("-" * 34 + "-" * (7 + 9 * len(SHORT)))
for term in ("2term", "3term"):
    for step in ("000512", "001024", "001536"):
        name = f"alps4b-s70-{term}-step{step}"
        f = os.path.join(W, name, "eval_summary_resumed.json")
        if not os.path.exists(f):
            print(f"{name:<34}{'—':>7}   (아직)")
            continue
        m = json.load(open(f))
        v = [m.get(k) for k in KEYS]
        if all(isinstance(x, (int, float)) for x in v):
            print(f"{name:<34}{100*sum(v)/5:7.2f}" + "".join(f"{100*x:9.2f}" for x in v))
        else:
            got = sum(isinstance(x, (int, float)) for x in v)
            print(f"{name:<34}{'—':>7}   ({got}/5 태스크만)")
print()
print("2048 엔드포인트 (B200 tp=4): 2term 39.12 / 3term 40.93")
print("위 6점은 A100 tp=1 — 하드웨어·TP 경로가 다릅니다")
PY
