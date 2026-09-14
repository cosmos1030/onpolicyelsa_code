#!/bin/bash
# Print milestone trajectories from whatever has finished so far.
# avg5 is the mean of the same five tasks resume_eval_lighteval.py averages.
#
# Roots are searched for */eval_summary_resumed.json, so this covers both
# layouts: the 4B milestones were pulled from the Hub into a work dir, while
# milestones trained on this cluster are scored in place under elsa/models.
#
#   bash a100_scripts/collect_milestone_scores.sh [ROOT ...]
ROOTS=("$@")
if [ ${#ROOTS[@]} -eq 0 ]; then
    ROOTS=(/home1/doyoonkim/projects/elsa/logs/milestone_eval
           /home1/doyoonkim/projects/elsa/models)
fi
/home1/doyoonkim/miniconda3/envs/rac/bin/python - "${ROOTS[@]}" <<'PY'
import glob, json, os, re, sys
KEYS = ["lighteval/math500", "lighteval/lcb", "lighteval/gpqa_diamond",
        "lighteval/gsm8k", "lighteval/ifeval_prompt"]
SHORT = ["math500", "lcb", "gpqa", "gsm8k", "ifeval"]
ENDPOINT = 10 ** 6   # sorts after every real step number

rows = []
for root in sys.argv[1:]:
    for f in glob.glob(os.path.join(root, "*", "eval_summary_resumed.json")):
        name = os.path.basename(os.path.dirname(f))
        # A step-keyed save carries stepNNNNNN; a run's FINAL save does not, so
        # matching only on stepNNNNNN would drop each arm's endpoint from its
        # own trajectory.
        m = re.search(r"step(\d{6})", name)
        if m:
            step = int(m.group(1))
        elif re.match(r"^gmp_s\d+pct_", name):
            step = ENDPOINT
        else:
            continue
        # 3-term arms carry the OPKD lambda in their tag; 2-term arms do not.
        arm = "3term" if ("3term" in name or "onpol_lmda" in name) else "2term"
        rows.append((arm, step, name, f))

if not rows:
    print("아직 채점된 마일스톤이 없습니다.")
    raise SystemExit

def label_of(name, step):
    return name + ("  [endpoint]" if step == ENDPOINT else "")

w = max(34, max(len(label_of(r[2], r[1])) for r in rows) + 2)
print(f"{'checkpoint':<{w}}{'avg5':>7}" + "".join(f"{s:>9}" for s in SHORT))
print("-" * (w + 7 + 9 * len(SHORT)))
for arm, step, name, f in sorted(rows):
    d = json.load(open(f))
    v = [d.get(k) for k in KEYS]
    label = label_of(name, step)
    if all(isinstance(x, (int, float)) for x in v):
        print(f"{label:<{w}}{100*sum(v)/5:7.2f}" + "".join(f"{100*x:9.2f}" for x in v))
    else:
        got = sum(isinstance(x, (int, float)) for x in v)
        print(f"{label:<{w}}{'—':>7}   ({got}/5 태스크만)")
print()
print("4B 2048 엔드포인트 (B200 tp=4): 2term 39.12 / 3term 40.93")
print("4B 마일스톤은 A100 tp=1 — 하드웨어·TP 경로가 다릅니다")
print("1.7B는 마일스톤·엔드포인트 모두 A100 tp=1 동일 경로")
PY
