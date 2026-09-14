#!/bin/bash
# Run this on the box that HAS the weights (the NHN /NHNHOME filesystem).
# Pushes the six ALPS-4B-S70 milestone checkpoints (512/1024/1536 x 2term/3term)
# to the Hub so an A100 box that does not mount /NHNHOME can score them.
#
# Why these six: --gmp_milestone_steps only SAVES an HF dir, it does not eval.
# The mseval_* jobs that were supposed to score them all failed on 2026-09-14
# (resume_eval_lighteval.py --tp_size defaults to 4, supervisor's eval1 kind
# hands out 1 GPU -> ray never gets a placement group -> 5/5 benchmarks timed
# out with exit 1). The weights are intact; only the scoring has to be redone.
#
#   PRIVATE=0 bash a100_scripts/push_milestones_to_hf.sh     # public, like the other 123
#   bash a100_scripts/push_milestones_to_hf.sh               # private (default)
set -euo pipefail

M=/NHNHOME/log-postech/doyoonkim/models
USER_NS=${USER_NS:-cosmos1030}
PRIVATE=${PRIVATE:-1}
export HF_TOKEN=$(cat /NHNHOME/log-postech/doyoonkim/secrets/hf_token)
PY=/NHNHOME/log-postech/doyoonkim/miniconda3/envs/rac/bin/python

# <local dir under $M>  <repo suffix>
ROWS=(
"gmp_s70pct_lr0.0001_step000512_20260913_150047_p2386905|alps4b-s70-2term-step000512"
"gmp_s70pct_lr0.0001_step001024_20260913_160242_p2386905|alps4b-s70-2term-step001024"
"gmp_s70pct_lr0.0001_step001536_20260913_170438_p2386905|alps4b-s70-2term-step001536"
"gmp_s70pct_lr0.0001_onpol_lmda0.33_step000512_20260913_152135_p2391204|alps4b-s70-3term-step000512"
"gmp_s70pct_lr0.0001_onpol_lmda0.33_step001024_20260913_163709_p2391204|alps4b-s70-3term-step001024"
"gmp_s70pct_lr0.0001_onpol_lmda0.33_step001536_20260913_175236_p2391204|alps4b-s70-3term-step001536"
)

for row in "${ROWS[@]}"; do
  d="${row%%|*}"; suffix="${row##*|}"
  src="$M/$d"; repo="$USER_NS/$suffix"
  [ -f "$src/config.json" ] || { echo "MISSING $src -- skipping"; continue; }
  echo "=== $repo  <-  $d ==="
  # ignore_patterns drops the empty lighteval_bench/ tree and the stale
  # eval_summary_resumed.json the failed 2026-09-14 run left behind; shipping
  # them would make the A100 run look like it already had results.
  PRIVATE="$PRIVATE" REPO="$repo" SRC="$src" $PY - <<'PYEOF'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo, src = os.environ["REPO"], os.environ["SRC"]
api.create_repo(repo_id=repo, exist_ok=True,
                private=os.environ["PRIVATE"] not in ("0", "false", ""))
api.upload_folder(
    folder_path=src, repo_id=repo,
    ignore_patterns=["lighteval_bench/**", "eval_summary_resumed.json",
                     ".zeroshot_done", ".eval_ctx.json"],
    commit_message="ALPS 4B S70 milestone checkpoint (weights only; unscored)",
)
print(f"pushed https://huggingface.co/{repo}")
PYEOF
done
echo "ALL DONE"
