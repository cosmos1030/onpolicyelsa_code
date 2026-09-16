#!/bin/bash
# Push the s80 ALPS+retrain checkpoints to the Hub so a box that does not mount
# /NHNHOME can score them. Run on the B200 box after the four runs finish.
#
# The B200 only evaluates each run's FINAL checkpoint, once, at profile=long
# seed=42. Everything else -- the 512/1024/1536 milestones, and seeds 0 and 1 --
# is scored elsewhere, which is why these have to leave this filesystem.
#
#   bash a100_scripts/push_s80_alps_retrain_to_hf.sh            # dry run, lists what it would push
#   PUSH=1 bash a100_scripts/push_s80_alps_retrain_to_hf.sh     # actually push
set -euo pipefail
M=/NHNHOME/log-postech/doyoonkim/models
USER_NS=${USER_NS:-cosmos1030}
PUSH=${PUSH:-0}
PY=/NHNHOME/log-postech/doyoonkim/miniconda3/envs/rac/bin/python
export HF_TOKEN=$(cat /NHNHOME/log-postech/doyoonkim/secrets/hf_token)

# _run_tag now carries the model size, and the OPKD lambda separates the arms:
# the 3-term arm gets an _onpol_lmda0.33 segment, the 2-term arm (OPKD 0.0) does
# not. That is the only thing in the directory name that tells them apart.
# A SCOUT s80 run saves under the same stem as the 3-term arm here
# (gmp_4b_s80pct_lr0.0001_onpol_lmda0.33_...): nothing in the directory name
# says ALPS-vs-SCOUT or fixed-vs-grown mask. SINCE_HOURS keeps this to the
# window the four ALPS+retrain runs wrote in; the dry run exists so the list
# gets read before anything is published.
SINCE_HOURS=${SINCE_HOURS:-24}
shopt -s nullglob
for d in "$M"/gmp_{4b,8b}_s80pct_lr0.0001*; do
  [ -f "$d/config.json" ] || continue
  [ -z "$(find "$d" -maxdepth 0 -mmin -$((SINCE_HOURS*60)))" ] && continue
  b=$(basename "$d")
  case "$b" in *_onpol_lmda*) arm=3term ;; *) arm=2term ;; esac
  case "$b" in gmp_8b_*) size=8b ;; *) size=4b ;; esac
  if [[ "$b" =~ _step([0-9]{6})_ ]]; then step="step${BASH_REMATCH[1]}"; else step="final"; fi
  repo="$USER_NS/alps${size}-s80-${arm}-${step}"
  echo "$b  ->  $repo"
  [ "$PUSH" = "1" ] || continue
  REPO="$repo" SRC="$d" $PY - <<'PYEOF'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo, src = os.environ["REPO"], os.environ["SRC"]
api.create_repo(repo_id=repo, exist_ok=True, private=False)
# The final dirs carry a lighteval_bench/ tree from their own long/seed-42 run.
# Shipping it would make the receiving box look like it already had results.
api.upload_folder(folder_path=src, repo_id=repo,
                  ignore_patterns=["lighteval_bench/**", "eval_summary_resumed.json",
                                   ".zeroshot_done", ".eval_ctx.json"],
                  commit_message="ALPS s80 + retrain checkpoint (weights only; unscored)")
print(f"  pushed https://huggingface.co/{repo}")
PYEOF
done
