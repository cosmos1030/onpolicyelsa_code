#!/bin/bash
#SBATCH --job-name=hf_upload
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-04:00:00
#SBATCH --output=/local-data/user-data/doyoonkim/logs/hf_upload_%j.out
#SBATCH --exclude=n3,n42,n51,n52,n54,n55,n58,n60,n76,n77,n80
exec 2>&1

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL_BASE=/home1/doyoonkim/projects/elsa/models

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export HF_HOME=/home1/doyoonkim/.cache/huggingface
export HF_DATASETS_OFFLINE=0
unset TRANSFORMERS_OFFLINE
unset HF_HUB_OFFLINE

mkdir -p /local-data/user-data/doyoonkim/logs

echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
echo "HF_TOKEN set: $([ -n "$HF_TOKEN" ] && echo YES || echo NO)"

upload() {
    local model_path=$1
    local repo_id=$2
    echo ""
    echo "=== Uploading: $repo_id ==="
    echo "    from: $model_path"
    if [ ! -d "$model_path" ]; then
        echo "    SKIP: directory not found"
        return
    fi
    $PYTHON -c "
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ['HF_TOKEN'])
api.create_repo('$repo_id', exist_ok=True, private=False)
api.upload_folder(
    folder_path='$model_path',
    repo_id='$repo_id',
    repo_type='model',
    ignore_patterns=['*.bin', '*.pt', 'optimizer*', 'scheduler*', 'trainer_state*', 'training_args*'],
)
print('    DONE: https://huggingface.co/$repo_id')
"
    if [ $? -ne 0 ]; then echo "    FAILED"; fi
}

# ── ELSA NTP-ADMM ────────────────────────────────────────────────────────────
upload \
  "$MODEL_BASE/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e_pruned0.5_kd_unknown_admm_lr0.0001_lmda0.001_kdlam1.0_20260719_152800" \
  "cosmos1030/elsa-ntp-admm-qwen3-1.7b-s50pct"

upload \
  "$MODEL_BASE/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e_pruned0.6_kd_unknown_admm_lr0.0001_lmda0.001_kdlam1.0_20260719_153938" \
  "cosmos1030/elsa-ntp-admm-qwen3-1.7b-s60pct"

upload \
  "$MODEL_BASE/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e_pruned0.7_kd_unknown_admm_lr0.0001_lmda0.005_kdlam1.0_20260720_052233" \
  "cosmos1030/elsa-ntp-admm-qwen3-1.7b-s70pct"

# ── SparseLLM ─────────────────────────────────────────────────────────────────
upload "$MODEL_BASE/qwen3_1.7b_sparsellm_s50pct" "cosmos1030/sparsellm-qwen3-1.7b-s50pct"
upload "$MODEL_BASE/qwen3_1.7b_sparsellm_s60pct" "cosmos1030/sparsellm-qwen3-1.7b-s60pct"
upload "$MODEL_BASE/qwen3_1.7b_sparsellm_s70pct" "cosmos1030/sparsellm-qwen3-1.7b-s70pct"

# ── ALPS ──────────────────────────────────────────────────────────────────────
upload "$MODEL_BASE/qwen3_1.7b_alps_s50pct" "cosmos1030/alps-qwen3-1.7b-s50pct"
upload "$MODEL_BASE/qwen3_1.7b_alps_s60pct" "cosmos1030/alps-qwen3-1.7b-s60pct"
upload "$MODEL_BASE/qwen3_1.7b_alps_s70pct" "cosmos1030/alps-qwen3-1.7b-s70pct"

echo ""
echo "##### ALL UPLOADS DONE #####"

cp /local-data/user-data/doyoonkim/logs/hf_upload_${SLURM_JOB_ID}.out \
   /home1/doyoonkim/projects/elsa/logs/hf_upload_${SLURM_JOB_ID}.out 2>/dev/null || true
