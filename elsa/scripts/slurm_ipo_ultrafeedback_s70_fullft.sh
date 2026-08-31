#!/bin/bash
#SBATCH --job-name=ipo_uf_s70_fullft
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --exclude=n3,n42,n46,n51,n52,n54,n55,n58,n60,n76,n77,n80,n91
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/ipo_uf_s70_fullft_%j.out
exec 2>&1

# Full-parameter (no LoRA) + MaskedAdam fork of slurm_ipo_ultrafeedback_s70.sh.
# That script's config used LoRA + merge_and_unload() for eval, which
# silently densified the pruned checkpoint's zero weights (the merge doesn't
# know about the pruning mask) -- verified on ipo_alpssft_s70_lr1e4_uf_lr1e-6_merged
# (zero_frac 0.70 -> ~0.0-0.0001 post-merge). This fork trains all parameters
# directly with DPOTrainer's new MaskedAdam-backed create_optimizer override
# (dpo_trainer.py) instead, so there's no merge step and no densification --
# the saved checkpoint IS the eval checkpoint.
#
# Usage: sbatch slurm_ipo_ultrafeedback_s70_fullft.sh <CONFIG_YAML> <OUTPUT_DIR> <TRAIN_WBID> [MAX_STEPS] [LR]
# e.g.:  sbatch slurm_ipo_ultrafeedback_s70_fullft.sh \
#          recipes/Qwen3-4B/dpo/config_ipo_ultrafeedback_s70_trgmp_fullft.yaml \
#          /home1/doyoonkim/projects/elsa/models/ipo_trgmp_s70_ultrafeedback_fullft_maskedadam \
#          657qk0cq

CONFIG_YAML=${1:?"Usage: <CONFIG_YAML> <OUTPUT_DIR> <TRAIN_WBID> [MAX_STEPS] [LR]"}
OUTPUT_DIR=${2:?"Usage: <CONFIG_YAML> <OUTPUT_DIR> <TRAIN_WBID> [MAX_STEPS] [LR]"}
TRAIN_WBID=${3:?"Usage: <CONFIG_YAML> <OUTPUT_DIR> <TRAIN_WBID> [MAX_STEPS] [LR]"}
MAX_STEPS=${4:--1}
LR_OVERRIDE=${5:-}

mkdir -p /local-data/user-data/$USER/job_$SLURM_JOB_ID/slurm
NFS_LOG_COPY="${OUTPUT_DIR}_slurmlog_${SLURM_JOB_ID}.out"
copy_log_to_nfs() {
  mkdir -p "$(dirname "$NFS_LOG_COPY")"
  cp "/local-data/user-data/$USER/job_$SLURM_JOB_ID/slurm/ipo_uf_s70_fullft_${SLURM_JOB_ID}.out" "$NFS_LOG_COPY" 2>/dev/null
}
trap copy_log_to_nfs EXIT

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rac_vllm084

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

cd /home1/doyoonkim/projects/RAC/open-r1-main
export PYTHONPATH=/home1/doyoonkim/projects/RAC/open-r1-main/src:/home1/doyoonkim/projects/RAC/open-r1-main/src/open_r1

echo "=== IPO (full-FT, MaskedAdam) on UltraFeedback: CONFIG=$CONFIG_YAML, MAX_STEPS=$MAX_STEPS, OUTPUT_DIR=$OUTPUT_DIR ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID  NODE=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

EXTRA_ARGS=()
if [ -n "$LR_OVERRIDE" ]; then
  EXTRA_ARGS+=(--learning_rate "$LR_OVERRIDE")
fi

accelerate launch --config_file recipes/plain_1gpu.yaml src/open_r1/dpo.py \
  --config "$CONFIG_YAML" \
  --output_dir "$OUTPUT_DIR" \
  --max_steps "$MAX_STEPS" \
  "${EXTRA_ARGS[@]}"
DPO_EXIT=$?

echo "=== DPO/IPO training done (exit $DPO_EXIT) ==="

EVAL_EXIT=0
if [ $DPO_EXIT -eq 0 ]; then
  # No merge step -- $OUTPUT_DIR is a plain full-parameter checkpoint already.
  # Full reasoning-bench eval (math500/lcb/gpqa/ifeval/gsm8k), same script and
  # "quick"=8192-budget profile already fixed for the GRPO eval jobs, so this
  # result is directly comparable to the GRPO/pre-GRPO table (not just MATH-500).
  echo "=== Running full reasoning-bench eval (quick/8192 profile) on $OUTPUT_DIR ==="
  conda deactivate
  conda activate rac
  bash /home1/doyoonkim/projects/elsa/scripts/slurm_gmp_eval_only.sh "$OUTPUT_DIR" "$TRAIN_WBID" 0.7 quick
  EVAL_EXIT=$?
  echo "=== Eval done (exit $EVAL_EXIT) ==="
fi

echo "=== DONE (dpo exit $DPO_EXIT, eval exit $EVAL_EXIT) ==="
exit $DPO_EXIT
