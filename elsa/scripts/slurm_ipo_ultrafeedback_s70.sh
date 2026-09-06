#!/bin/bash
#SBATCH --job-name=ipo_ultrafeedback_s70
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n91
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/ipo_ultrafeedback_s70_%j.out
exec 2>&1

# Usage: sbatch slurm_ipo_ultrafeedback_s70.sh [MAX_STEPS] [OUTPUT_DIR] [MODEL_NAME_OR_PATH] [LEARNING_RATE]
# Small-scale verification first: sbatch slurm_ipo_ultrafeedback_s70.sh 10 /home1/doyoonkim/projects/elsa/models/ipo_trgmp_s70_smoketest

MAX_STEPS=${1:--1}
OUTPUT_DIR=${2:-/home1/doyoonkim/projects/elsa/models/ipo_trgmp_s70_ultrafeedback}
MODEL_OVERRIDE=${3:-}
LR_OVERRIDE=${4:-}

mkdir -p /local-data/user-data/$USER/job_$SLURM_JOB_ID/slurm
NFS_LOG_COPY="${OUTPUT_DIR}_slurmlog_${SLURM_JOB_ID}.out"
copy_log_to_nfs() {
  mkdir -p "$(dirname "$NFS_LOG_COPY")"
  cp "/local-data/user-data/$USER/job_$SLURM_JOB_ID/slurm/ipo_ultrafeedback_s70_${SLURM_JOB_ID}.out" "$NFS_LOG_COPY" 2>/dev/null
}
trap copy_log_to_nfs EXIT

source ~/miniconda3/etc/profile.d/conda.sh
# Reuse rac_vllm084 -- same env already proven to work for this vendored
# open_r1_trl codebase (GRPO run). DPO/IPO doesn't touch vLLM at all, this
# is just for consistency/safety, not a functional requirement.
conda activate rac_vllm084

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

cd /home1/doyoonkim/projects/RAC/open-r1-main
export PYTHONPATH=/home1/doyoonkim/projects/RAC/open-r1-main/src:/home1/doyoonkim/projects/RAC/open-r1-main/src/open_r1

echo "=== IPO on UltraFeedback: TR-GMP-s70 base (pre-GRPO), MAX_STEPS=$MAX_STEPS, OUTPUT_DIR=$OUTPUT_DIR ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID  NODE=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

EXTRA_ARGS=()
if [ -n "$MODEL_OVERRIDE" ]; then
  EXTRA_ARGS+=(--model_name_or_path "$MODEL_OVERRIDE")
fi
if [ -n "$LR_OVERRIDE" ]; then
  EXTRA_ARGS+=(--learning_rate "$LR_OVERRIDE")
fi

accelerate launch --config_file recipes/plain_1gpu.yaml src/open_r1/dpo.py \
  --config recipes/Qwen3-4B/dpo/config_ipo_ultrafeedback_s70.yaml \
  --output_dir "$OUTPUT_DIR" \
  --max_steps "$MAX_STEPS" \
  "${EXTRA_ARGS[@]}"
DPO_EXIT=$?

echo "=== DPO/IPO training done (exit $DPO_EXIT) ==="

EVAL_EXIT=0
if [ $DPO_EXIT -eq 0 ]; then
  MERGED_DIR="${OUTPUT_DIR}_merged"
  echo "=== Running official MATH-500 eval (lighteval) on merged model: $MERGED_DIR ==="
  # lighteval needs the vllm>=0.10.0 env, not rac_vllm084 -- switch back to
  # the shared `rac` env just for this eval step (same one every other
  # MATH-500 re-eval in this project uses).
  conda deactivate
  conda activate rac
  EVAL_OUTPUT_DIR="${OUTPUT_DIR}_math500_eval"
  bash /home1/doyoonkim/projects/elsa/scripts/slurm_eval_math500_savedetails_4b.sh "$MERGED_DIR" "$EVAL_OUTPUT_DIR"
  EVAL_EXIT=$?
  echo "=== MATH-500 eval done (exit $EVAL_EXIT) ==="
fi

echo "=== DONE (dpo exit $DPO_EXIT, eval exit $EVAL_EXIT) ==="
exit $DPO_EXIT
